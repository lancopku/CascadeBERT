import collections
from typing import Union, List

import torch
from transformers import Trainer, is_torch_tpu_available
from transformers.trainer import nested_detach
from transformers.trainer_pt_utils import DistributedTensorGatherer, nested_concat
from transformers.trainer_utils import PredictionOutput, EvalPrediction

from transformers.utils import logging

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)

class CascadeBERTTrainer(Trainer):
    def __init__(self,
                 model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 require_exit_distribution=True,
                 model_layer_num=[2, 14],
                 **kwargs, ):
        super(CascadeBERTTrainer, self).__init__(model,
                                                 args,
                                                 data_collator,
                                                 train_dataset,
                                                 eval_dataset,
                                                 tokenizer,
                                                 model_init,
                                                 compute_metrics,
                                                 callbacks,
                                                 optimizers)
        self.require_exit_distribution = require_exit_distribution
        self.model_layer_num = model_layer_num
        
    def prediction_loop(
                self, dataloader, description, prediction_loss_only=False) -> PredictionOutput:

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
           prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
       # Note: in torch.distributed mode, there's no point in wrapping the model
       # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        paths_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
        paths_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels, paths = self.prediction_step(model, inputs, prediction_loss_only)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, dim=0)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, dim=0)
            if paths is not None:
                paths_host = paths if paths_host is None else nested_concat(paths_host, paths, dim=0)
 
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
 
           # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (
                    step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))

                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                paths_gatherer.add_arrays(self._gather_and_numpify(paths_host, "eval_path_ids"))
                # embeddings_gatherer.add_arrays(self._gather_and_numpify(embeddings_host, "eval_embedding"))
 
               # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, paths_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")
 
       # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
        labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
        paths_gatherer.add_arrays(self._gather_and_numpify(paths_host, "eval_path_ids"))
 
        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize()
        label_ids = labels_gatherer.finalize()
        paths = paths_gatherer.finalize()

       # add code here
        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}
 
        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss.mean().item()
        BASE_LAYER = 12

        if paths is not None:
            path_dist = paths.sum(axis=0)
            
            layer_cnt = 0
            total_layer = 0
            cur_layer = 0
            for i, k in enumerate(path_dist):
                cur_layer = self.model_layer_num[i]  # repetitive computation
                print("%d examples exit at %d layer model" % (k, cur_layer))
                layer_cnt += k * cur_layer
                total_layer += k * BASE_LAYER  # self.model_layer_num[-1]
            print('total examples', paths.sum())
            avg_layer = layer_cnt / paths.sum()
            metrics["expected_saving"] = 1 - (layer_cnt / total_layer)
            metrics["expected_acceleration"] = BASE_LAYER / avg_layer

       # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
            self, model, inputs, prediction_loss_only):

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().detach()
                logits = outputs[1]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[0]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
                # Remove the past from the logits.
                logits = logits[: self.args.past_index - 1] + logits[self.args.past_index:]
            if self.require_exit_distribution:
                paths = outputs[-1]
            else:
                paths = None


        logits = nested_detach(logits)

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels, paths)
