import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed, BertModel, AdamW,
)
from cascade_bert import CascadeBERTForSequenceClassification
# from transformers.trainer_utils import is_main_process
from trainer import CascadeBERTTrainer
from scipy.special import softmax

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json", "tsv"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    cascade_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models, split by ;"}
    )
    cascade_model_layers: str = field(
        metadata={"help": "layer number of each cascading model , split by ;, e.g., 2;12"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    saved_model_path: Optional[str] = field(
        default="", metadata={"help": "Trained model path"}
    )
    infer_mode: Optional[str] = field(
        default="small", metadata={"help": "infer mode: big, small, cascade"}
    )
    confidence_threshold: Optional[float] = field(
        default=1.0, metadata={"help": "threshold for judging the easy examples"}
    )
    confidence_margin: Optional[float] = field(
        default=0.3, metadata={"help": "confidence margin"}
    )
    dar_weight: Optional[float] = field(
        default=0.5, metadata={"help": "difficulty-aware regularization weight"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO, # if logging.WARN # is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    #if is_main_process(training_args.local_rank):
    #    transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    eval_and_test_datasets = load_dataset("glue", data_args.task_name)
    logger.info("Loading train data from %s" % data_args.train_file)
    difficulty_train_datasets = load_dataset(
        "json", data_files={"train": data_args.train_file})

    # Labels
    label_list = eval_and_test_datasets["train"].features["label"].names
    num_labels = len(label_list)
    # load complete models
    cascade_model_names = model_args.cascade_model_name_or_path.split(";")
    cascade_model_layers = [int(k) for k in model_args.cascade_model_layers.split(";")]

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else cascade_model_names[-1],
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else cascade_model_names[-1],
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    for i in range(1, len(cascade_model_layers)):
        cascade_model_layers[i] += cascade_model_layers[i - 1]
    print('Actual layer cost:', cascade_model_layers)

    cascade_models = [BertModel.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in cascade_model_names[-1]),
        config=config,
        cache_dir=model_args.cache_dir,
    ) for model_path in cascade_model_names]

    logger.info("Setting confidence margin to %.3f" % model_args.confidence_margin)
    logger.info("Setting margin loss weight to %.3f" % model_args.dar_weight)

    if model_args.saved_model_path != "":
        model = CascadeBERTForSequenceClassification.from_pretrained(
            model_args.saved_model_path,
            from_tf=bool(".ckpt" in model_args.saved_model_path),
            config=config,
            cache_dir=model_args.cache_dir,
            cascade_models=cascade_models,
            confidence_margin=model_args.confidence_margin,
            margin_loss_weight=model_args.dar_weight
        )
    else:
        model = CascadeBERTForSequenceClassification(config,
                                                     cascade_models=cascade_models,
                                                     confidence_margin=model_args.confidence_margin,
                                                     margin_loss_weight=model_args.dar_weight
                                                     )
    # setting up some configs
    logger.info("Setting infer mode to: %s" % model_args.infer_mode)
    model.set_infer_mode(model_args.infer_mode)
    logger.info("Setting infer mode to: %.6f" % model_args.confidence_threshold)
    model.set_threshold(model_args.confidence_threshold)

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}
    # Formation Rule
    difficulty_list = ["easy", "hard"]
    difficulty_to_id = None
    difficulty_to_id = {v: i for i, v in enumerate(difficulty_list)}
    print(label_to_id)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        if difficulty_to_id is not None and "difficulty" in examples:
            result["difficulty_labels"] = [difficulty_to_id[d] for d in examples["difficulty"]]
        return result

    difficulty_train_datasets = difficulty_train_datasets.map(preprocess_function, batched=True,
                                                              load_from_cache_file=not data_args.overwrite_cache)
    label_to_id = None  # mimic original script
    eval_and_test_datasets = eval_and_test_datasets.map(preprocess_function, batched=True,
                                                        load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = difficulty_train_datasets["train"]
    eval_dataset = eval_and_test_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    test_dataset = eval_and_test_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    assert data_args.task_name is not None, "task name cannot be None"
    metric = load_metric("glue", data_args.task_name)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Initialize our Trainer
    trainer = CascadeBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
        require_exit_distribution=True if model_args.infer_mode in ["cascade"] else False,
        model_layer_num=[2, 14] if model_args.infer_mode != "proxy" else [2, 12],
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.saevd_path if os.path.isdir(model_args.saved_model_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(eval_and_test_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)
            speed_up = eval_result['eval_expected_acceleration'] if 'eval_expected_acceleration' in eval_result else 1.0 
            output_eval_file = os.path.join(training_args.output_dir,  "eval_results_%s_TH%.6f_spd%.2f.txt" % (task,
                                                                                                              model_args.confidence_threshold,
                                                                                                              speed_up ))
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_prediction = trainer.predict(test_dataset=eval_dataset)
            output_prob_file = os.path.join(training_args.output_dir, f"eval_prob_{task}.npy")
            output_label_file = os.path.join(training_args.output_dir, f"eval_label_{task}.npy")
            logits = eval_prediction.predictions
            prob = softmax(logits, axis=-1)
            label = eval_prediction.label_ids
            # np.save(output_label_file, label)
            # np.save(output_prob_file, prob)
            eval_results.update(eval_result)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(eval_and_test_datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            prediction_ret = trainer.predict(test_dataset=test_dataset)
            predictions = prediction_ret.predictions

            speed_up = prediction_ret.metrics['eval_expected_acceleration']
            predictions = np.argmax(predictions, axis=1)
            output_test_file = os.path.join(training_args.output_dir, "test_results_%s_TH%.6f_spd%.2f.txt" % (task,
                                                                                                              model_args.confidence_threshold,
                                                                                                              speed_up))
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
