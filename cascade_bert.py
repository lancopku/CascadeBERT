from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert  import BertModel, BertPreTrainedModel


class CascadeBERTForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, cascade_models: List[BertModel] = None,
                 confidence_margin=0.5,
                 margin_loss_weight=1.0):
        super(CascadeBERTForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.cascade_models = nn.ModuleList(cascade_models)
        self.cascade_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels)
                                                  for _ in range(len(cascade_models))])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_params()
        self.infer_mode = 'big'
        self.threshold = 1.0
        self.softmax = nn.LogSoftmax(dim=-1)
        self.margin_loss = nn.MarginRankingLoss(margin=confidence_margin)  # used for small agent calibration
        self.margin_loss_weight = margin_loss_weight

    def init_params(self):
        for name, module in self.named_children():
            if 'cascade_models' not in name: # and 'big' not in name:
                module.apply(self._init_weights)
        self._init_weights(self)
        # Prune heads if needed
        if self.config.pruned_heads:
            self.prune_heads(self.config.pruned_heads)
        # Tie weights if needed
        self.tie_weights()

    def set_infer_mode(self, mode):
        self.infer_mode = mode

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_margin_loss_weight(self, margin_weight):
        self.margin_loss_weight = margin_weight

    def pair_loss(self, difficulty_labels, confidence, dif1=0, dif2=1):
        # we first try to adjust difficulty = 0  and difficulty = 1
        easy_idx = (difficulty_labels == dif1)
        hard_idx = (difficulty_labels == dif2)
        easy_conf = confidence[easy_idx]
        hard_conf = confidence[hard_idx]
        if len(easy_conf) == 0 or len(hard_conf) == 0:
            return 0.0
        uniform = torch.ones_like(hard_conf) / len(hard_conf)
        sampled_hard_idx = torch.multinomial(uniform, num_samples=len(easy_conf), replacement=True)
        rank_input1 = easy_conf
        rank_input2 = hard_conf[sampled_hard_idx]
        diff_label1 = 1.0 / (1.0 + difficulty_labels[easy_idx])  # 1.0
        diff_label2 = 1.0 / (1.0 + difficulty_labels[hard_idx][sampled_hard_idx])  # 0.5

        geq = torch.where(diff_label1 >= diff_label2, torch.ones_like(diff_label1), torch.zeros_like(diff_label1))
        less = torch.where(diff_label1 < diff_label2, -1 * torch.ones_like(diff_label2), torch.zeros_like(diff_label2))
        target = geq + less
        confidence_loss = self.margin_loss(rank_input1, rank_input2, target)
        return confidence_loss

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            difficulty_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()


        bsz = input_shape[0]
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        loss = None
        cascade_outputs = []
        for model in self.cascade_models:
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict, )
            cascade_outputs.append(outputs)

        cascade_logits = []

        for classifier, outputs in zip(self.cascade_classifiers, cascade_outputs):
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = classifier(pooled_output)
            cascade_logits.append(logits)

        paths = torch.zeros((bsz, len(cascade_logits)), device=device)

        if self.training or self.infer_mode =='cascade':
            if self.infer_mode == 'big':  # use biggest model
                logits = cascade_logits[-1]
            elif self.infer_mode == 'small':  # use smallest model
                logits = cascade_logits[0]
            elif self.infer_mode == 'cascade':  # cascading exiting
            # TODO: use dynamic forward in inference
                bsz = input_ids.size()[0]
                device = input_ids.device
                idx = torch.arange(bsz)
                emit_idxs = []
                emit_logits = []

                for i, class_logit in enumerate(cascade_logits):
                    if len(idx) == 0:
                        break
                    class_logit = class_logit[idx]
                    prob = F.softmax(class_logit, dim=-1)
                    confidence, _ = torch.max(prob, dim=-1)
                    emit_logit = class_logit[confidence > self.threshold]
                    emit_idx = idx[confidence > self.threshold]
                    paths[emit_idx, i] += 1  # record exiting path
                    emit_idxs.append(emit_idx)
                    emit_logits.append(emit_logit)
                    if i == len(cascade_logits) - 1:  # the final model is went
                        emit_idxs.append(idx[confidence <= self.threshold])
                        emit_logits.append(class_logit[confidence <= self.threshold])
                        paths[idx[confidence <= self.threshold], i] += 1
                    idx = idx[confidence <= self.threshold]  # idx for non-exited instances
                idx = torch.cat(emit_idxs, dim=0)
                logits = torch.cat(emit_logits, dim=0)
                _, order = torch.sort(idx)
                logits = logits[order]  # order it back
                cascade_outputs[-1] = cascade_outputs[-1] + (paths,)
            else:
                raise ValueError("Unsupported mode") 
        else:
            logits = cascade_logits[0] # take the small one 


        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(cascade_logits[0].view(-1), labels.view(-1))
                for c_logits in cascade_logits[1:]:
                    loss += loss_fct(c_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(cascade_logits[0].view(-1, self.num_labels), labels.view(-1))
                for c_logits in cascade_logits[1:]:
                    loss += loss_fct(c_logits.view(-1, self.num_labels), labels.view(-1))

            if difficulty_labels is not None and self.training:
                # add DAR  to guarantee easy examples produces higher confidence than difficulty examples
                c_logits = cascade_logits[0] # only support two-model now
                prob = F.softmax(c_logits, dim=-1)  # bsz, num_label
                confidence, _ = prob.max(dim=-1)
                loss += self.margin_loss_weight * self.pair_loss(difficulty_labels, confidence, dif1=0, dif2=1)

        output = (logits, paths)  #+ cascade_outputs[-1][2:]
        return ((loss,) + output) if loss is not None else output
