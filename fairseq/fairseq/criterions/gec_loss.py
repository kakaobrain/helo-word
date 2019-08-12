###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Edit-weighted cross-entropy loss, typically used for GEC.
"""

import math
import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('gec_loss')
class GECLossCriterion(FairseqCriterion):
    """A weighted cross-entropy criterion with
    an auxiliary source-side token-label classification."""
    def __init__(self, args, task):
        super().__init__(args, task)
        # Note: self.padding_idx defaults to the target dictionary's pad index
        self.src_padding_idx = task.source_dictionary.pad()

        self.edit_weighted_loss = args.edit_weighted_loss
        if self.edit_weighted_loss != 1.0:
            print(f"using edit-weighted MLE loss "
                  f"with scale {self.edit_weighted_loss}")
        self.edit_label_prediction = args.edit_label_prediction
        if self.edit_label_prediction > 0.0:
            print(f"using auxiliary edit label prediction loss "
                  f"with scale {self.edit_label_prediction}")

        # Check that the model provides required options.
        if (self.edit_label_prediction > 0.0 and
                not getattr(args, 'predict_edit_labels', None)):
            raise ValueError("model must have predict_edit_labels==True")
        if (self.edit_label_prediction == 0.0 and
                getattr(args, 'predict_edit_labels', None)):
            print("WARNING: edit labels are predicted by the model "
                  "but not included in the training objective.")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--edit-weighted-loss', type=float, default=3.0,
                            help='use edit-weighted MLE loss for targets'
                                 '(default: 3.0; ignored if 1.0)')
        parser.add_argument('--edit-label-prediction', type=float, default=1.0,
                            help='additionally predict edit labels from '
                                 'encoder outputs, using given scale.'
                                 '(default: 1.0; ignored if 0.0)')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, edit_label_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        if edit_label_loss is not None:
            loss = loss + self.edit_label_prediction * edit_label_loss
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'edit_label_loss': utils.item(edit_label_loss.data) if edit_label_loss is not None else 0.0,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))  # (B x T_dec) x V
        target = model.get_targets(sample, net_output).view(-1)  # (B x T_dec)

        # Compute token-level weights based on target-side token labels.
        # weights: edit_weighted_loss if tgt_labels == 1 else 1.0
        edit_weights = sample['tgt_labels'].float().view(-1)  # B x T_dec
        edit_weights = (self.edit_weighted_loss - 1.) * edit_weights + 1.
        loss = F.nll_loss(lprobs, target, ignore_index=self.padding_idx,
                          reduction='none')
        loss = edit_weights * loss
        if reduce:
            loss = torch.sum(loss)

        # Optionally add auxiliary loss from source-side edit label prediction.
        # Always reduced (dimension differs from loss).
        if self.edit_label_prediction > 0.0:
            # All three tensors have the same shape: (B x T_enc)
            src_nonpads = sample['net_input']['src_tokens'].ne(self.src_padding_idx)
            edit_logits = net_output[1]['edit_logits'].squeeze(-1).transpose(0, 1)
            src_labels = sample['src_labels'].float()
            edit_label_loss = F.binary_cross_entropy_with_logits(
                edit_logits[src_nonpads],
                src_labels[src_nonpads],
                reduction='sum'
            )
        else:
            edit_label_loss = None

        return loss, edit_label_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        edit_label_loss_sum = sum(log.get('edit_label_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'edit_label_loss': edit_label_loss_sum / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
