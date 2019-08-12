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
Copy-augmented Transformer with edit label prediction.
<https://arxiv.org/abs/1903.00138>

This file is based on transformer.py.
Any changes to the original file are marked with triple-quote comments.
(Search 'MODIFIED' for all changes made.)

Modules:
* CopyAugmentedTransformerELModel
* TransformerELEncoder
* CopyAugmentedTransformerELDecoder

Command line inputs:
    --arch {transformer_el, transformer_aux_el,
            copy_augmented_transformer, copy_augmented_transformer_aux_el,
            copy_augmented_transformer_el}
    [--copy-attention-heads N] [--alpha-warmup N] [--pad-copied-words]

Usage:
    DATA_NAME=bea19_nodup_word50k
    MODEL_NAME=copy_augmented_transformer
    python -m torch.distributed.launch --nproc_per_node 8 \
        $(which fairseq-train) $DATA_BIN/$DATA_NAME --ddp-backend no_c10d \
        --update-freq 16 --arch $MODEL_NAME --optimizer adam \
        --lr 0.001 --dropout 0.3 --max-tokens 2048 --min-lr '1e-09' \
        --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' \
        --adam-betas '(0.9, 0.98)' --share-all-embeddings \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --tensorboard-logdir $FAIRSEQ/logdir/$DATA_NAME/$MODEL_NAME'_adam_ls' \
        --save-dir $FAIRSEQ/checkpoints/$DATA_NAME/$MODEL_NAME'_adam_ls'

    DATA_NAME=bea19_nodup_word50k_el
    MODEL_NAME=copy_augmented_transformer_aux_el
    fairseq-train $GEC/fairseq/data-bin/$DATA_NAME --ddp-backend=no_c10d \
        --arch $MODEL_NAME --task gec --criterion gec_loss \
        --edit-weighted-loss 3.0 --edit-label-prediction 1.0 \
        --tensorboard-logdir $GEC/fairseq/logdir/$DATA_NAME/$MODEL_NAME \
        --save-dir $GEC/fairseq/checkpoints/$DATA_NAME/$MODEL_NAME

Architecture options:
use_copy_scores | predict_edit_labels | decode_with_edit_labels | arch
----------------|---------------------|-------------------------|------------------
False           | False               | False                   | n/a (same as base transformer)
False           | True                | False                   | transformer_aux_el
False           | False               | True                    | n/a (error)
False           | True                | True                    | transformer_el
True            | False               | False                   | copy_augmented_transformer
True            | True                | False                   | copy_augmented_transformer_aux_el [paper]
True            | False               | True                    | n/a (error)
True            | True                | True                    | copy_augmented_transformer_el
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    MultiheadAttention, AdaptiveSoftmax, SinusoidalPositionalEmbedding
)

from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture,
)

from .transformer import (
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer,
    Embedding, LayerNorm, Linear, PositionalEmbedding
)


@register_model('copy_augmented_transformer_el')
class CopyAugmentedTransformerELModel(FairseqModel):
    """
    Copy-augmented Transformer model for grammatical error correction.
    <https://arxiv.org/abs/1903.00138>

    Args:
        encoder (TransformerEncoder or TransformerELEncoder): the encoder
        decoder (TransformerDecoder or CopyAugmentedTransformerELDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off

        """
        MODIFIED: additional command line options
        """
        parser.add_argument('--use-copy-scores', type=options.eval_bool,
                            help='copy scores are calculated and '
                                 'added to decoder softmax outputs')
        parser.add_argument('--predict-edit-labels', type=options.eval_bool,
                            help='encoder also predicts whether '
                                 'each source token should be edited')
        parser.add_argument('--decode-with-edit-labels', type=options.eval_bool,
                            help='decoder uses edit labels through "gating"')
        parser.add_argument('--copy-attention-heads', type=int, metavar='N',
                            help='number of heads for copy scores attention '
                                 '(0: same number as decoder_attention_heads)')
        parser.add_argument('--alpha-warmup', type=int, default=0, metavar='N',
                            help='gradually allow alpha (copying ratio) to'
                                 'increase for the first N steps')
        parser.add_argument('--pad-copied-words', action='store_true',
                            help='zero out the probability of a word under '
                                 'the generative distribution, if it is '
                                 'a source word that could be copied')

        # Default Transformer hyperparameters
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        """
        MODIFIED: 
        Optionally use encoder with edit labels or a copy-augmented decoder.
        
        Also raise errors for invalid argument combinations.
        """
        if not (args.use_copy_scores or args.predict_edit_labels):
            raise ValueError("Either use_copy_scores or predict_edit_labels"
                             "must be True. Otherwise just use -a transformer.")
        if not args.predict_edit_labels and args.decode_with_edit_labels:
            raise ValueError("decode_with_edit_labels cannot be True "
                             "if predict_edit_labels is False")

        encoder_module = TransformerELEncoder
        decoder_module = CopyAugmentedTransformerELDecoder

        encoder = encoder_module(args, src_dict, encoder_embed_tokens)
        decoder = decoder_module(args, tgt_dict, decoder_embed_tokens)
        return CopyAugmentedTransformerELModel(encoder, decoder)


class TransformerELEncoder(FairseqEncoder):
    """
    Transformer "edit label" encoder consisting of *args.encoder_layers* layers.
    Each layer is a :class:`TransformerEncoderLayer`.

    **One model dimension is reserved Outputs edit labels for each input token.**

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        """
        MODIFIED: a readout layer for edit label prediction
        """
        self.predict_edit_labels = args.predict_edit_labels
        if self.predict_edit_labels:
            self.edit_label_layer = Linear(embed_dim, 1)
        else:
            self.edit_label_layer = None

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_edit_logits** (Tensor): output edit labels per token
                  in logits
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        """
        MODIFIED: edit labels (represented as logits)
        """
        if self.predict_edit_labels:
            edit_logits = self.edit_label_layer(x)
        else:
            edit_logits = None

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_edit_logits': edit_logits,  # T x B x 1
            'encoder_input_tokens': src_tokens,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        """
        MODIFIED: also reorder edit scores and input tokens, just in case
        """
        if encoder_out['encoder_edit_logits'] is not None:
            encoder_out['encoder_edit_logits'] = \
                encoder_out['encoder_edit_logits'].index_select(1, new_order)
        if encoder_out['encoder_input_tokens'] is not None:
            encoder_out['encoder_input_tokens'] = \
                encoder_out['encoder_input_tokens'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class CopyAugmentedTransformerELDecoder(FairseqIncrementalDecoder):
    """
    Copy-Augmented Transformer decoder consisting of *args.decoder_layers*
    layers. Each layer is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False,
                 left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        """
        MODIFIED: add copying mechanism as a separate multi-head attention
        """
        if args.use_copy_scores:
            assert not no_encoder_attn, \
                "copy scores cannot be computed if " \
                "there is no encoder-decoder attention"
            # Number of heads in copy attention layer is an optional argument
            self.copy_attention_heads = (args.decoder_attention_heads
                                         if args.copy_attention_heads == 0
                                         else args.copy_attention_heads)
            self.copy_attention = MultiheadAttention(
                embed_dim, self.copy_attention_heads,
                dropout=args.attention_dropout,
            )
            self.copy_balancing_layer = Linear(input_embed_dim, 1)
            if args.decode_with_edit_labels:
                raise NotImplementedError
        else:
            self.copy_attention = None
            self.copy_balancing_layer = None

        # Alpha scheduler & diagnostic checker
        self.alpha_warmup = args.alpha_warmup
        self.num_batches = 0
        self.num_copies = 0
        self.mean_alpha = 0.0

        # Zero out generative probability of a word if also in source sentence
        self.pad_copied_words = args.pad_copied_words

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            """
            MODIFIED: require share_input_output_embed for copying mechanism
            """
            raise NotImplementedError(
                "copying mechanism requires share_input_output_embed"
            )
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        """
        MODIFIED: combine with copy attention
        
        _Compared to the paper, we take the output of the attention layer to
        compute the copy-generate balancing factor ("alpha"). 
        
        shapes:
            x (Q): T_dec x B x C
            copy_scores: B x T_dec x T_enc
            alphas: B x T_dec x 1
        """
        if self.copy_attention is not None:
            assert encoder_out is not None
            # attn_output: T_dec x B x C, copy_scores: B x T_dec x T_enc
            attn_output, copy_scores = self.copy_attention(
                query=x,
                key=encoder_out['encoder_out'],
                value=encoder_out['encoder_out'],
                key_padding_mask=encoder_out['encoder_padding_mask'],
                static_kv=True,  # ??
                need_weights=True
            )
            # Balancing factor between generative & copy distributions
            alphas = torch.sigmoid(
                self.copy_balancing_layer(attn_output)
            ).transpose(0, 1)
        else:
            # Default values to be returned
            copy_scores = None
            alphas = None

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        batch_size, tgt_len, dim_model = x.size()

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            x = F.linear(x, self.embed_tokens.weight)

        """
        MODIFIED: combine copy distribution with generative distribution
        """
        gen_probs = None
        copy_probs = None
        self.num_batches += 1

        if self.copy_attention is not None:
            # Generative distribution: B x T_dec x V
            gen_logits = x
            vocab_size = self.embed_tokens.num_embeddings
            assert gen_logits.size() == (batch_size, tgt_len, vocab_size), \
                f"invalid shape for decoder logits " \
                f"(possibly changed while taking softmax)"

            # Copy distribution: B x T_dec x V
            copy_probs = compute_copy_probs(
                copy_scores, encoder_out['encoder_input_tokens'], vocab_size
            )

            # Optionally zero-out the probability of copied words
            if self.pad_copied_words:
                gen_logits.masked_fill_(copy_probs > 0, -1e8)

            # Optionally increase copying ratio during initial training steps
            if self.alpha_warmup > 0 and self.training:
                clamp_factor = min(1., self.num_batches / self.alpha_warmup)
                alphas = torch.clamp(alphas, max=clamp_factor)

            # alphas: B x T_dec x 1 -> B x T_dec x V
            alphas_ = alphas.expand(-1, -1, vocab_size)

            # Combine copy & generative distributions using alphas
            gen_probs = torch.softmax(gen_logits, dim=-1)
            combined_probs = (1 - alphas_) * gen_probs + alphas_ * copy_probs
            x = torch.log(combined_probs + 1e-8)  # stability... but inefficient

            # Diagnostic check
            mean_alpha = alphas.mean().item()
            if mean_alpha > 0.9:
                self.num_copies += 1
                print(f"WARNING: reached mean copying ratio of {mean_alpha:.5f}"
                      f", copy count: {self.num_copies}/{self.num_batches}")
            self.mean_alpha += mean_alpha
            if self.num_batches % 1000 == 0:
                print(f"INFO: number of batches {self.num_batches}, "
                      f"mean copying ratio (alpha): "
                      f"{self.mean_alpha / 1000:.5f}")
                self.mean_alpha = 0.0

        """
        MODIFIED: also return copy scores, balancing factors (alpha),
            copy & generative probs, as well as edit labels from the encoder.
            (None if unavailable.)
        """
        return x, {'attn': attn, 'inner_states': inner_states,
                   'alphas': alphas, 'copy_scores': copy_scores,
                   'gen_probs': gen_probs, 'copy_probs': copy_probs,
                   'edit_logits': encoder_out['encoder_edit_logits']}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


"""
MODIFIED: a helper that converts copy scores to a copy distribution over tokens
"""


def compute_copy_probs(copy_scores, src_tokens, vocab_size):
    """Converts copy scores on source tokens into a probability distribution.

    The `scatter_add()` operation is the same as:
    ```python
    for b in range(batch_size):
        for t in range(tgt_len):
            for s in range(src_len):
                copy_probs[b, t, src_tokens[b, s]] += copy_scores[b, t, s]
    ```

    :param copy_scores: torch.FloatTensor, (batch_size, tgt_len, src_len)
    :param src_tokens: torch.LongTensor, (batch_size, src_len)
    :param vocab_size: int
    :return: torch.FloatTensor, (batch_size, tgt_len, vocab_size)
    """
    batch_size, tgt_len, src_len = copy_scores.size()
    copy_probs = copy_scores.new_zeros((batch_size, tgt_len, vocab_size),
                                       requires_grad=copy_scores.requires_grad)
    src_tokens_expanded = src_tokens.unsqueeze(1).expand(-1, tgt_len, -1)
    return copy_probs.scatter_add(2, src_tokens_expanded, copy_scores)


"""
MODIFIED: command line architectures for the Transformer

Architecture options:
use_copy_scores | predict_edit_labels | decode_with_edit_labels | arch
----------------|---------------------|-------------------------|------------------
False           | False               | False                   | n/a (same as base transformer)
False           | True                | False                   | transformer_aux_el
False           | False               | True                    | n/a (error)
False           | True                | True                    | transformer_el
True            | False               | False                   | copy_augmented_transformer
True            | True                | False                   | copy_augmented_transformer_aux_el [paper]
True            | False               | True                    | n/a (error)
True            | True                | True                    | copy_augmented_transformer_el
"""


@register_model_architecture('copy_augmented_transformer_el', 'copy_augmented_transformer_el')
def base_architecture(args):
    """Copy-augmented transformer with edit-augmented encoder-decoder attention.

    Same model size as the original Transformer,
    except double FFN size (4096) and larger dropout rate (0.2).
    """

    # Defaults to all True.
    args.use_copy_scores = getattr(args, 'use_copy_scores', True)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', True)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', True)
    args.copy_attention_heads = getattr(args, 'copy_attention_heads', 0)
    args.alpha_warmup = getattr(args, 'alpha_warmup', 0)
    args.pad_copied_words = getattr(args, 'pad_copied_words', False)

    # Model size parameters.
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.2)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('copy_augmented_transformer_el', 'transformer_aux_el')
def transformer_aux_el(args):
    """Vanilla Transformer that also predicts edit labels as auxiliary task."""
    args.use_copy_scores = getattr(args, 'use_copy_scores', False)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', True)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', False)
    base_architecture(args)


@register_model_architecture('copy_augmented_transformer_el', 'transformer_el')
def transformer_el(args):
    """Vanilla Transformer with edit score-augmented encoder-decoder attention."""
    args.use_copy_scores = getattr(args, 'use_copy_scores', False)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', True)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', True)
    base_architecture(args)


@register_model_architecture('copy_augmented_transformer_el', 'copy_augmented_transformer')
def copy_augmented_transformer(args):
    """Copy-augmented Transformer *without* edit label prediction."""
    args.use_copy_scores = getattr(args, 'use_copy_scores', True)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', False)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', False)
    base_architecture(args)


@register_model_architecture('copy_augmented_transformer_el', 'copy_augmented_transformer_aux_el')
def copy_augmented_transformer_aux_el(args):
    """Copy-augmented Transformer with edit label prediction.
    *Most similar to the original paper's setup.*
    """
    args.use_copy_scores = getattr(args, 'use_copy_scores', True)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', True)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', False)
    base_architecture(args)


@register_model_architecture('copy_augmented_transformer_el', 'copy_augmented_transformer_aux_el_t2t')
def copy_augmented_transformer_aux_el_t2t(args):
    """Copy-augmented Transformer with edit label prediction.
    T2T ver: larger model size (1024-4096), layernorm *before* attention."""
    args.use_copy_scores = getattr(args, 'use_copy_scores', True)
    args.predict_edit_labels = getattr(args, 'predict_edit_labels', True)
    args.decode_with_edit_labels = getattr(args, 'decode_with_edit_labels', False)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)

    base_architecture(args)

