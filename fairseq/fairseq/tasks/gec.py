###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################

# Copyright (c) 2017-present, Facebook, Inc.  # All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Grammatical error correction:
    A sequence-to-sequence training task with edit-weighted MLE loss and
    an auxiliary edit label prediction loss.

This task inherits the default TranslationTask in translation.py.

Modules:
* GECTask

Command line inputs:
    --task gec --arch copy_augmented_transformer_el
    [--edit-weighted-loss SCALE] [--edit-label-prediction SCALE]

Usage:
    fairseq-train $GEC/fairseq/data-bin/$DATA_NAME --ddp-backend=no_c10d \
    --arch copy_augmented_transformer_aux_el --task gec --criterion gec_loss \
    --edit-weighted-loss 3.0 --edit-label-prediction 1.0 --max-tokens 4000 \
    --tensorboard-logdir $GEC/fairseq/logdir/$DATA_NAME/copy_augmented_transformer_aux_el \
    --save-dir $GEC/fairseq/checkpoints/$DATA_NAME/copy_augmented_transformer_aux_el
"""

import torch
import os.path
import itertools

from fairseq import tokenizer
from fairseq import options
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    TokenLabeledIndexedRawTextDataset,
    TokenLabeledLanguagePairDataset
)

from . import register_task
from .translation import TranslationTask


@register_task('gec')
class GECTask(TranslationTask):
    """
    GEC task for seq2seq networks with edit weights & labels (given by m3).

    See, e.g., <https://arxiv.org/abs/1903.00138>.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1,
                         padding_factor=8):
        """Build the dictionary from edit-labeled raw text inputs.

        Each file contains tokenized sentences along with their token labels:
        ```text
        My teacher is going to move to change his job .
        0 0 0 0 0 0 0 0 0 0 0
        And he took in my favorite subject like soccer .
        0 0 0 0 0 0 1 0 0 0
        ...
        ```
        A dictionary is built using only the tokens and not token labels.

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            # Write only tokens to a separate file.
            with open(filename) as f_in, \
                    open(f"{filename}.tokens", "w") as f_out:
                f_out.writelines(line for i, line in enumerate(f_in)
                                 if i % 2 == 0)
            # Add tokens to dictionary with multiprocessing.
            Dictionary.add_file_to_dictionary(f"{filename}.tokens", d,
                                              tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords,
                   padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup GEC task, including dictionary & model building."""

        """
        Similar to the translation task, but also load labels dictionaries
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            return TokenLabeledIndexedRawTextDataset.exists(filename)

        def indexed_dataset(path, dictionary):
            if TokenLabeledIndexedRawTextDataset.exists(path):
                print(f"| {split} | loading token-labeled dataset from "
                      f"raw text at {path}")
                return TokenLabeledIndexedRawTextDataset(path, dictionary)
            else:
                return None

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = TokenLabeledLanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TokenLabeledLanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary
        )

    def build_generator(self, args):
        """Replace SequenceGenerator with SequenceCopyGenerator."""
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.sequence_copygenerator import SequenceCopyGenerator
            return SequenceCopyGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
