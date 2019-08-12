###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch
import os.path
import itertools

from fairseq import utils
from fairseq.tokenizer import tokenize_line
from fairseq.data.language_pair_dataset import LanguagePairDataset

from . import data_utils


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    """Create batches out of items in a TokenLabeledLanguagePairDataset.

    Note that labels are always binary, meaning the same index may be used for
    both label and padding. Still, we can recover only the valid labels by
    looking at either src_tokens or src_lengths.
    """
    if len(samples) == 0:
        return {}

    def merge(key, tokens_or_labels, left_pad, move_eos_to_beginning=False):
        """Each source or target item is now a dict that looks like
        {'tokens': tokens, 'labels': labels}.
        """
        return data_utils.collate_tokens(
            [s[key][tokens_or_labels] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', 'tokens', left_pad=left_pad_source)
    src_labels = merge('source', 'labels', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source']['tokens'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_labels = src_labels.index_select(0, sort_order)

    prev_output_tokens = None
    tgt_tokens, tgt_labels = None, None
    if samples[0].get('target', None) is not None:
        tgt_tokens = merge('target', 'tokens', left_pad=left_pad_target)
        tgt_labels = merge('target', 'labels', left_pad=left_pad_target)
        tgt_tokens = tgt_tokens.index_select(0, sort_order)
        tgt_labels = tgt_labels.index_select(0, sort_order)
        ntokens = sum(len(s['target']['tokens']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                'tokens',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']['tokens']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': tgt_tokens,
        # additional token-level label targets
        'src_labels': src_labels,
        'tgt_labels': tgt_labels,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def encode_labels_line(labels_line, append_eos=True, reverse_order=False):
    """Custom helper:
    Encode a string of space-separated binary labels into LongTensor.

    Mimicks fairseq.data.dictionary.Dictionary.encode_line().
    eos always gets a zero token (no change).

    Returns a torch.IntTensor, analogous to dictionary's encode_line() method.
    """
    labels = [int(label) for label in tokenize_line(labels_line)]
    assert all([label in [0, 1] for label in labels]), \
        f"encode_labels_line: token-level labels must be binary!"
    if reverse_order:
        labels = list(reversed(labels))
    if append_eos:
        labels.append(0)
    return torch.tensor(labels, dtype=torch.int)


class TokenLabeledIndexedRawTextDataset(torch.utils.data.Dataset):
    """Takes a token-labeled text file as input and binarizes it in memory
    at instantiation. Original lines are also kept in memory.

    Builds upon fairseq.data.indexed_dataset.IndexedRawTextDataset.
    """

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.labels_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary):
        """Reads in a file that looks like:

        ```text
        My teacher is going to move to change his job .
        0 0 0 0 0 0 0 0 0 0 0
        And he took in my favorite subject like soccer .
        0 0 0 0 0 0 1 0 0 0
        ...
        ```

        For now, labels are always assumed to be in the labels dictionary.
        """

        with open(path, 'r', encoding='utf-8') as f:
            # Iterate over every pair of lines without loading all at once
            for tokens_line, labels_line in itertools.zip_longest(*[f]*2):
                # Tokens proceed the same as in a default IndexedDataset
                self.lines.append(tokens_line.strip('\n'))
                tokens = dictionary.encode_line(
                    tokens_line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
                # Labels stored in parallel
                labels = encode_labels_line(
                    labels_line,
                    append_eos=self.append_eos, reverse_order=self.reverse_order
                ).long()
                self.labels_list.append(labels)
                assert len(tokens) == len(labels), \
                    f"TokenLabeledIndexedRawTextDataset.read_data: " \
                    f"number of tokens ({len(tokens)}) and " \
                    f"labels ({len(labels)}) not matching"

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, i):
        """Returns a **dict** of tokens and labels."""
        self.check_index(i)
        return {
            "tokens": self.tokens_list[i],
            "labels": self.labels_list[i]
        }

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class TokenLabeledLanguagePairDataset(LanguagePairDataset):
    """
    A pair of TokenLabeledIndexedRawTextDataset containing binary token labels.
    Inherits LanguagePairDataset (everything else is the same except that each
    source & target item is now a dictionary of tokens & labels).

    Args:
        src (TokenLabeledIndexedRawTextDataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        src_labels_dict (~fairseq.data.Dictionary): source token labels vocabulary
        tgt (TokenLabeledIndexedRawTextDataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        tgt_labels_dict (~fairseq.data.Dictionary): target token labels vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        super().__init__(
            src, src_sizes, src_dict,
            tgt, tgt_sizes, tgt_dict,
            left_pad_source, left_pad_target,
            max_source_positions, max_target_positions,
            shuffle, input_feeding, remove_eos_from_source, append_eos_to_target
        )

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # **Change in TokenLabeled**: also append 0 to labels if EOS is added.
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index]["tokens"][-1] != eos:
                tgt_tokens = torch.cat([self.tgt[index]["tokens"],
                                        torch.LongTensor([eos])])
                tgt_labels = torch.cat([self.tgt[index]["labels"],
                                        torch.LongTensor([0])])
                tgt_item = {"tokens": tgt_tokens, "labels": tgt_labels}

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            # tokens and labels have same length
            if self.src[index]["tokens"][-1] == eos:
                src_tokens = self.src[index]["tokens"][:-1]
                src_labels = self.src[index]["labels"][:-1]
                src_item = {"tokens": src_tokens, "labels": src_labels}

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `src_labels` (LongTensor): 1D Tensor of the token-level
                    labels of each source sentence of shape `(bsz, src_len)`.
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)

        src_dummy = self.src_dict.dummy_sentence(src_len)
        tgt_dummy = self.tgt_dict.dummy_sentence(tgt_len)
        return self.collater([
            {
                'id': i,
                'source': {
                    "tokens": src_dummy,
                    "labels": torch.zeros_like(src_dummy),
                },
                'target': {
                    "tokens": tgt_dummy,
                    "labels": torch.zeros_like(tgt_dummy),
                } if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])
