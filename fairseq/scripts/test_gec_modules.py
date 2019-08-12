###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################

"""
Unit tests for custom modules added for the GEC task.
"""

import torch


"""
fairseq.models.copy_augmented_transformer_el
"""


def test_copy_augmented_transformer_el(args, device):
    """Build a copy-augmented transformer model."""
    from fairseq.models.copy_augmented_transformer_el import (
        TransformerELEncoder, CopyAugmentedTransformerELDecoder,
        CopyAugmentedTransformerELModel
    )
    # encoder = TransformerELEncoder(args, ...)
    # decoder = CopyAugmentedTransformerELDecoder(args, ...)
    # model = CopyAugmentedTransformerELModel(encoder, decoder)
    return


def test_compute_copy_probs(device):
    """Sanity check tester for compute_copy_logits."""
    from fairseq.models.copy_augmented_transformer_el import compute_copy_probs
    batch_size, tgt_len, src_len, vocab_size = 2, 3, 4, 10
    src_tokens = torch.Tensor([
        [7, 3, 1, 9],
        [5, 4, 2, 5]
    ]).to(device=device, dtype=torch.long)
    copy_scores = torch.Tensor([
        [[0.9, 0.1, 0.0, 0.0],
         [0.1, 0.7, 0.2, 0.0],
         [0.0, 0.0, 0.2, 0.8]],
        [[0.9, 0.0, 0.0, 0.1],
         [0.0, 0.9, 0.1, 0.0],
         [0.1, 0.0, 0.2, 0.7]],
    ]).requires_grad_().to(device=device)
    copy_probs = torch.Tensor([
        [[0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0],
         [0.0, 0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
         [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]],
        [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.2, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]]
    ]).to(device=device)
    probs = compute_copy_probs(copy_scores, src_tokens, vocab_size)
    assert probs.requires_grad, \
        "compute_copy_probs: output probs must require gradients in this test"
    if torch.allclose(probs, copy_probs):
        print("compute_copy_probs: test passed!")
    else:
        print("expected:", copy_probs)
        print("got:", probs)
        raise RuntimeError("compute_copy_probs: test failed")


"""
fairseq.data.token_labeled_language_pair_dataset
"""


def test_token_labeled_language_pair_dataset(args, device):
    from fairseq.data.token_labeled_language_pair_dataset import (
        TokenLabeledIndexedRawTextDataset, TokenLabeledLanguagePairDataset
    )
    # raw_text_dataset = TokenLabeledIndexedRawTextDataset(args)
    # dataset = TokenLabeledLanguagePairDataset(raw_text_dataset, args)
    return


"""
Run all tests
"""


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_copy_augmented_transformer_el(None, device)
    test_compute_copy_probs(device)
    test_token_labeled_language_pair_dataset(None, device)