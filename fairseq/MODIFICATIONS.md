# All `fairseq` Modifications for GEC

All newly added files have the following prefix at the beginning:
```python
###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################
```

Among all changes, the only modification that requires a **package rebuild** is
`fairseq/data/token_labeled_language_pair_dataset.py` and its import in 
`fairseq/data/__init__.py`, because datasets are not registered separately. 
Please run:
```bash
pip install --upgrade --editable .
```

## Modifications

1. `eval_lm_fp16.py`: single-line edit for fp16 lm evaluation
2. `fairseq/models/copy_augmented_transformer_el.py`: copy-augmented transformer + edit label prediction model definition
3. `fairseq/data/token_labeled_language_pair_dataset.py`: custom dataset loader for "m3" (i.e. ori-cor sentence pairs along with token-level edit labels)
4. `fairseq/data/__init__.py`: include `token_labeled_language_pair_dataset` in the module definition (somehow there's no registry for datasets)
5. `fairseq/criterion/gec_loss.py`: weighted cross-entropy using target-side edit labels, along with an auxiliary source-side edit label prediction loss.
6. `fairseq/tasks/gec.py`: define a GEC task using custom models, datasets, and losses
7. `fairseq/sequence_copygenerator.py`: a "fork" of `fairseq/sequence_generator.py` that also keeps track of & returns copy scores in decoding
8. `generate_or_copy.py`: generation with `<unk>`'s replaced based on copy scores
9. `fairseq/scripts/test_gec_modules.py`: unit tests for newly created modules
10. `lm_scorer.py`: scoring using pre-trained neural language models 
