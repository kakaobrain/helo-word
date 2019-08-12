# helo_word
**A Neural Grammatical Error Correction System Built on Better Pre-training and Sequential Transfer Learning**

Code accompanying Team Kakao&Brain's submission to the 
[ACL 2019 BEA Workshop Shared Task](https://www.cl.cam.ac.uk/research/nl/bea2019st/).  
(`helo_word` is our informal team name.)

Paper: https://arxiv.org/abs/1907.01256

ACL Anthology: https://www.aclweb.org/anthology/papers/W/W19/W19-4423/

## Authors

[YJ Choe](https://yjchoe.github.io/)^, 
[Jiyeon Ham](https://github.com/hammouse)^, 
[Kyubyong Park](https://github.com/Kyubyong)^, 
Yeoil Yoon^

^Equal contribution.

## Installation

Requires Python 3.

```bash
# apt-get packages (required for hunspell & pattern)
apt-get update
apt-get install libhunspell-dev libmysqlclient-dev -y

# pip packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt
python -m spacy download en

# custom fairseq (fork of 0.6.1 with gec modifications)
pip install --editable fairseq

# errant
git clone https://github.com/chrisjbryant/errant

# pattern3 (see https://www.clips.uantwerpen.be/pages/pattern for any installation issues)
pip install pattern3
python -c "import site; print(site.getsitepackages())"
# ['PATH_TO_SITE_PACKAGES']
cp tree.py PATH_TO_SITE_PACKAGES/pattern3/text/
```

## Download & Preprocess Data

```bash
python preprocess.py
```

## Restricted Track

- Prepare data for the restricted track
    ```bash
    python prepare.py --track 1
    ```
- Pre-train
    - If you train the model, the system will automatically create a checkpoint directory.
    - Fill it in {ckpt_dir}.
    - Also fill in the number of GPUs used for training in {ngpu}.
    ```bash
    python train.py --track 1 --train-mode pretrain --model base --ngpu {ngpu}
    python evaluate.py --track 1 --subset valid --ckpt-dir {ckpt_dir}
    ```
- Train
    - If you evaluate the model, the system will automatically create an output directory.
    - Fill the previous model output directory into {prev_model_output_dir}.
    ```bash
    python train.py --track 1 --train-mode train --model base --ngpu {ngpu} \
        --lr 1e-4 --max-epoch 40 --reset --prev-model-output-dir {prev_model_output_dir}
    python evaluate.py --track 1 --subset valid --ckpt-dir {ckpt_dir}
    ```
- Fine-tune
    - Fill the best validation report into {prev_model_output_fpath}.
    - Then `error_type_control.py` will give you a list of error types to be removed.
    - Fill them into {remove_error_type_lst}.
    ```bash
    python train.py --track 1 --train-mode finetune --model base --ngpu {ngpu} \
        --lr 5e-5 --max-epoch 80 --reset --prev-model-output-dir {prev_model_output_dir}
    python evaluate.py --track 1 --subset valid --ckpt-dir {ckpt_dir}
    python error_type_control.py --report {prev_model_output_fpath} \
        --max_error_types 10 --n_simulations 1000000
    python evaluate.py --track 1 --subset test --ckpt-fpath {ckpt_fpath} \
        --remove-unk-edits --remove-error-type-lst {remove_error_type_lst} \
        --apply-rerank --preserve-spell --max-edits 7 
    ```

## Low Resource Track

- Prepare data for the low resource track
    ```bash
    python prepare.py --track 3
    ```
- Pre-train
    ```bash
    python train.py --track 3 --train-mode pretrain --model base --ngpu {ngpu}
    python evaluate.py --track 3 --subset valid --ckpt-dir {ckpt_dir}
    ```
- Train
    ```bash
    python train.py --track 3 --train-mode finetune --model base --ngpu {ngpu} \
        --max-epoch 40 --prev-model-output-dir {prev_model_output_dir} 
    python evaluate.py --track 3 --subset valid --ckpt-dir {ckpt_dir}
    python evaluate.py --track 3 --subset test --ckpt-fpath {ckpt_fpath} \
        --remove-unk-edits --remove-error-type-lst {remove_error_type_lst} \
        --apply-rerank --preserve-spell --max-edits 7 
    ```

## A Note on `fairseq`

We ran our Transformer models using [`fairseq-0.6.1`](https://github.com/pytorch/fairseq/releases/tag/v0.6.1). 
We had to make several modifications to the package though,
including our own implementation of the copy-augmented Transformer model.
You can find all of our modifications in [`fairseq/MODIFICATIONS.md`](fairseq/MODIFICATIONS.md).

## Citation

If you use our code for research, please cite our work as:
```bibtex
@inproceedings{choe-etal-2019-neural,
    title = "A Neural Grammatical Error Correction System Built On Better Pre-training and Sequential Transfer Learning",
    author = "Choe, Yo Joong  and
      Ham, Jiyeon  and
      Park, Kyubyong  and
      Yoon, Yeoil",
    booktitle = "Proceedings of the Fourteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-4423",
    pages = "213--227",
    abstract = "Grammatical error correction can be viewed as a low-resource sequence-to-sequence task, because publicly available parallel corpora are limited.To tackle this challenge, we first generate erroneous versions of large unannotated corpora using a realistic noising function. The resulting parallel corpora are sub-sequently used to pre-train Transformer models. Then, by sequentially applying transfer learning, we adapt these models to the domain and style of the test set. Combined with a context-aware neural spellchecker, our system achieves competitive results in both restricted and low resource tracks in ACL 2019 BEAShared Task. We release all of our code and materials for reproducibility.",
}
```
