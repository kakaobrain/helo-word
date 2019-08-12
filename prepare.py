import logging
import argparse

from gec.track import choice_track
from gec import util


logging.getLogger().setLevel(logging.INFO)


def main(args):
    track = choice_track(args.track)

    prepare_text(track)

    for train_mode in track.train_modes:
        databin_path = track.get_databin_path(train_mode)
        trainpref, validpref = track.get_pref(train_mode)
        prepare_binary(databin_path, trainpref, validpref, track.fp.BPE_VOCAB)


def prepare_text(track):
    logging.info(f"[Prepare] 1. prepare for the text data")
    fp = track.fp

    if track.TRACK_NUM == 0:
        logging.info("[Prepare] 1-1. pretrain")
        util.maybe_prompt(fp.DAE_ORI0, f"cat {fp.GUTENBERG_ORI0} {fp.TATOEBA_ORI0} {fp.WIKI103_ORI0} > {fp.DAE_ORI0}")
        util.maybe_prompt(fp.DAE_COR0, f"cat {fp.GUTENBERG_COR0} {fp.TATOEBA_COR0} {fp.WIKI103_COR0} > {fp.DAE_COR0}")

        logging.info("[Prepare] 1-2. train")
        util.maybe_prompt(fp.TRAIN_ORI0, f"cat {fp.FCE_TOK_ORI} {fp.LANG8_TOK_ORI} {fp.NUCLE_TOK_ORI} > {fp.TRAIN_ORI0}")
        util.maybe_prompt(fp.TRAIN_COR0, f"cat {fp.FCE_TOK_COR} {fp.LANG8_TOK_COR} {fp.NUCLE_TOK_COR} > {fp.TRAIN_COR0}")

        logging.info("[Prepare] 1-3. finetune")
        util.maybe_prompt(fp.FINETUNE_ORI0, f"cp {fp.NUCLE_TOK_ORI} {fp.FINETUNE_ORI0}")
        util.maybe_prompt(fp.FINETUNE_COR0, f"cp {fp.NUCLE_TOK_COR} {fp.FINETUNE_COR0}")

        logging.info("[Prepare] 1-4. valid")
        util.maybe_prompt(fp.VALID_ORI0, f"cp {fp.CONLL2013_TOK_ORI} {fp.VALID_ORI0}")
        util.maybe_prompt(fp.VALID_COR0, f"cp {fp.CONLL2013_TOK_COR} {fp.VALID_COR0}")

    elif track.TRACK_NUM == 1:
        logging.info("[Prepare] 1-1. pretrain")
        util.maybe_prompt(fp.DAE_ORI1, f"cat {fp.GUTENBERG_ORI1} {fp.TATOEBA_ORI1} {fp.WIKI103_ORI1} > {fp.DAE_ORI1}")
        util.maybe_prompt(fp.DAE_COR1, f"cat {fp.GUTENBERG_COR1} {fp.TATOEBA_COR1} {fp.WIKI103_COR1} > {fp.DAE_COR1}")

        logging.info("[Prepare] 1-2. train")
        util.maybe_prompt(fp.TRAIN_ORI1,
                          f"cat {fp.FCE_TOK_ORI} {fp.LANG8_TOK_ORI} {fp.NUCLE_TOK_ORI} {fp.WI_TRAIN_TOK_ORI} > {fp.TRAIN_ORI1}")
        util.maybe_prompt(fp.TRAIN_COR1,
                          f"cat {fp.FCE_TOK_COR} {fp.LANG8_TOK_COR} {fp.NUCLE_TOK_COR} {fp.WI_TRAIN_TOK_COR} > {fp.TRAIN_COR1}")

        logging.info("[Prepare] 1-3. finetune")
        util.maybe_prompt(fp.FINETUNE_ORI1, f"cp {fp.WI_TRAIN_TOK_ORI} {fp.FINETUNE_ORI1}")
        util.maybe_prompt(fp.FINETUNE_COR1, f"cp {fp.WI_TRAIN_TOK_COR} {fp.FINETUNE_COR1}")

        logging.info("[Prepare] 1-4. valid")
        util.maybe_prompt(fp.VALID_ORI1, f"cp {fp.WI_DEV_TOK_ORI} {fp.VALID_ORI1}")
        util.maybe_prompt(fp.VALID_COR1, f"cp {fp.WI_DEV_TOK_COR} {fp.VALID_COR1}")

    elif track.TRACK_NUM == 3:
        logging.info("[Prepare] 1-1. pretrain")
        util.maybe_prompt(fp.DAE_ORI3, f"cat {fp.GUTENBERG_ORI3} {fp.TATOEBA_ORI3} {fp.WIKI103_ORI3} > {fp.DAE_ORI3}")
        util.maybe_prompt(fp.DAE_COR3, f"cat {fp.GUTENBERG_COR3} {fp.TATOEBA_COR3} {fp.WIKI103_COR3} > {fp.DAE_COR3}")

        logging.info("[Prepare] 1-2. finetune")
        util.maybe_prompt(fp.FINETUNE_ORI3, f"cp {fp.WI_DEV_3K_TOK_ORI} {fp.FINETUNE_ORI3}")
        util.maybe_prompt(fp.FINETUNE_COR3, f"cp {fp.WI_DEV_3K_TOK_COR} {fp.FINETUNE_COR3}")

        logging.info("[Prepare] 1-3. valid")
        util.maybe_prompt(fp.VALID_ORI3, f"cp {fp.WI_DEV_1K_TOK_ORI} {fp.VALID_ORI3}")
        util.maybe_prompt(fp.VALID_COR3, f"cp {fp.WI_DEV_1K_TOK_COR} {fp.VALID_COR3}")


def prepare_binary(databin_path, trainpref, validpref, vocab):
    logging.info(f"[Prepare] 2. create binary data on {databin_path}")
    prompt = f"fairseq-preprocess --source-lang ori --target-lang cor " \
             f"--trainpref {trainpref} --validpref {validpref} " \
             f"--srcdict {vocab} --tgtdict {vocab} --destdir {databin_path}"

    util.maybe_prompt(databin_path, prompt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=int, required=True, help="track number")
    args = parser.parse_args()

    main(args)
