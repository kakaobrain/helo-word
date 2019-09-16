import logging
import os
from glob import glob
import argparse
from gec import filepath, word_tokenize, bpe, perturb, m2, spell


logging.basicConfig(level=logging.INFO)


def maybe_do(fp, func, inputs):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        func(*inputs)

def maybe_download(dir, cmd):
    if os.listdir(dir) != []:
        logging.info(f"skip this step as {dir} is NOT empty")
    else:
        print(cmd)
        for sub_cmd in cmd.split("&"):
            print(sub_cmd)
            os.system(sub_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 1. word-tokenize
    parser.add_argument("--max_tokens", type=int, default=150,
                        help="Maximum number of tokens in a sample")

    # 2. train bpe model
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="vocabulary size")

    # 3. perturbation -> bpe-tokenize
    parser.add_argument("--min_cnt", type=int, default=4)
    parser.add_argument("--word_change_prob", type=float, default=.9)
    parser.add_argument("--type_change_prob", type=float, default=.1)
    parser.add_argument("--n_epochs", type=int, nargs="+", default=[1, 12, 5],
                        help="list of n_epochs of gutenberg, tatoeba, and wiki103")

    args = parser.parse_args()

    fp = filepath.FilePath()
    fp.make_dirs()

    logging.info("STEP 0. Download data")
    logging.info("STEP 0-1. Download Gutenberg Text")
    maybe_download(fp.gutenberg,
                   f"gdown https://drive.google.com/uc?id=0B2Mzhc7popBga2RkcWZNcjlRTGM & "
                   f"unzip Gutenberg.zip -d {fp.gutenberg} & "
                   f"rm Gutenberg.zip")

    logging.info("STEP 0-2. Download Tatoeba")
    maybe_download(fp.tatoeba,
                   f"wget http://downloads.tatoeba.org/exports/sentences.tar.bz2 & "
                   f"tar -C {fp.tatoeba} -xjf sentences.tar.bz2 &"
                   f"rm sentences.tar.bz2")

    logging.info("STEP 0-3. Download Wiki-103")
    maybe_download(fp.wiki103,
                   f"wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip & "
                   f"sleep 10s & "
                   f"unzip wikitext-103-v1.zip -d {fp.wiki103} & "
                   f"mv {fp.wiki103}/wikitext-103/wiki.train.tokens {fp.wiki103}/wiki.train.tokens &"
                   f"rm wikitext-103-v1.zip"
                   )


    # TODO: make these directories at filepath.py
    # make directories
    for dir_name in [f"{fp.bea19}", f"{fp.wi}", f"{fp.fce}", f"{fp.conll2013}", f"{fp.conll2014}"]:
        try:
            os.mkdir(dir_name)
        except: pass

    logging.info("STEP 0-4. Download FCE")
    maybe_download(f"{fp.bea19}/fce",
                   f"wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz & "
                   f"tar -C {fp.bea19} -xvzf fce_v2.1.bea19.tar.gz $"
                   f"rm fce_v2.1.bea19.tar.gz")

    logging.info("STEP 0-5. Download WI+LOCNESS")
    maybe_download(f"{fp.bea19}/wi+locness",
                   f"wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz & "
                   f"tar -C {fp.bea19} -xvzf wi+locness_v2.1.bea19.tar.gz &"
                   f"rm wi+locness_v2.1.bea19.tar.gz & "
                   f"wget https://www.cl.cam.ac.uk/research/nl/bea2019st/data/ABCN.bea19.test.orig & "
                   f"mv ABCN.bea19.test.orig {fp.WI_TEST_ORI}")

    logging.info("STEP 0-6. Download LANG8")
    logging.info(f"NO PUBLIC DATA AVAILABLE.\n "
                 f"Please visit 'https://www.cl.cam.ac.uk/research/nl/bea2019st/' to obtain data and extract file to {fp.nucle_m2}/*m2")


    logging.info("STEP 0-7. Download Conll 2013, 2014")
    maybe_download(fp.conll2013,
                   f"wget https://www.comp.nus.edu.sg/~nlp/conll13st/release2.3.1.tar.gz & "
                   f"tar -C {fp.conll2013} -xvzf &"
                   f"rm release2.3.1.tar.gz")

    maybe_download(fp.conll2014,
                   f"wget https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz & "
                   f"tar -C {fp.conll2014} -xvzf conll14st-test-data.tar.gz &"
                   f"rm conll14st-test-data.tar.gz")

    logging.info("STEP 0-8. Download language model")
    os.makedirs(os.path.dirname(filepath.Path.lm_path), exist_ok=True)
    os.makedirs(os.path.dirname(filepath.Path.lm_dict), exist_ok=True)
    maybe_download(filepath.Path.lm_databin,
                   f"wget https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2 & "
                   f"tar -xvf wiki103_fconv_lm.tar.bz2 & "
                   f"mv wiki103.pt {filepath.Path.lm_path} & "
                   f"mv dict.txt {filepath.Path.lm_dict} & "
                   f"rm wiki103_fconv_lm.tar.bz2")

    # logging.info("STEP 0-8. Download M2 Scorer")
    # maybe_download(fp.m2scorer, "wget https://www.comp.nus.edu.sg/~nlp/sw/m2scorer.tar.gz")
    # maybe_download(fp.errant, "git clone https://github.com/chrisjbryant/errant.git")

    logging.info("STEP 1. Word-tokenize the original files and merge them")
    logging.info("STEP 1-1. gutenberg")
    fpaths = sorted(glob(f'{fp.gutenberg}/Gutenberg/txt/*.txt'))
    maybe_do(fp.GUTENBERG_TXT, word_tokenize.gutenberg,
             (fpaths, fp.GUTENBERG_TXT, args.max_tokens))

    logging.info("STEP 1-2. tatoeba")
    fpath = f'{fp.tatoeba}/sentences.csv'
    maybe_do(fp.TATOEBA_TXT, word_tokenize.tatoeba,
             (fpath, fp.TATOEBA_TXT, args.max_tokens))

    logging.info("STEP 1-3. wiki103")
    fpath = f'{fp.wiki103}/wiki.train.tokens'
    maybe_do(fp.WIKI103_TXT, word_tokenize.wiki103,
             (fpath, fp.WIKI103_TXT, args.max_tokens))

    logging.info("STEP 2. Train bpe model")
    maybe_do(fp.BPE_MODEL, bpe.train,
             (fp.GUTENBERG_TXT, fp.BPE_MODEL.replace(".model", ""), args.vocab_size, 1.0, 'bpe'))

    logging.info("STEP 3. Split wi.dev into wi.dev.3k and wi.dev.1k")
    fpaths = sorted(glob(f'{fp.wi_m2}/*.dev.gold.bea19.m2'))
    wi_dev_3k_m2 = f'{fp.wi_m2}/ABCN.dev.gold.bea19.3k.m2'
    wi_dev_1k_m2 = f'{fp.wi_m2}/ABCN.dev.gold.bea19.1k.m2'
    maybe_do(wi_dev_3k_m2, m2.split_m2,
             (fpaths, wi_dev_3k_m2, wi_dev_1k_m2, 0.75))

    logging.info("STEP 4. Perturb and make parallel files")
    for track_no in ("1", "3", "0"):
        logging.info(f"Track {track_no}")
        logging.info("STEP 4-1. writing perturbation scenario")
        if track_no=="1":
            dir = f"{fp.wi_m2}/*train*.m2"
        elif track_no=="3":
            dir = f"{fp.wi_m2}/*dev.*3k*.m2"
        else:
            dir = f"{fp.nucle_m2}/*nucle*.m2"
        word2ptbs = perturb.make_word2ptbs(sorted(glob(dir)), args.min_cnt)

        logging.info("STEP 4-2. gutenberg")
        maybe_do(eval(f"fp.GUTENBERG_ORI{track_no}"), perturb.do,
                 (word2ptbs, fp.BPE_MODEL, fp.GUTENBERG_TXT,
                  eval(f"fp.GUTENBERG_ORI{track_no}"), eval(f"fp.GUTENBERG_COR{track_no}"), args.n_epochs[0],
                  args.word_change_prob, args.type_change_prob))

        logging.info("STEP 4-3. tatoeba")
        maybe_do(eval(f"fp.TATOEBA_ORI{track_no}"), perturb.do,
                 (word2ptbs, fp.BPE_MODEL, fp.TATOEBA_TXT,
                  eval(f"fp.TATOEBA_ORI{track_no}"), eval(f"fp.TATOEBA_COR{track_no}"), args.n_epochs[1],
                  args.word_change_prob, args.type_change_prob))

        logging.info("STEP 4-4. wiki103")
        maybe_do(eval(f"fp.WIKI103_ORI{track_no}"), perturb.do,
                 (word2ptbs, fp.BPE_MODEL, fp.WIKI103_TXT,
                  eval(f"fp.WIKI103_ORI{track_no}"), eval(f"fp.WIKI103_COR{track_no}"), args.n_epochs[2],
                  args.word_change_prob, args.type_change_prob))

    logging.info("STEP 5. m2 to parallel")
    logging.info("STEP 5-1. fce")
    maybe_do(fp.FCE_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.fce_m2}/*m2')), fp.FCE_ORI, fp.FCE_COR, False, True))

    logging.info("STEP 5-2. lang8")
    maybe_do(fp.LANG8_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.lang8_m2}/*m2')), fp.LANG8_ORI, fp.LANG8_COR, True, True))

    logging.info("STEP 5-3. nucle")
    maybe_do(fp.NUCLE_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.nucle_m2}/*m2')), fp.NUCLE_ORI, fp.NUCLE_COR, False, True))

    logging.info("STEP 5-4. wi train")
    maybe_do(fp.WI_TRAIN_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.wi_m2}/*train*m2')), fp.WI_TRAIN_ORI, fp.WI_TRAIN_COR, False, True))

    logging.info("STEP 5-5. wi dev")
    maybe_do(fp.WI_DEV_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.wi_m2}/ABCN.dev.gold.bea19.m2')), fp.WI_DEV_ORI, fp.WI_DEV_COR, False, False))

    # logging.info("STEP 5-6. wi test")
    # if os.path.exists(WI_TEST_ORI): logging.info(f"skip this step as {WI_TEST_ORI} already exists.")
    # else: m2.m2_to_parallel(glob(f'{wi_m2}/*test*m2'), WI_TEST_ORI, WI_TEST_COR, False, True)

    logging.info("STEP 5-7. wi dev 3k. For track 3 only.")
    maybe_do(fp.WI_DEV_3K_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.wi_m2}/ABCN.dev.gold.bea19.3k.m2')), fp.WI_DEV_3K_ORI, fp.WI_DEV_3K_COR, False, False))

    logging.info("STEP 5-8. wi dev 1k. For track 3 only.")
    maybe_do(fp.WI_DEV_1K_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.wi_m2}/ABCN.dev.gold.bea19.1k.m2')), fp.WI_DEV_1K_ORI, fp.WI_DEV_1K_COR, False, False))

    logging.info("STEP 5-9. conll2013. For track 0 only.")
    maybe_do(fp.CONLL2013_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.conll2013_m2}/official-preprocessed.m2')), fp.CONLL2013_ORI, fp.CONLL2013_COR, False, False))

    logging.info("STEP 5-10. conll2014. For track 0 only.")
    maybe_do(fp.CONLL2014_ORI, m2.m2_to_parallel,
             (sorted(glob(f'{fp.conll2014_m2}/official-2014.combined.m2')), fp.CONLL2014_ORI, fp.CONLL2014_COR, False, False))

    logging.info("STEP 6. spell-check")
    logging.info("STEP 6-1. fce")
    maybe_do(fp.FCE_SP_ORI, spell.check, (fp.FCE_ORI, fp.FCE_SP_ORI))

    logging.info("STEP 6-2. lang8")
    maybe_do(fp.LANG8_SP_ORI, spell.check, (fp.LANG8_ORI, fp.LANG8_SP_ORI))

    logging.info("STEP 6-3. nucle")
    maybe_do(fp.NUCLE_SP_ORI, spell.check, (fp.NUCLE_ORI, fp.NUCLE_SP_ORI))

    logging.info("STEP 6-4. wi train")
    maybe_do(fp.WI_TRAIN_SP_ORI, spell.check, (fp.WI_TRAIN_ORI, fp.WI_TRAIN_SP_ORI))

    logging.info("STEP 6-5. wi dev")
    maybe_do(fp.WI_DEV_SP_ORI, spell.check, (fp.WI_DEV_ORI, fp.WI_DEV_SP_ORI))

    logging.info("STEP 6-6. wi test")
    maybe_do(fp.WI_TEST_SP_ORI, spell.check, (fp.WI_TEST_ORI, fp.WI_TEST_SP_ORI))

    logging.info("STEP 6-7. wi dev 3k")
    maybe_do(fp.WI_DEV_3K_SP_ORI, spell.check, (fp.WI_DEV_3K_ORI, fp.WI_DEV_3K_SP_ORI))

    logging.info("STEP 6-8. wi dev 1k")
    maybe_do(fp.WI_DEV_1K_SP_ORI, spell.check, (fp.WI_DEV_1K_ORI, fp.WI_DEV_1K_SP_ORI))

    logging.info("STEP 6-9. conll 2013")
    maybe_do(fp.CONLL2013_SP_ORI, spell.check, (fp.CONLL2013_ORI, fp.CONLL2013_SP_ORI))

    logging.info("STEP 6-10. conll 2014")
    maybe_do(fp.CONLL2014_SP_ORI, spell.check, (fp.CONLL2014_ORI, fp.CONLL2014_SP_ORI))

    #
    logging.info("STEP 7. bpe-tokenize")
    logging.info("STEP 7-1. fce")
    maybe_do(fp.FCE_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.FCE_SP_ORI, fp.FCE_TOK_ORI))
    maybe_do(fp.FCE_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.FCE_COR, fp.FCE_TOK_COR))

    logging.info("STEP 7-2. lang8")
    maybe_do(fp.LANG8_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.LANG8_SP_ORI, fp.LANG8_TOK_ORI))
    maybe_do(fp.LANG8_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.LANG8_COR, fp.LANG8_TOK_COR))

    logging.info("STEP 7-3. nucle")
    maybe_do(fp.NUCLE_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.NUCLE_SP_ORI, fp.NUCLE_TOK_ORI))
    maybe_do(fp.NUCLE_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.NUCLE_COR, fp.NUCLE_TOK_COR))

    logging.info("STEP 7-4. wi train")
    maybe_do(fp.WI_TRAIN_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_TRAIN_SP_ORI, fp.WI_TRAIN_TOK_ORI))
    maybe_do(fp.WI_TRAIN_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_TRAIN_COR, fp.WI_TRAIN_TOK_COR))

    logging.info("STEP 7-5. wi dev")
    maybe_do(fp.WI_DEV_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_SP_ORI, fp.WI_DEV_TOK_ORI))
    maybe_do(fp.WI_DEV_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_COR, fp.WI_DEV_TOK_COR))

    logging.info("STEP 7-6. wi test")
    maybe_do(fp.WI_TEST_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_TEST_SP_ORI, fp.WI_TEST_TOK_ORI))
    # maybe_do(fp.WI_TEST_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_TEST_COR, fp.WI_TEST_TOK_COR))

    logging.info("STEP 7-7. wi dev 3k")
    maybe_do(fp.WI_DEV_3K_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_3K_SP_ORI, fp.WI_DEV_3K_TOK_ORI))
    maybe_do(fp.WI_DEV_3K_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_3K_COR, fp.WI_DEV_3K_TOK_COR))

    logging.info("STEP 7-8. wi dev 1k")
    maybe_do(fp.WI_DEV_1K_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_1K_SP_ORI, fp.WI_DEV_1K_TOK_ORI))
    maybe_do(fp.WI_DEV_1K_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.WI_DEV_1K_COR, fp.WI_DEV_1K_TOK_COR))

    logging.info("STEP 7-9. conll2013")
    maybe_do(fp.CONLL2013_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.CONLL2013_SP_ORI, fp.CONLL2013_TOK_ORI))
    maybe_do(fp.CONLL2013_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.CONLL2013_COR, fp.CONLL2013_TOK_COR))

    logging.info("STEP 7-10. conll2014")
    maybe_do(fp.CONLL2014_TOK_ORI, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.CONLL2014_SP_ORI, fp.CONLL2014_TOK_ORI))
    maybe_do(fp.CONLL2014_TOK_COR, bpe.bpe_tokenize, (fp.BPE_MODEL, fp.CONLL2014_COR, fp.CONLL2014_TOK_COR))


