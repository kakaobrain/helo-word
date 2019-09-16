import os


class Path:

    # set your root path here
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # hunspell
    aff = f"{root}/data/language_model/dicts/en_AU.aff"
    dic = f"{root}/data/language_model/dicts/en_wiki_rev.dic"

    # capital word dict
    cap_word_dic = f"{root}/data/language_model/dicts/cap_words_dic"

    # lm
    lm_path = f"{root}/data/language_model/wiki103.pt"
    lm_databin = f"{root}/data/language_model/data-bin"
    lm_dict = f"{root}/data/language_model/data-bin/dict.txt"

    # postprocess
    parallel_to_m2 = f"{root}/errant/parallel_to_m2.py"

    # scorer
    errant = f"{root}/errant/compare_m2.py"
    m2scorer = f"{root}/data/conll2014/m2scorer/m2scorer.py"


class FilePath(object):
    '''define file and directory names and their paths
       lowercase variables are directories, and uppercase ones are files.'''
    def __init__(self):

        # set your root path here
        self.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.gutenberg = f"{self.root}/data/gutenberg"
        self.tatoeba = f"{self.root}/data/tatoeba"
        self.wiki103 = f"{self.root}/data/wiki103"
        self.bea19 = f'{self.root}/data/bea19'
        self.conll2013 = f'{self.root}/data/conll2013'
        self.conll2014 = f'{self.root}/data/conll2014'
        self.jfleg = f'{self.root}/data/jfleg'
        self.bpe_model = f'{self.root}/data/bpe-model'
        self.parallel = f'{self.root}/data/parallel'
        self.fce = f'{self.bea19}/fce'
        self.wi = f'{self.bea19}/wi+locness'

        self.fce_m2 = f'{self.fce}/m2'
        self.lang8_m2 = f'{self.bea19}/lang8.bea19'
        self.nucle_m2 = f'{self.bea19}/nucle3.3/bea2019'
        self.wi_m2 = f'{self.wi}/m2'

        self.conll2013_m2 = f'{self.conll2013}/release2.3.1/revised/data'
        self.conll2014_m2 = f'{self.conll2014}/conll14st-test-data/noalt'

        # for dae
        self.GUTENBERG_TXT = f"{self.gutenberg}/gutenberg.txt"
        self.TATOEBA_TXT = f"{self.tatoeba}/tatoeba.txt"
        self.WIKI103_TXT = f"{self.wiki103}/wiki103.txt"

        # bpe
        self.BPE_MODEL = f"{self.bpe_model}/gutenberg.model"
        self.BPE_VOCAB = f"{self.bpe_model}/gutenberg.vocab"

        # dae tok
        self.GUTENBERG_ORI1 = f"{self.parallel}/tok/gutenberg.tok.wi.train.ori"
        self.GUTENBERG_COR1 = f"{self.parallel}/tok/gutenberg.tok.wi.train.cor"
        self.TATOEBA_ORI1 = f"{self.parallel}/tok//tatoeba.tok.wi.train.ori"
        self.TATOEBA_COR1 = f"{self.parallel}/tok/tatoeba.tok.wi.train.cor"
        self.WIKI103_ORI1 = f"{self.parallel}/tok/wiki103.tok.wi.train.ori"
        self.WIKI103_COR1 = f"{self.parallel}/tok/wiki103.tok.wi.train.cor"

        self.GUTENBERG_ORI3 = f"{self.parallel}/tok/gutenberg.tok.wi.dev.3k.ori"
        self.GUTENBERG_COR3 = f"{self.parallel}/tok/gutenberg.tok.wi.dev.3k.cor"
        self.TATOEBA_ORI3 = f"{self.parallel}/tok/tatoeba.tok.wi.dev.3k.ori"
        self.TATOEBA_COR3 = f"{self.parallel}/tok/tatoeba.tok.wi.dev.3k.cor"
        self.WIKI103_ORI3 = f"{self.parallel}/tok/wiki103.tok.wi.dev.3k.ori"
        self.WIKI103_COR3 = f"{self.parallel}/tok/wiki103.tok.wi.dev.3k.cor"

        self.GUTENBERG_ORI0 = f"{self.parallel}/tok/gutenberg.tok.nucle.ori"
        self.GUTENBERG_COR0 = f"{self.parallel}/tok/gutenberg.tok.nucle.cor"
        self.TATOEBA_ORI0 = f"{self.parallel}/tok/tatoeba.tok.nucle.ori"
        self.TATOEBA_COR0 = f"{self.parallel}/tok/tatoeba.tok.nucle.cor"
        self.WIKI103_ORI0 = f"{self.parallel}/tok/wiki103.tok.nucle.ori"
        self.WIKI103_COR0 = f"{self.parallel}/tok/wiki103.tok.nucle.cor"

        # raw
        self.FCE_ORI = f"{self.parallel}/raw/fce.ori"
        self.FCE_COR = f"{self.parallel}/raw/fce.cor"
        self.LANG8_ORI = f"{self.parallel}/raw/lang8.ori"
        self.LANG8_COR = f"{self.parallel}/raw/lang8.cor"
        self.NUCLE_ORI = f"{self.parallel}/raw/nucle.ori"
        self.NUCLE_COR = f"{self.parallel}/raw/nucle.cor"
        self.WI_TRAIN_ORI = f"{self.parallel}/raw/wi.train.ori"
        self.WI_TRAIN_COR = f"{self.parallel}/raw/wi.train.cor"
        self.WI_DEV_ORI = f"{self.parallel}/raw/wi.dev.ori"
        self.WI_DEV_COR = f"{self.parallel}/raw/wi.dev.cor"
        self.WI_TEST_ORI = f"{self.parallel}/raw/ABCN.test.bea19.orig"
        # self.WI_TEST_COR = f"{self.parallel}/raw/wi.test.cor"

        self.WI_DEV_3K_ORI = f"{self.parallel}/raw/wi.dev.3k.ori"
        self.WI_DEV_3K_COR = f"{self.parallel}/raw/wi.dev.3k.cor"
        self.WI_DEV_1K_ORI = f"{self.parallel}/raw/wi.dev.1k.ori"
        self.WI_DEV_1K_COR = f"{self.parallel}/raw/wi.dev.1k.cor"

        self.CONLL2013_ORI = f"{self.parallel}/raw/conll2013.ori"
        self.CONLL2013_COR = f"{self.parallel}/raw/conll2013.cor"
        self.CONLL2014_ORI = f"{self.parallel}/raw/conll2014.ori"
        self.CONLL2014_COR = f"{self.parallel}/raw/conll2014.cor"
        self.JFLEG_ORI = f"{self.jfleg}/test/test.src"

        # sp
        self.FCE_SP_ORI = f"{self.parallel}/sp/fce.sp.ori"
        self.LANG8_SP_ORI = f"{self.parallel}/sp/lang8.sp.ori"
        self.NUCLE_SP_ORI = f"{self.parallel}/sp/nucle.sp.ori"
        self.WI_TRAIN_SP_ORI = f"{self.parallel}/sp/wi.train.sp.ori"
        self.WI_DEV_SP_ORI = f"{self.parallel}/sp/wi.dev.sp.ori"
        self.WI_TEST_SP_ORI = f"{self.parallel}/sp/wi.test.sp.ori"

        self.WI_DEV_3K_SP_ORI = f"{self.parallel}/sp/wi.dev.3k.sp.ori"
        self.WI_DEV_1K_SP_ORI = f"{self.parallel}/sp/wi.dev.1k.sp.ori"

        self.CONLL2013_SP_ORI = f"{self.parallel}/sp/conll2013.sp.ori"
        self.CONLL2014_SP_ORI = f"{self.parallel}/sp/conll2014.sp.ori"
        self.JFLEG_SP_ORI = f"{self.parallel}/sp/jfleg.sp.ori"

        # tok
        self.FCE_TOK_ORI = f"{self.parallel}/tok/fce.tok.ori"
        self.FCE_TOK_COR = f"{self.parallel}/tok/fce.tok.cor"
        self.LANG8_TOK_ORI = f"{self.parallel}/tok/lang8.tok.ori"
        self.LANG8_TOK_COR = f"{self.parallel}/tok/lang8.tok.cor"
        self.NUCLE_TOK_ORI = f"{self.parallel}/tok/nucle.tok.ori"
        self.NUCLE_TOK_COR = f"{self.parallel}/tok/nucle.tok.cor"
        self.WI_TRAIN_TOK_ORI = f"{self.parallel}/tok/wi.train.tok.ori"
        self.WI_TRAIN_TOK_COR = f"{self.parallel}/tok/wi.train.tok.cor"
        self.WI_DEV_TOK_ORI = f"{self.parallel}/tok/wi.dev.tok.ori"
        self.WI_DEV_TOK_COR = f"{self.parallel}/tok/wi.dev.tok.cor"
        self.WI_TEST_TOK_ORI = f"{self.parallel}/tok/wi.test.tok.ori"
        # self.WI_TEST_TOK_COR = f"{self.parallel}/tok/wi.test.tok.cor"

        self.WI_DEV_3K_TOK_ORI = f"{self.parallel}/tok/wi.dev.3k.tok.ori"
        self.WI_DEV_3K_TOK_COR = f"{self.parallel}/tok/wi.dev.3k.tok.cor"
        self.WI_DEV_1K_TOK_ORI = f"{self.parallel}/tok/wi.dev.1k.tok.ori"
        self.WI_DEV_1K_TOK_COR = f"{self.parallel}/tok/wi.dev.1k.tok.cor"

        self.CONLL2013_TOK_ORI = f"{self.parallel}/tok/conll2013.tok.ori"
        self.CONLL2013_TOK_COR = f"{self.parallel}/tok/conll2013.tok.cor"
        self.CONLL2014_TOK_ORI = f"{self.parallel}/tok/conll2014.tok.ori"
        self.CONLL2014_TOK_COR = f"{self.parallel}/tok/conll2014.tok.cor"
        self.JFLEG_TOK_ORI = f"{self.parallel}/tok/jfleg.tok.ori"
        # self.JFLEG_TOK_COR = f"{self.parallel}/tok/jfleg.tok.cor"

        # track1
        self.DAE_ORI1 = f"{self.root}/track1/parallel/dae.ori"
        self.DAE_COR1 = f"{self.root}/track1/parallel/dae.cor"
        self.TRAIN_ORI1 = f"{self.root}/track1/parallel/train.ori"
        self.TRAIN_COR1 = f"{self.root}/track1/parallel/train.cor"
        self.FINETUNE_ORI1 = f"{self.root}/track1/parallel/finetune.ori"
        self.FINETUNE_COR1 = f"{self.root}/track1/parallel/finetune.cor"
        self.VALID_ORI1 = f"{self.root}/track1/parallel/valid.ori"
        self.VALID_COR1 = f"{self.root}/track1/parallel/valid.cor"
        self.TEST_ORI1 = f"{self.root}/track1/parallel/test.ori"
        self.TEST_COR1 = f"{self.root}/track1/parallel/test.cor"

        # track3
        self.DAE_ORI3 = f"{self.root}/track3/parallel/dae.ori"
        self.DAE_COR3 = f"{self.root}/track3/parallel/dae.cor"
        self.FINETUNE_ORI3 = f"{self.root}/track3/parallel/finetune.ori"
        self.FINETUNE_COR3 = f"{self.root}/track3/parallel/finetune.cor"
        self.VALID_ORI3 = f"{self.root}/track3/parallel/valid.ori"
        self.VALID_COR3 = f"{self.root}/track3/parallel/valid.cor"
        self.TEST_ORI3 = f"{self.root}/track3/parallel/test.ori"
        self.TEST_COR3 = f"{self.root}/track3/parallel/test.cor"

        # track0
        self.DAE_ORI0 = f"{self.root}/track0/parallel/dae.ori"
        self.DAE_COR0 = f"{self.root}/track0/parallel/dae.cor"
        self.TRAIN_ORI0 = f"{self.root}/track0/parallel/train.ori"
        self.TRAIN_COR0 = f"{self.root}/track0/parallel/train.cor"
        self.FINETUNE_ORI0 = f"{self.root}/track0/parallel/finetune.ori"
        self.FINETUNE_COR0 = f"{self.root}/track0/parallel/finetune.cor"
        self.VALID_ORI0 = f"{self.root}/track0/parallel/valid.ori"
        self.VALID_COR0 = f"{self.root}/track0/parallel/valid.cor"
        self.TEST_ORI0 = f"{self.root}/track0/parallel/test.ori"
        self.TEST_COR0 = f"{self.root}/track0/parallel/test.cor"
        self.TEST_ORI_JFLEG0 = f"{self.root}/track0/parallel/test_jfleg.ori"
        self.TEST_COR_JFLEG0 = f"{self.root}/track0/parallel/test_jfleg.cor"

        # make_dirs
        self.make_dirs()

    def make_dirs(self):
        fnames = [getattr(self, attr) for attr in dir(self) if attr[0].isupper()]
        for fname in fnames:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
