from abc import abstractmethod
import os

from . import util
from .filepath import FilePath


def choice_track(track_num):
    if track_num == 0:
        return Track0()

    if track_num == 1:
        return Track1()

    if track_num == 3:
        return Track3()


class Track:
    def __init__(self, track_num):
        self.fp = FilePath()
        self.TRACK_NUM = track_num
        self.TRACK_PATH = f"{self.fp.root}/track{track_num}"

    @property
    def train_modes(self):
        raise NotImplementedError

    @property
    def subsets(self):
        raise NotImplementedError

    def get_databin_path(self, train_mode):
        assert train_mode in self.train_modes
        return f"{self.TRACK_PATH}/data-bin/{train_mode}"

    def get_ckpt_dir(self, train_mode, model, lr=5e-4, dropout=0.3, seed=None, prev_model_dir=None):

        def _get_ckpt_dir_basename(train_mode, model, lr, dropout, seed, prev_model_dir):
            basenames = []
            if prev_model_dir is not None:
                prev_model_basename = util.get_basename(prev_model_dir, include_path=False, include_extension=False)
                basenames.append(prev_model_basename)

            basename = f"{train_mode}-{model}-lr{lr}-dr{dropout}"
            if seed is not None:
                basename += f"-s{seed}"
            basenames.append(basename)

            return "_".join(basenames)

        ckpt_basename = _get_ckpt_dir_basename(train_mode, model, lr, dropout, seed, prev_model_dir)

        return f"{self.TRACK_PATH}/ckpt/{ckpt_basename}"

    def get_output_dir(self, ckpt):
        def _get_output_dir_from_ckpt_dir(ckpt_dir):
            dir_basename = util.get_basename(ckpt_dir, include_path=False)
            return f"{self.TRACK_PATH}/outputs/{dir_basename}"

        def _get_output_dir_from_ckpt_fpath(ckpt_fpath):
            ckpts = ckpt_fpath.split(':')

            # not ensemble
            if len(ckpts) == 1:
                ckpt_dir = os.path.dirname(ckpt_fpath)
                return _get_output_dir_from_ckpt_dir(ckpt_dir)

            # ensemble
            else:
                dirname_lst = []
                for ckpt in ckpts:
                    ckpt_dir = os.path.dirname(ckpt)
                    ckpt_dir_basename = util.get_basename(ckpt_dir, include_path=False)
                    dirname_lst.append(ckpt_dir_basename)
                return f"{self.TRACK_PATH}/outputs/" + ":".join(dirname_lst)

        if os.path.isdir(ckpt):
            return _get_output_dir_from_ckpt_dir(ckpt)
        else:
            return _get_output_dir_from_ckpt_fpath(ckpt)

    @abstractmethod
    def get_subset_datapath(self, subset):
        raise NotImplementedError

    @staticmethod
    def get_model_config(model, lr, dropout, max_epoch, seed, reset=False):
        assert model in ['base', 'copy', 't2t']
        if model == 'base':
            model_config = f"--arch transformer --share-all-embeddings " \
                 f"--optimizer adam --lr {lr} --label-smoothing 0.1 --dropout {dropout} " \
                 f"--max-tokens 4000 --min-lr '1e-09' --lr-scheduler inverse_sqrt " \
                 f"--weight-decay 0.0001 --criterion label_smoothed_cross_entropy " \
                 f"--max-epoch {max_epoch} --warmup-updates 4000 --warmup-init-lr '1e-07' --max-tokens 4000 " \
                 f"--adam-betas '(0.9, 0.98)' --save-interval-updates 5000 "

        elif model == 'copy':
            model_config = f"--ddp-backend=no_c10d --arch copy_augmented_transformer " \
                f"--update-freq 8 --alpha-warmup 10000 --optimizer adam --lr {lr} " \
                f"--dropout {dropout} --max-tokens 4000 --min-lr '1e-09' --save-interval-updates 5000 " \
                f"--lr-scheduler inverse_sqrt --weight-decay 0.0001 --max-epoch {max_epoch} " \
                f"--warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' "

        else:   # model == 't2t':

            model_config = f"--arch transformer_wmt_en_de_big_t2t --share-all-embeddings " \
                           f"--criterion label_smoothed_cross_entropy --label-smoothing 0.1 " \
                           f"--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 " \
                           f"--lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --max-epoch {max_epoch} " \
                           f"--warmup-updates 4000 --lr {lr} --min-lr '1e-09' --dropout {dropout} " \
                           f"--weight-decay 0.0 --max-tokens 4000 --save-interval-updates 5000 "

        if seed is not None:
            model_config += f"--seed {seed} "
        if reset:
            model_config += f"--reset-optimizer --reset-lr-scheduler "

        return model_config


class Track0(Track):
    def __init__(self):
        super(Track0, self).__init__(0)

    train_modes = ['pretrain', 'train', 'finetune']
    subsets = ['valid', 'conll2014', 'jfleg']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI0)[0]
        elif train_mode == 'train':
            trainpref = os.path.splitext(self.fp.TRAIN_ORI0)[0]
        else:  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI0)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI0)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.conll2013_m2}/official-preprocessed.m2"
            ori_path = self.fp.CONLL2013_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = "m2scorer"

        elif subset == 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = "m2scorer"

        else:  # 'jfleg':
            gold_m2 = None
            ori_path = self.fp.JFLEG_ORI
            ori_bpe_path = self.fp.JFLEG_TOK_ORI
            gen_subset = None
            scorer_type = "jfleg"

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type


class Track1(Track):
    def __init__(self):
        super(Track1, self).__init__(1)

    train_modes = ['pretrain', 'train', 'finetune', 'dev']
    subsets = ['valid', 'test', 'conll2014']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI1)[0]
        elif train_mode == 'train':
            trainpref = os.path.splitext(self.fp.TRAIN_ORI1)[0]
        elif train_mode == 'finetune':  # finetune
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI1)[0]
        else:
            trainpref = os.path.splitext(self.fp.VALID_ORI1)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI1)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.m2"
            ori_path = self.fp.WI_DEV_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'test':
            gold_m2 = None
            ori_path = self.fp.WI_TEST_ORI
            ori_bpe_path = self.fp.WI_TEST_TOK_ORI
            gen_subset = None
            scorer_type = None

        else:  # 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type


class Track3(Track):
    def __init__(self):
        super(Track3, self).__init__(3)

    train_modes = ['pretrain', 'finetune']
    subsets = ['valid', 'test', 'conll2014']

    def get_pref(self, train_mode):
        assert train_mode in self.train_modes
        if train_mode == 'pretrain':
            trainpref = os.path.splitext(self.fp.DAE_ORI3)[0]
        else:
            trainpref = os.path.splitext(self.fp.FINETUNE_ORI3)[0]
        validpref = os.path.splitext(self.fp.VALID_ORI3)[0]
        return trainpref, validpref

    def get_subset_datapath(self, subset):
        assert subset in self.subsets

        if subset == 'valid':
            gold_m2 = f"{self.fp.wi_m2}/ABCN.dev.gold.bea19.1k.m2"
            ori_path = self.fp.WI_DEV_1K_ORI
            ori_bpe_path = None
            gen_subset = "valid"
            scorer_type = 'errant'

        elif subset == 'test':
            gold_m2 = None
            ori_path = self.fp.WI_TEST_ORI
            ori_bpe_path = self.fp.WI_TEST_TOK_ORI
            gen_subset = None
            scorer_type = None

        else:  # 'conll2014':
            gold_m2 = f"{self.fp.conll2014_m2}/official-2014.combined.m2"
            ori_path = self.fp.CONLL2014_ORI
            ori_bpe_path = self.fp.CONLL2014_TOK_ORI
            gen_subset = None
            scorer_type = 'm2scorer'

        return gold_m2, ori_path, ori_bpe_path, gen_subset, scorer_type
