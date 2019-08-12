########################
# CUSTOM MODULE FOR GEC
########################

'''
############

usage:

import lm_scorer
from lm_scorer import LMScorer
from fairseq import options


parser = lm_scorer.get_lm_scorer_parser()
parsed_args = options.parse_args_and_arch(parser)
lmscorer = LMScorer(parsed_args)
score_dict = lmscorer.score(error_sent_lst)

#############

above file need below arguments:

python python_file.py $DATA_BIN \  # dictionary path
  --path $PRETRAINED_LM_CKPT_PATH \  # LM model checkpoint
  --quiet  # not to print all the sentences

##############
'''



import numpy as np
import torch
from collections import namedtuple

from fairseq import options, progress_bar, tasks, utils, tokenizer, data
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module


Batch = namedtuple('Batch', 'srcs tokens lengths')

class LMScorer(object):
    def __init__(self, parsed_args):
        self.args = parsed_args
        import_user_module(parsed_args)
        assert parsed_args.path is not None, '--path required for evaluation'

        print(parsed_args)

        self.use_cuda = torch.cuda.is_available() and not parsed_args.cpu

        self.task = tasks.setup_task(parsed_args)

        # Load ensemble
        print('| loading model(s) from {}'.format(parsed_args.path))
        self.models, args = utils.load_ensemble_for_inference(
            parsed_args.path.split(':'), self.task, model_arg_overrides=eval(parsed_args.model_overrides),
        )

        for model in self.models:
            model.make_generation_fast_()
            if self.use_cuda:
                model.cuda()

        for arg in vars(parsed_args).keys():
            if arg not in {'self_target', 'future_target', 'past_target', 'tokens_per_sample',
                           'output_size_dictionary'}:
                setattr(args, arg, getattr(parsed_args, arg))
        self.task = tasks.setup_task(args)

        self.gen_timer = StopwatchMeter()
        self.scorer = SequenceScorer(self.task.target_dictionary)


    def score_sent(self, line):
        score_dict = self.score([line])
        return score_dict[0]

    def make_batches(self, lines):
        token_lst = [self.task.source_dictionary.encode_line(line, add_if_not_exist=False).long()
                     for line in lines]
        length_lst = torch.LongTensor([tokens.numel() for tokens in token_lst])

        ds = data.TokenBlockDataset(token_lst, length_lst, self.args.tokens_per_sample, pad=self.task.dictionary.pad(),
                                    eos=self.task.dictionary.eos(),
                                    break_mode='eos', include_targets=True)
        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'
        itr = self.task.get_batch_iterator(
            dataset=data.MonolingualDataset(ds, ds.sizes, self.task.dictionary, self.task.target_dictionary,
                                            add_eos_for_other_targets, shuffle=False, targets=self.task.targets),
            max_tokens=self.args.max_tokens or 3000,
            max_sentences=self.args.max_sentences,
            max_positions=utils.resolve_max_positions(*[
                model.max_positions() for model in self.models
            ]),
            num_shards=self.args.num_shards,
            shard_id=self.args.shard_id,
            ignore_invalid_inputs=True,
            num_workers=self.args.num_workers,
        ).next_epoch_itr(shuffle=False)

        return itr



    def score(self, lines):


        batch = self.make_batches(lines)

        sample_score_dict = {}



        # with progress_bar.build_progress_bar(self.args, itr) as t:
        for sample in batch:
            sample_id_lst = sample['id']
            sample = utils.move_to_cuda(sample) if self.use_cuda else sample
            if 'net_input' not in sample:
                continue

            hypos = self.scorer.generate(self.models, sample)

            # print(hypos)


            for sample_id, hypos_i in zip(sample_id_lst, hypos):
                hypo = hypos_i[0]
                pos_scores = hypo['positional_scores']

                inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
                if inf_scores.any():
                    print('| Skipping tokens with inf scores:',
                          self.task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                    pos_scores = pos_scores[(~inf_scores).nonzero()]
                sample_score = pos_scores.sum().cpu()
                count = pos_scores.numel()

                w_lst = []
                word_prob = []
                for i in range(len(hypo['tokens'])):
                    w_ind = hypo['tokens'][i].item()
                    w = self.task.dictionary[w_ind]
                    word_prob.append((w, pos_scores[i].item()))
                    w_lst.append(w)

                sample_score = -sample_score / count

                if not self.args.quiet:
                    if self.args.output_sent:
                        print('H-{}\t{}\t{}'.format(sample_id, sample_score, ' '.join(w_lst)))
                    else:
                        print('H-{}\t{}'.format(sample_id, sample_score))
                sample_score_dict[sample_id.item()] = sample_score.item()
                # print(sample_id, sample_score.item())


        return sample_score_dict


def main(parsed_args):
    lmscorer = LMScorer(parsed_args)
    print(parsed_args.fpath)
    fd = open(parsed_args.fpath, 'r')
    lmscorer.score(fd.read().splitlines())
    fd.close()


def get_lm_scorer_parser(default_task='language_modeling'):
    parser = options.get_parser('Evaluate Language Model', default_task)
    options.add_dataset_args(parser, gen=True)
    options.add_common_eval_args(parser)
    add_lm_scorer_args(parser)
    return parser


def add_lm_scorer_args(parser):
    group = parser.add_argument_group('LM Scorer')
    group.add_argument('--fpath', help='file path')
    group.add_argument('--output-sent', action='store_true')


def cli_main():
    parser = get_lm_scorer_parser()
    args = options.parse_args_and_arch(parser)


    main(args)


if __name__ == '__main__':
    cli_main()
