###############################################################################
# CUSTOM MODULE FOR GEC
###############################################################################

#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Translate pre-processed data with a trained model.

**This custom generate function is only suited for the GEC task using
copy-augmented transformers. The task argument defaults to 'gec'.**

**Modification for copy-augmented transformers: optionally replace <unk>'s
with source tokens that got the highest copy scores.**

Search 'MODIFIED' for all changes made.
"""

import torch

from fairseq import bleu, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.utils import import_user_module


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    import_user_module(args)

    """
    MODIFIED: The GEC task uses token-labeled raw text datasets, which 
    require raw text to be used.
    """
    assert args.raw_text, \
        f"--raw-text option is required for copy-based generation."

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    has_copy_scores = True
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            """
            MODIFIED: Use copy scores to replace <unk>'s with raw source words.
            
            use_copy_scores may be False with non-copy-based transformers that
            only use edit labels (e.g., transformer_aux_el and transformer_el).
            """
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            use_copy_scores = hypos[0][0].get('copy_scores', None) is not None
            if has_copy_scores and not use_copy_scores:
                print("| generate_or_copy.py | INFO | "
                      "Model does not include copy scores. "
                      "Generating hypotheses without replacing UNKs.")
                has_copy_scores = False
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                """
                MODIFIED: Replace <unk>s with raw source tokens. 
                This is analogous to the case where align_dict is provided
                in the original generate.py.
                """
                rawtext_dataset = task.dataset(args.gen_subset)
                src_str = rawtext_dataset.src.get_original_text(sample_id)
                tokenized_src_str = rawtext_dataset.src_dict.string(
                    src_tokens, bpe_symbol=args.remove_bpe
                )
                target_str = rawtext_dataset.tgt.get_original_text(sample_id)

                if not args.quiet:
                    if src_dict is not None:
                        # Raw source text
                        print('S-{}\t{}'.format(sample_id, src_str))
                        # Tokenized source text
                        print('K-{}\t{}'.format(sample_id, tokenized_src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for k, hypo in enumerate(hypos[i][:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    """
                    MODIFIED: Replace predicted <unk>s with the source token
                    that received the highest score.
                    """
                    raw_src_tokens = src_str.split()
                    final_hypo_tokens_str = []
                    for tgt_position, hypo_token in enumerate(hypo_tokens):
                        if use_copy_scores and hypo_token == tgt_dict.unk():
                            # See sequence_copygenerator.py#L292 for details.
                            copy_scores = hypo['copy_scores'][:, tgt_position].cpu()
                            assert len(copy_scores) - 1 == len(raw_src_tokens), \
                                f"length of copy scores do not match input source tokens " \
                                f"(copy_scores: {copy_scores}, raw_src_tokens: {raw_src_tokens})"
                            src_position = torch.argmax(copy_scores).item()
                            # Don't copy if attending to an EOS (not ideal).
                            if src_position == len(raw_src_tokens):
                                print("WARNING: copy score highest at EOS.")
                            else:
                                final_hypo_tokens_str.append(
                                    raw_src_tokens[src_position]
                                )
                            print('U-{}\t{}\t{}'.format(
                                sample_id,
                                tgt_position,
                                ' '.join(map(
                                    lambda x: '{:.4f}'.format(x),
                                    copy_scores.tolist(),
                                )),
                            ))
                        else:
                            final_hypo_tokens_str.append(tgt_dict[hypo_token])

                    # Note: raw input tokens could be included here.
                    final_hypo_str = ' '.join(
                        [token for token in final_hypo_tokens_str
                         if token != tgt_dict.eos_word]
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], final_hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    # Score only the top hypothesis
                    if has_target and k == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))
    return scorer


def cli_main():
    """
    MODIFIED: task defaults to gec
    """
    parser = options.get_generation_parser(default_task='gec')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
