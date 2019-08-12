import hunspell

import copy
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from lm_scorer import LMScorer
from .filepath import Path
from .fix_tokenization_errors import fix

verbose = False
NUM_OF_SUGGESTIONS = 16

SUGGESTION_CACHE = {}
CAPITAL_WORDS_DIC = {}


def load_lm(lm_path, data_bin):
    args = argparse.Namespace(
        path=lm_path,
        data=data_bin,
        fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0,
        fp16_scale_window=None, fpath=None, future_target=False,
        gen_subset='test', lazy_load=False, log_format=None, log_interval=1000,
        max_sentences=None, max_tokens=None, memory_efficient_fp16=False,
        min_loss_scale=0.0001, model_overrides='{}', no_progress_bar=True,
        num_shards=1, num_workers=4, output_dictionary_size=-1,
        output_sent=False, past_target=False,
        quiet=True, raw_text=False, remove_bpe=None, sample_break_mode=None,
        seed=1, self_target=False, shard_id=0, skip_invalid_size_inputs_valid_test=False,
        task='language_modeling', tensorboard_logdir='', threshold_loss_scale=None,
        tokens_per_sample=1024, user_dir=None, cpu=False)
    return LMScorer(args)

def load_capital_dictionary(cap_word_dic):
    with open(cap_word_dic) as f:
        for line in f.readlines():
            line = line.strip("\n")
            keyword, correction = line.split(",")
            CAPITAL_WORDS_DIC[keyword] = correction


def suggest(text, is_first, speller, n=NUM_OF_SUGGESTIONS):
    if (text, is_first) in SUGGESTION_CACHE:
        return SUGGESTION_CACHE[(text, is_first)]

    if text.lower() in CAPITAL_WORDS_DIC:
        suggestions = [CAPITAL_WORDS_DIC[text.lower()]]

    else:
        suggestions = []

    is_upper = text[0].isupper()

    if speller.spell(text):
        SUGGESTION_CACHE[(text, is_first)] = suggestions
        return suggestions

    if text.isalpha():
        if is_upper:
            pass

        elif is_first:
            raw_suggestions = speller.suggest(text)
            for suggestion in raw_suggestions:
                suggestions.append(suggestion[0].upper() + suggestion[1:])

        else:
            raw_suggestions = speller.suggest(text)

            for suggestion in raw_suggestions:
                if suggestion[0].isupper():
                    continue
                suggestions.append(suggestion)

            if len(suggestions)==0:
                suggestions = []
            else:
                suggestions = suggestions[:n]

    if len(suggestions)==0:
        SUGGESTION_CACHE[(text, is_first)] = []
        return []
    else:
        returns = []
        for suggestion in suggestions:
            if '-' in suggestion:
                pass
            else:
                returns.append(suggestion)

        if len(returns) == 0:
            SUGGESTION_CACHE[(text, is_first)] = [text]
            return [text]
        else:
            SUGGESTION_CACHE[(text, is_first)] = returns
            return returns

def spellcheck(model, fin, fout, speller):

    with open(fout, 'w') as fout:
        lines = open(fin, 'r').read().splitlines()
        for line in tqdm(lines):
            corrected_sents = []
            sents = sent_tokenize(line)
            for sent in sents:
                sent = fix(sent)
                tokens = copy.deepcopy(sent).split()
                for i, word in enumerate(tokens):
                    if i==0:
                        is_first = True
                    else:
                        is_first = False

                    suggestions = suggest(word, is_first, speller)
                    if len(suggestions) == 0: continue
                    if len(suggestions) == 1:
                        if suggestions[0] == word: continue

                    candidates = []
                    for suggestion in suggestions:
                        tokens[i] = suggestion
                        candidates.append(" ".join(tokens))

                    # inspect
                    winner = ""

                    # For Neural Net
                    min_score = 1000

                    scores = model.score(candidates)
                    # scores = {idx: idx for idx in range(len(candidates))}

                    for idx, s in zip(range(len(candidates)), suggestions):
                        # print(s, score)
                        score = scores[idx]
                        if score < min_score:
                            min_score = score
                            winner = s

                    if verbose:
                        print("ORIGINAL: " + sent)
                        print("TARGET: " + word.text)
                        print("SUGGEST: " + ','.join(suggestions))
                        for idx, candidates in zip(range(len(scores)), candidates):
                            print("{}: ".format(scores[idx]) + candidates)
                        print("WINNER: " + winner)
                        input()

                    tokens[i] = winner.replace(" ", "_")
                corrected_sents.append(" ".join(tokens).replace("_", " "))
            fout.write(" ".join(corrected_sents) + "\n")

def check(fin, fout,
         aff=Path.aff,
         dic=Path.dic,
         lm_path=Path.lm_path,
         data_bin=Path.lm_databin,
         cap_word_dic=Path.cap_word_dic
         ):

    model = load_lm(lm_path, data_bin)
    load_capital_dictionary(cap_word_dic)

    speller = hunspell.HunSpell(dic, aff)

    spellcheck(model, fin, fout, speller=speller)
