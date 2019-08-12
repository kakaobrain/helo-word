import numpy as np
import random
import argparse
import re
from tqdm import tqdm


ERROR_TYPES = ['M:ADJ', 'M:ADV', 'M:CONJ', 'M:CONTR', 'M:DET', 'M:NOUN', 'M:NOUN:POSS', 'M:OTHER', 'M:PART', 'M:PREP',
               'M:PRON', 'M:PUNCT', 'M:VERB', 'M:VERB:FORM', 'M:VERB:TENSE', 'R:ADJ', 'R:ADJ:FORM', 'R:ADV', 'R:CONJ',
               'R:CONTR', 'R:DET', 'R:MORPH', 'R:NOUN', 'R:NOUN:INFL', 'R:NOUN:NUM', 'R:NOUN:POSS', 'R:ORTH', 'R:OTHER',
               'R:PART', 'R:PREP', 'R:PRON', 'R:PUNCT', 'R:SPELL', 'R:VERB', 'R:VERB:FORM', 'R:VERB:INFL', 'R:VERB:SVA',
               'R:VERB:TENSE', 'R:WO', 'U:ADJ', 'U:ADV', 'U:CONJ', 'U:CONTR', 'U:DET', 'U:NOUN', 'U:NOUN:POSS',
               'U:OTHER', 'U:PART', 'U:PREP', 'U:PRON', 'U:PUNCT', 'U:VERB', 'U:VERB:FORM', 'U:VERB:TENSE']


def parse(report):
    # get summary
    text = open(report, 'r').read()
    srch = re.search("(?s)\nTP[^\n]+F0\.5\n([^\n]+)", text)
    if srch is None:
        raise ValueError("There is no summary")
    TP, FP, FN, Prec, Rec, Fscore = srch.group(1).split()
    TP, FP, FN, Prec, Rec, Fscore = int(TP), int(FP), int(FN), float(Prec), float(Rec), float(Fscore)

    # get edits
    error_type2scores = dict()
    for line in text.splitlines():
        if line[:2] in ("U:", "M:", "R:"):
            error_type, tp, fp, fn, _, _, _ = line.strip().split()
            error_type2scores[error_type] = int(tp), int(fp), int(fn)

    return TP, FP, FN, Prec, Rec, Fscore, error_type2scores

def get_score(error_type2scores, dropped_error_types=[]):
    '''
    error_type2scores: e.g., 'U:VERB' : (4, 17, 25)
    dropped_error_types: list.

    Returns
        precision, recall, fscore
    '''
    TP, FP, FN = 0, 0, 0
    for error_type, (tp, fp, fn) in error_type2scores.items():
        if error_type in dropped_error_types:
            fn += tp
            FN += fn
        else:
            TP += tp
            FP += fp
            FN += fn

    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    fscore = (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall)

    precision = round(precision, 4)
    recall = round(recall, 4)
    fscore = round(fscore, 4)

    return TP, FP, FN, precision, recall, fscore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str, required=True,
                        help="report file path")
    parser.add_argument("--max_error_types", type=int, default=20)
    parser.add_argument("--n_simulations", type=int, default=100000)

    hp = parser.parse_args()

    TP, FP, FN, Prec, Rec, Fscore, error_type2scores = parse(hp.report)

    # simulation
    _TP, _FP, _FN = 0, 0, 0
    _precision, _recall, _fscore = 0, 0, 0
    best = ""
    for _ in tqdm(range(hp.n_simulations)):
        n_dropped_error_types = random.randint(0, hp.max_error_types)
        dropped_error_types = np.random.choice(ERROR_TYPES, n_dropped_error_types, replace=False)
        dropped_error_types = dropped_error_types.tolist()

        tp, fp, fn, precision, recall, fscore = get_score(error_type2scores, dropped_error_types)
        if fscore > _fscore:
            _precision, _recall, _fscore = precision, recall, fscore
            _TP, _FP, _FN = tp, fp, fn
            best = dropped_error_types

    print(f"Best results after {hp.n_simulations} times of simulation")
    print(f"dropped error types: {best}")
    print(f"TP: {TP} -> {_TP}")
    print(f"FP: {FP} -> {_FP}")
    print(f"FN: {FN} -> {_FN}")

    print(f"precision: {Prec} -> {_precision}")
    print(f"recall: {Rec} -> {_recall}")
    print(f"fscore: {Fscore} -> {_fscore}")