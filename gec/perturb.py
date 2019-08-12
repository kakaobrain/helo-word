import random
import re, os
import sentencepiece as spm
from tqdm import tqdm
from collections import Counter
from nltk import pos_tag
from pattern3.en import conjugate, pluralize, singularize
import logging
import multiprocessing

PREPOSITIONS = [
    '', 'of', 'with', 'at', 'from', 'into', 'during', 'including', 'until', 'against', 'among', 'throughout',
    'despite', 'towards', 'upon', 'concerning', 'to', 'in', 'for', 'on', 'by', 'about', 'like',
    'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under', 'within', 'along',
    'following', 'across', 'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out', 'around', 'down'
    'off', 'above', 'near']

VERB_TYPES = ['inf', '1sg', '2sg', '3sg', 'pl', 'part', 'p', '1sgp', '2sgp', '3sgp', 'ppl', 'ppart']

def change_type(word, tag, change_prob):
    global PREPOSITIONS, VERB_TYPES
    if tag == "IN":
        if random.random() < change_prob:
            word = random.choice(PREPOSITIONS)
    elif tag == "NN":
        if random.random() < change_prob:
            word = pluralize(word)
    elif tag == "NNS":
        if random.random() < change_prob:
            word = singularize(word)
    elif "VB" in tag:
        if random.random() < change_prob:
            verb_type = random.choice(VERB_TYPES)
            word = conjugate(word, verb_type)
    return word

def make_word2ptbs(m2_files, min_cnt):
    '''Error Simulation
    m2: string. m2 file path.
    min_cnt: int. minimum count
    '''
    word2ptbs = dict()  # ptb: pertubation
    for m2_file in m2_files:
        entries = open(m2_file, 'r').read().strip().split("\n\n")
        for entry in entries:
            skip = ("noop", "UNK", "Um")
            S = entry.splitlines()[0][2:] + " </s>"
            words = S.split()
            edits = entry.splitlines()[1:]

            skip_indices = []
            for edit in edits:
                features = edit[2:].split("|||")
                if features[1] in skip: continue
                start, end = features[0].split()
                start, end = int(start), int(end)
                word = features[2]

                if start == end:  # insertion -> deletion
                    ptb = ""
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                elif start + 1 == end and word == "":  # deletion -> substitution
                    ptb = words[start] + " " + words[start + 1]
                    word = words[start + 1]
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                    skip_indices.append(start)
                    skip_indices.append(start + 1)
                elif start + 1 == end and word != "" and len(word.split()) == 1:  # substitution
                    ptb = words[start]
                    if word in word2ptbs:
                        word2ptbs[word].append(ptb)
                    else:
                        word2ptbs[word] = [ptb]
                    skip_indices.append(start)
                else:
                    continue

            for idx, word in enumerate(words):
                if idx in skip_indices: continue
                if word in word2ptbs:
                    word2ptbs[word].append(word)
                else:
                    word2ptbs[word] = [word]

    # pruning
    _word2ptbs = dict()
    for word, ptbs in word2ptbs.items():
        ptb2cnt = Counter(ptbs)

        ptb_cnt_li = []
        for ptb, cnt in ptb2cnt.most_common(len(ptb2cnt)):
            if cnt < min_cnt: break
            ptb_cnt_li.append((ptb, cnt))

        if len(ptb_cnt_li) == 0: continue
        if len(ptb_cnt_li) == 1 and ptb_cnt_li[0][0] == word: continue

        _ptbs = []
        for ptb, cnt in ptb_cnt_li:
            _ptbs.extend([ptb] * cnt)

        _word2ptbs[word] = _ptbs

    return _word2ptbs

def apply_perturbation(words, word2ptbs, word_change_prob, type_change_prob):
    word_tags = pos_tag(words)

    sent = []
    for (_, t), w in zip(word_tags, words):
        if w in word2ptbs and random.random() > 1-word_change_prob:
            oris = word2ptbs[w]
            w = random.choice(oris)
        else:
            w = change_type(w, t, type_change_prob)
        sent.append(w)

    try:
        sent = " ".join(sent)
        sent = re.sub("[ ]+", " ", sent)
    except:
        return None

    return sent

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

def count_lines(f):
    with open(f, "r", encoding="utf-8",errors='ignore') as f:
        return sum(bl.count("\n") for bl in blocks(f))

def make_parallel(inputs):
    word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob, start, end = inputs
    logging.info("Load sentencepiece model")
    sp = spm.SentencePieceProcessor()
    sp.Load(bpe_model)

    ori_dir = os.path.join(os.path.dirname(ori), f"working/ori")
    cor_dir = os.path.join(os.path.dirname(ori), f"working/cor")
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(cor_dir, exist_ok=True)

    with open(f'{ori_dir}/{start}', 'w') as ori, open(f'{cor_dir}/{start}', 'w') as cor:
        for _ in tqdm(range(n_epochs)):
            i = 0
            for line in open(txt, 'r'):
                i += 1
                if start <= i < end:
                    words = line.strip().split()
                    perturbation = apply_perturbation(words, word2ptbs, word_change_prob, type_change_prob)
                    if perturbation is None: continue
                    ori_pieces = sp.EncodeAsPieces(perturbation.strip())
                    cor_pieces = sp.EncodeAsPieces(line.strip())
                    ori.write(" ".join(ori_pieces) + "\n")
                    cor.write(" ".join(cor_pieces) + "\n")
                if i > end:
                    break

def do(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob):
    print("# multiprocessing settings")
    n_cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(n_cpus)

    print("# prepare inputs")
    n_lines = count_lines(txt)

    start_li = list(range(0, n_lines, n_lines // n_cpus))
    start_end_li = [(start, start + n_lines // n_cpus) for start in start_li]
    inputs_li = [(word2ptbs, bpe_model, txt, ori, cor, n_epochs, word_change_prob, type_change_prob, start, end) \
                 for start, end in start_end_li]

    print("# work")
    p.map(make_parallel, inputs_li)
    p.close()
    p.join()

    print("# work done!")

    print("# concat...")
    os.system(f"cat {os.path.dirname(ori)}/working/ori/* > {ori}")
    os.system(f"cat {os.path.dirname(cor)}/working/cor/* > {cor}")
    os.system(f"rm -r {os.path.dirname(ori)}/working")
    print("All done!")
