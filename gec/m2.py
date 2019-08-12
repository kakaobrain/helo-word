import random
import os
from itertools import combinations
from tqdm import tqdm

from .filepath import Path


def get_all_coder_ids(edits):
    coder_ids = set()
    for edit in edits:
        edit = edit.split("|||")
        coder_id = int(edit[-1])
        coder_ids.add(coder_id)
    coder_ids = sorted(list(coder_ids))
    return coder_ids


def m2_to_parallel(m2_files, ori, cor, drop_unchanged_samples, all):

    ori_fout = None
    if ori is not None:
        ori_fout = open(ori, 'w')
    cor_fout = open(cor, 'w')

    # Do not apply edits with these error types
    skip = {"noop", "UNK", "Um"}
    for m2_file in m2_files:
        entries = open(m2_file).read().strip().split("\n\n")
        for entry in entries:
            lines = entry.split("\n")
            ori_sent = lines[0][2:]  # Ignore "S "
            cor_tokens = lines[0].split()[1:]  # Ignore "S "
            edits = lines[1:]
            offset = 0

            coders = get_all_coder_ids(edits) if all == True else [0]
            for coder in coders:
                for edit in edits:
                    edit = edit.split("|||")
                    if edit[1] in skip: continue  # Ignore certain edits
                    coder_id = int(edit[-1])
                    if coder_id != coder: continue  # Ignore other coders
                    span = edit[0].split()[1:]  # Ignore "A "
                    start = int(span[0])
                    end = int(span[1])
                    cor = edit[2].split()
                    cor_tokens[start + offset:end + offset] = cor
                    offset = offset - (end - start) + len(cor)

                cor_sent = " ".join(cor_tokens)
                if drop_unchanged_samples and ori_sent == cor_sent:
                    continue

                if ori is not None:
                    ori_fout.write(ori_sent + "\n")
                cor_fout.write(cor_sent + "\n")


def m2_to_cor(m2, cor):
    m2_to_parallel(m2_files=[m2], ori=None, cor=cor, drop_unchanged_samples=False, all=True)


def parallel_to_m2(ori, cor, m2):
    p2m = f"python {Path.parallel_to_m2} -orig {ori} -cor {cor} -out {m2}"
    os.system(p2m)


def sys_to_cor(system_out, cor_path):

    def detokenize(sent):
        '''restores a bpe-segmented sentence
            '''
        return sent.replace(" ", "").replace("▁", " ").strip()

    lines = open(system_out, 'r').read().strip().splitlines()
    hypo_lst = []
    for line in lines:
        if line.startswith("H"):
            cols = line.split("\t")
            num = int(cols[0].split("-")[-1])
            hypo = (num, cols[-1])
            hypo_lst.append(hypo)

    def _sorted(x):
        x = sorted(x, key=lambda x: x[0])
        return [elem[1] for elem in x]

    hypo_lst = _sorted(hypo_lst)
    with open(cor_path, 'w') as fout:
        fout.write("\n".join(detokenize(sent) for sent in hypo_lst))
        fout.write("\n")


def line_to_edit(m2_line):
    if not m2_line.startswith("A"):
        return None
    features = m2_line.split("|||")
    span = features[0].split()
    start, end = int(span[1]), int(span[2])
    error_type = features[1]
    replace_token = features[2]
    return start, end, error_type, replace_token


def remove_m2(m2_entries, filter_type_lst, filter_token):
    noop = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"

    output_entries = []
    for m2_entry in m2_entries:
        lines = m2_entry.splitlines()
        sent = lines[0]
        edits = lines[1:]
        preserve_m2 = [sent]

        for line in edits:
            start, end, error_type, replace_token = line_to_edit(line)
            if filter_type_lst is not None:
                if error_type not in filter_type_lst:
                    preserve_m2.append(line)
            if filter_token is not None:
                if filter_token not in replace_token:
                    preserve_m2.append(line)

        if len(preserve_m2) == 1:
            preserve_m2.append(noop)

        output_entries.append("\n".join(preserve_m2))
    return output_entries


def sort_m2_lines(m2_lines):
    m2_dict = dict()
    for line in m2_lines:
        s, _, _, _ = line_to_edit(line)
        m2_dict[s] = line
    return [i[1] for i in sorted(m2_dict.items())]


def m2edits_to_cor(ori, m2_lines):
    _m2_lines = sort_m2_lines(m2_lines)

    skip = {"noop", "UNK", "Um"}
    cor_sent = ori.split()

    offset = 0
    for edit in _m2_lines:
        edit = edit.split("|||")
        if edit[1] in skip: continue  # Ignore certain edits
        span = edit[0].split()[1:]  # Ignore "A "
        start = int(span[0])
        end = int(span[1])
        cor = edit[2].split()
        cor_sent[start + offset:end + offset] = cor
        offset = offset - (end - start) + len(cor)
    return " ".join(cor_sent) + "\n"


def get_edit_combinations(edits):
    edit_combinations = []
    for i in range(len(edits) + 1):
        edit_combinations.extend(combinations(edits, i))

    return edit_combinations


def apply_lm_rerank(m2_entries, preserve_spell, max_edits, lm_scorer):
    noop = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"

    output_entries = []
    for m2_entry in tqdm(m2_entries):
        lines = m2_entry.splitlines()
        sent = lines[0][2:]
        edits = lines[1:]
        preserve_m2 = []
        rerank_m2 = []

        for line in edits:
            start, end, error_type, replace_token = line_to_edit(line)
            if error_type == 'noop':
                preserve_m2.append(line)
            elif preserve_spell and "SPELL" in error_type:
                preserve_m2.append(line)
            else:
                rerank_m2.append(line)

        if max_edits is None or 0 < len(rerank_m2) < max_edits:
            edit_comb = get_edit_combinations(rerank_m2)

            cor_sents = []
            for e in edit_comb:
                cor = m2edits_to_cor(sent, preserve_m2 + list(e))
                cor_sents.append(cor)

            score_dict = lm_scorer.score(cor_sents)

            min_idx = sorted(score_dict, key=score_dict.get, reverse=False)[0]
            sorted_m2_lines = sort_m2_lines(preserve_m2 + list(edit_comb[min_idx]))
            if len(sorted_m2_lines) == 0:
                output_entries.append("\n".join([lines[0], noop]))
            else:
                output_entries.append("\n".join([lines[0]] + sorted_m2_lines))

        else:
            output_entries.append(m2_entry)
    return output_entries


def get_m2_entries(m2_file):
    m2_entries = open(m2_file).read().strip().split('\n\n')
    return m2_entries


def write_m2_entries(m2_entries, m2_file):
    with open(m2_file, 'w') as fout:
        fout.write("\n\n".join(m2_entries))

def split_m2(fpaths, finetune_m2, valid_m2, finetune_split_ratio):
    """
    split m2 files into finetune and valid files
    """
    with open(finetune_m2, 'w') as finetune_out, open(valid_m2, 'w') as valid_out:
        for fpath in fpaths:
            if 'ABCN' in fpath: continue
            lines = open(fpath, 'r').read().split('\n\n')
            for line in lines:
                if line.startswith('S'):
                    if random.random() < finetune_split_ratio:
                        finetune_out.write(line+'\n\n')
                    else:
                        valid_out.write(line+'\n\n')
