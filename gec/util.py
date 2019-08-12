import os
import logging
import re
from glob import glob


def maybe_do(fp, func, inputs):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        func(*inputs)


def maybe_prompt(fp, prompt):
    if os.path.exists(fp):
        logging.info(f"skip this step as {fp} already exists")
    else:
        os.system(prompt)


def get_scores(report_fname, scorer):
    assert scorer in ["errant", "m2scorer"]

    report = open(report_fname, 'r')

    try:
        if scorer == "errant":
            line = report.read().strip().splitlines()[-2]
            tokens = line.split()
            precision, recall, fscore = tokens[-3], tokens[-2], tokens[-1]

        else:  # m2scorer
            line = report.read().strip().splitlines()
            precision = line[0].split()[-1]
            recall = line[1].split()[-1]
            fscore = line[2].split()[-1]
    except:
        logging.error(f"[Util] cannot get scores from {report_fname}")
        precision, recall, fscore = 0, 0, 0

    precision = float(precision) * 100
    recall = float(recall) * 100
    fscore = float(fscore) * 100

    # precision = int(float(precision) * 100)
    # recall = int(float(recall) * 100)
    # fscore = int(float(fscore) * 100)

    return precision, recall, fscore


def find_highest_score(output_dir, ori_path, scorer_type):
    data_basename = get_basename(ori_path, include_path=False, include_extension=False)
    files = glob(os.path.join(output_dir, f"*{data_basename}.report"))

    highest_fscore = 0
    highest_basename = ''

    for report_fname in files:
        precision, recall, fscore = get_scores(report_fname, scorer_type)
        if fscore > highest_fscore:
            highest_fscore  = fscore
            highest_basename = get_basename(report_fname, include_path=True, include_extension=False).replace(data_basename, '')[:-1]

    highest_ckpt = f"{highest_basename}.pt".replace("outputs", "ckpt")
    if highest_basename == '':
        print(f'cannot find highest basename from {output_dir}')
        exit()

    return highest_fscore, highest_ckpt


def get_sorted_ckpts(ckpt_dir, epoch_start=1, epoch_interval=1):
    files = glob(os.path.join(ckpt_dir, "*pt"))
    epoch_files = []
    for f in files:
        epoch = f.split("/")[-1].split(".")[0].replace("checkpoint", "").split("_")[-1]
        try:
            epoch = int(epoch)
        except ValueError:
            continue
        epoch_files.append((epoch, f))
    epoch_files = sorted(epoch_files, key=lambda x: x[0])
    files = [f for epoch, f in epoch_files if epoch >= epoch_start]
    files = files[::epoch_interval]  # skip some

    return files


def get_basename(path, include_path=True, include_extension=True):
    if path is None:
        return None

    if os.path.isdir(path) and path[-1] == '/':
        path = path[:-1]
    base = os.path.basename(path)

    if os.path.isfile(path) and not include_extension:
        base = '.'.join(base.split('.')[:-1])

    if include_path:
        dirpath = os.path.dirname(path)
        return f'{dirpath}/{base}'
    else:
        return base


def get_cor_path(system_out, remove_unk_edits, remove_error_type_lst,
                 apply_rerank, preserve_spell, max_edits):
    cor_path = get_basename(system_out, include_extension=False)
    if remove_unk_edits:
        cor_path += '-unk'
    if len(remove_error_type_lst) > 0:
        cor_path += '-'.join(remove_error_type_lst)
    if apply_rerank:
        cor_path += '-rerank'
    if preserve_spell:
        cor_path += '-spell'
    if max_edits is not None:
        cor_path += f'-max-{max_edits}'
    return f"{cor_path}.cor"


def change_ckpt_dir(ckpt_fpath, new_ckpt_dir):
    fpath_basename = os.path.basename(ckpt_fpath)
    dirname = os.path.dirname(ckpt_fpath).split('/')[:-1]
    new_ckpt_basename = os.path.basename(new_ckpt_dir)
    dirname = '/'.join(dirname + [new_ckpt_basename])
    return f"{dirname}/{fpath_basename}"
