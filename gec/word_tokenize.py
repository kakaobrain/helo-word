import logging
import os
import spacy # 1.9
import re
from glob import glob
from nltk.tokenize import sent_tokenize
logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en")


def gutenberg(fpaths, fout, max_tokens):
    global nlp

    with open(fout, 'w') as fout:
        for fpath in fpaths:
            logging.info(f"Working on {fpath}")
            paras = open(fpath, 'r', errors='ignore').read().split("\n\n")
            for para in paras:
                para = para.replace("\n", " ").replace("_", "").replace("  ", " ")
                sents = sent_tokenize(para)
                for sent in sents:
                    if len(sent) < 2: continue
                    doc = nlp.tokenizer(sent)
                    tokens = [token.text for token in doc]
                    if len(tokens) <= max_tokens:
                        fout.write(" ".join(tokens) + "\n")


def tatoeba(fpath, fout, max_tokens):
    global nlp

    with open(fout, 'w') as fout:
        logging.info(f"Working on {fpath}")
        for line in open(fpath, 'r'):
            sent = line.strip()
            _, lang, sent = sent.split("\t")
            if lang != "eng": continue

            doc = nlp.tokenizer(sent)
            tokens = [token.text for token in doc]
            if len(tokens) <= max_tokens:
                fout.write(" ".join(tokens) + "\n")


def wiki103(fpath, fout, max_tokens):
    global nlp
    proc = re.compile('^[A-Za-z0-9,.:/;\-@"!?()<>\[\]\']*$')
    with open(fout, 'w') as fout:
        logging.info(f"Working on {fpath}")
        for para in open(fpath, 'r').read().splitlines():
            p = para.strip()
            if p.startswith('=') or p == '': continue
            sents = sent_tokenize(p)
            for sent in sents:
                if len(sent) < 2: continue
                doc = nlp.tokenizer(sent)
                tokens = [token.text if proc.match(token.text) else '<unk>' for token in doc]
                if len(tokens) <= max_tokens:
                    line = " ".join(tokens).replace('< unk >', '<unk>')
                    if '(' in line and ')' not in line:
                        fout.write(line + " ")
                    else:
                        fout.write(line + "\n")
