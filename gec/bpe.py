import sentencepiece as spm

def train(inp, model_prefix, vocab_size, character_coverage, model_type):
    train = f'--input={inp} --model_prefix={model_prefix} \
            --vocab_size={vocab_size} --character_coverage={character_coverage} \
            --model_type={model_type}'
    spm.SentencePieceTrainer.Train(train)

    adjust_vocab(model_prefix + ".vocab")

def adjust_vocab(vocab_fpath):
    '''Adjust bpe vocab file such that it fits the fairseq dict format.'''
    adjusted = []
    for line in open(vocab_fpath, 'r').read().splitlines()[3:]:
        tok = line.split()
        if len(tok) > 1:
            adjusted.append(" ".join(tok))

    with open(vocab_fpath, 'w') as fout:
        fout.write("\n".join(adjusted))

def bpe_tokenize(model, fin, fout):
    '''
    model: str. model fp
    fin: input fp
    fout: output fp
    '''
    sp = spm.SentencePieceProcessor()
    sp.Load(model)

    with open(fout, 'w') as fout:
        for line in open(fin, 'r'):
            tokens = sp.EncodeAsPieces(line.strip())
            fout.write(" ".join(tokens) + "\n")

