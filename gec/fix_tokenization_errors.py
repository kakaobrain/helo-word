import re

def space_puncts(sent):
    '''
    e.g., noise!He -> noise ! He
    '''
    symbols = '[:.,!?"]'
    _sent = []
    for word in sent.strip().split():
        detect = re.search(f"{symbols}[A-Za-z]", word)
        if detect is not None:
            if not word.count(".") > 1:  # e.g., `U.S.` is correct.
                word = re.sub(f"({symbols})", r" \1 ", word)
        _sent.append(word)

    return " ".join(_sent)

def space_contracts(sent):
    '''
    e.g., haven't -> have n't
    '''
    clitics = ("n't", "'ll", "'s", "'m", "'ve")
    _sent = []
    for w in sent.split():
        for clitic in clitics:
            w = re.sub(f"([A-Za-z])({clitic})", r"\1 \2", w)
        _sent.append(w)
    return " ".join(_sent)

def collapse_spaces(sent):
    sent = re.sub(" +", " ", sent)
    return sent

def fix(sent):
    sent = space_puncts(sent)
    sent = space_contracts(sent)
    sent = collapse_spaces(sent)
    return sent