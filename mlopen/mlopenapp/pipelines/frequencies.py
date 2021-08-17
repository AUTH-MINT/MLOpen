import numpy as np


def build_freqs(statements, sentiment_list):
    freqs = {}
    for statement, sentiment in zip(statements, sentiment_list):
        for word in statement:
            pair = (word, sentiment)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
            return freqs


def statement_to_freq(statement, freqs):
    x = np.zeros((2,))
    for word in statement:
        if (word, 1) in freqs:
            x[0] += freqs[(word, 1)]
        if (word, 0) in freqs:
            x[1] += freqs[(word, 0)]
    return x


def get_posneg(x, freqs):
    posneg = [statement_to_freq(statement, freqs) for statement in x]
    return posneg
