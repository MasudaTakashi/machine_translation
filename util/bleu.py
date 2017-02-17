# coding: utf-8

import sys
import codecs
import math
from collections import defaultdict

# calculate BLEU score between two corpus
def BLEU(ws_hyp, ws_ref, **kwargs):
    # check args
    max_n = 4
    if 'maxn' in kwargs and isinstance(kwargs['maxn'], int) and kwargs['maxn'] > 0:
        max_n = kwargs['maxn']
    dump = False
    if 'dump' in kwargs and kwargs['dump'] is True:
        dump = True

    # calc N-gram precision
    np = 1
    for n in range(1, max_n+1):
        numer = 0
        denom = 0
        for hyp, ref in zip(ws_hyp, ws_ref):
            if len(hyp) < n:
                continue
            possible_ngram = defaultdict(lambda: 0)
            for i in range(len(ref)-(n-1)):
                possible_ngram[tuple(ref[i:i+n])] += 1
            for i in range(len(hyp)-(n-1)):
                key = tuple(hyp[i:i+n])
                if key in possible_ngram and possible_ngram[key] > 0:
                    possible_ngram[key] -= 1
                    numer += 1
            denom += len(hyp)-(n-1)
        if dump:
            print('P(n=%d) = %f (%d/%d)' % (n, numer/float(denom), numer, denom))
        np *= ((numer/float(denom))**(1/float(max_n)))

    # calc brevity penalty
    sumlen_hyp = sum(len(x) for x in ws_hyp)
    sumlen_ref = sum(len(x) for x in ws_ref)
    bp = min(1.0, math.exp(1.0-sumlen_ref/float(sumlen_hyp)))
    if dump:
        print('BP = %f (HYP:%d, REF:%d)' % (bp, sumlen_hyp, sumlen_ref))

    # calc final score
    bleu = bp * np
    if dump:
        print('BLEU = %f' % bleu)
    return bleu

# calculate BLEU+1 score between two sentences
def BLEUp1(hyp, ref, **kwargs):
    # check args
    max_n = 4
    if 'maxn' in kwargs and isinstance(kwargs['maxn'], int) and kwargs['maxn'] > 0:
        max_n = kwargs['maxn']

    # calc N-gram precision
    np = 0
    for n in range(1, max_n+1):
        numer = 0 if n == 1 else 1
        possible_ngram = defaultdict(lambda: 0)
        for i in range(len(ref)-(n-1)):
            possible_ngram[tuple(ref[i:i+n])] += 1
        for i in range(len(hyp)-(n-1)):
            key = tuple(hyp[i:i+n])
            if key in possible_ngram and possible_ngram[key] > 0:
                possible_ngram[key] -= 1
                numer += 1
        if numer == 0: # no shared unigram
            return 0
        denom = (0 if n == 1 else 1) + max(0, len(hyp)-(n-1))

        np += math.log(numer) - math.log(denom)

    # calc brevity penalty
    bp = min(1.0, math.exp(1.0-len(ref)/float(len(hyp))))

    # calc final score
    bleu = bp * math.exp(np/max_n)
    return bleu

def main():
    if len(sys.argv) != 3:
        print('USAGE: python bleu.py <file:HYP> <file:REF>')
        return

    ws_hyp = []
    with codecs.open(sys.argv[1], 'r', 'utf-8') as fp:
        for l in fp:
            ls = l.strip().split(' ')
            ws_hyp.append(ls)

    ws_ref = []
    with codecs.open(sys.argv[2], 'r', 'utf-8') as fp:
        for l in fp:
            ls = l.strip().split(' ')
            ws_ref.append(ls)

    BLEU(ws_hyp, ws_ref, dump=True)

def calc_bleu(hyp_file, ref_file):
    ws_hyp = []
    with codecs.open(hyp_file, 'r', 'utf-8') as fp:
        for l in fp:
            ls = l.strip().split(' ')
            ws_hyp.append(ls)

    ws_ref = []
    with codecs.open(ref_file, 'r', 'utf-8') as fp:
        for l in fp:
            ls = l.strip().split(' ')
            ws_ref.append(ls)

    return BLEU(ws_hyp, ws_ref, dump=True)

if __name__ == '__main__':
    main()