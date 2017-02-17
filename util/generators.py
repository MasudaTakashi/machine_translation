def batch(generator, batch_size):
    batch = []
    is_tuple = False
    for l in generator:
        is_tuple = isinstance(l, tuple)
        batch.append(l)
        if len(batch) == batch_size:
            yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch
            batch = []
    if batch:
        yield tuple(list(x) for x in zip(*batch)) if is_tuple else batch

# 長さでソートしながらバッチ作成
# @params generator1    src側
# @params generator2    trg側
# @params pooling       バッチ作る際の候補数
# @params order         src側(0), trg側(1)どちらの長さでソートするか
def sorted_parallel(generator1, generator2, pooling, order=0):
    gen1 = batch(generator1, pooling)
    gen2 = batch(generator2, pooling)
    for batch1, batch2 in zip(gen1, gen2):
        #yield from sorted(zip(batch1, batch2), key=lambda x: len(x[1]))
        for x in sorted(zip(batch1, batch2), key=lambda x: len(x[order])):
            yield x

def word_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield l.split()

def letter_list(filename):
    with open(filename) as fp:
        for l in fp:
            yield list(''.join(l.split()))
