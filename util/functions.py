from argparse import ArgumentParser
import sys
import math
import datetime
from .const import *

def trace(*args):
    print(datetime.datetime.now(), '...', *args, file=sys.stderr)
    sys.stderr.flush()

def fill_batch(batch, token=EOS.s, is_reverse=False):
    max_len = max(len(x) for x in batch)
    if is_reverse:
        return [[word for word in reversed(x)] + [token] * (max_len - len(x) + 1) for x in batch]
    else:
        return [x + [token] * (max_len - len(x) + 1) for x in batch]

def fill_batch2(batch, start_token=SOS.s, end_token=EOS.s, is_reverse=False):
    max_len = max(len(x) for x in batch)
    return [[start_token] + x + [end_token] * (max_len - len(x) + 1) for x in batch]

def vtos(v, fmt='%.8e'):
    return ' '.join(fmt % x for x in v)

def stov(s, tp=float):
    return [tp(x) for x in s.split()]

def removeLabel(sentence_l):
    return [word_with_label.split("/")[0]  if word_with_label[0:5] != "<unk>" else word_with_label for word_with_label in sentence_l]

def removeAllLabel(sentence_l):
    return [word_with_label.split("/")[0] for word_with_label in sentence_l]

def log_likelihood(softmax, ans_indexes):
    result = 0.0
    for i in range(len(ans_indexes)):
        log = math.log(softmax[i][ans_indexes[i]])
        if math.isnan(log):
            result += -700.0
        else:
            result += log
    return result

def parse_args():
    def_vocab = 0   # def_vocab = 32768
    def_embed = 256
    def_hidden = 256
    def_lr = 0.5
    def_epoch = 1000
    def_minibatch = 128
    def_beamwidth = 1
    def_generation_limit = 100

    p = ArgumentParser(description='neural machine trainslation')

    p.add_argument('mode', help='\'train\' or \'test\'')
#    p.add_argument('src_vocab', type=str, help='path of source vocab file to load')
#    p.add_argument('trg_vocab', type=str, help='path of target vocab file to load')
    p.add_argument('-o', default="", type=str, help='name of result folder', dest='folder_name')
    p.add_argument("--output", help="output sentence when train and evaluate", action="store_true")
    p.add_argument("--score", help="not output any files when dev, only calc score", action="store_true")
    p.add_argument("--all", help="use all vocab (not use word vocab, use str vocab) when test", action="store_true")
    p.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)', dest='gpu_id')
    p.add_argument('--lr', '-lr', default=def_lr, type=float, help="optimizer's initial learning rate (SGD) (default: %f)" % def_lr, dest='lr')
    p.add_argument('--model', default="", type=str, help='[in] model filename')
    p.add_argument('--embed', default=def_embed, metavar='INT', type=int, help='embedding layer size (default: %d)' % def_embed)
    p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int, help='hidden layer size (default: %d)' % def_hidden)
    p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int, help='maximum number of training epoch (default: %d)' % def_epoch)
    p.add_argument('--minibatch', "--batch", default=def_minibatch, metavar='INT', type=int, help='minibatch size (default: %d)' % def_minibatch)
    p.add_argument('--beamwidth', "--beam", default=def_beamwidth, metavar='INT', type=int, help='beam width (default: %d)' % def_beamwidth)
    p.add_argument('--generation-limit', default=def_generation_limit, metavar='INT', type=int, help='maximum number of sentence words to be generated (default: %d)' % def_generation_limit)

    args = p.parse_args()

    # check args
    try:
        if args.mode not in ['train', 'dev', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
        if args.mode in ["dev, ""test"] and args.model == "": raise ValueError('you must set model file')
        if args.embed < 1: raise ValueError('you must set --embed >= 1')
        if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
        if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
        if args.minibatch < 1: raise ValueError('you must set --minibatch >= 1')
        if args.beamwidth < 1: raise ValueError('you must set --beamwidth >= 1')
        if args.generation_limit < 1: raise ValueError('you must set --generation-limit >= 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    if args.mode == "test": args.minibatch = 1
    if args.mode == "dev":
        if args.epoch == def_epoch: args.epoch = 1

    return args


def typetoi(type):
    return types.index(type)