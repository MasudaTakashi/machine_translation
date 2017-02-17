import sys
import math
import numpy as np

from net import BasicEncoderDecoderModel
from util.functions import trace, fill_batch, parse_args
from util.vocabulary import Vocabulary
from util import generators as gens
from util.controller import Controller
from util.wrapper import wrapper
from util.const import *



if __name__ == '__main__':
    args = parse_args()

    trace('initializing ...')
    wrapper = wrapper(args.gpu_id)
    wrapper.init()

    trace('loading vocab ...')
#    src_vocab = Vocabulary.load(args.src_vocab)
#    trg_vocab = Vocabulary.load(args.trg_vocab)
    src_vocab = Vocabulary.load(VOCAB_SRC)
    trg_vocab = Vocabulary.load(VOCAB_TRG)

    controller = Controller(args.folder_name)

    if args.mode == 'train': controller.train_model(BasicEncoderDecoderModel, src_vocab, trg_vocab, args)
    elif args.mode == 'dev': controller.dev_model(BasicEncoderDecoderModel, src_vocab, trg_vocab, args)
    elif args.mode == 'test': controller.test_model(BasicEncoderDecoderModel, src_vocab, trg_vocab, args)
