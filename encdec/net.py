import copy
import pickle
import numpy as np
from chainer import optimizers
from chainer import serializers
from chainer import functions
from util.wrapper  import wrapper
from util.const import *
from model.encdec import EncoderDecoderModel

class BasicEncoderDecoderModel(EncoderDecoderModel):
    pass
