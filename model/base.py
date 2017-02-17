import math
import copy
import pickle
import numpy as np
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import functions
from util.wrapper  import wrapper
from util.functions import trace
from util.const import *

class BaseModel(object):
    def __init__(self, gpu_id, output):
        self.wrapper = wrapper(gpu_id)
        self.output = output
        self.with_label = False
        self.xp = cuda.cupy if gpu_id >= 0 else np
        self.is_reverse = self.is_reverse_input()
        self.custom_init()

    def custom_init(self):
        pass

    def init_model(self):
        self.make_model()
        # softmaxを求めるための行列を定義
        vocab = self.trg_vocab
        raw_matrix = [[1 if vocab.itow(column) == raw else 0 for column in range(len(vocab))] for raw in range(vocab.size())]
        self.matrix = self.xp.array(raw_matrix).T

    def make_model(self):
        pass

    def init_optimizer(self, lr):
        #self.opt = optimizers.SGD(lr=lr)
        self.opt = optimizers.AdaDelta()
        self.opt.setup(self.model)
        self.opt.clip_grads(5)

    def save(self, filename):
        self.wrapper.begin_model_access(self.model)
        serializers.save_hdf5(filename, self.model)
        #serializers.save_hdf5(filename[:-6]+".state", self.opt)
        self.wrapper.end_model_access(self.model)

    @classmethod
    def load(cls, args, src_vocab, trg_vocab):
        self = cls(args.gpu_id, args.output)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.n_embed = args.embed
        self.n_hidden = args.hidden
        self.init_model()
        serializers.load_hdf5(args.model, self.model)
        #serializers.load_hdf5(args.model[:-6]+".state", self.opt)
        return self

    @classmethod
    def new(cls, args, src_vocab, trg_vocab):
        self = cls(args.gpu_id, args.output)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.n_embed = args.embed
        self.n_hidden = args.hidden
        self.init_model()
        return self

    def forward(self, is_training, src_batch, trg_batch=None, generation_limit=None, beamwidth=None, is_greedy=False, use_all_vocab=False):
        hyp_batch = loss = accuracy = log_perp = None
        return hyp_batch, loss, accuracy, log_perp

    def train(self, src_batch, trg_batch):
        self.opt.zero_grads()
        hyp_batch, accum_loss, accum_accuracy, sum_log_likelihood = self.forward(True, src_batch, trg_batch=trg_batch)
        accum_loss.backward()
        self.opt.update()
        return hyp_batch, sum_log_likelihood, self.wrapper.get_data(accum_accuracy).reshape(())

    def evaluate(self, src_batch, trg_batch):
        hyp_batch, accum_loss, accum_accuracy, sum_log_likelihood = self.forward(True, src_batch, trg_batch=trg_batch)
        return hyp_batch, sum_log_likelihood, self.wrapper.get_data(accum_accuracy).reshape(())

    def predict(self, src_batch, generation_limit, beamwidth, use_all_vocab=False):
        return self.forward(False, src_batch, generation_limit=generation_limit, beamwidth=beamwidth, use_all_vocab=False)

    def predict_greedy(self, src_batch, generation_limit, beamwidth):
        return self.forward(False, src_batch, generation_limit=generation_limit, beamwidth=beamwidth, is_greedy=True)

    def softmax(self, x):
        vocab = self.trg_vocab
        softmax = functions.softmax(x).data
        if self.with_label:
            # 確率を単語ごとに寄せ集め
            softmax_by_word = softmax.dot(self.matrix)
            return softmax_by_word
        return softmax

    def is_reverse_input(self):
        return False

    def make_input_weighting_vec(self, input_word_index, index2vec_unit, softmax_all_):
        m = self.model
        trg_stoi = self.trg_vocab
        trg_wtois = self.trg_vocab.wtois
        softmax_all = softmax_all_.data[0]

        vec = self.wrapper.zeros((1, 4*self.n_hidden))
        sum_prob = float(0)
        for str_i in trg_wtois(input_word_index):
            str_i_var = self.wrapper.make_var([str_i], dtype=np.int32)
            new_vec = index2vec_unit(str_i_var)
            vec += new_vec * softmax_all[str_i]
            sum_prob += softmax_all[str_i]
        vec /= float(sum_prob)

        return vec
