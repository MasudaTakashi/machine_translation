import math
import copy
import pickle
import numpy as np
from chainer import optimizers
from chainer import serializers
from chainer import functions
from util.functions  import log_likelihood
from util.wrapper  import wrapper
from util.const import *
from .base import BaseModel

class AttentionalTranslationModel(BaseModel):
    def make_model(self):
        initialW = np.random.uniform
        self.model = self.wrapper.make_model(
            # input embedding
            w_xi = functions.EmbedID(len(self.src_vocab), self.n_embed),
            # forward encoder
            w_ia = functions.Linear(self.n_embed,   4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_embed))),
            w_aa = functions.Linear(self.n_hidden,  4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_hidden))),
            # backward encoder
            w_ib = functions.Linear(self.n_embed,   4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_embed))),
            w_bb = functions.Linear(self.n_hidden,  4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_hidden))),
            # attentional weight estimator
            w_aw = functions.Linear(self.n_hidden,  self.n_hidden,      initialW=initialW(-0.1, 0.1, (  self.n_hidden,  self.n_hidden))),
            w_bw = functions.Linear(self.n_hidden,  self.n_hidden,      initialW=initialW(-0.1, 0.1, (  self.n_hidden,  self.n_hidden))),
            w_pw = functions.Linear(self.n_hidden,  self.n_hidden,      initialW=initialW(-0.1, 0.1, (  self.n_hidden,  self.n_hidden))),
            w_we = functions.Linear(self.n_hidden,  1,                  initialW=initialW(-0.1, 0.1, (              1,  self.n_hidden))),
            # decoder
            w_ap = functions.Linear(self.n_hidden, self.n_hidden,       initialW=initialW(-0.1, 0.1, (  self.n_hidden,  self.n_hidden))),
            w_bp = functions.Linear(self.n_hidden, self.n_hidden,       initialW=initialW(-0.1, 0.1, (  self.n_hidden,  self.n_hidden))),
            w_yp = functions.EmbedID(len(self.trg_vocab), 4 * self.n_hidden),
            w_pp = functions.Linear(self.n_hidden,  4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_hidden))),
            w_cp = functions.Linear(self.n_hidden,  4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_hidden))),
            w_dp = functions.Linear(self.n_hidden,  4 * self.n_hidden,  initialW=initialW(-0.1, 0.1, (4*self.n_hidden,  self.n_hidden))),
            w_py = functions.Linear(self.n_hidden,  len(self.trg_vocab),initialW=initialW(-0.1, 0.1, (len(self.trg_vocab), self.n_hidden))),
        )

    def forward(self, is_training, src_batch, trg_batch=None, generation_limit=None, beamwidth=None, is_greedy=False, use_all_vocab=False):
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = len(src_batch)
        hidden_size = self.n_hidden
        src_len = len(src_batch[0])
        trg_len = len(trg_batch[0]) - 1 if is_training else generation_limit
        src_stoi = self.src_vocab.stoi
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        trg_itow = self.trg_vocab.itow
        trg_word = self.trg_vocab.word
        trg_wstois = self.trg_vocab.wstois

        hidden_zeros = self.wrapper.zeros((batch_size, hidden_size))
        sum_e_zeros = self.wrapper.zeros((batch_size, 1))

        # make embedding
        list_x = [None] * src_len
        for l in range(src_len):
            s_x = self.wrapper.make_var([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
            list_x[l] = s_x

        # forward encoding
        c = hidden_zeros
        s_a = hidden_zeros
        list_a = [s_a] * src_len
        for l in range(src_len):
            s_x = list_x[l]
            s_i = tanh(m.w_xi(s_x))
            c, s_a = lstm(c, m.w_ia(s_i) + m.w_aa(s_a))
            list_a[l] = s_a

        # backward encoding
        c = hidden_zeros
        s_b = hidden_zeros
        list_b = [s_b] * src_len
        for l in reversed(range(src_len)):
            s_x = list_x[l]
            s_i = tanh(m.w_xi(s_x))
            c, s_b = lstm(c, m.w_ib(s_i) + m.w_bb(s_b))
            list_b[l] = s_b

        # decoding
        c = hidden_zeros
        s_p = tanh(m.w_ap(list_a[-1]) + m.w_bp(list_b[0]))
        s_y = self.wrapper.make_var([trg_stoi(SOS.s) for k in range(batch_size)], dtype=np.int32)

        if is_training:

            hyp_batch = [[] for _ in range(batch_size)]
            accum_loss = self.wrapper.zeros(())
            accum_accuracy = self.wrapper.zeros(())
            sum_log_likelihood = 0.0

            for l in range(trg_len):
                # calculate attention weights
                list_e = [None] * src_len
                sum_e = sum_e_zeros
                for n in range(src_len):
                    s_w = tanh(m.w_aw(list_a[n]) + m.w_bw(list_b[n]) + m.w_pw(s_p))
                    r_e = functions.exp(m.w_we(s_w))
                    list_e[n] = r_e
                    sum_e += r_e

                # make attention vector
                s_c = hidden_zeros
                s_d = hidden_zeros
                for n in range(src_len):
                    s_e = list_e[n] / sum_e
                    s_c += functions.reshape(functions.batch_matmul(list_a[n], s_e), (batch_size, hidden_size))
                    s_d += functions.reshape(functions.batch_matmul(list_b[n], s_e), (batch_size, hidden_size))

                # generate next word
                c, s_p = lstm(c, m.w_yp(s_y) + m.w_pp(s_p) + m.w_cp(s_c) + m.w_dp(s_d))
                r_y = m.w_py(s_p)
                softmax = self.softmax(r_y) # @todo: if self.outputの後でもいい疑惑

                if self.output:
                    output = softmax.argmax(1)
                    for k in range(batch_size):
                        hyp_batch[k].append(trg_itos(output[k]))

                s_t = self.wrapper.make_var([trg_stoi(trg_batch[k][l + 1]) for k in range(batch_size)], dtype=np.int32)
                ans_word_indexes = [trg_itow(trg_stoi(trg_batch[k][l + 1])) for k in range(batch_size)]
                accum_loss += functions.softmax_cross_entropy(r_y, s_t)
                accum_accuracy += functions.accuracy(r_y, s_t)
                sum_log_likelihood += log_likelihood(softmax, ans_word_indexes)
                s_y = s_t

            return hyp_batch, accum_loss, accum_accuracy, sum_log_likelihood

        else:
            if is_greedy:
                hyp_batch = [[] for _ in range(batch_size)]

                for l in range(trg_len):
                    # calculate attention weights
                    list_e = [None] * src_len
                    sum_e = sum_e_zeros
                    for n in range(src_len):
                        s_w = tanh(m.w_aw(list_a[n]) + m.w_bw(list_b[n]) + m.w_pw(s_p))
                        r_e = functions.exp(m.w_we(s_w))
                        list_e[n] = r_e
                        sum_e += r_e

                    # make attention vector
                    s_c = hidden_zeros
                    s_d = hidden_zeros
                    for n in range(src_len):
                        s_e = list_e[n] / sum_e
                        s_c += functions.reshape(functions.batch_matmul(list_a[n], s_e), (batch_size, hidden_size))
                        s_d += functions.reshape(functions.batch_matmul(list_b[n], s_e), (batch_size, hidden_size))

                    # generate next word
                    c, s_p = lstm(c, m.w_yp(s_y) + m.w_pp(s_p) + m.w_cp(s_c) + m.w_dp(s_d))
                    r_y = m.w_py(s_p)
                    output = self.softmax(r_y).argmax(1)
                    for k in range(batch_size):
                        hyp_batch[k].append(trg_word(int(output[k])))

                    if all(EOS.s in set(hyp_batch[k]) for k in range(batch_size)): break
                    s_y = self.wrapper.make_var(trg_wstois(output, functions.softmax(r_y)), dtype=np.int32)
                return hyp_batch

            else:
                state = {
                    "c":c,
                    "s_p":s_p,
                    "s_y":s_y,
                    "list_a":list_a,
                    "list_b":list_b,
                }
                if use_all_vocab:
                    return self.beam_search_for_batch_1_use_all_vocab(state, generation_limit, beamwidth)
                else:
                    return self.beam_search_for_batch_1(state, generation_limit, beamwidth)


    def beam_search_for_batch_1(self, state, generation_limit, beamwidth):
        assert len(state["list_a"]) == len(state["list_b"])
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = 1
        src_len = len(state["list_a"])
        hidden_size = self.n_hidden
        src_stoi = self.src_vocab.stoi
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        trg_wtoi = self.trg_vocab.wtoi
        trg_wtos = self.trg_vocab.wtos
        hidden_zeros = self.wrapper.zeros((batch_size, hidden_size))
        sum_e_zeros = self.wrapper.zeros((batch_size, 1))

        beam = []
        init_beam_item = {"c":state["c"], "s_p":state["s_p"], "input_vec":m.w_yp(state["s_y"]), "sentence":[], "score":1.0}
        beam.append(init_beam_item)
        for _ in range(generation_limit):
            # 今のビームから次のビーム候補作成
            new_beam_candidate = [None] * (len(beam) * beamwidth)
            new_beam_candidate_index = 0
            for beam_item in beam:
                # calculate attention weights
                list_e = [None] * src_len
                sum_e = sum_e_zeros
                for n in range(src_len):
                    s_w = tanh(m.w_aw(state["list_a"][n]) + m.w_bw(state["list_b"][n]) + m.w_pw(beam_item["s_p"]))
                    r_e = functions.exp(m.w_we(s_w))
                    list_e[n] = r_e
                    sum_e += r_e

                # make attention vector
                s_c = hidden_zeros
                s_d = hidden_zeros
                for n in range(src_len):
                    s_e = list_e[n] / sum_e
                    s_c += functions.reshape(functions.batch_matmul(state["list_a"][n], s_e), (batch_size, hidden_size))
                    s_d += functions.reshape(functions.batch_matmul(state["list_b"][n], s_e), (batch_size, hidden_size))

                # generate softmax
                new_c, new_s_p = lstm(beam_item["c"], beam_item["input_vec"] + m.w_pp(beam_item["s_p"]) + m.w_cp(s_c) + m.w_dp(s_d))
                r_y = m.w_py(new_s_p)
                softmax = self.softmax(r_y)
                softmax_tmp = copy.deepcopy(softmax)
                softmax_all = functions.softmax(r_y)

                # beamsizeだけ候補を追加
                for _ in range(beamwidth):
                    # 予測単語
                    best_word_index = self.xp.argmax(softmax_tmp)
                    new_sentence = beam_item["sentence"][:]
                    word_with_label = trg_wtos(best_word_index, softmax_all)
                    new_sentence.append(word_with_label)

                    # スコアは「単語」(「品詞情報付き文字列」ではない) の出現確率の積
                    new_score = beam_item["score"] * softmax[0][best_word_index]

                    #new_s_y = self.wrapper.make_var([trg_stoi(word_with_label)], dtype=np.int32)
                    new_input_vec = self.make_input_weighting_vec(best_word_index, m.w_yp, softmax_all)

                    new_beam_item = {"c":new_c, "s_p":new_s_p, "input_vec":new_input_vec, "sentence":new_sentence, "score":new_score}
                    new_beam_candidate[new_beam_candidate_index] = new_beam_item
                    new_beam_candidate_index += 1

                    # 今回ビームに追加したものをソフトマックスから除外
                    softmax_tmp[0][best_word_index] = -1.0

            # 次のビームを作成
            new_beam = [None]  * beamwidth
            new_beam_candidate_tmp = copy.deepcopy(new_beam_candidate)
            for i in range(beamwidth):
                beam_candidate_score_list = self.xp.array([new_beam_candidate_item["score"] for new_beam_candidate_item in new_beam_candidate_tmp])
                best_index = self.xp.argmax(beam_candidate_score_list)
                new_beam[i] = new_beam_candidate[best_index]
                new_beam_candidate_tmp[best_index]["score"] = -1.0  # 追加したものを候補作成リストから除外

            # ビーム中でスコアが最も高いものが終了タグで終わっていれば終了
            if new_beam[0]["sentence"][-1] == EOS.s: return [new_beam[0]["sentence"]]
            else:
                beam = new_beam
        return [beam[0]["sentence"]] # 出力制限長まで達した場合は一番優秀なスコアのものを選択

    def beam_search_for_batch_1_use_all_vocab(self, state, generation_limit, beamwidth):
        assert len(state["list_a"]) == len(state["list_b"])
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = 1
        src_len = len(state["list_a"])
        hidden_size = self.n_hidden
        src_stoi = self.src_vocab.stoi
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        trg_wtoi = self.trg_vocab.wtoi
        trg_wtos = self.trg_vocab.wtos
        hidden_zeros = self.wrapper.zeros((batch_size, hidden_size))
        sum_e_zeros = self.wrapper.zeros((batch_size, 1))

        beam = []
        init_beam_item = {"c":state["c"], "s_p":state["s_p"], "s_y":state["s_y"], "sentence":[], "score":1.0}
        beam.append(init_beam_item)
        for _ in range(generation_limit):
            # 今のビームから次のビーム候補作成
            new_beam_candidate = [None] * (len(beam) * beamwidth)
            new_beam_candidate_index = 0
            for beam_item in beam:
                # calculate attention weights
                list_e = [None] * src_len
                sum_e = sum_e_zeros
                for n in range(src_len):
                    s_w = tanh(m.w_aw(state["list_a"][n]) + m.w_bw(state["list_b"][n]) + m.w_pw(beam_item["s_p"]))
                    r_e = functions.exp(m.w_we(s_w))
                    list_e[n] = r_e
                    sum_e += r_e

                # make attention vector
                s_c = hidden_zeros
                s_d = hidden_zeros
                for n in range(src_len):
                    s_e = list_e[n] / sum_e
                    s_c += functions.reshape(functions.batch_matmul(state["list_a"][n], s_e), (batch_size, hidden_size))
                    s_d += functions.reshape(functions.batch_matmul(state["list_b"][n], s_e), (batch_size, hidden_size))

                # generate softmax
                new_c, new_s_p = lstm(beam_item["c"], m.w_yp(beam_item["s_y"]) + m.w_pp(beam_item["s_p"]) + m.w_cp(s_c) + m.w_dp(s_d))
                r_y = m.w_py(new_s_p)
                softmax_all = functions.softmax(r_y)
                softmax_tmp = copy.deepcopy(softmax_all)

                # beamsizeだけ候補を追加
                for _ in range(beamwidth):
                    # 予測単語
                    best_str_index = self.xp.argmax(softmax_tmp)
                    new_sentence = beam_item["sentence"][:]
                    new_sentence.append(trg_itos(best_str_index))

                    # スコアは「品詞情報付き文字列」(「単語」ではない) の出現確率の積
                    new_score = beam_item["score"] * softmax_all[0][best_str_index]
                    new_s_y = self.wrapper.make_var([best_str_index], dtype=np.int32)

                    new_beam_item = {"c":new_c, "s_p":new_s_p, "s_y":new_s_y, "sentence":new_sentence, "score":new_score}
                    new_beam_candidate[new_beam_candidate_index] = new_beam_item
                    new_beam_candidate_index += 1

                    # 今回ビームに追加したものをソフトマックスから除外
                    softmax_tmp[0][best_str_index] = -1.0

            # 次のビームを作成
            new_beam = [None]  * beamwidth
            new_beam_candidate_tmp = copy.deepcopy(new_beam_candidate)
            for i in range(beamwidth):
                beam_candidate_score_list = self.xp.array([new_beam_candidate_item["score"] for new_beam_candidate_item in new_beam_candidate_tmp])
                best_index = self.xp.argmax(beam_candidate_score_list)
                new_beam[i] = new_beam_candidate[best_index]
                new_beam_candidate_tmp[best_index]["score"] = -1.0  # 追加したものを候補作成リストから除外

            # ビーム中でスコアが最も高いものが終了タグで終わっていれば終了
            if new_beam[0]["sentence"][-1] == EOS.s: return [new_beam[0]["sentence"]]
            else:
                beam = new_beam
        return [beam[0]["sentence"]] # 出力制限長まで達した場合は一番優秀なスコアのものを選択
