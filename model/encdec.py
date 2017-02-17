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

class EncoderDecoderModel(BaseModel):
    def make_model(self):
        initialW = np.random.uniform
        self.model = self.wrapper.make_model(
            # encoder
            w_xi = functions.EmbedID(len(self.src_vocab), self.n_embed),
            w_ip = functions.Linear(self.n_embed,   4 * self.n_hidden, initialW=initialW(-0.1, 0.1, (    4*self.n_hidden, self.n_embed))),
            w_pp = functions.Linear(self.n_hidden,  4 * self.n_hidden, initialW=initialW(-0.1, 0.1, (    4*self.n_hidden, self.n_hidden))),
            # decoder
            w_pq = functions.Linear(self.n_hidden,  4 * self.n_hidden, initialW=initialW(-0.1, 0.1, (    4*self.n_hidden, self.n_hidden))),
            w_qj = functions.Linear(self.n_hidden,       self.n_embed, initialW=initialW(-0.1, 0.1, (       self.n_embed, self.n_hidden))),
            w_jy = functions.Linear(self.n_embed, len(self.trg_vocab), initialW=initialW(-0.1, 0.1, (len(self.trg_vocab), self.n_embed))),
            w_yq = functions.EmbedID(len(self.trg_vocab), 4 * self.n_hidden),
            w_qq = functions.Linear(self.n_hidden,  4 * self.n_hidden, initialW=initialW(-0.1, 0.1, (    4*self.n_hidden, self.n_hidden))),
        )

    def forward(self, is_training, src_batch, trg_batch=None, generation_limit=None, beamwidth=None, is_greedy=False, use_all_vocab=False):
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        batch_size = len(src_batch)
        src_len = len(src_batch[0])
        src_stoi = self.src_vocab.stoi
        trg_stow = self.trg_vocab.stow
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        trg_itow = self.trg_vocab.itow
        trg_word = self.trg_vocab.word
        trg_wstois = self.trg_vocab.wstois

        s_c = self.wrapper.zeros((batch_size, self.n_hidden))
        s_p = None

        # encoding
        for l in range(src_len):
            s_x = self.wrapper.make_var([src_stoi(src_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
            s_i = tanh(m.w_xi(s_x))
            if s_p is None:
                s_c, s_p = lstm(s_c, m.w_ip(s_i))
            else:
                s_c, s_p = lstm(s_c, m.w_ip(s_i) + m.w_pp(s_p))
        s_c, s_q = lstm(s_c, m.w_pq(s_p))

        hyp_batch = [[] for _ in range(batch_size)]

        # decoding
        if is_training:
            accum_loss = self.wrapper.zeros(())
            accum_accuracy = self.wrapper.zeros(())
            sum_log_likelihood = 0.0
            trg_len = len(trg_batch[0])

            for l in range(trg_len):
                s_j = tanh(m.w_qj(s_q))
                r_y = m.w_jy(s_j)

                s_t = self.wrapper.make_var([trg_stoi(trg_batch[k][l]) for k in range(batch_size)], dtype=np.int32)
                accum_loss += functions.softmax_cross_entropy(r_y, s_t)
                accum_accuracy += functions.accuracy(r_y, s_t)
                softmax = self.softmax(r_y)
                ans_word_indexes = [trg_itow(trg_stoi(trg_batch[k][l])) for k in range(batch_size)]
                sum_log_likelihood += log_likelihood(softmax, ans_word_indexes)
                if self.output:
                    output = softmax.argmax(1)
                    for k in range(batch_size):
                        hyp_batch[k].append(self.trg_vocab.word(int(output[k])))

                s_c, s_q = lstm(s_c, m.w_yq(s_t) + m.w_qq(s_q)) # 一個前の入力は品詞付き文字列

            return hyp_batch, accum_loss, accum_accuracy, sum_log_likelihood
        else:
            search = not is_greedy
            if search:  # beam探索
                state = {"s_c":s_c, "s_q":s_q}
                if use_all_vocab:
                    return self.beam_search_for_batch_1_use_all_vocab(state, generation_limit, beamwidth)
                else:
                    return self.beam_search_for_batch_1(state, generation_limit, beamwidth)
            else:   # 貪欲にスコアが高いものを出力
                while len(hyp_batch[0]) < generation_limit:
                    s_j = tanh(m.w_qj(s_q))
                    r_y = m.w_jy(s_j)

                    output = self.softmax(r_y).argmax(1)
                    for k in range(batch_size):
                        hyp_batch[k].append(trg_word(output[k]))

                    if all(EOS.s in set(hyp_batch[k]) for k in range(batch_size)): break

                    s_y = self.wrapper.make_var(trg_wstois(output, functions.softmax(r_y)), dtype=np.int32)
                    s_c, s_q = lstm(s_c, m.w_yq(s_y) + m.w_qq(s_q))

                return hyp_batch

    def beam_search_for_batch_1(self, lstm_state, generation_limit, beamwidth):
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        src_stoi = self.src_vocab.stoi
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        s_c = lstm_state["s_c"]
        s_q = lstm_state["s_q"]

        beam = []
        init_beam_item = {"s_c":s_c, "s_q":s_q, "sentence":[], "score":1.0}
        beam.append(init_beam_item)
        for _ in range(generation_limit):
            # 今のビームから次のビーム候補作成
            new_beam_candidate = []
            for beam_item in beam:
                s_j = tanh(m.w_qj(beam_item["s_q"]))
                r_y = m.w_jy(s_j)
                softmax = self.softmax(r_y)
                softmax_tmp = copy.deepcopy(softmax)
                softmax_all = functions.softmax(r_y)
                for _ in range(beamwidth):
                    new_s_c = copy.deepcopy(beam_item["s_c"])
                    new_s_q = copy.deepcopy(beam_item["s_q"])
                    # 予測単語
                    best_word_index = self.xp.argmax(softmax_tmp)
                    new_sentence = beam_item["sentence"][:]
                    #word_with_label = self.trg_vocab.wtos(best_word_index, softmax_all)
                    #new_sentence.append(word_with_label)
                    new_sentence.append(self.trg_vocab.wtos(best_word_index, softmax_all))

                    # スコアは「単語」(「品詞情報付き文字列」ではない) の出現確率の積
                    new_score = beam_item["score"] * softmax[0][best_word_index]

                    # 出力単語を入力して次の隠れ層を生成
                    #s_y = self.wrapper.make_var([trg_stoi(word_with_label)], dtype=np.int32)
                    input_vec = self.make_input_weighting_vec(best_word_index, m.w_yq, softmax_all)
                    new_s_c, new_s_q = lstm(new_s_c, input_vec + m.w_qq(new_s_q))

                    new_beam_item = {"s_c":new_s_c, "s_q":new_s_q, "sentence":new_sentence, "score":new_score}
                    new_beam_candidate.append(new_beam_item)

                    # 今回ビームに追加したものをソフトマックスから除外
                    softmax_tmp[0][best_word_index] = -1.0

            # 次のビームを作成
            new_beam = []
            new_beam_candidate_tmp = copy.deepcopy(new_beam_candidate)
            for _ in range(beamwidth):
                beam_candidate_score_list = self.xp.array([new_beam_candidate_item["score"] for new_beam_candidate_item in new_beam_candidate_tmp])
                best_index = self.xp.argmax(beam_candidate_score_list)
                new_beam.append(new_beam_candidate[best_index])
                new_beam_candidate_tmp[best_index]["score"] = -1.0  # 追加したものを候補作成リストから除外

            # ビーム中でスコアが最も高いものが終了タグで終わっていれば終了
            if new_beam[0]["sentence"][-1] == EOS.s: return [new_beam[0]["sentence"]]
            else:
                beam = new_beam
        return [beam[0]["sentence"]] # 出力制限長まで達した場合は一番優秀なスコアのものを選択

    def beam_search_for_batch_1_use_all_vocab(self, lstm_state, generation_limit, beamwidth):
        m = self.model
        tanh = functions.tanh
        lstm = functions.lstm
        src_stoi = self.src_vocab.stoi
        trg_stoi = self.trg_vocab.stoi
        trg_itos = self.trg_vocab.itos
        s_c = lstm_state["s_c"]
        s_q = lstm_state["s_q"]

        beam = []
        init_beam_item = {"s_c":s_c, "s_q":s_q, "sentence":[], "score":1.0}
        beam.append(init_beam_item)
        for _ in range(generation_limit):
            # 今のビームから次のビーム候補作成
            new_beam_candidate = []
            for beam_item in beam:
                s_j = tanh(m.w_qj(beam_item["s_q"]))
                r_y = m.w_jy(s_j)
                softmax_all = functions.softmax(r_y)
                softmax_tmp = copy.deepcopy(softmax_all)
                for _ in range(beamwidth):
                    new_s_c = copy.deepcopy(beam_item["s_c"])
                    new_s_q = copy.deepcopy(beam_item["s_q"])
                    # 予測単語
                    # vocabの中から最大確率の文字列
                    best_str_index = self.xp.argmax(softmax_tmp)
                    new_sentence = beam_item["sentence"][:]
                    new_sentence.append(trg_itos(best_str_index))

                    # スコアは「品詞情報付き文字列」（「単語」ではない) の出現確率の積
                    new_score = beam_item["score"] * softmax_all[0][best_str_index]

                    s_y = self.wrapper.make_var([best_str_index], dtype=np.int32)
                    new_s_c, new_s_q = lstm(new_s_c, m.w_yq(s_y) + m.w_qq(new_s_q))

                    new_beam_item = {"s_c":new_s_c, "s_q":new_s_q, "sentence":new_sentence, "score":new_score}
                    new_beam_candidate.append(new_beam_item)

                    # 今回ビームに追加したものをソフトマックスから除外
                    softmax_tmp[0][best_str_index] = -1.0

            # 次のビームを作成
            new_beam = []
            new_beam_candidate_tmp = copy.deepcopy(new_beam_candidate)
            for _ in range(beamwidth):
                beam_candidate_score_list = self.xp.array([new_beam_candidate_item["score"] for new_beam_candidate_item in new_beam_candidate_tmp])
                best_index = self.xp.argmax(beam_candidate_score_list)
                new_beam.append(new_beam_candidate[best_index])
                new_beam_candidate_tmp[best_index]["score"] = -1.0  # 追加したものを候補作成リストから除外

            # ビーム中でスコアが最も高いものが終了タグで終わっていれば終了
            if new_beam[0]["sentence"][-1] == EOS.s: return [new_beam[0]["sentence"]]
            else:
                beam = new_beam
        return [beam[0]["sentence"]] # 出力制限長まで達した場合は一番優秀なスコアのものを選択
