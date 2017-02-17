import pickle
import numpy as np
from collections import defaultdict
from .const import *

def dd():
    return 0

def unk():
    return UNK.i

class OldVocabulary(object):
    def __init__(self):
        self.__stoi = None
        self.__itos = None
        self.__size = None

    def __len__(self):
        return self.__size

    def stoi(self, s):
        return self.__stoi[s]

    def itos(self, i):
        return self.__itos[i]

    @classmethod
    def new(cls, list_generator, size, thres):
        # 語彙サイズが指定されているとき重要単語から語彙にするため、出現頻度を保持
        word_freq = defaultdict(dd)
        for words in list_generator:
            for word in words:
                word_freq[word] += 1

        # 頻度によって制限
        word_freq_ = defaultdict(dd)
        for key in word_freq:
            if word_freq[key] >= thres: word_freq_[key] = word_freq[key]
        word_freq = word_freq_

        self = cls()
        self.__size = size if size > 0 else len(word_freq)+3

        n_const_vocab = 0
        self.__stoi = defaultdict(unk)   # 見知らぬ単語は<unk>になる
        self.__stoi[UNK.s] = UNK.i; n_const_vocab += 1
        self.__stoi[SOS.s] = SOS.i; n_const_vocab += 1
        self.__stoi[EOS.s] = EOS.i; n_const_vocab += 1
        self.__itos = [''] * self.__size
        self.__itos[UNK.i] = UNK.s
        self.__itos[SOS.i] = SOS.s
        self.__itos[EOS.i] = EOS.s

        for i, (k, v) in zip(range(self.__size - n_const_vocab), sorted(word_freq.items(), key=lambda x: -x[1])):
            self.__stoi[k] = i + n_const_vocab
            self.__itos[i + n_const_vocab] = k

        return self

    def save(self, filename):
        save_data = {
            "stoi":self.__stoi,
            "itos":self.__itos,
            "size":self.__size,
        }
        with open(filename, "wb") as fp:
            pickle.dump(save_data, fp)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as fp:
            load_data = pickle.load(fp)
        self = cls()
        self.__stoi = load_data["stoi"]
        self.__itos = load_data["itos"]
        self.__size = load_data["size"]
        return self


class Vocabulary(object):
    def __init__(self):
        self.__stoi = None
        self.__itos = None
        self.__size = None
        self.__stow = None
        self.__itow = None
        self.__word = None

    # 品詞ありの辞書のサイズ
    def __len__(self):
        return len(self.__stoi)

    # 品詞なしの単語のみの辞書サイズ
    def size(self):
        return self.__size

    # @param:   文字列
    # @return:  品詞情報付き文字列のインデックス
    def stoi(self, s):
        if s in self.__itos:
            return self.__stoi[s]
        else:
            # 単語/品詞 自体が未知語
            # -> <unk>/品詞 として認識
            s_info = s.split("/")
            if len(s_info) == 2:
                s_tag = s_info[1]
                return self.__stoi[UNK.s+"/"+s_tag]
            else:
                # 品詞情報なし -> <unk> として認識
                return self.__stoi[UNK.s]

    # @param:   品詞情報付き文字列のインデックス
    # @return:  文字列
    def itos(self, i):
        return self.__itos[i]

    # @param:   文字列
    # @return:  単語のインデックス
    def stow(self, s):
        return self.__stow[s]

    # @param:   品詞情報付き文字列のインデックス
    # @return:  単語のインデックス
    def itow(self, i):
        return self.__itow[i]

    # 品詞付き単語で、最大の確率をもつものを返す
    # @param:   単語のインデックス(list)
    # @param:   品詞情報まで含めた文字列に対するsoftmax(ndarray)
    # @return:  品詞情報付き文字列のインデックス(list)
    def wstois(self, ws, softmax_all):
        softmax_all = softmax_all.data
        assert len(ws) == len(softmax_all)
        batch_size = len(ws)
        is_target_word = np.array([[1 if self.__itow[i] == ws[bi] else 0 for i in range(len(self.__itow))] for bi in range(batch_size)])
        softmax_all_only_trg_word = softmax_all * is_target_word    # 該当単語以外は0の確率分布
        return softmax_all_only_trg_word.argmax(1)

    # 品詞付き単語で、最大の確率をもつものを返す
    # @param:   単語のインデックス
    # @param:   品詞情報まで含めた文字列に対するsoftmax(ndarray)
    # @return:  品詞情報付き文字列のインデックス
    def wtoi(self, w, softmax_all):
        softmax_all_data = softmax_all.data
        is_target_word = np.array([1 if self.__itow[i] == w else 0 for i in range(len(self.__itow))])
        softmax_all_only_trg_word = softmax_all_data * is_target_word    # 該当単語以外は0の確率分布
        return softmax_all_only_trg_word.argmax(1)

    def wtois(self, w):
        return [i for i in range(len(self.__itow)) if self.__itow[i] == w]

    def wtos(self, w, softmax_all):
        i = self.wtoi(w, softmax_all)
        return self.itos(i)

    # @param:   単語のインデックス
    # @return:  単語
    def word(self, w):
        return self.__word[w]

    @classmethod
    def new(cls, list_generator, size, thres, with_label):
        if with_label:
            labels = tags
        else:
            labels = []


        # 語彙サイズが指定されているとき重要単語から語彙にするため、出現頻度を保持
        str_freq    = defaultdict(dd)
        word_freq   = defaultdict(dd)
        for strs in list_generator:
            for str in strs:
                word = str.split("/")[0] # 品詞情報付きの場合"単語/品詞"から単語を取得
                str_freq[str]   += 1
                word_freq[word] += 1

        # 頻度によって制限
        word_freq_ = defaultdict(dd)
        for key in word_freq:
            if word_freq[key] >= thres: word_freq_[key] = word_freq[key]
        word_freq = word_freq_

        if with_label:
            print("word_freq::%d" % len(word_freq))
            print("str_freq::%d" % len(str_freq))
        else:
            print("n_vocab_word::%d" % len(word_freq))
            print("n_all_word::%d" % len(str_freq))

        self = cls()
        # size未指定時は全単語数と開始終了未知語
        self.__size = size if size > 0 else len(word_freq)+len(symbols)
        self.__stoi = {}
        self.__itos = [""] * (len(str_freq) + len(symbols) + len(labels))
        self.__word = [""] * self.__size
        self.__stow = {}
        self.__itow = [""] * (len(str_freq) + len(symbols) + len(labels))

        # 辞書中に必ず存在するもの
        # 開始、終了、未知語
        n_const_vocab = 0
        for symbol in symbols:
            self.__stoi[symbol.s] = symbol.i
            self.__itos[symbol.i] = symbol.s
            self.__word[symbol.i] = symbol.s
            self.__stow[symbol.s] = symbol.i
            self.__itow[symbol.i] = symbol.i
            n_const_vocab += 1
        # 品詞ラベル付き未知語
        for label in labels:
            index   = n_const_vocab
            str     = UNK.s + "/" + label
            self.__stoi[str]    = index
            self.__stow[str]    = UNK.i
            self.__itos[index]  = str
            self.__itow[index]  = UNK.i
            n_const_vocab += 1


        # その他辞書の要素を形成
        for i, (k, v) in zip(range(self.__size - len(symbols)), sorted(word_freq.items(), key=lambda x: -x[1])):
            self.__word[i + len(symbols)] = k
        i = 0
        for k, v in sorted(str_freq.items(), key=lambda x: -x[1]):
            word = k.split("/")[0]
            if word in self.__word:
                self.__stoi[k] = i + n_const_vocab
                self.__itos[i + n_const_vocab] = k
                word_index = self.__word.index(word)
                self.__stow[k] = word_index
                self.__itow[i + n_const_vocab] = word_index
                i += 1

        # 変数中の""を削除
        self.__itos = [x for x in self.__itos if x is not ""]
        self.__itow = [x for x in self.__itow if x is not ""]

        return self

    def save(self, filename):
        save_data = {
            "stoi":self.__stoi,
            "itos":self.__itos,
            "size":self.__size,
            "word":self.__word,
            "stow":self.__stow,
            "itow":self.__itow
        }
        with open(filename, "wb") as fp:
            pickle.dump(save_data, fp)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as fp:
            load_data = pickle.load(fp)
        self = cls()
        self.__stoi = load_data["stoi"]
        self.__itos = load_data["itos"]
        self.__size = load_data["size"]
        self.__word = load_data["word"]
        self.__stow = load_data["stow"]
        self.__itow = load_data["itow"]
        return self

    def print_data(self, target):
        for i in range(len(target)):
            print(target[i])