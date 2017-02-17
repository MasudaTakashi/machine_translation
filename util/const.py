# coding:UTF-8

# path
from os import path
PROJECT_PATH = path.dirname( path.abspath( __file__ ) )+ "/../../"
CORPUS_PATH = PROJECT_PATH + "data/corpus/"
SRC_PATH = PROJECT_PATH + "src/"

### コーパス関連
# 参照するコーパス
USE_EACH_FILE_FLG   = 1
CORPUS_SET = 0

if CORPUS_SET == 0:
    # 動作確認用で5文のみ。田中コーパスの先頭5文
    # 学習データと開発データとテストデータが全て同じ
    # vocab_sizeは20にしてあるので未知語あり
    TRAIN_FILE_SRC  = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    VALID_FILE_SRC  = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    TEST_FILE_SRC   = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/tanaka/5.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/tanaka/5.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"tanaka/corpus_file_5.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"tanaka/corpus_file_5.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/tanaka/5.label.en.vocab"
elif CORPUS_SET == 1:
    # 動作確認用で5文のみ。田中コーパスの先頭5文
    # 学習データと開発データとテストデータが全て同じ
    # 未知語なし
    TRAIN_FILE_SRC  = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    VALID_FILE_SRC  = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    TEST_FILE_SRC   = CORPUS_PATH+"tanaka/corpus_file_5.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"tanaka/corpus_file_5.eu"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/tanaka/5.full.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/tanaka/5.full.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"tanaka/corpus_file_5.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"tanaka/corpus_file_5.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/tanaka/5.label.full.en.vocab"
elif CORPUS_SET == 5:
    # tanakaコーパス1000文
    # vpcab_sizeに制限なし
    TRAIN_FILE_SRC  = CORPUS_PATH+"oda/train1000.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"oda/train1000.en"
    VALID_FILE_SRC  = CORPUS_PATH+"oda/test100.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"oda/test100.en"
    TEST_FILE_SRC   = CORPUS_PATH+"oda/test100.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"oda/test100.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/oda/all/train1000.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/oda/all/train1000.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"oda/train1000.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"oda/test100.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/oda/all/train1000.label.en.vocab"
elif CORPUS_SET == 6:
    # tanakaコーパス1000文
    # vpcab_sizeに制限なし
    # vocabを新しく作り直したもの
    TRAIN_FILE_SRC  = CORPUS_PATH+"oda/train1000.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"oda/train1000.en"
    VALID_FILE_SRC  = CORPUS_PATH+"oda/test100.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"oda/test100.en"
    TEST_FILE_SRC   = CORPUS_PATH+"oda/test100.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"oda/test100.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/oda/all/train1000_.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/oda/all/train1000_.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"oda/train1000.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"oda/test100.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/oda/all/train1000_.label.en.vocab"
elif CORPUS_SET == 10:
    # NLP2016用
    # vocab_sizeに制限なし
    TRAIN_FILE_SRC  = CORPUS_PATH+"nlp_last/train.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"nlp_last/train.en"
    VALID_FILE_SRC  = CORPUS_PATH+"nlp_last/dev.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"nlp_last/dev.en"
    TEST_FILE_SRC   = CORPUS_PATH+"nlp_last/test1000.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"nlp_last/test1000.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/nlp_last/all/train.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/nlp_last/all/train.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/train.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/dev.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/nlp_last/all/train.label.en.vocab"
elif CORPUS_SET == 11:
    # NLP2016用
    # vocab_sizeに制限あり
    TRAIN_FILE_SRC  = CORPUS_PATH+"nlp_last/train.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"nlp_last/train.en"
    VALID_FILE_SRC  = CORPUS_PATH+"nlp_last/dev.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"nlp_last/dev.en"
    TEST_FILE_SRC   = CORPUS_PATH+"nlp_last/test1000.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"nlp_last/test1000.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/nlp_last/notall/train.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/nlp_last/notall/train.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/train.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/dev.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/nlp_last/notall/train.label.en.vocab"
elif CORPUS_SET == 12:
    # NLP2016用
    # vocab_size::4096
    TRAIN_FILE_SRC  = CORPUS_PATH+"nlp_last/train.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"nlp_last/train.en"
    VALID_FILE_SRC  = CORPUS_PATH+"nlp_last/dev.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"nlp_last/dev.en"
    TEST_FILE_SRC   = CORPUS_PATH+"nlp_last/test1000.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"nlp_last/test1000.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/train.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/dev.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.label.en.vocab"
elif CORPUS_SET == 13:
    # NLP2016用
    # vocab_size::4096
    TRAIN_FILE_SRC  = CORPUS_PATH+"nlp_last/train.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"nlp_last/train.en"
    VALID_FILE_SRC  = CORPUS_PATH+"nlp_last/dev.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"nlp_last/dev.en"
    TEST_FILE_SRC   = CORPUS_PATH+"nlp_last/sample.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"nlp_last/sample.ja"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/train.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"nlp_last/dev.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/nlp_last/4096/train.label.en.vocab"
elif CORPUS_SET == 20:
    # 卒論用？
    # vocab_sizeに制限あり
    TRAIN_FILE_SRC  = CORPUS_PATH+"bachelor/train-all-clean.ja"
    TRAIN_FILE_TRG  = CORPUS_PATH+"bachelor/train-all-clean.en"
    VALID_FILE_SRC  = CORPUS_PATH+"bachelor/dev.ja"
    VALID_FILE_TRG  = CORPUS_PATH+"bachelor/dev.en"
    TEST_FILE_SRC   = CORPUS_PATH+"bachelor/test.ja"
    TEST_FILE_TRG   = CORPUS_PATH+"bachelor/test.en"
    VOCAB_SRC       = PROJECT_PATH+"data/vocab/bachelor/train-all-clean.ja.vocab"
    VOCAB_TRG       = PROJECT_PATH+"data/vocab/bachelor/train-all-clean.en.vocab"
    TRAIN_FILE_TRG_WL  = CORPUS_PATH+"bachelor/train-all-clean.label.en"
    VALID_FILE_TRG_WL  = CORPUS_PATH+"bachelor/dev.label.en"
    VOCAB_TRG_WL       = PROJECT_PATH+"data/vocab/bachelor/train-all-clean.label.en.vocab"


### RNN関連
MAX_EPOCH = 1000


### 変数
class token(object):
    def __init__(self, string, index):
        self.__s = string
        self.__i = index

    def __str__(self):
        return self.__s

    def __getattr__(self, key):
        if key == 'index' or key == "i":
            return self.__i
        elif key == "token" or key == "str" or key == "tok" or key == "s":
            return self.__s
        else:
            raise AttributeError

#SKIP_INDEX  = -1
UNK = token("<unk>", 0)
SOS = token("<s>", 1)
EOS = token("<eos>", 2)
UNK_CC  = token("<unk>/CC", 3)
UNK_CD  = token("<unk>/CD", 4)
UNK_DT  = token("<unk>/DT", 5)
UNK_EX  = token("<unk>/EX", 6)
UNK_FW  = token("<unk>/FW", 7)
UNK_IN  = token("<unk>/IN", 8)
UNK_JJ  = token("<unk>/JJ", 9)
UNK_JJR = token("<unk>/JJR", 10)
UNK_JJS = token("<unk>/JJS", 11)
UNK_LS  = token("<unk>/LS", 12)
UNK_MD  = token("<unk>/MD", 13)
UNK_NN  = token("<unk>/NN", 14)
UNK_NNP = token("<unk>/NNP", 15)
UNK_NNPS= token("<unk>/NNPS", 16)
UNK_NNS = token("<unk>/NNS", 17)
UNK_POS = token("<unk>/POS", 18)
UNK_PDT = token("<unk>/PDT", 19)
UNK_PRPd= token("<unk>/PRP$", 20)
UNK_PRP = token("<unk>/PRP", 21)
UNK_RB  = token("<unk>/RB", 22)
UNK_RBR = token("<unk>/RBR", 23)
UNK_RBS = token("<unk>/RBS", 24)
UNK_RP  = token("<unk>/RP", 25)
UNK_SYM = token("<unk>/SYM", 26)
UNK_TO  = token("<unk>/TO", 27)
UNK_UH  = token("<unk>/UH", 28)
UNK_URL = token("<unk>/URL", 29)
UNK_VB  = token("<unk>/VB", 30)
UNK_VBD = token("<unk>/VBD", 31)
UNK_VBD = token("<unk>/VBG", 32)
UNK_VBN = token("<unk>/VBN", 33)
UNK_VBP = token("<unk>/VBP", 34)
UNK_VBZ = token("<unk>/VBZ", 35)
UNK_WDT = token("<unk>/WDT", 36)
UNK_WP  = token("<unk>/WP", 37)
UNK_WPd = token("<unk>/WP$", 38)
UNK_WRB = token("<unk>/WRB", 39)
UNK_k   = token("<unk>/,", 40)
UNK_p   = token("<unk>/.", 41)
UNK_c   = token("<unk>/:", 42)
UNK_d   = token("<unk>/$", 43)
UNK_s   = token("<unk>/#", 44)
UNK_q   = token('<unk>/"', 45)
UNK_l   = token("<unk>/(", 46)
UNK_r   = token("<unk>/)", 47)

symbols = [UNK, SOS, EOS]

tags = [
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "POS",
    "PDT",
    "PRP$",
    "PRP",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    "URL",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    ",",
    ".",
    ":",
    "$",
    "#",
    "''",
    "``"
    "(",
    ")",
]

# juman中の品詞リスト
types = [
    "形容詞",
    "連体詞",
    "副詞",
    "判定詞",
    "助動詞",
    "接続詞",
    "指示詞",
    "感動詞",
    "名詞",
    "動詞",
    "助詞",
    "接頭辞",
    "接尾辞",
    "特殊",
    "未定義語",
]
