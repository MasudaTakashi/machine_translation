import os
import sys
import time
import math
import copy
import numpy as np
from datetime import datetime

from .functions import trace, fill_batch, fill_batch2, removeLabel, removeAllLabel
from . import generators as gens
from .const import *
from . import bleu
from . import ribes

class Controller:
    def __init__(self, folder_name, fill_option=False, with_label=False):
        ### 結果出力用のディレクトリを準備 ###
        self.resultFolder = "result" + datetime.now().strftime("%Y%m%d%H%M%S")
        if folder_name != "": self.resultFolder = folder_name
        os.mkdir(self.resultFolder)
        self.with_label = with_label
        self.fill_SOS = fill_option
        self.fill_batch = fill_batch2 if fill_option else fill_batch
        # コーパスの設定
        self.train_src  = TRAIN_FILE_SRC
        self.train_trg  = TRAIN_FILE_TRG
        self.dev_src    = VALID_FILE_SRC
        self.dev_trg    = VALID_FILE_TRG
        self.test_src   = TEST_FILE_SRC
        self.test_trg   = TEST_FILE_TRG
        if self.with_label:
            self.train_trg  = TRAIN_FILE_TRG_WL
            self.dev_trg    = VALID_FILE_TRG_WL


    def train_model(self, net, src_vocab, trg_vocab, args):
        self.lr = args.lr
        self.initial_epoch = 0
        # モデル準備
        if args.model == "":
            trace('making model ...')
            model = net.new(args, src_vocab, trg_vocab)
        else:
            trace('loading model ...')
            model = net.load(args, src_vocab, trg_vocab)
            self.initial_epoch = int(args.model[-9:-6])
        # 情報をファイルに出力
        self.outputInfo(args, model, src_vocab, trg_vocab)

        # プロット用ファイルにヘッダ出力
        with open(self.resultFolder+"/result.txt", "a") as result_file:
            result_file.write("#epoch\ttrain_l\tvalidate_l\ttrain_a\tvalidate_a\tlearn_time[sec]\n")

        n_train_sentence = self.getLineNum(self.train_src, self.train_trg)
        prev_likelihood = float("-inf")
        model.init_optimizer(self.lr)
        for epoch in range(args.epoch):
            trace('epoch %d/%d: ' % (epoch+self.initial_epoch+1, args.epoch))
            trained = 0
            raw_src_batch = gens.word_list(self.train_src)
            raw_trg_batch = gens.word_list(self.train_trg)
            batches = gens.batch(gens.sorted_parallel(raw_src_batch, raw_trg_batch, 100 * args.minibatch), args.minibatch)


            sum_log_likelihood = 0.0
            sum_accuracy    = 0.0
            batch_loop_cnt  = 0
            start_time = time.time()
            for src_batch, trg_batch in batches:
                src_batch = self.fill_batch(src_batch, is_reverse=model.is_reverse)
                trg_batch = self.fill_batch(trg_batch)
                K = len(src_batch)
                hyp_batch, log_likelihood, accuracy = model.train(src_batch, trg_batch)

                sum_log_likelihood += log_likelihood
                sum_accuracy += accuracy

                if args.output:
                    for k in range(K):
                        if self.with_label == True:
                            trg_batch[k] = removeLabel(trg_batch[k])
                            hyp_batch[k] = removeLabel(hyp_batch[k])
                        trace('epoch %3d/%3d, sample %8d' % (epoch+self.initial_epoch+1, args.epoch, trained + k + 1))
                        src = [x if x != EOS.s else '*' for x in src_batch[k]]
                        trg = [x if x != EOS.s else '*' for x in trg_batch[k]]
                        hyp = [x if x != EOS.s else '*' for x in hyp_batch[k]]
                        if model.is_reverse:
                            src.reverse()
                        trace('  src = ' + ' '.join(src))
                        trace('  trg = ' + ' '.join(trg))
                        trace('  hyp = ' + ' '.join(hyp))
                else:
                    self.printProgressInfo(n_train_sentence, trained)
                trained += K
                batch_loop_cnt += 1

            self.clearProgressInfo()
            # 1エポックあたりの学習時間
            run_time = time.time() - start_time

            trace('saving model ...')
            model.save(self.resultFolder + '/epoch-%03d.model' % (epoch+self.initial_epoch+1))

            # 評価
            train_log_likelihood = sum_log_likelihood / n_train_sentence
            train_accuracy = sum_accuracy / batch_loop_cnt
            validate_log_likelihood, validate_accuracy = self.validate(model, args)

            # 損失の出力
            trace('log_likelihood and accuracy:')
            trace('  train   : %f, %f' % (train_log_likelihood,     train_accuracy))
            trace('  validate: %f, %f' % (validate_log_likelihood,  validate_accuracy))
            sys.stderr.write('\n\n')
            with open(self.resultFolder+"/result.txt", "a") as result_file:
                result_file.write("%d\t%f\t%f\t%f\t%f\t%f\n" % (epoch+self.initial_epoch+1,
                                                                train_log_likelihood, validate_log_likelihood,
                                                                train_accuracy, validate_accuracy,
                                                                run_time))
            # 学習率のスケジューリング
            is_schedule = False
            if is_schedule:
                if epoch + 1 >= 5:
                    # 5エポック以降は半分に
                    model.opt.lr /= 2
                print(model.opt.lr)

        trace('finished.')

    def validate(self, model, args):
        raw_src_batch = gens.word_list(self.dev_src)
        raw_trg_batch = gens.word_list(self.dev_trg)
        batches = gens.batch(gens.sorted_parallel(raw_src_batch, raw_trg_batch, 100 * args.minibatch), args.minibatch)

        n_validate_sentence = self.getLineNum(self.dev_src, self.dev_trg)
        trained = 0
        batch_loop_cnt = 0
        sum_log_likelihood = 0.0
        sum_accuracy = 0.0
        for src_batch, trg_batch in batches:
            src_batch = self.fill_batch(src_batch, is_reverse=model.is_reverse)
            trg_batch = self.fill_batch(trg_batch)
            K = len(src_batch)
            hyp_batch, log_likelihood, accuracy = model.evaluate(src_batch, trg_batch)

            sum_log_likelihood += log_likelihood
            sum_accuracy += accuracy

            if args.output:
                for k in range(K):
                    if self.with_label == True:
                        trg_batch[k] = removeLabel(trg_batch[k])
                        hyp_batch[k] = removeLabel(hyp_batch[k])
                    trace('sample %8d' % (trained + k + 1))
                    src = [x if x != EOS.s else '*' for x in src_batch[k]]
                    trg = [x if x != EOS.s else '*' for x in trg_batch[k]]
                    hyp = [x if x != EOS.s else '*' for x in hyp_batch[k]]
                    if model.is_reverse:
                        src.reverse()
                    trace('  src = ' + ' '.join(src))
                    trace('  trg = ' + ' '.join(trg))
                    trace('  hyp = ' + ' '.join(hyp))
            else:
                self.printProgressInfo(n_validate_sentence, trained)
            trained += K
            batch_loop_cnt += 1

        self.clearProgressInfo()

        log_likelihood = sum_log_likelihood / n_validate_sentence
        accuracy = sum_accuracy / batch_loop_cnt
        return log_likelihood, accuracy


    def test_model(self, net, src_vocab, trg_vocab, args):
        trace('loading model ...')
        model = net.load(args, src_vocab, trg_vocab)
        with open(self.resultFolder+"/result.txt", "a") as fp:
            fp.write("#loaded model path: %s" % args.model)

        self.outputInfo(args, model, src_vocab, trg_vocab)

        trace('generating translation ...')
        generated = 0
        n_test_sentence = self.getLineNum(self.test_src, self.test_trg)
        hyp_file = self.resultFolder+"/hyp.txt"
        all_file = self.resultFolder+"/all.txt"
        fp_hype_file = open(hyp_file, 'w')
        fp_all_file = open(all_file, "w")

        raw_src_batch = gens.batch(gens.word_list(self.test_src), args.minibatch)
        raw_trg_batch = gens.batch(gens.word_list(self.test_trg), args.minibatch)
        for src_batch, trg_batch in zip(raw_src_batch, raw_trg_batch):
            src_batch = self.fill_batch(src_batch, is_reverse=model.is_reverse)
            trg_batch = self.fill_batch(trg_batch)
            K = len(src_batch)

            hyp_batch = model.predict(src_batch, args.generation_limit, args.beamwidth, use_all_vocab=args.all)
            generated += K
            self.printProgressInfo(n_test_sentence, generated)

            for k in range(K):
                hyp = hyp_batch[k]
                src = src_batch[k]
                trg = trg_batch[k]
                hyp.append(EOS.s)
                hyp = hyp[:hyp.index(EOS.s)]    # EOSタグが出るまで出力
                src = src[:src.index(EOS.s)]
                trg = trg[:trg.index(EOS.s)]
                hyp_wl = hyp[:]
                if self.with_label == True:
                    hyp = removeAllLabel(hyp)
                    hyp_wl = removeLabel(hyp_wl)

                print(' '.join(hyp), file=fp_hype_file)

                if model.is_reverse:
                    src.reverse()
                print("src: "+' '.join(src), file=fp_all_file)
                print("trg: "+' '.join(trg), file=fp_all_file)
                print("hyp: "+' '.join(hyp_wl), file=fp_all_file)
                print("", file=fp_all_file)
        self.clearProgressInfo()

        fp_hype_file.close()
        fp_all_file.close()
        bleu_val = bleu.calc_bleu(hyp_file, TEST_FILE_TRG)
        trace('BLEU :  %8f' % (bleu_val))
        with open(self.resultFolder+"/result.txt", "a") as fp:
            fp.write("#BLEU : %8f" % (bleu_val))

        trace('finished.')


    def dev_model(self, net, src_vocab, trg_vocab, args_init):
        args = copy.deepcopy(args_init)
        model_number = args_init.epoch
        n_dev_sentence = self.getLineNum(self.dev_src, self.dev_trg)
        self.outputInfo(args_init, None, src_vocab, trg_vocab)
        with open(self.resultFolder+"/result.txt", "a") as fp:
            fp.write("#epoch\tBLEU\tRIBES\n")

        while True:
            model = None
            while True:
                try:
                    if args.score:
                        self.hyp = args_init.model+"/hyp%s.txt" % (str(model_number).zfill(3))
                    else:
                        args.model = args_init.model + "/epoch-%s.model" % (str(model_number).zfill(3))
                        trace('loading model ... %3d' % model_number)
                        model = net.load(args, src_vocab, trg_vocab)
                    break
                except OSError:
                    if model_number == 1:
                        trace("error! - miss the name of model folder")
                    trace("finished.")
                    exit()

            hyp_file = None
            if args.score:
                hyp_file = self.hyp
            else:
                trace('generating translation ...')
                generated = 0
                hyp_file = self.resultFolder+"/hyp%s.txt" % (str(model_number).zfill(3))
                fp_hype_file = open(hyp_file, 'w')

                raw_src_batch = gens.batch(gens.word_list(self.dev_src), args.minibatch)
                raw_trg_batch = gens.batch(gens.word_list(self.dev_trg), args.minibatch)
                for src_batch, trg_batch in zip(raw_src_batch, raw_trg_batch):
                    src_batch = self.fill_batch(src_batch, is_reverse=model.is_reverse)
                    K = len(src_batch)

                    hyp_batch = model.predict_greedy(src_batch, args.generation_limit, args.beamwidth)
                    generated += K
                    self.printProgressInfo(n_dev_sentence, generated)

                    for k in range(K):
                        hyp = hyp_batch[k]
                        hyp.append(EOS.s)
                        hyp = hyp[:hyp.index(EOS.s)]    # EOSタグが出るまで出力
                        if self.with_label == True:
                            hyp = removeAllLabel(hyp)

                        print(' '.join(hyp), file=fp_hype_file)
                self.clearProgressInfo()

                fp_hype_file.close()
            bleu_val = bleu.calc_bleu(hyp_file, VALID_FILE_TRG)
            ribes_val = ribes.calc_ribes(hyp_file, VALID_FILE_TRG)
            trace('BLEU :  %8f' % (bleu_val))
            with open(self.resultFolder+"/result.txt", "a") as fp:
                fp.write("%d\t%f\t%f\n" % (model_number, bleu_val, ribes_val))
            model_number += 1

    # 進捗バーを出力
    def printProgressInfo(self, end, now):
        MAX_LEN = 50
        progress = 1.0 if now == end-1 else 1.0 * now / end
        BAR_LEN = MAX_LEN if now == end-1 else int(MAX_LEN * progress)
        progressbar_str =  ('[' + '=' * BAR_LEN +
                            ('>' if BAR_LEN < MAX_LEN else '=') +
                            ' ' * (MAX_LEN - BAR_LEN) +
                            '] %.1f%% (%d/%d)' % (progress * 100., now, end))
        sys.stderr.write('\r' +  progressbar_str)
        sys.stderr.flush()

    def clearProgressInfo(self):
        sys.stderr.write('\n')

    def getLineNum(self, filepath_eu, filepath_ja):
        n_data_e = sum(1 for line in open(filepath_eu, "r"))
        n_data_j = sum(1 for line in open(filepath_ja, "r"))
        assert n_data_e == n_data_j
        assert n_data_e != 0
        return n_data_e

    def outputInfo(self, args, model, src_vocab, trg_vocab):
        output_file = open(self.resultFolder+"/result.txt", "a")
        output_file.write("##########################\n")
        output_file.write("#*****  About Corpus *****\n")
        output_file.write("# the number of sentence:\n")
        output_file.write("#\tTrain: {0}\n".format(self.getLineNum(self.train_src, self.train_trg)))
        output_file.write("#\tDev: {0}\n".format(self.getLineNum(self.dev_src, self.dev_trg)))
        output_file.write("#\tTest: {0}\n".format(self.getLineNum(self.test_src, self.test_trg)))
        output_file.write("# the vocabulary:\n")
        output_file.write("#\tsrc: {0}\n".format(len(src_vocab)))
        output_file.write("#\ttrg: {0}\n".format(len(trg_vocab)))
        output_file.write("#\t\tsrc vocab_filepath: {0}\n".format(VOCAB_SRC))
        output_file.write("#\t\ttrg vocab_filepath: {0}\n".format(VOCAB_TRG))
        output_file.write("#\n")
        output_file.write("#*****  About RNN *****\n")
        output_file.write("# vector size:\n")
        output_file.write("#\tembed: {0}\n".format(args.embed))
        output_file.write("#\thidden: {0}\n".format(args.hidden))
        output_file.write("# batchsize: {0}\n".format(args.minibatch))
        output_file.write("# beamwidth: {0}\n".format(args.beamwidth))
        output_file.write("# learning rate: {0}\n".format(args.lr))
        output_file.write("#\n")
        output_file.write("##########################\n")
        output_file.write("#\n")
        output_file.close()
