import os
import json
import sys
import re
import jieba
import math
import pickle
import logging
import traceback
import pandas as pd
import numpy as np
from common.classification_report import ClassificationReport  # NOQA: E402
from sklearn.metrics import fbeta_score
from common.log_factory import logger


class MarkovReport:
    def __init__(
        self, vocab_file=None, vocab_max_size=10000, logger=None, model_file=None
    ) -> None:
        # 日志
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

        model_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(model_dir, "saved_mode")
        # 字典
        if vocab_file is not None:
            self.vocab_file = vocab_file
        else:
            self.vocab_file = os.path.join(data_dir, "vocab.txt")
        self.vocab_max_size = vocab_max_size
        if os.path.exists(self.vocab_file):
            self.logger.info("字段加载")
            self.load_dict()
        else:
            self.vocabulary = None

        # 模型文件
        if not os.path.exists("saved_mode"):
            os.mkdir("saved_mode")
        if model_file is not None:
            self.model_file = model_file
        else:
            self.model_file = os.path.join(data_dir, "markov_report.pkl")
        if os.path.exists(self.model_file):
            self.logger.info("模型加载")
            self.load_model(self.model_file)

    def load_dict(self):
        reader = open(self.vocab_file, encoding="utf-8")
        line = [x.split(",") for x in reader.readlines()]
        self.vocabulary = dict([(x[1], int(x[0])) for x in line])
        self.vocab_size = len(self.vocabulary)
        self.logger.info("vocab is loaded, vocab_size %s", self.vocab_size)

    def load_model(self, model_file):
        # 加载模型
        self.logger.info("start load model")
        reader = open(model_file, "rb")
        model = pickle.load(reader)
        self.log_prob_mat = model["mat"]
        self.vocab_size = len(self.log_prob_mat)
        self.thresh = model["thresh"]
        self.min_prob_value = np.min(np.array(self.log_prob_mat))

        # 加载字典
        reader = open(self.vocab_file, encoding="utf-8")
        lines = [x.split(",") for x in reader.readlines()]
        lines = lines[: self.vocab_size]
        self.vocabulary = dict([(x[1], int(x[0])) for x in lines])

        self.logger.info("success load model")
        self.logger.info("thresh:%s" % self.thresh)
        self.logger.info("vocab_size:%s" % self.vocab_size)

    @staticmethod
    def normalize(line):
        rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5，。！；]")
        line = rule.sub("", line)
        return line

    def clean_data(self, line):
        line = MarkovReport.normalize(line)
        cut_list = jieba.lcut(line)
        cut_list = [x for x in cut_list if x in self.vocabulary]
        if len(cut_list) <= 2:
            return []
        cut_set = set(cut_list)
        new_line = "".join(cut_set)
        new_line = re.sub(r"[a-zA-Z0-9，。！\s\r\n；]", "", new_line)
        if len(new_line.strip()) <= 3:
            return []

        return cut_list

    def ngram(self, n, words_list):
        for start in range(0, len(words_list) - n + 1):
            yield words_list[start : start + n]

    def generate_dict(self, train_file):
        reader = open(file=train_file, encoding="utf-8")
        vocabulary = {}
        while True:
            line = reader.readline().strip()
            if not line:
                break
            normalized_line = MarkovReport.normalize(line)
            cut_list = jieba.lcut(normalized_line)
            for w in cut_list:
                if w not in vocabulary:
                    vocabulary[w] = 0
                vocabulary[w] += 1
        if len(vocabulary) < self.vocab_max_size:
            line = "\r\n".join(
                [
                    f"{index},{key},{vocabulary[key]}"
                    for index, key in enumerate(vocabulary)
                ]
            )
            self.vocab_size = len(vocabulary)
        else:
            vocab_items = vocabulary.items()
            vocab_items = sorted(vocab_items, key=lambda x: x[1], reverse=True)
            vocab_items = vocab_items[: self.vocab_max_size]
            self.vocabulary = dict(vocab_items)
            self.vocab_size = self.vocab_max_size
            line = "\r\n".join(
                [f"{index},{x[0]},{x[1]}" for index, x in enumerate(vocab_items)]
            )
        writer = open(self.vocab_file, "w", encoding="utf-8")
        writer.write(line)
        writer.close()
        self.logger.info("vocab is generated")
        self.load_dict()

    def train(self, train_file, construct_dict=False):
        # 重构字典
        if construct_dict or not os.path.exists(self.vocab_file):
            self.logger.info("generate dict")
            self.generate_dict(train_file)
        k = self.vocab_size
        self.logger.info("vocab_size %s", self.vocab_size)
        self.log_prob_mat = [[10 for i in range(k)] for i in range(k)]

        for data in open(train_file).readlines():
            field = data.strip().split("\001")
            line = field[2]
            cut_list = self.clean_data(line)
            if len(cut_list) == 0:
                continue
            for entry in self.ngram(2, cut_list):
                word1 = entry[0]
                word2 = entry[1]
                word1_idx = self.vocabulary[word1]
                word2_idx = self.vocabulary[word2]
                self.log_prob_mat[word1_idx][word2_idx] += 1

        for i, row in enumerate(self.log_prob_mat):
            s = float(sum(row))
            for j in range(len(row)):
                row[j] = math.log(row[j] / s)

        self.min_prob_value = np.min(np.array(markov_report.log_prob_mat))

    def val(self, val_file, model_file=None):
        if model_file is not None:
            self.load_model(model_file)

        val_probs = self.cal_test_file(val_file)
        val_df = pd.DataFrame(
            val_probs,
            columns=["task_id", "label", "prob", "cut/len", "cut_size", "len", "line"],
        )
        bad_df = val_df[val_df["label"] == 1]
        self.logger.info(bad_df.describe())

        true_label = val_df["label"].values
        cr = ClassificationReport(labels=["非乱码", "乱码"], logger=self.logger)
        thresh_list = []
        for i in range(5, 100, 5):
            pct = i / 100.0
            thresh = bad_df["prob"].quantile(i / 100)
            self.logger.info(f"pct:{pct}, thresh:{thresh}")
            predict_label = [1 if x < thresh else 0 for x in val_df["prob"].values]
            cr.show_cm_report(y_true=true_label, y_pred=predict_label)
            f1 = fbeta_score(y_true=true_label, y_pred=predict_label, beta=0.5)
            thresh_list.append([thresh, pct, f1])
        thresh_list = sorted(thresh_list, key=lambda x: x[2], reverse=True)
        best_model = thresh_list[0]
        self.logger.info(
            "thresh:%s, pct:%s, f1:%s", best_model[0], best_model[1], best_model[2]
        )
        pickle.dump(
            {"mat": self.log_prob_mat, "thresh": best_model[0]},
            open(self.model_file, "wb"),
        )

    def test(self, test_file, model_file=None):
        if model_file is not None:
            self.load_model(model_file)

        probs = []
        for data in open(test_file).readlines():
            field = data.strip().split("\001")
            line = field[2]
            predict_rs = self.predict(line, self.thresh)

            probs.append(
                [
                    field[0],
                    int(field[1]),
                    predict_rs["prob"],
                    predict_rs["reason"],
                    predict_rs["predict"],
                    field[2],
                ]
            )

        test_df = pd.DataFrame(
            probs,
            columns=["task_id", "label", "prob", "reason", "predict", "line"],
        )

        cr = ClassificationReport(labels=["非乱码", "乱码"], logger=logger)
        cr.show_cm_report(
            y_true=test_df["label"].values, y_pred=test_df["predict"].values
        )
        test_df.to_excel("test.xlsx")
        return test_df

    def predict(self, line, thresh):
        rs = {"predict": 0, "reason": "", "prob": 0}
        norm_line = self.normalize(line)
        cut_list = jieba.lcut(norm_line)
        cut_list = [x for x in cut_list if x in self.vocabulary]

        if len(cut_list) <= 2:
            if len(line) >= 10:
                rs["predict"] = 1
                rs["reason"] = "after cut is far more less"
                return rs
            else:
                rs["reason"] = "after cut is far more less"
                return rs
        cut_set = set(cut_list)
        new_line = "".join(cut_set)
        new_line = re.sub(r"[a-zA-Z0-9，。！\s\r\n；]", "", new_line)
        if len(new_line.strip()) <= 3:
            if len(line) >= 10:
                rs["predict"] = 1
                rs["reason"] = "after filter is far more less"
                return rs
            else:
                rs["reason"] = "after filter is far more less"
                return rs

        prob = self.avg_transition_prob(cut_list, len(line))
        rs["reason"] = "model"
        rs["prob"] = prob
        if prob < thresh:
            rs["predict"] = 1
            return rs
        else:
            rs["predict"] = 0
            return rs

    def cal_test_file(self, test_file):
        probs = []
        for data in open(test_file).readlines():
            field = data.strip().split("\001")
            line = field[2]
            cut_list = self.clean_data(line)
            if len(cut_list) == 0:
                continue
            prob = self.avg_transition_prob(cut_list, len(line))
            cut_size = len("".join(cut_list))
            pct = cut_size / len(line)
            probs.append(
                [field[0], int(field[1]), prob, pct, cut_size, len(line), field[2]]
            )
        return probs

    """ Return the average transition prob from l through log_prob_mat. """

    def avg_transition_prob(self, cut_list, size):
        log_prob = 0.0
        transition_ct = 0
        all_size = len("".join(cut_list))
        for entry in self.ngram(2, cut_list):
            word1 = entry[0]
            word2 = entry[1]
            try:
                word1_idx = self.vocabulary[word1]
                word2_idx = self.vocabulary[word2]
                log_prob += self.log_prob_mat[word1_idx][word2_idx]
            except Exception as e:
                logger.error(f"word1:{word1}, word2:{word2}")
                logger.error(traceback.format_exc())
                raise e
            transition_ct += 1
        if size * 0.84 < all_size:
            short_size = 0
        else:
            short_size = size * 0.8 - all_size
        log_prob += self.min_prob_value * short_size

        # The exponentiation translates from log probs to probs.
        return math.exp(log_prob / (short_size + transition_ct or 1))

    def test_single_line(self, line):
        cut_list = self.clean_data(line)
        if len(cut_list) == 0:
            return
        if len(set(cut_list)) == 1 and cut_list[0].strip() == "":
            return
        for entry in self.ngram(2, cut_list):
            word1 = entry[0]
            word2 = entry[1]
            word1_idx = self.vocabulary[word1]
            word2_idx = self.vocabulary[word2]
            log_prob = self.log_prob_mat[word1_idx][word2_idx]
            print(f"log_prob:{log_prob}, word1:{word1}, word2:{word2}")


def write_file(data_list, write_file):
    writer = open(write_file, "w", encoding="utf-8")
    line = "\r\n".join(["\001".join([str(y) for y in x]) for x in data_list])
    writer.write(line)
    writer.close()


def parse_file(report_file):
    reader = open(report_file, encoding="utf-8")
    header = reader.readline().strip().split("\001")
    data_list = []
    decode_error = 0
    while True:
        try:
            line = reader.readline().strip()
            if not line:
                line = reader.readline().strip()
                if not line:
                    break
            field = line.split("\001")
            field_dict = dict(zip(header, field))
            if "task_id" not in field_dict:
                continue
            task_id = field_dict["task_id"]
            report_content = field_dict["report_content"]
            report_content = json.loads(report_content)
        except:
            decode_error += 1
            continue

        desc = report_content["report_content"]
        desc = desc.replace("\r", "。")
        desc = desc.replace("\n", "。")
        desc = desc.replace("\r\n", "。")
        report_type2 = field_dict["report_type2"]
        if report_type2 == "乱码举报":
            label = 1
        else:
            label = 0

        data_list.append([task_id, label, desc])
    logger.info("decode_error: %s", decode_error)
    logger.info("data is loaded")
    return data_list


def generate_train_val(report_file):
    data_list = parse_file(report_file)
    positive_data_list = [x for x in data_list if x[1] == 0]
    test_negative_data_list = [x for x in data_list if x[1] == 1]
    pct = 0.1
    test_positive_size = int(len(data_list) * pct)
    test_positive_data_list = positive_data_list[-test_positive_size:]
    train_positive_data_list = positive_data_list[:-test_positive_size]
    negative_size = int(len(test_negative_data_list) * pct)
    test_negative_data_list = test_negative_data_list[:negative_size]
    val_data_list = test_positive_data_list + test_negative_data_list

    write_file(train_positive_data_list, train_file)
    write_file(val_data_list, val_file)


def generate_test(report_file):
    data_list = parse_file(report_file)
    write_file(data_list, test_file)


if __name__ == "__main__":
    if sys.platform == "darwin":
        DATA_DIR = "/Users/xiangyuwang/Desktop/report"
        MODEL_DIR = "/Users/xiangyuwang/Desktop/report"
    elif sys.platform == "linux":
        DATA_DIR = "/dockerdata/txwsreport_automation/data/"
        MODEL_DIR = "/dockerdata/txwsreport_automation/gibberish"
    report_file = os.path.join(DATA_DIR, "kfreport_report_info_0714_0108_data.csv")
    train_file = os.path.join(MODEL_DIR, "train.txt")
    val_file = os.path.join(MODEL_DIR, "val.txt")
    vocab_file = os.path.join(MODEL_DIR, "vocab.txt")
    model_file = os.path.join(DATA_DIR, "markov_report.pkl")

    # generate_train_val(report_file)

    markov_report = MarkovReport(
        vocab_file=vocab_file, vocab_max_size=10000, logger=logger
    )
    # markov_report.train(train_file, construct_dict=False)
    # markov_report.val(val_file)

    test_report_file = os.path.join(DATA_DIR, "kfreport_report_0122_0128_data.csv")
    test_file = os.path.join(MODEL_DIR, "test.txt")
    generate_test(test_report_file)
    markov_report.test(test_file, model_file)

    probs = []
    for data in open(test_file).readlines():
        field = data.strip().split("\001")
        line = field[2]
        predict_rs = markov_report.predict(line, markov_report.thresh)

        probs.append(
            [
                field[0],
                int(field[1]),
                predict_rs["prob"],
                predict_rs["reason"],
                predict_rs["predict"],
                field[2],
            ]
        )

    test_df = pd.DataFrame(
        probs,
        columns=["task_id", "label", "prob", "reason", "predict", "line"],
    )

    cr = ClassificationReport(labels=["非乱码", "乱码"], logger=logger)
    cr.show_cm_report(y_true=test_df["label"].values, y_pred=test_df["predict"].values)
    test_df.to_csv("test.csv")

    test_df[(test_df.predict == 1) & (test_df.label == 0)]
