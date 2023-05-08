import os
import sys
import re
import time
import logging
import pandas as pd


# 引用库
from langdetect import detect

# 当文本过短或模糊时，判断出来的结果会不确定；
# 如果要让结果唯一，添加以下两行：
from langdetect import DetectorFactory
from common_model.gibberish.zh_character_trans import ZhCharacterTrans  # NOQA: E402
from common_model.gibberish.markov_train import MarkovReport  # NOQA: E402


DetectorFactory.seed = 0


class GibberishDetector:
    def __init__(self, vocab_file=None, model_file=None, logger=None) -> None:
        # 日志
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger
        # 繁体字转简体字
        self.zh_trans = ZhCharacterTrans()
        # 马尔科夫模型
        self.markov_report = MarkovReport(
            vocab_file=vocab_file, logger=logger, model_file=model_file
        )

    # 过滤数字+字母
    def filter_num_letters(self, line):
        pattern = re.compile(r"[a-zA-Z0-9_]")
        line_sub = re.sub(pattern, "", line)
        return line_sub

    # 过滤http链接
    def filter_http(self, line):
        pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        line_sub = re.sub(pattern, "", line)
        return line_sub

    def is_empty(self, line):
        if line.strip() == "":
            return True
        return False

    def is_special_square(self, line):
        pattern_list = ["�", "◆"]
        pattern = re.compile("|".join(pattern_list))
        rs = re.findall(pattern, line)
        if len(rs) > 0:
            return True
        return False

    # 非gb2312字符集字符占比
    def is_unrecognize_character_unusual(self, line):
        unrecognize_cnt = 0
        for w in line:
            try:
                w.encode("gb2312")
            except UnicodeEncodeError:
                unrecognize_cnt += 1
        if len(line) <= 15:
            return False
        pct = -1 if len(line) == 0 else unrecognize_cnt / len(line)
        if pct > 0.2:
            return True
        return False

    def is_ar(self, line):
        try:
            if detect(line) == "ar":
                return True
        except:
            return False
        return False

    def predict(self, line):
        rs = {"predict": 0, "reason": "", "prob": 0}

        # 空字符串
        empty = self.is_empty(line)
        if empty:
            rs["reason"] = "empty"
            rs["type"] = "others"
            rs["predict"] = 0
            return rs

        # 过滤http链接
        start_time = time.time()
        try:
            line_sub = self.filter_http(line)
            # 过滤后位空串
            if self.is_empty(line_sub):
                rs["reason"] = "http"
                rs["type"] = "others"
                rs["predict"] = 1
                return rs
        except:
            line_sub = line
        self.logger.info("filter http cost:%s" % (time.time() - start_time))

        # 纯数字
        if line_sub.isdigit():
            if len(line_sub) == 11:
                rs["reason"] = "11 numbers"
                rs["type"] = "others"
                rs["predict"] = 0
            rs["reason"] = "pure number"
            rs["predict"] = 1
            return rs

        # 过滤数字+字母链接
        start_time = time.time()
        try:
            line_sub = self.filter_num_letters(line_sub)
            # 过滤后位空串
            if self.is_empty(line_sub):
                rs["reason"] = "num letters"
                rs["type"] = "others"
                rs["predict"] = 1
                return rs
        except:
            pass
        self.logger.info("filter num letters cost:%s" % (time.time() - start_time))

        # 乱码square
        start_time = time.time()
        try:
            if self.is_special_square(line_sub):
                rs["reason"] = "square"
                rs["type"] = "messy code"
                rs["predict"] = 1
                return rs
        except:
            pass
        self.logger.info("filter num letters cost:%s" % (time.time() - start_time))

        # 繁体转简体
        start_time = time.time()
        line_sc = self.zh_trans.tc_2_sc(line_sub)
        self.logger.info("tc_2_sc cost:%s" % (time.time() - start_time))

        # 不可识别字符占比异常
        start_time = time.time()
        unrecognize_character_unusual = self.is_unrecognize_character_unusual(line_sc)
        self.logger.info(
            "is_unrecognize_character_unusual cost:%s" % (time.time() - start_time)
        )
        if unrecognize_character_unusual:

            # 维吾尔语检测
            start_time = time.time()
            ar = self.is_ar(line_sc)
            self.logger.info("is_ar cost:%s" % (time.time() - start_time))
            if ar:
                rs["reason"] = "ar"
                rs["type"] = "ar"
                rs["predict"] = 0
                return rs

            rs["reason"] = "unrecognize_character"
            rs["type"] = "messy code"
            rs["predict"] = 1
            return rs

        if len(line_sc) <= 20:
            rs["reason"] = "too short"
            rs["type"] = "messy code"
            rs["predict"] = 0
            return rs

        predict_rs = self.markov_report.predict(line, self.markov_report.thresh)
        if predict_rs["predict"] == 1:
            predict_rs["type"] = "messy code"
        else:
            predict_rs["type"] = "others"
        return predict_rs


if __name__ == "__main__":
    if sys.platform == "darwin":
        DATA_DIR = "/Users/xiangyuwang/Desktop/report"
        MODEL_DIR = "/Users/xiangyuwang/Desktop/report"
    elif sys.platform == "linux":
        DATA_DIR = "/dockerdata/txwsreport_automation/data/"
        MODEL_DIR = "/dockerdata/txwsreport_automation/gibberish"
    vocab_file = os.path.join(MODEL_DIR, "vocab.txt")
    model_file = os.path.join(DATA_DIR, "markov_report.pkl")
    test_file = os.path.join(MODEL_DIR, "test_20230129_20230204.txt")
    predict_file = os.path.join(MODEL_DIR, "predict_20230129_20230204.csv")

    from common.log_factory import logger

    line = "=◆◆◆)◆已报警处理◆MqI◆t◆◆走私狩猎保护动物◆◆◆◆◆◆◆ Eo◆e◆盗取他人qq号◆◆.l8◆◆x◆(:}◆◆V利用qq散播犯罪信息uowu◆◆s◆◆辱骂腾讯客服V◆◆◆◆X永久封禁◆vL>白号◆c◆◆+◆◆◆EV贩卖VPN◆4BV◆◆fT◆Ry◆;◆◆◆N辱骂腾讯客服◆C<◆◆◆o极速出结果◆◆;a◆S ◆k◆◆N◆◆极速出结果◆o◆K◆永久封禁◆◆◆辱骂习近平◆梧桐wwg拷打白号◆"
    gibberish_detector = GibberishDetector(
        vocab_file=vocab_file, logger=logger, model_file=model_file
    )
    predict_rs = gibberish_detector.predict(line)
    print(predict_rs)
    # probs = []
    # counter = 0
    # for data in open(test_file).readlines():
    #     counter += 1
    #     if counter == 1:
    #         continue
    #     field = data.strip().split("\001")
    #     line = field[2]
    #     predict_rs = gibberish_detector.predict(line)
    #     if counter % 1000 == 0:
    #         print("counter:%s" % counter)
    #     probs.append(
    #         [
    #             field[0],
    #             int(field[1]),
    #             predict_rs["prob"],
    #             predict_rs["reason"],
    #             predict_rs["predict"],
    #             field[2],
    #         ]
    #     )

    # test_df = pd.DataFrame(
    #     probs,
    #     columns=["task_id", "label", "prob", "reason", "predict", "line"],
    # )
    # test_df.to_csv(predict_file, index=False, sep="\001")
