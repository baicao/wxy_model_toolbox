import os
import sys
import json
import traceback
from urllib.parse import unquote_plus
import datetime
import dill as pickle
from common.log_factory import logger
from common_person.feature_utils import default_get_value
from common_talk.talks_parser import TalkParser
from common_talk.talks_feature import replace_line_feed


class ComplainPredictor(object):

    def __init__(self,
                 qq_pb_model_dir,
                 wx_pb_model_dir,
                 complain_model_cache_file: str) -> None:
        # pylint: disable=import-outside-toplevel
        try:
            from teen_sex_predicter import TeenSexPredicter

            self.teen_sex_model = TeenSexPredicter()
            logger.info("load teen_sex_model success")
        except:  # pylint:disable=bare-except
            logger.error("load txws model error %s", traceback.format_exc())
            raise RuntimeError("init TeenSexPredicter failed")
        try:
            from txws_predicter import TextInferenceQQ

            self.txws_qq_model = TextInferenceQQ(model_dir=qq_pb_model_dir,
                                                 logger=logger)
            self.txws_qq_model.runner.load_model()
            logger.info("load txws_qq_model success")
        except:  # pylint:disable=bare-except
            logger.error("load txws model error %s", traceback.format_exc())
            raise RuntimeError("init TextInferenceQQ failed")
        try:
            from txws_predicter import TextInferenceWX

            self.txws_wx_model = TextInferenceWX(model_dir=wx_pb_model_dir,
                                                 logger=logger)
            self.txws_wx_model.runner.load_model()
            logger.info("load txws_wx_model success")
        except:  # pylint:disable=bare-except
            logger.error("load txws model error %s", traceback.format_exc())
            raise RuntimeError("init TextInferenceWX failed")
        # pylint: enable=import-outside-toplevel
        self.complain_model_dict = {}
        self.uncached_eviuin = 0
        self.complain_model_cache_file = complain_model_cache_file
        self.load_complain_model_cache()

    def load_complain_model_cache(self):
        if os.path.exists(self.complain_model_cache_file):
            with open(self.complain_model_cache_file, "rb") as f:
                temp_dict = pickle.load(f)
                self.complain_model_dict.update(temp_dict)
                logger.info("complain_model_dict:%s",
                            len(self.complain_model_dict))

    def save_complain_model_cache(self):
        with open(self.complain_model_cache_file, "wb") as f:
            pickle.dump(self.complain_model_dict, f)

    def add_cache(self, input_account, complain_list):
        self.complain_model_dict[input_account] = complain_list
        self.uncached_eviuin += 1
        if self.uncached_eviuin > 500:
            cache_size = len(self.complain_model_dict)
            logger.info("save %s complain_model_dict success", cache_size)
            self.uncached_eviuin = 0
            # self.save_complain_model_cache()

    def effective_complain(self, complain_entry, be_reported):
        other_txt_evidence, reporter = "", ""
        if "strImpeachSrvParam" in complain_entry:
            other_txt_evidence = complain_entry["strImpeachSrvParam"]
        if "strUin" in complain_entry:
            reporter = str(complain_entry["strUin"])
        if "strEvilUin" in complain_entry and complain_entry[
            "strEvilUin"] != "":
            be_reported = str(complain_entry["strEvilUin"])

        # 没有举报对话，且没有图片
        if other_txt_evidence == "":
            return (False, ""), None, 0.0
        other_txt_evidence = replace_line_feed(other_txt_evidence)
        parse = TalkParser(
            line=other_txt_evidence,
            reporter=reporter,
            be_reported=be_reported,
        )

        rs = parse.is_valid_complain(
            check_diags=True,
            check_paragraph=True,
            check_be_reported=True,
            keep_other=False,
        )

        return rs, parse

    def get_complain_model_predict(
            self,
            complaint_info,
            account,
            account_type=None,
            dead_day=datetime.datetime.now().strftime("%Y-%m-%d 00:00:00"),
            teen_sex_model=True,
            txws_model=True,
    ):
        if account in self.complain_model_dict:
            return self.complain_model_dict[account]
        else:
            complain_list = self.__get_complain_model_predict(
                complaint_info,
                account,
                account_type,
                dead_day,
                teen_sex_model,
                txws_model,
            )
            self.add_cache(account, complain_list)
            return complain_list

    def __get_complain_model_predict(
            self,
            complaint_info,
            account,
            account_type=None,
            dead_day=datetime.datetime.now().strftime("%Y-%m-%d 00:00:00"),
            teen_sex_model=True,
            txws_model=True,
    ):
        if account_type is None:
            if account.isdigit():
                account_type = "qq"
            elif account.startswith("wxid_"):
                account_type = "wx"
            else:
                account_type = "wx"
        if complaint_info == "":
            return []
        if isinstance(complaint_info, list):
            pass
        else:
            if account_type == "qq":
                try:
                    complaint_info = json.loads(unquote_plus(complaint_info))
                    # print("complaint_info:%s" % complaint_info)
                except:  # pylint: disable=bare-except
                    print("parse qq complain error")
                    logger.error("parse qq %s complain error %s", account,
                                 traceback.format_exc())
                    return []
            elif account_type == "wx":
                try:
                    complaint_info = default_get_value(complaint_info,
                                                       "report_record")

                except:  # pylint: disable=bare-except
                    print("parse wx complain error")
                    logger.error("parse wx %s complain error %s", account,
                                 traceback.format_exc())
                    return []

        complain_list = []
        invalid_complain_list = []
        # print("complaint_info:%s" % complaint_info)
        try:
            for index, record_entry in enumerate(complaint_info):
                reporter, content, task_id, other_txt_evidence = "", "", "", ""
                if ("strImpeachSrvParam" not in record_entry
                        and "evidencetext" not in record_entry):
                    # print("strImpeachSrvParam")
                    continue

                if "add_time" not in record_entry and "reporttime" not in record_entry:
                    # print("add_time")
                    continue
                if account_type == "qq":
                    other_txt_evidence = record_entry["strImpeachSrvParam"]
                    if "strUin" not in record_entry:
                        # print("strUin")
                        continue
                    reporter = record_entry["strUin"]
                    task_id = (record_entry["id"] if "id" in record_entry
                                                     and record_entry["id"] != "" else index)
                    complaint_time = record_entry["add_time"]
                elif account_type == "wx":
                    complaint_time = datetime.datetime.fromtimestamp(
                        record_entry["reporttime"]).strftime(
                        "%Y-%m-%d 00:00:00")
                    other_txt_evidence = record_entry["evidencetext"]
                    reporter = str(record_entry["reporttime"])
                    task_id = index

                    record_entry["strImpeachSrvParam"] = record_entry[
                        "evidencetext"]
                    record_entry["id"] = index
                    record_entry["strUin"] = str(record_entry["reporttime"])
                    record_entry["add_time"] = complaint_time
                if complaint_time >= dead_day:
                    # print("complaint_time >= dead_day:", complaint_time)
                    # print("dead_day:", dead_day)
                    continue

                complain_parse = self.effective_complain(record_entry, account)
                # print("complain_parse:", complain_parse)
                if account_type is None:
                    account_type = complain_parse[1].account_type
                if not complain_parse[0][0]:
                    invalid_complain_list.append({
                        "talks":
                            other_txt_evidence,
                        "desc":
                            content,
                        "reporter":
                            reporter,
                        "be_reported":
                            account,
                        "task_id":
                            task_id,
                        "complaint_time":
                            datetime.datetime.strptime(
                                complaint_time, "%Y-%m-%d %H:%M:%S").timestamp(),
                        "valid":
                            False,
                        "valid_reason":
                            complain_parse[0][1],
                    })
                    # print("invalid_complain_list:%s" % invalid_complain_list)
                    continue
                complain_parse = complain_parse[1]

                complain_list.append({
                    "talks":
                        other_txt_evidence,
                    "desc":
                        content,
                    "reporter":
                        reporter,
                    "be_reported":
                        account,
                    "task_id":
                        task_id,
                    "complain_diags":
                        complain_parse.diags,
                    "complain_paragraph":
                        complain_parse.paragraph,
                    "complaint_time":
                        datetime.datetime.strptime(
                            complaint_time, "%Y-%m-%d %H:%M:%S").timestamp(),
                    "valid":
                        True,
                })

            # 儿色模型预测
            if teen_sex_model:
                self.teen_sex_model.predict_bulk(complain_list)

            # 腾讯卫士模型预测
            if txws_model:
                eval_datas = [
                    "|".join([y.WORDS for y in x["complain_diags"]])
                    for x in complain_list
                ]
                if len(eval_datas) > 0:
                    if account_type == "qq":
                        txws_model_rs = self.txws_qq_model.inference(
                            eval_text_list=eval_datas,
                            return_sentence_attention=False,
                            return_word_attention=False,
                        )

                    elif account_type == "wx":

                        txws_model_rs = self.txws_wx_model.inference(
                            eval_text_list=eval_datas,
                            return_sentence_attention=False,
                            return_word_attention=False,
                        )
                    for i, _ in enumerate(complain_list):
                        complain_list[i]["txws_prob"] = txws_model_rs[0][i]
                        complain_list[i]["txws_type"] = txws_model_rs[1][i]

        except:  # pylint: disable=bare-except
            logger.error("%s socialReportComplaintInfo error %s", account,
                         traceback.format_exc())

        complain_list.extend(invalid_complain_list)
        return complain_list


if __name__ == "__main__":
    from platform import system

    if system() == "Darwin":
        ROOT_DIR = "/Users/xiangyuwang/Desktop/MacBookProHome/clean"
        QQ_PB_MODEL_DIR = (
            "/Users/xiangyuwang/Software/txws_text_classification/saved_model/qq"
        )
        WX_PB_MODEL_DIR = (
            "/Users/xiangyuwang/Software/txws_text_classification/saved_model/wx"
        )
        complain_model_cache_file = os.path.join(
            ROOT_DIR, "complain_model_dict_cache.pkl")
    elif system() == "Linux":
        ROOT_DIR = "/dockerdata/gisellewang/natural_person/ziranren_teen_sex/data"
        QQ_PB_MODEL_DIR = "/dockerdata/sunyyao/tutorial/tf_model/saved_model/qq"
        WX_PB_MODEL_DIR = "/dockerdata/sunyyao/tutorial/tf_model/saved_model/wx"
        complain_model_cache_file = os.path.join(
            ROOT_DIR, "complain_model_dict_cache.pkl")
    else:
        sys.exit()

    teen_sex_extrator = ComplainPredictor(QQ_PB_MODEL_DIR, WX_PB_MODEL_DIR,
                                          complain_model_cache_file)
