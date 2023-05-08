import os
import sys
import re
import time
import json
import pickle
import traceback
import logging
from urllib.parse import unquote
import datetime
import faiss
import numpy as np
from collections import Counter

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.abspath(os.path.dirname(FILE_PATH) + os.path.sep + ".")
sys.path.append(FILE_DIR + "/../")  # 设置引入模块的根目录
from common.parse_server import CONFIG  # NOQA: E402
from common.ces_template import CESTemplate  # NOQA: E402
from common_model.similarity_search.clip_chinese_emd import CLIPChinese


"""
计算图文相似度，以及文本相似度的脚本
"""


class BulkSearch:
    def __init__(self, date=None, data_dir=None, logger=None) -> None:

        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.pkg_dir = os.path.dirname(os.path.abspath(__file__))
            pkg_name = "data"
            self.data_dir = os.path.join(self.pkg_dir, pkg_name)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if date is None:
            self.date = datetime.datetime.now()
        else:
            self.date = datetime.datetime.strptime(date, "%Y%m%d")

        self.clip_model = CLIPChinese(logger=logger)

        self.search_days = 7
        # faiss index 参数
        self.dim = 512
        self.measure = faiss.METRIC_INNER_PRODUCT
        self.param = "Flat"
        self.topK = 10
        self.batch_size = 100

    def generate_emd(self, start_datetime, emd_file):
        end_datetime = start_datetime + datetime.timedelta(days=1)
        self.logger.info(f"export data from {start_datetime} to {end_datetime}")
        desc_list, task_id_list = self.export_data(start_datetime, end_datetime)
        self.logger.info(f"export data size {len(desc_list)}")
        self.logger.info("draw embeddings")
        emds_list = self.clip_model.predict(desc_list)
        self.logger.info(f"emds_list:{emds_list.shape}")
        self.logger.info(f"desc_list:{len(desc_list)}")
        self.logger.info(f"task_id_list:{len(task_id_list)}")
        # 数据保存
        with open(emd_file, "wb") as writer:
            data = {
                "emd": emds_list,
                "task_id": task_id_list,
                "data_list": desc_list,
            }
            pickle.dump(data, writer)
        return emds_list, task_id_list, desc_list

    def build_index(self):
        index_date_list, embeddings, task_id_list, data_list = [], [], [], []
        for i in range(1, self.search_days + 1):
            start_datetime = self.date - datetime.timedelta(days=i)
            self.logger.info(f"load {start_datetime} data")
            index_date_str = start_datetime.strftime("%Y%m%d")
            index_date_list.append(index_date_str)
            task_id, emds_list, desc_list = [], [], []
            emd_file = os.path.join(self.data_dir, f"emb_{index_date_str}.pkl")
            if os.path.exists(emd_file):
                self.logger.info("load data from emd_file %s" % (emd_file))
                try:
                    with open(emd_file, "rb") as file:
                        emd_dict = pickle.load(file)
                        emds_list = emd_dict["emd"]
                        task_id = emd_dict["task_id"]
                        desc_list = emd_dict["data_list"]

                except:
                    self.logger.error("load from file error", traceback.format_exc())
                    self.logger.info("generate embedding")
                    emds_list, task_id, desc_list = self.generate_emd(
                        start_datetime, emd_file
                    )
            else:
                self.logger.info("not find emd_file %s" % (emd_file))
                self.logger.info("generate embedding")
                emds_list, task_id, desc_list = self.generate_emd(
                    start_datetime, emd_file
                )

            try:
                assert len(task_id) == len(emds_list) == len(desc_list)
                self.logger.info(f"load {start_datetime} success size:{len(task_id)}")
            except:
                self.logger.error(
                    f"data size not match, regenrate index, task_id_list:{len(task_id)}, embeddings:{len(emds_list)}, data_list:{len(desc_list)}"
                )
                emds_list, task_id, desc_list = self.generate_emd(
                    start_datetime, emd_file
                )

            embeddings.extend(emds_list)
            task_id_list.extend(task_id)
            data_list.extend(desc_list)

        self.logger.info(f"all embeddings {len(embeddings)}")
        self.logger.info(f"all task_id_list {len(task_id_list)}")
        self.logger.info(f"all data_list {len(data_list)}")

        self.task_id_list = task_id_list
        self.data_list = data_list
        self.logger.info(f"start build index from {index_date_list}")
        start = time.time()

        embeddings = np.array(embeddings)
        self.faiss_index = faiss.index_factory(self.dim, self.param, self.measure)
        self.faiss_index.add(embeddings)  # 将向量库中的向量加入到index中
        self.logger.info(f"build index cost:{time.time()-start}")
        self.logger.info(self.faiss_index.ntotal)  # 输出index中包含的向量总数，

    def export_data(self, start_datetime, end_datetime):
        task_fields = [
            "task_id",
            "username",
            "report_content",
            "add_time",
            "reply_content",
            "report_type2",
            "report_result",
            "report_username_list",
            "rt1",
            "rt2",
        ]

        tag_config = CONFIG["ES_WX_SOCIAL_REPORT_DEAL"]
        tag_config["index"] = "kfreport_report_info"
        ces_template = CESTemplate(**tag_config)
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "add_time": {
                                    "gte": start_datetime.strftime("%Y-%m-%d 00:00:00"),
                                    "lte": end_datetime.strftime("%Y-%m-%d 00:00:00"),
                                }
                            }
                        },
                    ]
                }
            },
            "_source": task_fields,
        }
        self.logger.info("query:%s" % query_body)
        data_list, total_size = ces_template.search(query_body, size=2000)
        self.logger.info("total_size:%s" % total_size)
        desc_list, task_id_list = [], []
        for data in data_list:
            source = data["_source"]
            if "report_content" not in source:
                continue
            if "task_id" not in source:
                continue
            task_id = source["task_id"]
            report_content = source["report_content"]

            if report_content.strip() == "":
                continue
            try:
                report_content = unquote(report_content)
                report_content = json.loads(report_content)
                if "report_content" not in report_content:
                    continue
                report_content = report_content["report_content"]
                report_content_temp = report_content[:510]
                report_content_temp = "".join(
                    re.findall(r"[0-9a-zA-Z\u4e00-\u9fa5\s。，]", report_content_temp)
                )
                if report_content_temp.strip() == "":
                    continue
                if len(report_content_temp) < 30:
                    continue
                if len(report_content_temp) >= 510:
                    report_content_temp = report_content_temp[:510]
                task_id_list.append(task_id)
                desc_list.append(report_content_temp)
            except:
                self.logger.error(task_id, traceback.format_exc())
                continue
        return desc_list, task_id_list

    @staticmethod
    def deal_line(line):
        line = line.replace("\r", "")
        line = line.replace("\n", "")
        line = line.replace("\r\n", "")
        return line

    def predict(self, desc_list, task_id_list):
        self.logger.info(
            f"desc_list:{len(desc_list)}, task_id_list:{len(task_id_list)}"
        )
        iter = int(len(desc_list) / self.batch_size) + 1
        result = []
        for i in range(iter):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(desc_list))
            batch_desc_list = desc_list[start_idx:end_idx]
            batch_desc_list = [x[:512] if len(x) > 512 else x for x in batch_desc_list]
            try:
                emds_list = self.clip_model.predict(batch_desc_list)
            except:
                self.logger.error("predict error " + str(batch_desc_list))
                continue

            D, I = self.faiss_index.search(emds_list, self.topK)
            D = np.array(D)
            x, y = np.where(D > 0.9)
            counter = Counter(x)

            for match_index in counter:
                if counter[match_index] < 10:
                    continue
                count = counter[match_index]
                if count < 10:
                    continue

                content = desc_list[start_idx + match_index]
                task_id = task_id_list[start_idx + match_index]

                related_task_id = ",".join(
                    [self.task_id_list[id] for id in I[match_index][:count]]
                )

                predict_rs = {
                    "predict": 1,
                    "reason": related_task_id,
                    "task_id": task_id,
                    "content": content,
                }
                result.append(predict_rs)
        return result


if __name__ == "__main__":
    from common.log_factory import logger

    bulk_search_server = BulkSearch(date="20230305", logger=logger)
    bulk_search_server.build_index()
    desc_list = [
        "iiltltili00C母e",
        "以上用户在QQ群威胁他人，利用QQ无违规封号恶意举报他人。多次威胁并勒索钱财，QQ无违规严重违规，并贩卖QQ举报接口，已经危害许多人，以上账号QQ举报多次无效果，希望官方可以严查，严重违规理应永久封号",
    ]
    bulk_search_server.predict(desc_list, task_id_list=["1", "2"])
