#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : perrytang
# @Time    : 2023/04/03 16:24
# @Desc    :

import requests
import torch
import sys
import math
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import concurrent.futures
import logging
from io import BytesIO

from PIL import Image

MODEL_DIR = "/data/report"
sys.path.append(MODEL_DIR)
from common_model.gpu_manager import GPUManager
from common_model.similarity_search.bulk_search import BulkSearch


class ClipModel:
    def __init__(self, max_workers=10, logger=logging):
        gm = GPUManager()
        self.gpu_index = gm.auto_choice()
        print("ClipModel cuda index ", self.gpu_index)
        self.device = f"cuda:{self.gpu_index}" if torch.cuda.is_available() else "cpu"
        model_path = "/dockerdata/gisellewang/porn/openai/clip-vit-large-patch14/"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.classify_model = CLIPModel.from_pretrained(model_path)
        self.classify_processor = CLIPProcessor.from_pretrained(
            model_path, tokenizer=tokenizer
        )
        self.classify_model = self.classify_model.to(self.device)
        self.real_label = [
            "a photo of a young woman",
            "a photo of a sex body",
            "a photo of a cat",
            "a photo of a dog",
            "a photo of landscape",
            "others",
        ]
        self.batch_size = 10
        self.real_id_2_label = dict(zip(range(len(self.real_label)), self.real_label))
        self.logger = logger
        self.logger.info("加载clip_model完成")
        self.image_download_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )

    def img_load(self, url_list):
        img_list, valid_url_list = [], []
        for url in url_list:
            response = requests.get(url, timeout=5)
            status_code = response.status_code
            if status_code in ["200", 200]:
                image = Image.open(BytesIO(response.content))
                img_list.append(image)
                valid_url_list.append(url)
            else:
                continue
        return img_list, valid_url_list

    def inference(self, image_list):
        model_rs_list = []
        iter = math.ceil(len(image_list) / self.batch_size)
        for i in range(iter):
            start = self.batch_size * i
            end = min(self.batch_size * (i + 1), len(image_list))
            batch_image_list = image_list[start:end]
            batch_model_rs = self._inference(batch_image_list)
            model_rs_list.extend(batch_model_rs)
        return model_rs_list

    def _inference(self, image_list):
        clip_rs_list = []
        if len(image_list) > 0:
            inputs = self.classify_processor(
                text=self.real_label,
                images=image_list,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self.device)
            outputs = self.classify_model(**inputs)
            logits_per_image = (
                outputs.logits_per_image
            )  # this is the image-text similarity score
            probs = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities
            probs = probs.detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            # logger.info(f"preds:{preds}")
            preds_list = preds
            for i in range(len(preds_list)):
                pred = preds_list[i]
                out = [
                    str(pred),
                    self.real_id_2_label[pred],
                    str(probs[i][pred]),
                    "|".join([str(x) for x in probs[i]]),
                ]
                # logger.info(f"out:{out}")
                clip_rs_list.append(out)
        return clip_rs_list

    @staticmethod
    def download(image_url):
        response = requests.get(image_url)
        if response.status_code == 200:
            try:
                img = Image.open(BytesIO(response.content))
                return [img], [image_url]
            except:  # pylint: disable=bare-except
                return [], []
        return [], []

    def clip_predict_url_pool(self, image_url_list):
        self.logger.info("all image urls %s" % len(image_url_list))
        futures = []
        for image_url in image_url_list:
            future = self.image_download_pool.submit(ClipModel.download, image_url)
            futures.append(future)

        # 轮询 future 对象，直到所有进程都完成
        valid_image_list, valid_image_url_list = [], []
        valid_image_size, counter = 0, 0
        return_rs = {}
        while futures:
            # 遍历所有未完成的 future 对象
            for future in concurrent.futures.as_completed(futures):
                # 如果进程已经完成，则从未完成的 future 列表中移除它
                if future.done():
                    counter += 1
                    futures.remove(future)
                    # 获取进程的输出并打印出来
                    image, valid_image_url = future.result()
                    valid_image_list.extend(image)
                    valid_image_url_list.extend(valid_image_url)
                    if len(valid_image_list) >= 10:
                        self.logger.info(
                            f"counter:{counter}, valid image:{valid_image_size}",
                        )
                        rs_list = self.inference(valid_image_list)
                        temp_return_rs = dict(zip(valid_image_url_list, rs_list))
                        return_rs.update(temp_return_rs)
                        valid_image_size += len(valid_image_list)

                        valid_image_list = []
                        valid_image_url_list = []

        if len(valid_image_list) > 0:
            valid_image_size += len(valid_image_list)
            rs_list = self.inference(valid_image_list)
            temp_return_rs = dict(zip(valid_image_url_list, rs_list))
            return_rs.update(temp_return_rs)
        self.logger.info("valid image %s" % valid_image_size)
        return return_rs


if __name__ == "__main__":
    import time

    # bulk_server = BulkSearch(date="20230305")

    clip_model = ClipModel()
    url_list = [
        "http://thirdqq.qlogo.cn/g?b=sdk&k=03hryGEgzsoRBJEDmJ7xrw&kti=ZCwzygAAAAI&s=640&t=0",
        "http://thirdqq.qlogo.cn/g?b=sdk&k=cDnR7jClaBs7bGfoY2K6Fw&kti=ZCvfjwAAAAA&s=640&t=1677904536",
        "http://thirdqq.qlogo.cn/g?b=sdk&k=NSyKcR68wq1xE1YWnwiaNQQ&kti=ZCwzzAAAAAE&s=640&t=1667354821",
        "http://vweixinthumb.tc.qq.com/150/20250/snsvideodownload?filekey=30340201010420301e020200960402534804101821f15d8f78444ce52e968b74cb81ab020225a6040d00000004627466730000000132&hy=SH&storeid=563e4a995000514b94337b1e80000009600004f1a53482b300980b65f6c252&bizid=1023",
        "http://vweixinthumb.tc.qq.com/150/20250/snsvideodownload?filekey=30340201010420301e020200960402534804100655350efeca0f6971cbf43c1e5434c802022834040d00000004627466730000000132&hy=SH&storeid=563ec9ba50007eafd4337b1e80000009600004f1a53480af8bb01e6c4d9bbb&bizid=1023",
        "http://wx.qlogo.cn/mmhead/ver_1/7udH9q2YiaKOGjgJD8lINESdZFfhqF9X15zPUia2MbUhiayamQUV7IaxshzmzMwTUm5u8pgVKnVIOwibicHjZ0468ic4uYTf5V2Th28ib2n6Ka0M8c/0",
        "http://wx.qlogo.cn/mmhead/ver_1/OGz2KxcuY6vuGnNSJ2Y63ib91lwIx9Mffd0OtwltfCTDQcbibudkHiavQVpxayNlzoFesM6yvLVcia45Xp2CccZqHS9jnAnZLP3NBvEUyjcQaYM/0",
        "http://shmmsns.qpic.cn/mmsns/7KAk8217Wsm5CHeOVajN4qCXmcibxXTuhLfjO016lDeWibs3YzREOUw4J1ScrVqfSetIsIW7rb2X8/150",
        "http://shmmsns.qpic.cn/mmsns/7KAk8217Wsm5CHeOVajN4uTeOFhDPnAhft3kvPYN56P9MuibKtJ3t3T2RxNrtD9cviamAKDydGwiaU/150",
        "http://shmmsns.qpic.cn/mmsns/7KAk8217Wsm5CHeOVajN4s0fALGibTiaGpNBVibT3ch63RbudgCE4vDrgcjGu3JOwuJ55NTM1dhqw4/150",
        "http://shmmsns.qpic.cn/mmsns/7KAk8217Wsm5CHeOVajN4okpVO1mf1iaicztWvDJyzOOJWwPKzdgMjcf5ic5gIg5z8VUEbTYYnRWy8/150",
        "http://shmmsns.qpic.cn/mmsns/ibC8nosc9924jW94clQrO0bicBibZW3icfODKvSiaXyaANS4MYRhcnR0oQMLMQVGGEEVrcHsSZ1ibNtXw/150",
        "http://thirdqq.qlogo.cn/g?b=sdk&k=oIJaGiagianuDNOU57ExpAFQ&kti=ZCvfkgAAAAI&s=640&t=1661355963",
    ]

    start = time.time()
    clip_result = clip_model.clip_predict_url_pool(url_list)
    print(clip_result)
