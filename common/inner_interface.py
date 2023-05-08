#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : kerwinnli
# @Time    : 2023/2/27 16:44
# @Desc    : 内部接口工具类，例如恶意度接口、黑词接口
import json
import sys
import traceback
import uuid
import aiohttp
import asyncio
import logging

import requests

MODEL_DIR = "/data/report"
sys.path.append(MODEL_DIR)
from common.ces_template import CESTemplate
from common.parse_server import CONFIG
from common_talk.talks_parser import TalkParser, ROLE_TYPE
from util.data_format_util import load_json
from config.config import (
    EVIL_URL,
    SERVER_URL,
    OCR_LAHEI_URL,
    REBOOT_HOOK_URL,
    UNMATCHED_TYPE,
)
from urllib.parse import unquote
import logging

HEADERS = {"Content-Type": "application/json"}


def alarm(msg):
    data = {"msgtype": "text", "text": {"content": msg}}
    requests.post(url=REBOOT_HOOK_URL, headers=HEADERS, data=json.dumps(data))

def alarm_url(msg:str, url: str, message_type: str):
    data = {"msgtype": message_type, message_type: {"content": msg}}
    requests.post(url=url, headers=HEADERS, data=json.dumps(data))

def check_report_porn_or_not(report_info, logger=logging):
    offend_action = ""
    porn_match_words = ""
    if "offend_action" in report_info:
        offend_action = report_info["offend_action"]
    if offend_action != "色情":
        report_content = ""
        if "report_content" not in report_info:
            return []
        report_content = report_info["report_content"]
        report_content = unquote(report_content)
        report_content = json.loads(report_content)
        if "report_content" not in report_content:
            return []
        model_list = ["320005"]
        report_content = report_content["report_content"]
        report_content = report_content.replace("\r", "")
        report_content = report_content.replace("\n", "")
        report_content = report_content.replace("\r\n", "")
        execute_evil_result_dict = execute_evil(
            report_info=report_content, model_list=model_list, logger=logger
        )
        logger.info("execute_evil_result_dict", execute_evil_result_dict)
        model_code = execute_evil_result_dict["code"]
        if model_code != "100":
            return []
        data = execute_evil_result_dict["data"]
        if len(data) == 0:
            return []
        data = data[0]
        if "hit_info_list" not in data and len(data["hit_info_list"]) == 0:
            return []
        hit_info_list = data["hit_info_list"]
        for hit_entry in hit_info_list:
            if "model_id" not in hit_entry:
                continue
            model_id = hit_entry["model_id"]
            if model_id == "320005":
                porn_match_words = hit_entry["model_keywords"]

    if offend_action.find("色情") != -1 or porn_match_words != "":
        return True, offend_action, porn_match_words
    else:
        return False, offend_action, porn_match_words


"""
    跑黑词库接口
"""


def execute_black_word(report_info: str, dimension: str, uniqueKey: str):
    try:
        black_word_param = {
            "dimension": dimension,
            "sentence": report_info,
            "uniqueKey": uniqueKey,
        }
        return requests.post(
            url=BLACK_WORD_URL, headers=HEADERS, data=json.dumps([black_word_param])
        ).json()
    except Exception as e:
        alarm("黑词接口异常: " + traceback.format_exc())
        traceback.print_exc()
    return None


def judge_black_word(resp: dict, uniqueKey: str) -> int:
    if resp is not None and "code" in resp and "data" in resp:
        code = resp["code"]
        data = resp["data"]
        if code == 100 and uniqueKey in data:
            matched_list = data[uniqueKey]
            result = 1
            if matched_list is not None and len(matched_list) > 0:
                for matched_dict in matched_list:
                    if "strategy" in matched_dict and "match" in matched_dict:
                        match = matched_dict["match"]
                        strategy = matched_dict["strategy"]
                        if match and result != -1:
                            if strategy == UNMATCHED_TYPE:
                                result = -1
    return -1


"""
    跑恶意度接口
"""


def execute_evil_bulk(reporpost_data: list, logger=logging) -> dict:
    try:
        request_id = str(uuid.uuid4())
        post_data = {"word_list": reporpost_data, "full_rs": 1}
        logger.info(f"[{request_id}] REQUEST {reporpost_data}")
        response = requests.post(
            url=EVIL_URL, data=json.dumps(post_data), headers=HEADERS
        )
        logger.info(f"[{request_id}] RESPONSE {response.content}")
        return response.json()

    except:
        alarm("恶意度接口异常: " + traceback.format_exc())
        logger.error(f"[{request_id}] ERROR {traceback.format_exc()}")
    return {}


def execute_evil(report_info: str, model_list: list, logger=logging) -> dict:
    try:
        request_id = str(uuid.uuid4())
        request_param = {
            "word_list": [{"id": request_id, "content": report_info}],
            "model_list": model_list,
        }
        logger.info(f"[{request_id}] REQUEST {request_param}")
        response = requests.post(
            url=EVIL_URL, data=json.dumps(request_param), headers=HEADERS
        )
        logger.info(f"[{request_id}] RESPONSE {response.content}")
        return response.json()

    except:
        alarm("恶意度接口异常: " + traceback.format_exc())
        logger.error(f"[{request_id}] ERROR {traceback.format_exc()}")
    return {}


def execute_evil_diag(
    str_uin: str, str_evil_uin: str, report_info: str, model_list: list, logger=logging
) -> dict:
    try:
        request_id = str(uuid.uuid4())
        diags = data_diags(str_uin, str_evil_uin, report_info)
        print(f"diags:{diags}")
        request_param = {
            "word_list": [{"id": request_id, "content": diags, "diag_tag": 1}],
            "model_list": model_list,
            "diag_tag": 1,
        }
        logger.info(f"[{request_id}] REQUEST {request_param}")
        response = requests.post(
            url=EVIL_URL, data=json.dumps(request_param), headers=HEADERS
        )
        logger.info(f"[{request_id}] RESPONSE {response.content}")
        return response.json()

    except:
        alarm("恶意度接口异常: " + traceback.format_exc())
        logger.error(f"[{request_id}] ERROR {traceback.format_exc()}")
    return {}


def data_diags(str_uin, str_evil_uin, str_impeach_srv_param):
    task_info = [
        str_impeach_srv_param,
        str_uin,
        str_evil_uin,
    ]
    parse = TalkParser(*task_info)
    diags = parse.line_2_diags(keep_other=True)
    return diags


def judge_evil_result(resp: dict) -> list:
    hit_model_id_list = []
    if resp is not None and len(resp) > 0 and "data" in resp:
        data = resp["data"]
        for data_info in data:
            if "rs" in data_info and "hit_info_list" in data_info:
                rs = data_info["rs"]
                hit_info_list = data_info["hit_info_list"]
                if rs == 1:
                    for hit_info in hit_info_list:
                        if "model_id" in hit_info:
                            model_id = hit_info["model_id"]
                            hit_model_id_list.append(model_id)
    return hit_model_id_list


def query_crm_label(account: str, label_id: str) -> list:
    query_body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"account": {"value": account}}},
                    {"term": {"label_id": {"value": label_id}}},
                ]
            }
        }
    }
    result_list = []
    interface_template = CESTemplate(**CONFIG["CRM_LABEL_INDEX_CONFIG"])
    try:
        data_list, size = interface_template.search(body=query_body, size=1000)
        if data_list is not None and len(data_list) > 0:
            for data in data_list:
                source = data["_source"]
                result_list.append(source)
    except Exception:
        pass
    return result_list


async def ocr_coroutine(session, img_request):
    # print(f"开始读取图片ocr:{img_request}")
    async with session.post(url=SERVER_URL, json=img_request) as res:
        # print(f"res:{res}")
        content = await res.content.read()
        content_str = content.decode("utf-8")
        content_json = load_json(content_str)
        # print(f"content_json:{content_json}")
        return content_json


async def async_ocr(input_img_list):
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(ocr_coroutine(session, img_request))
            for img_request in input_img_list
        ]
        content = await asyncio.gather(*tasks)
        # content = await asyncio.wait(tasks)
    return content


def ocr_sync(data: str):
    try:
        return requests.post(url=SERVER_URL, data=data).json()
    except Exception:
        alarm("ocr接口异常: " + traceback.format_exc())
    return None


def ocr_lahei(data: str):
    try:
        return requests.post(url=OCR_LAHEI_URL, data=data).json()
    except Exception:
        alarm("ocr接口异常: " + traceback.format_exc())
    return None


if __name__ == "__main__":
    str_uin = "1966346565"
    str_evil_uin = "1472231279"
    report_info = 'chatmsg:[uin=1472231279;content=每天退化三次的恐龙，人类历史上最强的废材;type=1][uin=1472231279;content=像你这种可恶的家伙只能演电视剧里的一陀粪;type=1][uin=1472231279;content=你是进化不完全的生命体，基因突变的外星人;type=1][uin=1472231279;content=连如花都帅你10倍以上;type=1][uin=1472231279;content=10倍石油浓度的沉积原料，被毁容的麦当劳叔叔;type=1][uin=1472231279;content=想要自杀只会有人劝你不要留下尸体以免污染环境;type=1][uin=1472231279;content=比不上路边被狗洒过尿的口香糖;type=1][uin=1472231279;content=喷出来的口水比SARS还致命;type=1][uin=1472231279;content=找女朋友得去动物园甚至要离开地球;type=1][uin=1472231279;content=耍酷装帅的话人类就只得用无性生殖;type=1][uin=1472231279;content=你摸过的键盘上连阿米吧原虫都活不下去;type=1][uin=1472231279;content=只要你抬头臭氧层就会破洞;type=1][uin=1472231279;content=装可爱的话可以瞬间解决人口膨胀的问题;type=1][uin=1472231279;content=如果你的丑陋可以发电的话全世界的核电厂都可以停摆;type=1][uin=1472231279;content=白痴可以当你的老师，智障都可以教你说人话;type=1][uin=1472231279;content=手榴弹看到你会自爆;type=1][uin=1472231279;content=要移民火星是为了要离开你;type=1][uin=1472231279;content=你去过的名胜全部变古迹，你去过的古迹会变成历史;type=1][uin=1472231279;content=去打仗的话子弹飞弹会忍不住向你飞;type=1][uin=1472231279;content=反正横竖一句话：别让我再看见你，要是见着了你;type=1][uin=1472231279;content=别人要开飞机去撞双子星才行而你只要跳伞就有同样的威力;type=1][uin=1472231279;content=长的惊险.....有创意啊;type=1][uin=1472231279;content=18辈子都没干好事才会认识你，连丢进太阳都嫌不够环保;type=1][uin=1472231279;content=我一定要把你灭了！;type=1][uin=1472231279;content=你小时候被猪亲过吧?;type=1][uin=1472231279;content=你长的很 爱国 很敬业 很有骨气;type=1][uin=1472231279;content=长得真有创意，活得真有勇气！;type=1][uin=1472231279;content=你长的真tm后现代;type=1][uin=1472231279;content=你长的好象车祸现场;type=1][uin=1472231279;content=你长的外形不准 比例没打好;type=1][uin=1472231279;content=你干嘛用屁股挡住脸啊！;type=1][uin=1472231279;content=我觉得世界上就只有两种人能吸引人，一种是特漂亮的一种就是你这样的;type=1][uin=1472231279;content=你的长相很提神的说!!;type=1][uin=1472231279;content=你需要回炉重造;type=1][uin=1472231279;content=他们怎么能管你叫猪呢？？这太不像话了！总不能人家长的像什么就叫人家什么吧！怎么能说你长得像猪;type=1][uin=1472231279;content=靠，你TMD长得太好认了。;type=1][uin=1472231279;content=长的很科幻,长的很抽象!;type=1][uin=1472231279;content=见过丑的，没见过这么丑的。乍一看挺丑，仔细一看更丑！;type=1][uin=1472231279;content=长的很无辜，长的对不起人民对不起党。;type=1][uin=1472231279;content=你长的拖慢网速，你长的太耗内存;type=1][uin=1472231279;content=你光着身子追我两公里 我回一次头都算我是流氓!;type=1][uin=1472231279;content=把你脸上的分辨率调低点好吗？;type=1][uin=1472231279;content=你长的违章!;type=1][uin=1472231279;content=国际脸孔世界通用;type=1][uin=1472231279;content=很惋惜的看着他说："手术能整回来吗？";type=1][uin=1472231279;content=你的长相突破了人类的想象...;type=1][uin=1472231279;content=你张的很野兽派嘛！！　;type=1][uin=1472231279;content=你还没有进化完全，长的象人真的难为你了。;type=1][uin=1472231279;content=我想看着你说话，可你为什么把脸埋在你的屁股里？...哦？对不起，我不知道那是你的脸，那你的屁股哪;type=1][uin=1472231279;content=我也不想打击你了。你去动物园看看有没有合适的工作适合你，你这样在街上乱跑很容易被警察射杀的。;type=1]'
    model_list = ["947520"]
    rs = execute_evil_diag(str_uin, str_evil_uin, report_info, model_list)
    print(f"rs:{rs}")
