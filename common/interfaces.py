import socket
import time
import re
import select
import traceback
import random
import datetime
import json
import hashlib
from urllib.parse import unquote_plus
import uuid
import requests
from requests.adapters import HTTPAdapter
import simplejson
import logging
from common.ces_template import CESTemplate

# the max retries for http connect
MAX_RETRIES = 3
s = requests.Session()
s.mount("http://", HTTPAdapter(max_retries=MAX_RETRIES))
s.mount("https://", HTTPAdapter(max_retries=MAX_RETRIES))

logger = logging
trace_logger = logging
interface_logger = logging


def curl_post(url, post_data, headers=None):
    if headers is None:
        headers = {"Content-type": "application/json"}
    response = requests.post(
        url, data=post_data.encode("utf-8"), headers=headers, timeout=5
    )
    content = response.json()
    return content


def get_value(sou, key, default_key: set = None):
    begin = sou.find(key)
    match = re.findall(r"&[a-zA-Z0-9_-]{2,20}=", sou)
    end = -1
    if len(match) > 0:
        for m in match:
            if default_key is not None and m in default_key:
                end = sou.find(m)
                break
            elif default_key is None:
                end = sou.find(m)
                break
    if begin == -1:
        return ""
    if end != -1:
        if begin == 0:
            value = sou[len(key) + 1 : end]
            return value.strip()
        else:
            sub = sou[begin:]
            return get_value(sub, key, default_key)
    else:
        value = sou[len(key) + 1 :]
        return value.strip()


def get_user_trace(
    account: str,
    start_time: str,
    end_time: str,
    account_type: str,
    relation_only="false",
    prefix="",
):
    """获取指定手机号的用户的指定范围时间的行为轨迹"""
    if len(start_time) < 12:
        start_time = f"{start_time} 00:00:00"
    if len(end_time) < 12:
        end_time = f"{end_time} 00:00:00"

    url = f"http://kfdata.cm.com/interface/userProfile/getUserProfile?relation_only={relation_only}\
        &relation_type=0&account={account}&start_time={start_time}&end_time={end_time}&account_type={account_type}"

    request_id = uuid.uuid1()
    if relation_only == "false":
        trace_logger.info("[%s-%s] Request %s", prefix, request_id, url)
    else:
        logger.info("[%s-%s] Request %s", prefix, request_id, url)
    result = requests.get(url, timeout=50)
    if relation_only:
        trace_logger.info("[%s-%s] %s Response %s", prefix, request_id, account, result)
    else:
        logger.info("[%s-%s] %s Response %s", prefix, request_id, account, result)
    result1 = result.replace("\r", "")
    result1 = result1.replace("\n", "")
    result1 = result1.replace("\r\n", "")
    result = simplejson.loads(result)
    return result


def get_md5(s_total) -> str:
    """
    取085账号的md5加密
    Parameters
    ----------
    taskid
    s

    Returns
    -------

    """
    # 时间戳使用账号在库里的时间
    m = hashlib.md5()
    m.update(str(s_total).encode("utf-8"))
    return m.hexdigest()


def get_random_number() -> str:
    """
    按要求生成10位数随机数

    Returns
    -------

    """
    return "".join(str(random.choice(range(10))) for _ in range(10))


def get_fromtype_oa() -> str:
    """
    用到oa的格式
    Returns
    -------

    """
    ran_num = get_random_number()
    time_now = int(time.time())
    fromtype = f"GS_{time_now}_{ran_num}_cache"
    return fromtype


def get_related_account_by_trace(account: str, account_type: str = "2") -> str:
    """query account to get related account

    Args:
        phone_number (str): input account
        account_type (str): 2-phone 1-qq 0-openid

    Returns:
        str: openid
    """
    openid_list, phone_list, qq_list = None, None, None
    url = f"http://kfdata.cm.com/interface/userProfile/getUserProfile?\
        relation_only=true&relation_type=-1&account={account}\
        &account_type={account_type}"

    try:
        logger.info("Request %s", url)
        res = requests.get(url, timeout=50)
        logger.info("Response %s", res.json())
        res_dict: dict = res.json()
        relations = res_dict.get("msg", {}).get("relations")
        openid_list = (
            relations["OPENID"]
            if ("OPENID" in relations and relations["OPENID"])
            else []
        )
        phone_list = (
            relations["PHONE"] if "PHONE" in relations and relations["PHONE"] else []
        )
        qq_list = relations["QQ"] if "QQ" in relations and relations["QQ"] else []
    except:  # pylint: disable=bare-except
        logger.error("Error %s %s", url, traceback.format_exc())
    return openid_list, phone_list, qq_list


def get_openid_by_easygraph(phone_number: str) -> str:
    """
    curl -XPOST 9.138.64.19:8091/easygraph/edge/queryEdges \
    -H 'Content-Type: Application/json;charset=utf-8' \
    -d '{"columnName": "v_phone","id":"18948239916","skip": 1}'
    """
    openid = ""
    base_url = "http://9.138.64.19:8091/easygraph/edge/queryEdges"
    post_data = {"columnName": "v_phone", "id": phone_number, "skip": 1}
    header = {"Content-Type": "Application/json"}
    try:
        logger.info("Request %s", post_data)
        res = requests.post(base_url, json=post_data, headers=header, timeout=30)
        logger.info("Response %s", res.json())
        res_list: list = res.json()
        # print(res_list)
        # avoid None situation
        if len(res_list) == 0:
            return openid
        results = res_list[0].get("results")
        if len(results) > 0:
            for item in results:
                openid: str = item.get("to") or ""
                if openid != "" and openid.startswith("o4N"):
                    break
    except:  # pylint: disable=bare-except
        logger.error("Error %s %s", post_data, traceback.format_exc())

    return openid


def get_target_ups(tcp_config, tcp_back_config, fin_key, fin_val, fout_key):
    req = f"command=CmdXYUPS&type=get&finKey={fin_key}&finVal={fin_val}&foutKey={fout_key}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def get_wx_info(
    tcp_config,
    tcp_back_config,
    account,
    account_type,
    ticket,
    fromtype,
    fromuserid,
    task_id,
):
    """封号情况

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        account (str): 账号
        accont_type (str): 账号类型
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号
        ticket (str): ticket

    Returns:
        dict: 返回串
    """
    req = f"command=FlowGetWxInfo&input1={account}&type={account_type}&fromuserid={fromuserid}\
        &fromtype={fromtype}&__task_id={task_id}&__ticket={ticket}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "SendInfoToWxInfo")
    content = unquote_plus(content)
    if content:
        content = simplejson.loads(content)
        return content
    return {}


def get_wx_sns(
    tcp_config,
    tcp_back_config,
    username,
    ticket,
    task_id,
    fromtype,
    fromuserid,
):
    """微信社交

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        username (str): 微信号
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号
        ticket (str): ticket

    Returns:
        dict: 返回串
    """
    req = f"command=FlowWxQueryAppealInfo&proc=sns&rtx={fromuserid}&cookie={ticket}&username={username}\
        &fromuserid={fromuserid}\
        &fromtype={fromtype}&__task_id={task_id}&__ticket={ticket}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    output2 = get_value(recv, "output2")
    if output2:
        output2 = simplejson.loads(unquote_plus(output2))
        return output2
    return {}


# 获取ip
def get_ip():
    ip_addr = ""
    try:
        soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # doesn't even have to be reachable
        soc.connect(("10.255.255.255", 1))
        ip_addr = soc.getsockname()[0]
    except:  # pylint: disable=bare-except
        ip_addr = "127.0.0.1"
    finally:
        if soc is not None:
            soc.close()
    return ip_addr


def get_complaint_id(
    tcp_config,
    tcp_back_config,
    payid,
    token,
    taskid,
    fromtype_str,
    time_stamp,
):
    """10
    取085账号的投诉ID，后面有接口用到
    Parameters
    ----------
    taskid
    payid
    token

    Returns
    -------

    """
    response = ""
    complaint_id = ""
    request = f"command=FlowWxPayRecheck&plant_type=payclean&service_type=10000&account={payid}@wx.tenpay.com\
        &task_id={taskid}&token={token}&time={time_stamp}&fromuserid=GS&fromtype={fromtype_str}\r\n"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, request)
    response = tcp_request(tcp_config, tcp_back_config, request)
    interface_logger.info("[%s] Response %s", request_id, response)
    response = unquote_plus(response)
    content = json.loads(get_value(response, "content"))
    complaint_id = content["uinque_id"]
    interface_logger.info("[%s] Response complaint_id %s", request_id, complaint_id)
    return complaint_id


def get_real_name_payid(
    tcp_config,
    tcp_back_config,
    ori_payid,
    payid,
    fromtype,
    fromuserid,
    ticket,
    task_id=None,
):
    """11
    取账号的姓名
    Parameters
    ----------
    payid
    token
    complaintid

    Returns
    -------

    """
    real_names = ""
    payid_set = set()
    local_ip = get_ip()
    if task_id is None or task_id == "":
        task_id = task_generator()
    time_stamp = int(time.time())
    s_total = f"{payid}@wx.tenpay.com|{task_id}|{time_stamp}|kf_inverse_query_key"
    id_token = get_md5(s_total)
    complaint_id = get_complaint_id(
        tcp_config=tcp_config,
        tcp_back_config=tcp_back_config,
        payid=payid,
        token=id_token,
        taskid=task_id,
        fromtype_str=fromtype,
        time_stamp=time_stamp,
    )
    if payid == ori_payid:
        main_uin = ""
        query_uin_type = 1
    else:
        main_uin = ori_payid
        query_uin_type = 2
    request = (
        "command=FlowKFWxPayInfoQuery&service_type=2255&operator_id={fromuserid}&"
        "tcoa_ticket={ticket}&complaint_id={complaint_id}&complaint_channel=可疑团伙清洗&"
        "client_ip={local_ip}&auth_type=5&rtx={fromuserid}&"
        "fromtype={fromtype}&fromuserid={fromuserid}&__task_id={id_token}&__ticket={ticket}&"
        "key_info_type=2&opt_time={time_stamp}&key_need_special_query=1&"
        "key_info_acctid={payid}@wx.tenpay.com&"
        "query_uin_type={query_uin_type}&main_uin={main_uin}\r\n".format(
            fromuserid=fromuserid,
            fromtype=fromtype,
            ticket=ticket,
            complaint_id=complaint_id,
            local_ip=local_ip,
            id_token=id_token,
            time_stamp=time_stamp,
            payid=payid,
            query_uin_type=query_uin_type,
            main_uin=main_uin,
        )
    )
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, request)
    response = tcp_request(tcp_config, tcp_back_config, request)
    interface_logger.info("[%s] Response %s", request_id, response)
    response = unquote_plus(response)
    content = get_value(response, "content")
    if content == "":
        return real_names, payid_set
    content_json = json.loads(content)
    if "name" in content_json:
        real_names = content_json["name"]
        for item in content_json["identity_uin_list"]:
            payid_set.add(item["accid"].split("@")[0])
    return real_names, payid_set


def get_wxinfo(tcp_config, tcp_back_config, account, account_type):
    # account_type 5-uin 3-wxid 2-phone 1-qq
    req = f"command=CmdGetUserInfoIvr&appname=wx_ivr&input={account}&type={account_type}&f=json"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


# qq转openid
# [王者qq]command=CmdSapmg&input1=15210&uin=876016037&appid=1104466820
# [和平qq]command=CmdSapmg&input1=15210&uin=876016037&appid=1106467070


def game_openid_2_wx(tcp_config, tcp_back_config, openid, appid):
    # appid 王者-wx95a3a4d7c627e07d 和平-wxc4c0253df149f02d 客服-wxc8cfdff818e686b9
    req = f"command=CmdWxInfo&input1=getuserinfo_kf&input6=wx_kf&f=json&openid={openid}&appid={appid}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    wx_info = get_value(recv, "SendInfoToWxInfo")
    if wx_info:
        wx_info = simplejson.loads(wx_info)
        return wx_info
    return {}


def game_openid_2_qq(tcp_config, tcp_back_config, openid, appid):
    # appid 王者-1104466820 和平-1106467070
    req = f"command=CmdSapmg&input1=15282&appid={appid}&openid={openid}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    output90 = get_value(recv, "output90")
    if output90:
        output90 = simplejson.loads(output90)
        if len(output90) > 0:
            output90 = output90[0]
            if "uin" in output90:
                return output90["uin"]
    return ""


def get_secret_phone(tcp_config, tcp_back_config, qq):
    """通过qq请求密保手机

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict): 副ip
        qq (str): qq账号
    """
    req = f"command=CmdMbstatQryApi&uin={qq}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    return content


def get_account_status(
    tcp_config, tcp_back_config, account, fromtype, fromuserid, ticket
):
    """账号质量,微信同设备

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        account (str): 账号
        account_type (str): 账号类型
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        ticket (str): ticket

    Returns:
        [type]: [description]
    """
    req = f"command=FlowReportSecondCheckInfo&operate_type=get_fraudticketinfo&fromuserid={fromuserid}\
        &fromtype={fromtype}&__ticket={ticket}&uin={account}&type=WxId"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    fraudticketinfo = get_value(recv, "fraudticketinfo")
    fraudsubserviceinfo = get_value(recv, "fraudsubserviceinfo")
    info = get_value(recv, "spaminfokfInfo")
    usergamble = get_value(recv, "usergamble")
    return fraudticketinfo, fraudsubserviceinfo, info, usergamble


def is_friends(
    tcp_config, tcp_back_config, username1, username2, fromtype, fromuserid, ticket
):
    """绑卡记录

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        username1 (str): 微信号1
        username2 (str): 微信号2
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        ticket (str): ticket

    Returns:
        dict: 返回串
    """
    req = f"command=CmdWxInfo&serverType=getaddressbookinfo&input1=getaddressbookinfo&appname=wx_kf\
        &f=json&input={username1}&rtx={fromuserid}&cookieInfo={ticket}&type=3\
        &friendInput={username2}&friendType=3&fromtype={fromtype}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def task_generator():
    """
    虚拟工单生成
    Returns
    -------

    """
    # 21052609130389075414
    # 21011512295582024619
    # 21024215997098266608生成
    task_id = "GS"
    now = datetime.datetime.now()
    current_time = now.strftime("%y%m%d%H%M%S")
    task_id = task_id + current_time
    for _ in range(0, 8):
        task_id = task_id + str(random.randint(0, 9))
    return task_id


def creid_2_accounts(
    tcp_config, tcp_back_config, creid, fromtype, fromuserid, ticket, task_id
):
    """查询身份证下的其他账号

    Args:
        tcp_config (dict)): 主ip
        tcp_back_config (dict): 副ip
        creid (str): 身份证
        fromtype (str): 来源
        fromuserid (str): rtx
        ticket (str): ticket
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    req = f"command=FlowRealNameAuthRightQuery&service_type=cft&account={creid}\
    &creType=1&task_id={task_id}&rtx_name={fromuserid}&rtx_ticket={ticket}&fromtype={fromtype}\
    &fromuserid={fromuserid}&__task_id={task_id}&__ticket={ticket}&client_ip=100.77.35.149"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    record_list = get_value(recv, "record_list")
    if record_list:
        content = simplejson.loads(unquote_plus(record_list))
        return content
    return {}


def qq_2_creid(tcp_config, tcp_back_config, account):
    """查询QQ账号的身份证

    Args:
        tcp_config (dict)): 主ip
        tcp_back_config (dict): 副ip
        account (str): QQ
        fromtype (str): 来源
        fromuserid (str): rtx
        ticket (str): ticket
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    req = f"command=CmdCftInfo&clientip=100.77.35.149&query_type=p_query_user_detail_c\
        &uin={account}&curtype=1&query_attach=QUERY_USERINFO|QUERY_USERATT"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def bind_cards(tcp_config, tcp_back_config, username, fromtype, fromuserid, task_id):
    """绑卡记录

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        username (str): 微信号
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    req = f"command=FlowWechatAccountInfo&input1=GetWeChatAllBankCardInfo&input37={username}&fromtype={fromtype}\
        &fromuserid={fromuserid}&client_ip=10.82.201.48&__task_id={task_id}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "output90")
    if content:
        content = simplejson.loads(content)
        return content
    return {}


def ban_user_info(
    tcp_config,
    tcp_back_config,
    account,
    account_type,
    ticket,
    fromuserid,
):
    """封号情况

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        account (str): 账号
        accont_type (str): 账号类型
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    req = f"command=FlowWxBanUserInfo&rtx={fromuserid}&type={account_type}&account={account}&cookie={ticket}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "output90")
    if content:
        content = simplejson.loads(content)
        return content
    return {}


def ban_qq_info(tcp_config, tcp_back_config, account, ticket, fromuserid, task_id):
    """QQ封号情况

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        account (str): qq
        accont_type (str): 账号类型
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    opt_time = str(int(time.time()))
    appkey = "M685E476970ad8543CD32b45a0bbfd43"
    sign = get_md5(opt_time + fromuserid + "20002" + appkey)
    req = f"command=FlowKFSecurityCustomerServProxy&service_type=20002&uin={account}&lock_type=0&function_type=3\
    &opt_time={opt_time}&operator={fromuserid}&sign={sign}&token_type=2&__ticket={ticket}&__task_id={task_id}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "output90")
    if content:
        content = simplejson.loads(content)
        return content
    return {}


def wx_beat_record(
    tcp_config,
    tcp_back_config,
    username,
    task_id,
    ticket,
    fromtype,
    fromuserid,
):
    """微信场景举报

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        username (str): 微信号
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号

    Returns:
        dict: 返回串
    """
    req = f"command=FlowFriendsterBeatInfo&type=3&proc=track&rtx={fromuserid}&ticket={ticket}\
        &username={username}&fromtype={fromtype}&fromuserid={fromuserid}&client_ip=10.82.200.42&__task_id={task_id}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    output2 = get_value(recv, "output2")
    if output2:
        output2 = simplejson.loads(unquote_plus(output2))
        return output2
    return {}


def wx_chat_report_es(es_conf, accounts, size=2000):
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request wx_chat_report_es %s", request_id, accounts)
    es_template = CESTemplate(**es_conf)
    body = {"query": {"bool": {"must": [{"terms": {"strEvilUin": accounts}}]}}}
    scroll_id, total_size, hits = es_template.init_scroll(body, size)
    interface_logger.info(
        "[%s] Response wx_chat_report_es total_size %s", request_id, total_size
    )
    result = []
    for hit in hits:
        result.append(hit["_source"])
    hit_size = len(hits)
    fetch_size = hit_size
    while scroll_id is not None and fetch_size < total_size:
        scroll_id, hits = es_template.search_scroll(scroll_id, scroll="2m")
        for hit in hits:
            result.append(hit["_source"])
        hit_size = len(hits)
        fetch_size += hit_size
    return result


def wx_chat_report(
    tcp_config, tcp_back_config, username, fromtype, fromuserid, task_id, ticket
):
    """微信场景举报

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict)): 副ip
        username (str): 微信号usename or alias
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号
        ticket (str)): ticket

    Returns:
        dict: 返回串
    """
    req = f"command=FlowReportQueryExpose&type=WxId&uin={username}&ticket={ticket}&\
        fromtype={fromtype}&fromuserid={fromuserid}&client_ip=10.82.200.65&__task_id={task_id}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    report_record = get_value(recv, "report_record")
    if report_record:
        report_record = simplejson.loads(unquote_plus(report_record))
        return report_record
    return {}


def txws_report(username):
    response = requests.get(
        f"http://kfreport.cm.com/Runscript/getReportRecord?username={username}"
    )

    return response.json()


def wx_pay_report(
    tcp_config, tcp_back_config, accid, fromtype, fromuserid, task_id, ticket, commseq
):
    """微信社交保障接口，带交易单号的举报单，来自订单场景和聊天场景

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict): 备ip
        accid (str): 085账号
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号
        ticket (str): ticket

    Returns:
        dict: 返回串
    """
    if not accid.endswith("@wx.tenpay.com"):
        accid += "@wx.tenpay.com"
    req = f"command=FlowCftFlexibleBlack&input1=1&querytype=139&input5={accid}@wx.tenpay.com\
        &rtx={fromuserid}&fromtype={fromtype}&fromuserid={fromuserid}&__task_id={task_id}\
            &__ticket={ticket}&CommSeq={commseq}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def accid_2_username(tcp_config, tcp_back_config, accid):
    """支付账号转微信号

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict): 备ip
        accid (str): 085账号

    Returns:
        dict: 返回串
    """
    if not accid.endswith("@wx.tenpay.com"):
        accid += "@wx.tenpay.com"
    req = f"command=CmdKFWxPayInfo&auth_type=1&service_type=2253&key_info_type=2&key_info_acctid={accid}"
    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(content)
        if "query_wxusername_result" in content and content["query_wxusername_result"]:
            query_wxusername_result = content["query_wxusername_result"][0]
            if "wx_user_name" in query_wxusername_result:
                wx_user_name = query_wxusername_result["wx_user_name"]
                return wx_user_name
    return ""


def wx_bind_history(
    tcp_config, tcp_back_config, accid, fromtype, fromuserid, task_id, ticket, commseq
):
    """微信绑定解绑记录

    Args:
        tcp_config (dict): 主ip
        tcp_back_config (dict): 备ip
        accid (str): 085账号
        fromtype (str): 请求方
        fromuserid (str): 请求rtx
        task_id (str): 工单号
        ticket (str): ticket

    Returns:
        dict: 返回串
    """
    if not accid.endswith("@wx.tenpay.com"):
        accid += "@wx.tenpay.com"
    req = f"command=FlowCftFlexibleBlack&input1=1&querytype=139&input5={accid}@wx.tenpay.com\
        &rtx={fromuserid}&fromtype={fromtype}&fromuserid={fromuserid}&__task_id={task_id}\
            &__ticket={ticket}&CommSeq={commseq}"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "content")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def get_qq_profile(tcp_config, tcp_back_config, qq, fromuserid, ticket):
    req = f"command=CmdOidb&input1=Query0x480&input5={qq}&input9=1\
        &input7=20002,20031,20009,20014,20015,20005,20007,20006,20010,20011,20012,20013,20019,20021\
        &fromuserid={fromuserid}&sessionKey={ticket}&dwAppID=0&cKeyType=34"

    request_id = uuid.uuid1()
    interface_logger.info("[%s] Request %s", request_id, req)
    recv = tcp_request(tcp_config, tcp_back_config, req)
    interface_logger.info("[%s] Response %s", request_id, recv)
    content = get_value(recv, "output13")
    if content:
        content = simplejson.loads(unquote_plus(content))
        return content
    return {}


def tcp_request(main_config, backups_config, tcpdata, timeout=5):
    """socket 请求"""
    address = (main_config["ip"], main_config["port"])
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setblocking(1)
    client.settimeout(timeout)

    socket_conn = 0
    try:
        socket_conn = client.connect(address)
    except:  # pylint: disable=bare-except
        socket_conn = -1

    if socket_conn == -1:
        client.close()
        address = (backups_config["ip"], backups_config["port"])
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        socket_conn = client.connect(address)

    tcpdata = tcpdata + "\r\n"
    tcpdata = tcpdata.encode("gbk")
    client.send(tcpdata)
    client.setblocking(0)
    recv_data = ""
    while recv_data.find("\r\n") <= 0:
        infds, _, errfds = select.select([client], [], [], timeout)
        if len(infds) == 0 or len(errfds) > 0:
            break
        once_data = client.recv(1024)
        recv_data = recv_data + once_data.decode("utf8", "ignore")
        if not recv_data:
            break
    if recv_data.find("\r\n") <= 0:
        recv_data = recv_data[:-2]

    return recv_data


def test_openid():
    phone_number = "13083659707"
    print(get_related_account_by_trace(phone_number))
    print(get_openid_by_easygraph(phone_number))


if __name__ == "__main__":
    test_openid()
    get_related_account_by_trace("16651090192")
