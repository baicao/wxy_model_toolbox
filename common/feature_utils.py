import re
import json
import datetime
import time
import traceback
from urllib.parse import unquote_plus, unquote
from common.interfaces import get_value


def is_pure_english(keyword):
    """
    @param keyword: 需要判断是否为纯英文的字符串
    @return:
    all()函数：用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE：如果是返回 True，否则返回 False。
    ord()函数以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值: 英文的ASCII码数值小于128
    """
    return all(ord(i) < 128 for i in keyword)


# 判断字符串是否匹配
def is_py_match(text, py_deformation_set):
    text = text.lower()
    for pattern in py_deformation_set:
        if text.find(pattern) != -1:
            return (True, pattern)
    return (False, None)


# 从实名中提取拼音或者子串
def real_name_to_deformation(name):
    py_deformation_set = set()
    zh_deformation_set = set()
    if len(name) == "":
        return py_deformation_set, zh_deformation_set
    p = Pinyin()
    real_name_py = p.get_pinyin(name)
    real_name_py_list = real_name_py.split("-")

    # 首字母
    initial_py = "".join([x[0] for x in real_name_py_list])
    initial_py1 = "".join([x[0] + x[0] for x in real_name_py_list])
    initial_py2 = "".join([x[0] + x[0] + x[0] for x in real_name_py_list])
    py_deformation_set.update([initial_py, initial_py1, initial_py2])

    # 名字的英文
    if len(name) >= 3:
        initial_py = "".join([x[0] for x in real_name_py_list[-2:]])
        initial_py1 = "".join([x[0] + x[0] for x in real_name_py_list[-2:]])
        initial_py2 = "".join([x[0] + x[0] + x[0] for x in real_name_py_list[-2:]])
        py_deformation_set.update([initial_py, initial_py1, initial_py2])
        zh_deformation_set.update([name[-2:]])

    return py_deformation_set, zh_deformation_set


def wx_userinfo(baseuserinfo):
    if baseuserinfo == "":
        return ""
    elif isinstance(baseuserinfo, dict):
        pass
    elif baseuserinfo.startswith("result="):
        baseuserinfo = default_get_value(baseuserinfo, "content")
    else:
        baseuserinfo = json.loads(baseuserinfo)

    fields = [
        "avatar",
        "hot_friend_count",
        "age",
        "is_real_name",
        "alis_name",
        "realiable_succ_cnt",
        "gender",
        "evil_score",
        "signature",
        "nick_name",
        "is_bind_card",
        "wxid",
        "sns_img",
    ]
    wx_info = {}
    for f in fields:
        if f in baseuserinfo:

            wx_info[f] = baseuserinfo[f]
        else:
            wx_info[f] = ""
    return wx_info


# 实名与微信资料相印证
def wxinfo_connect_realname(py_deformation_set, wx_info):
    if len(py_deformation_set) == 0:
        return (False, "")
    pattern = re.compile("|".join(py_deformation_set))
    for field_name in ["alias", "nickname", "signature", "username"]:
        rs = field_pattern_match(pattern, wx_info, field_name)
        if rs[0]:
            return rs
    return (False, "")


# 实名与QQ资料相印证
def qqinfo_connect_realname(py_deformation_set, qq_info):
    if len(py_deformation_set) == 0:
        return (False, "")
    pattern = re.compile("|".join(py_deformation_set))
    for field_name in ["qqnick", "personal", "signature"]:
        rs = field_pattern_match(pattern, qq_info, field_name)
        if rs[0]:
            return rs
    return (False, "")


# 获取QQ基础信息
def get_qq_info(qq_info, qq_signature):
    qq_info_id_2_name = {
        "20014": "好友验证",
        "20007": "qqPhone",
        "20002": "qqnick",
        "20009": "gender",
        "20012": "occupation",
        "20013": "homepage",
        "20019": "personal",
        "20021": "college",
        "20031": "birthday",
        "20006": "address",
    }
    if qq_info == "":
        return ""
    elif isinstance(qq_info, dict):
        pass
    elif qq_info.startswith("result="):
        qq_info = default_get_value(qq_info, "status")
    else:
        qq_info = json.loads(qq_info)
    new_qq_info = {}
    for qq_entry in qq_info:
        if len(qq_entry) != 1:
            continue
        key = list(qq_entry.keys())[0]
        if key in qq_info_id_2_name:
            new_qq_info[qq_info_id_2_name[key]] = unquote_plus(
                unquote_plus(list(qq_entry.values())[0])
            )

    qq_signature = get_value(
        qq_signature,
        "longNick",
        default_key=set(["&output1=", "&output12=", "&output2=", "&result=", "&uin="]),
    )
    new_qq_info["signature"] = qq_signature
    return new_qq_info


# 别名印证
def alias_validation(interfaces, py_deformation_set, seed_secret_mobile):
    alias = ""
    wx_info_list = default_get_value(interfaces["getBaseInfo"], "SendInfoToWxInfo")
    if "alias" in wx_info_list:
        alias = wx_info_list["alias"]
        # 1、微信别名和实名人姓名简写印证， 如实名人张三，别名zs*****
        for name_py in py_deformation_set:
            if name_py in alias:
                return 1
        # 2、微信别名和作恶帐号印证，如作恶帐号123456，别名xx123456
        # 3、微信别名和密保手机、绑定手机关联，如密保手机、绑定手机13xxxxxxx，别名zs13xxxxxxx
        # 种子帐号密保手机
        re_str = seed_secret_mobile[:3] + "\d+" + seed_secret_mobile[-2:]
        secretphone = re.findall(re_str, seed_secret_mobile)
        if len(secretphone) > 0:
            return 1
        # 微信绑定手机
        if "mobile" in wx_info_list:
            search_secret_mobile = wx_info_list["mobile"]
            if len(search_secret_mobile) == 11:
                if search_secret_mobile in alias:
                    return 1
                return 0
            return 0
    else:
        return 0


def field_pattern_match(pattern, info, field_name):
    if field_name in info and info[field_name] != "":
        field_value = info[field_name]
        if field_value.startswith("wxid_"):
            return False, ""
        find_rs = re.findall(pattern, field_value)
        if len(find_rs) > 0:
            return True, f"{field_name}:{find_rs[0]}"
    return False, ""


def default_get_value(info, key, default_key: set = None):
    info_json = {}
    try:
        if info == "":
            return {}
        info_unquote = get_value(info, key, default_key)
        if info == "":
            return {}
        else:
            try:
                info_json = json.loads(info_unquote)
            except:  # pylint: disable=bare-except
                info = unquote_plus(info_unquote)
                info_json = json.loads(info)
    except:  # pylint: disable=bare-except
        pass
    return info_json


def get_qq_portrait(portrait_info):
    fields = [
        "qq_age",
        "fraud_ph_min_suc_regtime",
        "fraud_phone_province",
        "qq_recent30_deleted_fri_uincnt",
        "qq_yh_level",
        "qq_recent30_del_fri_uincnt",
        "qq_lockcnt_3mouth",
        "qq_recent30_accept_uincnt",
        "qq_feedback_score",
        "qq_uin_phone_num",
        "puin_occupation",
        "fraud_ph_allopatric_reqistration_times",
        "beat_reason",
        "puin_industry",
        "qq_steal_abnormal_score",
        "qq_recent30_added_friend_uincnt",
        "qq_max_login_suc_time",
        "qq_phone_uin_num",
        "rspvalid",
        "fraud_mig_phone_score",
        "fraud_phone_country",
        "qq_recent30_accepted_addfri_uincnt",
        "qq_msg_active_score",
        "fraud_phone_city",
        "qq_fdbk_idx",
        "beat_personal",
        "fraud_ph_regtimes",
        "qq_credit_score",
        "qq_7days_allopatry_times",
        "account_credit_level",
        "fraud_ph_allopatric_registration_uin_cnt",
        "fraud_ph_max_suc_regtime",
        "qq_login_flag",
        "qq_recent30_addfri_uincnt",
        "beat_common",
        "fraud_allopatric_registration",
    ]
    feature = dict([(x, -1) for x in fields])
    try:
        if portrait_info == "":
            return ""
        elif isinstance(portrait_info, dict):
            pass
        elif portrait_info.startswith("result="):
            portrait_info = default_get_value(
                portrait_info,
                "content",
                set(["&error_msg="]),
            )
        else:
            portrait_info = json.loads(portrait_info)
        for f in fields:
            if f in portrait_info:
                feature[f] = portrait_info[f]

    except:  # pylint: disable=bare-except
        pass
    return feature


def get_wx_portrait(portrait_info):
    feature = {
        "quality": -1,
        "rented": -1,
        "credit_rank": -1,
        "active": -1,
        "is_job_type_primary_school": -1,
        "job_name": "",
        "job_type": "",
        "org_name": "",
        "org_type": "",
        "regcountry": "",
    }
    try:
        if portrait_info == "":
            return ""
        elif isinstance(portrait_info, dict):
            pass
        elif portrait_info.startswith("result="):
            portrait_info = default_get_value(
                portrait_info,
                "spaminfokfInfo",
                set(["&fraudsubserviceinfo=", "&usergamble=", "&errmsg="]),
            )
        else:
            portrait_info = json.loads(portrait_info)

        if "result" in portrait_info:
            spaminfokf_info = portrait_info["result"]
            if "misinfo" in spaminfokf_info:
                misinfo = spaminfokf_info["misinfo"]
                if "regcountry" in misinfo:
                    feature["regcountry"] = misinfo["regcountry"]
            if "fraudinfo" in spaminfokf_info:
                fraudinfo = spaminfokf_info["fraudinfo"]
                if "securitylevel" in fraudinfo:
                    securitylevel = fraudinfo["securitylevel"]
                    if "quality" in securitylevel:
                        feature["quality"] = securitylevel["quality"]
                    if "rented" in securitylevel:
                        feature["rented"] = securitylevel["rented"]
                    if "credit_rank" in securitylevel:
                        feature["credit_rank"] = securitylevel["credit_rank"]
                        if feature["credit_rank"] == 1:
                            feature["is_credit_rank_equal1"] = 1
                    if "active" in securitylevel:
                        feature["active"] = securitylevel["active"]

                    if "job_name" in securitylevel:
                        feature["job_name"] = securitylevel["job_name"]
                    if "job_type" in securitylevel:
                        feature["job_type"] = securitylevel["job_type"]
                        if feature["job_type"] == "中小学":
                            feature["is_job_type_primary_school"] = 1
                        if feature["job_type"] != "":
                            feature["has_job"] = 1
                    if "org_name" in securitylevel:
                        feature["org_name"] = securitylevel["org_name"]
                        if feature["org_name"] != "":
                            feature["is_org_name_available"] = 1
                    if "org_type" in securitylevel:
                        feature["org_type"] = securitylevel["org_type"]
    except:  # pylint: disable=bare-except
        pass
    return feature


def get_real_name(real_name_spread):
    if real_name_spread == "":
        return ""
    elif isinstance(real_name_spread, dict):
        pass
    elif real_name_spread.startswith("result="):
        real_name_spread = default_get_value(real_name_spread, "content")
    else:
        real_name_spread = json.loads(real_name_spread)

    if "name" in real_name_spread and real_name_spread["name"] != "":
        related_realname = real_name_spread["name"]
        return related_realname
    return ""


def login_in_last30d(login_trace, deadline=None):
    if isinstance(login_trace, dict):
        pass
    elif login_trace.startswith("output1="):
        login_trace = default_get_value(login_trace, "output2")
    else:
        login_trace = json.loads(login_trace)

    if deadline is None:
        deadline = float(datetime.datetime.now().strftime("%s.%f"))
    latest_logintime = 0
    if "logindev" in login_trace:
        logindev = login_trace["logindev"]
        for dev_entry in logindev:
            if "lastlogintime" in dev_entry:
                lastlogintime = dev_entry["lastlogintime"]
                if lastlogintime > latest_logintime:
                    latest_logintime = lastlogintime
    if "loginip" in login_trace:
        loginip = login_trace["loginip"]
        for ip_entry in loginip:
            if "lastlogintime" in ip_entry:
                lastlogintime = ip_entry["lastlogintime"]
                if lastlogintime > latest_logintime:
                    latest_logintime = lastlogintime

    if latest_logintime == 0:
        return (None, latest_logintime)
    gap = deadline - latest_logintime
    if gap / 3600 / 24 < 30:
        return (True, latest_logintime)
    return (False, latest_logintime)


def open_wx_time(basic_info):
    if isinstance(basic_info, dict):
        pass
    elif basic_info.startswith("result=0"):
        basic_info = default_get_value(basic_info, "SendInfoToWxInfo")
    else:
        basic_info = json.loads(basic_info)

    if "opentime" in basic_info:
        opentime = basic_info["opentime"]
        opentime = opentime.replace("+", " ")
        opentime_datetime = datetime.datetime.strptime(opentime, "%Y-%m-%d %H:%M:%S")
        opentime_stamp = float(opentime_datetime.strftime("%s.%f"))
        return (True, opentime_stamp)
    return (False, None)


def latest_and_most_login_location(login_trace, level="province"):
    if isinstance(login_trace, dict):
        pass
    elif login_trace.startswith("output1="):
        login_trace = default_get_value(login_trace, "output2")
    else:
        login_trace = json.loads(login_trace)

    lastlogintime, mostcnt = 0, 0
    latest_entry, most_entry = None, None
    if "loginip" in login_trace:
        loginip = login_trace["loginip"]
        for entry in loginip:
            if level not in entry or entry[level] == "":
                continue
            if "lastlogintime" in entry:
                if entry["lastlogintime"] > lastlogintime:
                    lastlogintime = entry["lastlogintime"]
                    latest_entry = entry

            if "cnt" in entry:
                cnt = entry["cnt"]
                if cnt > mostcnt:
                    most_entry = entry
                    mostcnt = cnt
    return latest_entry, most_entry


# 获取微信最后登陆地
def get_lastest_login(login_trace):
    if isinstance(login_trace, dict):
        pass
    elif login_trace.startswith("output1="):
        login_trace = default_get_value(login_trace, "output2")
    else:
        login_trace = json.loads(login_trace)
    latest = -1
    if "logindev" in login_trace:
        for entry in login_trace["logindev"]:
            lastlogintime = entry["lastlogintime"]
            if lastlogintime > latest:
                latest = lastlogintime
    if "loginip" in login_trace:
        for entry in login_trace["loginip"]:
            lastlogintime = entry["lastlogintime"]
            if lastlogintime > latest:
                latest = lastlogintime
    return latest


# 获取QQ登陆地
def get_qq_login(login_trace):
    fields = login_trace.split("&")
    login_trace_dict = {}
    for field in fields:
        k, v = tuple(field.split("="))
        login_trace_dict[k] = v
    return login_trace_dict


def is_activate(login_trace, deadline=None):
    latest = get_lastest_login(login_trace)
    if latest == -1:
        return False
    if deadline is None:
        deadline = float(datetime.datetime.now().strftime("%s.%f"))
    if (deadline - latest) > 3 * 30 * 24 * 3600:
        return False
    return True


def get_login_device(login_trace):
    md5_set = set()
    if "logindev" in login_trace:
        logindev = login_trace["logindev"]
        for entry in logindev:
            md5 = entry["md5"]
            md5_set.add(md5)
    return md5_set


def get_latest_device(login_trace):
    lastlogintime = 0
    device_entry = None
    if "logindev" in login_trace:
        logindev = login_trace["logindev"]
        for entry in logindev:
            if entry["md5"] == "":
                continue
            if entry["lastlogintime"] > lastlogintime:
                device_entry = entry
                lastlogintime = entry["lastlogintime"]
    return device_entry


def same_login_location_mix_type(
    login_trace1, account_type1, login_trace2, account_type2
):
    if login_trace1 == "":
        return (False, "login_trace1 no data")
    if login_trace2 == "":
        return (False, "login_trace2 no data")
    if account_type1 == "qq":
        l1 = get_qq_login(login_trace1)
    elif account_type1 == "wx":
        _, l1 = latest_and_most_login_location(login_trace1)

    if account_type2 == "qq":
        l2 = get_qq_login(login_trace2)
    elif account_type2 == "wx":
        _, l2 = latest_and_most_login_location(login_trace2)
    if (
        l1 is not None
        and l2 is not None
        and l1["province"] != "未知"
        and l1["province"] != ""
        and l1["province"] == l2["province"]
    ):
        return (True, "{},{}".format(l1["province"], l2["province"]))

    if l1 is None or "province" not in l1:
        return (False, "{},{}".format("", l2["province"]))
    if l2 is None or "province" not in l2:
        return (False, "{},{}".format(l1["province"], ""))
    return (False, "{},{}".format(l1["province"], l2["province"]))


# 默认第一个登录地是生活号登陆地
def same_login_location(login_trace1, login_trace2, level="province"):
    if isinstance(login_trace1, dict):
        pass
    elif login_trace1.startswith("output1="):
        login_trace1 = default_get_value(login_trace1, "output2")
    else:
        login_trace1 = json.loads(login_trace1)

    if isinstance(login_trace2, dict):
        pass
    elif login_trace2.startswith("output1="):
        login_trace2 = default_get_value(login_trace2, "output2")
    else:
        login_trace2 = json.loads(login_trace2)
    latest_entry1, most_entry1 = latest_and_most_login_location(login_trace1, level)
    _, most_entry2 = latest_and_most_login_location(login_trace2, level)
    location_list = [""] * 3
    if most_entry1 is not None:
        location_list[0] = most_entry1[level]
    if latest_entry1 is not None:
        location_list[1] = latest_entry1[level]
    if most_entry2 is not None:
        location_list[2] = most_entry2[level]

    if len(set(location_list)) == 1:
        return (True, location_list[0])
    return (False, ",".join(location_list))


# 默认第一个登录地是生活号登陆地
def same_latest_device(login_trace1, login_trace2):
    if isinstance(login_trace1, dict):
        pass
    elif login_trace1.startswith("output1="):
        login_trace1 = default_get_value(login_trace1, "output2")
    else:
        login_trace1 = json.loads(login_trace1)

    if isinstance(login_trace2, dict):
        pass
    elif login_trace2.startswith("output1="):
        login_trace2 = default_get_value(login_trace2, "output2")
    else:
        login_trace2 = json.loads(login_trace2)

    latest_device1 = get_latest_device(login_trace1)
    latest_device2 = get_latest_device(login_trace2)
    if latest_device1 is None or latest_device2 is None:
        return (False, "lack data")

    if latest_device1["md5"] == latest_device2["md5"]:
        return (True, latest_device1["md5"])
    m1 = latest_device1["md5"]
    m2 = latest_device2["md5"]
    return (False, f"{m1},{m2}")


def get_device_logintime(login_trace, target_md5):
    if isinstance(login_trace, dict):
        pass
    elif login_trace.startswith("output1="):
        login_trace = default_get_value(login_trace, "output2")
    else:
        login_trace = json.loads(login_trace)

    if "logindev" in login_trace:
        logindev = login_trace["logindev"]
        for entry in logindev:
            md5 = entry["md5"]
            firstlogintime, lastlogintime, logincnt = 0, 0, 0
            if "firstlogintime" in entry:
                firstlogintime = entry["firstlogintime"]
            if "lastlogintime" in entry:
                lastlogintime = entry["lastlogintime"]
            if "logincnt" in entry:
                logincnt = entry["logincnt"]
            if md5 == target_md5:
                return (firstlogintime, lastlogintime, logincnt)
    return (0, 0, 0)


# 信安色情模型
def xinan_pron_detect(
    xinan_qq_scenes_info, dead_timestamp=datetime.datetime.now().timestamp()
):
    xinan_pron_reporter = set()
    if xinan_qq_scenes_info == "":
        return xinan_pron_reporter
    if isinstance(xinan_qq_scenes_info, dict):
        pass
    elif xinan_qq_scenes_info.startswith("result="):
        xinan_qq_scenes_info = default_get_value(xinan_qq_scenes_info, "content")
    else:
        xinan_qq_scenes_info = json.loads(xinan_qq_scenes_info)
    if "query_Data" not in xinan_qq_scenes_info:
        return xinan_pron_reporter
    if "result_data" not in xinan_qq_scenes_info["query_Data"]:
        return xinan_pron_reporter
    xinan_qq_scenes_list = xinan_qq_scenes_info["query_Data"]["result_data"]
    for entry in xinan_qq_scenes_list:
        if "resultType" not in entry:
            continue
        if "ulluin" not in entry:
            continue
        if "ullimpeachtime" not in entry:
            continue
        xinan_model_rs = entry["resultType"]
        ullimpeachtime = entry["ullimpeachtime"]
        try:
            ullimpeachtime = datetime.datetime.strptime(
                ullimpeachtime, "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            if ullimpeachtime > dead_timestamp:
                continue
        except:  # pylint: disable=bare-except
            traceback.print_exc()
            continue
        if xinan_model_rs == "色情":
            xinan_pron_reporter.add(entry["ulluin"])
    return xinan_pron_reporter


# 投诉场景色情
def tousu_porn_detect(
    complain_list, dead_timestamp=datetime.datetime.now().timestamp()
):
    porn_set = set()
    for complain_entry in complain_list:
        content, task_id, other_txt_evidence = "", "", ""
        if "task_id" in complain_entry:
            task_id = complain_entry["task_id"]
        if "complaint_time" not in complain_entry:
            continue
        if "reporter" not in complain_entry:
            continue
        if "talks" in complain_entry:
            other_txt_evidence = complain_entry["talks"]
        if other_txt_evidence == "":
            continue
        if "complaint_time" in complain_entry:
            complaint_time = complain_entry["complaint_time"]
            if complaint_time > dead_timestamp:
                continue
        reporter = complain_entry["reporter"]
        if not complain_entry["valid"]:
            continue
        if "txws_type" in complain_entry and complain_entry["txws_type"] in [
            "色情",
            "色情低俗",
        ]:
            porn_set.add(reporter)
        # if "predict" in complain_entry and complain_entry["predict"] == 1:
        if ("prob" in complain_entry and complain_entry["prob"] > 0.35) or (
            "predict" in complain_entry and complain_entry["predict"] == 1
        ):
            porn_set.add(reporter)

    return porn_set


def match_device(login_trace1, login_trace2):
    if isinstance(login_trace1, dict):
        pass
    elif login_trace1.startswith("output1="):
        login_trace1 = default_get_value(login_trace1, "output2")
    else:
        login_trace1 = json.loads(login_trace1)

    if isinstance(login_trace2, dict):
        pass
    elif login_trace2.startswith("output1="):
        login_trace2 = default_get_value(login_trace2, "output2")
    else:
        login_trace2 = json.loads(login_trace2)

    devices1 = get_login_device(login_trace1)
    devices2 = get_login_device(login_trace2)
    if len(devices1 & devices2) > 0:
        return (True, list(devices1 & devices2))
    return (False, None)


def match_login_province(login_trace1, login_trace2):
    if isinstance(login_trace1, dict):
        pass
    elif login_trace1.startswith("output1="):
        login_trace1 = default_get_value(login_trace1, "output2")
    else:
        login_trace1 = json.loads(login_trace1)

    if isinstance(login_trace2, dict):
        pass
    elif login_trace2.startswith("output1="):
        login_trace2 = default_get_value(login_trace2, "output2")
    else:
        login_trace2 = json.loads(login_trace2)

    latest_entry1, most_entry1 = latest_and_most_login_location(
        login_trace1, "province"
    )
    latest_entry2, most_entry2 = latest_and_most_login_location(
        login_trace2, "province"
    )

    provinces1 = set()
    if latest_entry1 is not None:
        provinces1.add(latest_entry1["province"])
    if most_entry1 is not None:
        provinces1.add(most_entry1["province"])
    provinces2 = set()
    if latest_entry2 is not None:
        provinces2.add(latest_entry2["province"])
    if most_entry2 is not None:
        provinces2.add(most_entry2["province"])
    common_province = provinces1 & provinces2
    if len(common_province) > 0:
        return (True, list(common_province)[0])
    return (False, None)


def find_porn_qun(info, account_type="wx"):
    qun_porn_pattern = r"🍉|🌸|扩列|夜场娱乐|高端城市花园|交友|同城|福利群|全网聊天|网红|吃瓜|lsp|聚集地幼儿园|初高中|视频软件|直播|软件库|成年视频\
        |男星|女星|明星房|奶片|网红视频|资料|玉足美脚|涩涩|抖淫|抖阴|正能量|正太|彩虹|白袜资料|稚初|瓜瓜精品|茶|模特"

    if account_type == "qq":
        qun_list = get_qq_room_topics(info)
    elif account_type == "wx":
        qun_list = get_wx_room_topics(info)
    for qun in qun_list:
        if "uint32_role_in_group" in qun and qun["uint32_role_in_group"] in [
            20,
            30,
            "20",
            "30",
        ]:
            for key in [
                "roomtopic",
                "announcement",
                "string_group_name",
                "string_group_memo",
            ]:
                if key in qun and qun[key].strip() != "":
                    rs = re.findall(qun_porn_pattern, qun[key].strip())
                    if len(rs) > 0:
                        return (True, rs)
    return (False, "")


def parse_wx_basic_info(input_str: str, column: str):
    if input_str is not None and column is not None:
        contents = input_str.split("&")
        for content in contents:
            arr = content.split("=")
            if len(arr) > 1 and arr[0] == column:
                return arr[1]
    return None


def get_wx_room_topics(wx_room_topic_info):
    roomtopic_list = []
    if wx_room_topic_info == "":
        return roomtopic_list
    elif isinstance(wx_room_topic_info, list):
        pass
    elif wx_room_topic_info.startswith("result="):
        wx_room_topic_info = default_get_value(wx_room_topic_info, "content")
    else:
        wx_room_topic_info = json.loads(wx_room_topic_info)
    for room_info in wx_room_topic_info:
        room_recog_label_str = ""
        if (
            "room_recog_label_str" in room_info
            and room_info["room_recog_label_str"].strip() != ""
        ):
            room_recog_label_str = room_info["room_recog_label_str"]
            if (
                str(room_recog_label_str).strip() != ""
                and str(room_recog_label_str) != "无标签"
            ):
                roomtopic_list.append({"string_group_name": room_recog_label_str})
            # string_group_name
        if "roomtopic" not in room_info or room_info["roomtopic"].strip() == "":
            continue
        roomtopic = room_info["roomtopic"].strip()
        roomlatestmsg_ts = ""
        if "roomlatestmsg_ts" in room_info:
            roomlatestmsg_ts = room_info["roomlatestmsg_ts"].strip()
        roomid = ""
        if "roomid" in room_info:
            roomid = room_info["roomid"].strip()

        announcement = ""
        if "announcement" in room_info:
            announcement = room_info["announcement"].strip()
        roomtopic_list.append(
            {
                "roomtopic": roomtopic,
                "announcement": announcement,
                "string_group_name": roomtopic,
                "roomid": roomid,
                "room_recog_label_str": room_recog_label_str,
                "roomlatestmsg_ts": roomlatestmsg_ts,
            }
        )
    return roomtopic_list


def get_qq_room_topics(qq_room_topic_info):
    qq_group_list = []
    if qq_room_topic_info == "":
        return qq_group_list
    qq_room_topic_info = unquote_plus(qq_room_topic_info)
    qq_room_topic_info = json.loads(qq_room_topic_info)

    for room_info in qq_room_topic_info:
        if (
            "string_group_name" not in room_info
            or room_info["string_group_name"].strip() == ""
        ):
            continue
        string_group_name = room_info["string_group_name"].strip()
        string_group_memo = ""
        if "string_group_memo" in room_info:
            string_group_memo = room_info["string_group_memo"].strip()
        qq_group_list.append(
            {
                "string_group_name": string_group_name,
                "string_group_memo": string_group_memo,
            }
        )
    return qq_group_list


def timediff_to_str(start_timestamp, end_timestamp):
    # 计算场景投诉作恶时常，单位：天
    start_timestamp = datetime.datetime.utcfromtimestamp(start_timestamp)
    end_timestamp = datetime.datetime.utcfromtimestamp(end_timestamp)
    result = end_timestamp - start_timestamp
    return result.days


def filter_login_trains(login_trace, create_time):
    if isinstance(login_trace, dict):
        pass
    elif login_trace.startswith("output1="):
        login_trace = default_get_value(login_trace, "output2")
    else:
        login_trace = json.loads(login_trace)
    login_trace_new = {"loginip": [], "logindev": []}
    if "loginip" in login_trace:
        loginip = login_trace["loginip"]
        for entr in loginip:
            if (
                entr["firstlogintime"]
                < datetime.datetime.strptime(
                    create_time, "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            ):
                login_trace_new["loginip"].append(entr)
    if "logindev" in login_trace:
        loginip = login_trace["logindev"]
        for entr in loginip:
            if (
                entr["firstlogintime"]
                < datetime.datetime.strptime(
                    create_time, "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            ):
                login_trace_new["logindev"].append(entr)
    return login_trace_new


def parse_complain_time(input_time_list):
    evil_duriation = 0
    if len(input_time_list) <= 1:
        return evil_duriation
    time_list = sorted(input_time_list)
    start_time = time_list[0]
    end_time = time_list[-1]
    evil_duriation = timediff_to_str(int(start_time), int(end_time))
    return evil_duriation


def parse_qq_lastLoginArea(input_str: str):
    last_login_time = datetime.datetime.strptime(
        "1970-01-01 08:00:00", "%Y-%m-%d %H:%M:%S"
    )
    content = input_str.split("&", 1)
    if len(content) != 2:
        return last_login_time
    lastLoginTime = content[0].split("=", 1)
    if len(lastLoginTime) != 2:
        return last_login_time
    if len(lastLoginTime[1]) == 0:
        return last_login_time
    last_login_time = datetime.datetime.strptime(lastLoginTime[1], "%Y-%m-%d %H:%M:%S")
    return last_login_time


def parse_qq_register_time(input_str: str):
    last_login_time = datetime.datetime.strptime(
        "1970-01-01 08:00:00", "%Y-%m-%d %H:%M:%S"
    )
    contents = input_str.split("&")
    for content in contents:
        arr = content.split("=")
        if len(arr) > 1 and arr[0] == "registerTime":
            if len(arr[1]) == 0:
                return last_login_time
            return arr[1]
    return last_login_time


def parse_qq_level(input_str: str):
    qq_level = ""
    content = input_str.split("&")
    if len(content) != 9:
        return qq_level
    Level = content[3].split("=", 1)
    if len(Level) != 2:
        return qq_level
    qq_level = Level[1]
    return qq_level


def parse_wx_credit_level(input_str: str):
    credit_rank = ""
    input_str = unquote(input_str)
    content = input_str.split("&", 3)
    if len(content) != 4:
        return credit_rank
    content_r = content[3].rsplit("&", 2)
    if len(content_r) != 3:
        return credit_rank
    getAccountFraudRiskLevel_list = content_r[0].split("=", 1)
    if len(getAccountFraudRiskLevel_list) != 2:
        return credit_rank
    getAccountFraudRiskLevel = getAccountFraudRiskLevel_list[1]
    wx_portrait = get_wx_portrait(getAccountFraudRiskLevel)
    credit_rank = wx_portrait["credit_rank"]
    return credit_rank


def parse_wx_last_login(input_str: str):
    last_login_time = datetime.datetime.strptime(
        "1970-01-01 08:00:00", "%Y-%m-%d %H:%M:%S"
    )
    login_time_list = []
    input_str = unquote(input_str)
    content = input_str.split("&", 1)
    if len(content) != 2:
        return last_login_time
    content_split = content[1].split("=", 1)
    if len(content_split) != 2:
        return last_login_time
    output2 = content_split[1]
    if output2 == "":
        return last_login_time
    output2_json = json.loads(output2)
    for login_type in output2_json:
        for login_detail in output2_json[login_type]:
            login_time = login_detail["lastlogintime"]
            login_time_list.append(login_time)
    login_time_list.sort(reverse=True)
    if len(login_time_list) == 0:
        return last_login_time
    last_login_time = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(login_time_list[0])
    )
    last_login_time = datetime.datetime.strptime(last_login_time, "%Y-%m-%d %H:%M:%S")
    return last_login_time


if __name__ == "__main__":
    pass
