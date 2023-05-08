import datetime
import re
from typing import List
import pandas as pd


def convert_time(x, time_format: str = "%Y-%m-%d %H:%M:%S"):
    try:
        if isinstance(x, datetime.datetime):
            return x
        elif isinstance(x, str):
            return datetime.datetime.strptime(x, time_format)
        elif isinstance(x, int):
            return datetime.datetime.fromtimestamp(x, time_format)
    except:  # pylint: disable=bare-except
        return x


def is_phone(x) -> bool:
    """judge x is phone
    phone refer to 11 digit mobile phone.

    Args:
        x (str): input

    Returns:
        bool: output
    """
    if x.startswith("1") and len(x) == 11 and x.isdigit():
        return True
    return False


def is_qq(x) -> bool:
    """
    check is x is qq
    qq is a set of number between 5-10, starts from 10000.
    Args:
        x (str):

    Returns:
        bool: qq or not
    """
    if len(x) >= 5 and len(x) <= 10 and x >= "10000" and x.isdigit():
        return True
    return False


def is_wx() -> bool:
    pass


def simple_tie(series: pd.Series) -> List:
    series = series[~pd.isna(series)]
    series = series.apply(lambda x: str(x).replace("_x000D_", "").replace(
        "\r", "").replace("\n", ""))
    series = series.apply(lambda x: re.split(r"[，、|]", str(x)))
    series = series[series.apply(lambda x: isinstance(x, list))]
    result_tie = series.tolist()
    result_list = [
        item.strip() for inner in result_tie for item in inner
        if len(item) == 11 and re.match(r"^1[3-9]\d{9}$", item)
    ]
    return result_list


def read_data_from_excel(file_path: str, sheet_name: str) -> tuple:
    """
    Args:
        file_path:xlsx fp
        sheet_name:str,can not be empty
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    date_key = "日期"
    df[date_key] = df[date_key].apply(convert_time)
    df = df[df[date_key].apply(lambda x: isinstance(x, datetime.datetime))]
    base_date = datetime.datetime(year=2021, month=11, day=1)
    mask = df[date_key].apply(lambda x: x >= base_date)
    df = df[mask]

    phone_key = "手机号"
    phone_list = simple_tie(df[phone_key])
    # print(phone_list[:10])

    qq_or_wx_key = "用户账号（填写QQ/微信账号）"
    qq_or_wx_list = simple_tie(df[qq_or_wx_key])
    # print(qq_or_wx_list[:10])
    return (phone_list, qq_or_wx_list)


if __name__ == "__main__":
    # test_openid()
    # test_read_excel()
    # crytor=PrpCrypt()
    pass
