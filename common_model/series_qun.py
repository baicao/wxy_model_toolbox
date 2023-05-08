import math
import collections
from hellonlp.ChineseWordSegmentation.hyperparameters import Hyperparamters as hp
from hellonlp.ChineseWordSegmentation.modules import (
    get_ngram_frequence_infomation,
    calcul_ngram_entropy,
    calcul_ngram_pmi,
)


class SeriesQun:
    def __init__(self) -> None:
        self.min_n = 2
        self.max_n = 5
        self.min_freq = 3

    def cal_score(self, ngram_freq, ngram_keys):

        # Get left and right ngram entropy
        left_right_entropy = calcul_ngram_entropy(
            ngram_freq, ngram_keys, range(self.min_n, self.max_n + 1)
        )
        # Get pmi ngram entropy
        mi = calcul_ngram_pmi(ngram_freq, ngram_keys, range(self.min_n, self.max_n + 1))
        # Join keys of entropy and keys of pmi
        joint_phrase = mi.keys() & left_right_entropy.keys()

        # Word liberalization
        def word_liberalization(el, er):
            return math.log(
                (el * hp.e**er + 0.00001) / (abs(el - er) + 1), hp.e
            ) + math.log((er * hp.e**el + 0.00001) / (abs(el - er) + 1), hp.e)

        word_info_scores = {
            word: (
                mi[word][0],
                mi[word][1],
                left_right_entropy[word][0],
                left_right_entropy[word][1],
                min(left_right_entropy[word][0], left_right_entropy[word][1]),
                word_liberalization(
                    left_right_entropy[word][0], left_right_entropy[word][1]
                )
                + mi[word][1],
            )
            for word in joint_phrase
        }

        # Drop some special word that end with "的" like "XX的,美丽的,漂亮的"
        target_ngrams = word_info_scores.keys()
        start_chars = collections.Counter([n[0] for n in target_ngrams])
        end_chars = collections.Counter([n[-1] for n in target_ngrams])
        threshold = int(len(target_ngrams) * 0.004)
        threshold = max(50, threshold)
        invalid_start_chars = set(
            [char for char, count in start_chars.items() if count > threshold]
        )
        invalid_end_chars = set(
            [char for char, count in end_chars.items() if count > threshold]
        )
        invalid_target_ngrams = set(
            [
                n
                for n in target_ngrams
                if (n[0] in invalid_start_chars or n[-1] in invalid_end_chars)
            ]
        )
        # Remove some words invalids
        for n in invalid_target_ngrams:
            word_info_scores.pop(n)
        return word_info_scores

    def series_name(self, data):

        ngram_freq, ngram_keys = get_ngram_frequence_infomation(
            data,
            self.min_n,
            self.max_n,
            chunk_size=hp.chunk_size,
            min_freq=self.min_freq,
        )
        word_info_scores = self.cal_score(ngram_freq, ngram_keys)
        new_words = self.drop_repeat_short_term(ngram_freq, word_info_scores)
        series_qun = self.group_qun(new_words, data)

        return series_qun

    def drop_repeat_short_term(self, ngram_freq, word_info_scores):
        ngram_keys = [("".join(x), x) for x in ngram_freq.keys()]
        ngram_keys = sorted(ngram_keys, key=lambda x: len(x[0]))
        delete_key = set()
        for i, (key, key_tuple) in enumerate(ngram_keys):
            if len(key) < self.min_n:
                continue
            for j in range(i + 1, len(ngram_keys)):
                compare_key, compare_key_tuple = ngram_keys[j]
                if len(compare_key) == len(key):
                    continue
                if (
                    key in compare_key
                    and ngram_freq[key_tuple] == ngram_freq[compare_key_tuple]
                ):
                    delete_key.add(key_tuple)
        new_words = set()
        for key in word_info_scores:
            if key not in delete_key:
                new_words.add(
                    (
                        "".join(key),
                        tuple(list(word_info_scores[key]) + [ngram_freq[key]]),
                    )
                )
        return new_words

    def group_qun(self, new_words, data):
        new_words = sorted(new_words, key=lambda x: x[1][-1], reverse=True)
        group_qun = {}
        for d in data:
            for new_entry in new_words:
                new_word = new_entry[0]
                if new_word in d:
                    if new_word not in group_qun:
                        group_qun[new_word] = []
                    group_qun[new_word].append(d)
                    break
        series_qun = {}
        for key in group_qun.keys():
            if len(group_qun[key]) >= self.min_n:
                series_qun[key] = group_qun[key]
        return series_qun


if __name__ == "__main__":

    data = [
        "【乌市】青春交友群",
        "乌市🍒灯红酒绿",
        "乌市🍒美术馆2",
        "乌市🍒汇珊园",
        "乌市🍒幸福花园",
        "乌市🍒美术馆3",
        "乌市🍒美术馆1",
        "乌市🍒️恒大之星",
        "乌市🍒️美琳花源",
        "乌市💕️桃花岛3",
        "乌市💋夜色撩人",
        "乌鲁木齐🈷️百味小",
        "乌鲁木齐🈷️仙女聚",
        "乌鲁木齐🈷️仙女聚",
        "乌鲁木齐🈷️仙女聚",
        "乌鲁木齐🈷️仙女聚",
        "乌鲁木齐之夜6",
        "乌鲁木齐之夜5",
        "乌鲁木齐之夜10",
        "乌鲁木齐都是聚会1",
        "乌鲁木齐都市大聚",
        "同城交友（1）",
        "同城交友（3）",
        "乌市✔美女大本营",
        "	乌鲁木齐💋可a2女神",
        "乌市欢乐汇",
        "乌市大风吹",
        "乌鲁木齐都市大聚会10",
        "乌鲁木齐2020JP（2）",
        "乌鲁木齐2020JP（3）",
        "乌鲁木齐2020JP（4）",
        "乌鲁木齐2020JP（7）",
        "乌市新市区交友群（5",
        "乌市高端",
    ]
    data2 = [
        "a辽鞍海🌼荔枝红茶4️",
        "a辽鞍海🌼荷叶冬瓜6️",
        "a辽鞍海🌼蜜桃乌龙7️",
        "A✡️辽鞍海✡️大眼",
        "A✡️辽鞍海✡️大眼",
        "A✡️辽鞍海✡️大眼",
        "A✡️辽鞍海✡️大眼",
        "🍓鞍山'漫漫长夜😍无",
        "🍓鞍山'💮浪漫满屋",
        "🍓鞍山'风花雪月，梦",
        "辽阳鞍山Vip贵族❹",
        "辽阳鞍山Vip贵族❺",
        "辽鞍娱乐{5}",
        "辽鞍娱乐{3}",
        "花儿朵朵③",
        "花儿朵朵⑩",
        "花儿朵朵⑨",
        "花儿朵朵⑦",
        "花儿朵朵④",
        "花儿朵朵⑥",
        "梅兰竹菊",
        "小天王经典娱乐83",
        "a小鞍@小海@小辽2000",
        "辽阳阳光永在风雨后二",
        "JS小聚酒馆2000",
        "人间烟火🎇2000",
        "小天王娱乐经典",
        "大灰狼",
        "	下辈子不让你孤单",
        "襄平娱乐①",
        "	鞍❤️辽江南小蛮腰",
        "沈阳夜未央⑤",
        "辽鞍鹊桥相会1群2000人",
        "孟泰后山练功🤛🤜",
        "需要点咸味",
        "拉人给福利5个人一部",
        "鞍山交友娱乐群",
        "辽鞍皇家贵族人(1)",
        "同城🉑娱乐共享群",
        "后宫佳丽三千",
        "小天王经典娱乐",
        "辽鞍☃️炫彩斑斓",
        "沈阳夜未央②",
        "鞍辽☞你的笑真的很美",
        "🛏️上快乐大本营",
        "天下至尊舍我其谁1000",
        "一二三四五",
        "✨我的女孩你别碰✨",
        "辽阳鞍山学生VIP",
        "后宫佳丽三千",
    ]
    data3 = [
        "沈阳美女雷霆交友群",
        "南京💐师范职业技术",
        "雷霆二年一班",
        "南京女神🌺交友",
        "王者沈阳醉酒当歌",
        "《昆明飞飞鱼》",
        "雷霆世纪家园",
        "沈阳美女雷霆丘比特",
        "沈阳同城学生交友1k",
        "沈阳美女🎀验证群",
        "沈阳摩天大厦",
        "十里桃花2群",
        "雷霆世纪佳缘",
        "沈阳后宫佳丽三千人",
        "王者沈阳美女交友群⑤",
        "上海闵行灰领达人",
        "沈阳酒醉金迷夜逍遥4",
        "苏州～此妖、未成精",
        "雷霆二年二班",
        "雷霆初一二班",
        "沈阳旺仔俱乐部",
        "南京🍀大学生👗论坛",
        "王者王者荣耀（苏城）",
        "南京💖紫罗兰之夜",
        "沈阳聊天交友大群A（",
        "成都💋耍💄耍💗💗",
        "辽宁沈阳不夜城1000",
        "南京小姐姐群",
    ]
    data4 = [
        "一二次200k🈲️躺群",
        "一二次已开🈲 纯三 陆瑀",
        "一二次klq200k🚫躺群",
        "一二次klq60k🈲️纯三",
        "一二次100k🈲纯三躺群TNT",
        "一二次200k扩列群🈲️纯三",
        "一二次扩列群188k🈲纯三tnt躺群",
        "国乙已开200k🈲纯三 不懂群名",
        "国乙自由扩150k🈲️看不懂群名",
        "国乙klq已开🈲️TNT、纯三",
    ]
    sq = SeriesQun()
    series_quns = sq.series_name(data4)
    print(series_quns)
