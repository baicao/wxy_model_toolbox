import sys
import os
import qqseg
from enum import Enum
import traceback

TC_UTF8 = 0
QQSEG_METHOD = 0
JIEBA_METHOD = 1
PADDING = 0
UNKNOWN = 1
"""
define const flag
"""
# @Deprecated 使用人名识别，推荐使用 TC_CRF 开关替代
TC_CN = 0x00000100
# 开启ne识别，并且基于CRF识别人名、地名、机构名
TC_CRF = 0x00000100
# 开启基于深度学习的命名实体识别
# 目前只支持人名识别，开启后，地名、机构名的识别结果仍由CRF模型提供
TC_NER_DL = 0x00000101
# 基于词典匹配识别其他类别的实体：影视、小说、软件、游戏、音乐等
# 必须同时开启TC_CRF才能生效
TC_OTHER_NE = 0x40000000
# IP类命名实体识别（小说、电影、电视剧、综艺节目以及歌曲、专辑等）
TC_IP = 0x0001100
# // @Deprecated
TC_VIDEO = 0x0001100
# 产品名识别
TC_PRODUCTION = 0x00002100
# 进行词性标注
TC_POS = 0x00000004
# 使用繁体转简体
TC_T2S = 0x00000200
# 进行英文大小写转换
TC_U2L = 0x00000020
# 进行全角半角转换
TC_S2D = 0x00000010
# @Deprecated 人名作为一个词整体输出，推荐使用 TC_PER_W 替代
TC_CN_W = 0x00010000
# 人名 person 作为一个词整体输出
# NOTE: 此开关与TC_ANE互斥，不能同时使用
TC_PER_W = 0x00010000

# 机构名 organization 作为一个词整体输出
TC_ORG_W = 0x00020000

# 地名 location 作为一个词整体输出
TC_LOC_W = 0x00040000

# 影视剧名 video 作为一个词整体输出
TC_VIDEO_W = 0x00000040

# @Deprecated 使用自定义词典，推荐使用 TC_CUS替代
TC_USR = 0x00000008
# 使用自定义词典
TC_CUS = 0x00000008

# 开启附加词切分
TC_ADD = 0x10000000

# 开启人名调整
# NOTE: 此开关和TC_CN_W/TC_PER_W互斥，不能同时使用
TC_ANE = 0x20000000

# use simple mode in ascii for weixin
TC_SIM = 0x08000000

# 字符串编码格式
# w GBK编码
TC_GBK = 0x00100000

# UTF-8编码
TC_UTF8 = 0x00200000

# 使用规则
TC_RUL = 0x00000080

# ASCII字符串作为整体切分
TC_ASC = 0x00080000

# 单字切词
TC_SIG = 0x00000002

# ASCII字符串使用语言模型切分
TC_ASCII_LM = 0x01000000

# ASCII字符串识别出拼音，需要打开TC_ASCII_LM开关
TC_ASCII_LM_PINYIN = 0x02000000

# 对ASCII字符串分割结果中连续的长度小于等于6的字母或数字串进行合并，至少需要打开TC_ASCII_LM开关
TC_ASCII_LM_COMB = 0x04000000

# 基于深度学习方法的分词
TC_SEG_DL = 0x00000800


COARS_GRAINED_METHOD = "MixWord"
FINE_GRAINED_METHOD = "FineWord"


class CutStyle(Enum):
    CUT_CLOSE_INSTANT = 1
    CUT_DELAY_CLOSE = 2


class QQsegHandler(object):
    def __init__(
        self, close_type=CutStyle.CUT_DELAY_CLOSE, qqseg_mode=None, userdict_dir=None
    ):
        try:
            if qqseg_mode is None:
                self.qqseg_mode = (
                    TC_CRF
                    | TC_POS
                    | TC_IP
                    | TC_PRODUCTION
                    | TC_ASC
                    | TC_ASCII_LM_PINYIN
                    | TC_CUS
                    | TC_NER_DL
                )
            else:
                self.qqseg_mode = qqseg_mode
            qqseq_package_dir = self._found_install_dir(package_name="qqseg")
            self.initialize_source_dir = os.path.join(qqseq_package_dir, "qqseg_data")
            self.close_type = close_type

            self.userdict_id = 1
            self.userdict_dir = userdict_dir
            if userdict_dir is not None:
                self.userdict_id = qqseg.TCAddCustomDict(self.userdict_dir)
                current_userdict_id = qqseg.TCGetWorkingCustomDictID()
                print(f"load {self.userdict_dir} userdict id :{self.userdict_id}")

            if close_type == CutStyle.CUT_DELAY_CLOSE:
                qqseg.TCInitSeg(self.initialize_source_dir)
                self.seg_handle = None
            current_userdict_id = qqseg.TCGetWorkingCustomDictID()
            print(f"current userdict id:{current_userdict_id}")
        except:
            print("init failed", traceback.format_exc())

    @staticmethod
    def _found_install_dir(package_name):
        installed_dir = None
        for p in sys.path:
            installed_dir = os.path.join(p, package_name)
            if os.path.exists(installed_dir):
                print("found installed dir -> {}".format(installed_dir))
                break
        return installed_dir

    def get_named_entities(self, text):
        if self.close_type == CutStyle.CUT_CLOSE_INSTANT or self.seg_handle is None:
            qqseg.TCInitSeg(self.initialize_source_dir)
            self.seg_handle = qqseg.TCCreateSegHandle(self.qqseg_mode)

        token_list = []
        results = []
        batch_text = []
        if isinstance(text, list):
            batch_text = text
        elif isinstance(text, str):
            batch_text = [text]
        else:
            batch_text = [str(text)]
        for text in batch_text:
            token_list = []
            rs = qqseg.TCSegment(
                self.seg_handle, text, len(text.encode("utf-8")), qqseg.TC_UTF8
            )
            if rs:
                num_basic_tokens = qqseg.TCGetPhraseCnt(self.seg_handle)
                for i in range(num_basic_tokens):
                    basic_token = qqseg.TCGetPhraseTokenAt(self.seg_handle, i)
                    entity_id = basic_token.cls
                    entity_name = qqseg.TCNeTypeId2Str(entity_id)
                    if entity_name == "OTHERS" or entity_name == "TIME":
                        continue

                    token_list.append(
                        {
                            "word": basic_token.word,
                            "entity_name": entity_name,
                            "pos": basic_token.sidx_ch,
                        }
                    )
                results.append(token_list)

        if self.close_type == CutStyle.CUT_CLOSE_INSTANT:
            qqseg.TCCloseSegHandle(self.seg_handle)
            qqseg.TCUnInitSeg()
        if isinstance(text, list):
            return results
        else:
            return results[0]

    def cut(self, text):
        if self.close_type == CutStyle.CUT_CLOSE_INSTANT or self.seg_handle is None:
            qqseg.TCInitSeg(self.initialize_source_dir)
            self.seg_handle = qqseg.TCCreateSegHandle(self.qqseg_mode)
        # switch_rs = qqseg.TCSwitchCustomDictTo(self.seg_handle, self.userdict_id)
        # current = qqseg.TCGetWorkingCustomDictID()
        token_list = []
        batch_text = []
        results = []
        if isinstance(text, list):
            batch_text = text
        elif isinstance(text, str):
            batch_text = [text]
        else:
            batch_text = [str(text)]
        for text in batch_text:
            token_list = []
            rs = qqseg.TCSegment(
                self.seg_handle, text, len(text.encode("utf-8")), qqseg.TC_UTF8
            )
            if rs:
                num_basic_tokens = qqseg.TCGetResultCnt(self.seg_handle)
                for i in range(num_basic_tokens):
                    basic_token = qqseg.TCGetBasicTokenAt(self.seg_handle, i)
                    pos = qqseg.TCPosId2Str(basic_token.pos)
                    token_list.append({"word": basic_token.word, "pos": pos})
                results.append(token_list)

        if self.close_type == CutStyle.CUT_CLOSE_INSTANT:
            qqseg.TCCloseSegHandle(self.seg_handle)
            qqseg.TCUnInitSeg()
        if isinstance(text, list):
            return results
        else:
            return results[0]

    def lcut(self, text):
        rs = self.cut(text=text)
        return [x["word"] for x in rs]

    def cut_close(self):
        if hasattr(self, "seg_handle") and self.seg_handle is not None:
            qqseg.TCCloseSegHandle(self.seg_handle)
            qqseg.TCUnInitSeg()


if __name__ == "__main__":

    batch_text = ["5️⃣年级3️⃣班英语群", "三人行"]
    # 初始化
    handler = QQsegHandler()
    # 批量切分
    rs = handler.cut(batch_text, CutStyle.CUT_BATCH)
    # 单个切分，速度慢
    rs = handler.cut("5️⃣年级3️⃣班英语群", CutStyle.CUT_SINGLE)
    # 单个切分，需要单独close，速度中等
    rs, seg_handler = handler.cut("5️⃣年级3️⃣班英语群", CutStyle.CUT_DELAY_CLOSE)
    print(rs)
    handler.cut_close(seg_handler)
