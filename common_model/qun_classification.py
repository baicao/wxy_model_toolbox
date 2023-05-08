import os
import sys
import re
from enum import Enum
import pandas as pd
from common_model.cut_handler import CutHandler
from common_person.feature_utils import is_pure_english 


class GroupType (Enum):
    # 未知
    UNK = "未知"
    # 学校群
    SCHOOL_GROUP = "学校群"
    # 疾病群
    DISEASE_GROUP = "疾病_其他群"
    # 弱势群体疾病群
    DISEASE_VULNERABLE_GROUP = "疾病_弱势群体群"
    # 家庭群
    FAMILY_GROUP = "家庭群"
    # 兴趣群
    INTEREST_GROUP = "兴趣群"
    # 小区群
    RESIDENCE_GROUP = "小区群"
    # 低保群
    LOW_INCOME = "低保户群"
    # 军人群
    SERVICEMAN_GROUP = "军人群"
    # 粉丝群
    FAN_GROUP = "粉丝群"
    # 游戏群
    GAME_GROUP = "游戏群"
    # 购物群
    CONSUME_GROUP = "消费_购物群"
    # 医美群
    AESTHETIC_MEDICINE_GROUP = "消费_医美群"
    # 休闲娱乐场所群
    LEISURE_GROUP = "消费_休闲娱乐群"
    # 商铺
    SHOP_GROUP = "商铺"


class QunCls():

    def __init__(self,) -> None:

        class_dir = os.path.dirname(sys.modules[QunCls.__module__].__file__)
        data_dir = os.path.join(class_dir, "qq_qun_data")
        userdict_dir = os.path.join(data_dir, "userdict/disease")
        print("class_dir", class_dir)
        data_dir = os.path.join(class_dir, "qq_qun_data")

        disease_kg_file = os.path.join(data_dir, "疾病图谱.xlsx")
        print("disease_kg_file", disease_kg_file)

        # 初始化切词
        self.cut_model = CutHandler(
            type="qqseg",
            userdict_dir=userdict_dir,
        )
        self.disease_kg = self.load_disease(disease_kg_file)
        self.seg_handler = None
        self.counter = 0
        self.disease_pattern = self.generate_disease_pattern(disease_kg_file)
    
    def generate_disease_pattern(self, disease_kg_file):
        disease_kg_df = pd.read_excel(disease_kg_file)
        disease_set = set()
        for i in range(len(disease_kg_df)):
            entry = disease_kg_df.iloc[i]
            alias = entry["识别词"]
            is_risk = entry["是否风险病类"]
            if alias != alias or is_risk != is_risk:
                continue
            if is_risk != "是":
                continue
            alias_split = alias.split(",")
            for a in alias_split:
                if len(a) == "":
                    continue
                if is_pure_english(a):
                    continue
                if len(a) <= 1 or len(a) >= 10:
                    continue
                disease_set.add(a)
        disease_pattern = re.compile("|".join(disease_set))
        return disease_pattern

    def load_disease(self, disease_kg_file):
        disease_kg_df = pd.read_excel(disease_kg_file)
        disease_dict = {}
        for i in range(len(disease_kg_df)):
            entry = disease_kg_df.iloc[i]
            name = entry["name"]
            alias = entry["识别词"]
            if len(name) == "":
                continue
            if len(name) <= 1 or len(name) >= 10:
                continue
            category = entry["category"]
            if category.find("中医") == -1:
                disease_dict[name] = entry
            if alias != alias:
                continue
            alias_split = alias.split(",")
            for a in alias_split:
                if len(a) == "":
                    continue
                if is_pure_english(a):
                    continue
                if len(a) <= 1 or len(a) >= 10:
                    continue
                if category.find("中医") == -1:
                    disease_dict[a] = entry
        return disease_dict

    def close(self):
        self.cut_model.cut_close(self.seg_handler)

    def drop_location(self, text):
        entities = self.cut_model.get_named_entities(text)
        new_text_list = []
        index = 0
        for entity in entities:
            if entity["entity_name"] == "LOCATION":
                entity_word = entity["word"]
                new_text_list.append(text[index:entity["pos"]])
                index = len(entity_word) + index
        new_text_list.append(text[index:])
        clean_pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 匹配不是中文、大小写、数字的其他字符
        new_text = ''.join(new_text_list)
        new_text = clean_pattern.sub('', new_text)
        return new_text

    def inference_disease(self, text):
        group_type = (GroupType.UNK, "")
        ill_pattern = r'病友|病友交流群|结疗群'
        ill_rs = re.findall(ill_pattern, text)
        if len(ill_rs) > 0:
            group_type = (GroupType.DISEASE_VULNERABLE_GROUP, ','.join(ill_rs))
            return group_type
        
        new_text = self.drop_location(text)
        if len(new_text) == "":
            return group_type
        disease_rs = re.findall(self.disease_pattern, new_text)
        if len(disease_rs) > 0:
            group_type = (GroupType.DISEASE_VULNERABLE_GROUP, ','.join(disease_rs))
        return group_type
    
if __name__ == "__main__":

    qun_model = QunCls()
    test = ["鲜花与牛粪", "发红包神器群聊", "济南二区【人事群】", "【济南二区】开灯反馈（苏琳琳）","苏皖(二部，三部）骨干1群", "体检卡"]
    test2 = ["温州腹膜透析群"]
    rs = qun_model.inference_disease(test2[0])
    print(rs)




    
    