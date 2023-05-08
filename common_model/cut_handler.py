import os


class CutHandler(object):
    def __init__(self, type="qqseg", userdict_dir=None) -> None:
        self.type = type
        self._init_handler(userdict_dir)

    def load_userdict(self, userdict_dir):
        userdict = set()
        for userdict_file in os.listdir(userdict_dir):
            userdict_path = os.path.join(userdict_dir, userdict_file)
            if os.path.exists(userdict_path):
                temp_userdict = open(userdict_path, encoding="gbk").readlines()
                temp_userdict = [x.strip() for x in temp_userdict]
                userdict.update(temp_userdict)
        return userdict

    def _init_handler(self, userdict_dir):
        if self.type == "qqseg":
            from common_model.qqseg_handler import QQsegHandler, CutStyle

            self.handler = QQsegHandler(
                CutStyle.CUT_DELAY_CLOSE, userdict_dir=userdict_dir
            )
        elif self.type == "jieba":
            import jieba as jieba_handler
            import jieba.posseg

            self.handler = jieba_handler
            userdict = self.load_userdict(userdict_dir)
            for w in userdict:
                self.handler.add_word(w)

    def seg(self, text):
        if self.type == "qqseg":
            rs = self.handler.cut(text)
        elif self.type == "jieba":
            seg_cut = self.handler.posseg.cut(text)
            rs = []
            for word, flag in seg_cut:
                rs.append({"word": word, "pos": flag})
        return rs

    def lcut(self, text):
        rs = self.handler.lcut(text)
        return rs

    def get_named_entities(self, text):
        rs = self.handler.get_named_entities(text)
        return rs

    def close(self):
        if self.type == "qqseg":
            self.handler.cut_close()


if __name__ == "__main__":
    cut_model = CutHandler(
        type="qqseg",
        userdict_dir="/Users/xiangyuwang/Desktop/less_refund/disease/userdict/disease",
    )
    qun_list = [
        "武警路库房租主群",
        "血液病c座水果蔬菜店",
        "台州甲状腺群",
        "甲友群(乐乐群)",
        "关节风湿（永丰谢医生）",
        "儿医血液肿瘤专科病友群",
        "新乡肿瘤登记",
        "河南省慢性病监测信息管理系统",
        "航空总医院病友交流群",
        "血液瘤",
        "上海儿童医学中心交流一群",
        "儿童医学中心病友交流群",
        "儿医血液肿瘤专科病友群",
        "885帮扶【上海儿童中心】病房礼物",
        "【分享】S水滴病友之家-宋洋21",
        "儿童医学中心病友交流群",
        "省医儿科血液病-1",
        "不生病的奥秘新群",
        "《不生病的奥秘》栏目LN",
        "五院病友群",
        "⛳️天津血液病病友结疗群",
        "精神病科104号",
        "永和街职业病防治企业群",
        "富丽华隔离群，小病毒快走开😡",
        "卫辉传统文化真实好病案例群",
        "学员群第十届感染性疾病诊断学习班",
        "田李村慢性病服务群",
        "二院北院住院病历沟通群",
        "平安血液病科2022",
        "平安健康集团检验病理专业群",
        "二院新病历试点沟通群",
        "精神病院",
        "🙏555 投诉人全家老小被生温病",
    ]
    for qun in qun_list:
        rs = cut_model.get_named_entities(qun)
        print(f"qun:{qun}, rs:{rs}")

    text = "武警路库房租主群"
    entities = cut_model.get_named_entities(text)
    new_text = []
    index = 0
    for entity in entities:
        if entity["entity_name"] == "LOCATION":
            entity_word = entity["word"]
            new_text.append(text[index : entity["pos"]])
            index = len(entity_word) + index
    new_text.append(text[index:])
