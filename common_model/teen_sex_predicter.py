import os
import sys
from platform import system
import configparser
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa  # pylint: disable=unused-import # noqa: F401
import onnxruntime as rt 

if system() == "Darwin":
    MODEL_DIR = "/Users/xiangyuwang/Software/"
elif system() == "Linux":
    MODEL_DIR = "/dockerdata/nature_person/cron_scripts/"
else:
    sys.exit()
sys.path.append(MODEL_DIR)
# pylint: disable=wrong-import-position
from teen_sex_v3.tokenizer import TokenSeqBuilder
from common.log_factory import logger
# pylint: enable=wrong-import-position


class TeenSexPredicter:

    def __init__(self) -> None:
        package_name = "teen_sex_v3"
        self.package_dir = self._found_install_dir(package_name)
        logger.info("package_dir:{}".format(self.package_dir))
        model_file = os.path.join(self.package_dir, "saved_model")
        config_file = os.path.join(self.package_dir,
                                   "textcnn_config_class2.ini")
        vocabulary_path = os.path.join(
            self.package_dir, "textcnn_array_2class/token_dict_text_cnn.txt")
        # 加载配置
        cfg = configparser.ConfigParser()
        cfg.read(config_file, encoding="utf-8")
        labels = cfg.get("MODEL", "lables").split(",")
        labels_2_id = dict(zip(labels, range(len(labels))))
        dimension = cfg.getint("MODEL", "dimension")
        # 模型加载

        # try:
        #     from common.gpu_manager import GPUManager
        #
        #     gm = GPUManager()
        #     gpu_id = gm.auto_choice_gpuid()
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # except Exception as e:
        #     pass

        self.pb_mode = tf.keras.models.load_model(model_file)
        self.seq_builder = TokenSeqBuilder(
            line_sep="\t",
            config_dir=MODEL_DIR,
            vocabulary_path=vocabulary_path,
            label_map=labels_2_id,
            dimension=dimension,
        )
        self.seq_builder.initialize_jieba_userdict(
            os.path.join(self.package_dir, "userdict.txt"))
        self.seq_builder.initialize_jieba_userdict(
            os.path.join(self.package_dir, "userdict_teen_sex.txt"))
        self.seq_builder.initialize_stopwords(
            os.path.join(self.package_dir, "stopwords.txt"))

    def _found_install_dir(self, package_name):
        installed_dir = None
        for p in sys.path:
            installed_dir = os.path.join(p, package_name)
            if os.path.exists(installed_dir):
                logger.info("found installed dir -> {}".format(installed_dir))
                break
        return installed_dir

    def predict(self, sample, desc, reporter, be_reported, keep_other=False):
        handle_result = self.seq_builder._handle_line(sample,
                                                      desc,
                                                      reporter,
                                                      be_reported,
                                                      keep_other=keep_other)

        if len(handle_result) > 0:
            (total_x, _, ids) = self.seq_builder.construct_token_sequence(
                token_result=[(handle_result, "儿色", "1")],
                words_size=300,
                userdict_file="userdict_teen_sex.txt",
            )
            if len(ids) > 0:




                probs = self.pb_mode.predict(total_x)
                print("probs", probs)
                prob = np.squeeze(probs)
                predict = 1 if prob >= 0.5 else 0

                sess = rt.InferenceSession('keras_model.onnx')
                input_name = sess.get_inputs()[0].name
                print("input_name", input_name)
                total_x = total_x.astype('float32')
                res= sess.run(None, {input_name:total_x})
                res = res[0]
                print("res", res)
                return (predict, prob)

        return (0, -1)

    def predict_bulk(self, complain_list, keep_other=False):
        data_list = []
        for i, _ in enumerate(complain_list):
            data = complain_list[i]
            handle_result = self.seq_builder._handle_line(
                data["talks"],
                data["desc"],
                data["reporter"],
                data["be_reported"],
                keep_other=keep_other,
            )
            if len(handle_result) > 0:
                data_list.append((handle_result, "儿色", i))
        (total_x, _, ids) = self.seq_builder.construct_token_sequence(
            token_result=data_list,
            words_size=300,
            userdict_file="userdict_teen_sex.txt",
        )
        id_2_predict = {}
        if len(ids) > 0:
            probs = self.pb_mode.predict(total_x)
            prob = np.squeeze(probs, axis=1)
            predict = [1 if p >= 0.5 else 0 for p in prob]
            id_2_predict = dict([(ids[i], (predict[i], prob[i]))
                                 for i in range(len(ids))])
        for i, _ in enumerate(complain_list):
            if i in id_2_predict:
                complain_list[i].update({
                    "predict": id_2_predict[i][0],
                    "prob": id_2_predict[i][1],
                })
            else:
                complain_list[i].update({
                    "predict": 0,
                    "prob": -1,
                })
        return complain_list

    def deal_line(self, line: str):
        line = line.replace("\r", "")
        line = line.replace("\n", "")
        line = line.replace("\r\n", "")
        return line


if __name__ == "__main__":
    #  pylint: disable=line-too-long
    D = (
        "wxid_pjtv7ksitore21",
        "wxid_050gfozxqmgw22",  # noqa E501
        "|被举报人:小程序微信10撸91元                                                                                                                                 曲线到多少倍就赚多少倍钱无限刷1.50倍，一天刷800+️最高到50倍，即赚1000+                                                                                                     每天稳赚1000十下单立马逃跑，在1.2--1.5倍之间点逃跑必赚，切记不要贪，贪之必亏️打开WiFi如果进不了充值️请打开流量进去充值教程:逃跑几倍就赚本金的几倍，假如我们下单十块，在1.2倍的时候点击逃跑，那么我们就等于赚了2块，下五十，就赚10块，多玩几把几百块就可以到手，保存二维码微信扫一扫直接进，不用下载！，不绑卡，提现秒到微信速度进！",  # noqa E501
    )
    E = (
        "wxid_gwqid7kvb86m22",
        "wxid_ahz8m0betaeu22",
        "|其他人(3673685592):@所有人 南北盛德国际11号大空|被举报人:今天有没有07年的|其他人(3673685592):@所有人 明天会大量上新，中圈2位，小贵一位|其他人(205084939):这些是明天新加的人？|其他人(205084939):@粥粥 今天试了几个钟|其他人(205084939):杭州小妹子  全被你试完了|其他人(3673685592):图一04年，C+；图二，首下海，新人05年，婴儿肥；图三，在校06年高中生，第一次做。@所有人 |其他人(576143320):@粥粥 你一天试几次啊|其他人(3673685592):好货才会上架，不好的货一般不轻易上架[抱拳]|被举报人:价位|被举报人:@粥粥 价位|其他人(3673685592):@红  鱼 价位要明天出|被举报人:@粥粥 发的都是那里的|其他人(3673685592):我会统筹安排，分布在慧仁家园，远大金地，盛德国际|其他人(3673685592):@小干妈 撤回|其他人(3673685592):@所有人 9号，58号，68号，最后一波了[吃瓜][吃瓜][吃瓜]|其他人(2991442575):@粥",  # noqa E501
    )
    E1 = (
        "wxid_gwqid7kvb86m22",
        "wxid_ahz8m0betaeu22",
        "|被举报人:@所有人 南北盛德国际11号大空|其他人(181221557):今天有没有07年的|被举报人:@所有人 明天会大量上新，中圈2位，小贵一位|其他人(205084939):这些是明天新加的人？|其他人(205084939):@粥粥 今天试了几个钟|其他人(205084939):杭州小妹子  全被你试完了|被举报人:图一04年，C+；图二，首下海，新人05年，婴儿肥；图三，在校06年高中生，第一次做。@所有人 |其他人(576143320):@粥粥 你一天试几次啊|被举报人:好货才会上架，不好的货一般不轻易上架[抱拳]|其他人(181221557):价位|其他人(181221557):@粥粥 价位|被举报人:@红  鱼 价位要明天出|其他人(181221557):@粥粥 发的都是那里的|被举报人:我会统筹安排，分布在慧仁家园，远大金地，盛德国际|被举报人:@小干妈 撤回|被举报人:@所有人 9号，58号，68号，最后一波了[吃瓜][吃瓜][吃瓜]|其他人(2991442575):@粥粥 ✌️不|其他人(2733197070):今天8号在吗|被举报人:@@ 在|被举报人:资料我做",  # noqa E501
    )
    #  pylint: enable=line-too-long

    model = TeenSexPredicter()
    rs = model.predict(E1[2], "", E1[0], E1[1])

    # complain_list = [
    #     {
    #         "talks": D[2],
    #         "desc": "",
    #         "reporter": D[0],
    #         "be_reported": D[1],
    #     },
    #     {
    #         "talks": E[2],
    #         "desc": "",
    #         "reporter": E[0],
    #         "be_reported": E[1],
    #     },
    #     {
    #         "talks": E1[2],
    #         "desc": "",
    #         "reporter": E1[0],
    #         "be_reported": E1[1],
    #     },
    # ]
    # rs = model.predict_bulk(complain_list)
