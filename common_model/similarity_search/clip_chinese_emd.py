import os
import torch
import sys
import logging

FILE_PATH = os.path.abspath(__file__)
FILE_DIR = os.path.abspath(os.path.dirname(FILE_PATH) + os.path.sep + ".")
sys.path.append(FILE_DIR + "/../")  # 设置引入模块的根目录
sys.path.append(FILE_DIR + "/../../")  # 设置引入模块的根目录
from common_model.similarity_search.component.model import BertCLIPModel
from common_model.gpu_manager import GPUManager
from transformers import CLIPProcessor
from PIL import Image
import time
import numpy as np


class CLIPChinese:
    def __init__(self, logger=None) -> None:
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

        model_name_or_path = "clip-vit-bert-chinese-1M"
        self.pkg_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(self.pkg_dir, model_name_or_path)
        print("data_dir", data_dir)
        # 加载模型
        gm = GPUManager()
        self.gpu_index = gm.auto_choice()
        print("CLIPChinese cuda index ", self.gpu_index)
        self.model, self.clip_processor = self.load_model_and_processor(data_dir)
        self.device = f"cuda:{self.gpu_index}" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using {self.device} device")
        self.model.to(self.device)
        self.batch_size = 400

    @staticmethod
    def load_model_and_processor(model_name_or_path):
        # 加载模型
        model = BertCLIPModel.from_pretrained(model_name_or_path)
        # note: 代码库默认使用CLIPTokenizer, 这里需要设置自己需要的tokenizer的名称
        CLIPProcessor.tokenizer_class = "BertTokenizerFast"
        processor = CLIPProcessor.from_pretrained(model_name_or_path)
        return model, processor

    def process_data(self, texts, image_files=None):
        # 如果存在需要对图片进行预处理，则读取文件
        if image_files is not None:
            images = [Image.open(x).convert("RGB") for x in image_files]
        else:
            images = None
        # 预处理
        inputs = self.clip_processor(
            images=images, text=texts, return_tensors="pt", padding=True
        )
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        return inputs.to(self.device)

    def predict(self, texts):

        iter = int(len(texts) / self.batch_size + 1)
        text_embeds_list = []
        for i in range(iter):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(texts))
            start = time.time()
            self.logger.info(f"start_idx:{start_idx}, end_idx:{end_idx}")
            batch_texts = texts[start_idx:end_idx]
            inputs = self.process_data(batch_texts, None)
            with torch.no_grad():
                text_embeds = self.model.get_text_features(**inputs)
                self.logger.info(f"emd cost:{time.time()-start}")
                # normalized features
                # todo 是否需要normlize
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                text_embeds_list.extend(text_embeds.detach().cpu().numpy())
                self.logger.info(f"to cpu cost:{time.time()-start}")

        # 输出为True，代表该类index不需要训练，只需要add向量进去即可
        text_embeds_list = np.array(text_embeds_list)

        return text_embeds_list


if __name__ == "__main__":
    clip = CLIPChinese()
    texts = ["VPN本是明令禁止的辱骂国家主席cac无视国家法律永久封禁快速封禁"]
    text_embeds_list = clip.predict(texts)
