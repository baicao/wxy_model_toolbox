import os
import torch
from common_model.gpu_manager import GPUManager


class HuggingBase:
    def __init__(self, model_name) -> None:
        # 加载模型
        try:
            gm = GPUManager()
            self.gpu_index = gm.auto_choice()
            print("cuda index ", self.gpu_index)
            self.device = (
                f"cuda:{self.gpu_index}" if torch.cuda.is_available() else "cpu"
            )
        except:
            self.device = "cpu"
        print("device", self.device)
        self.model_name = model_name
        self.hugging_folder = "hugging_models"
        self.local_model = os.path.join(self.hugging_folder, self.model_name)
        self.onnx_folder = "onnx_models"

    def hugging_model_export(self):
        if not os.path.exists(self.hugging_folder):
            os.mkdir(self.hugging_folder)
        self.model.save_pretrained(self.local_model)
        self.tokenizer.save_pretrained(self.local_model)
