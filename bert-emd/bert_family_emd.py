import os
import sys
import onnx
import onnxruntime
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from hugging_base import HuggingBase
from onnxconverter_common import float16

# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/


class BertFamilyEmd(HuggingBase):
    def __init__(self, model_name, inference_type="torch") -> None:
        super().__init__(model_name)
        model_name_2_hugging_name = {
            "chinese-roberta-wwm-ext-large": "hfl/chinese-roberta-wwm-ext-large",
            "chinese-roberta-wwm-ex": "hfl/chinese-roberta-wwm-ext",
            "chinese-bert-wwm-ext": "hfl/chinese-bert-wwm-ext",
            "chinese-bert-www": "hfl/chinese-bert-www",
            "rbt3": "hfl/rbt3",
            "rbtl3": "hfl/rbtl3",
            "chinese-macbert-base": "hfl/chinese-macbert-base",
            "chinese-electra-180g-small-discriminator": "hfl/chinese-electra-180g-small-discriminator",
            "bert-base-chinese": "bert-base-chinese",
            "albert-base-chinese-cluecorpussmall": "uer/albert-base-chinese-cluecorpussmall",
            "chinese_roberta_L-2_H-128": "uer/chinese_roberta_L-2_H-128",
        }
        # 如果存在本地模型，优先加载本地模型，如果不存在从hugging上下载或者直接从cache中加载
        self.model_name = model_name
        if os.path.exists(self.local_model):
            self.hugging_name = self.local_model
        elif model_name in model_name_2_hugging_name:
            self.hugging_name = model_name_2_hugging_name[model_name]
        self.inference_type = inference_type

        self.batch_size = 100
        self.tokenizer = AutoTokenizer.from_pretrained(self.hugging_name)

        if inference_type == "onnx":
            self.onnx_file = os.path.join(self.onnx_folder, self.model_name + ".onnx")
        elif inference_type == "onnx_fp16":
            self.onnx_file = os.path.join(
                self.onnx_folder, self.model_name + "_fp16.onnx"
            )
        else:
            # 如果需要加载一个预训练的语言模型来完成各种自然语言处理任务，应该使用 AutoModel。
            # AutoModel的输出是'last_hidden_state', 'pooler_output', 'hidden_states'
            # last_hidden_state：是BERT模型的最后一层所有token的隐藏状态，形状为[batch_size, sequence_length, hidden_size]。这个输出包含了BERT模型对输入的所有token的编码信息，可以用于训练下游任务，如文本分类、命名实体识别等。
            # pooler_output：是BERT模型的最后一层的第一个token（即[CLS]）的隐藏状态，形状为[batch_size, hidden_size]。这个输出可以被用作整个句子的表示，可以用于句子级别的任务，如情感分析、相似度计算等。
            # hidden_states：是BERT模型所有层的所有token的隐藏状态，形状为[num_hidden_layers+1, batch_size, sequence_length, hidden_size]。这个输出包含了BERT模型每一层对输入的编码信息，可以用于分析模型的行为，如可视化注意力权重等。

            # 如果需要加载一个预训练的遮盖语言模型来填充文本中的空缺，应该使用 AutoModelForMaskedLM。
            # AutoModelForMaskedLM的输出是'logits','hidden_states'
            self.model = AutoModel.from_pretrained(
                self.hugging_name, output_hidden_states=True
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"load {self.model_name} success")

        if hasattr(self, "onnx_file"):
            print("providers", onnxruntime.get_available_providers())
            if os.path.exists(self.onnx_file):
                self.ort_session = onnxruntime.InferenceSession(
                    self.onnx_file, providers=["CPUExecutionProvider"]
                )
                print(f"load {self.onnx_file} success")
            else:
                print(f"load {self.onnx_file} fail, generate file first")
                sys.exit()

    def inference(self, sentences):
        if self.inference_type == "torch":
            return self.torch_inference(sentences)
        elif self.inference_type.startswith("onnx"):
            return self.onnx_inference(sentences)

    def torch_inference(self, sentences):
        if sentences is None:
            return False, []
        elif isinstance(sentences, str):
            sentences = [sentences]
        sentence_embedding = []
        input_ids, _, attention_masks = self.text_process(sentences)
        iter = int(len(input_ids) / self.batch_size) + 1
        for i in range(iter):
            start = self.batch_size * i
            end = min(self.batch_size * (i + 1), len(input_ids))
            batch_input_ids = input_ids[start:end]
            batch_attention_masks = attention_masks[start:end]
            input_ids_tensor = torch.stack(batch_input_ids, dim=0).to(self.device)
            attention_masks_tensor = torch.stack(batch_attention_masks, dim=0).to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(input_ids_tensor, None, attention_masks_tensor)
            emd = outputs["pooler_output"]
            sentence_embedding.extend(emd.detach().cpu().numpy())
        sentence_embedding = np.array(sentence_embedding)
        return True, sentence_embedding

    def text_process(self, sentences, return_tensors="pt"):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        if isinstance(sentences, str):
            sentences = [sentences]
        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors=return_tensors,  # Return pytorch tensors.
            )
            print(encoded_dict.keys())

            # Add the encoded sentence to the list.
            input_ids.extend(encoded_dict["input_ids"])
            token_type_ids.extend(encoded_dict["token_type_ids"])
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.extend(encoded_dict["attention_mask"])
        return input_ids, token_type_ids, attention_masks

    def to_onnx(self, fp16=True):
        if not hasattr(self, "model"):
            print("torch model not load")
            sys.exit()
        if not os.path.exists(self.onnx_folder):
            os.mkdir(self.onnx_folder)
        self.onnx_file = os.path.join(self.onnx_folder, self.model_name + ".onnx")
        (
            dummy_input_ids,
            _,
            dummy_attention_masks,
        ) = self.text_process("测试案例")
        dummy_input_ids = torch.stack(dummy_input_ids, dim=0).to(self.device)
        dummy_attention_masks = torch.stack(dummy_attention_masks, dim=0).to(
            self.device
        )
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_masks),
            f=self.onnx_file,
            input_names=["input_ids", "attention_mask"],
            # output_names 需要跟输出order匹配，这样才能在inference中根据名字取出具体的输出
            output_names=["last_hidden_state", "pooler_output", "hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
            },
            do_constant_folding=True,
            opset_version=13,
        )
        if fp16:
            self.onnx_fp16_file = os.path.join(
                self.onnx_folder, self.model_name + "_fp16.onnx"
            )
            model = onnx.load(self.onnx_file)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, self.onnx_fp16_file)

        # self.ort_session = onnxruntime.InferenceSession(
        #     self.onnx_file, providers=onnxruntime.get_available_providers()
        # )
        # onnx_model = onnx.load(self.onnx_file)
        # onnx.checker.check_model(onnx_model)  # 检查文件模型是否正确
        # onnx.helper.printable_graph(onnx_model.graph)  # 输出计算图

    def onnx_inference(self, sentences):
        if sentences is None:
            return False, []
        elif isinstance(sentences, str):
            sentences = [sentences]
        sentence_embedding = []
        input_ids, _, attention_masks = self.text_process(
            sentences, return_tensors="np"
        )
        iter = int(len(input_ids) / self.batch_size) + 1
        for i in range(iter):
            start = self.batch_size * i
            end = min(self.batch_size * (i + 1), len(input_ids))
            batch_input_ids = input_ids[start:end]
            batch_attention_masks = attention_masks[start:end]

            batch_input_ids = np.array(batch_input_ids)
            batch_attention_masks = np.array(batch_attention_masks)

            input = {
                self.ort_session.get_inputs()[0].name: batch_input_ids,
                self.ort_session.get_inputs()[1].name: batch_attention_masks,
            }
            # 指定hidden_states只会输出第一层，需要设置为None，可能是因为名为hidden_states的层有13层
            outs = self.ort_session.run(["pooler_output"], input)
            outs = list(outs)
            sentence_embedding.extend(outs)
        sentence_embedding = np.array(sentence_embedding)
        return (True, sentence_embedding)


if __name__ == "__main__":
    text = ["我喜欢吃西红柿炒鸡蛋", "这是测试数据"]
    test_model_list = [
        "hfl/chinese-roberta-wwm-ext-large",
        "hfl/chinese-roberta-wwm-ext",
        "hfl/chinese-bert-wwm-ext",
        "hfl/rbt3",
        "hfl/rbtl3",
        "hfl/chinese-macbert-base",
        "hfl/chinese-electra-180g-small-discriminator",
        "bert-base-chinese",
        "uer/albert-base-chinese-cluecorpussmall",
        "uer/chinese_roberta_L-2_H-128",
    ]
    for model_name in ["bert-base-chinese"]:
        bert_emd_server = BertFamilyEmd(model_name)
        bert_emd_server.to_onnx(fp16=True)
        import time

        start = time.time()
        torch_out = bert_emd_server.inference(text)
        print(f"pytorch cost:{time.time()-start}")
        print("torch_out", torch_out)

        bert_emd_onnx_server = BertFamilyEmd(model_name, inference_type="onnx")
        start = time.time()
        onnx_out = bert_emd_onnx_server.inference(text)
        print(f"onnx cost:{time.time()-start}")
        print("onnx_out", onnx_out)

        bert_emd_onnx_fp16_server = BertFamilyEmd(
            model_name, inference_type="onnx_fp16"
        )
        start = time.time()
        onnx_fp16_out = bert_emd_onnx_fp16_server.inference(text)
        print(f"onnx cost:{time.time()-start}")
        print("onnx_fp16_out", onnx_fp16_out)
