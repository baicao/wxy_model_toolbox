import time
import torch
from transformers import CLIPProcessor
from component.model import BertCLIPModel
import onnxruntime


def model_2_onnx(
    dummy_input,
    input_names,
    output_names,
    onnx_model_file,
):
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_model_file,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,  # the model's input names
        output_names=output_names,  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "id_size"},
            "pixel_values": {0: "batch_size"},
            "attention_mask": {0: "batch_size", 1: "id_size"},
        },
        # 可以打印模型结果，以及输出
        # verbose=True,
    )


def cpu_onnx_predict(onnx_model_file, inputs):
    sess = onnxruntime.InferenceSession(
        onnx_model_file,
        providers=["CPUExecutionProvider"],
    )
    input_data = {
        "input_ids": inputs["input_ids"].numpy(),
        "pixel_values": inputs["pixel_values"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }
    # CUDAExecutionProvider
    start = time.time()
    outputs = sess.run(output_names, input_data)
    print(f"onnx cost:{time.time()-start}")
    print(outputs)


if __name__ == "__main__":
    model_name_or_path = "clip-vit-bert-chinese-1M"
    model = BertCLIPModel.from_pretrained(model_name_or_path)

    # note: 代码库默认使用CLIPTokenizer, 这里需要设置自己需要的tokenizer的名称
    CLIPProcessor.tokenizer_class = "BertTokenizerFast"
    processor = CLIPProcessor.from_pretrained(model_name_or_path)

    onnx_model_file = "bert_model.onnx"
    texts = [
        "群聚众赌博他们以红包的形式开奖唆使我们去下注 中了就把我们踢出去 我被骗了好多钱 害我无家可归 妻离子散，里面还有很多青少年未成年被骗 他们没有自控能力 输了就去偷去骗来赌博 严重影响我 还我们一个绿色的交友平台 赌博终究害人害己",
        "@.@@此人販爽簧片84え%@C@@@販爽外挂&quot;M1=◆老虎机@@k◆永久祢結3@@=+@gEi代碣fpf@%◆@@。。^@@@◆此人辱写他人0@@]@@0@侵犯財亭@@@@@辱0咢中国@@@&lt;&gt;/)@@ZZ@@1D@@@@Qay@@@@ii@@@◆@@All全部管理員群主賦号fost@Y@cA@@cv&gt;-@@Rbb@-@@&amp;b@@@R--a@@0。。@@fY@@@na@0@@。。@%%@@RR封号処理ll@@@@@@@@@&gt;@@Y]@@@@@j@:@@小号吏挂大号挂群@@@@@@@E--&amp;f,@@R@@◆@@0],R@@%@0@--@を@@@①@@をj@@@:@@@@@@O@--.8f,@@R@@4@-@@-.@@j@R@@@--@-%@@@-@@@」-@@-@@R@S@@@f@@@SB@希望封号永久処理/-:-+C+++。。@@@@%Rb@@@%%@RR封号処理ll@@0BDD@@@&gt;@@Y@@@@@j@:@@小号吏挂大号挂群@@@@@0@E--.&amp;f-,@@R@@4@-@0D。。@0@j@@E@:@@群公告售",
        "◆◆②◆◆⑨◆〉◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆ii◆◆◆◆◆②◆⑦②◆②◆◆&lt;◆⑦&lt;◆◆◆◆◆◆,◆,⑦◆◆◆◆◆◆◆◆◆◆◆◆◆ili◆◆◆◆◆00◆◆◆②C④◆◆+++母◆,◆◆◆◆◆◆◆⑦◆◆②◆◆◆,◆◆◆◆◆◆◆◆,◆◆,◆+++◆◆◆◆◆◆◆◆+++e◆◆◆◆◆◆◆◆◆◆",
        "❼❼发布大型色情◆s◆H涉嫌诈编◆迟◆d❹Dr恶意炒作◆N◆g8_t❺Ma❼ai违法违规●?td◆=*xc❼◆N◆严重影响快手社区c◆秋◆89N❼mN✧(引人不适)✧c❷)◎请官方仔细查它账号34❹P●Su❼Sz?N◆之3视频含有引人限流现家m！3&#x27;in◆KGq●A◆ICISh◆E己M◆◆涉嫌未成年人不良语言◆坎@❼l永久封號"
        "3570357851账号多次提供违规服务3570357851",
    ]
    input_names = ["input_ids", "pixel_values", "attention_mask"]
    output_names = ["onnx::MatMul_3148"]
    inputs = processor(images=None, text=texts, return_tensors="pt", padding=True)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    inputs["pixel_values"] = torch.zeros((len(texts), 3, 224, 224))
    dummy_input = (
        inputs["input_ids"],
        inputs["pixel_values"],
        inputs["attention_mask"],
    )

    # 模型转化
    model_2_onnx(
        dummy_input,
        input_names,
        output_names,
        onnx_model_file,
    )

    # onnx预测
    cpu_onnx_predict(onnx_model_file, inputs)

    # torch预测
    start = time.time()
    inputs.pop("pixel_values")
    feature = model.get_text_features(**inputs)
    feature = feature / feature.norm(dim=-1, keepdim=True)
    print(f"model cost:{time.time()-start}")
    print(feature)
