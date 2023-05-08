import sys
import tensorflow
import tensorflow_addons
import tf2onnx
import onnxruntime as rt

# sys.path.append(
#     "/Users/xiangyuwang/Works/KFBusiness/natural_person/scripts/natural_person/common_model"
# )
from common_model.teen_sex_predicter import TeenSexPredicter

model = TeenSexPredicter()
output_path =  "model.onnx"
tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
