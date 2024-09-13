# int8とfloatの誤差を比較する

import os
import sys

import ailia_tflite
import numpy as np
import cv2


def find_and_append_util_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        potential_util_path = os.path.join(current_dir, 'util')
        if os.path.exists(potential_util_path):
            sys.path.append(potential_util_path)
            return
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Couldn't find 'util' directory. Please ensure it's in the project directory structure.")


find_and_append_util_path()


from model_utils import format_input_tensor, get_output_tensor
from image_utils import normalize_image


def predict_tflite(tflite_path, input_data):
    # ailiaでtfliteファイルを読み込み
    interpreter = ailia_tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # 入力の形式チェック
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    if any(input_shape[i] != input_data.shape[i] for i in range(4)):
        print("input shape mismatch", input_shape, input_data.shape)
        return
    inputs = format_input_tensor(input_data, input_details, 0)
    interpreter.set_tensor(input_details[0]['index'], inputs)

    # 推論
    interpreter.invoke()

    infos = interpreter.get_node_infos()
    tensors_dict = []
    for node in infos:
        # tensor_idx = node["output_tensor_index"]
        # output_tensor = interpreter.get_tensor(tensor_idx)
        output_details = [node["output_detail"]]
        outputs = get_output_tensor(interpreter, output_details, 0)
        tensors_dict.append({"tensor": outputs, "index": node["index"], "operator": node["operator_name"]})
        # print("Node", node["index"], node["operator_name"], node["output_tensor_index"], output_tensor.shape)
        # print("TensorHash",hashlib.sha224(output_tensor.tobytes()).hexdigest())

    return tensors_dict


def main():
    # 入力画像読み込み
    input_data = cv2.imread("clock.jpg")
    input_data = cv2.resize(input_data, (224, 224))
    input_data = normalize_image(input_data, normalize_type="ImageNet")
    input_data = np.expand_dims(input_data, 0)

    # 推論を行う
    tensors_dict_float = predict_tflite("vgg16_pytorch_float32.tflite", input_data)
    tensors_dict_int8 = predict_tflite("vgg16_pytorch_quant_recalib.tflite", input_data)

    # 比較
    print("---------------------------------------")
    print("Display mean square errors of output tensors")
    print("---------------------------------------")

    for idx in range(len(tensors_dict_float)):
        diff = np.mean(np.square(tensors_dict_float[idx]["tensor"] - tensors_dict_int8[idx]["tensor"]))
        print(tensors_dict_float[idx]["index"],
              tensors_dict_int8[idx]["index"],
              tensors_dict_float[idx]["operator"],
              tensors_dict_int8[idx]["operator"], diff)


if __name__ == '__main__':
    main()
