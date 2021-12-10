import enum
import sys
import time

import cv2
import numpy as np

import resnet50_labels

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'ImageNet classification Model', IMAGE_PATH, None
)
parser.add_argument(
    '--shape', type=int, 
    help='change input image shape (Please specify one int value to change width and height).'
)
parser.add_argument(
    '--float', action='store_true',
    help='use float model.'
)
args = update_parser(parser)

import tensorflow as tf
import ailia_tflite

# ======================
# Parameters 2
# ======================
MODEL_NAME = 'resnet50_quant_verification'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/resnet50/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    image_paths = args.input.split(",")
    input_data = None
    if args.shape:
        height = width = args.shape
    for path in image_paths:
        dtype = np.int8
        if args.float:
            dtype = np.float32
        image = load_image(
            path,
            (height, width),
            normalize_type='Caffe',
            gen_input_ailia_tflite=True,
            bgr_to_rgb=False,
            output_type=dtype
        )
        if input_data is None:
            input_data = image
        else:
            input_data = np.concatenate([input_data, image])

    # net initialize
    interpreter_ailia = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter_tf = ailia_tflite.Interpreter(model_path=MODEL_PATH)

    interpreter_ailia.allocate_tensors()
    interpreter_tf.allocate_tensors()

    input_details = interpreter_ailia.get_input_details()
    output_details = interpreter_ailia.get_output_details()
    
    output_details = sorted(output_details, key=lambda x:x['index'])

    final_output_index = 0
    for i in range(len(output_details)):
        if output_details[i]["name"]=="StatefulPartitionedCall:176":
            final_output_index = i

    interpreter_tf.set_tensor(input_details[0]['index'], input_data)
    interpreter_tf.invoke()

    interpreter_ailia.set_tensor(input_details[0]['index'], input_data)
    interpreter_ailia.invoke()

    preds_ailia = interpreter_ailia.get_tensor(output_details[final_output_index]["index"])
    preds_tf_lite = interpreter_tf.get_tensor(output_details[final_output_index]["index"])
    
    for i, name in enumerate(image_paths):
        print(f"=== {name} ===")
        print("ailia")
        print_results([preds_ailia[i]], resnet50_labels.imagenet_category)
        print("tflite")
        print_results([preds_tf_lite[i]], resnet50_labels.imagenet_category)

    for i in range(len(output_details)):
        preds_ailia = interpreter_ailia.get_tensor(output_details[i]["index"])
        preds_tf_lite = interpreter_tf.get_tensor(output_details[i]["index"])
        if np.allclose(preds_ailia[0],preds_tf_lite[0]):
            result = "match"
        else:
            result = "unmatch"
        print(output_details[i]["name"],result)
    
    print('Script finished successfully.')



def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    # image mode
    recognize_from_image()


if __name__ == '__main__':
    main()
