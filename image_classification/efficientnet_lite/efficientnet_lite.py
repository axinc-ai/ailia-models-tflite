import enum
import sys
import time

import cv2
import numpy as np

import efficientnet_lite_labels

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

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# Parameters 2
# ======================
if args.float:
    MODEL_NAME = 'efficientnetliteb0_float'
else:
    MODEL_NAME = 'efficientnetliteb0_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/efficientnet_lite/'


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
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if len(image_paths) > 1 or args.shape:
        print(f"update input shape {[len(image_paths), height, width, 3]}")
        interpreter.resize_tensor_input(0, [len(image_paths), height, width, 3])
        interpreter.allocate_tensors()
    
    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
    
    for i, name in enumerate(image_paths):
        print(f"=== {name} ===")
        print_results([preds_tf_lite[i]], efficientnet_lite_labels.imagenet_category)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='Caffe',
            bgr_to_rgb=False, output_type=np.int8
        )

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])

        plot_results(
            input_image, preds_tf_lite, efficientnet_lite_labels.imagenet_category
        )
        cv2.imshow('frame', input_image)
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(input_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        args.input = args.input[0]
        recognize_from_image()


if __name__ == '__main__':
    main()
