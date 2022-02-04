import sys
import time
import numpy as np

import cv2

import blazeface_utils as but

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# PARAMETERS
# ======================
MODEL_NAME = 'blazeface'
MODEL_FLOAT_PATH = 'face_detection_front.tflite'
MODEL_INT_PATH = 'face_detection_front_128_full_integer_quant.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/{MODEL_NAME}/'

IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'result.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazeFace is a fast and light-weight face detector.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-f', '--float',
    action='store_true',
    help='By default, the full integer quantization (8-bit) model is used, ' +
    'but with this option, you can switch to using the full precision ' +
    'floating point (32-bit) model.'
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
    )
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='127.5',
        gen_input_ailia_tflite=True
    )

    # net initialize
    if args.float:
        model_path = MODEL_FLOAT_PATH
    else:
        model_path = MODEL_INT_PATH
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds_tf_lite = {}
            if args.float:
                preds_tf_lite[0] = interpreter.get_tensor(output_details[0]['index'])   #1x896x16 regressors
                preds_tf_lite[1] = interpreter.get_tensor(output_details[1]['index'])   #1x896x1 classificators
            else:
                preds_tf_lite[0] = interpreter.get_tensor(output_details[1]['index'])   #1x896x16 regressors
                preds_tf_lite[1] = interpreter.get_tensor(output_details[0]['index'])   #1x896x1 classificators
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = interpreter.get_tensor(output_details[0]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter.get_tensor(output_details[1]['index'])   #1x896x1 classificators
        else:
            preds_tf_lite[0] = interpreter.get_tensor(output_details[1]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter.get_tensor(output_details[0]['index'])   #1x896x1 classificators

    # postprocessing
    detections = but.postprocess(preds_tf_lite)

    # generate detections
    for detection in detections:
        but.plot_detections(org_img, detection, save_image_path=args.savepath)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.float:
        model_path = "face_detection_front.tflite"
    else:
        model_path = "face_detection_front_128_full_integer_quant.tflite"
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
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
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )

        # inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = interpreter.get_tensor(output_details[0]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter.get_tensor(output_details[1]['index'])   #1x896x1 classificators
        else:
            preds_tf_lite[0] = interpreter.get_tensor(output_details[1]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter.get_tensor(output_details[0]['index'])   #1x896x1 classificators

        # postprocessing
        detections = but.postprocess(preds_tf_lite)
        but.show_result(input_image, detections)
        cv2.imshow('frame', input_image)

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
    check_and_download_models(MODEL_FLOAT_PATH, REMOTE_PATH)
    check_and_download_models(MODEL_INT_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        args.input = args.input[0]
        recognize_from_image()


if __name__ == '__main__':
    main()
