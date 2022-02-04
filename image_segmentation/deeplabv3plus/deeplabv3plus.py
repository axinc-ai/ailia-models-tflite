import sys
import time

import cv2
import numpy as np

from deeplab_utils import *

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'DeepLab is a state-of-art deep learning model '
    'for semantic image segmentation.', IMAGE_PATH, SAVE_IMAGE_PATH
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# MODEL PARAMETERS
# ======================
MODEL_NAME = 'deeplab_v3_plus_mnv2_decoder_256_integer_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/deeplabv3plus/'


# ======================
# Main functions
# ======================
def segment_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # prepare input data
    org_img = cv2.imread(args.input)
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='127.5',
        gen_input_ailia_tflite=True,
    )

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]
            end = int(round(time.time() * 1000))
            print(f'ailia processing time {end - start} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]

    # postprocessing
    seg_img = preds_tf_lite.astype(np.uint8)
    seg_img = label_to_color_image(seg_img)
    org_h, org_w = org_img.shape[:2]
    seg_img = cv2.resize(seg_img, (org_w, org_h))
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
    seg_overlay = cv2.addWeighted(org_img, 1.0, seg_img, 0.9, 0)

    cv2.imwrite(args.savepath, seg_overlay)
    print('Script finished successfully.')


def segment_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
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
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]

        # postprocessing
        seg_img = preds_tf_lite.astype(np.uint8)
        seg_img = label_to_color_image(seg_img)
        org_h, org_w = input_image.shape[:2]
        seg_img = cv2.resize(seg_img, (org_w, org_h))
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
        seg_overlay = cv2.addWeighted(input_image, 1.0, seg_img, 0.9, 0)

        cv2.imshow('frame', seg_overlay)

        # save results
        if writer is not None:
            writer.write(seg_overlay)

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
        segment_from_video()
    else:
        # image mode
        args.input = args.input[0]
        segment_from_image()


if __name__ == '__main__':
    main()
