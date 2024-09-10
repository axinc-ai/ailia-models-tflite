import sys
import time

import cv2
import numpy as np

from deeplab_utils import *

# import original modules
import os
es = os.path.abspath(__file__).split('/')
util_path = os.path.join('/', *es[:es.index('ailia-models-tflite') + 1], 'util')
sys.path.append(util_path)
from utils import file_abs_path, get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor  # noqa: E402
from image_utils import load_image  # noqa: E402
import webcamera_utils  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


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

if args.shape:
    IMAGE_HEIGHT = args.shape
    IMAGE_WIDTH = args.shape

# ======================
# MODEL PARAMETERS
# ======================
if args.float:
    MODEL_NAME = 'deeplab_v3_plus_mnv2_decoder_256'
else:
    MODEL_NAME = 'deeplab_v3_plus_mnv2_decoder_256_integer_quant'
MODEL_PATH = file_abs_path(__file__, f'{MODEL_NAME}.tflite')
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/deeplabv3plus/'


# ======================
# Main functions
# ======================
def segment_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        logger.info(f"update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        interpreter.allocate_tensors()

    logger.info('Start inference...')

    for image_path in args.input:
        # prepare input data
        org_img = cv2.imread(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='127.5',
            gen_input_ailia_tflite=True,
        )

        # quantize input data
        input_data = format_input_tensor(input_data, input_details, 0)

        # inference
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]

        # postprocessing
        if args.float:
            preds_tf_lite = preds_tf_lite[:,:,0]
        seg_img = preds_tf_lite.astype(np.uint8)
        seg_img = label_to_color_image(seg_img)
        org_h, org_w = org_img.shape[:2]
        seg_img = cv2.resize(seg_img, (org_w, org_h))
        seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
        seg_overlay = cv2.addWeighted(org_img, 1.0, seg_img, 0.9, 0)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')        
        cv2.imwrite(args.savepath, seg_overlay)
    logger.info('Script finished successfully.')


def segment_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video, args.camera_width, args.camera_height)

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
        if args.float:
            preds_tf_lite = preds_tf_lite[:,:,0]

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
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        segment_from_video()
    else:
        # image mode
        segment_from_image()


if __name__ == '__main__':
    main()
