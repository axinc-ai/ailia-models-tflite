import time
import sys

import numpy as np
import cv2

import googlenet_labels

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# PARAMETERS 1
# ======================
IMAGE_PATH = 'pizza.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0  # for webcam mode


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'GoogLeNet is a CNN architecture that won ImageNet2014', IMAGE_PATH, None,
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite


# ======================
# PARAMETERS 2
# ======================
if args.float:
    MODEL_NAME = 'googlenet_float32'
else:
    MODEL_NAME = 'googlenet_quant_recalib.tflite'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/googlenet/'


# ======================
# Main functions
# ======================
def recognize_from_image(interpreter):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        print(f"update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        interpreter.allocate_tensors()

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        input_data = None
        dtype = np.int8
        if args.float:
            dtype = np.float32
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            keep_aspect_ratio=False,
            gen_input_ailia_tflite=True,
            bgr_to_rgb=True,
            output_type=dtype,
        )
        
        # quantize input data
        inputs = format_input_tensor(input_data, input_details, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]['index'], inputs)
                interpreter.invoke()
                preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            interpreter.set_tensor(input_details[0]['index'], inputs)
            interpreter.invoke()
            preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
        
        preds_tf_lite_int8 = interpreter.get_tensor(output_details[0]['index'])

        # show results
        print_results([preds_tf_lite[0],preds_tf_lite_int8[0]], googlenet_labels.imagenet_category)

    logger.info('Script finished successfully.')


def recognize_from_video(interpreter):

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

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        dtype = np.int8
        if args.float:
            dtype = np.float32
        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='255',
            bgr_to_rgb=True, output_type=dtype
        )

        # Inference
        inputs = format_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        preds_tf_lite = get_output_tensor(interpreter, output_details, 0)

        # show results
        plot_results(
            input_image, preds_tf_lite, googlenet_labels.imagenet_category
        )

        cv2.imshow('frame', frame)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    if args.profile:
        interpreter.set_profile_mode(True)
    interpreter.allocate_tensors()

    if args.video is not None:
        # video mode
        recognize_from_video(interpreter)
    else:
        # image mode
        recognize_from_image(interpreter)


if __name__ == '__main__':
    main()
