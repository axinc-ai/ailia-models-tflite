import sys
import time

import cv2
import numpy as np
from hrnet_utils import smooth_output, save_pred, gen_preds_img_np


import os
es = os.path.abspath(__file__).split('/')
util_path = os.path.join('/', *es[:es.index('ailia-models-tflite') + 1], 'util')
sys.path.append(util_path)
from utils import file_abs_path, get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor  # noqa: E402
from image_utils import load_image, preprocess_image  # noqa: E402
import webcamera_utils  # noqa: E402

from logging import getLogger
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'test.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 1024
MODEL_NAMES = ['HRNetV2-W48', 'HRNetV2-W18-Small-v1', 'HRNetV2-W18-Small-v2']
NORMALIZE_TYPE="255"


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'High-Resolution networks for semantic segmentations.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-a', '--arch', metavar="ARCH",
    default='HRNetV2-W18-Small-v2',
    choices=MODEL_NAMES,
    help='model architecture:  ' + ' | '.join(MODEL_NAMES) +
         ' (default: HRNetV2-W18-Small-v2)'
)
parser.add_argument(
    '--smooth',  # '-s' has already been reserved for '--savepath'
    action='store_true',
    help='result image will be smoother by applying bilinear upsampling'
)
args = update_parser(parser)


# ======================
# MODEL PARAMETERS
# ======================
if args.float:
    MODEL_NAME = args.arch
else:
    MODEL_NAME = args.arch + "_integer_quant"


MODEL_PATH = file_abs_path(__file__, f'{MODEL_NAME}.tflite')
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/hrnet/'


if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite


if args.shape:
    IMAGE_HEIGHT = args.shape
    IMAGE_WIDTH = args.shape


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.env_id or args.delegate_path is not None:
            interpreter = ailia_tflite.Interpreter(
                model_path=MODEL_PATH, 
                memory_mode=args.memory_mode, 
                flags=args.flags,
                env_id = args.env_id,
                experimental_delegates = delegate_obj(args.delegate_path)
            )
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        logger.info(f'update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}')
        interpreter.resize_tensor_input(
            input_details[0]["index"], 
            [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]
        )
        interpreter.allocate_tensors()
 
    logger.info("Start inference...")

    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type=NORMALIZE_TYPE,
            gen_input_ailia_tflite=True
        )

        # quantize input data
        input_data = format_input_tensor(input_data, input_details, 0)

        # inference
        if args.benchmark:
            logger.info("BENCHMARK mode")
            average_time = 0

            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')

            logger.info(f"\taverage time {average_time / args.benchmark_count} ms")
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

        preds_tf_lite = interpreter.get_tensor(output_details[0]["index"])[0]
        preds_tf_lite = smooth_output(preds_tf_lite, IMAGE_HEIGHT, IMAGE_WIDTH)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        save_pred(preds_tf_lite, savepath, IMAGE_HEIGHT, IMAGE_WIDTH)


def recognize_from_video():
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(
                model_path=MODEL_PATH, 
                memory_mode=args.memory_mode, 
                flags=args.flags
            )
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

    while True:
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_data = preprocess_image(
            frame, 
            (IMAGE_HEIGHT, IMAGE_WIDTH), 
            NORMALIZE_TYPE, 
            batch_dim=True,
            keep_aspect_ratio=True,
            reverse_color_channel=True,
            chan_first=False,
            output_type=np.float32,
            return_scale_pad=False,
            tta="none"
        )
        input_data = format_input_tensor(input_data, input_details, 0)

        # inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]
        preds_tf_lite = smooth_output(preds_tf_lite, IMAGE_HEIGHT, IMAGE_WIDTH)
        preds_tf_lite = gen_preds_img_np(preds_tf_lite, IMAGE_HEIGHT, IMAGE_WIDTH)

        cv2.imshow("Inference result", preds_tf_lite)

        # save results
        if writer is not None:
            writer.write(preds_tf_lite)

    capture.release()
    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    check_and_download_models(MODEL_PATH, REMOTE_PATH)
    if args.video is not None:
        recognize_from_video()
    else:
        recognize_from_image()


if __name__ == "__main__":
    main()
