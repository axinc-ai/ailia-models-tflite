import sys
import time

import matplotlib.pyplot as plt
import cv2
import numpy as np
from hrnet_utils import smooth_output, save_pred


sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor  # noqa: E402
from image_utils import load_image  # noqa: E402
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
# MODEL_PATH = "model_float16_quant.tflite"
# MODEL_PATH = "/Users/daisukeakagawa/work/ailia/hrnet/models/HRNetV2-W18-Small-v2/HRNetV2-W18-Small-v2_saved_model/model_float16_quant.tflite"
MODEL_PATH = "HRNetV2-W48/output_quant.tflite"
NORMALIZE_TYPE="127.5"


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
        if args.flags or args.memory_mode:
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
                avertage_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')

            logger.info(f"\taverage time {average_time / args.benchmark_count} ms")
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

        preds_tf_lite = interpreter.get_tensor(output_details[0]["index"])[0]
        # TODO Normal outputにするにはテンソルの形状を合わせる必要あり
        preds_tf_lite = smooth_output(preds_tf_lite, IMAGE_HEIGHT, IMAGE_WIDTH)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        save_pred(preds_tf_lite, savepath, IMAGE_HEIGHT, IMAGE_WIDTH)


def recognize_from_video():
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
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

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type=NORMALIZE_TYPE
        )

        # inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])[0]

        # if args.float:
        #     preds_tf_lite = preds_tf_lite[:,:,0]

        # postprocessing
        # seg_img = preds_tf_lite.astype(np.uint8)
        # seg_img = label_to_color_image(seg_img)
        # org_h, org_w = input_image.shape[:2]
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
    # check_and_download_models()
    if args.video is not None:
        recognize_from_video()
    else:
        recognize_from_image()

if __name__ == "__main__":
    main()
