import sys
import time

import ailia
import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402

from u2net_utils import load_image, norm, save_result, transform  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_SIZE = 320
MODEL_LISTS = ['small', 'large']
OPSET_LISTS = ['10', '11']

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('U square net', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-a', '--arch', metavar='ARCH',
    default='large', choices=MODEL_LISTS,
    help='model lists: ' + ' | '.join(MODEL_LISTS)
)
parser.add_argument(
    '-o', '--opset', metavar='OPSET',
    default='11', choices=OPSET_LISTS,
    help='opset lists: ' + ' | '.join(OPSET_LISTS)
)
parser.add_argument(
    '-w', '--width',
    default=IMAGE_SIZE, type=int,
    help='The segmentation width and height for u2net. (default: 320)'
)
parser.add_argument(
    '-h', '--height',
    default=IMAGE_SIZE, type=int,
    help='The segmentation height and height for u2net. (default: 320)'
)
parser.add_argument(
    '--rgb',
    action='store_true',
    help='Use rgb color space (default: bgr)'
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# Model select
# ======================
if args.opset == "10":
    MODEL_PATH = 'u2net_full_integer_quant.tflite' if args.arch == 'large' else 'u2netp_full_integer_quant.tflite'
else:
    MODEL_PATH = 'u2net_opset11_full_integer_quant.tflite' \
        if args.arch == 'large' else 'u2netp_opset11_full_integer_quant.tflite'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/u2net/'


# ======================
# Utils
# ======================

def get_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        input_tensor = input_tensor.clip(0, 255)
        return input_tensor.astype(dtype)
    else:
        return tensor


# ======================
# Main functions
# ======================

def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    # select inference engine
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)

        # prepare input data
        input_data, h, w = load_image(
            image_path,
            scaled_size=(args.width,args.height),
            rgb_mode=args.rgb
        )

        # inference
        logger.info('Start inference...')

        inputs = get_input_tensor(input_data.astype(np.float32), input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()

        # ここは仮
        details = output_details[0]
        quant_params = details['quantization_parameters']
        int_tensor = interpreter.get_tensor(details['index'])
        real_tensor = int_tensor - quant_params['zero_points']
        real_tensor = real_tensor.astype(np.float32) * quant_params['scales']


        logger.info(f'saved at : {SAVE_IMAGE_PATH}')
        save_result(real_tensor, SAVE_IMAGE_PATH, [h, w])

    logger.info('Script finished successfully.')


if __name__ == '__main__':
    main()
