import sys
import time

import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results  # noqa: E402
import webcamera_utils  # noqa: E402


# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'lenna.png'
SAVE_IMAGE_PATH = 'output.jpg'


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Super Resolution Model', IMAGE_PATH, SAVE_IMAGE_PATH
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
    MODEL_NAME = 'espcn'
else:
    MODEL_NAME = 'espcn_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/espcn/'


# ======================
# Utils
# ======================
def get_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details['quantization_parameters']
        input_tensor = tensor / quant_params['scales'] + quant_params['zero_points']
        input_tensor = input_tensor.clip(-128, 127)
        return input_tensor.astype(dtype)
    else:
        return tensor

def get_real_tensor(interpreter, output_details, idx):
    details = output_details[idx]
    if details['dtype'] == np.uint8 or details['dtype'] == np.int8:
        quant_params = details['quantization_parameters']
        int_tensor = interpreter.get_tensor(details['index'])
        real_tensor = int_tensor - quant_params['zero_points']
        real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
    else:
        real_tensor = interpreter.get_tensor(details['index'])
    return real_tensor

# ======================
# Main functions
# ======================

import PIL
from PIL import Image, ImageFilter

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

def recognize_from_image():
    logger.info('Start inference...')

    for test_img_path in args.input:
        upscale_factor = 3

        img = Image.open(test_img_path)
        lowres_input = get_lowres_image(img, upscale_factor)
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = np.asarray(y)
        y = np.expand_dims(y, axis=2)
        y = y.astype("float32") / 255.0
        input_data = np.expand_dims(y, axis=0)

        # net initialize
        if args.tflite:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        else:
            if args.flags or args.memory_mode:
                interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
            else:
                interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.resize_tensor_input(0, input_data.shape)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        inputs = get_input_tensor(input_data, input_details, 0)

        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]['index'], inputs)
                interpreter.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            interpreter.set_tensor(input_details[0]['index'], inputs)
            interpreter.invoke()
        out_img_y = get_real_tensor(interpreter, output_details, 0)
        out_img_y = out_img_y[0,:,:,0]

        out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
            "RGB"
        )
        savepath = get_savepath(args.savepath, test_img_path)
        out_img.save(savepath)
    logger.info('Script finished successfully.')


def recognize_from_video():
    logger.info("Video mode is not implemented")


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
