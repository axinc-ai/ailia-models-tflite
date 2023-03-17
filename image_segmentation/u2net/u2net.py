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

from u2net_utils import imread, load_image, norm, save_result, transform  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.png'
SAVE_IMAGE_PATH = 'output.png'
SAVE_VIDEO_FRAME_PATH = 'video_frame.png' 
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
    '-c', '--composite',
    action='store_true',
    help='Composite input image and predicted alpha value'
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
if args.float:
    model_param = 'float32'
else:
    model_param = 'full_integer_quant'
if args.arch == 'large':
    model_size = 'u2net'
else:
    model_size = 'u2netp'
if args.opset == '10':
    MODEL_PATH = f'{model_size}_{model_param}.tflite'
else:
    MODEL_PATH = f'{model_size}_opset11_{model_param}.tflite'
    
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

def recognize_from_video(interpreter):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        if args.rgb and image.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_data = transform(frame, (args.width, args.height))

        inputs = get_input_tensor(input_data.astype(np.float32), input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()

        details = output_details[0]
        quant_params = details['quantization_parameters']
        int_tensor = interpreter.get_tensor(details['index'])
        real_tensor = int_tensor - quant_params['zero_points']
        real_tensor = real_tensor.astype(np.float32) * quant_params['scales']

        # 各フレームごとに推論結果を重ねて保存
        pred = norm(real_tensor[0])
        if args.rgb and image.shape[2] == 3:
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        pred = cv2.resize(pred, (f_w, f_h))

        frame[:, :, 0] = frame[:, :, 0] * pred + 64 * (1 - pred)
        frame[:, :, 1] = frame[:, :, 1] * pred + 177 * (1 - pred)
        frame[:, :, 2] = frame[:, :, 2] * pred

        cv2.imshow('frame', frame.astype(np.uint8))
        frame_shown = True

    capture.release()
    logger.info('Script finished successfully.')


def recognize_from_image(interpreter):

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

        details = output_details[0]
        dtype = details['dtype']
        if dtype == np.uint8 or dtype == np.int8:
            quant_params = details['quantization_parameters']
            int_tensor = interpreter.get_tensor(details['index'])
            real_tensor = int_tensor - quant_params['zero_points']
            real_tensor = real_tensor.astype(np.float32) * quant_params['scales']
        else:
            real_tensor = interpreter.get_tensor(details['index'])

        save_path = args.savepath
        logger.info(f'saved at : {save_path}')
        save_result(real_tensor, save_path, [h, w])

        # composite
        if args.composite:
            image = imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            image[:, :, 3] = cv2.resize(real_tensor[0], (w, h)) * 255
            cv2.imwrite(save_path, image)

    logger.info('Script finished successfully.')


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
    
    if args.video is not None:
        # video mode
        recognize_from_video(interpreter)
    else:
        # image mode
        recognize_from_image(interpreter)


if __name__ == '__main__':
    main()
