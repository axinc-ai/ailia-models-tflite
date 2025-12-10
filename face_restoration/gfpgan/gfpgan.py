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
IMAGE_PATH = 'face_03.png'
SAVE_IMAGE_PATH = 'output.jpg'


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'GFPGAN Model', IMAGE_PATH, SAVE_IMAGE_PATH
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
    MODEL_NAME = 'gfpgan_float'
else:
    MODEL_NAME = 'gfpgan_int8'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/gfpgan/'


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

def recognize_from_image():
    logger.info('Start inference...')

    for test_img_path in args.input:
        img = cv2.imread(test_img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb = rgb.astype("float32") / 255.0
        rgb = (rgb - 0.5) / 0.5
        input_data = np.expand_dims(rgb, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        # net initialize
        if args.tflite:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        else:
            if args.flags or args.memory_mode or args.env_id:
                interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id)
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
        out_img = get_real_tensor(interpreter, output_details, 0)

        out_img = out_img + 1
        out_img *= 127.5
        out_img = out_img.clip(0, 255)
        out_img = out_img.astype(np.uint8)
        out_img = np.transpose(out_img, (0, 2, 3, 1))
        out_img = out_img[0,:,:,::-1]

        savepath = get_savepath(args.savepath, test_img_path)
        cv2.imwrite(savepath, out_img)
    logger.info('Script finished successfully.')


def recognize_from_video():

    capture = webcamera_utils.get_capture(args.video, args.camera_width, args.camera_height)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)

    i_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 4
    i_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 4

    interpreter.resize_tensor_input(0, (1, i_h, i_w, 1))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[0], frame.shape[1]
        frame = frame[h//2:h//2+i_h, w//2:w//2+i_w, :]

        y = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        y = y.astype("float32") / 255.0
        y = (y - 0.5) / 0.5
        input_data = np.expand_dims(y, axis=0)
        input_data = np.transpose(input_data, (0, 3, 1, 2))

        inputs = get_input_tensor(input_data, input_details, 0)

        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        out_img = get_real_tensor(interpreter, output_details, 0)
        
        out_img = out_img + 1
        out_img *= 127.5
        out_img = out_img.clip(0, 255)
        out_img = out_img.astype(np.uint8)
        out_img = np.transpose(out_img, (0, 2, 3, 1))
        out_img = out_img[0,:,:,::-1]

        cv2.imshow('frame', out_img)
        frame_shown = True
        # # save results
        # if writer is not None:
        #     writer.write(output_img)

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
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
