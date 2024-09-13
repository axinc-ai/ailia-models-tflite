import os
import sys
import time
from logging import getLogger   # noqa: E402

import cv2
import numpy as np


def find_and_append_util_path():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        potential_util_path = os.path.join(current_dir, 'util')
        if os.path.exists(potential_util_path):
            sys.path.append(potential_util_path)
            return
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("Couldn't find 'util' directory. Please ensure it's in the project directory structure.")

find_and_append_util_path()


from utils import file_abs_path, get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
import webcamera_utils  # noqa: E402


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
MODEL_PATH = file_abs_path(__file__, f'{MODEL_NAME}.tflite')
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

def recognize_from_image():
    logger.info('Start inference...')

    for test_img_path in args.input:
        img = cv2.imread(test_img_path)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]

        y = np.expand_dims(y, axis=2)
        y = y.astype("float32") / 255.0
        input_data = np.expand_dims(y, axis=0)

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
        out_img_y = get_real_tensor(interpreter, output_details, 0)
        out_img_y = out_img_y[0,:,:,0]

        out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        
        out_img_y = out_img_y.astype(np.uint8)
        out_img_cr = cv2.resize(cr, (out_img_y.shape[1], out_img_y.shape[0]), cv2.INTER_CUBIC).astype(np.uint8)
        out_img_cb = cv2.resize(cb, (out_img_y.shape[1], out_img_y.shape[0]), cv2.INTER_CUBIC).astype(np.uint8)

        out_img = np.zeros((out_img_y.shape[0], out_img_y.shape[1], 3)).astype(np.uint8)
        out_img[:, :, 0] = out_img_y
        out_img[:, :, 1] = out_img_cr
        out_img[:, :, 2] = out_img_cb

        out_img = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2BGR)

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

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = ycrcb[:,:,0], ycrcb[:,:,1], ycrcb[:,:,2]

        y = np.expand_dims(y, axis=2)
        y = y.astype("float32") / 255.0
        input_data = np.expand_dims(y, axis=0)

        inputs = get_input_tensor(input_data, input_details, 0)

        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        out_img_y = get_real_tensor(interpreter, output_details, 0)
        out_img_y = out_img_y[0,:,:,0]
        
        out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))

        out_img_y = out_img_y.astype(np.uint8)

        out_img_cr = cv2.resize(cr, (out_img_y.shape[1], out_img_y.shape[0]), cv2.INTER_CUBIC).astype(np.uint8)
        out_img_cb = cv2.resize(cb, (out_img_y.shape[1], out_img_y.shape[0]), cv2.INTER_CUBIC).astype(np.uint8)

        out_img = np.zeros((out_img_y.shape[0], out_img_y.shape[1], 3)).astype(np.uint8)
        out_img[:, :, 0] = out_img_y
        out_img[:, :, 1] = out_img_cr
        out_img[:, :, 2] = out_img_cb

        out_img = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2BGR)

        #bilinear_img = cv2.resize(frame, (out_img_y.shape[1], out_img_y.shape[0]))
        #out_img[:, 0:out_img.shape[1]//2, :] = bilinear_img[:, 0:out_img.shape[1]//2, :]

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
