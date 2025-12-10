import os
import sys
import time
from logging import getLogger  # noqa: E402

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


import webcamera_utils  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from utils import file_abs_path, get_base_parser, get_savepath, update_parser, delegate_obj  # noqa: E402


logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'lenna.png'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 64    # net.get_input_shape()[3]
IMAGE_WIDTH = 64     # net.get_input_shape()[2]
OUTPUT_HEIGHT = 256  # net.get_output_shape()[3]
OUTPUT_WIDTH = 256   # net.get_output.shape()[2]


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Single Image Super-Resolution', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-p', '--padding', action='store_true',
    help=('Instead of resizing input image when loading it, ' +
          ' padding input and output image')
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

if args.shape:
    IMAGE_WIDTH = args.shape
    IMAGE_HEIGHT = args.shape

# ======================
# Parameters 2
# ======================
if args.float:
    MODEL_NAME = 'srresnet.opt_float32'
else:
    MODEL_NAME = 'srresnet.opt_full_integer_quant'
MODEL_PATH = file_abs_path(__file__, f'{MODEL_NAME}.tflite')
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/srresnet/'


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
        input_data = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            rgb=True,
            normalize_type='255',
            gen_input_ailia_tflite=True,
            keep_aspect_ratio=False,
        )
        
        inputs = format_input_tensor(input_data, input_details, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]['index'], inputs)
                interpreter.invoke()
                preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            interpreter.set_tensor(input_details[0]['index'], inputs)
            interpreter.invoke()
            preds_tf_lite = get_output_tensor(interpreter, output_details, 0)

        # postprocessing
        output_img = preds_tf_lite[0]
        logger.info(f"{output_img.shape}")
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img * 255)
    logger.info('Script finished successfully.')


def tiling(interpreter, img):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        print(f"update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        interpreter.allocate_tensors()

    h, w = img.shape[0], img.shape[1]

    padding_w = int((w + IMAGE_WIDTH - 1) / IMAGE_WIDTH) * IMAGE_WIDTH
    padding_h = int((h+IMAGE_HEIGHT-1) / IMAGE_HEIGHT) * IMAGE_HEIGHT
    scale = int(OUTPUT_HEIGHT / IMAGE_HEIGHT)
    output_padding_w = padding_w * scale
    output_padding_h = padding_h * scale
    output_w = w * scale
    output_h = h * scale

    logger.debug(f'input image : {h}x{w}')
    logger.debug(f'output image : {output_w}x{output_h}')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = img[np.newaxis, :, :, :]

    pad_img = np.zeros((1, padding_h, padding_w, 3))
    pad_img[:, 0:h, 0:w, :] = img

    output_pad_img = np.zeros((1, output_padding_h, output_padding_w, 3))
    tile_x = int(padding_w / IMAGE_WIDTH)
    tile_y = int(padding_h / IMAGE_HEIGHT)

    inputs = format_input_tensor(pad_img, input_details, 0)

    if args.float:
        inputs = inputs.astype(np.float32)
    else:
        inputs = inputs.astype(np.int8)

    # Inference 
    start = int(round(time.time() * 1000))
    for y in range(tile_y):
        for x in range(tile_x):
            interpreter.set_tensor(input_details[0]['index'], inputs[:, y*IMAGE_HEIGHT:(y+1)*IMAGE_HEIGHT, x*IMAGE_WIDTH:(x+1)*IMAGE_WIDTH, :])
            interpreter.invoke()
            preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
            
            output_pad_img[
                :,
                y*OUTPUT_HEIGHT:(y+1)*OUTPUT_HEIGHT,
                x*OUTPUT_WIDTH:(x+1)*OUTPUT_WIDTH,
                :
            ] = preds_tf_lite

    end = int(round(time.time() * 1000))
    logger.info(f'ailia processing time {end - start} ms')

    # Postprocessing
    output_img = output_pad_img[0, :output_h, :output_w, :]
    output_img = output_img.astype(np.float32)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

    return output_img


def imread(filename, flags=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        logger.error(f"File does not exist: {filename}")
        sys.exit()
    data = np.fromfile(filename, np.int8)
    img = cv2.imdecode(data, flags)
    return img


def recognize_from_image_tiling(interpreter):

    # processing
    # input image loop
    for image_path in args.input:
        # prepare input data
        # TODO: FIXME: preprocess is different, is it intentionally...?
        logger.info(image_path)
        img = imread(image_path)
        output_img = tiling(interpreter, img)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, output_img * 255)
    logger.info('Script finished successfully.')


def recognize_from_video(interpreter):

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

    frame_shown = False
    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        h, w = frame.shape[0], frame.shape[1]
        frame = frame[h//2:h//2+h//4, w//2:w//2+w//4, :]

        output_img = tiling(interpreter, frame)

        cv2.imshow('frame', output_img)
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

    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
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
        if args.padding:
            recognize_from_image_tiling(interpreter)
        else:
            recognize_from_image(interpreter)


if __name__ == '__main__':
    main()
