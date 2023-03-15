import os
import sys
import time

import cv2
import numpy as np

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import resize_image, load_image, normalize_image  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor #★  # noqa: E402
from utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer, preprocess_frame  # noqa: E402

logger = getLogger(__name__)


# ======================
# Parameters
# ======================
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/midas/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'input_depth.png'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_MULTIPLE_OF = 32


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MiDaS model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-t', '--model_type', default='large', choices=('large', 'small'),
    help='model type: large or small. small can be specified only for version 2.1 model.'
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
    MODEL_NAME = 'midas_float32'
else:
    MODEL_NAME = 'midas_full_integer_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'

REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/midas/'


# ======================
# Main functions
# ======================
def constrain_to_multiple_of(x, min_val=0, max_val=None):
    y = (np.round(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if y < min_val:
        y = (np.ceil(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    return y


def midas_resize(image, target_height, target_width):
    # Resize while keep aspect ratio.
    h, w, c = image.shape
    scale_height = target_height / h
    scale_width = target_width / w
    if scale_width < scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = constrain_to_multiple_of(
        scale_height * h, max_val=target_height
    )
    new_width = constrain_to_multiple_of(
        scale_width * w, max_val=target_width
    )
    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def recognize_from_image(interpreter):

    src = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    h, w, c = src.shape

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        input_data = None
        dtype = np.int8
        if args.float:
            dtype = np.float32
        image = load_image( 
            image_path,
            (IMAGE_HEIGHT,IMAGE_WIDTH),
            normalize_type='ImageNet',
            gen_input_ailia_tflite=True,
            bgr_to_rgb=False,
            output_type=dtype,
            keep_aspect_ratio=False,
        )
        input_data = image

        # quantize input data
        inputs = format_input_tensor(input_data, input_details, 0)

        # inference
        logger.info('Start inference...')
        
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke() #★

        preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
        

        depth_min = preds_tf_lite.min()
        depth_max = preds_tf_lite.max()
        max_val = (2 ** 16) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (preds_tf_lite - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        out = out.transpose(1, 2, 0)
        out, scale, padding = resize_image(out, (h, w), keep_aspect_ratio=False)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out.astype("uint16"))


    logger.info('Script finished successfully.')


def recognize_from_video(interpreter):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = get_capture(args.video)

    # allocate output buffer
    f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    zero_frame = np.zeros((f_h,f_w,3))
    resized_img = midas_resize(zero_frame, IMAGE_HEIGHT, IMAGE_WIDTH)
    save_h, save_w = resized_img.shape[0], resized_img.shape[1]

    output_frame = np.zeros((save_h,save_w*2,3))

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        logger.warning(
            'currently, video results cannot be output correctly...'
        )
        writer = get_writer(args.savepath, save_h, save_w * 2)
    else:
        writer = None

    input_shape_set = False
    frame_shown = False
    while(True):
        ret, frame = capture.read()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('depth', cv2.WND_PROP_VISIBLE) == 0:
            break
        
        frame_mini, scale, padding = resize_image(frame, (save_h, save_w), keep_aspect_ratio=False)
        frame_resize, scale, padding = resize_image(frame, (IMAGE_HEIGHT, IMAGE_WIDTH), keep_aspect_ratio=False)

        # prepare input data
        dtype = np.int8
        if args.float:
            dtype = np.float32
        input_image, input_data = preprocess_frame(
            frame_resize, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='ImageNet',
            bgr_to_rgb=False, output_type=dtype
        )

        inputs = format_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        preds_tf_lite = get_output_tensor(interpreter, output_details, 0)


        # normalize to 16bit
        depth_min = preds_tf_lite.min()
        depth_max = preds_tf_lite.max()
        max_val = (2 ** 16) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (preds_tf_lite - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        # convert to 8bit
        res_img = (out.transpose(1, 2, 0)/256).astype("uint8")
        res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)

        res_img, scale, padding = resize_image(res_img, (save_h, save_w), keep_aspect_ratio=False)

        output_frame[:,save_w:save_w*2,:]=res_img
        output_frame[:,0:save_w,:]=frame_mini
        output_frame = output_frame.astype("uint8")

        cv2.imshow('depth', output_frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(output_frame)

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
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags) #★
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