import sys
import time

import numpy as np
import cv2

# import original modules
sys.path.append('../../util')
import webcamera_utils  # noqa: E402
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from detector_utils import load_image # noqa: E402

# logger
from logging import getLogger  # noqa: E402

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

IMAGE_PATH = 'demo.png'
SAVE_IMAGE_PATH = 'output.png'

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 896

CATEGORY = {
    'BG': 0,
    'road': 1,
    'curb': 2,
    'mark': 3,
}

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'road-segmentation-adas', IMAGE_PATH, SAVE_IMAGE_PATH
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
    MODEL_NAME = 'road-segmentation-adas-0001_float32'
else:
    MODEL_NAME = 'road-segmentation-adas-0001_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/road-segmentation-adas/'


# ======================
# Secondaty Functions
# ======================

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c])
    return image


def draw_result(image, objects):
    for ctgry, color in (
            ('road', (0, 255, 0)),
            ('curb', (0, 0, 255)),
            ('mark', (232, 162, 0))):
        i = CATEGORY[ctgry]
        mask = objects == i
        image = apply_mask(image, mask, color)

    return image


# ======================
# Main functions
# ======================

def preprocess(img):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = np.expand_dims(img, axis=0)
    return img


def post_processing(output, img_size):
    output = np.argmax(output[0], axis=2)

    output = cv2.resize(
        output.astype(np.uint8),
        (img_size[1], img_size[0]), cv2.INTER_NEAREST)

    return output


def predict(img, interpreter):
    h, w = img.shape[:2]

    # initial preprocesses
    img = preprocess(img)

    logger.debug(f'input image shape: {img.shape}')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.float:
        img = img.astype(np.float32)
    else:
        img = img.astype(float) 

    # quantize input data
    inputs = format_input_tensor(img, input_details, 0)

    # feedforward
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()
    preds_tf_lite = get_output_tensor(interpreter, output_details, 0)

    output = preds_tf_lite

    # post processes
    objects = post_processing(output, (h, w))

    return objects


def recognize_from_image(interpreter):

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                objects = predict(img, interpreter)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing time {end - start} ms')
        else:
            objects = predict(img, interpreter)
        
        res_img = draw_result(img, objects)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)


def recognize_from_video(interpreter):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        objects = predict(frame, interpreter)

        # draw segmentation area
        frame = draw_result(frame, objects)

        # show
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()


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
