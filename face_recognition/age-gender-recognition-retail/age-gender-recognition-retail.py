import sys
import os
import time
from logging import getLogger

import cv2
import numpy as np

import blazeface_utils as but

sys.path.append('../../util')
import webcamera_utils  # noqa
from image_utils import load_image  # noqa
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa
from utils import get_base_parser, get_savepath, update_parser  # noqa

_this = os.path.dirname(os.path.abspath(__file__))
top_path = os.path.dirname(os.path.dirname(_this))

sys.path.append(os.path.join(top_path, 'face_detection/blazeface'))
from blazeface_utils import crop_blazeface  # noqa

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

FACE_MIN_SCORE_THRESH = 0.5

IMAGE_PATH = 'demo.jpg'##'me.png'##'girl.jpg'##'man.jpg'##'demo.jpg'##'smile.jpg'##'girl_crop.jpg'##'w.jpg'##'smile.jpg'##'who.jpg'##'girl.jpg'##'who.jpg'##'girl.jpg'##'who.jpg'##'girl.jpg'##'w_crop.jpg'##'smile.jpg'##'demo.jpg'##'girl_crop.jpg'##'demo.jpg'##'smile.jpg'##'demo.jpg'##
IMAGE_SIZE = 62

IMAGE_HEIGHT_DET = 128
IMAGE_WIDTH_DET = 128

SAVE_IMAGE_PATH = 'output.png'

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'age-gender-recognition', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-d', '--detector', action='store_true',
    help='Use face detection.'
)
args = update_parser(parser)

detection = args.detector if args.detector else None

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# Parameters 2
# ======================
if args.float:
    MODEL_NAME = 'age-gender-recognition-retail-0013_float32'
else:
    MODEL_NAME = '' ##
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/age-gender-recognition-retail/'

if args.float:
    BLAZEFACE_MODEL_NAME = 'face_detection_front'
else:
    BLAZEFACE_MODEL_NAME = '' ##
BLAZEFACE_MODEL_PATH = f'{BLAZEFACE_MODEL_NAME}.tflite'
BLAZEFACE_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/blazeface/'


# ======================
# Secondaty Functions
# ======================

def crop_blazeface(obj, margin, frame):
    w = frame.shape[1]
    h = frame.shape[0]
    cx = (obj[1] + (obj[3] - obj[1])/2) * w
    cy = (obj[0] + (obj[2] - obj[0])/2) * h
    cw = max((obj[3] - obj[1]) * w * margin, (obj[2] - obj[0]) * h * margin)
    fx = max(cx - cw/2, 0)
    fy = max(cy - cw/1.4, 0)
    fw = min(cw, w-fx)
    fh = min(cw, h-fy)
    top_left = (int(fx), int(fy))
    bottom_right = (int((fx+fw)), int(fy+fh))
    crop_img = frame[
        top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], 0:3
    ]
    return crop_img, top_left, bottom_right

# ======================
# Main functions
# ======================

def recognize_image(interpreter_agender, interpreter_det, input_data, image):

    input_details_det = interpreter_det.get_input_details()
    output_details_det = interpreter_det.get_output_details()

    input_details_agender = interpreter_agender.get_input_details()
    output_details_agender = interpreter_agender.get_output_details()

    # inference
    interpreter_det.set_tensor(input_details_det[0]['index'], input_data)
    interpreter_det.invoke()
    preds_tf_lite = {}
    if args.float:
        preds_tf_lite[0] = interpreter_det.get_tensor(output_details_det[0]['index'])
        preds_tf_lite[1] = interpreter_det.get_tensor(output_details_det[1]['index'])
    else:
        preds_tf_lite[0] = interpreter_det.get_tensor(output_details_det[1]['index'])
        preds_tf_lite[1] = interpreter_det.get_tensor(output_details_det[0]['index'])

    # postprocessing
    detections = but.postprocess(preds_tf_lite)

    # estimate age and gender
    for detection in detections:
        for obj in detection:
            # get detected face
            margin = 1.5
            crop_img, top_left, bottom_right = crop_blazeface(
                obj, margin, image
            )
            if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                continue

            img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = np.transpose(img, [0, 1, 3, 2])

            img = img.astype(np.float32)
            if not args.float:
                img = img.astype(float)

            inputs = format_input_tensor(img, input_details_agender, 0)
            
            # inference
            interpreter_agender.set_tensor(input_details_agender[0]['index'], inputs)
            interpreter_agender.invoke()
            preds_tf_lite_gen = get_output_tensor(interpreter_agender, output_details_agender, 0)
            preds_tf_lite_age = get_output_tensor(interpreter_agender, output_details_agender, 1)

            prob = preds_tf_lite_gen[0][0][0]
            age_conv3 = preds_tf_lite_age[0][0][0][0]

            i = np.argmax(prob)
            gender = 'Female' if i == 0 else 'Male'
            age = round(age_conv3 * 100)

            # display label
            LABEL_WIDTH = bottom_right[1] - top_left[1]
            LABEL_HEIGHT = 20
            if gender == "Male":
                color = (255, 128, 128)
            else:
                color = (128, 128, 255)
            cv2.rectangle(image, top_left, bottom_right, color, thickness=2)
            cv2.rectangle(
                image,
                top_left,
                (top_left[0] + LABEL_WIDTH, top_left[1] + LABEL_HEIGHT),
                color,
                thickness=-1,
            )

            text_position = (top_left[0], top_left[1] + LABEL_HEIGHT // 2)
            color = (0, 0, 0)
            fontScale = 0.5
            cv2.putText(
                image,
                "{} {}".format(gender, age),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale,
                color,
                1,
            )

    return image


def recognize_from_image(interpreter_agender, interpreter_det):

    input_details_agender = interpreter_agender.get_input_details()
    output_details_agender = interpreter_agender.get_output_details()

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        image = cv2.imread(image_path)

        if detection is not None:
            dtype = float
            if args.float:
                dtype = np.float32
            input_data = load_image(
                image_path,
                (IMAGE_HEIGHT_DET, IMAGE_HEIGHT_DET),
                normalize_type='127.5',
                gen_input_ailia_tflite=True,
                output_type=dtype,
                keep_aspect_ratio=False,
            )
            image = recognize_image(interpreter_agender, interpreter_det, input_data, image)

            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')
            cv2.imwrite(savepath, image)
            continue

        # prepare input data
        input_data = None
        dtype = float
        if args.float:
            dtype = np.float32
        img = load_image(
            image_path,
            (IMAGE_SIZE, IMAGE_SIZE),
            normalize_type='None',
            gen_input_ailia_tflite=False,
            output_type=dtype,
            keep_aspect_ratio=False,
        )

        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, [0, 1, 3, 2])

        inputs = format_input_tensor(img, input_details_agender, 0)

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time_estimation = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter_agender.set_tensor(input_details_agender[0]['index'], inputs)
                interpreter_agender.invoke()
                preds_tf_lite_gen = get_output_tensor(interpreter_agender, output_details_agender, 0)
                preds_tf_lite_age = get_output_tensor(interpreter_agender, output_details_agender, 1)
                end = int(round(time.time() * 1000))
                estimation_time = (end - start)

                # Logging
                logger.info(f'\tailia processing estimation time {estimation_time} ms')
                if i != 0:
                    total_time_estimation = total_time_estimation + estimation_time

            logger.info(f'\taverage time estimation {total_time_estimation / (args.benchmark_count - 1)} ms')
        else:
            interpreter_agender.set_tensor(input_details_agender[0]['index'], inputs)
            interpreter_agender.invoke()
            preds_tf_lite_gen = get_output_tensor(interpreter_agender, output_details_agender, 0)
            preds_tf_lite_age = get_output_tensor(interpreter_agender, output_details_agender, 1)

        prob = preds_tf_lite_gen[0][0][0]
        age_conv3 = preds_tf_lite_age[0][0][0][0]

        i = np.argmax(prob)
        logger.info(" gender is: %s (%.2f)" % ('Female' if i == 0 else 'Male', prob[i] * 100))
        logger.info(" age is: %d" % round(age_conv3 * 100))

    logger.info('Script finished successfully.')


def recognize_from_video(interpreter_agender, interpreter_det):
    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
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

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT_DET, IMAGE_WIDTH_DET, normalize_type='127.5'
        )

        frame = recognize_image(interpreter_agender, interpreter_det, input_data, frame)

        # show result
        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')


def main():
    # model files check and download
    logger.info('=== age-gender-recognition model ===')
    check_and_download_models(
        MODEL_PATH, REMOTE_PATH
    )

    # net age, gender initialize
    if args.tflite:
        interpreter_agender = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter_agender = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter_agender = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    if args.profile:
        interpreter_agender.set_profile_mode(True)
    interpreter_agender.allocate_tensors()


    det_model_path = det_remote_path = interpreter_det = None
    if detection:
        # model files check and download
        logger.info('=== face detection model ===')
        det_model_path, det_remote_path = BLAZEFACE_MODEL_PATH, BLAZEFACE_REMOTE_PATH
        check_and_download_models(
            det_model_path, det_remote_path
        )

        # net detector initialize
        if args.tflite:
            interpreter_det = tf.lite.Interpreter(model_path=det_model_path)
        else:
            if args.flags or args.memory_mode:
                interpreter_det = ailia_tflite.Interpreter(model_path=det_model_path, memory_mode = args.memory_mode, flags = args.flags)
            else:
                interpreter_det = ailia_tflite.Interpreter(model_path=det_model_path)
        if args.profile:
            interpreter_det.set_profile_mode(True)
        interpreter_det.allocate_tensors()

    # image mode
    if args.video is not None:
        # video mode
        recognize_from_video(interpreter_agender, interpreter_det)
    else:
        # image mode
        recognize_from_image(interpreter_agender, interpreter_det)


if __name__ == '__main__':
    main()
