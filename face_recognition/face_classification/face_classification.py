import os
import sys
import time

import cv2
import numpy as np

import blazeface_utils as but

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

import webcamera_utils  # noqa: E402
from image_utils import load_image  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from utils import get_base_parser, update_parser  # noqa: E402

logger = getLogger(__name__)


# ======================
# PARAMETERS
# ======================

IMAGE_PATH = 'lenna.png'
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_HEIGHT_DET = 128
IMAGE_WIDTH_DET = 128

EMOTION_MAX_CLASS_COUNT = 3
GENDER_MAX_CLASS_COUNT = 2
SLEEP_TIME = 0

EMOTION_CATEGORY = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]
GENDER_CATEGORY = ["female", "male"]

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Face Classificaiton Model (emotion & gender)',
    IMAGE_PATH,
    None,
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
    EMOTION_MODEL_NAME = 'emotion_miniXception_float32'
    GENDER_MODEL_NAME = 'gender_miniXception_float32'
    FACE_MODEL_NAME = 'face_detection_front'
else:
    EMOTION_MODEL_NAME = 'emotion_miniXception_quant'
    GENDER_MODEL_NAME = 'gender_miniXception_quant'
    FACE_MODEL_NAME = 'face_detection_front_128_full_integer_quant'
EMOTION_MODEL_PATH = f'{EMOTION_MODEL_NAME}.tflite'
GENDER_MODEL_PATH = f'{GENDER_MODEL_NAME}.tflite'
FACE_MODEL_PATH = f'{FACE_MODEL_NAME}.tflite'

REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/face_classification/'
FACE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/blazeface/'
FACE_MARGIN = 1.0


def imread(filename, flags=cv2.IMREAD_COLOR):
    if not os.path.isfile(filename):
        logger.error(f"File does not exist: {filename}")
        sys.exit()
    data = np.fromfile(filename, np.int8)
    img = cv2.imdecode(data, flags)
    return img


def crop_blazeface(obj, margin, frame):
    w = frame.shape[1]
    h = frame.shape[0]
    cx = (obj[1] + (obj[3] - obj[1])/2) * w
    cy = (obj[0] + (obj[2] - obj[0])/2) * h
    cw = max((obj[3] - obj[1]) * w * margin, (obj[2] - obj[0]) * h * margin)
    fx = max(cx - cw/2, 0)
    fy = max(cy - cw/2, 0)
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
def recognize_from_image(interpreter_emo, interpreter_gen):
    
    input_details_emo = interpreter_emo.get_input_details()
    output_details_emo = interpreter_emo.get_output_details()
    input_details_gen = interpreter_gen.get_input_details()
    output_details_gen = interpreter_gen.get_output_details()

    # input image loop
    for image_path in args.input:

        # prepare input data
        input_data = None
        dtype = np.float32
        image = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            rgb=True,
            normalize_type='127.5',
            gen_input_ailia_tflite=False,
            output_type=dtype,
        )
        if input_data is None:
            input_data = image
        else:
            input_data = np.concatenate([input_data, image])

        input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2GRAY)
        if not args.float:
            input_data = input_data.astype(float)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=3)
        
        inputs = format_input_tensor(input_data, input_details_emo, 0)

        # inference emotion
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                interpreter_emo.set_tensor(input_details_emo[0]['index'], inputs)
                interpreter_emo.invoke()
                preds_tf_lite = get_output_tensor(interpreter_emo, output_details_emo, 0)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\t[EMOTION MODEL] ailia processing time {end - start} ms'
                )
        else:
            interpreter_emo.set_tensor(input_details_emo[0]['index'], inputs)
            interpreter_emo.invoke()
            preds_tf_lite = get_output_tensor(interpreter_emo, output_details_emo, 0)
                    
        class_count = np.argsort(-preds_tf_lite[0][0][0])
        count = EMOTION_MAX_CLASS_COUNT
        logger.info(f'emotion_class_count={count}')

        # logger.info result
        for idx in range(count):
            logger.info(f'+ idx={idx}')
            info = class_count[idx]
            logger.info(f'  category={info} '
                        f'[ {EMOTION_CATEGORY[info]} ]')
            logger.info(f'  prob={preds_tf_lite[0][0][0][info]}')
        logger.info('')

        # inference gender
        if args.benchmark:
            logger.info('BENCHMARK mode')
            for i in range(5):
                start = int(round(time.time() * 1000))
                interpreter_gen.set_tensor(input_details_gen[0]['index'], inputs)
                interpreter_gen.invoke()
                preds_tf_lite = get_output_tensor(interpreter_gen, output_details_gen, 0)
                end = int(round(time.time() * 1000))
                logger.info(
                    f'\t[EMOTION MODEL] ailia processing time {end - start} ms'
                )
        else:
            interpreter_gen.set_tensor(input_details_gen[0]['index'], inputs)
            interpreter_gen.invoke()
            preds_tf_lite = get_output_tensor(interpreter_gen, output_details_gen, 0)
        
        class_count = np.argsort(-preds_tf_lite[0][0][0])
        count = GENDER_MAX_CLASS_COUNT
        logger.info(f'gender_class_count={count}')

        # logger.info reuslt
        for idx in range(count):
            logger.info(f'+ idx={idx}')
            info = class_count[idx]
            logger.info(f'  category={info} '
                        f'[ {GENDER_CATEGORY[info]} ]')
            logger.info(f'  prob={preds_tf_lite[0][0][0][info]}')
    logger.info('Script finished successfully.')


def recognize_from_video(interpreter_emo, interpreter_gen, interpreter_det):
    
    input_details_emo = interpreter_emo.get_input_details()
    output_details_emo = interpreter_emo.get_output_details()
    input_details_gen = interpreter_gen.get_input_details()
    output_details_gen = interpreter_gen.get_output_details()

    input_details_det = interpreter_det.get_input_details()
    output_details_det = interpreter_det.get_output_details()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        logger.warning('[WARNING] currently video results output feature '
                       'is not supported in this model!')
        # TODO: shape should be debugged!
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT_DET, IMAGE_WIDTH_DET
        )
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

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT_DET, IMAGE_WIDTH_DET, normalize_type='127.5'
        )
        render_image = input_image.copy()

        # inference
        interpreter_det.set_tensor(input_details_det[0]['index'], input_data)
        interpreter_det.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = interpreter_det.get_tensor(output_details_det[0]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter_det.get_tensor(output_details_det[1]['index'])   #1x896x1 classificators
        else:
            preds_tf_lite[0] = interpreter_det.get_tensor(output_details_det[1]['index'])   #1x896x16 regressors
            preds_tf_lite[1] = interpreter_det.get_tensor(output_details_det[0]['index'])   #1x896x1 classificators

        # postprocessing
        detections = but.postprocess(preds_tf_lite)

        for detection in detections:
            for obj in detection:
                # get detected face
                crop_img, top_left, bottom_right = crop_blazeface(
                    obj, FACE_MARGIN, input_image
                )

                if crop_img.shape[0] <= 0 or crop_img.shape[1] <= 0:
                    continue

                dtype = np.float32
                input_image2, input_data = webcamera_utils.preprocess_frame(
                crop_img, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5',
                bgr_to_rgb=False, output_type=dtype
                )

                input_data = np.squeeze(input_data, axis=0)
                input_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2GRAY)
                if not args.float:
                    input_data = input_data.astype(float)
                input_data = np.expand_dims(input_data, axis=0)
                input_data = np.expand_dims(input_data, axis=3)

                inputs = format_input_tensor(input_data, input_details_emo, 0)

                # emotion inference
                interpreter_emo.set_tensor(input_details_emo[0]['index'], inputs)
                interpreter_emo.invoke()
                preds_tf_lite = get_output_tensor(interpreter_emo, output_details_emo, 0)

                class_count = np.argsort(-preds_tf_lite[0][0][0])

                count = EMOTION_MAX_CLASS_COUNT
                logger.info('=' * 80)
                logger.info(f'emotion_class_count={count}')

                # logger.info result
                emotion_text = ""
                for idx in range(count):
                    logger.info(f'+ idx={idx}')
                    info = class_count[idx]
                    logger.info(
                        f'  category={info} ' +
                        f'[ {EMOTION_CATEGORY[info]} ]'
                    )
                    logger.info(f'  prob={preds_tf_lite[0][0][0][info]}')
                    if idx == 0:
                        emotion_text = (f'[ {EMOTION_CATEGORY[info]} ] '
                                        f'prob={preds_tf_lite[0][0][0][info]:.3f}')
                logger.info('')

                # gender inference
                gender_text = ""
                interpreter_gen.set_tensor(input_details_gen[0]['index'], inputs)
                interpreter_gen.invoke()
                preds_tf_lite = get_output_tensor(interpreter_gen, output_details_gen, 0)

                class_count = np.argsort(-preds_tf_lite[0][0][0])

                count = GENDER_MAX_CLASS_COUNT
                # logger.info reuslt
                for idx in range(count):
                    logger.info(f'+ idx={idx}')
                    info = class_count[idx]
                    logger.info(
                        f'  category={info} ' +
                        f'[ {GENDER_CATEGORY[info]} ]'
                    )
                    logger.info(f'  prob={preds_tf_lite[0][0][0][info]}')
                    if idx == 0:
                        gender_text = (f'[ {GENDER_CATEGORY[info]} ] '
                                       f'prob={preds_tf_lite[0][0][0][info]:.3f}')
                logger.info('')

                # display label
                LABEL_WIDTH = 400
                LABEL_HEIGHT = 20
                color = (255, 255, 255)
                cv2.rectangle(render_image, top_left, bottom_right, color, thickness=2)
                cv2.rectangle(
                    render_image,
                    top_left,
                    (top_left[0]+LABEL_WIDTH, top_left[1]+LABEL_HEIGHT),
                    color,
                    thickness=-1,
                )

                text_position = (top_left[0], top_left[1]+LABEL_HEIGHT//2)
                color = (0, 0, 0)
                fontScale = 0.5
                cv2.putText(
                    render_image,
                    emotion_text + " " + gender_text,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    color,
                    1,
                )

        # show result
        cv2.imshow('frame', render_image)
        frame_shown = True
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(render_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        EMOTION_MODEL_PATH, REMOTE_PATH
    )
    check_and_download_models(
        GENDER_MODEL_PATH, REMOTE_PATH
    )
    
    # net emotion initialize
    if args.tflite:
        interpreter_emo = tf.lite.Interpreter(model_path=EMOTION_MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter_emo = ailia_tflite.Interpreter(model_path=EMOTION_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter_emo = ailia_tflite.Interpreter(model_path=EMOTION_MODEL_PATH)
    if args.profile:
        interpreter_emo.set_profile_mode(True)
    interpreter_emo.allocate_tensors()

    # net gender initialize
    if args.tflite:
        interpreter_gen = tf.lite.Interpreter(model_path=GENDER_MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter_gen = ailia_tflite.Interpreter(model_path=GENDER_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter_gen = ailia_tflite.Interpreter(model_path=GENDER_MODEL_PATH)
    if args.profile:
        interpreter_gen.set_profile_mode(True)
    interpreter_gen.allocate_tensors()

    if args.video:
        check_and_download_models(
            FACE_MODEL_PATH, FACE_REMOTE_PATH
        )
        # net detector initialize
        if args.tflite:
            interpreter_det = tf.lite.Interpreter(model_path=FACE_MODEL_PATH)
        else:
            if args.flags or args.memory_mode:
                interpreter_det = ailia_tflite.Interpreter(model_path=FACE_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
            else:
                interpreter_det = ailia_tflite.Interpreter(model_path=FACE_MODEL_PATH)
        interpreter_det.allocate_tensors()

    if args.video is not None:
        # video mode
        recognize_from_video(interpreter_emo, interpreter_gen, interpreter_det)
    else:
        # image mode
        recognize_from_image(interpreter_emo, interpreter_gen)


if __name__ == '__main__':
    main()
