import sys
import time

import cv2
import numpy as np
from scipy.special import expit

import facemesh_utils as fut

import os
REPO_NAME = "ailia-models-tflite"
path_elements = os.path.abspath(__file__).split('/')
assert REPO_NAME in path_elements, f"""
    It seems you renamed the repo,
    please name it back to `{REPO_NAME}`,
    or update the REPO_NAME varibale to reflect the change you did"""
util_path = os.path.join('/', *path_elements[:path_elements.index(REPO_NAME) + 1], 'util')
sys.path.append(util_path)
from utils import file_abs_path, get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from image_utils import load_image, preprocess_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from facemesh_const import FACEMESH_TESSELATION

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'man.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
LANDMARK_WIDTH = 192


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'Face Mesh, an on-device real-time face recognition.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

if args.shape:
    LANDMARK_WIDTH = args.shape

# ======================
# Parameters 2
# ======================
DETECTION_MODEL_NAME = 'blazeface'
LANDMARK_MODEL_NAME = 'facemesh'
if args.float:
    DETECTOR_MODEL_PATH = f'face_detection_front.tflite'
    LANDMARK_MODEL_PATH = f'face_landmark.tflite'
else:
    DETECTOR_MODEL_PATH = f'face_detection_front_128_full_integer_quant.tflite'
    LANDMARK_MODEL_PATH = f'face_landmark_192_full_integer_quant_uint8.tflite'
DETECTOR_MODEL_PATH = file_abs_path(__file__, DETECTOR_MODEL_PATH)
LANDMARK_MODEL_PATH = file_abs_path(__file__, LANDMARK_MODEL_PATH)
DETECTOR_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/{DETECTION_MODEL_NAME}/'
LANDMARK_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/{LANDMARK_MODEL_NAME}/'


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

def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1, x2, x3, x4), (y1, y2, y3, y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0, 255, 0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0, 0, 0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0, 0, 0), 2)


def draw_landmarks(img, points, color=(0, 0, 255), size=2):
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=cv2.FILLED)


def draw_face_landmarks(
        image,
        landmark_list,
        size = 1):
    for connection in FACEMESH_TESSELATION:
        start_idx = connection[0]
        end_idx = connection[1]
        sx, sy = int(landmark_list[start_idx][0]), int(landmark_list[start_idx][1])
        ex, ey = int(landmark_list[end_idx][0]), int(landmark_list[end_idx][1])
        cv2.line(
            image, (sx,sy), (ex,ey),
            (0, 255, 0), size)

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.tflite:
        detector = tf.lite.Interpreter(model_path=DETECTOR_MODEL_PATH)
        estimator = tf.lite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    else:
        if args.memory_mode or args.flags or args.env_id or args.delegate_path is not None:
            detector = ailia_tflite.Interpreter(model_path=DETECTOR_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
            estimator = ailia_tflite.Interpreter(model_path=LANDMARK_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
        else:
            detector = ailia_tflite.Interpreter(model_path=DETECTOR_MODEL_PATH)
            estimator = ailia_tflite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    detector.allocate_tensors()
    det_input_details = detector.get_input_details()
    det_output_details = detector.get_output_details()
    estimator.allocate_tensors()
    est_input_details = estimator.get_input_details()
    est_output_details = estimator.get_output_details()

    if args.shape:
        logger.info(f"update input shape {[1, LANDMARK_WIDTH, LANDMARK_WIDTH, 3]}")
        estimator.resize_tensor_input(est_input_details[0]["index"], [1, LANDMARK_WIDTH, LANDMARK_WIDTH, 3])
        estimator.allocate_tensors()

    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = cv2.imread(image_path)
        input_data, scale, pad = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='127.5',
            gen_input_ailia_tflite=True,
            return_scale_pad=True
        )

        # inference
        logger.info('Start inference...')

        # Face detection
        det_input = get_input_tensor(input_data, det_input_details, 0)
        detector.set_tensor(det_input_details[0]['index'], det_input)
        detector.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 0)   #1x896x16 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 1)   #1x896x1 classificators
        else:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 1)   #1x896x16 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 0)   #1x896x1 classificators
        detections = fut.detector_postprocess(preds_tf_lite, file_abs_path(__file__, "anchors.npy"))

        if detections[0].size != 0:
            imgs, affines, box = fut.estimator_preprocess(
                src_img[:, :, ::-1], detections, scale, pad, LANDMARK_WIDTH
            )
            draw_roi(src_img, box)
            est_input = get_input_tensor(imgs, est_input_details, 0)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for _ in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                if detections[0].size != 0:
                    estimator.set_tensor(est_input_details[0]['index'], est_input)
                    estimator.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            # Face landmark estimation
            if detections[0].size != 0:
                estimator.set_tensor(est_input_details[0]['index'], est_input)
                estimator.invoke()

        # get result
        preds_tf_lite = {}
        if args.float:
            landmarks = get_real_tensor(estimator, est_output_details, 0)
            confidences = get_real_tensor(estimator, est_output_details, 1)
        else:
            landmarks = get_real_tensor(estimator, est_output_details, 1)
            confidences = get_real_tensor(estimator, est_output_details, 0)
        landmarks = landmarks.squeeze((1, 2))
        confidences = confidences.squeeze((1, 2))
        normalized_landmarks = landmarks / 192.0

        # postprocessing
        landmarks = fut.denormalize_landmarks(
            normalized_landmarks, affines, LANDMARK_WIDTH
        )
        for i in range(len(landmarks)):
            landmark, face_flag = landmarks[i], expit(confidences[i])
            if face_flag > 0:
                #draw_landmarks(src_img, landmark[:, :2], size=1)
                draw_face_landmarks(src_img, landmark[:, :2], size=1)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.tflite:
        detector = tf.lite.Interpreter(model_path=DETECTOR_MODEL_PATH)
        estimator = tf.lite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    else:
        if args.memory_mode or args.flags:
            detector = ailia_tflite.Interpreter(model_path=DETECTOR_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
            estimator = ailia_tflite.Interpreter(model_path=LANDMARK_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            detector = ailia_tflite.Interpreter(model_path=DETECTOR_MODEL_PATH)
            estimator = ailia_tflite.Interpreter(model_path=LANDMARK_MODEL_PATH)
    detector.allocate_tensors()
    det_input_details = detector.get_input_details()
    det_output_details = detector.get_output_details()
    estimator.allocate_tensors()
    est_input_details = estimator.get_input_details()
    est_output_details = estimator.get_output_details()

    capture = get_capture(args.video, args.camera_width, args.camera_height)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_data, scale, pad = preprocess_image(
            frame,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='127.5',
            reverse_color_channel=True,
            chan_first=False,
            return_scale_pad=True
        )

        # inference
        # Face detection
        det_input = get_input_tensor(input_data, det_input_details, 0)
        detector.set_tensor(det_input_details[0]['index'], det_input)
        detector.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 0)   #1x896x16 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 1)   #1x896x1 classificators
        else:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 1)   #1x896x16 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 0)   #1x896x1 classificators
        detections = fut.detector_postprocess(preds_tf_lite, file_abs_path(__file__, "anchors.npy"))

        # Face landmark estimation
        if detections[0].size != 0:
            imgs, affines, box = fut.estimator_preprocess(
                frame[:, :, ::-1], detections, scale, pad, LANDMARK_WIDTH
            )
            draw_roi(frame, box)

            est_input = get_input_tensor(imgs, est_input_details, 0)
            estimator.set_tensor(est_input_details[0]['index'], est_input)
            estimator.invoke()
            preds_tf_lite = {}
            if args.float:
                landmarks = get_real_tensor(estimator, est_output_details, 0)
                confidences = get_real_tensor(estimator, est_output_details, 1)
            else:
                landmarks = get_real_tensor(estimator, est_output_details, 1)
                confidences = get_real_tensor(estimator, est_output_details, 0)
            landmarks = landmarks.squeeze((1, 2))
            confidences = confidences.squeeze((1, 2))
            normalized_landmarks = landmarks / 192.0

            # postprocessing
            landmarks = fut.denormalize_landmarks(
                normalized_landmarks, affines, LANDMARK_WIDTH
            )
            for i in range(len(landmarks)):
                landmark, face_flag = landmarks[i], expit(confidences[i])
                if face_flag > 0:
                    #draw_landmarks(frame, landmark[:, :2], size=1)
                    draw_face_landmarks(frame, landmark[:, :2], size=1)

        visual_img = frame
        if args.video == '0': # Flip horizontally if camera
            visual_img = np.ascontiguousarray(frame[:,::-1,:])

        cv2.imshow('frame', visual_img)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        DETECTOR_MODEL_PATH, DETECTOR_REMOTE_PATH
    )
    check_and_download_models(
        LANDMARK_MODEL_PATH, LANDMARK_REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
