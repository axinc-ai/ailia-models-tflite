import sys
import time

import cv2
import numpy as np

import blazehand_utils as but

import os
es = os.path.abspath(__file__).split('/')
util_path = os.path.join('/', *es[:es.index('ailia-models-tflite') + 1], 'util')
sys.path.append(util_path)
from utils import file_abs_path, get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from image_utils import load_image, preprocess_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'person_hand.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'BlazeHand, an on-device real-time hand tracking.',
    IMAGE_PATH,
    SAVE_IMAGE_PATH,
)
parser.add_argument(
    '--hands',
    metavar='NUM_HANDS',
    type=int,
    default=2,
    help='The maximum number of hands tracked (=2 by default)'
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

if args.shape:
    IMAGE_HEIGHT = args.shape
    IMAGE_WIDTH = args.shape

# ======================
# Parameters 2
# ======================
DETECTION_MODEL_NAME = 'blazepalm'
LANDMARK_MODEL_NAME = 'blazehand'
if args.float:
    DETECTOR_MODEL_PATH = f'palm_detection_builtin.tflite'
    LANDMARK_MODEL_PATH = f'hand_landmark_new_256x256_float32.tflite'
else:
    DETECTOR_MODEL_PATH = f'palm_detection_builtin_256_full_integer_quant.tflite'
    LANDMARK_MODEL_PATH = f'hand_landmark_new_256x256_full_integer_quant.tflite'
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

def draw_landmarks(img, points, connections=[], color=(0, 0, 255), size=2):
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), size)
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size+1, color, thickness=cv2.FILLED)


# ======================
# Main functions
# ======================
def recognize_from_image(detector, estimator):
    if args.profile:
        detector.set_profile_mode(True)

    detector.allocate_tensors()
    det_input_details = detector.get_input_details()
    det_output_details = detector.get_output_details()
    estimator.allocate_tensors()
    est_input_details = estimator.get_input_details()
    est_output_details = estimator.get_output_details()

    if args.shape:
        print(f"update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}")
        detector.resize_tensor_input(det_input_details[0]["index"], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        detector.allocate_tensors()

    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = cv2.imread(image_path)
        input_data, scale, pad = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='255',
            gen_input_ailia_tflite=True,
            return_scale_pad=True
        )

        # inference
        logger.info('Start inference...')

        # Palm detection
        det_input = get_input_tensor(input_data, det_input_details, 0)

        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for _ in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                detector.set_tensor(det_input_details[0]['index'], det_input)
                detector.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            detector.set_tensor(det_input_details[0]['index'], det_input)
            detector.invoke()

        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 0)   #1x2944x18 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 1)   #1x2944x1 classificators
        else:
            preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 1)   #1x2944x18 regressors
            preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 0)   #1x2944x1 classificators
        detections = but.detector_postprocess(preds_tf_lite, file_abs_path(__file__, "anchors.npy"))

        # Hand landmark estimation
        presence = [0, 0]  # [left, right]
        if detections[0].size != 0:
            imgs, affines, _ = but.estimator_preprocess(
                src_img, detections, scale, pad
            )

            landmarks = np.zeros((0,63))
            flags = np.zeros((0,1,1,1))
            handedness = np.zeros((0,1,1,1))

            for img_id in range(len(imgs)):
                est_input = get_input_tensor(np.expand_dims(imgs[img_id],axis=0), est_input_details, 0)
                estimator.set_tensor(est_input_details[0]['index'], est_input)
                estimator.invoke()

                landmarks = np.concatenate([landmarks, get_real_tensor(estimator, est_output_details, 2)], 0)
                flags = np.concatenate([flags, get_real_tensor(estimator, est_output_details, 0)], 0)
                handedness = np.concatenate([handedness, get_real_tensor(estimator, est_output_details, 1)], 0)

            normalized_landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
            normalized_landmarks = normalized_landmarks / 256.0
            flags = flags.squeeze((1, 2, 3))
            handedness = handedness.squeeze((1, 2, 3))

            # postprocessing
            landmarks = but.denormalize_landmarks(
                normalized_landmarks, affines
            )
            for i in range(len(flags)):
                landmark, flag, handed = landmarks[i], flags[i], handedness[i]
                if flag > 0.75:
                    if handed > 0.5: # Right handedness when not flipped camera input
                        presence[0] = 1
                    else:
                        presence[1] = 1
                    draw_landmarks(src_img, landmark[:,:2], but.HAND_CONNECTIONS, size=2)

            if presence[0] and presence[1]:
                hand_presence = 'Left and right'
            elif presence[0]:
                hand_presence = 'Right'
            elif presence[1]:
                hand_presence = 'Left'
            else:
                hand_presence = 'No hand'
            logger.info(f'Hand presence: {hand_presence}')

            savepath = get_savepath(args.savepath, image_path)
            logger.info(f'saved at : {savepath}')        
            cv2.imwrite(savepath, src_img)
    logger.info('Script finished successfully.')

    if args.profile:
        print(detector.get_summary())

def recognize_from_video(detector, estimator):
    detector.allocate_tensors()
    det_input_details = detector.get_input_details()
    det_output_details = detector.get_output_details()
    estimator.allocate_tensors()
    est_input_details = estimator.get_input_details()
    est_output_details = estimator.get_output_details()
    num_hands = args.hands
    thresh = 0.5
    tracking = False
    tracked_hands = np.array([0.0] * num_hands)
    rois = [None] * num_hands

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
            normalize_type='255',
            reverse_color_channel=True,
            chan_first=False,
            return_scale_pad=True
        )

        # inference
        # Perform palm detection on 1st frame and if at least 1 hand has low
        # confidence (not detected)
        if np.any(tracked_hands < thresh):
            tracking = False
            # Palm detection
            det_input = get_input_tensor(input_data, det_input_details, 0)
            detector.set_tensor(det_input_details[0]['index'], det_input)
            detector.invoke()
            preds_tf_lite = {}
            if args.float:
                preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 0)   #1x2944x18 regressors
                preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 1)   #1x2944x1 classificators
            else:
                preds_tf_lite[0] = get_real_tensor(detector, det_output_details, 1)   #1x2944x18 regressors
                preds_tf_lite[1] = get_real_tensor(detector, det_output_details, 0)   #1x2944x1 classificators
            detections = but.detector_postprocess(preds_tf_lite, file_abs_path(__file__, "anchors.npy"))
            if detections[0].size > 0:
                tracking = True
                roi_imgs, affines, _ = but.estimator_preprocess(frame, [detections[0][:num_hands]], scale, pad)
        else:
            for i, roi in enumerate(rois):
                xc, yc, scale, theta = roi
                roi_img, affine, _ = but.extract_roi(frame, xc, yc, theta, scale)
                roi_imgs[i] = roi_img[0]
                affines[i] = affine[0]

        # Hand landmark estimation
        presence = [0, 0] # [left, right]
        if tracking:
            landmarks = np.zeros((0,63))
            hand_flags = np.zeros((0,1,1,1))
            handedness = np.zeros((0,1,1,1))

            for img_id in range(len(roi_imgs)):
                est_input = get_input_tensor(np.expand_dims(roi_imgs[img_id],axis=0), est_input_details, 0)
                estimator.set_tensor(est_input_details[0]['index'], est_input)
                estimator.invoke()

                landmarks = np.concatenate([landmarks, get_real_tensor(estimator, est_output_details, 2)], 0)
                hand_flags = np.concatenate([hand_flags, get_real_tensor(estimator, est_output_details, 0)], 0)
                handedness = np.concatenate([handedness, get_real_tensor(estimator, est_output_details, 1)], 0)

            normalized_landmarks = landmarks.reshape((landmarks.shape[0], -1, 3))
            normalized_landmarks = normalized_landmarks / 256.0
            hand_flags = hand_flags.squeeze((1, 2, 3))
            handedness = handedness.squeeze((1, 2, 3))

            # postprocessing
            landmarks = but.denormalize_landmarks(normalized_landmarks, affines)

            tracked_hands[:] = 0
            n_imgs = len(hand_flags)
            for i in range(n_imgs):
                landmark, hand_flag, handed = landmarks[i], hand_flags[i], handedness[i]
                if hand_flag > thresh:
                    if handed > 0.5: # Right handedness when not flipped camera input
                        presence[0] = 1
                    else:
                        presence[1] = 1
                    draw_landmarks(
                        frame, landmark[:, :2], but.HAND_CONNECTIONS, size=2
                    )

                    rois[i] = but.landmarks2roi(normalized_landmarks[i], affines[i])
                tracked_hands[i] = hand_flag

        if presence[0] and presence[1]:
            text = 'Left and right'
        elif presence[0]:
            text = 'Right'
        elif presence[1]:
            text = 'Left'
        else:
            text = 'No hand'

        visual_img = frame
        if args.video == '0': # Flip horizontally if camera
            visual_img = np.ascontiguousarray(frame[:,::-1,:])

        cv2.putText(visual_img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('frame', visual_img)

        # save results
        if writer is not None:
            cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        DETECTOR_MODEL_PATH, DETECTOR_REMOTE_PATH
    )
    check_and_download_models(
        LANDMARK_MODEL_PATH, LANDMARK_REMOTE_PATH
    )

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

    if args.video is not None:
        # video mode
        recognize_from_video(detector, estimator)
    else:
        # image mode
        recognize_from_image(detector, estimator)


if __name__ == '__main__':
    main()
