import sys
import time

import numpy as np
import cv2

import const


import os
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

from utils import file_abs_path, get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image as load_image_img, preprocess_image  # noqa: E402
from detector_utils import load_image as load_image_det  # noqa: E402
from nms_utils import nms # noqa: E402
import webcamera_utils  # noqa: E402
from pose_resnet_util import compute, keep_aspect  # noqa: E402

from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters
# ======================
IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.png'

COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]
THRESHOLD = 0.4
IOU = 0.45
POSE_THRESHOLD = 0.1


# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser(
    'Simple Baseline for Pose Estimation', IMAGE_PATH, SAVE_IMAGE_PATH,
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: '+str(THRESHOLD)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo. (default: '+str(IOU)+')'
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

if args.shape:
    DETECTION_SIZE = args.shape


# ======================
# Parameters 2
# ======================
if args.float:
    POSE_MODEL_NAME = 'pose_resnet_50_256x192_float32'
else:
    POSE_MODEL_NAME = 'pose_resnet_50_256x192_int8'
POSE_MODEL_PATH = file_abs_path(__file__, f'{POSE_MODEL_NAME}.tflite')
POSE_REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/pose_resnet/'

if args.float:
    DETECT_MODEL_NAME = 'yolov3-tiny-416'
else:
    DETECT_MODEL_NAME = 'yolov3-tiny-416_full_integer_quant'
DETECT_MODEL_PATH = file_abs_path(__file__, f'{DETECT_MODEL_NAME}.tflite')
DETECT_REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/yolov3-tiny/'


# ======================
# Display result
# ======================
def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def line(input_img, person, point1, point2):
    threshold = POSE_THRESHOLD
    if person.points[point1].score > threshold and\
       person.points[point2].score > threshold:
        color = hsv_to_rgb(255*point1/const.POSE_KEYPOINT_CNT, 255, 255)

        x1 = int(input_img.shape[1] * person.points[point1].x)
        y1 = int(input_img.shape[0] * person.points[point1].y)
        x2 = int(input_img.shape[1] * person.points[point2].x)
        y2 = int(input_img.shape[0] * person.points[point2].y)
        cv2.line(input_img, (x1, y1), (x2, y2), color, 5)


def display_result(input_img, person):
    line(input_img, person, const.POSE_KEYPOINT_NOSE,
         const.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, const.POSE_KEYPOINT_SHOULDER_LEFT,
         const.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, const.POSE_KEYPOINT_SHOULDER_RIGHT,
         const.POSE_KEYPOINT_SHOULDER_CENTER)

    line(input_img, person, const.POSE_KEYPOINT_EYE_LEFT,
         const.POSE_KEYPOINT_NOSE)
    line(input_img, person, const.POSE_KEYPOINT_EYE_RIGHT,
         const.POSE_KEYPOINT_NOSE)
    line(input_img, person, const.POSE_KEYPOINT_EAR_LEFT,
         const.POSE_KEYPOINT_EYE_LEFT)
    line(input_img, person, const.POSE_KEYPOINT_EAR_RIGHT,
         const.POSE_KEYPOINT_EYE_RIGHT)

    line(input_img, person, const.POSE_KEYPOINT_ELBOW_LEFT,
         const.POSE_KEYPOINT_SHOULDER_LEFT)
    line(input_img, person, const.POSE_KEYPOINT_ELBOW_RIGHT,
         const.POSE_KEYPOINT_SHOULDER_RIGHT)
    line(input_img, person, const.POSE_KEYPOINT_WRIST_LEFT,
         const.POSE_KEYPOINT_ELBOW_LEFT)
    line(input_img, person, const.POSE_KEYPOINT_WRIST_RIGHT,
         const.POSE_KEYPOINT_ELBOW_RIGHT)

    line(input_img, person, const.POSE_KEYPOINT_BODY_CENTER,
         const.POSE_KEYPOINT_SHOULDER_CENTER)
    line(input_img, person, const.POSE_KEYPOINT_HIP_LEFT,
         const.POSE_KEYPOINT_BODY_CENTER)
    line(input_img, person, const.POSE_KEYPOINT_HIP_RIGHT,
         const.POSE_KEYPOINT_BODY_CENTER)

    line(input_img, person, const.POSE_KEYPOINT_KNEE_LEFT,
         const.POSE_KEYPOINT_HIP_LEFT)
    line(input_img, person, const.POSE_KEYPOINT_ANKLE_LEFT,
         const.POSE_KEYPOINT_KNEE_LEFT)
    line(input_img, person, const.POSE_KEYPOINT_KNEE_RIGHT,
         const.POSE_KEYPOINT_HIP_RIGHT)
    line(input_img, person, const.POSE_KEYPOINT_ANKLE_RIGHT,
         const.POSE_KEYPOINT_KNEE_RIGHT)


def pose_estimation(boxes, scores, classes, interpreter_pose, img):
    dtype = np.int8
    if args.float:
        dtype = np.float32
    
    pose_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[0], img.shape[1]
    count = len(classes)

    pose_detections = []
    for idx in range(count):
        top_left = (int(w*boxes[idx][0]), int(h*boxes[idx][1]))
        bottom_right = (int(w*boxes[idx][2]), int(h*boxes[idx][3]))
        CATEGORY_PERSON = 0
        if classes[idx] != CATEGORY_PERSON:
            pose_detections.append(None)
            continue
        px1, py1, px2, py2 = keep_aspect(
            top_left, bottom_right, pose_img
        )
        crop_img = pose_img[py1:py2, px1:px2, :]

        offset_x = px1/img.shape[1]
        offset_y = py1/img.shape[0]
        scale_x = crop_img.shape[1]/img.shape[1]
        scale_y = crop_img.shape[0]/img.shape[0]
        detections = compute(
            interpreter_pose, crop_img, offset_x, offset_y, scale_x, scale_y, dtype
        )
        pose_detections.append(detections)
    return pose_detections


def plot_results(boxes, scores, classes, img, category, pose_detections, logging=True):
    h, w = img.shape[0], img.shape[1]
    count = len(classes)
    if logging:
        logger.info(f'object_count={count}')

    for idx in range(count):
        if logging:
            logger.info(f'+ idx={idx}')
            logger.info(f'  category={classes[idx]}[ {category[classes[idx]]} ]')
            logger.info(f'  prob={scores[idx]}')
            logger.info(f'  x1={boxes[idx][0]}')
            logger.info(f'  y1={boxes[idx][1]}')
            logger.info(f'  x2={boxes[idx][2]}')
            logger.info(f'  y2={boxes[idx][3]}')
        top_left = (int(w*boxes[idx][0]), int(h*boxes[idx][1]))
        bottom_right = (int(w*boxes[idx][2]), int(h*boxes[idx][3]))
        text_position = (int(w*boxes[idx][0])+4, int(h*boxes[idx][3]-8))

        # update image
        color = hsv_to_rgb(256 * classes[idx] / len(category), 255, 255)
        fontScale = w / 512.0
        cv2.rectangle(img, top_left, bottom_right, color, 4)

        cv2.putText(
            img,
            category[classes[idx]],
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        CATEGORY_PERSON = 0
        if classes[idx] != CATEGORY_PERSON:
            continue

        # pose detection
        px1, py1, px2, py2 = keep_aspect(
            top_left, bottom_right, img
        )
        detections = pose_detections[idx]
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 1)
        display_result(img, detections)

    return img


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


def filter_boxes(box_xywh, scores, w, h, padding, score_threshold=0.4):
    scores_max = np.max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = box_xywh[mask]
    pred_conf = scores[mask]
    class_boxes = class_boxes.reshape((scores.shape[0], -1, class_boxes.shape[-1]))
    pred_conf = pred_conf.reshape((scores.shape[0], -1, pred_conf.shape[-1]))

    xy_origin = padding[[2, 0]][np.newaxis, np.newaxis].astype(np.float32)
    box_xy = class_boxes[..., :2] - xy_origin
    box_wh = class_boxes[..., 2:]

    input_shape = np.array([h, w], dtype=np.float32)
    input_shape -= padding.reshape((2, 2)).sum(axis=1)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = np.concatenate([
        box_mins[..., 1:2],  # x_min
        box_mins[..., 0:1],  # y_min
        box_maxes[..., 1:2],  # x_max
        box_maxes[..., 0:1]  # y_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


# ======================
# Main functions
# ======================
def recognize_from_image(interpreter_pose, interpreter_detect):

    input_details = interpreter_detect.get_input_details()
    output_details = interpreter_detect.get_output_details()

    if args.shape:
        print(f"update input shape {[1, DETECTION_SIZE, DETECTION_SIZE, 3]}")
        interpreter_detect.resize_tensor_input(input_details[0]["index"], [1, DETECTION_SIZE, DETECTION_SIZE, 3])
        interpreter_detect.allocate_tensors()

    # input image loop
    for image_path in args.input:
        
        logger.info(image_path)
        det_w = 416
        det_h = 416
        input_data, _, pad = load_image_img(
            image_path,
            (det_h, det_w),
            normalize_type='255',
            gen_input_ailia_tflite=True,
            return_scale_pad=True,
        )

        # object detection
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for _ in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                inputs = get_input_tensor(input_data, input_details, 0)
                interpreter_detect.set_tensor(input_details[0]['index'], inputs)
                interpreter_detect.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            inputs = get_input_tensor(input_data, input_details, 0)
            interpreter_detect.set_tensor(input_details[0]['index'], inputs)
            interpreter_detect.invoke()

        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(interpreter_detect, output_details, 0)
            preds_tf_lite[1] = get_real_tensor(interpreter_detect, output_details, 1)
        else:
            preds_tf_lite[0] = get_real_tensor(interpreter_detect, output_details, 1)
            preds_tf_lite[1] = get_real_tensor(interpreter_detect, output_details, 0)

        boxes, pred_conf = filter_boxes(preds_tf_lite[1], preds_tf_lite[0],
            det_w, det_h, pad, score_threshold=args.threshold)
        boxes, scores, classes = nms(boxes[0], pred_conf[0],
            iou_threshold=args.iou, score_threshold=args.threshold)

        img = load_image_det(
            image_path
        )

        logger.info('Start inference...')
        # pose estimation
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                pose_detections = pose_estimation(boxes, scores, classes, interpreter_pose, img)
                end = int(round(time.time() * 1000))
                logger.info(f'\tailia processing detection time {end - start} ms')
                if i != 0:
                    total_time = total_time + (end - start)
            logger.info(f'\taverage detection time {total_time / (args.benchmark_count-1)} ms')
        else:
            pose_detections = pose_estimation(boxes, scores, classes, interpreter_pose, img)

        # plot result
        res_img = plot_results(boxes, scores, classes, img, COCO_CATEGORY, pose_detections)
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)
    logger.info('Script finished successfully.')


def recognize_from_video(interpreter_pose, interpreter_detect):

    input_details = interpreter_detect.get_input_details()
    output_details = interpreter_detect.get_output_details()

    capture = webcamera_utils.get_capture(args.video, args.camera_width, args.camera_height)
    if args.savepath != SAVE_IMAGE_PATH:
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

        det_w = 416
        det_h = 416
        input_data, _, pad = preprocess_image(
            frame,
            (det_h, det_w),
            normalize_type='255',
            reverse_color_channel=True,
            chan_first=False,
            return_scale_pad=True
        )

        # object detection
        logger.info('Start inference...')
        inputs = get_input_tensor(input_data, input_details, 0)
        interpreter_detect.set_tensor(input_details[0]['index'], inputs)
        interpreter_detect.invoke()
        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(interpreter_detect, output_details, 0)
            preds_tf_lite[1] = get_real_tensor(interpreter_detect, output_details, 1)
        else:
            preds_tf_lite[0] = get_real_tensor(interpreter_detect, output_details, 1)
            preds_tf_lite[1] = get_real_tensor(interpreter_detect, output_details, 0)

        boxes, pred_conf = filter_boxes(preds_tf_lite[1], preds_tf_lite[0],
            det_w, det_h, pad, score_threshold=args.threshold)
        boxes, scores, classes = nms(boxes[0], pred_conf[0],
            iou_threshold=args.iou, score_threshold=args.threshold)

        logger.info('Start inference...')
        # pose estimation
        pose_detections = pose_estimation(boxes, scores, classes, interpreter_pose, frame) 
        
        # plot result
        res_img = plot_results(boxes, scores, classes, frame, COCO_CATEGORY, pose_detections)

        cv2.imshow('frame', res_img)
        frame_shown = True
        # save results
        if writer is not None:
            writer.write(res_img)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(POSE_MODEL_PATH, POSE_REMOTE_PATH)
    check_and_download_models(DETECT_MODEL_PATH, DETECT_REMOTE_PATH)

    # net initialize
    if args.tflite:
        interpreter_pose = tf.lite.Interpreter(model_path=POSE_MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter_pose = ailia_tflite.Interpreter(model_path=POSE_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
        else:
            interpreter_pose = ailia_tflite.Interpreter(model_path=POSE_MODEL_PATH)
    interpreter_pose.allocate_tensors()

    # net initialize
    if args.tflite:
        interpreter_detect = tf.lite.Interpreter(model_path=DETECT_MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter_detect = ailia_tflite.Interpreter(model_path=DETECT_MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
        else:
            interpreter_detect = ailia_tflite.Interpreter(model_path=DETECT_MODEL_PATH)
    interpreter_detect.allocate_tensors()


    if args.video is not None:
        # video mode
        recognize_from_video(interpreter_pose, interpreter_detect)
    else:
        # image mode
        recognize_from_image(interpreter_pose, interpreter_detect)


if __name__ == '__main__':
    main()
