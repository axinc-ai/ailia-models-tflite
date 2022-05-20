import numpy as np
import time
import os
import sys
import cv2
import math

from yolox_utils import preproc as preprocess
from yolox_utils import postprocess, filter_predictions

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models, format_input_tensor, \
    get_output_tensor
from detector_utils import plot_results, write_predictions
import webcamera_utils

# logger
from logging import getLogger

logger = getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ======================
# Parameters
# ======================
MODEL_PARAMS = {#'yolox_nano': {'input_shape': [416, 416]},
                'yolox_tiny': {'input_shape': [416, 416]},
                #'yolox_s': {'input_shape': [640, 640]},
                #'yolox_m': {'input_shape': [640, 640]},
                #'yolox_l': {'input_shape': [640, 640]},
                #'yolox_darknet': {'input_shape': [640, 640]},
                #'yolox_x': {'input_shape': [640, 640]}
                }

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'output.jpg'

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

SCORE_THR = 0.4
NMS_THR = 0.45

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('yolox model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-m', '--model_name',
    default='yolox_tiny',
    choices=['yolox_tiny'],
    help='Only yolox_tiny is available currently.'
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
parser.add_argument(
    '-th', '--threshold',
    default=SCORE_THR, type=float,
    help='The detection threshold for yolo. (default: '+str(SCORE_THR)+')'
)
parser.add_argument(
    '-iou', '--iou',
    default=NMS_THR, type=float,
    help='The detection iou for yolo. (default: '+str(NMS_THR)+')'
)
parser.add_argument(
    '-n', '--normal',
    action='store_true',
    help='By default, the optimized model is used, but with this option, ' +
    'you can switch to the normal model for ailia TFLite Runtime 1.1.0'
)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

MODEL_NAME = args.model_name
if args.float:
    if not args.normal:
        stem = f'{MODEL_NAME}'
    else:
        logger.error("float model is supported for opt model only")
        sys.exit(1)
else:
    stem = f'{MODEL_NAME}_full_integer_quant'
if not args.normal:
    stem += '.opt'
MODEL_PATH = f'{stem}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/yolox/'

HEIGHT = MODEL_PARAMS[MODEL_NAME]['input_shape'][0]
WIDTH = MODEL_PARAMS[MODEL_NAME]['input_shape'][1]

if args.shape:
    HEIGHT = args.shape
    WIDTH = args.shape

# ======================
# Utils
# ======================
def compute(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inputs = format_input_tensor(input_data, input_details, 0)
    interpreter.set_tensor(input_details[0]['index'], inputs)
    interpreter.invoke()
    outputs = [get_output_tensor(interpreter, output_details, 0)]

    return outputs

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        if args.normal:
            print(f"update input shape {[1, 3, HEIGHT, WIDTH]}")
            interpreter.resize_tensor_input(input_details[0]["index"], [1, 3, HEIGHT, WIDTH])
        else:
            print(f"update input shape {[1,  HEIGHT, WIDTH, 3]}")
            interpreter.resize_tensor_input(input_details[0]["index"], [1, HEIGHT, WIDTH, 3])
        interpreter.allocate_tensors()

    if args.normal:
        swap = (2, 0, 1)
    else:
        swap = (0, 1, 2)

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')
        raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img, ratio = preprocess(raw_img, (HEIGHT, WIDTH), swap=swap)
        inputs = img[np.newaxis]
        logger.debug(f'input image shape: {raw_img.shape}')

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            total_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                outputs = compute(interpreter, inputs)
                end = int(round(time.time() * 1000))
                if i != 0:
                    total_time = total_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {total_time / (args.benchmark_count-1)} ms')
        else:
            outputs = compute(interpreter, inputs)

        predictions = postprocess(outputs[0], (HEIGHT, WIDTH))[0]
        dets = filter_predictions(predictions, raw_img, ratio, args.iou,
                                  args.threshold)
        if dets is not None:
            final_boxes, final_scores = dets[:, :4], dets[:, 4]
            final_cls_inds = dets[:, 5].astype(np.uint32)
            res_img = plot_results(
                raw_img, final_boxes, final_scores, final_cls_inds,
                COCO_CATEGORY, normalized_boxes=False, logger=logger)
        else:
            final_boxes = np.zeros((0))
            final_scores = np.zeros((0))
            final_cls_inds = np.zeros((0))
            res_img = raw_img

        # plot result
        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

        # write prediction
        if args.write_prediction:
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(
                pred_file, final_boxes, final_scores, final_cls_inds,
                normalized_boxes=False, classes=COCO_CATEGORY)

    logger.info('Script finished successfully.')

def recognize_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = f_h, f_w
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    if args.write_prediction:
        frame_count = 0
        frame_digit = int(math.log10(capture.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
        video_name = os.path.splitext(os.path.basename(args.video))[0]

    if args.normal:
        swap = (2, 0, 1)
    else:
        swap = (0, 1, 2)

    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        img, ratio = preprocess(frame, (HEIGHT, WIDTH), swap=swap)
        inputs = img[np.newaxis]
        outputs = compute(interpreter, inputs)
        predictions = postprocess(outputs[0], (HEIGHT, WIDTH))[0]
        dets = filter_predictions(predictions, frame, ratio, args.iou,
                                  args.threshold)

        visual_img = frame
        res_img = frame
        if dets is not None:
            final_boxes, final_scores = dets[:, :4], dets[:, 4]
            final_cls_inds = dets[:, 5].astype(np.uint32)
            visual_img = plot_results(
                frame, final_boxes, final_scores, final_cls_inds,
                COCO_CATEGORY, normalized_boxes=False, logger=logger)

            if args.video == '0': # Flip horizontally if camera
                visual_img = np.ascontiguousarray(frame[:,::-1,:])
                boxes_vis = final_boxes.copy()
                boxes_vis[:, [0, 2]] = frame.shape[1] - final_boxes[:, [2, 0]]
                visual_img = plot_results(
                    visual_img, boxes_vis, final_scores, final_cls_inds,
                    COCO_CATEGORY, normalized_boxes=False, logger=logger)

        cv2.imshow('frame', visual_img)

        # save results
        if writer is not None:
            writer.write(res_img)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, video_name, post_fix = '_%s' % (str(frame_count).zfill(frame_digit) + '_res'), ext='.png')
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(
                pred_file, final_boxes, final_scores, final_cls_inds,
                normalized_boxes=False, classes=COCO_CATEGORY)
            frame_count += 1

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
