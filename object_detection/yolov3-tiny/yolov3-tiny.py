import colorsys
import sys
import random
import time

import cv2
import numpy as np

sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from image_utils import load_image, preprocess_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from nms_utils import nms
from detector_utils import plot_results, write_predictions

# logger
from logging import getLogger   # noqa: E402
logger = getLogger(__name__)


# ======================
# Parameters 1
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
DETECTION_SIZE = 416 # Currently model only accepts this size


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('Yolov3 tiny model', IMAGE_PATH, SAVE_IMAGE_PATH)
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
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
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
MODEL_NAME = 'yolov3-tiny'
if args.float:
    MODEL_PATH = f'yolov3-tiny-416.tflite'
else:
    MODEL_PATH = f'yolov3-tiny-416_full_integer_quant.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/{MODEL_NAME}/'


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

def draw_bbox(image, out_boxes, out_scores, out_classes, classes=COCO_CATEGORY, show_label=True):
    num_boxes = len(out_boxes)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        coor[0] = int(coor[0] * image_w)
        coor[1] = int(coor[1] * image_h)
        coor[2] = int(coor[2] * image_w)
        coor[3] = int(coor[3] * image_h)

        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(c3[0]), int(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        print(f"update input shape {[1, DETECTION_SIZE, DETECTION_SIZE, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, DETECTION_SIZE, DETECTION_SIZE, 3])
        interpreter.allocate_tensors()

    # input image loop
    for image_path in args.input:
        # prepare input data
        logger.debug(f'input image: {image_path}')

        # prepare input data
        src_img = cv2.imread(image_path)
        det_w = DETECTION_SIZE
        det_h = DETECTION_SIZE
        input_data, _, pad = load_image(
            image_path,
            (det_h, det_w),
            normalize_type='255',
            gen_input_ailia_tflite=True,
            return_scale_pad=True,
        )

        # inference
        logger.info('Start inference...')
        if args.benchmark:
            logger.info('BENCHMARK mode')
            average_time = 0
            for _ in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                inputs = get_input_tensor(input_data, input_details, 0)
                interpreter.set_tensor(input_details[0]['index'], inputs)
                interpreter.invoke()
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                logger.info(f'\tailia processing time {end - start} ms')
            logger.info(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            inputs = get_input_tensor(input_data, input_details, 0)
            interpreter.set_tensor(input_details[0]['index'], inputs)
            interpreter.invoke()

        preds_tf_lite = {}
        if args.float:
            preds_tf_lite[0] = get_real_tensor(interpreter, output_details, 0)
            preds_tf_lite[1] = get_real_tensor(interpreter, output_details, 1)
        else:
            preds_tf_lite[0] = get_real_tensor(interpreter, output_details, 1)
            preds_tf_lite[1] = get_real_tensor(interpreter, output_details, 0)

        boxes, pred_conf = filter_boxes(preds_tf_lite[1], preds_tf_lite[0],
            det_w, det_h, pad, score_threshold=args.threshold)
        boxes, scores, classes = nms(boxes[0], pred_conf[0],
            iou_threshold=args.iou, score_threshold=args.threshold)
        src_img = draw_bbox(src_img, boxes, scores, classes)

        savepath = get_savepath(args.savepath, image_path)
        logger.info(f'saved at : {savepath}')        
        cv2.imwrite(savepath, src_img)

        # write prediction
        if args.write_prediction:
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(
                pred_file, boxes, scores, classes,
                normalized_boxes=False, classes=COCO_CATEGORY)

    logger.info('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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

        det_w = DETECTION_SIZE
        det_h = DETECTION_SIZE
        input_data, _, pad = preprocess_image(
            frame,
            (det_h, det_w),
            normalize_type='255',
            reverse_color_channel=True,
            chan_first=False,
            return_scale_pad=True
        )

        # inference
        inputs = get_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        preds_tf_lite = {}
        preds_tf_lite[0] = get_real_tensor(interpreter, output_details, 1)
        preds_tf_lite[1] = get_real_tensor(interpreter, output_details, 0)

        boxes, pred_conf = filter_boxes(preds_tf_lite[1], preds_tf_lite[0],
            det_w, det_h, pad, score_threshold=args.threshold)
        boxes, scores, classes = nms(boxes[0], pred_conf[0],
            iou_threshold=args.iou, score_threshold=args.threshold)

        visual_img = frame
        frame = draw_bbox(frame, boxes, scores, classes)
        cv2.imshow('frame', visual_img)

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    logger.info('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(
        MODEL_PATH, REMOTE_PATH
    )

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
