import os
import sys
import colorsys
import random
import time
from logging import getLogger   # noqa: E402

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


from utils import file_abs_path, get_base_parser, update_parser, delegate_obj  # noqa: E402
from webcamera_utils import get_capture, get_writer  # noqa: E402
from image_utils import load_image, preprocess_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402


logger = getLogger(__name__)


# ======================
# Parameters 1
# ======================
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

THRESHOLD = 0.4

MODEL_LIST=["lite0", "lite1", "edgeai", "automl"]

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('EfficientDetLite model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo. (default: '+str(THRESHOLD)+')'
)
parser.add_argument(
    '-m', '--model',
    default='lite0',
    choices=MODEL_LIST,
    help='Select model format'
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
MODEL_NAME = 'efficientdet_lite'
if args.model == 'lite0':
    if args.float:
        MODEL_PATH = f'efficientdet_lite0_float32.tflite'
    else:
        MODEL_PATH = f'efficientdet_lite0_integer_quant.tflite'
    DETECTION_SIZE = 320
elif args.model == 'lite1':
    if args.float:
        MODEL_PATH = f'efficientdet_lite1_float32.tflite'
    else:
        MODEL_PATH = f'efficientdet_lite1_integer_quant.tflite'
    DETECTION_SIZE = 384
elif args.model == 'edgeai':
    MODEL_PATH = f'efficientdet_lite1_relu_ti.tflite'
    DETECTION_SIZE = 384
elif args.model == 'automl':
    if args.float:
        MODEL_PATH = f'efficientdet-lite0_automl.tflite'
    else:
        MODEL_PATH = f'efficientdet-lite0_integer_quant_automl.tflite'
    DETECTION_SIZE = 320

MODEL_PATH = file_abs_path(__file__, MODEL_PATH)
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/{MODEL_NAME}/'


# ======================
# Utils
# ======================
def get_input_tensor(tensor, input_details, idx):
    details = input_details[idx]
    dtype = details['dtype']

    if args.model == 'edgeai' or args.model == 'automl':
        if dtype == np.uint8 or dtype == np.int8:
            input_tensor = tensor.clip(0, 255)
            return input_tensor.astype(dtype)
        else:
            return tensor / 255.0

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
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[i]
        if score<args.threshold:
            continue
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (int(c3[0]), int(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def reverse_padding(bboxes, pad):
    # bboxes = ymin xmin ymax xmax
    # pad = top bottom left right

    scale = 1.0
    if args.model == 'edgeai' or args.model == 'automl':
        scale = DETECTION_SIZE

    bboxes[:,0] = (bboxes[:,0] / scale - pad[0] / DETECTION_SIZE) * (DETECTION_SIZE/(DETECTION_SIZE-pad[0]-pad[1]))
    bboxes[:,2] = (bboxes[:,2] / scale - pad[0] / DETECTION_SIZE) * (DETECTION_SIZE/(DETECTION_SIZE-pad[0]-pad[1]))
    bboxes[:,1] = (bboxes[:,1] / scale - pad[2] / DETECTION_SIZE) * (DETECTION_SIZE/(DETECTION_SIZE-pad[2]-pad[3]))
    bboxes[:,3] = (bboxes[:,3] / scale - pad[2] / DETECTION_SIZE) * (DETECTION_SIZE/(DETECTION_SIZE-pad[2]-pad[3]))

# ======================
# Main functions
# ======================
def recognize_from_image():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH, memory_mode = args.memory_mode, flags = args.flags, env_id = args.env_id, experimental_delegates = delegate_obj(args.delegate_path))
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if args.shape:
        print(f"update input shape {[1, DETECTION_SIZE, DETECTION_SIZE, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, DETECTION_SIZE, DETECTION_SIZE, 3])
        interpreter.allocate_tensors()

    logger.info('Start inference...')
    for image_path in args.input:
        # prepare input data
        logger.info(image_path)
        src_img = cv2.imread(image_path)
        det_w = DETECTION_SIZE
        det_h = DETECTION_SIZE

        # input image is 0-255 RGB image
        input_data, _, pad = load_image(
            image_path,
            (det_h, det_w),
            normalize_type='None',
            bgr_to_rgb=True,
            gen_input_ailia_tflite=True,
            return_scale_pad=True,
            output_type=np.float32
        )

        # inference
        if args.profile:
            interpreter.set_profile_mode(True)
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

        if args.model == "edgeai" or args.model == "automl":
            outputs = get_real_tensor(interpreter, output_details, 0)
            bboxes = outputs[:,:,1:5]
            class_ids = outputs[:,:,6] - 1
            confs = outputs[:,:,5]
            print(outputs)
        else:
            if args.float:
                bboxes = get_real_tensor(interpreter, output_details, 0)
                class_ids = get_real_tensor(interpreter, output_details, 1)
                confs = get_real_tensor(interpreter, output_details, 2)
            else:
                bboxes = get_real_tensor(interpreter, output_details, 0)
                class_ids = get_real_tensor(interpreter, output_details, 2)
                confs = get_real_tensor(interpreter, output_details, 1)

        bboxes = bboxes[0]
        reverse_padding(bboxes, pad)

        class_ids = class_ids[0]
        confs = confs[0]
        src_img = draw_bbox(src_img, bboxes, confs, class_ids)

        logger.info(f'saved at : {args.savepath}')        
        cv2.imwrite(args.savepath, src_img)

        if args.profile:
            print(interpreter.get_summary())

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
            normalize_type='None',
            reverse_color_channel=True,
            chan_first=False,
            return_scale_pad=True
        )

        # inference
        inputs = get_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()

        if args.model == "edgeai" or args.model == "automl":
            outputs = get_real_tensor(interpreter, output_details, 0)
            bboxes = outputs[:,:,1:5]
            class_ids = outputs[:,:,6] - 1
            confs = outputs[:,:,5]
        else:
            if args.float:
                bboxes = get_real_tensor(interpreter, output_details, 0)
                class_ids = get_real_tensor(interpreter, output_details, 1)
                confs = get_real_tensor(interpreter, output_details, 2)
            else:
                bboxes = get_real_tensor(interpreter, output_details, 0)
                class_ids = get_real_tensor(interpreter, output_details, 2)
                confs = get_real_tensor(interpreter, output_details, 1)

        bboxes = bboxes[0]
        reverse_padding(bboxes, pad)

        class_ids = class_ids[0]
        confs = confs[0]

        frame = draw_bbox(frame, bboxes, confs, class_ids)
        cv2.imshow('frame', frame)

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
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        recognize_from_image()


if __name__ == '__main__':
    main()
