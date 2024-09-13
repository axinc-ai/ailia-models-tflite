import os
import sys
import time

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


from utils import file_abs_path, get_base_parser, update_parser, delegate_obj
from model_utils import check_and_download_models
from image_utils import load_image
from classifier_utils import plot_results, print_results
import webcamera_utils
import squeezenet_labels


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'clock.jpg'
IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'ImageNet classification Model', IMAGE_PATH, None
)
parser.add_argument(
    '--float', action='store_true',
    help='use float model.'
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
    MODEL_NAME = 'squeezenet_float'
else:
    MODEL_NAME = 'squeezenet_quant'
MODEL_PATH = file_abs_path(__file__, f'{MODEL_NAME}.tflite')
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/squeezenet/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    dtype = np.int8
    if args.float:
        dtype = np.float32
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='None',
        gen_input_ailia_tflite=True,
        bgr_to_rgb=False,
        output_type=dtype
    )

    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id or args.delegate_path is not None:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH,
                                                   memory_mode=args.memory_mode,
                                                   flags=args.flags,
                                                   env_id=args.env_id,
                                                   experimental_delegates=delegate_obj(args.delegate_path))
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        average_time = 0
        for i in range(args.benchmark_count):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])
            end = int(round(time.time() * 1000))
            average_time = average_time + (end - start)
            print(f'\tailia processing time {end - start} ms')
        print(f'\taverage time {average_time / args.benchmark_count} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])

    print_results(preds_tf_lite, squeezenet_labels.imagenet_category)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        if args.flags or args.memory_mode or args.env_id:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH,
                                                   memory_mode=args.memory_mode,
                                                   flags=args.flags,
                                                   env_id=args.env_id)
        else:
            interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video, args.camera_width, args.camera_height)

    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        save_h, save_w = webcamera_utils.calc_adjust_fsize(
            f_h, f_w, IMAGE_HEIGHT, IMAGE_WIDTH
        )
        writer = webcamera_utils.get_writer(args.savepath, save_h, save_w)
    else:
        writer = None

    while True:
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='None',
            bgr_to_rgb=False, output_type=np.uint8
        )

        # Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds_tf_lite = interpreter.get_tensor(output_details[0]['index'])

        plot_results(
            input_image, preds_tf_lite, squeezenet_labels.imagenet_category
        )
        cv2.imshow('frame', input_image)
        time.sleep(SLEEP_TIME)

        # save results
        if writer is not None:
            writer.write(input_image)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    print('Script finished successfully.')


def main():
    # model files check and download
    check_and_download_models(MODEL_PATH, REMOTE_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video()
    else:
        # image mode
        args.input = args.input[0]
        recognize_from_image()


if __name__ == '__main__':
    main()
