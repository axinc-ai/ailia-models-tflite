import enum
import sys
import time

import cv2
import numpy as np

import vision_transformer_labels

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser, get_savepath, delegate_obj  # noqa: E402
from model_utils import check_and_download_models, format_input_tensor, get_output_tensor  # noqa: E402
from image_utils import load_image  # noqa: E402
from classifier_utils import plot_results, print_results, write_predictions  # noqa: E402
import webcamera_utils  # noqa: E402

# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'daisy.jpg'
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

MAX_CLASS_COUNT = 3
SLEEP_TIME = 0

TTA_NAMES = ['none', '1_crop']


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser(
    'tf_flowers classification Model', IMAGE_PATH, None
)
parser.add_argument(
    '-w', '--write_prediction',
    action='store_true',
    help='Flag to output the prediction file.'
)
parser.add_argument(
    '--tta', '-t', metavar='TTA',
    default='none', choices=TTA_NAMES,
    help=('tta scheme: ' + ' | '.join(TTA_NAMES) +
          ' (default: none)')
)
args = update_parser(parser)
# for debug
args.tflite = True

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

if args.shape:
    IMAGE_WIDTH = args.shape
    IMAGE_HEIGHT = args.shape

# ======================
# Parameters 2
# ======================
if args.float:
    MODEL_NAME = 'vision_transformer_float'
else:
    MODEL_NAME = 'vision_transformer_quant'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = f'https://storage.googleapis.com/ailia-models-tflite/vision_transformer/'


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
    if args.profile:
        interpreter.set_profile_mode(True)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if args.shape:
        print(f"update input shape {[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]}")
        interpreter.resize_tensor_input(input_details[0]["index"], [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        interpreter.allocate_tensors()
    
    print('Start inference...')

    # image loop
    for image_path in args.input:
        # prepare input data
        input_data = None
        image = load_image(
            image_path,
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            normalize_type='127.5',
            gen_input_ailia_tflite=True,
            bgr_to_rgb=True,
            output_type=np.float32, # float
            tta=args.tta
        )
        if input_data is None:
            input_data = image
        else:
            input_data = np.concatenate([input_data, image])

        # quantize input data
        inputs = format_input_tensor(input_data, input_details, 0)

        # inference
        if args.benchmark:
            print('BENCHMARK mode')
            average_time = 0
            for i in range(args.benchmark_count):
                start = int(round(time.time() * 1000))
                interpreter.set_tensor(input_details[0]['index'], inputs)
                interpreter.invoke()
                preds_tf_lite = get_output_tensor(interpreter, output_details, 0)
                end = int(round(time.time() * 1000))
                average_time = average_time + (end - start)
                print(f'\tailia processing time {end - start} ms')
            print(f'\taverage time {average_time / args.benchmark_count} ms')
        else:
            interpreter.set_tensor(input_details[0]['index'], inputs)
            interpreter.invoke()
            preds_tf_lite = get_output_tensor(interpreter, output_details, 0)

        preds_tf_lite_int8 = interpreter.get_tensor(output_details[0]['index'])

        print(f"=== {image_path} ===")
        print_results([preds_tf_lite[0],preds_tf_lite_int8[0]], vision_transformer_labels.tf_flowers_category)

        # write prediction
        if args.write_prediction:
            savepath = get_savepath(args.savepath, image_path)
            pred_file = '%s.txt' % savepath.rsplit('.', 1)[0]
            write_predictions(pred_file, preds_tf_lite, vision_transformer_labels.tf_flowers_category)

        if args.profile:
            print(interpreter.get_summary())

    print('Script finished successfully.')


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

    while(True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break

        input_image, input_data = webcamera_utils.preprocess_frame(
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='Caffe',
            bgr_to_rgb=False, output_type=np.int8
        )

        # Inference
        inputs = format_input_tensor(input_data, input_details, 0)
        interpreter.set_tensor(input_details[0]['index'], inputs)
        interpreter.invoke()
        preds_tf_lite = get_output_tensor(interpreter, output_details, 0)

        plot_results(
            input_image, preds_tf_lite, vision_transformer_labels.tf_flowers_category
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
        recognize_from_image()


if __name__ == '__main__':
    main()
