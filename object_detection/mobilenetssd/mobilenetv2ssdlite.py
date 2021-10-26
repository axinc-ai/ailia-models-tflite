import sys
import time

import cv2

import mobilenetv2ssdlite_utils as mut

# import original modules
sys.path.append('../../util')
from utils import get_base_parser, update_parser  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from image_utils import load_image  # noqa: E402
from detector_utils import plot_results  # noqa: E402
import webcamera_utils  # noqa: E402


# ======================
# Parameters 1
# ======================
IMAGE_PATH = 'couple.jpg'
SAVE_IMAGE_PATH = 'output.png'
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300


# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('MultiBox Detector', IMAGE_PATH, SAVE_IMAGE_PATH)
args = update_parser(parser)

if args.tflite:
    import tensorflow as tf
else:
    import ailia_tflite

# ======================
# Parameters 2
# ======================
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_300_integer_quant_with_postprocess'
MODEL_PATH = f'{MODEL_NAME}.tflite'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models-tflite/mobilenetssd/'


# ======================
# Main functions
# ======================
def recognize_from_image():
    # prepare input data
    org_img = cv2.imread(args.input)
    input_data = load_image(
        args.input,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        normalize_type='127.5',
        gen_input_ailia_tflite=True,
    )

    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    try:
        interpreter.set_num_threads(4)
    except:
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # inference
    print('Start inference...')
    if args.benchmark:
        print('BENCHMARK mode')
        for i in range(5):
            start = int(round(time.time() * 1000))
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            count = interpreter.get_tensor(output_details[3]['index'])[0]
            end = int(round(time.time() * 1000))
            print(f'\tailia processing time {end - start} ms')
    else:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = interpreter.get_tensor(output_details[3]['index'])[0]

    # postprocessing
    mut.postprocessing(org_img, boxes, classes, scores)

    cv2.imwrite(args.savepath, org_img)
    print('Script finished successfully.')


def recognize_from_video():
    # net initialize
    if args.tflite:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=MODEL_PATH)
    try:
        interpreter.set_num_threads(4)
    except:
        pass
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    capture = webcamera_utils.get_capture(args.video)

    # create video writer if savepath is specified as video format
    if args.savepath != SAVE_IMAGE_PATH:
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
            frame, IMAGE_HEIGHT, IMAGE_WIDTH, normalize_type='127.5'
        )

        # inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = interpreter.get_tensor(output_details[3]['index'])[0]

        # postprocessing
        mut.postprocessing(input_image, boxes, classes, scores)

        cv2.imshow('frame', input_image)

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
