
import numpy as np
import sys
import cv2
import glob

import tensorflow as tf

import os

# logger
from logging import getLogger  # noqa: E402#


logger = getLogger(__name__)

IMAGE_HEIGHT = 384 # ここ
IMAGE_WIDTH = 384 
IMAGE_HEIGHT_SMALL = 256 #ここ
IMAGE_WIDTH_SMALL = 256
IMAGE_MULTIPLE_OF = 32 #? これなに data_type FP32 と何か関係ある？




# settings
input_model = "saved_model_384x384"
output_model = "./models/output_quant.tflite"


#image_folder = "./test_img"
args = sys.argv
image_folder = args[1] #calibration data directory


# load validation set
img_path = glob.glob(image_folder+"/*")
if len(img_path)==0:
    print("image not found")
    sys.exit(1)


def normalize_image(image, normalize_type='255'):
    """
    Normalize image

    Parameters
    ----------
    image: numpy array
        The image you want to normalize
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': simply dividing by 255.0
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet
        - 'None': no normalization

    Returns
    -------
    normalized_image: numpy array
    """
    if normalize_type == 'None':
        return image
    elif normalize_type == '255':
        return image / 255.0
    elif normalize_type == '127.5':
        return image / 127.5 - 1.0
    elif normalize_type == 'ImageNet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    else:
        logger.error(f'Unknown normalize_type is given: {normalize_type}')
        sys.exit()



def midas_imread(image_path):
    if not os.path.isfile(image_path):
        logger.error(f'{image_path} not found.')
        sys.exit() 

    image = cv2.imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) 

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, 'ImageNet')

    return cv2.resize(
        image, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC
    )



#quantize
def representative_dataset_gen():
  for i in range(len(img_path)):

    logger.info(img_path[i])
    img = midas_imread(img_path[i]) #画像の前処理
    print(img.shape)
    ary = np.asarray(img, dtype=np.float32)
    ary = np.expand_dims(ary, axis=0)
    
    print(ary.shape)
    yield [ary]



converter = tf.lite.TFLiteConverter.from_saved_model(input_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

with open(output_model, 'wb') as o_:
    o_.write(tflite_quant_model)
