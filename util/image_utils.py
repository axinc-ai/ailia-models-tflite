import sys
import os

import cv2
import numpy as np


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
    elif normalize_type == 'Caffe':
        mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
        image = image - mean
        image = np.minimum(image,127)
        image = np.maximum(image,-128)
        return image
    elif normalize_type == 'ImageNet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image


def resize_image(img, out_size, keep_aspect_ratio=True):
    """
    Resizes the input image to the desired size, keeping the original aspect
    ratio or not.

    Parameters
    ----------
    img: NumPy array
        The image to resize.
    out_size: (int, int)  (height, width)
        Resizes the image to the desired size.
    keep_aspect_ratio: bool (default: True)
        If true, resizes while keeping the original aspect ratio. Adds zero-
        padding if necessary.

    Returns
    -------
    resized: NumPy array
        Resized image
    scale: NumPy array
        Resized / original, (scale_height, scale_width)
    padding: NumPy array
        Zero padding, (top, bottom, left, right)
    """
    img_size = img.shape[:2]
    scale = img_size / np.array(out_size)
    padding = np.zeros(4, dtype=int)

    if keep_aspect_ratio:
        scale_long_side = np.max(scale)
        size_new = (img_size / scale_long_side).astype(int)
        padding = out_size - size_new
        padding = np.stack((padding // 2, padding - padding // 2), axis=1).flatten()
        scale[:] = scale_long_side
        resized = cv2.resize(img, (size_new[1], size_new[0]))
        resized = cv2.copyMakeBorder(resized, *padding, cv2.BORDER_CONSTANT, 0)
    else:
        resized = cv2.resize(img, (out_size[1], out_size[0]))

    return resized, scale, padding

def preprocess_image(
        img,
        out_size,
        normalize_type,
        keep_aspect_ratio=True,
        reverse_color_channel=False,
        chan_first=True,
        batch_dim=True,
        output_type=np.float32,
        return_scale_pad=False,
        tta="none"
    ):
    """
    Preprocess the image with various operations.

    Parameters
    ----------
    img: NumPy array
        The image to preprocess.
    out_size: (int, int)  (height, width)
        Resizes the image to the desired size.
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': output range: 0 and 1
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet.
        - 'None': no normalization
    keep_aspect_ratio: bool (default: True)
        If true, resizes while keeping the original aspect ratio. Adds zero-
        padding if necessary.
    reverse_color_channel: bool (default: False)
        If true, reverse the color channels (BGR <-> RGB).
    chan_first: bool (default: True)
        If true, return output with channel dimension first (C, H, W).
    batch_dim: bool (default: True)
        If true, return output with an additional batch dimension,
        e.g. (B, C, H, W).
    output_type: NumPy dtype (default: np.float32)
        If None, no conversion.
    return_scale_pad: bool (default: False)
        If true, returns the scale (scale_height, scale_width) and padding
        (top, bottom, left, right) used when resizing.
        (y_original, x_original) = ((y_resized, x_resized) - 
            (top_padding, left_padding)) * (scale_height, scale_width)

    Returns
    -------
    img_new: NumPy array
        Resized image
    scale: NumPy array (optional)
        Resized / original, (scale_height, scale_width)
    padding: NumPy array (optional)
        Zero-padding, (top, bottom, left, right)
    """
    img_new = normalize_image(img, normalize_type)
    if tta == "1_crop": # imagenet 1 crop mode (256x256 -> 224x224)
        pad = 16
        if img_new.shape[0] < img_new.shape[1]:
            img_new = cv2.resize(img_new, (int(img_new.shape[1]*(out_size[0]+pad*2)/img_new.shape[0]), (out_size[0]+pad*2)))
            img_new = img_new[pad:pad+out_size[0],(img_new.shape[1]-out_size[1])//2:(img_new.shape[1]-out_size[1])//2+out_size[1],:]
        else:
            img_new = cv2.resize(img_new, ((out_size[1]+pad*2), int(img_new.shape[0]*(out_size[1]+pad*2)/img_new.shape[1])))
            img_new = img_new[(img_new.shape[0]-out_size[0])//2:(img_new.shape[0]-out_size[0])//2+out_size[0],pad:pad+out_size[1],:]
        img_new = img_new.copy()

    img_new, scale, padding = resize_image(img_new, out_size,
        keep_aspect_ratio=keep_aspect_ratio)

    if len(img_new.shape) == 3:
        if reverse_color_channel:
            img_new = img_new[:, :, ::-1]
    elif len(img_new.shape) == 2:
        img_new = img_new[..., np.newaxis]
    else:
        raise AssertionError('Input must have at least 2 dimensions.')

    if chan_first:
        img_new = np.moveaxis(img_new, -1, 0)

    if batch_dim:
        img_new = img_new[np.newaxis]  # (batch_size, h, w, channel)

    if output_type is not None:
        img_new = img_new.astype(output_type)

    if return_scale_pad:
        return img_new, scale, padding
    else:
        return img_new

def load_image(
        image_path,
        image_shape,
        rgb=True,
        normalize_type='255',
        gen_input_ailia_tflite=False,
        bgr_to_rgb=True,
        output_type=np.float32,
        keep_aspect_ratio=True,
        return_scale_pad=False,
        tta="none"
    ):
    """
    Loads the image of the given path, performs the necessary preprocessing,
    and returns it.

    Parameters
    ----------
    image_path: string
        The path of image which you want to load.
    image_shape: (int, int)  (height, width)
        Resizes the loaded image to the size required by the model.
    rgb: bool, default=True
        Load as RGB or BGR image when True, as gray scale image when False.
    normalize_type: string
        Normalize type should be chosen from the type below.
        - '255': output range: 0 and 1
        - '127.5': output range : -1 and 1
        - 'ImageNet': normalize by mean and std of ImageNet.
        - 'None': no normalization
    gen_input_ailia_tflite: bool, default=False
        If True, convert the image to the form corresponding to ailia_tflite.
    bgr_to_rgb: bool (default: True)
        Convert image channels BGR to RGB
    output_type: NumPy dtype, default=np.float32
        If None, no conversion.

    Returns
    -------
    image: numpy array
    """
    # rgb == True --> cv2.IMREAD_COLOR
    # rbg == False --> cv2.IMREAD_GRAYSCALE
    if os.path.isfile(image_path):
        image = cv2.imread(image_path, int(rgb))
    else:
        print(f'[ERROR] {image_path} not found.')
        sys.exit()

    res = preprocess_image(image, image_shape, normalize_type,
        keep_aspect_ratio=keep_aspect_ratio, reverse_color_channel=bgr_to_rgb,
        chan_first=False, batch_dim=gen_input_ailia_tflite,
        output_type=output_type, return_scale_pad=return_scale_pad, tta=tta)

    return res


def get_image_shape(image_path):
    tmp = cv2.imread(image_path)
    height, width = tmp.shape[0], tmp.shape[1]
    return height, width


# (ref: https://qiita.com/yasudadesu/items/dd3e74dcc7e8f72bc680)
def draw_texts(img, texts, font_scale=0.7, thickness=2):
    h, w, c = img.shape
    offset_x = 10
    initial_y = 0
    dy = int(img.shape[1] / 15)
    color = (0, 0, 0)  # black

    texts = [texts] if type(texts) == str else texts

    for i, text in enumerate(texts):
        offset_y = initial_y + (i+1)*dy
        cv2.putText(img, text, (offset_x, offset_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)


def draw_result_on_img(img, texts, w_ratio=0.35, h_ratio=0.2, alpha=0.4):
    overlay = img.copy()
    pt1 = (0, 0)
    pt2 = (int(img.shape[1] * w_ratio), int(img.shape[0] * h_ratio))

    mat_color = (200, 200, 200)
    fill = -1
    cv2.rectangle(overlay, pt1, pt2, mat_color, fill)

    mat_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    draw_texts(mat_img, texts)
    return mat_img
