import colorsys
import os
import sys

import numpy as np
import cv2


def preprocessing_img(img):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    return img


def load_image(image_path):
    if os.path.isfile(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        print(f'[ERROR] {image_path} not found.')
        sys.exit()
    return preprocessing_img(img)


def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


# def plot_results(detector, img, category, segm_masks=None, logging=True):
#     """
#     :param detector: ailia.Detector, or list of ailia.DetectorObject
#     :param img: ndarray data of image
#     :param category: list of category_name
#     :param segm_masks:
#     :param logging: output log flg
#     :return:
#     """
#     h, w = img.shape[0], img.shape[1]
#     count = detector.get_object_count() if hasattr(detector, 'get_object_count') else len(detector)
#     if logging:
#         print(f'object_count={count}')

#     # prepare color data
#     colors = []
#     for idx in range(count):
#         obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]

#         # print result
#         if logging:
#             print(f'+ idx={idx}')
#             print(
#                 f'  category={obj.category}[ {category[obj.category]} ]'
#             )
#             print(f'  prob={obj.prob}')
#             print(f'  x={obj.x}')
#             print(f'  y={obj.y}')
#             print(f'  w={obj.w}')
#             print(f'  h={obj.h}')

#         color = hsv_to_rgb(256 * obj.category / (len(category) + 1), 255, 255)
#         colors.append(color)

#     # draw segmentation area
#     if segm_masks:
#         for idx in range(count):
#             mask = np.repeat(np.expand_dims(segm_masks[idx], 2), 3, 2).astype(np.bool)
#             color = colors[idx][:3]
#             fill = np.repeat(np.repeat([[color]], img.shape[0], 0), img.shape[1], 1)
#             img[:, :, :3][mask] = img[:, :, :3][mask] * 0.7 + fill[mask] * 0.3

#     # draw bounding box
#     for idx in range(count):
#         obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
#         top_left = (int(w * obj.x), int(h * obj.y))
#         bottom_right = (int(w * (obj.x + obj.w)), int(h * (obj.y + obj.h)))

#         color = colors[idx]
#         cv2.rectangle(img, top_left, bottom_right, color, 4)

#     # draw label
#     for idx in range(count):
#         obj = detector.get_object(idx) if hasattr(detector, 'get_object') else detector[idx]
#         text_position = (int(w * obj.x) + 4, int(h * (obj.y + obj.h) - 8))
#         fontScale = w / 512.0

#         color = colors[idx]
#         cv2.putText(
#             img,
#             category[obj.category],
#             text_position,
#             cv2.FONT_HERSHEY_SIMPLEX,
#             fontScale,
#             color,
#             1
#         )
#     return img


def plot_results(img, boxes, scores, class_ids, classes, normalized_boxes=True,
                 show_label=True, colors=None, logger=None):
    """Plot detections on the image

    Parameters
    ----------
    img : NumPy array
    boxes : NumPy array
        Detected bounding boxes
    scores : NumPy array
        Scores associated to bounding boxes
    class_ids : NumPy array
        Predicted class ID
    classes : list of str
        List of all classes/categories
    normalized_boxes : bool, optional, default=True
        Specify whether the bounding boxes are normalized or not
    show_label : bool, optional, default=True
        Specify whether to draw the labels in addition to the bounding boxes
    colors : NumPy array, optional
        Colors for each class, if None, automatically generate them
    logger : logging.Logger, optional
        If a Logger is given, log results

    Returns
    -------
    res : NumPy array
        Resulting image
    """
    nb_boxes = len(boxes)
    nb_classes = len(classes)
    res = img.copy()
    img_h, img_w, _ = img.shape
    max_side = max(img_h, img_w)
    labels_kwargs = []

    if colors is None:
        hsv_tuples = np.asarray([
            (1 * x / nb_classes, 1., 1.) for x in range(nb_classes)])
        colors = np.apply_along_axis(
            lambda x: colorsys.hsv_to_rgb(*x), 1, hsv_tuples)
        colors = (colors * 255).astype(np.uint8) # 0-255 BGR
        rng = np.random.default_rng(0)
        rng.shuffle(colors)

    if normalized_boxes:
        boxes_norm = boxes
        boxes_px = boxes * np.asarray([[img_w, img_h, img_w, img_h]],
                                      dtype=np.float32)
    else:
        boxes_norm = boxes / np.asarray([[img_w, img_h, img_w, img_h]],
                                        dtype=np.float32)
        boxes_px = boxes

    if logger is not None:
        logger.info(f'Object_count = {nb_boxes}')

    # Draw detection bounding boxes
    for i in range(nb_boxes):
        box = boxes_px[i]
        class_id = class_ids[i]
        score = scores[i]
        x0, y0, x1, y1 = box.astype(np.int64)

        # Log results
        if logger is not None:
            logger.info(f'+ Index = {i}')
            logger.info(
                f'  Category = {class_id} ({classes[class_id]})'
            )
            logger.info(f'  Probability = {score}')
            logger.info(f'  x = {boxes_norm[i][0]}')
            logger.info(f'  y = {boxes_norm[i][1]}')
            logger.info(f'  w = {boxes_norm[i][2] - boxes_norm[i][0]}')
            logger.info(f'  h = {boxes_norm[i][3] - boxes_norm[i][1]}')

        color = colors[class_id]
        l = np.asarray([300, 600, 1200])
        idx = np.searchsorted(l, max_side)
        if idx == len(l):
            bbox_thick = int(round(max_side / 2400))
        else:
            bbox_thick = max(1, idx)
        # Array of pure Python int is needed for cv2.rectangle,
        # not array of NumPy int
        cv2.rectangle(res, (x0, y0), (x1, y1), color.tolist(), bbox_thick)

        if show_label:
            text = f'{classes[class_id]}: {score * 100:.1f}%'
            txt_color = (0, 0, 0) if np.mean(color) > 127.5 else (255, 255, 255)
            txt_bbox_thick = max(
                1, min(int(round(0.75 * bbox_thick)), bbox_thick - 1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max_side / 2048
            txt_size = cv2.getTextSize(text, font, font_scale, txt_bbox_thick)[0]
            margin = max(1, int(round(0.1 * txt_size[1])))

            labels_kwargs.append([
                {
                    'pt1': (x0, y0 + margin),
                    'pt2': (
                        x0 + txt_size[0] + 2*margin, y0 + txt_size[1] + 2*margin
                    ),
                    'color': color.tolist(),
                    'thickness': -1,
                },
                {
                    'text': text,
                    'org': (x0, y0 + txt_size[1] + margin),
                    'fontFace': font,
                    'fontScale': font_scale,
                    'color': txt_color,
                    'thickness': txt_bbox_thick,
                }
            ])

    # Draw labels
    if show_label:
        for kwargs in labels_kwargs:
            cv2.rectangle(res, **kwargs[0])
            cv2.putText(res, **kwargs[1])

    return res


def write_predictions(file_name, boxes, scores, class_ids, normalized_boxes=True,
                      img_size=None, classes=None):
    """Plot detections on the image

    Parameters
    ----------
    file_name : str
        Output filepath
    boxes : NumPy array
        Detected bounding boxes
    scores : NumPy array
        Scores associated to bounding boxes
    class_ids : NumPy array
        Predicted class ID
    normalized_boxes : bool, optional, default=True
        Specify whether the bounding boxes are normalized or not
    img_size : tuple of int, optional
        If the image size (h, w) is given, the coordinates of the bounding
        boxes are multiplied or divided according to normalized_boxes
    classes : list of str, optional
        List of all classes/categories
    """
    boxes_ = boxes # Not a copy
    if img_size is not None:
        h, w = img_size
        a = np.asarray([[w, h, w, h]], dtype=np.float32)
        if normalized_boxes:
            boxes_ = boxes * a
            boxes_ = boxes_.round().astype(np.int64)
        else:
            boxes_ = boxes / a
    elif not normalized_boxes and not np.issubdtype(boxes.dtype, np.integer):
        boxes_ = boxes.round().astype(np.int64)
    count = len(boxes_)

    with open(file_name, 'w') as f:
        for idx in range(count):
            x0, y0, x1, y1 = boxes_[idx]
            x, y = x0, y0
            dw = x1 - x0
            dh = y1 - y0

            class_id = class_ids[idx]
            label = classes[class_id] if classes is not None else class_id
            label = label.replace(' ', '_')
            f.write(f'{label} {scores[idx]} {x} {y} {dw} {dh}\n')
