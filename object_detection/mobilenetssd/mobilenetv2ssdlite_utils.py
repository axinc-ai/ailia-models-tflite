import numpy as np
import cv2

LABELS = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    '???',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    '???',
    'backpack',
    'umbrella',
    '???',
    '???',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    '???',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    '???',
    'dining table',
    '???',
    '???',
    'toilet',
    '???',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    '???',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
]

SCORE_THRESHOLD = 0.6

def postprocessing(img, boxes, classes, scores):
    h, w = img.shape[:2]
    for box, classidx, score in zip(boxes, classes, scores):
        probability = score
        if probability >= SCORE_THRESHOLD:
            if (not np.isnan(box[0]) and
                not np.isnan(box[1]) and
                not np.isnan(box[2]) and
                not np.isnan(box[3])):
                pass
            else:
                continue
            ymin = int(box[0] * h)
            xmin = int(box[1] * w)
            ymax = int(box[2] * h)
            xmax = int(box[3] * w)
            if ymin > ymax:
                continue
            if xmin > xmax:
                continue
            classnum = int(classidx)
            probability = score
            # print(f'coordinates: ({xmin}, {ymin})-({xmax}, {ymax}).' +
            #     f' class: "{classnum}". probability: {probability:.2f}')
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, f'{LABELS[classnum]}: {probability:.2f}',
                (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)