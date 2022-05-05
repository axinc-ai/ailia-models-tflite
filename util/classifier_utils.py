import os
import sys
import numpy as np
import cv2

MAX_CLASS_COUNT = 3

RECT_WIDTH = 640
RECT_HEIGHT = 20
RECT_MARGIN = 2

def get_top_scores(classifier, top_k = MAX_CLASS_COUNT):
    classifier = classifier[0]
    top_scores = classifier.argsort()[-1 * top_k:][::-1]
    scores = classifier
    return top_scores, scores


def print_results(classifier, labels, top_k = MAX_CLASS_COUNT):
    top_scores, scores = get_top_scores(classifier)

    print('==============================================================')
    print(f'class_count={top_k}')
    for idx in range(top_k):
        print(f'+ idx={idx}')
        print(f'  category={top_scores[idx]}['
              f'{labels[top_scores[idx]]} ]')
        print(f'  prob={scores[top_scores[idx]]}')
        if len(classifier)>=2:
            print(f'  value={classifier[1][top_scores[idx]]}')



def hsv_to_rgb(h, s, v):
    bgr = cv2.cvtColor(
        np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]), 255)


def plot_results(input_image, classifier, labels, logging=True):
    x = RECT_MARGIN
    y = RECT_MARGIN
    w = RECT_WIDTH
    h = RECT_HEIGHT

    top_scores, scores = get_top_scores(classifier)

    if logging:
        print('==============================================================')
        print(f'class_count={MAX_CLASS_COUNT}')
    for idx in range(MAX_CLASS_COUNT):
        if logging:
            print(f'+ idx={idx}')
            print(f'  category={top_scores[idx]}['
                f'{labels[top_scores[idx]]} ]')
            print(f'  prob={scores[top_scores[idx]]}')
        
        text = f'category={top_scores[idx]}[{labels[top_scores[idx]]} ] prob={scores[top_scores[idx]]}'

        color = hsv_to_rgb(256 * top_scores[idx] / (len(labels)+1), 128, 255)

        cv2.rectangle(input_image, (x, y), (x + w, y + h), color, thickness=-1)
        text_position = (x+4, y+int(RECT_HEIGHT/2)+4)

        color = (0,0,0)
        fontScale = 0.5

        cv2.putText(
            input_image,
            text,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            color,
            1
        )

        y=y + h + RECT_MARGIN

def write_predictions(file_name, classifier, labels):
    top_k = 5
    top_scores, scores = get_top_scores(classifier, top_k)
    top_k = min(len(top_scores),top_k)
    with open(file_name, 'w') as f:
        for idx in range(top_k):
            f.write('%s %d %f\n' % (
                labels[top_scores[idx]].replace(' ', '_'),
                top_scores[idx],
                scores[top_scores[idx]]
            ))