import os
import sys
import numpy as np
import cv2

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def nms_between_categories(detections,w,h,categories = None,iou_threshold = 0.25):
    #Normally darknet use per class nms
    #But some cases need between class nms
    #https://github.com/opencv/opencv/issues/17111

    # remove overwrapped detection
    det = []
    keep = []
    for idx in range(len(detections)):
        obj = detections[idx]
        is_keep=True
        for idx2 in range(len(det)):
            if not keep[idx2]:
                continue
            box_a = [w*det[idx2].x,h*det[idx2].y,w*(det[idx2].x+det[idx2].w),h*(det[idx2].y+det[idx2].h)]
            box_b = [w*obj.x,h*obj.y,w*(obj.x+obj.w),h*(obj.y+obj.h)]
            iou = bb_intersection_over_union(box_a,box_b)
            if iou >= iou_threshold and (categories==None or ((det[idx2].category in categories) and (obj.category in categories))):
                if det[idx2].prob<=obj.prob:
                    keep[idx2]=False
                else:
                    is_keep=False
        det.append(obj)
        keep.append(is_keep)
    
    det = []
    for idx in range(len(detections)):
        if keep[idx]:
            det.append(detections[idx])

    return det

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def nms(boxes, scores, iou_threshold=0.25, score_threshold=0.4):
    """
    Per class nms
    """
    n_boxes = len(boxes)
    keep = np.full(n_boxes, False)
    remaining = np.full_like(keep, True)
    scores_filt = scores.max(axis=1)
    classes = np.argmax(scores, axis=1)
    indices = np.argsort(-scores_filt)

    for i, idx0 in enumerate(indices):
        if remaining[idx0] and scores_filt[idx0] > score_threshold:
            keep[idx0] = True

            for j in range(i+1, n_boxes):
                idx1 = indices[j]
                if remaining[idx1]:
                    box0 = boxes[idx0]
                    box1 = boxes[idx1]
                    iou = intersection_over_union(box0,box1)
                    if iou >= iou_threshold and classes[idx0] == classes[idx1]:
                        remaining[idx1] = False
            
            remaining[idx0] = False

    return boxes[keep], scores_filt[keep], classes[keep]