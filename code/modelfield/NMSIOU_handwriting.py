import numpy as np

def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i,0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i,0], boxes[order[1:], 1])
        xx2 = np.maximum(boxes[i,0], boxes[order[1:], 2])
        yy2 = np.maximum(boxes[i,0], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        union = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
        iou = intersection / union
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep
