import torch
import math

from utilities.constants import *

# predictions_to_bboxes
def predictions_to_bboxes(preds):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts predictions to bboxes
    - Bboxes given in x1y1x2y2 format
    - Predictions must have attributes in the last dimension
    ----------
    """

    device = preds.device

    # Dropping attribute dim of preds to replace with BBOX_N_ELEMS dim
    preds_shape = preds.shape[:-1]
    bboxes = torch.zeros((*preds_shape, BBOX_N_ELEMS), dtype=torch.float32, device=device)

    # Half width and height for offsetting from center point
    half_w = preds[..., YOLO_TW] / 2
    half_h = preds[..., YOLO_TH] / 2

    bboxes[..., BBOX_X1] = preds[..., YOLO_TX] - half_w
    bboxes[..., BBOX_Y1] = preds[..., YOLO_TY] - half_h
    bboxes[..., BBOX_X2] = preds[..., YOLO_TX] + half_w
    bboxes[..., BBOX_Y2] = preds[..., YOLO_TY] + half_h

    return bboxes

# bbox_iou_one_to_many
def bbox_iou_many_to_many(boxes_a, boxes_b):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes ious between all the boxes in a with all the boxes in b
    - Returns iou tensor of shape (n,m)  where n and m are the number of boxes_a and boxes_b respectively
    - For better performance, order such that the length of boxes_a is greater than the length of boxes_b
    ----------
    """

    device = boxes_a.device

    n_boxes_a = len(boxes_a)
    n_boxes_b = len(boxes_b)

    all_ious = torch.zeros((n_boxes_a, n_boxes_b), dtype=torch.float32, device=device)

    for i, box_b in enumerate(boxes_b):
        ious = bbox_iou_one_to_many(box_b, boxes_a)
        all_ious[..., i] = ious

    return all_ious

# bbox_iou_one_to_many
def bbox_iou_one_to_many(bbox_a, bboxes_b):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes IOU between bbox_a and all the bboxes in bboxes_b
    - Returns tensor of ious where each i corresponds to the iou between bbox_a and bboxes_b[i]
    ----------
    """

    bboxes_a = bbox_a.expand_as(bboxes_b)
    ious = bbox_iou(bboxes_a, bboxes_b)

    return ious

# bbox_ciou
# Modified from https://github.com/Zzh-tju/DIoU-darknet/blob/master/src/box.c (box_ciou)
def bbox_ciou(bboxes_a, bboxes_b):
    """
    ----------
    Author: Zzh-tju
    Modified: Damon Gwinn (gwinndr)
    ----------
    - Computes CIOU elementwise between the bboxes in a and the bboxes in b
    - Modified from https://github.com/Zzh-tju/DIoU-darknet/blob/master/src/box.c (box_ciou)
    - https://arxiv.org/abs/1911.08287
    ----------
    """

    device = bboxes_a.device

    shape = bboxes_a.shape[:-1]
    cious = torch.zeros(shape, dtype=torch.float32, device=device)

    ious = bbox_iou(bboxes_a, bboxes_b)
    union = bbox_union_box(bboxes_a, bboxes_b)

    union_w = union[..., BBOX_X2] - union[..., BBOX_X1]
    union_h = union[..., BBOX_Y2] - union[..., BBOX_Y1]

    c = (union_w * union_w) + (union_h * union_h)

    a = bboxes_a
    b = bboxes_b

    a_w = a[..., BBOX_X2] - a[..., BBOX_X1]
    a_h = a[..., BBOX_Y2] - a[..., BBOX_Y1]
    b_w = b[..., BBOX_X2] - b[..., BBOX_X1]
    b_h = b[..., BBOX_Y2] - b[..., BBOX_Y1]

    a_cx = a[..., BBOX_X1] + a_w / 2.0
    a_cy = a[..., BBOX_Y1] + a_h / 2.0
    b_cx = b[..., BBOX_X1] + b_w / 2.0
    b_cy = b[..., BBOX_Y1] + b_h / 2.0

    u = (a_cx - b_cx) * (a_cx - b_cx) + (a_cy - b_cy) * (a_cy - b_cy)
    d = u / c

    ar_b = b_w / b_h;
    ar_a = a_w / a_h;

    ar_term = 4.0 / (math.pi * math.pi) * (torch.atan(ar_b) - torch.atan(ar_a)) * (torch.atan(ar_b) - torch.atan(ar_a));
    alpha = ar_term / (1.0 - ious + ar_term + 0.000001);

    ciou_term = d + alpha * ar_term;

    cious = ious - ciou_term

    # If ciou is nan, set to iou
    nan_mask = torch.isnan(cious)
    cious[nan_mask] = ious[nan_mask]

    # print("  c: %f, u: %f, riou_term: %f\n" % (c, u, ciou_term))

    return cious

# bbox_iou
# Modified from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bbox_iou(bboxes_a, bboxes_b):
    """
    ----------
    Author: Johannes Meyer (meyerjo)
    Modified: Damon Gwinn (gwinndr)
    ----------
    - Computes IOU elementwise between the bboxes in a and the bboxes in b
    - Code modified from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    ----------
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = torch.max(bboxes_a[..., BBOX_X1], bboxes_b[..., BBOX_X1])
    yA = torch.max(bboxes_a[..., BBOX_Y1], bboxes_b[..., BBOX_Y1])
    xB = torch.min(bboxes_a[..., BBOX_X2], bboxes_b[..., BBOX_X2])
    yB = torch.min(bboxes_a[..., BBOX_Y2], bboxes_b[..., BBOX_Y2])

    # compute the area of intersection rectangle
    interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    bboxes_aArea = (bboxes_a[..., BBOX_X2] - bboxes_a[..., BBOX_X1]) * (bboxes_a[..., BBOX_Y2] - bboxes_a[..., BBOX_Y1])
    bboxes_bArea = (bboxes_b[..., BBOX_X2] - bboxes_b[..., BBOX_X1]) * (bboxes_b[..., BBOX_Y2] - bboxes_b[..., BBOX_Y1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (bboxes_aArea + bboxes_bArea - interArea)

    # If iou is not a number, we'll assume it's 0
    nan_mask = torch.isnan(iou)
    iou[nan_mask] = 0.0

    # return the intersection over union value
    return iou

# bbox_union_box
def bbox_union_box(bboxes_a, bboxes_b):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes the union bbox elementwise between the bboxes in a and the bboxes in b
    - Union bbox is the smallest box that encompasses both a and b
    ----------
    """

    device = bboxes_a.device

    shape = (*bboxes_a.shape[:-1], BBOX_N_ELEMS)
    union = torch.zeros(shape, dtype=torch.float32, device=device)

    a_x1 = bboxes_a[..., BBOX_X1]
    a_y1 = bboxes_a[..., BBOX_Y1]
    a_x2 = bboxes_a[..., BBOX_X2]
    a_y2 = bboxes_a[..., BBOX_Y2]

    b_x1 = bboxes_b[..., BBOX_X1]
    b_y1 = bboxes_b[..., BBOX_Y1]
    b_x2 = bboxes_b[..., BBOX_X2]
    b_y2 = bboxes_b[..., BBOX_Y2]

    union[..., BBOX_X1] = torch.min(a_x1, b_x1)
    union[..., BBOX_Y1] = torch.min(a_y1, b_y1)
    union[..., BBOX_X2] = torch.max(a_x2, b_x2)
    union[..., BBOX_Y2] = torch.max(a_y2, b_y2)

    return union
