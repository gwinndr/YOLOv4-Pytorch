import torch

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

    # return the intersection over union value
    return iou
