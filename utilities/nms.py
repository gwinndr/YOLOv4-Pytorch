import torch

from utilities.constants import *

from utilities.detections import filter_detections

# run_nms
def run_nms(dets, model, obj_thresh, nms_thresh=NMS_THRESHOLD):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Performs nms inplace on preds according to method and hyperparams specified by yolo_layer
    - Nms will decrease the lower of two class confidence scores of bboxes found to overlap
    - After nms is performed, low confidence detections are filtered out according to given obj_thresh
    - Input must be given in darknet detection format like those returned from extract_detections
    ----------
    """

    # Only need one since nms values must be consistent (see verify_yolo_hyperparams in configs.py)
    yolo_layer = model.get_yolo_layers()[0]

    if(yolo_layer.nms_kind == GREEDY_NMS):
        greedy_nms_inplace(dets, nms_thresh)

    # Filtering out detections without a class prob greater than obj_thresh
    filtered_dets = filter_detections(dets, obj_thresh)

    return filtered_dets

# greedy_nms_inplace
def greedy_nms_inplace(dets, nms_thresh=NMS_THRESHOLD):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Performs greedy nms inplace on given detections
    - Greedy nms forces the lower of two same class probabilities to 0 if the IOU between their respective
      bounding boxes is greater than NMS_THRESHOLD
    - Input must be given in darknet detection format like those returned from extract_detections
    ----------
    """

    bboxes = dets[..., DETECTION_X1:DETECTION_Y2+1]
    class_probs = dets[..., DETECTION_CLASS_START:]

    # Doing a cartesian product via for loops (TODO: Could be optimized to remove the for loop I'm sure)
    for i, bbox in enumerate(bboxes[:-1]):
        bboxes_b = bboxes[i+1:]
        bboxes_a = bbox.expand_as(bboxes_b)

        iou = bbox_iou(bboxes_a, bboxes_b)
        thresh_mask = iou > nms_thresh

        class_b = class_probs[i+1:]
        class_b_thresh = class_b[thresh_mask]
        class_a = class_probs[i].expand_as(class_b_thresh)

        class_zero_a = class_a < class_b_thresh
        class_zero_b = ~class_zero_a

        # masked_select does not share storage with the original tensor
        zero_mask_b = torch.zeros(class_b.size(), device=class_zero_b.device, dtype=class_zero_b.dtype)
        zero_mask_b[thresh_mask] = class_zero_b

        # If two bboxes overlap set the lower class probabilities to 0
        class_a[class_zero_a] = 0
        class_b[zero_mask_b] = 0

    return

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
