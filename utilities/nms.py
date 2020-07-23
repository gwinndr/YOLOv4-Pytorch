import torch

from utilities.constants import *

from utilities.detections import filter_detections
from utilities.bboxes import bbox_iou_one_to_many

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

    nms_kind = model.net_block.nms_kind

    if(nms_kind == GREEDY_NMS):
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

        ious = bbox_iou_one_to_many(bbox, bboxes_b)
        thresh_mask = ious > nms_thresh

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
