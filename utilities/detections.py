import torch

from .constants import *

from utilities.bboxes import predictions_to_bboxes, correct_boxes, is_valid_box

# extract_detections
def extract_detections(all_preds, obj_thresh):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Extracts detections across all batches from raw yolo predictions
    - Each extracted detection contains x1,y1,x2,y2 bbox coordinates along with class index and
      confidence (in that order)
    - Returns a list of detection tensors where each index of the list represents the batch index
    ----------
    """

    all_detections = []

    # Getting detections on a batch by batch basis
    for b, batch_preds in enumerate(all_preds):
        batch_detections = extract_detections_single_image(batch_preds, obj_thresh)
        all_detections.append(batch_detections)

    return all_detections

# correct_detections
def correct_detections(detections, image_info):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Maps detections from the preprocessed input tensor back to the original image
    ----------
    """

    ow = image_info.aug_w
    oh = image_info.aug_h
    nw = image_info.org_w
    nh = image_info.org_h
    offs_x = image_info.aug_pleft
    offs_y = image_info.aug_ptop
    embed_w = image_info.aug_embed_w
    embed_h = image_info.aug_embed_h

    boxes = detections[..., DETECTION_X1:DETECTION_Y2+1]
    boxes = correct_boxes(boxes, ow, oh, nw, nh,
                o_offs_x=offs_x, o_offs_y=offs_y, o_embed_w=embed_w, o_embed_h=embed_h,
                boxes_normalized=False)

    is_valid = is_valid_box(boxes, nw, nh, boxes_normalized=False)
    detections = detections[is_valid]
    boxes = boxes[is_valid]

    detections[..., DETECTION_X1:DETECTION_Y2+1] = boxes

    return detections

# extract_detections_single_image
def extract_detections_single_image(preds, obj_thresh):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Extracts detections from a single image given the raw yolo predictions
    - Each extracted detection contains x1,y1,x2,y2 bbox coordinates along with class index and
      confidence (in that order)
    ----------
    """

    # First filter out low objectness
    thresh_mask = preds[..., YOLO_OBJ] > obj_thresh
    valid_preds = preds[thresh_mask]

    # Converting to detections and filtering out detections with class prob lower than object thresh
    detections = predictions_to_detections(valid_preds)
    detections = filter_detections(detections, obj_thresh)

    return detections

# predictions_to_detections
def predictions_to_detections(preds):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts given preds to detections
    - This is a helper function that converts all given preds from prediction format to
      detection format, use extract_detections instead
    ----------
    """

    n_preds = len(preds)

    # Getting x1y1x2y2 bboxes from predictions
    bboxes = predictions_to_bboxes(preds)

    # Get full class probs by multiplying class score with objectness
    # We need a num values dimension on the object scores (unsqueeze) to get pytorch to broadcast properly
    preds_obj = preds[..., YOLO_OBJ].unsqueeze(1)
    class_scores = preds[..., YOLO_CLASS_START:]
    class_probs = class_scores * preds_obj

    n_classes = class_probs.shape[-1]
    detection_n_elems = DETECTION_CLASS_START + n_classes

    # Creating detections
    detections = torch.zeros((n_preds, detection_n_elems), dtype=preds.dtype, device=preds.device)
    detections[..., DETECTION_X1:DETECTION_Y2+1] = bboxes
    detections[..., DETECTION_CLASS_START:] = class_probs

    return detections

# filter_detections
def filter_detections(dets, obj_thresh):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Filters out detections without a class prob greater than obj_thresh
    - Returns detections with low confidence detections filtered out
    ----------
    """

    class_probs, _ = detections_best_class(dets)

    class_mask = class_probs > obj_thresh
    valid_dets = dets[class_mask]

    return valid_dets

# best_class
def detections_best_class(dets):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Returns the highest confidence class along with its confidence
    ----------
    """

    if(len(dets) == 0):
        confs = torch.empty((0,))
        cl = confs
    else:
        confs, cl = torch.max(dets[..., DETECTION_CLASS_START:], dim=-1)

    return confs, cl
