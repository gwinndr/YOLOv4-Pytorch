import torch

from .constants import *
from .nms import run_nms_inplace

# extract_detections
def extract_detections(all_preds, yolo_layers, obj_thresh):
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

    # Verifies that postprocessing hyperparameters are the same across all yolo layers
    # success = verify_yolo_hyperparams(yolo_layers)

    yolo_layer = yolo_layers[0]

    all_detections = []

    # Getting detections on a batch by batch basis
    for b, batch_preds in enumerate(all_preds):
        batch_detections = extract_detections_single_image(batch_preds, yolo_layer, obj_thresh)
        all_detections.append(batch_detections)

    # print(all_detections)
    return all_detections

# extract_detections_single_image
def extract_detections_single_image(preds, yolo_layer, obj_thresh):
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
    preds_thresh = preds[thresh_mask]

    detections = None

    if(len(preds_thresh) > 0):

        # Get full class probs by multiplying class score with objectness
        # We need a "fake" dimension here (unsqueeze) to get pytorch to broadcast properly
        preds_obj = preds_thresh[..., YOLO_OBJ].unsqueeze(1)
        class_probs = preds_thresh[..., YOLO_CLASS_START:]
        class_probs *= preds_obj

        # Perform NMS with class probabilities
        run_nms_inplace(preds_thresh, yolo_layer)
        # class_probs = preds_thresh[..., YOLO_CLASS_START:]

        # Filter out predictions without a class prob > thresh
        class_mask = class_probs > obj_thresh
        class_mask = torch.sum(class_mask, dim=1).bool()
        valid_preds = preds_thresh[class_mask]
        n_dets = len(valid_preds)

        # Convert to detection format
        if(n_dets > 0):
            detections = predictions_to_detections(valid_preds)


    # Dummy tensor to represent 0 detections (makes concatenations easier)
    if(detections is None):
        detections = torch.empty((0, DETECTION_N_ELEMS), device=preds.device, dtype=preds.dtype)


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
    best_class_prob, best_class_idx = torch.max(preds[..., YOLO_CLASS_START:], dim=1)
    best_class_prob = best_class_prob
    best_class_idx = best_class_idx

    half_w = preds[..., YOLO_TW] / 2
    half_h = preds[..., YOLO_TH] / 2

    detections = torch.zeros((n_preds, DETECTION_N_ELEMS), dtype=preds.dtype, device=preds.device)
    detections[..., DETECTION_X1] = preds[..., YOLO_TX] - half_w
    detections[..., DETECTION_Y1] = preds[..., YOLO_TY] - half_h
    detections[..., DETECTION_X2] = preds[..., YOLO_TX] + half_w
    detections[..., DETECTION_Y2] = preds[..., YOLO_TY] + half_h
    detections[..., DETECTION_CLASS_IDX] = best_class_idx
    detections[..., DETECTION_CLASS_PROB] = best_class_prob

    return detections
