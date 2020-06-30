import torch

from .constants import *
from .nms import run_nms_inplace
from .images import get_letterbox_image_embedding

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

# correct_detections
def correct_detections(detections, img_h, img_w, input_dim=INPUT_DIM_DEFAULT, letterboxed=LETTERBOX_DEFAULT):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Maps detections from the preprocessed input tensor back to the original image
    ----------
    """

    # Getting letterbox information for embedded image
    if(letterboxed):
        embed_h, embed_w, start_y, start_x = get_letterbox_image_embedding(img_h, img_w, input_dim)

    # Setting information such that there is no letterbox (tensor contains the full image)
    else:
        embed_h = input_dim
        embed_w = input_dim
        start_y = 0
        start_x = 0

    dets_x1 = detections[..., DETECTION_X1]
    dets_y1 = detections[..., DETECTION_Y1]
    dets_x2 = detections[..., DETECTION_X2]
    dets_y2 = detections[..., DETECTION_Y2]

    # Move embedded image back to top left
    dets_x1 -= start_x
    dets_y1 -= start_y
    dets_x2 -= start_x
    dets_y2 -= start_y

    # Normalize by the image within the letterbox
    dets_x1 /= embed_w
    dets_y1 /= embed_h
    dets_x2 /= embed_w
    dets_y2 /= embed_h

    # Map back to original image
    dets_x1 *= img_w
    dets_y1 *= img_h
    dets_x2 *= img_w
    dets_y2 *= img_h

    # Clamping dims to lie within the image
    if(CLAMP_DETECTIONS):
        torch.clamp(dets_x1, min=0, max=img_w, out=dets_x1)
        torch.clamp(dets_y1, min=0, max=img_h, out=dets_y1)
        torch.clamp(dets_x2, min=0, max=img_w, out=dets_x2)
        torch.clamp(dets_y2, min=0, max=img_h, out=dets_y2)

    return detections

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
