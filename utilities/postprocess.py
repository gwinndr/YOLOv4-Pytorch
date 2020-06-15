import torch
import random

from utilities.constants import *

# extract_detections
def extract_detections(all_preds, yolo_layers, obj_thresh=YOLO_OBJ_THRESH):
    # Verifies that postprocessing hyperparameters are the same across all yolo layers
    # success = verify_yolo_hyperparams(yolo_layers)


    yolo_layer = yolo_layers[0]

    all_detections = []

    # Getting detections on a batch by batch basis
    for b, batch_preds in enumerate(all_preds):
        batch_detections = extract_detections_batch(batch_preds, yolo_layer, obj_thresh=obj_thresh)
        all_detections.append(batch_detections)

    # print(all_detections)
    return all_detections

# get_bbox_image
def get_bbox_image(detections, image, class_names, verbose_output=False):
    image = image.copy()

    bboxes = detections[..., DETECTION_X1:DETECTION_Y2+1].cpu().numpy()
    classes = detections[..., DETECTION_CLASS_IDX].cpu().type(torch.int32).numpy()
    class_confs = detections[..., DETECTION_CLASS_PROB].cpu().numpy()

    for i in range(len(detections)):
        x1 = int(round(bboxes[i, BBOX_X1]))
        y1 = int(round(bboxes[i, BBOX_Y1]))
        x2 = int(round(bboxes[i, BBOX_X2]))
        y2 = int(round(bboxes[i, BBOX_Y2]))


        p1 = (x1, y1)
        p2 = (x2, y2)
        class_name = class_names[classes[i]]
        class_conf = class_confs[i]

        if(verbose_output):
            print("Class:", class_name)
            print("Conf: %.2f" % class_conf)
            print("Left_x:", x1)
            print("Left_y:", y1)
            print("Width:", x2-x1)
            print("Height:", y2-y1)
            print("")

        label = "%s %.2f" % (class_name, class_conf)

        color = random.choice(BBOX_COLORS)
        cv2.rectangle(image, p1, p2, color, BBOX_RECT_THICKNESS)

        t_dims = cv2.getTextSize(label, BBOX_FONT, BBOX_FONT_SCALE, BBOX_FONT_THICKNESS)[0]

        label_rect_x1 = x1
        label_rect_y1 = y1 - t_dims[CV2_TEXT_SIZE_H] - (BBOX_TEXT_TOP_PAD + BBOX_TEXT_BOT_PAD)
        label_rect_x2 = x1 + t_dims[CV2_TEXT_SIZE_W] + (BBOX_TEXT_LEFT_PAD + BBOX_TEXT_RIGHT_PAD)
        label_rect_y2 = y1

        label_rect_p1 = (label_rect_x1, label_rect_y1)
        label_rect_p2 = (label_rect_x2, label_rect_y2)

        cv2.rectangle(image, label_rect_p1, label_rect_p2, color, CV2_RECT_FILL)

        label_text_p = (x1 + BBOX_TEXT_LEFT_PAD, y1 - BBOX_TEXT_BOT_PAD)
        cv2.putText(image, label, label_text_p, BBOX_FONT, BBOX_FONT_SCALE, COLOR_BLACK, BBOX_FONT_THICKNESS);

    return image

# bbox_letterbox_to_image
def bbox_letterbox_to_image(detections, img_h, img_w, letter_dim):
    n_detections = len(detections)

    ratio = img_w / img_h

    if(img_w >= img_h):
        new_w = letter_dim
        new_h = round(new_w / ratio)
    else:
        new_h = letter_dim
        new_w = round(new_h * ratio)

    start_h = (letter_dim - new_h) // 2
    start_w = (letter_dim - new_w) // 2

    # Move embedded image back to top left
    detections[..., DETECTION_X1] -= start_w
    detections[..., DETECTION_Y1] -= start_h
    detections[..., DETECTION_X2] -= start_w
    detections[..., DETECTION_Y2] -= start_h

    # Normalize by the image within
    detections[..., DETECTION_X1] /= new_w
    detections[..., DETECTION_Y1] /= new_h
    detections[..., DETECTION_X2] /= new_w
    detections[..., DETECTION_Y2] /= new_h

    # Map back to original image
    detections[..., DETECTION_X1] *= img_w
    detections[..., DETECTION_Y1] *= img_h
    detections[..., DETECTION_X2] *= img_w
    detections[..., DETECTION_Y2] *= img_h

    return detections


# extract_detections_batch
def extract_detections_batch(preds, yolo_layer, obj_thresh=YOLO_OBJ_THRESH):
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
        class_mask = class_probs > YOLO_OBJ_THRESH
        class_mask = torch.sum(class_mask, dim=1).bool()
        valid_preds = preds_thresh[class_mask]
        n_dets = len(valid_preds)

        # Create detections
        if(n_dets > 0):
            best_class_prob, best_class_idx = torch.max(valid_preds[..., YOLO_CLASS_START:], dim=1)
            best_class_prob = best_class_prob
            best_class_idx = best_class_idx

            half_w = valid_preds[..., YOLO_TW] / 2
            half_h = valid_preds[..., YOLO_TH] / 2

            detections = torch.zeros((n_dets, DETECTION_N_ELEMS), dtype=valid_preds.dtype, device=valid_preds.device)
            detections[..., DETECTION_X1] = valid_preds[..., YOLO_TX] - half_w
            detections[..., DETECTION_Y1] = valid_preds[..., YOLO_TY] - half_h
            detections[..., DETECTION_X2] = valid_preds[..., YOLO_TX] + half_w
            detections[..., DETECTION_Y2] = valid_preds[..., YOLO_TY] + half_h
            detections[..., DETECTION_CLASS_IDX] = best_class_idx
            detections[..., DETECTION_CLASS_PROB] = best_class_prob


    # Dummy tensor to represent 0 detections (makes concatenations easier)
    if(detections is None):
        detections = torch.empty((0, DETECTION_N_ELEMS), device=preds.device, dtype=preds.dtype)


    return detections


# run_nms_inplace
def run_nms_inplace(preds, yolo_layer):
    if(yolo_layer.nms_kind == GREEDY_NMS):
        greedy_nms_inplace(preds)

    return

# greedy_nms_inplace
def greedy_nms_inplace(preds, nms_thresh=NMS_THRESHOLD):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Performs greedy nms on yolo predictions
    - Greedy nms forces the lower of two same class probabilities to 0 if the IOU between their respective
      bounding boxes is greater than NMS_THRESHOLD
    - Done in place on preds
    - Assumed that given predictions already have class scores multiplied by objectness
    ----------
    """

    attrs = preds[..., YOLO_TX:YOLO_OBJ]
    class_probs = preds[..., YOLO_CLASS_START:]

    bboxes = attrs.clone()

    half_w = attrs[..., YOLO_TW] / 2
    half_h = attrs[..., YOLO_TH] / 2

    bboxes[..., BBOX_X1] = attrs[..., YOLO_TX] - half_w
    bboxes[..., BBOX_Y1] = attrs[..., YOLO_TY] - half_h
    bboxes[..., BBOX_X2] = attrs[..., YOLO_TX] + half_w
    bboxes[..., BBOX_Y2] = attrs[..., YOLO_TY] + half_h


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

        # Using this with original class_b because masked_select does not share storage with the original tensor
        zero_mask_b = torch.zeros(class_b.size(), device=class_zero_b.device, dtype=class_zero_b.dtype)
        zero_mask_b[thresh_mask] = class_zero_b

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
