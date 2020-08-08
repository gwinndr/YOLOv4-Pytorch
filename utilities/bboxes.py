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

# correct_boxes
def correct_boxes(boxes, ow, oh, nw, nh,
    o_offs_x=None, o_offs_y=None, o_embed_w=None, o_embed_h=None,
    n_offs_x=None, n_offs_y=None, n_embed_w=None, n_embed_h=None,
    boxes_normalized=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Maps bounding boxes from one image to another
    - Used to map annotations to an augmented image, or map detections back to the original image
    - ow, oh: The width and height of the source
    - nw, nh: The width and height of the target
    - (optional) o_offs_x, o_offs_y: Image offset for the source (image embedded within an image = imageception)
    - (optional) o_embed_w, o_embed_h: Width and height of the embedded image for the source
    - (optional) n_offs_x, ..., n_embed_h: Same as o but for the target
    - (optional) boxes_normalized: Flag for if the given boxes are normalized according to ow and oh (will return normalized result)
    ----------
    """

    o_offs_x = 0.0 if o_offs_x is None else o_offs_x
    o_offs_y = 0.0 if o_offs_y is None else o_offs_y
    o_embed_w = ow if o_embed_w is None else o_embed_w
    o_embed_h = oh if o_embed_h is None else o_embed_h

    n_offs_x = 0.0 if n_offs_x is None else n_offs_x
    n_offs_y = 0.0 if n_offs_y is None else n_offs_y
    n_embed_w = nw if n_embed_w is None else n_embed_w
    n_embed_h = nh if n_embed_h is None else n_embed_h

    boxes = boxes.clone()

    x1 = boxes[..., BBOX_X1]
    y1 = boxes[..., BBOX_Y1]
    x2 = boxes[..., BBOX_X2]
    y2 = boxes[..., BBOX_Y2]

    ### ORIGINAL IMAGE ADJUSTMENTS ###
    # Map to full original image
    if(boxes_normalized):
        x1 *= ow
        y1 *= oh
        x2 *= ow
        y2 *= oh

    # Move embedded image back to top left
    x1 -= o_offs_x
    y1 -= o_offs_y
    x2 -= o_offs_x
    y2 -= o_offs_y

    # Normalize by the embedded image
    x1 /= o_embed_w
    y1 /= o_embed_h
    x2 /= o_embed_w
    y2 /= o_embed_h

    ### NEW IMAGE ADJUSTMENTS (inverse of original) ###
    # Map to image embedded within
    x1 *= n_embed_w
    y1 *= n_embed_h
    x2 *= n_embed_w
    y2 *= n_embed_h

    # Add offset
    x1 += n_offs_x
    y1 += n_offs_y
    x2 += n_offs_x
    y2 += n_offs_y

    # Clamp to lie within image
    torch.clamp(x1, min=0, max=nw, out=x1)
    torch.clamp(y1, min=0, max=nh, out=y1)
    torch.clamp(x2, min=0, max=nw, out=x2)
    torch.clamp(y2, min=0, max=nh, out=y2)

    # Normalize by the full image
    if(boxes_normalized):
        x1 /= nw
        y1 /= nh
        x2 /= nw
        y2 /= nh

    return boxes

# crop_boxes
def crop_boxes(boxes, ow, oh, crop_left, crop_top, crop_w, crop_h, boxes_normalized=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Crops out bounding boxes
    - Essentially smushes bboxes to fit within a certain crop
    - Some returned boxes may be invalid due to being outside the crop (use is_valid_box)
    - ow, oh: The width and height of the source
    - crop_left, crop_top: The top left point for the start of the crop
    - crop_w, crop_h: The width and height of the crop
    - (optional) boxes_normalized: Flag for if the given boxes are normalized according to ow and oh (will return normalized result)
    ----------
    """

    boxes = boxes.clone()

    x1 = boxes[..., BBOX_X1]
    y1 = boxes[..., BBOX_Y1]
    x2 = boxes[..., BBOX_X2]
    y2 = boxes[..., BBOX_Y2]

    crop_right = crop_left + crop_w
    crop_bot = crop_top + crop_h

    # Map to original image
    if(boxes_normalized):
        x1 *= ow
        y1 *= oh
        x2 *= ow
        y2 *= oh

    # Clamp to lie within crop
    torch.clamp(x1, min=crop_left, max=crop_right, out=x1)
    torch.clamp(y1, min=crop_top, max=crop_bot, out=y1)
    torch.clamp(x2, min=crop_left, max=crop_right, out=x2)
    torch.clamp(y2, min=crop_top, max=crop_bot, out=y2)

    # Move to top left
    x1 -= crop_left
    y1 -= crop_top
    x2 -= crop_left
    y2 -= crop_top

    # Normalize by the cropped image
    if(boxes_normalized):
        x1 /= crop_w
        y1 /= crop_h
        x2 /= crop_w
        y2 /= crop_h

    return boxes

# is_valid_box
def is_valid_box(boxes, img_w, img_h, boxes_normalized=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Validates that all boxes have an area > 0 and lie on the image plane
    - Returns a boolean mask (can use boxes[is_valid])
    ----------
    """

    boxes = boxes.clone()

    x1 = boxes[..., BBOX_X1]
    y1 = boxes[..., BBOX_Y1]
    x2 = boxes[..., BBOX_X2]
    y2 = boxes[..., BBOX_Y2]

    if(boxes_normalized):
        x1 *= img_w
        y1 *= img_h
        x2 *= img_w
        y2 *= img_h

    # Round to nearest pixel
    x1 = torch.round(x1)
    y1 = torch.round(y1)
    x2 = torch.round(x2)
    y2 = torch.round(y2)

    # Clamp within image
    torch.clamp(x1, min=0, max=img_w, out=x1)
    torch.clamp(y1, min=0, max=img_h, out=y1)
    torch.clamp(x2, min=0, max=img_w, out=x2)
    torch.clamp(y2, min=0, max=img_h, out=y2)

    # If clamp results in equal dimensions (box off image or has 0 area), it's invalid
    ne_x = (x1 != x2)
    ne_y = (y1 != y2)

    is_valid = torch.logical_and(ne_x, ne_y)

    return is_valid

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
