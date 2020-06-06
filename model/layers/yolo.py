import torch
import torch.nn as nn

from utilities.constants import *

# The big cheese
# YoloLayer
class YoloLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - A darknet Yolo layer
    ----------
    """

    # __init__
    def __init__(self, anchors, n_classes, ignore_thresh, truth_thresh, random=YOLO_RANDOM,
        jitter=YOLO_JITTER, scale_xy=YOLO_SCALEXY, iou_thresh=YOLO_IOU_THRESH, cls_norm=YOLO_CLS_NORM,
        iou_norm=YOLO_IOU_NORM, iou_loss=YOLO_IOU_LOSS, nms_kind=YOLO_NMS_KIND, beta_nms=YOLO_BETA_NMS,
        max_delta=YOLO_MAX_DELTA):

        super(YoloLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False

        self.anchors = anchors
        self.n_classes = n_classes
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - YoloLayer, just a placeholder for now
        ----------
        """

        return x
