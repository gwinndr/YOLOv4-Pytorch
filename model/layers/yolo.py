import torch
import torch.nn as nn

from utilities.constants import *
from utilities.devices import get_device

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

        self.scale_xy = scale_xy

    # forward
    def forward(self, x, input_dim):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - YoloLayer, just a placeholder for now
        ----------
        """

        grid_size = x.shape[X_DIM]
        grid_stride = input_dim // grid_size

        batch_num = x.shape[BATCH_DIM]
        attrs_per_anchor = self.n_classes + YOLO_N_BBOX_ATTRS
        n_anchors = len(self.anchors)
        n_grid = grid_size * grid_size

        anchors = [(anc[0] / grid_stride, anc[1]/ grid_stride) for anc in self.anchors]
        anchors = torch.tensor(anchors, device=get_device())

        # Moving channel dimension to the back to make matrix operations easier to perform
        # TODO: This makes the tensor non-continuous. Is there a way to do this better?
        # I'll try different methods and compare performance. I'll leave this for now.
        x = x.view(batch_num, n_anchors, attrs_per_anchor, n_grid)

        # Moving the grid_size to the second dimension
        # Makes it easier to do matrix operations (NOTE: No longer contiguous)
        x = x.permute(0,3,1,2)

        grid = torch.arange(start=0, end=grid_size, step=1, device=get_device())
        y_offset, x_offset = torch.meshgrid(grid,grid)
        x_offset = x_offset.flatten().repeat(n_anchors,1).permute(1,0)
        y_offset = y_offset.flatten().repeat(n_anchors,1).permute(1,0)

        # print(x_offset.shape)
        # print(grid_size)
        # print(x_offset[...,2])
        # print(y_offset[...,2])


        x[..., YOLO_TX:YOLO_TY+1] = \
            torch.sigmoid(x[..., YOLO_TX:YOLO_TY+1]) * self.scale_xy - (self.scale_xy - 1) / 2
        x[..., YOLO_TW:YOLO_TH+1] = \
            torch.exp(x[..., YOLO_TW:YOLO_TH+1]) * anchors

        x[..., YOLO_OBJ:] = torch.sigmoid(x[..., YOLO_OBJ:])

        x[..., YOLO_TX] += x_offset
        x[..., YOLO_TY] += y_offset
        x[..., YOLO_TX:YOLO_OBJ] *= grid_stride


        return x
