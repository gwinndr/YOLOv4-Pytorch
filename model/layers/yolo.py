import torch
import torch.nn as nn

from utilities.constants import *
from utilities.devices import get_device

# The big cheese
# YoloLayer
class YoloLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
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

        self.nms_kind = nms_kind

    # forward
    def forward(self, x, input_dim):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - YoloLayer, extracts predictions

        - When in detection mode, returns predictions filtered out by specified NMS type
        - All detections are returned regardless of object score and will need to be thresholded
        - TX, TY, TW, and TH are already set to the correct bbox dimensions according to the input image
          size and will require no post-processing.
        ----------
        """

        grid_dim = x.shape[X_DIM]
        grid_stride = input_dim // grid_dim

        batch_num = x.shape[BATCH_DIM]
        attrs_per_anchor = self.n_classes + YOLO_N_BBOX_ATTRS
        n_anchors = len(self.anchors)
        grid_size = grid_dim * grid_dim

        anchors = [(anc[0] / grid_stride, anc[1] / grid_stride) for anc in self.anchors]
        anchors = torch.tensor(anchors, device=get_device())

        # Combining grid_dims into one vector
        # Moving the grid_size to the second dimension for easier matrix operations
        x = x.view(batch_num, n_anchors, attrs_per_anchor, grid_size)
        x = x.permute(0,3,1,2).contiguous()

        # Grid offsets for each grid cell
        grid = torch.arange(start=0, end=grid_dim, step=1, device=get_device())
        y_offset, x_offset = torch.meshgrid(grid,grid)

        x_offset = x_offset.flatten()
        y_offset = y_offset.flatten()

        # The permute is to help pytorch broadcast offsets to each grid cell properly
        # expand is like repeat but shares memory
        x_offset = x_offset.expand(n_anchors, -1).permute(1,0)
        y_offset = y_offset.expand(n_anchors, -1).permute(1,0)

        x[..., YOLO_TX:YOLO_TY+1] = \
            torch.sigmoid(x[..., YOLO_TX:YOLO_TY+1]) * self.scale_xy - (self.scale_xy - 1) / 2
        x[..., YOLO_TW:YOLO_TH+1] = \
            torch.exp(x[..., YOLO_TW:YOLO_TH+1]) * anchors

        x[..., YOLO_OBJ:] = torch.sigmoid(x[..., YOLO_OBJ:])

        x[..., YOLO_TX] += x_offset
        x[..., YOLO_TY] += y_offset

        # Converting values from grid-relative to image-relative
        x[..., YOLO_TX:YOLO_TH+1] *= grid_stride

        # Combining the anchor and grid dimensions
        x = x.view(batch_num, grid_size*n_anchors, attrs_per_anchor)

        return x

    # to_string
    def to_string(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Converts this layer into a human-readable string
        ----------
        """

        return \
            "YOLO: ignore: %.2f  truth: %.2f  scalexy: %.2f  nms_t: %s" % \
            (self.ignore_thresh, self.truth_thresh, self.scale_xy, self.nms_kind)