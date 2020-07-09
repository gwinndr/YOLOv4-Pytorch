import torch
import torch.nn as nn

from utilities.constants import *
from utilities.bboxes import bbox_iou_one_to_many, predictions_to_bboxes, bbox_iou
from utilities.detections import extract_detections_single_image

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
    def __init__(self, anchors, anchor_mask, n_classes, ignore_thresh, truth_thresh, random=YOLO_RANDOM,
        jitter=YOLO_JITTER, scale_xy=YOLO_SCALEXY, iou_thresh=YOLO_IOU_THRESH, cls_norm=YOLO_CLS_NORM,
        iou_norm=YOLO_IOU_NORM, iou_loss=YOLO_IOU_LOSS, nms_kind=YOLO_NMS_KIND, beta_nms=YOLO_BETA_NMS,
        max_delta=YOLO_MAX_DELTA):

        super(YoloLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False
        self.is_output_layer = True

        self.all_anchors = anchors
        self.anchor_mask = anchor_mask
        self.n_classes = n_classes
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh

        self.scale_xy = scale_xy

        self.nms_kind = nms_kind

    # forward
    def forward(self, x, input_dim, anns=None):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - YoloLayer, return value varies when in train vs eval mode
        - See self.yolo_train for output info while training
        - See self.yolo_inference for output info while inferencing
        ----------
        """

        # if(self.training):
        if(anns is not None):
            out = self.yolo_train(x, input_dim, anns)
        else:
            out = self.yolo_inference(x, input_dim)

        return out

    # yolo_inference
    def yolo_inference(self, x, input_dim):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Extracts predictions:
            - Predictions are returned as [batch, n_preds, bbox_attributes]
            - bbox_attributes tx, ty, tw, and th have values relative to the input image
                (i.e. tx = 128 means pixel at x=128 on the image given to the model forward method)
            - Objectness and class scores are sigmoided.
            - Note: Class scores are not multiplied by objectness
        ----------
        """

        device = x.device

        grid_dim = x.shape[INPUT_H_DIM]
        grid_stride = input_dim // grid_dim

        batch_num = x.shape[INPUT_BATCH_DIM]
        attrs_per_anchor = self.n_classes + YOLO_N_BBOX_ATTRS
        n_anchors = len(self.anchor_mask)
        grid_size = grid_dim * grid_dim

        anchors = [self.all_anchors[i] for i in self.anchor_mask]
        anchors = [(anc[0] / grid_stride, anc[1] / grid_stride) for anc in anchors]
        anchors = torch.tensor(anchors, dtype=torch.float32, device=device)

        # Combining grid_dims into one vector
        x = x.view(batch_num, n_anchors, attrs_per_anchor, grid_size)

        # (batch, grid_size, n_anchors, attrs_per_anchor)
        x = x.permute(0,3,1,2).contiguous()

        # Performs yolo transforms (sigmoid, anchor offset, etc.)
        self.yolo_transform(x, anchors)

        # Adds grid cell offsets to tx and ty
        self.add_grid_offsets(x, grid_dim, n_anchors)

        # Converting values from grid relative to input image relative
        x[..., YOLO_TX:YOLO_TH+1] *= grid_stride

        # Combining the anchor and grid dimensions into one n_predictions dimension
        x = x.view(batch_num, grid_size*n_anchors, attrs_per_anchor)

        return x

    # yolo_train
    def yolo_train(self, x, input_dim, anns):

        device = x.device
        anns.requires_grad = False

        grid_dim = x.shape[INPUT_H_DIM]
        grid_stride = input_dim // grid_dim

        batch_num = x.shape[INPUT_BATCH_DIM]
        attrs_per_anchor = self.n_classes + YOLO_N_BBOX_ATTRS
        n_anchors = len(self.anchor_mask)
        grid_size = grid_dim * grid_dim

        all_anchors = [(anc[0] / grid_stride, anc[1] / grid_stride) for anc in self.all_anchors]
        mask_anchors = [all_anchors[i] for i in self.anchor_mask]

        all_anchors = torch.tensor(all_anchors, dtype=torch.float32, device=device, requires_grad=False)
        mask_anchors = torch.tensor(mask_anchors, dtype=torch.float32, device=device, requires_grad=False)

        # Combining grid_dims into one vector
        x = x.view(batch_num, n_anchors, attrs_per_anchor, grid_size)

        # (batch, grid_size, n_anchors, attrs_per_anchor)
        x = x.permute(0,3,1,2).contiguous()

        # Mapping annotations to grid
        anns = anns.clone()
        anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] /= grid_stride

        for b, batch_x in enumerate(x):
            batch_anns = anns[b]

            # Transforming a clone of x (no gradients)
            x_transformed = batch_x.detach().clone()
            self.yolo_transform(x_transformed, mask_anchors)
            self.add_grid_offsets(x_transformed, grid_dim, n_anchors)

            dets = extract_detections_single_image(x_transformed, OBJ_THRESH_DEFAULT)

            # Bounding boxes for x predictions
            x_boxes = predictions_to_bboxes(x_transformed)
            # x_boxes = dets[..., DETECTION_X1:DETECTION_Y2+1]

            # Annotations bounding boxes
            ann_boxes = batch_anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]

            # First get ignore indices (prediction boxes that overlap the gt by a certain threshold)
            ignore_mask = self.ignore_mask(x_boxes, ann_boxes, self.ignore_thresh)

            # ignore_mask = ignore_mask.view(grid_dim, grid_dim, n_anchors)
            # batch_x = batch_x.view(grid_dim, grid_dim, n_anchors, attrs_per_anchor)

            from utilities.configs import parse_names
            class_names = parse_names("./configs/coco.names")

            x_ign = x_transformed[ignore_mask]
            print("=====")
            print("IGNORE:")
            print(x_ign.shape)
            for ign in x_ign:
                cls = ign[YOLO_CLASS_START:]
                obj = ign[YOLO_OBJ]
                i = torch.argmax(cls)
                # print(i)
                # print(cls.shape)
                print(class_names[i], cls[i], obj)
                # print(cls)
            # print("=====")
            print("")

            print("DETECTIONS:")
            print("anns:")
            print(ann_boxes)
            for det in dets:
                i = torch.argmax(det[DETECTION_CLASS_START:])
                print(class_names[i])
                print(det[DETECTION_X1:DETECTION_Y2+1])

                # print(bbox_iou_one_to_many(det[DETECTION_X1:DETECTION_Y2+1], ann_boxes))
            print("")

        return None


    # yolo_transform
    def yolo_transform(self, x, anchors):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Yolo postprocessing transformation
        - Sigmoids tx and ty offset values
        - Multiplies exp'd tw and th values by anchor boxes
        ----------
        """

        # TX, TY, TW, and TH post-processing
        x[..., YOLO_TX:YOLO_TY+1] = \
            torch.sigmoid(x[..., YOLO_TX:YOLO_TY+1]) * self.scale_xy - (self.scale_xy - 1) / 2
        x[..., YOLO_TW:YOLO_TH+1] = \
            torch.exp(x[..., YOLO_TW:YOLO_TH+1]) * anchors

        # Sigmoid objectness and class scores
        x[..., YOLO_OBJ:] = torch.sigmoid(x[..., YOLO_OBJ:])

        return

    # add_grid_offsets
    def add_grid_offsets(self, x, grid_dim, n_anchors):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Adds grid position offsets to each tx and ty value
        ----------
        """

        device = x.device

        # Grid offsets for each grid cell
        grid = torch.arange(start=0, end=grid_dim, step=1, device=device)
        y_offset, x_offset = torch.meshgrid(grid,grid)

        x_offset = x_offset.flatten()
        y_offset = y_offset.flatten()

        # The permute is to help pytorch broadcast offsets to each grid cell properly
        # It just works, it's magic, I don't know, welcome to hell
        x_offset = x_offset.expand(n_anchors, -1).permute(1,0)
        y_offset = y_offset.expand(n_anchors, -1).permute(1,0)

        # Adding grid offsets to TX and TY
        x[..., YOLO_TX] += x_offset
        x[..., YOLO_TY] += y_offset

        return

    # ignore_mask
    def ignore_mask(self, x_boxes, ann_boxes, ignore_thresh):
        # Don't need gradients for ignore indices
        with torch.no_grad():
            device = x_boxes.device

            # Dropping attribute dim
            ignore_shape = x_boxes.shape[:-1]

            # Filling mask with False values
            ignore_mask = torch.full(ignore_shape, False, dtype=torch.bool, device=device, requires_grad=False)

            # Figuring out which predictions overlap an annotation by more than ignore_thresh
            for ann_box in ann_boxes:
                ious = bbox_iou_one_to_many(ann_box, x_boxes)
                # print(ious.shape)
                ann_ignore = ious > ignore_thresh

                torch.logical_or(ignore_mask, ann_ignore, out=ignore_mask)

        return ignore_mask


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
