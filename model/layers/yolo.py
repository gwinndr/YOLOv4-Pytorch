import torch
import torch.nn as nn

from utilities.constants import *
from utilities.bboxes import bbox_iou_one_to_many, predictions_to_bboxes, bbox_iou, bbox_iou_many_to_many, bbox_ciou
from utilities.detections import extract_detections_single_image

# Used for debugging
yolo_layer_num = 0

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

        self.iou_thresh = iou_thresh
        self.iou_loss = iou_loss

        self.iou_norm = iou_norm
        self.cls_norm = cls_norm

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
        # x = x.view(batch_num, n_anchors, attrs_per_anchor, grid_size)
        x = x.view(batch_num, n_anchors, attrs_per_anchor, grid_dim, grid_dim)

        # (batch, grid_size, n_anchors, attrs_per_anchor)
        # x = x.permute(0,3,1,2).contiguous()
        x = x.permute(0,3,4,1,2).contiguous()

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
        n_pred_anchors = len(self.anchor_mask)

        all_anchors = [(anc[0] / grid_stride, anc[1] / grid_stride) for anc in self.all_anchors]
        pred_anchors = [all_anchors[i] for i in self.anchor_mask]

        all_anchors = torch.tensor(all_anchors, dtype=torch.float32, device=device, requires_grad=False)
        pred_anchors = torch.tensor(pred_anchors, dtype=torch.float32, device=device, requires_grad=False)

        # Viewing yolo attributes
        x = x.view(batch_num, n_pred_anchors, attrs_per_anchor, grid_dim, grid_dim)

        # Moving grid dimensions to the front of the tensor after batch_num
        # (batch, grid_dim, grid_dim, n_masked_anchors, attrs_per_anchor)
        x = x.permute(0,3,4,1,2).contiguous()

        tot_bbox_loss = 0.0
        tot_obj_loss = 0.0
        tot_cls_loss = 0.0
        tot_n_resp = 0

        for b, batch_x in enumerate(x):
            # Removing padding labels
            batch_anns = anns[b]
            ann_mask = (batch_anns[..., ANN_BBOX_CLASS] != ANN_PAD_VAL)
            batch_anns = batch_anns[ann_mask]

            # Mapping annotations to grid
            batch_anns = batch_anns.clone()
            batch_anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] *= grid_dim

            n_anns = len(batch_anns)
            ann_boxes = batch_anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
            ann_cls = batch_anns[..., ANN_BBOX_CLASS].type(torch.int32)

            # Following do not require gradients
            with torch.no_grad():

                # Indexes for responsible anchors (h,w,a) and their corresponding targets
                resp_pred_idxs, resp_anns = self.responsible_predictions(all_anchors, batch_anns)
                resp_pred_h, resp_pred_w, resp_pred_a = resp_pred_idxs

                # ious between predictions and annotations for masks
                x_ann_ious = self.iou_preds_anns(batch_x, batch_anns, grid_dim, pred_anchors)

                # Ignore mask, True if predictions overlap a ground truth by more than ignore threshold
                ignore_mask = x_ann_ious > self.ignore_thresh
                ignore_mask = ignore_mask.sum(dtype=torch.bool, dim=1) # logical_or along annotation dim
                ignore_mask = ignore_mask.view(grid_dim, grid_dim, n_pred_anchors)

                # Responsible predictions are always ignored since they have a separate loss calculation
                ignore_mask[resp_pred_h, resp_pred_w, resp_pred_a] = True

            # end torch.no_grad()

            # Loss objects
            mse = nn.MSELoss(reduction = "mean")
            sse = nn.MSELoss(reduction = "sum")

            # Transforming responsible predictions
            resp_preds = batch_x[resp_pred_h, resp_pred_w, resp_pred_a]
            resp_ancs = pred_anchors[resp_pred_a]
            self.yolo_transform(resp_preds, resp_ancs)
            resp_preds[..., YOLO_TX] += resp_pred_w
            resp_preds[..., YOLO_TY] += resp_pred_h

            n_resp = len(resp_preds)

            # Setting up targets
            target_zeros = torch.zeros(resp_preds.shape[:-1], dtype=torch.float32, device=device, requires_grad=False)
            target_ones = target_zeros + 1.0

            ##### BBOX LOSS #####
            resp_pred_boxes = predictions_to_bboxes(resp_preds)
            resp_anns_boxes = resp_anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
            bbox_cious = bbox_ciou(resp_pred_boxes, resp_anns_boxes)
            bbox_loss = sse(bbox_cious, target_ones)

            ##### OBJECTNESS LOSS #####
            pred_obj = resp_preds[..., YOLO_OBJ]
            resp_obj_loss = sse(pred_obj, target_ones)

            ##### CLASSIFICATION LOSS #####
            pred_cls = resp_preds[..., YOLO_CLASS_START:]
            resp_anns_cls = resp_anns[..., ANN_BBOX_CLASS].type(torch.long)

            target_cls = torch.zeros(pred_cls.shape, dtype=torch.float32, device=device, requires_grad=False)
            resp_idxs = torch.arange(start=0, end=n_resp, step=1, device=device) # just to help index properly
            target_cls[resp_idxs, resp_anns_cls] = 1.0

            cls_loss = sse(pred_cls, target_cls)

            ##### ZERO-OBJECTNESS LOSS #####
            # Ignored predictions are not penalized
            non_resp_objs = batch_x[~ignore_mask][..., YOLO_OBJ]
            non_resp_objs = torch.sigmoid(non_resp_objs)
            target_zeros = torch.zeros(non_resp_objs.shape, dtype=torch.float32, device=device)

            non_resp_obj_loss = sse(non_resp_objs, target_zeros)

            # Fixing NaNs
            loss_zero = torch.zeros((1,), dtype=torch.float32, device=device, requires_grad=True)
            if(torch.isnan(bbox_loss)):
                bbox_loss = loss_zero
            if(torch.isnan(resp_obj_loss)):
                resp_obj_loss = loss_zero
            if(torch.isnan(cls_loss)):
                cls_loss = loss_zero
            if(torch.isnan(non_resp_obj_loss)):
                non_resp_obj_loss = loss_zero

            # Full objectness loss
            obj_loss = resp_obj_loss + non_resp_obj_loss

            tot_bbox_loss += bbox_loss
            tot_obj_loss += obj_loss
            tot_cls_loss += cls_loss
            tot_n_resp += n_resp

        avg_bbox_loss = (tot_bbox_loss / batch_num) * self.iou_norm
        avg_obj_loss = (tot_obj_loss / batch_num) * self.cls_norm
        avg_cls_loss = (tot_cls_loss / batch_num) * self.cls_norm

        total_loss = avg_bbox_loss + avg_obj_loss + avg_cls_loss

        print("bbox_loss: %.4f  obj_loss: %.4f  cls_loss: %.4f  tot_loss: %.4f  batch_size: %d  count: %d" % \
                (avg_bbox_loss, avg_obj_loss, avg_cls_loss, total_loss, batch_num, tot_n_resp))
        print("")

        return total_loss


    # yolo_transform
    def yolo_transform(self, x, anchors, objcls=True):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Yolo postprocessing transformation
        - Sigmoids tx and ty offset values and scales by scale_xy
        - Multiplies exp'd tw and th values by anchor boxes
        - If objcls, also sigmoids objectness and class scores
        ----------
        """

        # TX and TY post-processing
        x[..., YOLO_TX:YOLO_TY+1] = \
            torch.sigmoid(x[..., YOLO_TX:YOLO_TY+1]) * self.scale_xy - (self.scale_xy - 1) / 2

        # TW and TY post-processing
        x[..., YOLO_TW:YOLO_TH+1] = \
            torch.exp(x[..., YOLO_TW:YOLO_TH+1]) * anchors

        # Sigmoid objectness and class scores
        if(objcls):
            x[..., YOLO_OBJ] = torch.sigmoid(x[..., YOLO_OBJ])
            x[..., YOLO_CLASS_START:] = torch.sigmoid(x[..., YOLO_CLASS_START:])

        return

    # add_grid_offsets
    def add_grid_offsets(self, x, grid_dim, n_anchors):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Adds grid position offsets to each tx and ty value
        - Expect x to be tensor of shape (..., grid_dim, grid_dim, n_anchors, yolo_attrs)
        ----------
        """

        device = x.device

        # Grid offsets for each grid cell
        grid = torch.arange(start=0, end=grid_dim, step=1, device=device)
        y_offset, x_offset = torch.meshgrid(grid,grid)

        # Expanding offsets to be of the shape (grid_dim, grid_dim, n_anchors)
        x_offset = x_offset.expand(n_anchors, grid_dim, grid_dim).permute(1,2,0)
        y_offset = y_offset.expand(n_anchors, grid_dim, grid_dim).permute(1,2,0)

        # Adding grid offsets to TX and TY
        x[..., YOLO_TX] += x_offset
        x[..., YOLO_TY] += y_offset

        return

    # responsible_predictions
    def responsible_predictions(self, all_anchors, anns):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Computes indexes and corresponding annotations for responsible predictions
        - Responsible predictions:
            - Predictions whose grid cell is centered on an annotation and whose anchor prior is responsible
            - Responsible anchor priors overlap an annotation more than any other anchor prior
            - If any anchor prior overlaps an annotation by more than iou_thresh, it is considered responsible
            - Annotations have at minimum one responsible prediction across all yolo layers
            - There may not be any responsible predictions on this yolo layer since all anchor priors are considered
        - Returns responsible prediction indices (grid_h, grid_w, grid_a) and their corresponding annotations
        ----------
        """

        device = all_anchors.device
        n_anchors = len(all_anchors)
        n_pred_anchors = len(self.anchor_mask)
        n_anns = len(anns)

        ann_boxes = anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
        ann_boxes_topleft = ann_boxes.clone()
        ann_boxes_topleft[..., ANN_BBOX_X2] -= ann_boxes_topleft[..., ANN_BBOX_X1]
        ann_boxes_topleft[..., ANN_BBOX_Y2] -= ann_boxes_topleft[..., ANN_BBOX_Y1]
        ann_boxes_topleft[..., ANN_BBOX_X1] = 0.0
        ann_boxes_topleft[..., ANN_BBOX_Y1] = 0.0

        all_anchor_boxes = torch.zeros((n_anchors, BBOX_N_ELEMS), dtype=torch.float32, device=device)
        all_anchor_boxes[..., BBOX_X2:BBOX_Y2+1] = all_anchors

        # IOUs between annotations and anchor boxes are the driving force behind responsible predictions
        ann_anc_ious = bbox_iou_many_to_many(ann_boxes_topleft, all_anchor_boxes)

        # Responsible anchors, True if anchor overlaps corresponding ground truth (anns) by more than some threshold
        resp_anc_mask = ann_anc_ious > self.iou_thresh

        # Anchors with the highest iou for an annotation are always responsible regardless of iou_thresh
        best_ancs = torch.argmax(ann_anc_ious, dim=1)
        ann_idxs = torch.arange(start=0, end=n_anns, step=1, device=device) # just to help index properly
        resp_anc_mask[ann_idxs, best_ancs] = True

        # Subset out anchors tied to this yolo layer's predictions
        masked_idxs = torch.tensor(self.anchor_mask, dtype=torch.long, device=device)
        resp_anc_mask = resp_anc_mask[..., masked_idxs]

        # Corresponding annotations for each responsible prediction
        #(n_anns, n_masked_anchors, n_bbox_elems)
        anns_expanded = anns.expand((n_pred_anchors, n_anns, ANN_BBOX_N_ELEMS)).permute(1,0,2)
        resp_anns = anns_expanded[resp_anc_mask]

        # Grid coordinates for each responsible prediction
        resp_anns_w = resp_anns[..., ANN_BBOX_X2] - resp_anns[..., ANN_BBOX_X1]
        resp_anns_h = resp_anns[..., ANN_BBOX_Y2] - resp_anns[..., ANN_BBOX_Y1]
        resp_anns_cx = resp_anns[..., ANN_BBOX_X1] + resp_anns_w / 2.0
        resp_anns_cy = resp_anns[..., ANN_BBOX_Y1] + resp_anns_h / 2.0

        # Grid coordinates are the floor of the center coordinate for the corresponding annotation bbox
        resp_pred_h = resp_anns_cy.type(torch.long)
        resp_pred_w = resp_anns_cx.type(torch.long)

        # Anchor box idxs for responsible predictions
        pred_anchor_idxs = torch.arange(start=0, end=n_pred_anchors, step=1, device=device)
        pred_anchor_idxs = pred_anchor_idxs.expand(n_anns, n_pred_anchors)
        resp_pred_a = pred_anchor_idxs[resp_anc_mask]

        # The full index list for responsible predictions
        resp_pred_idxs = (resp_pred_h, resp_pred_w, resp_pred_a)

        return resp_pred_idxs, resp_anns

    # iou_preds_anns
    def iou_preds_anns(self, x, anns, grid_dim, pred_anchors):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Computes ious between predictions (x) and annotations
        - Returns an n x m iou tensor where n is the number of predictions and m is the number of annotations
        ----------
        """

        n_anchors = len(pred_anchors)

        # Getting a flattened copy of txtytwth values for iou computations
        x_ts = x[..., YOLO_TX:YOLO_TH+1].clone()
        self.yolo_transform(x_ts, pred_anchors, objcls=False)
        self.add_grid_offsets(x_ts, grid_dim, n_anchors)
        x_ts = x_ts.view(-1, x_ts.shape[-1])

        # Flattened annotations
        anns = anns.view(-1, anns.shape[-1])

        x_boxes = predictions_to_bboxes(x_ts)
        ann_boxes = anns[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]

        x_ann_ious = bbox_iou_many_to_many(x_boxes, ann_boxes)

        return x_ann_ious

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
