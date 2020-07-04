import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2

from utilities.constants import *
from utilities.devices import cpu_device

from utilities.preprocessing import preprocess_image_train
from utilities.images import load_image, image_to_tensor
from utilities.inferencing import inference_on_image
from utilities.detections import detections_best_class

# CocoDataset
class CocoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Pytorch Dataset object for MS-COCO
    - __getitem__ should only be used for training, see coco_evaluate_bbox for validation
    ----------
    """

    # __init__
    def __init__(self, image_root, input_dim, letterbox, annotation_file=None):
        self.image_root = image_root
        self.annotation_file = annotation_file
        self.input_dim = input_dim
        self.letterbox = letterbox

        self.coco_api = COCO(self.annotation_file)
        self.img_ids = self.coco_api.getImgIds()

    # __len__
    def __len__(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Retuns the number of images given by the annotation file
        ----------
        """

        return len(self.img_ids)

    # __getitem__
    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns pre-processed input tensor with the bbox labels properly mapped to it
        - Should only be used while training, see coco_evaluate_bbox for validation
        ----------
        """

        # Image ids given by index
        img_id = self.img_ids[idx]
        image = self.load_image_by_id(img_id)

        anns = self.load_annotations_by_id(img_id)

        # preprocessing
        img_tensor, anns = preprocess_image_train(image, anns, self.input_dim, self.letterbox, force_cpu=True)

        return img_tensor, anns

    # load_image_by_id
    def load_image_by_id(self, img_id):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Given an image id, returns the corresponding cv2 image
        - Returns None if invalid
        ----------
        """

        # Getting full file path prepended with 0's such that there's 12 characters
        img_file = os.path.join(self.image_root, "%012d.jpg" % img_id)

        return load_image(img_file)

    # load_annotations_by_id
    def load_annotations_by_id(self, img_id):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Given an image id, returns the corresponding bbox annotations as a tensor
        - Annotations are in Darknet annotation format (x1, y1, x2, y2, coco_80_class)
        ----------
        """

        ann_id = self.coco_api.getAnnIds(imgIds=img_id)
        coco_anns = self.coco_api.loadAnns(ann_id)

        n_boxes = len(coco_anns)
        darknet_anns = torch.zeros((n_boxes, ANN_BBOX_N_ELEMS), device=cpu_device(), dtype=torch.float32)

        for i, coco_ann in enumerate(coco_anns):
            coco_bbox = coco_ann["bbox"]
            coco_91 = coco_ann["category_id"]

            # Converting coco_91 to coco_80
            try:
                coco_80 = COCO_80_TO_91.index(coco_91)
            except ValueError:
                print("load_annotations_by_id: Warning: Could not convert coco 91 class", coco_91, "to coco 80. Ignoring this annotation.")
                continue

            x1 = coco_bbox[COCO_ANN_BBOX_X]
            y1 = coco_bbox[COCO_ANN_BBOX_Y]
            x2 = x1 + coco_bbox[COCO_ANN_BBOX_W]
            y2 = y1 + coco_bbox[COCO_ANN_BBOX_H]

            darknet_anns[i, ANN_BBOX_X1] = x1
            darknet_anns[i, ANN_BBOX_Y1] = y1
            darknet_anns[i, ANN_BBOX_X2] = x2
            darknet_anns[i, ANN_BBOX_Y2] = y2
            darknet_anns[i, ANN_BBOX_CLASS] = coco_80

        return darknet_anns

# coco_evaluate_bbox
def coco_evaluate_bbox(coco_dataset, model, network_dim, obj_thresh, letterbox, max_imgs=0):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Evaluates model on the given coco_dataset
    - Returns precision and recall statistics given by pycocotools
    - Can limit images evaluated with max_imgs argument. If max_imgs <= 0, all images in the dataset are evaluated.
    ----------
    """

    # Evaluate all images in the given dataset if max_imgs <= 0
    if(max_imgs <= 0):
        max_imgs = len(coco_dataset)

    model.eval()
    with torch.no_grad():
        yolo_layers = model.get_yolo_layers()
        imgs_done = 0

        model_dts = []
        all_img_ids = []
        for i in range(max_imgs):
            img_id = coco_dataset.img_ids[i]

            # Inferencing on coco image
            image = coco_dataset.load_image_by_id(img_id)
            detections = inference_on_image(model, image, network_dim, obj_thresh, letterbox)

            # Converting yolo detections to coco detection format (appends to model_dts)
            detections_to_coco_format(detections, img_id, model_dts)

            # Appending to all_img_ids for COCOeval
            all_img_ids.append(img_id)
            print("Evaluated %d out of %d" % (i+1, max_imgs), end=CARRIAGE_RETURN, flush=True)

        print("")

        # Evaluation of coco images using the API
        coco_dts = coco_dataset.coco_api.loadRes(model_dts)
        coco_evaluator = COCOeval(coco_dataset.coco_api, coco_dts, COCO_ANN_TYPE_BBOX)
        coco_evaluator.params.imgIds = all_img_ids
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    return coco_evaluator.stats

# detections_to_coco_format
def detections_to_coco_format(detections, img_id, model_dts=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts yolo model detections to coco detection format as expected by pycocotools
    - If model_dts is None, returns a list of converted detections. If model_dts is a list, converted detections are appended to it in_place.
    ----------
    """

    if(model_dts is None):
        model_dts = []

    class_confs, classes = detections_best_class(detections)

    for i, dt in enumerate(detections):
        x1 = float(dt[DETECTION_X1])
        y1 = float(dt[DETECTION_Y1])
        width = float(dt[DETECTION_X2]) - x1
        height = float(dt[DETECTION_Y2]) - y1
        class_conf = float(class_confs[i])

        # The API expects the coco 91 class format
        coco_80 = int(classes[i])
        coco_91 = COCO_80_TO_91[coco_80]

        coco_dt = dict()
        coco_dt["image_id"] = img_id
        coco_dt["bbox"] = [x1, y1, width, height]
        coco_dt["score"] = class_conf
        coco_dt["category_id"] = coco_91

        model_dts.append(coco_dt)

    return model_dts
