import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2

from utilities.constants import *

from utilities.file_io import load_image

from utilities.inferencing import inference_on_image

# CocoDataset
class CocoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Pytorch Dataset object for MS-COCO
    ----------
    """

    # __init__
    def __init__(self, image_root, annotation_file=None, input_dim=INPUT_DIM_DEFAULT, letterbox=LETTERBOX_DEFAULT):
        self.image_root = image_root
        self.annotation_file = annotation_file
        self.input_dim = input_dim
        self.letterbox = letterbox

        self.coco_api = COCO(self.annotation_file)
        self.img_ids = self.coco_api.getImgIds()

    # __len__
    def __len__(self):
        return len(self.img_ids)

    # __getitem__
    def __getitem__(self, idx):
        # Image ids given by index
        img_id = self.img_ids[idx]

        # Getting full file path prepended with 0's such that there's 12 characters
        img_file = os.path.join(self.image_root, "%012d.jpg" % img_id)

        image = cv2.imread(img_file)
        img_tensor, img_info = preprocess_image_eval(image, self.input_dim, self.letterbox, force_cpu=True)

        return img_tensor, img_id, img_info

    # load_image_by_id
    def load_image_by_id(self, img_id):
        # Getting full file path prepended with 0's such that there's 12 characters
        img_file = os.path.join(self.image_root, "%012d.jpg" % img_id)

        return load_image(img_file)


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

    for dt in detections:
        x1 = float(dt[DETECTION_X1])
        y1 = float(dt[DETECTION_Y1])
        width = float(dt[DETECTION_X2]) - x1
        height = float(dt[DETECTION_Y2]) - y1

        # The API expects the coco 91 class format
        coco_80 = round(float(dt[DETECTION_CLASS_IDX]))
        coco_91 = COCO_80_TO_91[coco_80]

        coco_dt = dict()
        coco_dt["image_id"] = img_id
        coco_dt["bbox"] = [x1, y1, width, height]
        coco_dt["score"] = float(dt[DETECTION_CLASS_PROB])
        coco_dt["category_id"] = coco_91

        model_dts.append(coco_dt)

    return model_dts
