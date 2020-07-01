import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import cv2

from utilities.constants import *

from utilities.devices import get_device
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
def coco_evaluate_bbox(coco_dataset, model, network_dim, obj_thresh, letterbox):
    model.eval()
    with torch.no_grad():
        n_images = len(coco_dataset)
        yolo_layers = model.get_yolo_layers()
        imgs_done = 0

        model_dts = []
        all_img_ids = []
        for img_id in coco_dataset.img_ids:
            # if(imgs_done >= 50):
            #     break

            image = coco_dataset.load_image_by_id(img_id)

            detections = inference_on_image(model, image, network_dim, obj_thresh, letterbox)

            detections_to_coco_format(detections, img_id, model_dts)
            all_img_ids.append(img_id)

            imgs_done += 1
            print("Evaluated %d out of %d" % (imgs_done, n_images), end=CARRIAGE_RETURN, flush=True)

        print("")

        # print(model_dts)
        coco_dts = coco_dataset.coco_api.loadRes(model_dts)
        coco_evaluator = COCOeval(coco_dataset.coco_api, coco_dts, COCO_ANN_TYPE_BBOX)
        coco_evaluator.params.imgIds = all_img_ids
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    return coco_evaluator.stats

def detections_to_coco_format(detections, img_id, model_dts=None):
    if(model_dts is None):
        model_dts = []

    for dt in detections:
        x1 = float(dt[DETECTION_X1])
        y1 = float(dt[DETECTION_Y1])
        width = float(dt[DETECTION_X2]) - x1
        height = float(dt[DETECTION_Y2]) - y1

        coco_80 = round(float(dt[DETECTION_CLASS_IDX]))
        coco_91 = COCO_80_TO_91[coco_80]

        coco_dt = dict()
        coco_dt["image_id"] = img_id
        coco_dt["bbox"] = [x1, y1, width, height]
        coco_dt["score"] = float(dt[DETECTION_CLASS_PROB])
        coco_dt["category_id"] = coco_91

        model_dts.append(coco_dt)

    return model_dts
