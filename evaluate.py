import torch

from utilities.constants import *

from datasets.coco import CocoDataset, coco_evaluate_bbox

from utilities.devices import get_device
from utilities.configs import parse_config
from utilities.weights import load_weights

def main():
    coco_image = "D:/Datasets/COCO/2017/val2017/"
    coco_ann = "D:/Datasets/COCO/2017/annotations/instances_val2017.json"

    cfg = "./configs/yolov4.cfg"
    weights = "./weights/yolov4.weights"
    class_f = "./configs/coco.names"

    print("Parsing config into model...")
    model = parse_config(cfg)
    if(model is None):
        return

    model = model.to(get_device())
    model.eval()

    print("Loading weights...")
    version, imgs_seen = load_weights(model, weights)

    val_dataset = CocoDataset(coco_image, coco_ann)

    print("")
    coco_evaluate_bbox(val_dataset, model, 608, OBJ_THRESH_DEFAULT, True)


if __name__ == "__main__":
    main()
