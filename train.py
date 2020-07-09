import torch

from utilities.constants import *
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights

from utilities.detections import extract_detections
from utilities.nms import run_nms
from utilities.bboxes import bbox_iou_one_to_many

from datasets.coco import CocoDataset
from utilities.images import tensor_to_image, draw_annotations, draw_detections

def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for training a darknet model
    ----------
    """


    # train_imgs = "D:/Datasets/COCO/2017/train2017"
    # train_anns = "D:/Datasets/COCO/2017/annotations/instances_train2017.json"
    train_imgs = "D:/Datasets/COCO/2017/val2017"
    train_anns = "D:/Datasets/COCO/2017/annotations/instances_val2017.json"

    config = "./configs/yolov4.cfg"
    weights = "./weights/yolov4.weights"

    obj_thresh = 0.25

    class_f = "./configs/coco.names"
    class_names = parse_names(class_f)

    model = parse_config(config)
    load_weights(model, weights)

    train_set = CocoDataset(train_imgs, input_dim=608, letterbox=True, annotation_file=train_anns)
    print("")

    x, anns = train_set[200]

    image = tensor_to_image(x)

    # model.train()
    model.eval()

    model.training_custom = True
    out = model(x.unsqueeze(0), anns.unsqueeze(0))
    # print(out)
    # print(anns)

    model.training_custom = False
    with torch.no_grad():
        out = model(x.unsqueeze(0))

    dets = extract_detections(out, obj_thresh)[0]
    dets = run_nms(dets, model, obj_thresh)

    # for ann in anns:
    #     ious = bbox_iou_one_to_many(ann[ANN_BBOX_X1:ANN_BBOX_Y2+1], dets[..., DETECTION_X1:DETECTION_Y2+1])
    #     print(ious)

    dets_image = draw_detections(dets, image, class_names)
    anns_image = draw_annotations(anns, image, class_names)

    # print(dets[..., DETECTION_X1:DETECTION_Y2+1])

    # cv2.imshow("Annotations", anns_image)
    # # cv2.waitKey(0)
    #
    # cv2.imshow("Detections", dets_image)
    # cv2.waitKey(0)

    return


if __name__ == "__main__":
    main()
