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

    train_set = CocoDataset(train_imgs, input_dim=608, letterbox=False, annotation_file=train_anns)
    print("")

    x, anns, id = train_set[200]
    x2, anns2, id = train_set[250]
    # print(id)
    # print(anns)

    # anns_w = anns[..., BBOX_X2] - anns[..., BBOX_X1]
    # anns_h = anns[..., BBOX_Y2] - anns[..., BBOX_Y1]
    # anns_x = anns[..., BBOX_X1]
    # anns_y = anns[..., BBOX_Y1]
    #
    # anns[..., BBOX_X1] = anns_x + anns_w / 2.0
    # anns[..., BBOX_Y1] = anns_y + anns_h / 2.0
    # anns[..., BBOX_X2] = anns_w
    # anns[..., BBOX_Y2] = anns_h
    #
    # print(anns)
    #
    # import sys
    # sys.exit(1)

    image = tensor_to_image(x)

    x_in = torch.stack((x, x2))
    anns_in = torch.stack((anns, anns2))
    # print(x_in.shape)
    # print(anns_in.shape)

    model.train()
    # out = model(x.unsqueeze(0), anns.unsqueeze(0))
    # out = model(x2.unsqueeze(0), anns2.unsqueeze(0))
    out = model(x_in, anns_in)
    # print(out)
    # print(anns)

    print(SEPARATOR)
    print("Error: %.4f" % (sum(out).item()/3))

    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))

    dets = extract_detections(out, obj_thresh)[0]
    dets = run_nms(dets, model, obj_thresh)

    for ann in anns:
        ious = bbox_iou_one_to_many(ann[ANN_BBOX_X1:ANN_BBOX_Y2+1], dets[..., DETECTION_X1:DETECTION_Y2+1])
        # print(ious)

    dets_image = draw_detections(dets, image, class_names)
    anns_image = draw_annotations(anns, image, class_names)

    # print(dets[..., DETECTION_X1:DETECTION_Y2+1])

    cv2.imshow("Annotations", anns_image)
    # cv2.waitKey(0)

    cv2.imshow("Detections", dets_image)
    cv2.waitKey(0)

    return


if __name__ == "__main__":
    main()
