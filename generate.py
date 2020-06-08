import torch
import cv2

from utilities.constants import *

from utilities.devices import gpu_device_name, get_device
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights
from utilities.preprocess import letterbox_image, image_to_tensor

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point for generating labels on a given image
    ----------
    """

    config_path = "./configs/yolov4.cfg"
    weight_path = "./weights/yolov4.weights"

    print("Parsing config into model...")
    model = parse_config(config_path)
    model.cuda()

    print("Loading weights...")
    version, imgs_seen = load_weights(model, weight_path)

    print("")
    print(SEPARATOR)
    print("DARKNET")
    print("GPU:", gpu_device_name())
    print("Config:", config_path)
    print("Weights:", weight_path)
    print("Version:", ".".join([str(v) for v in version]))
    print("Images seen:", imgs_seen)
    print(SEPARATOR)
    print("")

    image = cv2.imread("./examples/eagle.jpg")
    image2 = cv2.imread("./examples/giraffe.jpg")

    class_names = parse_names("./configs/coco.names")

    letterbox = letterbox_image(image, 512)
    letterbox2 = letterbox_image(image2, 512)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.imshow("image", letterbox)
    # cv2.waitKey(0)

    x = torch.stack([image_to_tensor(letterbox), image_to_tensor(letterbox2)])

    detections = model(x)
    for detection in detections:
        for i,batch in enumerate(detection):
            print("IMG:", i)
            for xy in batch:
                for anchor in xy:
                    if(anchor[YOLO_OBJ] > 0.75):
                        # print(anchor[YOLO_CLASS_START:])
                        score, class_idx = anchor[YOLO_CLASS_START:].max(0)
                        print(class_names[class_idx])
                        # print(class_names[int(class_idx)])


if __name__ == "__main__":
    main()
