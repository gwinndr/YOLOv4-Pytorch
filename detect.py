import torch
import cv2

from utilities.constants import *

from utilities.devices import gpu_device_name, get_device, use_cuda
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights
from utilities.images import preprocess_image_eval, tensor_to_image, map_dets_to_original_image, write_dets_to_image
from utilities.extract_detections import extract_detections

# main
def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    Entry point for generating labels on a given image
    ----------
    """

    with torch.no_grad():
        # use_cuda(False)

        config_path = "./configs/yolov4.cfg"
        weight_path = "./weights/yolov4.weights"
        # config_path = "./configs/yolov3.cfg"
        # weight_path = "./weights/yolov3.weights"

        print("Parsing config into model...")
        model = parse_config(config_path)
        model = model.to(get_device())
        model.eval()

        print("Parsing class names...")
        class_names = parse_names("./configs/coco.names")

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
        # image = cv2.imread("./examples/whale_tiger.jpg")
        image2 = cv2.imread("./examples/giraffe.jpg")
        # print(image.shape)

        obj_threshold = 0.25

        letterbox = preprocess_image_eval(image, 608, letterbox=True)
        letterbox2 = preprocess_image_eval(image2, 608, letterbox=True)

        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.imshow("image", letterbox)
        # cv2.waitKey(0)
        # x = letterbox.unsqueeze(0)
        x = torch.stack([letterbox, letterbox2])

        predictions = model(x)
        detections = extract_detections(predictions, model.get_yolo_layers())

        # new_image = get_bbox_image(detections[0], tensor_to_image(letterbox), class_names)
        # new_image = get_bbox_image(detections[1], tensor_to_image(letterbox2), class_names)
        # cv2.imshow("image", new_image)
        # cv2.waitKey(0)

        detections[0] = map_dets_to_original_image(detections[0], image.shape[0], image.shape[1], 608)
        new_image = write_dets_to_image(detections[0], image, class_names, verbose_output=True)
        # detections[1] = bbox_letterbox_to_image(detections[1], image2.shape[0], image2.shape[1], 608)
        # new_image = get_bbox_image(detections[1], image2, class_names)

        cv2.imshow("image", new_image)
        cv2.waitKey(0)
        # cv2.imwrite("./detection.png", new_image)


        # for detection in detections:
        #     for i,batch in enumerate(detection):
        #         print("IMG:", i)
        #         for pred in batch:
        #             if(pred[YOLO_OBJ] > obj_threshold):
        #                 # print(anchor[YOLO_CLASS_START:])
        #                 score, class_idx = pred[YOLO_CLASS_START:].max(0)
        #                 print(class_names[class_idx])
        #                 # print(class_names[int(class_idx)])




if __name__ == "__main__":
    main()
