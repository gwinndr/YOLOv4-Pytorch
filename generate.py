import torch

from utilities.constants import *

from utilities.devices import gpu_device_name
from utilities.configs import parse_config
from utilities.weights import load_weights

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

    x = torch.rand([1,3,320,320]).cuda()
    detections = model(x)
    for detection in detections:
        print(detection.shape)


if __name__ == "__main__":
    main()
