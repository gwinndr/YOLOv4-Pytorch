import torch

from utilities.configs import parse_config
# from utilities.weights import load_weights

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point for generating labels on a given image
    ----------
    """

    config_path = "./configs/yolov3.cfg"
    model = parse_config(config_path)
    model.cuda()

    print(torch.cuda.get_device_name(device=None))

    x = torch.rand((1,3,320,320)).cuda()

    detections = model(x)
    for detection in detections:
        print(detection.shape)

    # load_weights(model, "./weights/yolov3.weights")


if __name__ == "__main__":
    main()
