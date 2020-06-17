import argparse
import sys

from utilities.constants import *

# parse_detect_args
def parse_detect_args():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Argparse arguments for detect.py
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-input", type=str, default="./examples/dog.jpg", help="Image file to process (video file if --video)")
    parser.add_argument("-output", type=str, default="./detections.png", help="Output image file (video out if --video)")

    parser.add_argument("--video", action="store_true", help="Specifies input and output are video files")

    parser.add_argument("-cfg", type=str, default="./configs/yolov4.cfg", help="Yolo configuration file")
    parser.add_argument("-weights", type=str, default="./weights/yolov4.weights", help="Yolo weights file")
    parser.add_argument("-class_names", type=str, default="./configs/coco.names", help="Names for each class index")

    parser.add_argument("-obj_thresh", type=float, default=OBJ_THRESH_DEFAULT, help="Confidence threshold for filtering out predictions")

    parser.add_argument("--no_letterbox", action="store_true", help="Turns off image input letterboxing (degrades detector performance)")
    parser.add_argument("--no_show", action="store_true", help="Does not display output image with detections (ignored if --video)")

    parser.add_argument("--print_network", action="store_true", help="Print out each layer in the darknet model with their hyperparameters")
    parser.add_argument("--force_cpu", action="store_true", help="Forces the model to run on the cpu regardless of cuda status")

    return parser.parse_args()
