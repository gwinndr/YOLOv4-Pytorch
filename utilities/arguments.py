import argparse

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

    parser.add_argument("--benchmark", action="store_true", help="Benchmark fps on video. Default metric is the MODEL_ONLY approach (mirrors darknet). See -benchmark_method to switch metrics.")
    parser.add_argument("-benchmark_method", type=int, default=MODEL_ONLY, help="Sets benchmark method for --benchmark. Put 1 for MODEL_ONLY (default), 2 for MODEL_WITH_PP, and 3 for MODEL_WITH_IO.")

    parser.add_argument("-cfg", type=str, default="./configs/yolov4.cfg", help="Yolo configuration file")
    parser.add_argument("-weights", type=str, default="./weights/yolov4.weights", help="Yolo weights file")
    parser.add_argument("-class_names", type=str, default="./configs/coco.names", help="Names for each class index")

    parser.add_argument("-obj_thresh", type=float, default=OBJ_THRESH_DEFAULT, help="Confidence threshold for filtering out predictions")

    parser.add_argument("--no_letterbox", action="store_true", help="Turns off image input letterboxing (degrades detector performance)")
    parser.add_argument("--no_show", action="store_true", help="Does not display output image with detections (ignored if --video)")

    parser.add_argument("--print_network", action="store_true", help="Print out each layer in the darknet model with their hyperparameters")
    parser.add_argument("--force_cpu", action="store_true", help="Forces the model to run on the cpu regardless of cuda status")

    return parser.parse_args()

# parse_evaluate_args
def parse_evaluate_args():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Argparse arguments for evaluate.py
    ----------
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-images", type=str, default="./COCO/2017/val2017/", help="Folder containing validation images")
    parser.add_argument("-anns", type=str, default="./COCO/2017/annotations/instances_val2017.json", help="File containing bbox annotations for validation images")
    parser.add_argument("-max_imgs", type=int, default=0, help="Specifies upper bound on the number of validation images processed. If <= 0, all validation images are processed.")

    parser.add_argument("-cfg", type=str, default="./configs/yolov4.cfg", help="Yolo configuration file")
    parser.add_argument("-weights", type=str, default="./weights/yolov4.weights", help="Yolo weights file")

    parser.add_argument("-obj_thresh", type=float, default=OBJ_THRESH_DEFAULT, help="Confidence threshold for filtering out predictions")
    parser.add_argument("--no_letterbox", action="store_true", help="Turns off image input letterboxing (degrades detector performance)")

    parser.add_argument("--print_network", action="store_true", help="Print out each layer in the darknet model with their hyperparameters")
    parser.add_argument("--force_cpu", action="store_true", help="Forces the model to run on the cpu regardless of cuda status")

    return parser.parse_args()
