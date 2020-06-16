# Darknet in Pytorch
Currently supports Pytorch >= 1.5.0  
Verified to work with Yolov4 and Yolov3

![Dog Detection](https://lh3.googleusercontent.com/7fDn_S03jrwJNxWoUIB46VIB2_LXiccuQtS-8Xf8Uk0ooghJV3IcQ_p_r2IoG_CuExD2CXzINWbKfXrW37Rp25O2fxh3WiU7DFVuQJtLVzlBYs4HAnXJKmTHlJLSYNnTRC9ACbd_vA=w2400 "Dog Detection")

## About
This is a reproduction of the Darknet framework in Pytorch with support for YOLO training (TODO) and inferencing. The end goal of this project is to have a clean and full Darknet copy in Pytorch for research purposes (and because I'm bored and wanted a new project).

Currently, Alexey Bochkovskiy's Darknet framework (https://github.com/AlexeyAB/darknet) is the goto repo for all the hot new yolo implementations as Redmon has moved on from Yolo. Ultralytics is doing some awesome work as well, they're up to Yolov5 apparently (https://github.com/ultralytics/yolov5). I would like to keep this repo up to date with all the cool new things that come out of this space.

## TODO
The current TODO list. These are just the immediate tasks I am currently focused on and what you can expect in the near future. It is in no way complete :-)
* Implement YOLO training
* Write a detect_video.py for detection and output with video files
* Write an evaluation script to get baseline mAP on MS-COCO
* Write a script to calculate FPS and timing info for pre-processing, model inferencing, and post-processing

## How to run
The repo is currently set up to do inferencing on single images. Here is an example of how to run it:

```
python detect.py -img ./examples/eagle.jpg -output ./eagle_detect.png
```
There are more arguments you can specify as well, to see the full argument list, run the following:
```
python detect.py --help
```

### Bounding Boxes
Look in utilities.constants under the BBOX DRAWING header to tweak how the bounding boxes are output. Everything from color selection to label padding can be tweaked to how you like it.



