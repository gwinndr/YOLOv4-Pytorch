# Darknet in Pytorch
Currently supports Pytorch 1.5.1  
Verified to work with Yolov4 and Yolov3 (probably Yolov2 as well)  

Please give the full error message when reporting an issue (contains reproduction information)

Please click on [Releases](https://github.com/gwinndr/YOLOv4-Pytorch/releases) to get the most up to date and stable release.

![Dog Detection](https://lh3.googleusercontent.com/OyZTbeMh7E5C5LUMmWkfdgxFs38FTV7KQlHGir9Y-HNE1VJhnh80iMmem2Emdaq4P_u-jKSOFlQJ1PBut3mdiIZhbQqrqPQ7JNrZd9p-tkYDKadOd_leS7b2GIIwdO-L2GH7u_E1CQ=w2400 "Dog Detection")

## Requirements
### Pytorch
First install [pytorch](https://pytorch.org/get-started/locally/). I only support 1.5.1 at the moment.

Basically, if you have [CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base), run:
```
pip install torch==1.5.1+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pycocotools (Training and Evaluation)
If on Windows, run:
```
pip install pycocotools-windows
```
If on Linux, run:
```
pip install pycocotools
```

### The rest
After doing the above, run:
```
pip install -r requirements.txt
```

## About
This is a reproduction of the Darknet framework in Pytorch with support for YOLO training and inferencing. The end goal of this project is to have a pytorch implementation of all darknet layers and features. This will include not only the detector portion which is currently finished, but will also include the pre-training on ImageNet which is my next milestone. I used Alexey's Darknet framework to build this repository and verified (painstakingly at times) that they work semi-equivalently with deviations usually resulting from differences between pytorch and darknet ML layers in terms of precision.

Currently, Alexey Bochkovskiy's [Darknet framework](https://github.com/AlexeyAB/darknet) is the goto repo for all the hot new yolo stuff as Redmon has moved on from Yolo. Ultralytics is doing some awesome work as well, they're up to [Yolov5 apparently](https://github.com/ultralytics/yolov5). I would like to keep this repo up to date with all the cool new stuff that come out of this space, or at least I'll try to.

## How we got here
Short story. After wrapping up my graduate degree, I had the entire summer off until starting a new job. Paradise got boring after about 2 weeks and I decided I needed something to do. That something was YOLO. 

This is not the first time I've implemented YOLO, but I couldn't share my other one so here we are. The nice thing about this time around is that I could pretty much develop this "from the ground up". I could really take my time with certain things. Of course this can be a double edged sword because it's really easy to be lazy and do nothing all day when technically on vacation.

Another nice thing, is that I developed this alongside Alexey's Darknet. This was a good idea, as I discovered that I had actually done jitter wrong the first time (Josh, you should fix this if you haven't).

TLDR: I was bored, and this is sort of a capstone project to my collegiate career. Hope you like it :-).

## TODO
The current [TODO list](https://docs.google.com/document/d/1WvkFzX29_vPRy2sYOeLbDnoJ4yiQsT7GAHyvyxOsC2w/edit?usp=sharing). These are just the immediate tasks I am currently focused on and what you can expect in the near future. It is in no way complete.

## Weights
You will have to download pre-trained weights. Some download urls are listed below:
* [Yolov4](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
* [Yolov4 Pretrained](https://drive.google.com/file/d/13kN5sb0jJtoP9XVKu9Y2_Fqwo0BSz-Yj/view?usp=sharing)
* [Yolov3-spp](https://pjreddie.com/media/files/yolov3-spp.weights)
* [Yolov3](https://pjreddie.com/media/files/yolov3.weights)
* [Yolov3-tiny](https://pjreddie.com/media/files/yolov3-tiny.weights)

The configurations for these models are already included in the configs folder. There is also a copy of the coco class names there.


## How to run
You can change the network input dimensions by editing the width and height parameters in the given config file.   
Currently, I only support width=height dimensions to keep it simple. 

There's a bunch more stuff (mostly training related) in the config for you to edit if you want.

### Inferencing:
Right, so first off, you can get a list of the arguments by running:
```
python detect.py --help
```

To inference on a single image, you can go ahead and run:
```
python detect.py -input ./examples/eagle.jpg -output ./eagle_detect.png
```

That's cool, but maybe you want to run Yolov3? In that case:
```
python detect.py -cfg ./configs/yolov3 -weights ./weights/yolov3.weights
```

You can do benchmarking here as well if you have a video. Go ahead and run:
```
python detect.py -input ./vid.mp4  --video --benchmark
```
You can specify other benchmarking metrics, but I'd recommend the default which to just benchmark the model only.

### Train on COCO:
Training takes about a century give or take just to warn you. Firstly, you'll need to download [MS-COCO](https://cocodataset.org/#download). Get the 2017 version. At minimum you'll want train images, val images, and the train/val annotations.

Again, to get arguments, run:
```
python train.py --help
```

To train with coco on the defaults, I recommend you run:
```
python train.py -results ./results/default --epoch_csv --tensorboard
```
This will assume COCO is in ./COCO/2017, but you can change this with
```
python train.py -train_imgs .../train2017 -train_anns .../instances_train2017.json -val_imgs ... -val_anns ...
```

This will create a results folder. In it are info.txt which lists some training hyperparameters and the [net] block, a copy of the config you're using, and the weights for each epoch. If you run with --epoch_csv and --tensorboard, the repository will additionally write out mAP and mAR to an epoch csv and also write batch loss and epoch evaluation to tensorboard. I recommend you don't use --batch_csv as it uses quite a bit of storage space.

To get the neat visual loss diagrams with tensorboard, run:
```
tensorboard --logdir=./results/default/tensorboard
```

It will print out a url that you open in your preferred browser to see all the statistics.

### Training Minibatch Size:
We don't have infinite memory so some sacrifices have to be made. 

To change the minibatch size, you'll want to open the yolo config and edit the subdivisions parameter.  
I'll copy Alexey's recommendations here for **subdivision count** based on GPU memory:
| 8 - 12 GB | 16 - 24 GB | 32 GB |
|---|---|---|
| 32 subdivs  | 16 subdivs | 8 subdivs |

If you fall within the gpu range and are still getting out of memory errors, change **random=1.2** in the config (last yolo layer). That fixes it by making the range of input sizes during training a bit smaller. Obviously not the most ideal, but a higher minibatch size is probably better.

### Evaluate on COCO:
Evaluation gives you mAP and mAR on a validation set.

You know the drill, to get arguments, run:
```
python evaluate.py --help
```

To evaluate, you can run:
```
python evaluate.py -cfg ./configs/yolov4.cfg -weights ./weights/myweight.weights
```
This will assume COCO is in ./COCO/2017, but you can change this with
```
python evaluate.py -images .../val2017 -anns .../instances_val2017.json
```

## Results
As I mentioned, training takes forever. I only got a home computer to work with here. I will update this with results when it finishes. Once I got the results that prove I'm not crazy and it actually does work, I'll do the first 1.0 release!

It looked promising mAP-wise when I did some training (about 25 epochs) without any augmentations. I'm pretty optimistic this works.

I'll update with a "results so far" probably in a week or two.

## Benchmarks
### FPS on Nvidia GeForce 2070 Super 
| 608x608 | 512x512 | 416x416 | 320x320 | 
|---|---|---|---|
| 22.94  | 28.06  | 29.22  | 29.75 |

Does it run as fast as the original Darknet? No... But it's Python vs. C, what did you expect?  
As a note though, on my list for the far future, I'd like to do a C++ libtorch inferencer at some point. I'd be curious to see the speed.

## Contributing
Yeah, so let me know what you're fixing and why. Follow the style that's throughout and make sure you list yourself as a contributing author in the corresponding function header. I'd prefer that you have an issue corresponding to your contribution as well.

If you have a DGX or similarly ridiculously expensive piece of supercomputer, please give me benchmarks!



