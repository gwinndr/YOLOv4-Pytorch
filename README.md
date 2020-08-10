# Darknet in Pytorch
Currently supports Pytorch 1.5.1  
Verified to work with Yolov4 and Yolov3 (probably Yolov2 as well)

![Dog Detection](https://lh3.googleusercontent.com/OyZTbeMh7E5C5LUMmWkfdgxFs38FTV7KQlHGir9Y-HNE1VJhnh80iMmem2Emdaq4P_u-jKSOFlQJ1PBut3mdiIZhbQqrqPQ7JNrZd9p-tkYDKadOd_leS7b2GIIwdO-L2GH7u_E1CQ=w2400 "Dog Detection")

## About
This is a reproduction of the Darknet framework in Pytorch with support for YOLO training and inferencing. The end goal of this project is to have a pytorch implementation of all darknet layers and features. This will include not only the detector portion which is currently finished, but will also include the pre-training on ImageNet which is my next milestone. I used Alexey's Darknet framework to build this repository and verified (painstakingly at times) that they work semi-equivalently with deviations usually resulting from differences between pytorch and darknet ML layers in terms of precision.

Currently, Alexey Bochkovskiy's [Darknet framework](https://github.com/AlexeyAB/darknet) is the goto repo for all the hot new yolo stuff as Redmon has moved on from Yolo. Ultralytics is doing some awesome work as well, they're up to [Yolov5 apparently](https://github.com/ultralytics/yolov5). I would like to keep this repo up to date with all the cool new stuff that come out of this space, or at least I'll try to.

## How we got here
Short story. After wrapping up my graduate degree, I had the entire summer off until starting a new job. Paradise got boring after about 2 weeks and I decided I needed something to do. That something was YOLO. 

This is not the first time I've implemented YOLO, but I couldn't share my other one so here we are. The nice thing about not being on a company's time, is that I could pretty much develop this "from the ground up". Without a time and money crunch, I could really take my time with certain things which is always a plus. Of course this can be a double edged sword because it's really easy to be lazy and do nothing all day when there's no one to wave money in your face, but what can you do? 

Another nice thing, is this time around, I developed this alongside Alexey's Darknet. This was a good idea, as I discovered that I had actually done jitter wrong the first time around (Josh, you should fix this if you haven't). I'm glad to have been able to do everything right this time and have something special for my Github account.

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

This will create a results folder. In it are info.txt which the information on the training hyperparameter, a copy of the config you're using, and the weights for each epoch. If you run with --epoch_csv and --tensorboard, the repository will additionally write out mAP and mAR to an epoch csv and also write batch loss and epoch evaluation to tensorboard. I recommend you don't use --batch_csv as it uses quite a bit of storage space.

To get the neat visual loss diagrams with tensorboard, run:
```
tensorboard --logdir=./results/default/tensorboard
```

It will print out a url that you open in your preferred browser to see all the statistics.

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

I'll update with a "results so far" probably in a week or two.

## Benchmarks
### FPS on Nvidia GeForce 2070 Super 
| 608x608 | 512x512 | 416x416 | 320x320 | 
|---|---|---|---|
| 22.94  | 28.06  | 29.22  | 29.75 |

Does it run as fast as the original Darknet? No... But it's Python vs. C, what did you expect?  
As a note though, on my list for the far future, I'd like to do a C++ libtorch inferencer at some point. I'd be curious to see the speed.

## Contributing
Yeah, so let me know what you're fixing and why. Follow the style that's throughout and make sure you list yourself as a contributing offer in the corresponding function header. I'd prefer that you have an issue corresponding to your contribution as well.

If you have a DGX or similarly ridiculously expensive piece of supercomputer, please give me benchmarks!



