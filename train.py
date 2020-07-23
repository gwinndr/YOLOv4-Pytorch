import torch
from torch.utils.data import DataLoader

from utilities.constants import *
from utilities.devices import get_device
from utilities.arguments import parse_train_args
from utilities.configs import parse_config
from utilities.weights import load_weights

from utilities.augmentations import possible_image_sizings

from datasets.coco import CocoDataset

def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for training a darknet model
    ----------
    """

    args = parse_train_args()


    # train_imgs = "D:/Datasets/COCO/2017/train2017"
    # train_anns = "D:/Datasets/COCO/2017/annotations/instances_train2017.json"
    train_imgs = "D:/Datasets/COCO/2017/val2017"
    train_anns = "D:/Datasets/COCO/2017/annotations/instances_val2017.json"

    print("Parsing config into model...")
    model = parse_config(args.cfg)
    if(model is None):
        return

    model = model.to(get_device())
    model.train()

    model.print_network()

    # net = model.net_block
    #
    # # Network input dim
    # if(net["width"] != net["height"]):
    #     print("Error: Width and height must match in [net]")
    #     return
    #
    # print("Loading weights...")
    # version, imgs_seen = load_weights(model, args.weights)
    #
    # network_dim = int(net["width"])
    #
    # # Training and validation sets
    # train_set = CocoDataset(train_imgs, input_dim=network_dim, letterbox=True, annotation_file=train_anns)
    # print("")
    #
    # batch_size = int(net["batch"])
    # subdiv = int(net["subdivisions"])
    # subdiv_batch_size = batch_size // subdiv
    #
    # # Learn rate is divided by the batch size, so this will be our scaling term
    # initial_lr = 1.0 / batch_size
    #
    # train_loader = DataLoader(train_set, batch_size=subdiv_batch_size, shuffle=False)



    # print(SEPARATOR)
    # print("Error: %.4f" % (sum(out).item()/3))

    return


if __name__ == "__main__":
    main()
