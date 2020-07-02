import torch

from utilities.constants import *

from utilities.arguments import parse_evaluate_args
from datasets.coco import CocoDataset, coco_evaluate_bbox

from utilities.devices import gpu_device_name, get_device, use_cuda
from utilities.configs import parse_config
from utilities.weights import load_weights

def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for evaluating a darknet model
    ----------
    """

    args = parse_evaluate_args()

    # no_grad disables autograd so our model runs faster
    with torch.no_grad():
        if(args.force_cpu):
            print("----- WARNING: Model is using the CPU (--force_cpu), expect model to run slower! -----")
            use_cuda(False)

        print("Parsing config into model...")
        model = parse_config(args.cfg)
        if(model is None):
            return

        model = model.to(get_device())
        model.eval()

        # TODO
        network_dim = int(model.net_block["width"])

        print("Loading weights...")
        version, imgs_seen = load_weights(model, args.weights)

        print("")
        print(SEPARATOR)
        print("DARKNET")
        print("GPU:", gpu_device_name())
        print("Config:", args.cfg)
        print("Weights:", args.weights)
        print("Version:", ".".join([str(v) for v in version]))
        print("Images seen:", imgs_seen)
        print("Network Dim:", network_dim)
        print(SEPARATOR)
        print("")

        # Print network
        if(args.print_network):
            model.print_network()

        image_dir = args.images
        ann_file = args.anns
        max_imgs = args.max_imgs
        obj_thresh = args.obj_thresh
        letterbox = not args.no_letterbox

        val_dataset = CocoDataset(image_dir, ann_file)
        print("")

        coco_evaluate_bbox(val_dataset, model, network_dim, obj_thresh, letterbox, max_imgs)

    return


if __name__ == "__main__":
    main()
