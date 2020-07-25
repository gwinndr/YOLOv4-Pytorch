import torch
import torch.nn as nn
import sys

from utilities.constants import *

from .layers.yolo import YoloLayer

# Darknet
class Darknet(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - The darknet model
    - Supports Yolov4 training and inferencing
    ----------
    """

    # __init__
    def __init__(self, layer_modules, net_block=None, version="0.0.0", imgs_seen=0):
        super(Darknet, self).__init__()

        self.layer_modules = layer_modules
        self.net_block = net_block
        self.version = version
        self.imgs_seen = imgs_seen

        self.yolo_layers = []
        for l in self.layer_modules:
            if(isinstance(l, YoloLayer)):
                self.yolo_layers.append(l)

    # forward
    def forward(self, x, anns=None):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs the given input through the Yolo model
        - Train mode: Returns loss based on given annotations
        - Eval mode: Returns predictions
        ----------
        """

        model_out = []
        saved_outputs = []
        input_dim = x.shape[INPUT_H_DIM]

        for i, module in enumerate(self.layer_modules):
            if(module.is_output_layer):
                model_out.append(module(x, input_dim, anns=anns))

            elif(module.requires_layer_outputs):
                x = module(x, saved_outputs)

            else:
                x = module(x)

            # Saving outputs
            saved_outputs.append(x)

        if(anns is None):
            model_out = torch.cat(model_out, dim=PREDS_N_PREDS_DIM)

        return model_out

    # get_net_block
    def get_net_block(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns the net_block
        - net_block is a NetBlock class under model.net_block
        ----------
        """

        return self.net_block

    # get_yolo_layers
    def get_yolo_layers(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Gets yolo layers in order
        ----------
        """

        return self.yolo_layers

    # get_layers
    def get_layers(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns module_list of layers
        ----------
        """

        return self.layer_modules

    # print_network
    def print_network(self, f=sys.stdout):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Prints each layer of this darknet model
        ----------
        """

        print(SEPARATOR, file=f)
        print("START NETWORK PRINT:", file=f)
        print("", file=f)

        print(self.net_block.to_string(), file=f)
        print("", file=f)

        for i, module in enumerate(self.layer_modules):
            print("   ", i, ":::", module.to_string(), file=f)

        print("", file=f)
        print("END NETWORK PRINT:", file=f)
        print(SEPARATOR, file=f)
        print("", file=f)

        return
