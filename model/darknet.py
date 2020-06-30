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
    def __init__(self, layer_modules, net_block=None):
        super(Darknet, self).__init__()

        self.layer_modules = layer_modules
        self.net_block = net_block

        self.yolo_layers = []
        for l in self.layer_modules:
            if(isinstance(l, YoloLayer)):
                self.yolo_layers.append(l)

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs the given input through the Yolo model
        - Eval mode: Returns predictions (will need to run utilities.postprocess.extract_detections)
        ----------
        """
        predictions = []
        saved_outputs = []
        input_dim = x.shape[INPUT_H_DIM]

        for i, module in enumerate(self.layer_modules):
            if(module.is_output_layer):
                predictions.append(module(x, input_dim))

            elif(module.requires_layer_outputs):
                x = module(x, saved_outputs)

            else:
                x = module(x)

            # Saving outputs
            saved_outputs.append(x)

        predictions = torch.cat(predictions, dim=YOLO_OUT_N_PREDS_DIM)
        return predictions

    # get_net_block
    def get_net_block(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Returns the net_block
        - net_block is a dictionary containing all key-value pairs under the [net] heading
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
        for i, module in enumerate(self.layer_modules):
            print("   ", i, ":::", module.to_string(), file=f)

        print("", file=f)
        print("END NETWORK PRINT:", file=f)
        print(SEPARATOR, file=f)
        print("", file=f)

        return
