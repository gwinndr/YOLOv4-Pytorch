import torch
import torch.nn as nn
import copy

from utilities.constants import *

from .layers.yolo import YoloLayer
from .layers.convolutional import ConvolutionalLayer

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
        self.layer_outputs_needed = generate_required_layer_dict(layer_modules)

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
        layer_count_dict = copy.deepcopy(self.layer_outputs_needed)
        saved_outputs = dict()
        input_dim = x.shape[X_DIM]

        # print("INPUT:", x[0,0,0,0])

        for i, module in enumerate(self.layer_modules):
            # Running the model layer
            if(isinstance(module, YoloLayer)):
                predictions.append(module(x, input_dim))
                # return predictions

            elif(module.requires_layer_outputs):
                req_layers = module.get_required_layers()
                layer_outputs = extract_layer_outputs(i, req_layers, layer_count_dict, saved_outputs)

                x = module(x, layer_outputs)

            else:
                x = module(x)

            # f = ": %.2f" % float(x[0,0,0,0])
            # print(i, type(module).__name__, f)

            # Saving needed outputs
            if(i in layer_count_dict.keys()):
                saved_outputs[i] = x

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


# generate_required_layer_dict
def generate_required_layer_dict(layer_modules):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Returns a dictionary mapping layer numbers to the number of layers that need them
    - Used for tracking when outputs are needed more than once to save on memory
    - For example, if three separate layers need layer two's output, the dict entry for 2 will contain 3
    ----------
    """

    layer_dict = dict()
    for i, module in enumerate(layer_modules):
        if(module.requires_layer_outputs):
            layers = module.get_required_layers()

            for l in layers:
                if(l == 0):
                    print("generate_required_layer_dict: Error: Layer", i, "includes '0' in layer list")
                    return None
                if(l < 0):
                    l += i

                if(l in layer_dict.keys()):
                    layer_dict[l] += 1
                else:
                    layer_dict[l] = 1
    # for

    return layer_dict

# extract_layer_outputs
def extract_layer_outputs(cur_layer, layers, layer_count_dict, layer_output_dict):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Extracts given layers into a list of outputs from those layers
    - Automatically removes layer outputs from memory when layer_dict counter goes to 0
    - Used to get layer outputs needed for layers such as shortcut and route
    ----------
    """

    layer_outputs = []
    for l in layers:
        if(l < 0):
            l += cur_layer

        layer_outputs.append(layer_output_dict[l])

        # Freeing memory when the output is no longer needed
        layer_count_dict[l] -= 1
        if(layer_count_dict[l] == 0):
            del layer_count_dict[l]

    return layer_outputs
