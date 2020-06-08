import torch
import torch.nn as nn
import copy

from utilities.constants import *

from .layers.yolo import YoloLayer

# Darknet
class Darknet(nn.Module):
    """
    ----------
    Author: Damon Gwinn
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

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs the given input through the Yolo model
        - Eval mode: Returns detections
        ----------
        """
        detections = []
        layer_count_dict = copy.deepcopy(self.layer_outputs_needed)
        saved_outputs = dict()
        input_dim = x.shape[X_DIM]

        for i, module in enumerate(self.layer_modules):
            # Running the model layer
            if(isinstance(module, YoloLayer)):
                detections.append(module(x, input_dim))
                # return detections

            elif(module.requires_layer_outputs):
                req_layers = module.get_required_layers()
                layer_outputs = extract_layer_outputs(i, req_layers, layer_count_dict, saved_outputs)

                x = module(x, layer_outputs)

            else:
                x = module(x)

            # Saving needed outputs
            if(i in layer_count_dict.keys()):
                saved_outputs[i] = x

        return detections


    # get_net_block
    def get_net_block(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Returns the net_block
        - net_block is a dictionary containing all key-value pairs under the [net] heading
        ----------
        """

        return self.net_block

    # get_layers
    def get_layers(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Returns module_list of layers
        ----------
        """

        return self.layer_modules


# generate_required_layer_dict
def generate_required_layer_dict(layer_modules):
    """
    ----------
    Author: Damon Gwinn
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
    Author: Damon Gwinn
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
