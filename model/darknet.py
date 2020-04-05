import torch
import torch.nn as nn

# Darknet
class Darknet(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - The darknet model
    - Supports Yolov3 training and inferencing
    ----------
    """

    # __init__
    def __init__(self, layer_modules, net_block=None):
        super(Darknet, self).__init__()

        self.layer_modules = layer_modules
        self.net_block = net_block

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
