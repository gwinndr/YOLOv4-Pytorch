import torch
import torch.nn as nn

from utilities.constants import CHANNEL_DIM

# RouteLayer
class RouteLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - A darknet Route layer
    ----------
    """

    # __init__
    def __init__(self, layers):
        super(RouteLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = True

        self.layers = layers

    # get_required_layers
    def get_required_layers(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Indices of the routed (concatentated depth-wise) layers
        ----------
        """

        return self.layers

    # forward
    def forward(self, x, layer_outputs):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs the route layer
        - Must give a list of layer outputs specified by layers
        - For compatibility, takes in an x input, but does not use it
        ----------
        """

        if(len(layer_outputs) <= 0):
            print("RouteLayer: BUG: Empty list given for layer_outputs")
            return None

        return torch.cat(layer_outputs, dim=CHANNEL_DIM)
