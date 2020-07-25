import torch
import torch.nn as nn

from utilities.constants import *

# RouteLayer
class RouteLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - A darknet Route layer
    ----------
    """

    # __init__
    def __init__(self, layers):
        super(RouteLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = True
        self.is_output_layer = False

        self.layers = layers

    # get_required_layers
    def get_required_layers(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Indices of the routed (concatentated channel-wise) layers
        ----------
        """

        return self.layers

    # forward
    def forward(self, x, layer_outputs):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs the route layer
        - Takes a list of outputs from all previous layers
        - Takes in an x input, but does not use it (compatibility)
        ----------
        """

        outputs_to_route = []
        for l in self.layers:
            outputs_to_route.append(layer_outputs[l])

        if(len(outputs_to_route) <= 0):
            print("RouteLayer: BUG: No layers to route")
            return None

        return torch.cat(outputs_to_route, dim=INPUT_CHANNEL_DIM)

    # to_string
    def to_string(self):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Converts this layer into a human-readable string
        ----------
        """

        return \
            "ROUT: layers: %s" % \
            (",".join([str(l)for l in self.layers]))
