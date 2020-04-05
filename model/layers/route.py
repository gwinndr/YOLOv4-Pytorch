import torch
import torch.nn as nn

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

        self.layers = layers

    # get_route_layers
    def get_route_layers(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Absolute indices of the routed (concatentated depth-wise) layers
        ----------
        """

        return self.layers

    # forward
    def forward(self, layer_outputs):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs the route layer
        - Must give a list of layer outputs specified by layers
        ----------
        """

        if(len(layer_outputs) <= 0):
            print("RouteLayer: BUG: Empty list given for layer_outputs")
            return None

        return torch.cat(layer_outputs, dim=CHANNEL_DIM)
