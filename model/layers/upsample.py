import torch
import torch.nn as nn

from utilities.constants import *

# UpsampleLayer
class UpsampleLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - A darknet Upsample layer
    ----------
    """

    # __init__
    def __init__(self, stride):
        super(UpsampleLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False

        self.stride = stride
        self.upsample = nn.Upsample(scale_factor=self.stride, mode=UPSAMPLE_MODE)

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs upsampling on given input
        ----------
        """

        return self.upsample(x)
