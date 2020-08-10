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
    def __init__(self, stride=UPSAMP_STRIDE_DEF, mode=UPSAMP_MODE_DEF):
        super(UpsampleLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False
        self.is_output_layer = False

        self.stride = stride
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=stride, mode=mode)

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
            "UPSM: stride: %d  mode: %s" % \
            (self.stride, self.mode)
