import torch
import torch.nn as nn

from utilities.constants import *

# UpsampleLayer
class UpsampleLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - A darknet Upsample layer
    ----------
    """

    # __init__
    def __init__(self, stride):
        super(UpsampleLayer, self).__init__()

        self.stride = stride
        self.upsample = nn.Upsample(scale_factor=self.stride, mode=UPSAMPLE_MODE)

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs upsampling on given input
        ----------
        """

        return self.upsample(x)
