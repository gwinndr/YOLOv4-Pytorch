import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities.constants import *

# MaxpoolLayer
class MaxpoolLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - A darknet Maxpool layer
    ----------
    """

    # __init__
    def __init__(self, size, stride):
        super(MaxpoolLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False

        self.size = size
        self.stride = stride

        # Set to true when size is even and stride is 1
        self.special_pad = False

        # Have to handle even kernel sizes for yolov3-tiny
        # May be more edge cases to handle (TODO)
        # Fix thanks to Ayoosh Kathuria (https://github.com/ayooshkathuria/pytorch-yolo-v3)
        if(self.size % 2 == 0):
            if(self.stride == 1):
                self.special_pad = True
                self.padding = self.size - 1
                self.maxpool = nn.MaxPool2d(self.size, stride=self.padding)
            else:
                self.padding = 0
                self.maxpool = nn.MaxPool2d(self.size, stride=self.stride, padding=self.padding)

        # The normal case which will be the vast majority of cases
        else:
            self.padding = (self.size - 1) // 2
            self.maxpool = nn.MaxPool2d(self.size, stride=self.stride, padding=self.padding)



    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs maxpooling on given input
        ----------
        """

        if(self.special_pad):
            pad_right_bottom = (0, self.padding, 0, self.padding)
            x = F.pad(x, pad_right_bottom, mode=POOL_SPECIAL_PAD_MODE)

        return self.maxpool(x)

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
            "MXPL: size: %d  stride: %d  pad: %d" % \
            (self.size, self.stride, self.padding)
