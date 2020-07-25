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
    def __init__(self, size=MAXPL_SIZE_DEF, stride=MAXPL_STRIDE_DEF, padding=MAXPL_PAD_DEF):
        super(MaxpoolLayer, self).__init__()

        self.has_learnable_params = False
        self.requires_layer_outputs = False
        self.is_output_layer = False

        self.size = size
        self.stride = stride

        # Darknet defines the full pad for top-down and left-right
        # MaxPool2d defines it as padding per each individual side (half the top-down/left-right)
        self.padding = padding
        self.half_padding = padding // 2

        # Set to true when size is even and stride is 1
        # When special_pad we pad the right and the bottom by the full pad. It works for some reason.
        self.special_pad = False

        # Have to handle kernel of size 2 and stride 1 for yolov3-tiny
        # May be more edge cases to handle, requires experimentation (TODO)
        # Fix thanks to Ayoosh Kathuria (https://github.com/ayooshkathuria/pytorch-yolo-v3)
        if((self.size == 2) and (self.stride == 1)):
            self.special_pad = True
            self.maxpool = nn.MaxPool2d(self.size, stride=self.padding)

        # The normal case which will be the vast majority of cases
        else:
            self.maxpool = nn.MaxPool2d(self.size, stride=self.stride, padding=self.half_padding)



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
            x = F.pad(x, pad_right_bottom, mode="replicate")

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
            (self.size, self.stride, self.half_padding)
