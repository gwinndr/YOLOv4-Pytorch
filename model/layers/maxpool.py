import torch
import torch.nn as nn

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
