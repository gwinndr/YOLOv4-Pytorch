import torch
import torch.nn as nn

from utilities.constants import *

# ConvolutionalLayer
class ConvolutionalLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - A darknet Convolutional Layer
    - If padding is true, zero padding is applied as (size - 1) // 2 to preserve dimensions
    ----------
    """

    # __init__
    def __init__(self, in_channels, out_channels, size, stride, batch_normalize=CONV_DEFAULT_BN,
            padding=CONV_DEFAULT_PAD, activation=CONV_DEFAULT_ACTIV):
        super(ConvolutionalLayer, self).__init__()

        self.size = size
        self.stride = stride
        self.out_channels = out_channels
        self.batch_normalize = batch_normalize

        self.sequential = nn.Sequential()

        if(padding):
            self.padding = (self.size - 1) // 2
        else:
            self.padding = 0

        conv = nn.Conv2d(in_channels, out_channels, size, stride=stride, padding=self.padding)
        self.sequential.add_module("conv_2d", conv)

        # TODO: Are the default hyperparameters in Pytorch BatchNorm the same in Darknet?
        if(batch_normalize):
            bn = nn.BatchNorm2d(out_channels)
            self.sequential.add_module("batch_norm", bn)

        # Leaky Relu
        if(activation == "leaky"):
            leaky = nn.LeakyReLU(negative_slope=YOLO_LEAKY_SLOPE, inplace=True)
            self.sequential.add_module("leaky_relu", leaky)

        # Error case
        elif((activation != "linear") and (activation is not None)):
                print("ConvolutionalLayer: WARNING: Ignoring unrecognized activation:", activation)

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs the convolutional layer on the given input
        ----------
        """

        return self.sequential(x)
