import torch
import torch.nn as nn

from utilities.constants import *

from model.activations.mish import Mish

# ConvolutionalLayer
class ConvolutionalLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - A darknet Convolutional Layer
    - If pad is true, zero pad is applied as (size - 1) // 2 to preserve dimensions
    ----------
    """

    # __init__
    def __init__(self,
        in_channels, filters=CONV_FILT_DEF, size=CONV_SIZE_DEF, stride=CONV_STRIDE_DEF,
        batch_normalize=CONV_BN_DEF, pad=CONV_PAD_DEF, activation=CONV_ACTIV_DEF):

        super(ConvolutionalLayer, self).__init__()

        self.has_learnable_params = True
        self.requires_layer_outputs = False
        self.is_output_layer = False

        self.size = size
        self.stride = stride
        self.in_channels = in_channels
        self.filters = filters
        self.batch_normalize = batch_normalize
        self.activation = activation

        self.sequential = nn.Sequential()

        if(pad):
            self.padding = (self.size - 1) // 2
        else:
            self.padding = 0

        # Only one bias term which is applied to conv if no bn, bn otherwise
        bias = not batch_normalize

        conv = nn.Conv2d(in_channels, filters, size, stride=stride, padding=self.padding, bias=bias)
        self.sequential.add_module("conv_2d", conv)

        # Batch Normalizations
        if(batch_normalize):
            bn = nn.BatchNorm2d(filters)
            self.sequential.add_module("batch_norm", bn)

        # Leaky Relu
        if(activation == "leaky"):
            leaky = nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE, inplace=True)
            self.sequential.add_module("leaky_relu", leaky)

        # Mish
        elif(activation == "mish"):
            mish = Mish()
            self.sequential.add_module("mish", mish)

        # Error case
        elif((activation != "linear") and (activation is not None)):
                print("ConvolutionalLayer: WARNING: Ignoring unrecognized activation:", activation)


    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs the convolutional layer on the given input
        ----------
        """

        return self.sequential(x)

    # load_weights
    def load_weights(self, weight_data, start_pos=0):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Loads weights in weight_data starting at start_pos
        - Returns the new position in the list after loading all needed weights
        ----------
        """

        cur_pos = start_pos
        conv = self.sequential[0]

        if(self.batch_normalize):
            bn = self.sequential[1]

            # Number of weight parameters
            n_bias = bn.bias.numel()
            n_weights = bn.weight.numel()
            n_running_mean = bn.running_mean.numel()
            n_running_var = bn.running_var.numel()
            # print(n_bias, n_weights, n_running_mean, n_running_var)

            # Loading some numbers
            bn_bias = torch.from_numpy(weight_data[cur_pos : cur_pos + n_bias])
            cur_pos += n_bias

            bn_weights = torch.from_numpy(weight_data[cur_pos : cur_pos + n_weights])
            cur_pos += n_weights

            bn_running_mean = torch.from_numpy(weight_data[cur_pos : cur_pos + n_running_mean])
            cur_pos += n_running_mean

            bn_running_var = torch.from_numpy(weight_data[cur_pos : cur_pos + n_running_var])
            cur_pos += n_running_var

            # Shaping numbers into pytorch form
            bn_bias = bn_bias.view_as(bn.bias.data)
            bn_weights = bn_weights.view_as(bn.weight.data)
            bn_running_mean = bn_running_mean.view_as(bn.running_mean.data)
            bn_running_var = bn_running_var.view_as(bn.running_var.data)
            # print(":")
            # print(self.filters)
            # print(bn_bias.shape)
            # print(bn_weights.shape)
            # print(bn_running_mean.shape)
            # print(bn_running_var.shape)

            bn.bias.data.copy_(bn_bias)
            bn.weight.data.copy_(bn_weights)
            bn.running_mean.data.copy_(bn_running_mean)
            bn.running_var.data.copy_(bn_running_var)

        # Just a bias for convolutional
        else:
            n_bias = conv.bias.numel()

            conv_bias = torch.from_numpy(weight_data[cur_pos : cur_pos + n_bias])
            cur_pos += n_bias

            conv_bias = conv_bias.view_as(conv.bias.data)
            conv.bias.data.copy_(conv_bias)

        # Load weights for conv_2d
        n_weights = conv.weight.numel()

        conv_weights = torch.from_numpy(weight_data[cur_pos : cur_pos + n_weights])
        cur_pos += n_weights

        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)

        return cur_pos

    # write_weights
    def write_weights(self, o_stream):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Saves weights to o_stream
        - o_stream should be an open stream for writing binary ("wb" mode)
        ----------
        """

        conv = self.sequential[0]

        if(self.batch_normalize):
            bn = self.sequential[1]

            bn.bias.data.cpu().numpy().tofile(o_stream)
            bn.weight.data.cpu().numpy().tofile(o_stream)
            bn.running_mean.data.cpu().numpy().tofile(o_stream)
            bn.running_var.data.cpu().numpy().tofile(o_stream)

        # Just a bias for convolutional
        else:
            conv.bias.data.cpu().numpy().tofile(o_stream)

        # Write weights for conv_2d
        conv.weight.data.cpu().numpy().tofile(o_stream)

        return

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
            "CONV: size: %d  stride: %d  in_c: %d  out_c: %d  bn: %d  pad: %d  activ: %s" % \
            (self.size, self.stride, self.in_channels, self.filters, self.batch_normalize, self.padding, self.activation)
