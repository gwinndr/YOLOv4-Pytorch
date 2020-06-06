import torch
import torch.nn as nn

class Mish(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Implementation of the Mish activation function (https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf)
    - Mish is not present in Pytorch as of 1.5
    ----------
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - Runs mish activation on each element of the given input
        ----------
        """

        return x * torch.tanh( torch.log( 1 + torch.exp(x) ) )
