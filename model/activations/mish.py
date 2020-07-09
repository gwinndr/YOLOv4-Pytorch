import torch
import torch.nn as nn

from utilities.constants import *

class Mish(nn.Module):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Implementation of the Mish activation function (https://arxiv.org/ftp/arxiv/papers/1908/1908.08681.pdf)
    - Mish is not present in Pytorch as of 1.5
    ----------
    """

    def __init__(self, beta=MISH_BETA, threshold=MISH_THRESHOLD):
        super(Mish, self).__init__()

        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Runs mish activation on each element of the given input
        ----------
        """

        return x * torch.tanh( self.softplus(x) )
