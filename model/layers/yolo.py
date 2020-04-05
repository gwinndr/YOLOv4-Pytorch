import torch
import torch.nn as nn

# The big cheese
# YoloLayer
class YoloLayer(nn.Module):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - A darknet Yolo layer
    - Seriously though, why is jitter and random in this layer? It makes no sense...
    ----------
    """

    # __init__
    def __init__(self, anchors, n_classes, ignore_thresh, truth_thresh):
        super(YoloLayer, self).__init__()

        self.anchors = anchors
        self.n_classes = n_classes
        self.ignore_thresh = ignore_thresh
        self.truth_thresh = truth_thresh

    # forward
    def forward(self, x):
        """
        ----------
        Author: Damon Gwinn
        ----------
        - YoloLayer, just a placeholder for now
        ----------
        """

        return x
