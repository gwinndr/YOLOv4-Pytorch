from .constants import *

# ImageInfo
class ImageInfo:
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Contains information about the image being input into darknet
    - This class is used when inferencing to map detections back to the original image
    ----------
    """

    # __init__
    def __init__(self, image, network_dim):
        self.image = image
        self.network_dim = network_dim

        # Original image width and height
        self.img_h = image.shape[CV2_H_DIM]
        self.img_w = image.shape[CV2_W_DIM]

        # Letterboxing offset
        self.h_offset = 0
        self.w_offset = 0

        # Height and width of image embedded within the letterbox (or just the whole input tensor)
        self.embed_h = network_dim
        self.embed_w = network_dim

    # set_letterbox
    def set_letterbox(self, h_offset, w_offset, embed_h, embed_w):
        """
        ----------
        Author: Damon Gwinn (gwinndr)
        ----------
        - Sets letterboxing information for the image embedding
        ----------
        """

        self.h_offset = h_offset
        self.w_offset = w_offset
        self.embed_h = embed_h
        self.embed_w = embed_w
