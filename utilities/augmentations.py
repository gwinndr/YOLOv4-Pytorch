import torch
import cv2
import numpy as np

from utilities.constants import *
from utilities.images import image_float_to_uint8, image_uint8_to_float

##### LETTERBOXING #####
# letterbox_image
def letterbox_image(image, target_dim, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts the given image into a letterboxed image
    - Returned image does not share the same memory
    - If image_info given, sets letterbox info needed to map detections back to the original image
    ----------
    """

    # Creating blank input which we fill in
    letterbox = np.full((target_dim, target_dim, IMG_CHANNEL_COUNT), LETTERBOX_COLOR, dtype=np.float32)

    # Getting letterbox embedding information
    img_w = image.shape[CV2_W_DIM]
    img_h = image.shape[CV2_H_DIM]
    embed_h, embed_w, start_y, start_x = get_letterbox_image_embedding(img_h, img_w, target_dim)

    end_y = start_y + embed_h
    end_x = start_x + embed_w

    # Resizing works better (higher mAP) when using uint8 for some reason
    if(image.dtype == np.float32):
        image = image_float_to_uint8(image)

    # Resizing image to the embedding dimensions
    embed_dim = (embed_w, embed_h)
    embedding_img = cv2.resize(image, embed_dim, interpolation=CV2_INTERPOLATION)

    embedding_img = image_uint8_to_float(embedding_img)

    # Embedding the normalized resized image into the input tensor (Set equal if not letterboxing)
    letterbox[start_y:end_y, start_x:end_x, :] = embedding_img

    # If image_info given, sets information needed to map detections back to the original image
    if(image_info is not None):
        image_info.set_letterbox(start_y, start_x, embed_h, embed_w)

    return letterbox

# get_letterbox_image_embedding
def get_letterbox_image_embedding(img_h, img_w, target_letterbox_dim):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes embedding information for a letterbox input format
    - Information is the size of the embedded image and where the embedded image is in the letterbox
    ----------
    """

    ratio = img_w / img_h

    if(img_w >= img_h):
        embed_w = target_letterbox_dim
        embed_h = round(embed_w / ratio)
    else:
        embed_h = target_letterbox_dim
        embed_w = round(embed_h * ratio)

    start_x = (target_letterbox_dim - embed_w) // 2
    start_y = (target_letterbox_dim - embed_h) // 2

    return embed_h, embed_w, start_y, start_x
