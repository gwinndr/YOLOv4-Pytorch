import torch
import cv2
import numpy as np

from utilities.constants import *
from utilities.images import image_float_to_uint8, image_uint8_to_float

##### IMAGE RESIZING #####
# image_resize
def image_resize(image, target_dim, annotations=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Resizes a cv2 image to the desired dimensions
    - target_dim should be of the form (width, height)
    - If annotations are given, maps annotations to the new image (in_place)
    - Interpolation given by CV2_INTERPOLATION in constants.py
    ----------
    """

    new_img = cv2.resize(image, target_dim, interpolation=CV2_INTERPOLATION)

    # Mapping annotations to the new image
    if(annotations is not None):
        img_h = image.shape[CV2_H_DIM]
        img_w = image.shape[CV2_W_DIM]
        new_h = new_img.shape[CV2_H_DIM]
        new_w = new_img.shape[CV2_W_DIM]

        # Normalizing
        annotations[..., ANN_BBOX_X1] /= img_w
        annotations[..., ANN_BBOX_Y1] /= img_h
        annotations[..., ANN_BBOX_X2] /= img_w
        annotations[..., ANN_BBOX_Y2] /= img_h

        # Mapping to new image dimensions
        annotations[..., ANN_BBOX_X1] *= new_w
        annotations[..., ANN_BBOX_Y1] *= new_h
        annotations[..., ANN_BBOX_X2] *= new_w
        annotations[..., ANN_BBOX_Y2] *= new_h


    return new_img

##### LETTERBOXING #####
# letterbox_image
def letterbox_image(image, target_dim, annotations=None, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts the given image into a letterboxed image
    - Returned image does not share the same memory
    - If annotations given, maps the annotations to the new image letterbox (in_place)
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
    embedding_img = image_resize(image, embed_dim, annotations=annotations)

    embedding_img = image_uint8_to_float(embedding_img)

    # Embedding the normalized resized image into the input tensor (Set equal if not letterboxing)
    letterbox[start_y:end_y, start_x:end_x, :] = embedding_img

    # Adding letterbox offsets to annotations
    if(annotations is not None):
        annotations[..., ANN_BBOX_X1] += start_x
        annotations[..., ANN_BBOX_Y1] += start_y
        annotations[..., ANN_BBOX_X2] += start_x
        annotations[..., ANN_BBOX_Y2] += start_y

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
