import torch

from utilities.constants import *
from utilities.devices import get_device, cpu_device

from utilities.images import image_to_tensor, image_uint8_to_float
from utilities.image_info import ImageInfo
from utilities.augmentations import letterbox_image, image_resize

# preprocess_image_eval
def preprocess_image_eval(image, target_dim, letterbox, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet input format with dimensions target_dim x target_dim
    - Letterboxing recommended for best results
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
    - Returns preprocessed input tensor and image info object for mapping detections back
    ----------
    """

    # Tracking image information to map detections back
    image_info = ImageInfo(image, target_dim)

    if(force_cpu):
        device = cpu_device()
    else:
        device = get_device()

    # Letterbox to preserve aspect ratio vs. not caring and resizing
    if(letterbox):
        input_image = letterbox_image(image, target_dim, image_info=image_info)
    else:
        input_image = image_resize(input_image, (target_dim, target_dim))

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    return input_tensor, image_info

# preprocess_image_train
def preprocess_image_train(image, annotations, target_dim, letterbox, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet input format with dimensions target_dim x target_dim
    - Will map annotations to the new image
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
    ----------
    """

    # Copy of annotations since augmentations change them in_place
    annotations = annotations.clone()

    if(force_cpu):
        device = cpu_device()
    else:
        device = get_device()

    # Letterbox to preserve aspect ratio vs. not caring and resizing
    if(letterbox):
        input_image = letterbox_image(image, target_dim, annotations=annotations)
    else:
        input_image = image_resize(image, (target_dim, target_dim), annotations=annotations)

    # Normalizing annotations
    annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] /= target_dim

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    return input_tensor, annotations
