import torch

from utilities.constants import *
from utilities.devices import get_device, cpu_device

from utilities.images import image_to_tensor, image_uint8_to_float
from utilities.image_info import ImageInfo
from utilities.augmentations import letterbox_image

# preprocess_image_eval
def preprocess_image_eval(image, target_dim=INPUT_DIM_DEFAULT, letterbox=LETTERBOX_DEFAULT, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet eval input format with dimensions target_dim x target_dim
    - Output will convert cv2 channels from BGR to RGB
    - Letterboxing recommended for best results
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
        - DataLoader with num_workers > 1 needs the output to be on the cpu
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
        input_image = cv2.resize((target_dim, target_dim), interpolation=CV2_INTERPOLATION)

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    return input_tensor, image_info
