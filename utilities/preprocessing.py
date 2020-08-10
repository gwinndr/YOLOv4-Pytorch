import torch
import cv2

from utilities.constants import *
from utilities.devices import get_device, cpu_device

from utilities.bboxes import is_valid_box
from utilities.images import image_to_tensor, image_uint8_to_float
from utilities.image_info import ImageInfo
from utilities.augmentations import letterbox_image, image_resize, augment_image, create_mosaic

# preprocess_image_eval
def preprocess_image_eval(image, target_dim, letterbox=False, show_img=False, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet input format with dimensions target_dim x target_dim
    - show_img will show the augmented input image
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
    - Returns preprocessed input tensor and image info object for mapping detections back
    ----------
    """

    # Tracking image information to map detections back
    image_info = ImageInfo(image)

    if(force_cpu):
        device = cpu_device()
    else:
        device = get_device()

    # Letterbox to preserve aspect ratio vs. not caring and resizing
    if(letterbox):
        input_image = letterbox_image(image, target_dim, image_info=image_info)
    else:
        input_image = image_resize(image, (target_dim, target_dim), image_info=image_info)

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    # Show image (if applicable)
    if(show_img):
        cv2.imshow("Augmented Input Image", image_info.aug_image)
        cv2.waitKey(0)

    return input_tensor, image_info

# preprocess_image_train
def preprocess_image_train(image, annotations, netblock, target_dim, augment=False, letterbox=False, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet input format with dimensions target_dim x target_dim
    - Annotations are assumed to be normalized (0-1) relative to the input image
    - Will map annotations to the transformed input and filter out invalid annotations
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
    ----------
    """

    if(force_cpu):
        device = cpu_device()
    else:
        device = get_device()

    if((augment) and (letterbox)):
        print("preprocess_image_train: WARNING: Cannot letterbox and augment at the same time. Not letterboxing...")
        letterbox = False

    # Copy of annotations since augmentations change them in_place
    annotations = annotations.clone()

    if(augment):
        input_image = augment_image(image, netblock, target_dim, annotations=annotations)
    elif(letterbox):
        input_image = letterbox_image(image, target_dim, annotations=annotations)
    else:
        input_image = image_resize(image, (target_dim, target_dim))

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    # Filtering out bad annotations
    boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
    is_valid = is_valid_box(boxes, target_dim, target_dim, boxes_normalized=True)
    annotations = annotations[is_valid]

    return input_tensor, annotations

# preprocess_image_mosaic
def preprocess_images_mosaic(images, annotations, netblock, target_dim, force_cpu=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a list of 4 images and a list of 4 annotations tensors to a mosaic image
    - Annotations are assumed to be normalized (0-1) relative to the input image
    - Will map annotations to the mosaic and filter out invalid annotations
    - force_cpu will force the return type to be on the cpu, otherwise uses the default device
    ----------
    """

    if(force_cpu):
        device = cpu_device()
    else:
        device = get_device()

    if(len(images) != 4):
        print("preprocess_image_mosaic: ERROR: Images must be a list of 4 images")
        return None, None
    if(len(annotations) != 4):
        print("preprocess_image_mosaic: ERROR: Annotation must be a list of 4 normalized annotation tensors")
        return None, None

    # Returned annotations is a tensor
    input_image, annotations = create_mosaic(images, netblock, target_dim, images_annotations=annotations)

    # Converting to tensor
    input_tensor = image_to_tensor(input_image, device=device)

    # Filtering out bad annotations
    boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
    is_valid = is_valid_box(boxes, target_dim, target_dim, boxes_normalized=True)
    annotations = annotations[is_valid]

    return input_tensor, annotations
