import torch
import cv2
import numpy as np

from utilities.constants import *
from utilities.devices import get_device, cpu_device

def preprocess_image(image, target_dim=INPUT_DIM_DEFAULT, letterbox=LETTERBOX_DEFAULT):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet format with dimensions target_dim x target_dim
    - Output will convert channels from BGR to RGB
    ----------
    """

    tensor = None

    if(letterbox):
        tensor = torch.full((target_dim, target_dim, CV2_N_IMG_DIM), LETTERBOX_COLOR, dtype=TORCH_FLOAT, device=get_device())

        img_w = image.shape[CV2_W_DIM]
        img_h = image.shape[CV2_H_DIM]

        ratio = img_w / img_h

        if(img_w >= img_h):
            new_w = target_dim
            new_h = round(new_w / ratio)
        else:
            new_h = target_dim
            new_w = round(new_h * ratio)

        new_dim = (new_w, new_h)
        new_img = cv2.resize(image, new_dim, interpolation=CV2_INTERPOLATION)

        # rgb
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        start_h = (target_dim - new_h) // 2
        start_w = (target_dim - new_w) // 2
        end_h = start_h + new_h
        end_w = start_w + new_w

        new_img = torch.tensor(new_img, dtype=TORCH_FLOAT, device=get_device()) / 255.0

        tensor[start_h:end_h, start_w:end_w, :] = new_img

    else:
        new_img = cv2.resize(image, (target_dim, target_dim), interpolation=CV2_INTERPOLATION)
        # rgb
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        tensor = torch.tensor(new_img, dtype=TORCH_FLOAT, device=get_device()) / 255.0

    tensor = tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM).contiguous()
    return tensor

def image_to_tensor(image):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts cv2 image into a darknet input tensor
    - Image output is first converted to RGB, then normalized
    ----------
    """

    # rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # normalize
    norm_image = image.astype(np.float32) / 255.0

    # then convert to tensor
    tensor = torch.tensor(norm_image, device=get_device(), dtype=TORCH_FLOAT)
    tensor = tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM).contiguous()
    return tensor

def tensor_to_image(tensor):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts darknet input tensor into cv2 image
    - Tensor is first converted to uint8, then converted to BGR
    ----------
    """

    # Moving channel dimension back to normal
    tensor = tensor.permute(INPUT_H_DIM, INPUT_W_DIM, INPUT_C_DIM).contiguous()

    # Convert back to uint8
    image = tensor.to(cpu_device()).numpy()
    image *= 255.0
    image = image.astype(np.uint8)

    # bgr
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
