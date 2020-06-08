import torch
import cv2
import numpy as np

from utilities.constants import *
from utilities.devices import get_device

def letterbox_image(image, target_dim):
    letterbox = np.full((target_dim, target_dim, CV2_N_IMG_DIM), LETTERBOX_COLOR, dtype=np.uint8)

    img_w = image.shape[CV2_W_DIM]
    img_h = image.shape[CV2_H_DIM]

    ratio = img_w / img_h

    if(img_w >= img_h):
        new_w = target_dim
        new_h = round(new_w / ratio)
    else:
        new_h = target_dim
        new_w = round(target_dim * ratio)

    new_dim = (new_w, new_h)
    new_img = cv2.resize(image, new_dim, interpolation=CV2_INTERPOLATION)

    start_h = (target_dim - new_h) // 2
    start_w = (target_dim - new_w) // 2
    end_h = start_h + new_h
    end_w = start_w + new_w

    letterbox[start_h:end_h, start_w:end_w, :] = new_img

    return letterbox

def image_to_tensor(image):
    # rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # normalize
    norm_image = image.astype(np.float32) / 255.0

    # then convert to tensor
    tensor = torch.tensor(norm_image, device=get_device())
    tensor = tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM)
    return tensor
