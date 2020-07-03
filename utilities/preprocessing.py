import torch

from utilities.constants import *
from utilities.devices import get_device, cpu_device

from utilities.image_info import ImageInfo

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
        output_device = cpu_device()
    else:
        output_device = get_device()

    # Creating blank input which we fill in
    input_tensor = torch.full((target_dim, target_dim, IMG_CHANNEL_COUNT), LETTERBOX_COLOR, dtype=torch.float32, device=output_device)

    # Getting letterbox embedding information
    if(letterbox):
        img_w = image.shape[CV2_W_DIM]
        img_h = image.shape[CV2_H_DIM]
        embed_h, embed_w, start_y, start_x = get_letterbox_image_embedding(img_h, img_w, target_dim)

        image_info.set_letterbox(start_y, start_x, embed_h, embed_w)

    # Setting embedding information to include no letterbox space
    else:
        embed_h = target_dim
        embed_w = target_dim
        start_y = 0
        start_x = 0

    end_y = start_y + embed_h
    end_x = start_x + embed_w

    # Resizing image to the embedding dimensions
    new_img_dim = (embed_w, embed_h)
    new_img = cv2.resize(image, new_img_dim, interpolation=CV2_INTERPOLATION)

    # Converting image from BGR to RGB
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    # Normalizing the image values and converting to torch tensor
    new_img = torch.tensor(new_img, dtype=torch.float32, device=output_device) / 255.0

    # Embedding the normalized resized image into the input tensor (Set equal if not letterboxing)
    input_tensor[start_y:end_y, start_x:end_x, :] = new_img
    input_tensor = input_tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM).contiguous()

    return input_tensor, image_info

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
