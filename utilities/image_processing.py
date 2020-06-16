import torch
import cv2
import numpy as np
import random

from utilities.constants import *
from utilities.devices import get_device

# preprocess_image_eval
def preprocess_image_eval(image, target_dim=INPUT_DIM_DEFAULT, letterbox=LETTERBOX_DEFAULT):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts a cv2 image into Darknet eval input format with dimensions target_dim x target_dim
    - Output will convert cv2 channels from BGR to RGB
    - Letterboxing recommended for best results
    ----------
    """

    # Creating blank input which we fill in
    input_tensor = torch.full((target_dim, target_dim, IMG_CHANNEL_COUNT), LETTERBOX_COLOR, dtype=TORCH_FLOAT, device=get_device())

    # Getting letterbox embedding information
    if(letterbox):
        img_w = image.shape[CV2_W_DIM]
        img_h = image.shape[CV2_H_DIM]
        embed_h, embed_w, start_y, start_x = get_letterbox_image_embedding(img_h, img_w, target_dim)

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
    new_img = torch.tensor(new_img, dtype=TORCH_FLOAT, device=get_device()) / 255.0

    # Embedding the normalized resized image into the input tensor (Set equal if not letterboxing)
    input_tensor[start_y:end_y, start_x:end_x, :] = new_img
    input_tensor = input_tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM).contiguous()

    return input_tensor

# map_detections_to_original_image
def map_dets_to_original_image(detections, img_h, img_w, input_dim=INPUT_DIM_DEFAULT, letterboxed=LETTERBOX_DEFAULT):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Maps detections from the preprocessed input image back to the original image
    ----------
    """

    # Getting letterbox information for embedded image
    if(letterboxed):
        embed_h, embed_w, start_y, start_x = get_letterbox_image_embedding(img_h, img_w, input_dim)

    # Setting information such that there is no letterbox (tensor contains the full image)
    else:
        embed_h = input_dim
        embed_w = input_dim
        start_y = 0
        start_x = 0

    # Move embedded image back to top left
    detections[..., DETECTION_X1] -= start_x
    detections[..., DETECTION_Y1] -= start_y
    detections[..., DETECTION_X2] -= start_x
    detections[..., DETECTION_Y2] -= start_y

    # Normalize by the image within the letterbox
    detections[..., DETECTION_X1] /= embed_w
    detections[..., DETECTION_Y1] /= embed_h
    detections[..., DETECTION_X2] /= embed_w
    detections[..., DETECTION_Y2] /= embed_h

    # Map back to original image
    detections[..., DETECTION_X1] *= img_w
    detections[..., DETECTION_Y1] *= img_h
    detections[..., DETECTION_X2] *= img_w
    detections[..., DETECTION_Y2] *= img_h

    return detections

# write_detections_to_image
def write_dets_to_image(detections, image, class_names, verbose_output=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Creates a new image with the detections shown as bounding boxes
    - verbose_output will additionally print the detections to console
    - See BBOX DRAWING section in utilities.constants to control how bounding boxes are drawn
    ----------
    """

    image = image.copy()

    bboxes = detections[..., DETECTION_X1:DETECTION_Y2+1].cpu().numpy()
    classes = detections[..., DETECTION_CLASS_IDX].cpu().type(torch.int32).numpy()
    class_confs = detections[..., DETECTION_CLASS_PROB].cpu().numpy()

    for i in range(len(detections)):
        x1 = int(round(bboxes[i, BBOX_X1]))
        y1 = int(round(bboxes[i, BBOX_Y1]))
        x2 = int(round(bboxes[i, BBOX_X2]))
        y2 = int(round(bboxes[i, BBOX_Y2]))

        # Bbox rectangle top-left and bottom-right
        p1 = (x1, y1)
        p2 = (x2, y2)

        class_name = class_names[classes[i]]
        class_conf = class_confs[i]

        # Extended output is similar to darknet's
        if(verbose_output):
            print("Class:", class_name)
            print("Conf: %.2f" % class_conf)
            print("Left_x:", x1)
            print("Left_y:", y1)
            print("Width:", x2-x1)
            print("Height:", y2-y1)
            print("")

        label = "%s %.2f" % (class_name, class_conf)

        # Drawing the full bounding box
        color = random.choice(BBOX_COLORS)
        cv2.rectangle(image, p1, p2, color, BBOX_RECT_THICKNESS)

        # Getting the label text size
        t_dims = cv2.getTextSize(label, BBOX_FONT, BBOX_FONT_SCALE, BBOX_FONT_THICKNESS)[0]

        # Drawing a label rectangle above the bbox to make the text pop out better
        label_rect_x1 = x1
        label_rect_y1 = y1 - t_dims[CV2_TEXT_SIZE_H] - (BBOX_TEXT_TOP_PAD + BBOX_TEXT_BOT_PAD)
        label_rect_x2 = x1 + t_dims[CV2_TEXT_SIZE_W] + (BBOX_TEXT_LEFT_PAD + BBOX_TEXT_RIGHT_PAD)
        label_rect_y2 = y1

        label_rect_p1 = (label_rect_x1, label_rect_y1)
        label_rect_p2 = (label_rect_x2, label_rect_y2)

        cv2.rectangle(image, label_rect_p1, label_rect_p2, color, CV2_RECT_FILL)

        # Placing the label text within the label rectangle
        label_text_p = (x1 + BBOX_TEXT_LEFT_PAD, y1 - BBOX_TEXT_BOT_PAD)
        cv2.putText(image, label, label_text_p, BBOX_FONT, BBOX_FONT_SCALE, COLOR_BLACK, BBOX_FONT_THICKNESS);

    return image

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

# image_to_tensor
def image_to_tensor(image):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts cv2 image into a pytorch input tensor
    - Helper function, recommended you use one of preprocess_image_eval or preprocess_image_train
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

# tensor_to_image
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
    image = tensor.cpu().numpy()
    image *= 255.0
    image = image.astype(np.uint8)

    # bgr
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image
