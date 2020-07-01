import torch
import cv2
import numpy as np

from .constants import *
from .devices import get_device, cpu_device

from .image_info import ImageInfo

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

# draw_detections
def draw_detections(detections, image, class_names, verbose_output=False):
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

    bboxes = detections[..., DETECTION_X1:DETECTION_Y2+1]
    classes = detections[..., DETECTION_CLASS_IDX]
    class_confs = detections[..., DETECTION_CLASS_PROB]

    # Sort by confidence (better bbox color stability on videos)
    _, indices = torch.sort(class_confs, dim=0, descending=True)
    bboxes = bboxes[indices].cpu().numpy()
    classes = classes[indices].cpu().type(torch.int32).numpy()
    class_confs = class_confs[indices].cpu().numpy()

    n_colors = len(BBOX_COLORS)

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

        if(BBOX_INCLUDE_CLASS_CONF):
            label = "%s %.2f" % (class_name, class_conf)
        else:
            label = class_name

        # Drawing the full bounding box
        # color = random.choice(BBOX_COLORS)
        color = BBOX_COLORS[i % n_colors]
        cv2.rectangle(image, p1, p2, color, BBOX_RECT_THICKNESS)

        # Getting the label text size
        t_dims = cv2.getTextSize(label, BBOX_FONT, BBOX_FONT_SCALE, BBOX_FONT_THICKNESS)[0]

        # Drawing a label rectangle above the bbox at the top left to make the text pop out better
        label_rect_x1 = x1
        label_rect_y1 = y1 - t_dims[CV2_TEXT_SIZE_H] - (BBOX_TEXT_TOP_PAD + BBOX_TEXT_BOT_PAD)
        label_rect_x2 = x1 + t_dims[CV2_TEXT_SIZE_W] + (BBOX_TEXT_LEFT_PAD + BBOX_TEXT_RIGHT_PAD)
        label_rect_y2 = y1

        # Label to go into the rectangle
        label_x = x1 + BBOX_TEXT_LEFT_PAD
        label_y = y1 - BBOX_TEXT_BOT_PAD

        # Will put the label below the bbox at the top left if it clips outside the image
        if(label_rect_y1 < 0):
            move_y = (y1 - label_rect_y1)

            label_rect_y1 += move_y
            label_rect_y2 += move_y
            label_y += move_y

        label_rect_p1 = (label_rect_x1, label_rect_y1)
        label_rect_p2 = (label_rect_x2, label_rect_y2)
        label_text_p = (label_x, label_y)

        # Drawing the label rectangle with the text inside
        cv2.rectangle(image, label_rect_p1, label_rect_p2, color, CV2_RECT_FILL)
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
    tensor = torch.tensor(norm_image, device=get_device(), dtype=torch.float32)
    tensor = tensor.permute(CV2_C_DIM, CV2_H_DIM, CV2_W_DIM).contiguous()
    return tensor
