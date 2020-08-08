import torch
import cv2
import numpy as np
import random

from utilities.constants import *
from utilities.bboxes import correct_boxes, crop_boxes
from utilities.rando import rand_scale
from utilities.images import image_float_to_uint8, image_uint8_to_float
from utilities.image_info import ImageInfo

##### IMAGE RESIZING #####
# image_resize
def image_resize(image, target_dim, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Resizes a cv2 image to the desired dimensions
    - target_dim should be of the form (width, height)
    - Annotations should be already normalized so image resizing does not have an effect
    - Interpolation given by CV2_INTERPOLATION in constants.py
    ----------
    """

    new_img = cv2.resize(image, target_dim, interpolation=CV2_INTERPOLATION)

    # Setting dimension information for new image
    if(image_info is not None):
        image_info.set_augmentation(new_img)
        image_info.set_embedding_dimensions(new_w, new_h)


    return new_img

# possible_image_sizings
def possible_image_sizings(init_dim, rand_coef, resize_step):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes a list of possible image resizings based on the initial network dimensions and the rand_coef
    - List is in order from smallest to largest
    - The rand_coef is a float value greater than 1.0 that defines how far the image sizing is
      allowed to stray from the initial sizing
    - The resize_step is equivalent to the network stride (32 for yolov4 and yolov3 for example)
    ----------
    """

    max_scale = rand_coef
    min_scale = 1.0/max_scale

    max_dim = round(max_scale * init_dim / resize_step + 1) * resize_step;
    min_dim = round(min_scale * init_dim / resize_step + 1) * resize_step;

    # np.arange stop is non-inclusive
    max_dim += resize_step

    dim_list = np.arange(min_dim, max_dim, resize_step, dtype=np.int32).tolist()

    return dim_list

##### MOSAIC #####
# create_mosaic
def create_mosaic(images, jitter, resize_coef, target_dim, annotations=None):

    if(len(images) != 4):
        print("create_mosaic: Error: Mosaic must be given a list of 4 images")
        return None

    # Create placement image
    mosaic_img = np.zeros((target_dim, target_dim, IMG_CHANNEL_COUNT), dtype=np.float32)

    cut_start = round(target_dim * MOSAIC_MIN_OFFSET)
    cut_end = round(target_dim * (1.0 - MOSAIC_MIN_OFFSET))

    # The center point that all 4 images meet at
    cut_x = random.randint(cut_start, cut_end)
    cut_y = random.randint(cut_start, cut_end)

    # Go through each image and place inside the mosaic
    for i, img in enumerate(images):
        if(img.dtype == np.uint8):
            img = image_uint8_to_float(img)

        image_info = ImageInfo(img)

        ow = img.shape[CV2_W_DIM]
        oh = img.shape[CV2_H_DIM]

        # Jitter information
        pleft, pright, ptop, pbot = get_jitter_embedding(ow, oh, jitter, resize_coef)

        # Mosaic force corner
        if(bool(random.randint(0,1))):
            # Top left
            if (i == 0):
                pright += pleft
                pleft = 0
                pbot += ptop
                ptop = 0
            # Top right
            elif (i == 1):
                pleft += pright
                pright = 0
                pbot += ptop
                ptop = 0
            # Bottom left
            elif (i == 2):
                pright += pleft
                pleft = 0
                ptop += pbot
                pbot = 0
            # Bottom right
            else: # (i == 3)
                pleft += pright
                pright = 0
                ptop += pbot
                pbot = 0

        # Jittering image beforehand
        jitter_embedding = (pleft, pright, ptop, pbot)
        jittered_img = jitter_image_precalc(img, jitter_embedding, target_dim, image_info=image_info)

        place_image_mosaic(mosaic_img, jittered_img, image_info, cut_x, cut_y, i, annotations=annotations)

    return mosaic_img

def place_image_mosaic(placement_image, image, image_info, cut_x, cut_y, i_num, annotations=None):

    ow = image.shape[CV2_W_DIM]
    oh = image.shape[CV2_H_DIM]
    pw = placement_image.shape[CV2_W_DIM]
    ph = placement_image.shape[CV2_H_DIM]

    # top left
    if(i_num == 0):
        placem_x1 = 0
        placem_x2 = cut_x
        placem_y1 = 0
        placem_y2 = cut_y
    # top right
    elif(i_num == 1):
        placem_x1 = cut_x
        placem_x2 = pw
        placem_y1 = 0
        placem_y2 = cut_y
    # Bottom left
    elif(i_num == 2):
        placem_x1 = 0
        placem_x2 = cut_x
        placem_y1 = cut_y
        placem_y2 = ph
    # Bottom right
    else: #(i_num == 3):
        placem_x1 = cut_x
        placem_x2 = pw
        placem_y1 = cut_y
        placem_y2 = ph

    needed_w = placem_x2 - placem_x1
    needed_h = placem_y2 - placem_y1

    pleft = image_info.aug_pleft
    ptop = image_info.aug_ptop
    avail_w = image_info.aug_embed_w
    avail_h = image_info.aug_embed_h
    full_w = image_info.aug_w
    full_h = image_info.aug_h

    # Fix for when we need to go outside the embedded image
    pleft, avail_w = correct_mosaic_placement(pleft, avail_w, needed_w, full_w)
    ptop, avail_h = correct_mosaic_placement(ptop, avail_h, needed_h, full_h)

    # Basically in reverse to placement
    # top left = start at bottom right
    if(i_num == 0):
        pright = pleft + avail_w
        pbot = ptop + avail_h
        pleft = pright - needed_w
        ptop = pbot - needed_h
    # top right = start at bottom left
    elif(i_num == 1):
        pright = pleft + needed_w
        pbot = ptop + avail_h
        pleft = pleft
        ptop = pbot - needed_h
    # Bottom left = start at top right
    elif(i_num == 2):
        pright = pleft + avail_w
        pbot = ptop + needed_h
        pleft = pright - needed_w
        ptop = ptop
    # Bottom right = start at top left
    else: #(i_num == 3):
        pright = pleft + needed_w
        pbot = ptop + needed_h
        pleft = pleft
        ptop = ptop

    placement_image[placem_y1:placem_y2, placem_x1:placem_x2] = image[ptop:pbot, pleft:pright]

    return

# correct_mosaic_placement
def correct_mosaic_placement(p, avail, needed, full):
    corrected_p = p
    corrected_avail = avail

    # Correction needed
    if(avail < needed):
        # Will have some of the padding bleed from the left/top only if needed
        end = p + needed
        if(end > full):
            amount_over = end - full
            corrected_p -= amount_over

        corrected_avail = needed

    return corrected_p, corrected_avail

##### IMAGE JITTER #####
def jitter_image(image, jitter, resize_coef, target_dim, annotations=None):
    ow = image.shape[CV2_W_DIM]
    oh = image.shape[CV2_H_DIM]

    precalc = get_jitter_embedding(ow, oh, jitter, resize_coef)
    jitter_img = jitter_image_precalc(image, precalc, target_dim, annotations=annotations)

    return jitter_img

# jitter_image_precalc
def jitter_image_precalc(image, precalc, target_dim, annotations=None):
    if(image.dtype == np.uint8):
        image = image_uint8_to_float(image)

    ow = image.shape[CV2_W_DIM]
    oh = image.shape[CV2_H_DIM]

    pleft, pright, ptop, pbot = precalc

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    # Image cropping and placement (intersection of image and p rectangle)
    crop_x1 = max(0, pleft)
    crop_y1 = max(0, ptop)
    crop_x2 = min(ow, pleft + swidth)
    crop_y2 = min(oh, ptop + sheight)

    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    # No need to do anything further if there's no image cropping
    if((crop_x1 == 0) and (crop_y1 == 0) and (crop_w == ow) and (crop_h == oh)):
        new_img = image

        # Image info placement
        dst_x1_norm = 0.0
        dst_y1_norm = 0.0
        dst_w_norm = 1.0
        dst_h_norm = 1.0
    else:
        # Just how darknet does it, it sort of reflects the dimension placement
        dst_x1 = max(0, -pleft)
        dst_y1 = max(0, -ptop)
        dst_x2 = dst_x1 + crop_w
        dst_y2 = dst_y1 + crop_h

        # Setting up the new image
        img_mean = np.array(cv2.mean(image))
        new_img = np.zeros((sheight, swidth, IMG_CHANNEL_COUNT), dtype=np.float32)
        new_img[..., :] = img_mean[:IMG_CHANNEL_COUNT]

        # Cropping out the original and placing into the new image
        new_img[dst_y1:dst_y2, dst_x1:dst_x2] = image[crop_y1:crop_y2, crop_x1:crop_x2]

        # Image info placement
        dst_x1_norm = dst_x1 / swidth
        dst_y1_norm = dst_y1 / sheight
        dst_w_norm = crop_w / swidth
        dst_h_norm = crop_h / sheight

    # Resizing to our target dim
    new_dim = (target_dim, target_dim)
    new_img = image_resize(new_img, new_dim)

    # Setting annotations
    if(annotations is not None):
        start_x = round(dst_x1_norm * target_dim)
        start_y = round(dst_y1_norm * target_dim)
        embed_w = round(dst_w_norm * target_dim)
        embed_h = round(dst_h_norm * target_dim)

        boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]

        # First crop out the boxes from the image
        boxes = crop_boxes(boxes, ow, oh, crop_x1, crop_y1, crop_w, crop_h, boxes_normalized=True)

        # Then map to image within dst
        boxes = correct_boxes(boxes, crop_w, crop_h, target_dim, target_dim,
                    n_offs_x=start_x, n_offs_y=start_y, n_embed_w=embed_w, n_embed_h=embed_h,
                    boxes_normalized=True)

        annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] = boxes

    return new_img

# get_jitter_embedding
def get_jitter_embedding(width, height, jitter, resize):
    # Jitter allowance
    dw = width * jitter
    dh = height * jitter

    # Resizing bounds
    resize_down = 1.0/resize if resize > 1.0 else resize
    resize_up = 1.0/resize if resize < 1.0 else resize

    min_rdw = width * (1 - (1 / resize_down)) / 2
    min_rdh = height * (1 - (1 / resize_down)) / 2
    max_rdw = width * (1 - (1 / resize_up)) / 2
    max_rdh = height * (1 - (1 / resize_up)) / 2

    # The jitter placement
    pleft = round(random.uniform(-dw, dw))
    pright = round(random.uniform(-dw, dw))
    ptop = round(random.uniform(-dh, dh))
    pbot = round(random.uniform(-dh, dh))

    # Downsize only
    if(resize < 1.0):
        max_rdw = 0
        max_rdh = 0

    pleft += round(random.uniform(min_rdw, max_rdw))
    pright += round(random.uniform(min_rdw, max_rdw))
    ptop += round(random.uniform(min_rdh, max_rdh))
    pbot += round(random.uniform(min_rdh, max_rdh))

    return pleft, pright, ptop, pbot


##### HSV SHIFTING #####
# hsv_shift_image
def hsv_shift_image(image, hue, saturation, exposure, image_info=None, precalc=None):

    # Very important this conversion happens, otherwise CV2_HSV_H_MAX is wrong
    if(image.dtype == np.uint8):
        image = image_uint8_to_float(image)

    if(precalc is None):
        dhue, dsat, dexp = get_hsv_shifting(hue, saturation, exposure)
    else:
        dhue, dsat, dexp = precalc

    hue_term = dhue * CV2_HSV_H_MAX

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Ze shift
    h += hue_term
    s *= dsat
    v *= dexp

    # This fix prevents weird results with artifacting
    if(dhue < 0):
        h[h < 0.0] += CV2_HSV_H_MAX
    else:
        h[h > CV2_HSV_H_MAX] -= CV2_HSV_H_MAX

    hsv = cv2.merge([h, s, v])
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if(image_info is not None):
        image_info.set_augmentation(new_img)

    return new_img

# get_hsv_shifting
def get_hsv_shifting(hue, saturation, exposure):
    dhue = random.uniform(-hue, hue);
    dsat = rand_scale(saturation);
    dexp = rand_scale(exposure);

    return dhue, dsat, dexp


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
    embedding_img = image_resize(image, embed_dim)

    embedding_img = image_uint8_to_float(embedding_img)

    # Embedding the normalized resized image into the input tensor (Set equal if not letterboxing)
    letterbox[start_y:end_y, start_x:end_x, :] = embedding_img

    # Adding letterbox offsets to annotations
    if(annotations is not None):
        boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]
        boxes = correct_boxes(boxes, img_w, img_h, target_dim, target_dim,
                    n_offs_x=start_x, n_offs_y=start_y, n_embed_w=embed_w, n_embed_h=embed_h,
                    boxes_normalized=True)
        annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] = boxes

    # Sets the image topleft offset and the embedding dimensions
    if(image_info is not None):
        image_info.set_augmentation(letterbox)
        image_info.set_offset(start_x, start_y)
        image_info.set_embedding_dimensions(embed_w, embed_h)

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
