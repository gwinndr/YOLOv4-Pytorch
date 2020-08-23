import torch
import cv2
import numpy as np
import random

from utilities.constants import *
from utilities.bboxes import correct_boxes, crop_boxes
from utilities.rando import rand_scale
from utilities.images import image_float_to_uint8, image_uint8_to_float
from utilities.image_info import ImageInfo

##### AUGMENTATION FOR SINGLE IMAGE #####
def augment_image(image, netblock, target_dim, annotations=None, image_info=None, jitter=True, hsv=True, flip=True):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Augments a single image
    - Given annotations should be normalized
    - Takes a network block as argument and maps given annotations to the new image
    - Width and height given by network block are overrided by target_dim
    - Can toggle jitter, hsv, and flip args to False to force those augmentations not to run
    - Returns augmented image
    ----------
    """

    if(image.dtype == np.uint8):
        image = image_uint8_to_float(image)

    aug_img = image.copy()

    if(image_info is None):
        image_info = ImageInfo(image)

    # First jitter
    if(jitter):
        aug_img = jitter_image(aug_img, netblock.jitter, netblock.resize, target_dim,
                    annotations=annotations, image_info=image_info)

    # Then hsv
    if(hsv):
        aug_img = hsv_shift_image(aug_img, netblock.hue, netblock.saturation, netblock.exposure,
                    image_info=image_info)

    # Finally, flip with half probability
    flip = (flip and netblock.flip and bool(random.randint(0,1)))
    if(flip):
        aug_img = flip_image(aug_img, annotations=annotations, image_info=image_info)

    # Resize if needed
    aw = aug_img.shape[CV2_W_DIM]
    ah = aug_img.shape[CV2_H_DIM]
    if((aw != target_dim) or (ah != target_dim)):
        aug_img = image_resize(aug_img, (target_dim, target_dim), image_info=image_info)

    return aug_img


##### MOSAIC #####
# create_mosaic
def create_mosaic(images, netblock, target_dim, images_annotations=None, jitter=True, hsv=True, flip=True):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Creates a mosaic image out of a list of 4 images
    - Given annotations should be a list of 4 normalized annotation tensors
    - Takes a network block as argument and maps given annotations to the new mosaic
    - Can toggle jitter, hsv, and flip args to False to force those augmentations not to run
    - Width and height given by network block are overrided by target_dim
    - Returns mosaic and all annotations as one tensor (recommend you fix with is_valid_box)
    ----------
    """

    if(len(images) != 4):
        print("create_mosaic: Error: Mosaic must be given a list of 4 images")
        return None

    if((images_annotations is not None) and (len(images_annotations) != len(images))):
        print("create_mosaic: Error: For mosaic, annotations must be None or a list of 4")
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

        annotations = images_annotations[i] if images_annotations is not None else None
        image_info = ImageInfo(img)

        ow = img.shape[CV2_W_DIM]
        oh = img.shape[CV2_H_DIM]

        # Need to jitter separately since we need the embedding information
        pleft, pright, ptop, pbot = get_jitter_embedding(ow, oh, netblock.jitter, netblock.resize)

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

        try:
            jittered_img = jitter_image_precalc(img, jitter_embedding, target_dim, annotations=annotations, image_info=image_info)
        except:
            print("Error with jittering image inside create_mosaic:")
            print("jitter_embedding:", jitter_embedding)
            raise

        try:
            # Do rest of the augmentations
            aug_image = augment_image(jittered_img, netblock, target_dim, annotations=annotations, image_info=image_info, jitter=False)
        except:
            print("Mosaic image index:", i)
            raise

        try:
            # Place the image on the mosaic
            place_image_mosaic(mosaic_img, aug_image, image_info, cut_x, cut_y, i, annotations=annotations)
        except:
            print("Error with placing mosaic image:")
            print("placement image shape:", mosaic_img.shape)
            print("aug image shape:", aug_image.shape)
            print("jitter_embedding:", jitter_embedding)
            print("cut_x:", cut_x)
            print("cut_y:", cut_y)
            print("image num:", i)
            raise

    # Combining all annotations together into one tensor
    if(images_annotations is not None):
        all_annotations = torch.cat(images_annotations, dim=0)
    else:
        all_annotations = None

    # Flip the whole image with half probability
    flip = (flip and netblock.flip and bool(random.randint(0,1)))
    if(flip):
        mosaic_img = flip_image(mosaic_img, annotations=all_annotations)

    return mosaic_img, all_annotations

# place_image_mosaic
def place_image_mosaic(placement_image, image, image_info, cut_x, cut_y, i_num, annotations=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Helper for create_mosaic
    - Given annotations should be normalized
    - Places an image on the placement according to mosaic rules
    - cut_x and cut_y is the dividing point in the image
    - i_num is the current image number (0-3 inclusive)
    ----------
    """

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
    pleft_fix, avail_w_fix = correct_mosaic_placement(pleft, avail_w, needed_w, full_w)
    ptop_fix, avail_h_fix = correct_mosaic_placement(ptop, avail_h, needed_h, full_h)

    # For properly mapping annotations
    offset_x_fix = abs(pleft - pleft_fix)
    offset_y_fix = abs(ptop - ptop_fix)
    embed_w_fix = abs(avail_w - avail_w_fix)
    embed_h_fix = abs(avail_h - avail_h_fix)

    pleft = pleft_fix
    avail_w = avail_w_fix
    ptop = ptop_fix
    avail_h = avail_h_fix

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

    # Placing the mosaic
    placement_image[placem_y1:placem_y2, placem_x1:placem_x2] = image[ptop:pbot, pleft:pright]

    # Mapping annotations
    if(annotations is not None):
        boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]

        # Account for offset in case of fix
        pleft_adj = pleft + offset_x_fix
        ptop_adj = ptop + offset_y_fix
        actual_w = needed_w - embed_w_fix
        actual_h = needed_h - embed_h_fix
        placem_x1_adj = placem_x1 + offset_x_fix
        placem_y1_adj = placem_y1 + offset_y_fix

        # Crop by adjusted dims
        boxes = crop_boxes(boxes, ow, oh, pleft_adj, ptop_adj, actual_w, actual_h, boxes_normalized=True)

        # Map to adjusted image embedded in the mosaic
        boxes = correct_boxes(boxes, actual_w, actual_h, pw, ph,
                    n_offs_x=placem_x1_adj, n_offs_y=placem_y1_adj, n_embed_w=actual_w, n_embed_h=actual_h,
                    boxes_normalized=True)

        annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] = boxes

    return

# correct_mosaic_placement
def correct_mosaic_placement(p, avail, needed, full):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Helper for place_image_mosaic
    - Corrects given point, p, and available dim, avail, so that avail >= needed
    - Function may do no corrections if avail >= needed already
    - Function assumes that full >= needed
    - Returns corrected p and avail
    - p: point of embedded image
    - avail: length of embedded image
    - needed: required length for mosaic
    - full: the full length of the augmentation
    ----------
    """

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
    nw = new_img.shape[CV2_W_DIM]
    nh = new_img.shape[CV2_H_DIM]

    # Setting dimension information for new image
    if(image_info is not None):
        image_info.set_augmentation(new_img)
        image_info.set_embedding_dimensions(nw, nh)


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


##### IMAGE JITTER #####
def jitter_image(image, jitter, resize_coef, target_dim, annotations=None, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes image jitter
    - Given annotations should be normalized
    - Jitter value must be a value less that .5 otherwise it could crash due to negative image dimensions
    - resize_coef should be a value greater than 0 and less than 2
    - This function computes get_jitter_embedding information and passes it to jitter_image_precalc
    - See jitter_image_precalc for more info.
    ----------
    """

    ow = image.shape[CV2_W_DIM]
    oh = image.shape[CV2_H_DIM]

    precalc = get_jitter_embedding(ow, oh, jitter, resize_coef)

    try:
        jitter_img = jitter_image_precalc(image, precalc, target_dim, annotations=annotations, image_info=image_info)
    except:
        print("Error with jittering image:")
        print("ow:", ow)
        print("oh:", oh)
        print("jitter precalc:", precalc)
        raise

    return jitter_img

# jitter_image_precalc
def jitter_image_precalc(image, precalc, target_dim, annotations=None, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes image jitter based on precalculated values from get_jitter_embedding
    - Given annotations should be normalized
    - Jitter will create a new image of varying dimensions based on precalc values
    - Image may be a smaller image embedded in a bigger image
    - Result is resized to target_dim
    - Returns augmented image
    ----------
    """

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
        # Negation to guarantee placement lines up on the destination image
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

    # Needed by image info and annotations
    # Will take the floor to ensure we never go outside the image bounds
    start_x = int(dst_x1_norm * target_dim)
    start_y = int(dst_y1_norm * target_dim)
    embed_w = int(dst_w_norm * target_dim)
    embed_h = int(dst_h_norm * target_dim)

    # Setting annotations
    if(annotations is not None):
        boxes = annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1]

        # First crop out the boxes from the image
        boxes = crop_boxes(boxes, ow, oh, crop_x1, crop_y1, crop_w, crop_h, boxes_normalized=True)

        # Then map to image within dst
        boxes = correct_boxes(boxes, crop_w, crop_h, target_dim, target_dim,
                    n_offs_x=start_x, n_offs_y=start_y, n_embed_w=embed_w, n_embed_h=embed_h,
                    boxes_normalized=True)

        annotations[..., ANN_BBOX_X1:ANN_BBOX_Y2+1] = boxes

    # Setting image info
    if(image_info is not None):
        image_info.set_augmentation(new_img)
        image_info.set_offset(start_x, start_y)
        image_info.set_embedding_dimensions(embed_w, embed_h)

    return new_img

# get_jitter_embedding
def get_jitter_embedding(width, height, jitter, resize):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes randomized jitter embedding based on source width and height and jitter-resize coefficients
    - jitter should be a value greater than 0 and less than .5
    - resize should be a value greater than 0 and less than 2
    ----------
    """

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
def hsv_shift_image(image, hue, saturation, exposure, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes an hsv shifted image
    - Hue should be a value less than 1
    - Saturation and exposure should be values greater than 1 and less than 2
    - See hsv_shift_image_precalc for more info.
    ----------
    """

    precalc = get_hsv_shifting(hue, saturation, exposure)

    try:
        new_img = hsv_shift_image_precalc(image, precalc, image_info=image_info)
    except:
        print("Error with HSV shifting:")
        print("HSV precalc:", precalc)
        raise

    return new_img

# hsv_shift_image_precalc
def hsv_shift_image_precalc(image, precalc, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes an hsv shifted image based on precalculated values
    - dhue * CV2_HSV_H_MAX is added as a shift to hue
    - dsat and dexp are multiplied as a scaling to sat and exp
    - hsv shifting does not affect annotations
    ----------
    """

    # Very important this conversion happens, otherwise CV2_HSV_H_MAX is wrong
    if(image.dtype == np.uint8):
        image = image_uint8_to_float(image)

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

    # Fix for if values go over 1 or under 0
    new_img[new_img > 1.0] = 1.0
    new_img[new_img < 0.0] = 0.0

    if(image_info is not None):
        image_info.set_augmentation(new_img)

    return new_img

# get_hsv_shifting
def get_hsv_shifting(hue, saturation, exposure):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Computes randomized hsv coefficients
    - hue should be a value less than 1
    - Saturation and exposure should be values greater than 1 and less than 2
    ----------
    """

    dhue = random.uniform(-hue, hue);
    dsat = rand_scale(saturation);
    dexp = rand_scale(exposure);

    return dhue, dsat, dexp

##### IMAGE FLIP #####
def flip_image(image, annotations=None, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Horizontally flips an image and maps annotations to the new image
    - Also fixes image_info offset information to reflect the flip
    - Given annotations should be normalized
    ----------
    """

    new_img = cv2.flip(image, CV2_FLIP_HORIZONTAL)

    # Annotations
    if(annotations is not None):
        x1 = annotations[..., ANN_BBOX_X1]
        x2 = annotations[..., ANN_BBOX_X2]

        # Make the center of the image at position 0
        x1 -= 0.5
        x2 -= 0.5

        # Flip their position on this new (-0.5 -> 0.5) scale
        x1 = -x1
        x2 = -x2

        # Map back to (0 -> 1)
        x1 += 0.5
        x2 += 0.5

        # Swap because x1 is now greater than x2
        temp = x1
        x1 = x2
        x2 = temp

        annotations[..., ANN_BBOX_X1] = x1
        annotations[..., ANN_BBOX_X2] = x2

    # image_info
    if(image_info is not None):
        half_ow = image.shape[CV2_W_DIM] / 2.0
        pleft = image_info.aug_pleft
        embed_w = image_info.aug_embed_w

        # Fixing pleft in a similar way to annotations
        pleft -= half_ow
        pleft = -pleft
        pleft += half_ow
        pleft -= embed_w

        image_info.aug_pleft = int(pleft)
        image_info.set_augmentation(new_img)


    return new_img


##### LETTERBOXING #####
# letterbox_image
def letterbox_image(image, target_dim, annotations=None, image_info=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Converts the given image into a letterboxed image
    - Given annotations should be normalized
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
