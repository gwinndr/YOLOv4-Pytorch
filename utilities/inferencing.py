from utilities.constants import *

from utilities.devices import get_device
from utilities.image_processing import preprocess_image_eval, map_dets_to_original_image, write_dets_to_image
from utilities.extract_detections import extract_detections

# inference_on_single_image
def inference_on_single_image(model, image, network_dim, letterbox):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Runs inference detection given a yolo model
    - Assumes image is a cv2 image with 3 color channels
    - Returns detections for this image (does not return a list like in inference_on_images)
    - verbose_output prints the detections to console
    ----------
    """

    input = (image,)
    detections = inference_on_images(model, input, network_dim, letterbox)

    # Returned detections is not in list format (since it's only one image)
    detections = detections[0]

    return detections

# inference_on_images
def inference_on_images(model, images, network_dim, letterbox):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Runs inference detection given a yolo model
    - Assumes images is a list of cv2 images with 3 color channels
    - Returns detections for each image as a list
    - verbose_output prints the detections to console
    ----------
    """

    n_images = len(images)

    # Preprocessing
    if(n_images == 1):
        x = preprocess_image_eval(images[0], network_dim, letterbox).unsqueeze(0)
    else:
        x = torch.zeros((n_images, IMG_CHANNEL_COUNT, network_dim, network_dim), dtype=TORCH_FLOAT, device=get_device())
        for i, image in enumerate(images):
            x[i] = preprocess_image_eval(image, network_dim, letterbox)

    # Running the model
    predictions = model(x)

    # Extracting detections
    detections = extract_detections(predictions, model.get_yolo_layers())

    for i in range(n_images):
        img_h = images[i].shape[CV2_H_DIM]
        img_w = images[i].shape[CV2_W_DIM]

        detections[i] = map_dets_to_original_image(detections[i], img_h, img_w, network_dim, letterbox)

    return detections

# inference_on_video
def inference_on_video(model, video_in, video_out, class_names, network_dim, letterbox, frame_print_mod=DETECT_VIDEO_FRAME_MOD):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Runs inference detection given a yolo model
    - Computes detections on all frames in video_in and writes the resulting frames to video_out
    - frame_print_mod controls when to print out the number of frames processed.
        For example: 100 prints out number of frames processed every 100 frames
    ----------
    """

    # Reading video frame by frame, getting detections, and writing to output video
    done = False
    frame_count = 0
    while(not done):
        ret, frame = video_in.read()
        # ret is False when there's no more frames in the video
        if(ret):
            detections = inference_on_single_image(model, frame, network_dim, letterbox)
            output_frame = write_dets_to_image(detections, frame, class_names, verbose_output=False)

            video_out.write(output_frame)

            # See utilities.constants.py to tweak frame modulus
            frame_count += 1
            if((frame_count % frame_print_mod == 0) and (frame_print_mod > 0)):
                print("Processed frames:", frame_count)

        else:
            done = True
