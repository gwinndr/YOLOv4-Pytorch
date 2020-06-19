import time

from utilities.constants import *

from utilities.devices import get_device, synchronize_device
from utilities.image_processing import preprocess_image_eval, map_dets_to_original_image, write_dets_to_image
from utilities.extract_detections import extract_detections

# inference_on_single_image
def inference_on_single_image(model, image, network_dim, obj_thresh, letterbox):
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

    img_h = image.shape[CV2_H_DIM]
    img_w = image.shape[CV2_W_DIM]

    # Preprocessing
    x = preprocess_image_eval(image, network_dim, letterbox).unsqueeze(0)

    # Running the model
    predictions = model(x)

    # Postprocessing
    detections = extract_detections(predictions, model.get_yolo_layers(), obj_thresh)
    detections = map_dets_to_original_image(detections[0], img_h, img_w, network_dim, letterbox)

    # Note, not in list format (since it's only one image)
    return detections

# inference_on_images
def inference_on_images(model, images, network_dim, obj_thresh, letterbox, benchmark_model=False):
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
    x = torch.zeros((n_images, IMG_CHANNEL_COUNT, network_dim, network_dim), dtype=TORCH_FLOAT, device=get_device())
    for i, image in enumerate(images):
        x[i] = preprocess_image_eval(image, network_dim, letterbox)

    # Running the model
    predictions = model(x)

    # Postprocessing
    detections = extract_detections(predictions, model.get_yolo_layers(),obj_thresh)

    for i in range(n_images):
        img_h = images[i].shape[CV2_H_DIM]
        img_w = images[i].shape[CV2_W_DIM]

        detections[i] = map_dets_to_original_image(detections[i], img_h, img_w, network_dim, letterbox)

    return detections

# inference_video_to_video
def inference_video_to_video(model, video_in, video_out, class_names, obj_thresh, network_dim, letterbox, benchmark=NO_BENCHMARK, verbose=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Runs inference detection given a yolo model
    - Computes detections on all frames in video_in and writes the resulting frames to video_out
    - verbose prints frames processed with running fps tracking (if applicable)
    - Returns None if no benchmark, otherwise returns benchmark fps based on benchmark enum:
        - NO_BENCHMARK: Does not run any benchmarking (None)
        - MODEL_ONLY: Only running the darknet model
        - MODEL_WITH_PP: Darknet model with all pre and post-processing
        - MODEL_WITH_IO: Darknet model with all pre and post-processing plus file io time
    ----------
    """

    # Benchmarking
    fps = None
    sum_time = 0

    # Reading video frame by frame, getting detections, and writing to output video
    done = False
    frame_count = 0
    while(not done):
        # start MODEL_WITH_IO
        if(benchmark == MODEL_WITH_IO):
            synchronize_device()
            start_time = time.time()

        # Read video frame
        ret, frame = video_in.read()

        # ret is False when there's no more frames in the video
        if(ret):
            # start MODEL_WITH_PP
            if(benchmark == MODEL_WITH_PP):
                synchronize_device()
                start_time = time.time()

            # Preprocessing
            x = preprocess_image_eval(frame, network_dim, letterbox).unsqueeze(0)

            # start MODEL_ONLY
            if(benchmark == MODEL_ONLY):
                synchronize_device()
                start_time = time.time()

            # Running the model
            predictions = model(x)

            # end MODEL_ONLY
            if(benchmark == MODEL_ONLY):
                synchronize_device()
                sum_time += time.time() - start_time

            # Postprocessing
            frame_h = frame.shape[CV2_H_DIM]
            frame_w = frame.shape[CV2_W_DIM]
            detections = extract_detections(predictions, model.get_yolo_layers(), obj_thresh)
            detections = map_dets_to_original_image(detections[0], frame_h, frame_w, network_dim, letterbox)
            output_frame = write_dets_to_image(detections, frame, class_names, verbose_output=False)

            # end MODEL_WITH_PP
            if(benchmark == MODEL_WITH_PP):
                synchronize_device()
                sum_time += time.time() - start_time

            # Output video
            video_out.write(output_frame)

            # end MODEL_WITH_IO
            if(benchmark == MODEL_WITH_IO):
                synchronize_device()
                sum_time += time.time() - start_time

            frame_count += 1

            if(verbose):
                print("Processed frames:", frame_count, end="")

                if(benchmark != NO_BENCHMARK):
                    fps_so_far = frame_count / sum_time
                    print("  FPS: %.2f" % fps_so_far, end=CARRIAGE_RETURN, flush=True)
                else:
                    print("", end=CARRIAGE_RETURN, flush=True)
        else:
            done = True
    # while

    if(benchmark != NO_BENCHMARK):
        fps = frame_count / sum_time

    if(verbose):
        print("")

    return fps
