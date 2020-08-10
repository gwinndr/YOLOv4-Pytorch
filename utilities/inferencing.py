import torch
import time

from utilities.constants import *

from utilities.devices import synchronize_device

from utilities.images import draw_detections
from utilities.preprocessing import preprocess_image_eval
from utilities.detections import extract_detections, correct_detections
from utilities.nms import run_nms

# inference
def inference(model, input_tensor, obj_thresh):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Runs yolo model and extracts detections
    - Input should be a tensor of the form: (BATCH, N_CHANNEL, H_DIM, W_DIM)
    - Returned detections are relative to the input tensor (see correct_detections in utilities.detections)
    ----------
    """

    model.eval()
    with torch.no_grad():
        # Running the model
        predictions = model(input_tensor)

        # Extracting detections from predictions
        detections = extract_detections(predictions, obj_thresh)

    return detections

# inference_on_image
def inference_on_image(model, image, network_dim, obj_thresh, letterbox=False, show_img=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Similar to inference except takes in an image as input (simpler to use)
    - Performs all needed pre and post processing
    - Returned detections are relative to the input image
    - show_img will show the augmented input image
    ----------
    """

    # Preprocessing
    input_tensor, img_info = preprocess_image_eval(image, network_dim, letterbox=letterbox, show_img=show_img)
    input_tensor = input_tensor.unsqueeze(0) # batch dim

    detections = inference(model, input_tensor, obj_thresh)[0]

    # Postprocessing
    detections = run_nms(detections, model, obj_thresh)
    detections = correct_detections(detections, img_info)

    return detections

# inference_video_to_video
def inference_video_to_video(model, video_in, video_out, class_names, network_dim, obj_thresh, letterbox=False, benchmark=NO_BENCHMARK, verbose=False):
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
            x, img_info = preprocess_image_eval(frame, network_dim, letterbox=letterbox)
            x = x.unsqueeze(0) # batch dim

            # start MODEL_ONLY
            if(benchmark == MODEL_ONLY):
                synchronize_device()
                start_time = time.time()

            # Running the model
            detections = inference(model, x, obj_thresh)[0]

            # end MODEL_ONLY
            if(benchmark == MODEL_ONLY):
                synchronize_device()
                sum_time += time.time() - start_time

            # Postprocessing
            detections = run_nms(detections, model, obj_thresh)
            detections = correct_detections(detections, img_info)
            output_frame = draw_detections(detections, frame, class_names, verbose_output=False)

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
