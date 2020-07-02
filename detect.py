import torch
import time

from utilities.constants import *

from utilities.arguments import parse_detect_args
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights

from utilities.devices import gpu_device_name, get_device, use_cuda
from utilities.file_io import load_image

from utilities.inferencing import inference_on_image, inference_video_to_video
from utilities.images import draw_detections

# main
def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for generating labels on a given image
    ----------
    """

    args = parse_detect_args()

    # Benchmarking information
    if(not args.benchmark):
        benchmark = NO_BENCHMARK
    elif(not args.video):
        print("Warning: Benchmarking is only available with a video input")
        benchmark = NO_BENCHMARK
    else:
        benchmark = args.benchmark_method
        if((benchmark < MODEL_ONLY) or (benchmark > MODEL_WITH_IO)):
            print("Unrecognized -benchmark_method. Please use 1 (MODEL_ONLY), 2 (MODEL_WITH_PP), or 3 (MODEL_WITH_IO)")
            return

    # no_grad disables autograd so our model runs faster
    with torch.no_grad():
        if(args.force_cpu):
            print("----- WARNING: Model is using the CPU (--force_cpu), expect model to run slower! -----")
            use_cuda(False)

        print("Parsing config into model...")
        model = parse_config(args.cfg)
        if(model is None):
            return

        model = model.to(get_device())
        model.eval()

        # Network input dim
        if(model.net_block["width"] != model.net_block["height"]):
            print("Error: Width and height must match in [net]")
            return

        network_dim = int(model.net_block["width"])

        print("Parsing class names...")
        class_names = parse_names(args.class_names)
        if(class_names is None):
            return

        print("Loading weights...")
        version, imgs_seen = load_weights(model, args.weights)

        print("")
        print(SEPARATOR)
        print("DARKNET")
        print("GPU:", gpu_device_name())
        print("Config:", args.cfg)
        print("Weights:", args.weights)
        print("Version:", ".".join([str(v) for v in version]))
        print("Images seen:", imgs_seen)
        print("Network Dim:", network_dim)
        print(SEPARATOR)
        print("")

        # Print network
        if(args.print_network):
            model.print_network()

        obj_thresh = args.obj_thresh
        letterbox = not args.no_letterbox

        ##### IMAGE DETECTION #####
        if(not args.video):
            image = load_image(args.input)
            if(image is None):
                return

            detections = inference_on_image(model, image, network_dim, obj_thresh, letterbox)
            output_image = draw_detections(detections, image, class_names, verbose_output=True)

            cv2.imwrite(args.output, output_image)

            if(not args.no_show):
                cv2.imshow("Detections", output_image)
                cv2.waitKey(0)

        ##### VIDEO DETECTION #####
        else:
            # Warm up the model for more accurate benchmarks
            if(benchmark != NO_BENCHMARK):
                print("Warming up model for benchmarks...")
                for i in range(BENCHMARK_N_WARMUPS):
                    warmup = torch.rand((IMG_CHANNEL_COUNT, network_dim, network_dim), dtype=torch.float32, device=get_device())
                    model(warmup.unsqueeze(0))
                print("Done!")
                print("")

            # Load video capture object
            video_in = cv2.VideoCapture(args.input)
            if(video_in.isOpened()):
                # Getting input video hyperparameters for the output video
                vid_w  = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_dims = (vid_w, vid_h)
                fourcc = int(video_in.get(cv2.CAP_PROP_FOURCC))
                fps = int(video_in.get(cv2.CAP_PROP_FPS))

                video_out = cv2.VideoWriter(args.output, fourcc, fps, vid_dims, isColor=True)

                fps = inference_video_to_video(model, video_in, video_out, class_names, network_dim, obj_thresh, letterbox, benchmark, verbose=True)
                print("")

                video_in.release()
                video_out.release()

                if(benchmark == MODEL_ONLY):
                    print("Model fps: %.2f" % fps)
                elif(benchmark == MODEL_WITH_PP):
                    print("Model fps with pre/post-processing: %.2f" % fps)
                elif(benchmark == MODEL_WITH_IO):
                    print("Model fps with file io and pre/post-processing: %.2f" % fps)

    return

if __name__ == "__main__":
    main()
