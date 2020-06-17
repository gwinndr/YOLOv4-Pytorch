import torch

from utilities.constants import *

from utilities.arguments import parse_detect_args
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights


from utilities.devices import gpu_device_name, get_device, use_cuda
from utilities.file_reading import load_image, load_frame

from utilities.inferencing import inference_on_single_image, inference_on_video
from utilities.image_processing import write_dets_to_image

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
        print(SEPARATOR)
        print("")

        # Print network
        if(args.print_network):
            model.print_network()

        # TODO
        network_dim = int(model.net_block["width"])
        letterbox = not args.no_letterbox

        ##### IMAGE DETECTION #####
        if(not args.video):
            image = load_image(args.input)
            if(image is None):
                return

            detections = inference_on_single_image(model, image, network_dim, letterbox)
            output_image = write_dets_to_image(detections, image, class_names, verbose_output=True)

            cv2.imwrite(args.output, output_image)

            if(not args.no_show):
                cv2.imshow("Detections", output_image)
                cv2.waitKey(0)

        ##### VIDEO DETECTION #####
        else:
            video_in = cv2.VideoCapture(args.input)
            if(video_in.isOpened()):
                # Getting input video hyperparameters for the output video
                vid_w  = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
                vid_h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_dims = (vid_w, vid_h)
                fourcc = int(video_in.get(cv2.CAP_PROP_FOURCC))
                fps = int(video_in.get(cv2.CAP_PROP_FPS))

                video_out = cv2.VideoWriter(args.output, fourcc, fps, vid_dims, CV2_IS_COLOR)

                inference_on_video(model, video_in, video_out, class_names, network_dim, letterbox, DETECT_VIDEO_FRAME_MOD)

                video_in.release()
                video_out.release()

    return

if __name__ == "__main__":
    main()
