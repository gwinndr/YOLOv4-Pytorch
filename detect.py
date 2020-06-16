import torch
import cv2

from utilities.constants import *
from utilities.arguments import parse_detect_args

from utilities.devices import gpu_device_name, get_device, use_cuda
from utilities.configs import parse_config, parse_names
from utilities.weights import load_weights
from utilities.image_processing import preprocess_image_eval, tensor_to_image, map_dets_to_original_image, write_dets_to_image
from utilities.extract_detections import extract_detections
from utilities.file_reading import load_image

# main
def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for generating labels on a given image, image folder, or video
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

        # Loading image
        image = load_image(args.img)
        if(image is None):
            return

        img_h = image.shape[CV2_H_DIM]
        img_w = image.shape[CV2_W_DIM]

        # Preprocessing
        letterbox = not args.no_letterbox
        x = preprocess_image_eval(image, network_dim, letterbox).unsqueeze(0)

        # Running the model
        predictions = model(x)

        # Extracting detections
        detections = extract_detections(predictions, model.get_yolo_layers())
        detections = detections[0]
        detections = map_dets_to_original_image(detections, img_h, img_w, network_dim, letterbox)

        # Putting detections on the image
        new_image = write_dets_to_image(detections, image, class_names, verbose_output=True)

        cv2.imwrite(args.output_img, new_image)

        if(not args.no_show):
            cv2.imshow("Detections", new_image)
            cv2.waitKey(0)



if __name__ == "__main__":
    main()
