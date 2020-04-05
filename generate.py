from utilities.configs import parse_config

# main
def main():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Entry point for generating labels on a given image
    ----------
    """

    config_path = "./configs/yolov3.cfg"
    model = parse_config(config_path)
    model.cuda()

    print(model)


if __name__ == "__main__":
    main()
