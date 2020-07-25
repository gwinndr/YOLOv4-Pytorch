import numpy as np

# load_weights
def load_weights(model, weights_file):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Loads weights from file into the darknet model
    ----------
    """

    with open(weights_file, "rb") as i_stream:
        # Header:
        # 1) Major version - int32
        # 2) Minor version - int32
        # 3) Build number - int32
        # 4) Num images seen - int64
        version = np.fromfile(i_stream, dtype = np.int32, count = 3)
        imgs_seen = np.fromfile(i_stream, dtype = np.int64, count = 1)[0]

        # Converting version to string "major.minor.build"
        version = ".".join([str(v) for v in version])

        model.version = version
        model.imgs_seen = int(imgs_seen)

        # The rest are weights
        weights = np.fromfile(i_stream, dtype = np.float32)

        layers = model.get_layers()
        cur_pos = 0

        for layer in layers:
            if(layer.has_learnable_params):
                if(cur_pos == len(weights)):
                    print("")
                    print("======= WARNING: Weights file has less weights than learnable parameters =======")
                    break

                cur_pos = layer.load_weights(weights, cur_pos)

        if(cur_pos != len(weights)):
            print("")
            print("======= WARNING: Weights file has more weights than learnable parameters =======")

        return
