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
                    print("")
                    break

                cur_pos = layer.load_weights(weights, cur_pos)

        if(cur_pos != len(weights)):
            print("")
            print("======= WARNING: Weights file has more weights than learnable parameters =======")
            print("")

        return

# write_weights
def write_weights(model, weights_file, max_layers=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Saves weights from darknet model to file
    - Will truncate or append 0s to make the version in format major.minor.build
    - max_layer will allow you to write only up to a certain layer (not including [net])
    ----------
    """

    with open(weights_file, "wb") as o_stream:
        # Header:
        # 1) Major version - int32
        # 2) Minor version - int32
        # 3) Build number - int32
        # 4) Num images seen - int64
        version = [int(v) for v in model.version.split(".")]
        if(len(version) < 3):
            version.extend([0] * 3 - len(version))

        version = np.array(version, dtype=np.int32)[:3]
        imgs_seen = np.int64(model.imgs_seen)

        version.tofile(o_stream)
        imgs_seen.tofile(o_stream)

        layers = model.get_layers()

        # Prepping max_layers
        if(max_layers is None):
            max_layers = len(layers)
        if((max_layers < 0) or (max_layers > len(layers))):
            print("====== WARNING: Trying to write weights with invalid max_layers=%d. Writing all layers. =====" % (max_layers))
            max_layers = len(layers)

        # Writing layers out
        for i in range(max_layers):
            layer = layers[i]
            if(layer.has_learnable_params):
                layer.write_weights(o_stream)

        return
