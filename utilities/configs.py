import torch
import torch.nn as nn

from .constants import *
from model.darknet import Darknet
from model.layers.convolutional import ConvolutionalLayer
from model.layers.maxpool import MaxpoolLayer
from model.layers.route import RouteLayer
from model.layers.shortcut import ShortcutLayer
from model.layers.upsample import UpsampleLayer
from model.layers.yolo import YoloLayer

# parse_names
def parse_names(names_path):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses a names file into a list where each index has the name of the corresponding class prediction
    - File should have each class name on each line in order of class prediction index
    ----------
    """

    with open(names_path, "r") as i_stream:
        lines = i_stream.readlines()

    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]

    return lines

# parse_config
def parse_config(config_path):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses a darknet configuration file and returns a pytorch darknet model
    - Pytorch darknet model also contains the network_block

    - A block is a dictionary of named parameters. For example to access batch in a net block,
      you would use block["batch"].
    - "DARKNET_CONFIG_BLOCK_TYPE" is a special parameter that tells what type the block is (convolutional, maxpool, etc.).
    - A network block contains a list of darknet hyperparameters (see [net] block of a darknet config)
    - The pytorch darknet model is instantiated on the CPU, you will need to send it to a CUDA
      device if you want to use the GPU
    ----------
    """

    with open(config_path, "r") as i_stream:
        lines = i_stream.readlines()

    # Remove trailing whitespace, empty lines, and comments
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = [line for line in lines if line[0] != '#']

    blocks = parse_lines_into_blocks(lines)
    model = parse_blocks_into_model(blocks)

    # Making sure hyperparameters that need to be the same, are the same
    consistent = verify_yolo_hyperparams(model.get_yolo_layers())
    if(not consistent):
        model = None

    return model

# verify_yolo_hyperparams
def verify_yolo_hyperparams(yolo_layers):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Verifies hyperparams that should be the same across yolo layers (like nms_kind) are the same
    - Returns True if all is good, False if all is not good
    ----------
    """

    if(len(yolo_layers) > 1):
        ref_layer = yolo_layers[0]

        for yolo_layer in yolo_layers[1:]:
            if(yolo_layer.nms_kind != ref_layer.nms_kind):
                print("verify_yolo_hyperparams: Error: nms_kind not consistent across all yolo layers")
                return False

    return True

# parse_lines_into_blocks
def parse_lines_into_blocks(lines):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses given lines into a list of blocks in order they appear
    ----------
    """

    blocks = []
    new_block = None

    for i, line in enumerate(lines):
        # Checking if it's a block header
        if((line[0] == '[') or (line[-1] == ']')):
            if(new_block is not None):
                blocks.append(new_block)

            new_block = {}
            new_block[DARKNET_CONFIG_BLOCK_TYPE] = line[1:-1]

        # Error out if no block header
        elif(new_block is None):
            print("parse_lines_into_blocks: Error on line", i, ": No block header before data being set")
            return None

        # Parse out data for the block (key=value)
        else:
            key_value = line.split("=")
            key_value = [s.strip() for s in key_value]

            if(len(key_value) != 2):
                print("parse_lines_into_blocks: Error on line", i, ": Bad syntax (key=value)")
                return None

            new_block[key_value[0]] = key_value[1]
    # end for

    if(new_block is not None):
        blocks.append(new_block)

    return blocks

# parse_blocks_into_model
def parse_blocks_into_model(blocks):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses a list of blocks and returns the darknet model
    - Model defaults on CPU
    ----------
    """

    net_block = None
    darknet_model = None
    layer_output_channels = []
    module_list = nn.ModuleList()

    # Config must start with net block
    net_block = blocks[0]
    if(net_block[DARKNET_CONFIG_BLOCK_TYPE] != "net"):
        print("parse_blocks_into_model: Error: Config file must start with [net] block.")
        return None

    layer_count = 0
    for block in blocks[1:]:
        layer, out_channels = parse_block(block, layer_output_channels, layer_count)

        module_list.append(layer)
        layer_output_channels.append(out_channels)
        layer_count += 1

    # end for

    # Checking for error state
    if(None not in module_list):
        darknet_model = Darknet(module_list, net_block)

    return darknet_model

# parse_block
def parse_block(block, layer_output_channels, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses a block into a darknet layer
    - Needs a list of all previous layer output channels
    - Returns the parsed layer and its output_channels
    - Output channels for the layer can be None if no output
    ----------
    """

    block_type = block[DARKNET_CONFIG_BLOCK_TYPE]
    parsed_layer = None
    out_channels = None

    # The incoming channel count (some layers don't use this)
    if(len(layer_output_channels) == 0):
        cur_channels = IMG_CHANNEL_COUNT
    else:
        cur_channels = layer_output_channels[-1]

    # Convolutional
    if(block_type == "convolutional"):
        parsed_layer, out_channels = parse_convolutional_block(block, cur_channels, layer_idx)

    # Maxpool
    elif(block_type == "maxpool"):
        parsed_layer = parse_maxpool_block(block, layer_idx)
        out_channels = cur_channels

    # Route
    elif(block_type == "route"):
        parsed_layer, out_channels = parse_route_block(block, layer_output_channels, layer_idx)

    # Shortcut
    elif(block_type == "shortcut"):
        parsed_layer =  parse_shortcut_block(block, layer_idx)
        out_channels = cur_channels

    # Upsample
    elif(block_type == "upsample"):
        parsed_layer = parse_upsample_block(block, layer_idx)
        out_channels = cur_channels

    # Yolo
    elif(block_type == "yolo"):
        parsed_layer = parse_yolo_block(block, layer_idx)
        out_channels = None # Yolo layers do not have an output

    # Invalid net block
    elif(block_type == "net"):
        print("parse_block: Error on layer idx", layer_idx, ": Should only be one [net] block at the very start of the config")

    # Unrecognized
    else:
        print("parse_blocks_into_model: Error on layer idx", layer_idx, ": Unrecognized block type:", block_type)


    return parsed_layer, out_channels


##### LAYER PARSING #####

# parse_convolutional_block
def parse_convolutional_block(block, in_channels, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses convolutional block into a ConvolutionalLayer
    ----------
    """

    conv_layer = None

    batch_normalize = CONV_BN
    filters = None
    size = None
    stride = None
    pad = CONV_PAD
    activation = CONV_ACTIV

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "batch_normalize"):
            batch_normalize = bool(int(v))
        elif(k == "filters"):
            out_channels = int(v)
        elif(k == "size"):
            size = int(v)
        elif(k == "stride"):
            stride = int(v)
        elif(k == "pad"):
            pad = bool(int(v))
        elif(k == "activation"):
            activation = v
        else:
            print("parse_convolutional_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # Validation
    if(out_channels is None):
        print("parse_convolutional_block: Error: Missing 'filters' entry")
        error_state = True
    if(size is None):
        print("parse_convolutional_block: Error: Missing 'size' entry")
        error_state = True
    if(stride is None):
        print("parse_convolutional_block: Error: Missing 'stride' entry")
        error_state = True

    # Setting up ConvolutionalLayer
    if(not error_state):
        conv_layer = ConvolutionalLayer(in_channels, out_channels, size, stride, batch_normalize, pad, activation)

    return conv_layer, out_channels

# parse_maxpool_block
def parse_maxpool_block(block, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses maxpool block into a MaxpoolLayer
    ----------
    """

    maxpool_layer = None

    size = None
    stride = None

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "size"):
            size = int(v)
        elif(k == "stride"):
            stride = int(v)
        else:
            print("parse_maxpool_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # end for

    # Validation
    if(size is None):
        print("parse_maxpool_block: Error on layer_idx", layer_idx, ": Missing 'size' entry")
        error_state = True
    if(stride is None):
        print("parse_maxpool_block: Error on layer_idx", layer_idx, ": Missing 'stride' entry")
        error_state = True

    # Setting up MaxpoolLayer
    if(not error_state):
        maxpool_layer = MaxpoolLayer(size, stride)

    return maxpool_layer

# parse_route_block
def parse_route_block(block, layer_output_channels, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses route block into a RouteLayer
    ----------
    """

    route_layer = None
    out_channels = None

    layers = None

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "layers"):
            layers = [int(l.strip()) for l in v.split(",")]
        else:
            print("parse_route_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # end for

    # Validation
    if(layers is None):
        print("parse_route_block: Error on layer_idx", layer_idx, ": Missing 'layers' entry")
        error_state = True

    # Setting up RouteLayer
    if(not error_state):
        route_layer = RouteLayer(layers)
        out_channels = 0
        for l in layers:
            out_channels += layer_output_channels[l]

    # end if

    return route_layer, out_channels

# parse_shortcut_block
def parse_shortcut_block(block, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses shortcut block into a ShortcutLayer
    ----------
    """

    shortcut_layer = None

    from_entry = None
    activation = SHCT_ACTIV

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "from"):
            from_entry = int(v)
        elif(k == "activation"):
            activation = v
        else:
            print("parse_shortcut_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # end for

    # Validation
    if(from_entry is None):
        print("parse_shortcut_block: Error on layer_idx", layer_idx, ": Missing 'from' entry")
        error_state = True

    # Setting up ShortcutLayer
    if(not error_state):
        shortcut_layer = ShortcutLayer(from_entry, activation)

    return shortcut_layer

# parse_upsample_block
def parse_upsample_block(block, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses upsample block into a UpsampleLayer
    ----------
    """

    upsample_layer = None

    stride = None

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "stride"):
            stride = int(v)
        else:
            print("parse_upsample_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # end for

    # Validation
    if(stride is None):
        print("parse_upsample_block: Error on layer_idx", layer_idx, ": Missing 'stride' entry")
        error_state = True

    # Setting up UpsampleLayer
    if(not error_state):
        upsample_layer = UpsampleLayer(stride)

    return upsample_layer

# parse_yolo_block
def parse_yolo_block(block, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses yolo block into a YoloLayer
    ----------
    """

    yolo_layer = None

    mask = None
    anchors = None
    classes = None
    num = None
    jitter = YOLO_JITTER
    ignore_thresh = None
    truth_thresh = None
    random = YOLO_RANDOM
    scale_xy = YOLO_SCALEXY
    iou_thresh = YOLO_IOU_THRESH
    cls_norm = YOLO_CLS_NORM
    iou_norm = YOLO_IOU_NORM
    iou_loss = YOLO_IOU_LOSS
    nms_kind = YOLO_NMS_KIND
    beta_nms = YOLO_BETA_NMS
    max_delta = YOLO_MAX_DELTA

    error_state = False

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "mask"):
            mask = [int(m) for m in v.split(",")]
        elif(k == "classes"):
            classes = int(v)
        elif(k == "num"):
            num = int(v)
        elif(k == "jitter"):
            jitter = float(v)
        elif(k == "ignore_thresh"):
            ignore_thresh = float(v)
        elif(k == "truth_thresh"):
            truth_thresh = float(v)
        elif(k == "random"):
            random = bool(int(v))
        elif(k == "scale_x_y"):
            scale_xy = float(v)
        elif(k == "iou_thresh"):
            iou_thresh = float(v)
        elif(k == "cls_normalizer"):
            cls_norm = float(v)
        elif(k == "iou_normalizer"):
            iou_norm = float(v)
        elif(k == "iou_loss"):
            iou_loss = v
        elif(k == "nms_kind"):
            nms_kind = v
        elif(k == "beta_nms"):
            beta_nms = float(v)
        elif(k == "max_delta"):
            beta_nms = int(v)
        elif(k == "anchors"):
            anchors_temp = [int(m) for m in v.split(",")]
            if(len(anchors_temp) % 2 != 0):
                print("parse_yolo_block: Error on layer_idx", layer_idx, ": All anchors must have width and height")
                return None

            i = 0
            anchors = []
            while i < len(anchors_temp):
                anchor = (anchors_temp[i], anchors_temp[i+1])
                anchors.append(anchor)

                i += 2

        else:
            print("parse_yolo_block: Error on layer_idx", layer_idx, ": Unrecognized block key:", k)
            error_state = True

    # end for

    # Validation
    if(mask is None):
        print("parse_yolo_block: Error on layer_idx", layer_idx, ": Missing 'mask' entry")
        error_state = True
    if(anchors is None):
        print("parse_yolo_block: Error on layer_idx", layer_idx, ": Missing 'anchors' entry")
        error_state = True
    if(classes is None):
        print("parse_yolo_block: Error on layer_idx", layer_idx, ": Missing 'classes' entry")
        error_state = True
    if(num is None):
        print("parse_yolo_block: Error on layer_idx", layer_idx, ": Missing 'num' entry")
        error_state = True
    if(ignore_thresh is None):
        print("parse_yolo_block: Error on layer_idx", layer_idx, ": Missing 'ignore_thresh' entry")
        error_state = True

    # Anchor mask validation
    if(not error_state):
        if(num != len(anchors)):
            print("parse_yolo_block: Error on layer_idx", layer_idx, ": 'num' entry must equal number of anchors")
            error_state = True
        else:
            for m in mask:
                if(m < 0 or m >= num):
                    print("parse_yolo_block: Error on layer_idx", layer_idx, ": Mask index is invalid for given anchors")


    # Setting up YoloLayer
    if(not error_state):
        yolo_layer = YoloLayer(anchors, mask, classes, ignore_thresh, truth_thresh, random,
                                jitter, scale_xy, iou_thresh, cls_norm, iou_norm, iou_loss,
                                nms_kind, beta_nms, max_delta)

    return yolo_layer
