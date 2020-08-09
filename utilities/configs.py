import torch
import torch.nn as nn

from .constants import *
from model.darknet import Darknet
from model.net_block import NetBlock
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

    net_block = model.net_block
    yolo_layers = model.get_yolo_layers()

    ##### Alexey's darknet just uses the last yolo layer without verification #####
    ##### You can uncomment it if it makes you feel better, but make sure to update random in yolov4.cfg #####
    # # Making sure hyperparameters that need to be the same, are the same
    # consistent = verify_yolo_hyperparams(yolo_layers)
    # if(not consistent):
    #     model = None
    # else:
    #     yolo_hyperparams_to_netblock(net_block, yolo_layers)

    yolo_hyperparams_to_netblock(net_block, yolo_layers)

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

    if(type(yolo_layers) not in [list,tuple]):
        return True

    if(len(yolo_layers) > 1):
        ref_layer = yolo_layers[0]

        for yolo_layer in yolo_layers[1:]:
            is_bad = False

            if(yolo_layer.nms_kind != ref_layer.nms_kind):
                print("verify_yolo_hyperparams: Error: nms_kind not consistent across all yolo layers")
                is_bad = True
            if(yolo_layer.jitter != ref_layer.jitter):
                print("verify_yolo_hyperparams: Error: jitter not consistent across all yolo layers")
                is_bad = True
            if(yolo_layer.random != ref_layer.random):
                print("verify_yolo_hyperparams: Error: random not consistent across all yolo layers")
                is_bad = True

            if(is_bad):
                return False

    return True

# yolo_hyperparams_to_netblock
def yolo_hyperparams_to_netblock(net_block, yolo_layers):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Sets net to have yolo hyperparameters parameters (nms_kind, jitter, etc.)
    - Does not verify all yolo layers have the same augmentations params
    - Uses the hyperparams from the last yolo layer (if yolo_layers is a list or tuple)
    ----------
    """

    if(type(yolo_layers) in [list,tuple]):
        yolo_layer = yolo_layers[-1]
    else:
        yolo_layer = yolo_layers

    net_block.jitter = yolo_layer.jitter
    net_block.random = yolo_layer.random
    net_block.resize = yolo_layer.resize
    net_block.nms_kind = yolo_layer.nms_kind

    return


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

    net_block = parse_net_block(net_block)

    layer_count = 0
    for block in blocks[1:]:
        layer, out_channels = parse_layer_block(block, layer_output_channels, layer_count)

        module_list.append(layer)
        layer_output_channels.append(out_channels)
        layer_count += 1

    # end for

    # Checking for error state
    if(None not in module_list):
        darknet_model = Darknet(module_list, net_block)

    return darknet_model

# parse_layer_block
def parse_layer_block(block, layer_output_channels, layer_idx):
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


##### BLOCK PARSING #####
# find_option
def find_option(block, key, type, default, loud=False, layer_name=None, layer_idx=-1):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Finds the option specified by key and sets the value according to type
    - If option not found, uses default
    - If default is used and loud is True, prints that the default is being used
    - layer_name (str) and layer_idx (int) further specify layer details when printing with loud
    ----------
    """

    if(key in block.keys()):
        val = block[key]
    else:
        val = default
        if(loud):
            if(layer_name is None):
                label = ""
            else:
                label = "%s at idx %d:" % (layer_name, layer_idx)

            print(label, "Using default:", default, "for key:", key)

    if(val is not None):
        val = type(val)

    return val

# find_option_list
def find_option_list(block, key, type, default, loud=False, layer_name=None, layer_idx=-1):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Finds the option specified by key and creates a list of values according to type
    - Value is parsed according to a ','. Default is assumed to be a list.
    - If option not found, uses default
    - If default is used and loud is True, prints that the default is being used
    - layer_name (str) and layer_idx (int) further specify layer details when printing with loud
    ----------
    """

    if(key in block.keys()):
        val = block[key]
        val = val.split(",")
        val = [s.strip() for s in val]
    else:
        val = default
        if(loud):
            if(layer_name is None):
                label = ""
            else:
                label = "%s at idx %d:" % (layer_name, layer_idx)

            print(label, "Using default:", default, "for key:", key)

    if(val is not None):
        val = [type(v) for v in val]

    return val

# parse_net_block
def parse_net_block(block):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses network block into a NetBlock
    ----------
    """

    layer_name = "NetworkBlock"
    layer_idx = -1

    batch = find_option(block, "batch", int, NET_BATCH_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    subdiv = find_option(block, "subdivisions", int, NET_SUBDIV_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    width = find_option(block, "width", int, NET_W_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    height = find_option(block, "height", int, NET_H_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    channels = find_option(block, "channels", int, NET_CH_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    momentum = find_option(block, "momentum", float, NET_MOMEN_DEF)
    decay = find_option(block, "decay", float, NET_DECAY_DEF)
    angle = find_option(block, "angle", float, NET_ANG_DEF)
    saturation = find_option(block, "saturation", float, NET_SATUR_DEF)
    exposure = find_option(block, "exposure", float, NET_EXPOS_DEF)
    hue = find_option(block, "hue", float, NET_HUE_DEF)
    flip = find_option(block, "flip", int, NET_FLIP_DEF);
    lr = find_option(block, "learning_rate", float, NET_LR_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    burn_in = find_option(block, "burn_in", int, NET_BURN_DEF)
    power = find_option(block, "power", float, NET_POW_DEF)
    max_batches = find_option(block, "max_batches", int, NET_MAX_BAT_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    policy = find_option(block, "policy", str, NET_POL_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    steps = find_option_list(block, "steps", int, NET_STEP_DEF)
    scales = find_option_list(block, "scales", float, NET_SCALE_DEF)
    mosaic = find_option(block, "mosaic", int, NET_MOSAIC_DEF)
    resize_step = find_option(block, "resize_step", int, NET_RESIZE_STEP_DEF)

    net_block = NetBlock(
        batch=batch, subdivisions=subdiv, width=width, height=height, channels=channels,
        momentum=momentum, decay=decay, angle=angle, saturation=saturation, exposure=exposure,
        hue=hue, flip=flip, lr=lr, burn_in=burn_in, power=power, max_batches=max_batches,
        policy=policy, steps=steps, scales=scales, mosaic=mosaic, resize_step=resize_step
    )

    return net_block

# parse_convolutional_block
def parse_convolutional_block(block, in_channels, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses convolutional block into a ConvolutionalLayer
    ----------
    """

    layer_name = "ConvolutionalLayer"

    filters = find_option(block, "filters", int, CONV_FILT_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    size = find_option(block, "size", int, CONV_SIZE_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    stride = find_option(block, "stride", int, CONV_STRIDE_DEF)
    batch_normalize = find_option(block, "batch_normalize", int, CONV_BN_DEF)
    pad = find_option(block, "pad", int, CONV_PAD_DEF)
    activation = find_option(block, "activation", str, CONV_ACTIV_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)

    conv_layer = ConvolutionalLayer(
        in_channels, filters=filters, size=size, stride=stride, batch_normalize=batch_normalize,
        pad=pad, activation=activation
    )

    return conv_layer, filters

# parse_maxpool_block
def parse_maxpool_block(block, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses maxpool block into a MaxpoolLayer
    ----------
    """

    layer_name = "MaxpoolLayer"

    size = find_option(block, "size", int, MAXPL_SIZE_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    stride = find_option(block, "stride", int, MAXPL_STRIDE_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    padding = find_option(block, "padding", int, default=size-1)

    maxpool_layer = MaxpoolLayer(size=size, stride=stride, padding=padding)

    return maxpool_layer

# parse_route_block
def parse_route_block(block, layer_output_channels, layer_idx):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Parses route block into a RouteLayer
    - layer_output_channels should be a list containing all output channels up to this current layer
    ----------
    """

    layer_name = "RouteLayer"
    out_channels = 0

    layers = find_option_list(block, "layers", int, default=None)

    error_state = False

    # Validation
    if(layers is None):
        print("route: Error on layer_idx", layer_idx, ": Missing required 'layers' entry")
        error_state = True

    # Setting up RouteLayer
    if(not error_state):
        route_layer = RouteLayer(layers)

        for l in layers:
            out_channels += layer_output_channels[l]
    else:
        route_layer = None

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

    layer_name = "ShortcutLayer"

    from_layer = find_option(block, "from", int, default=None)
    activation = find_option(block, "activation", str, SHCT_ACTIV_DEF)

    error_state = False

    # Validation
    if(from_layer is None):
        print("shortcut: Error on layer_idx", layer_idx, ": Missing required 'from' entry")
        error_state = True

    # Setting up ShortcutLayer
    if(not error_state):
        shortcut_layer = ShortcutLayer(from_layer, activation=activation)
    else:
        shortcut_layer = None

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

    layer_name = "UpsampleLayer"

    stride = find_option(block, "stride", int, UPSAMP_STRIDE_DEF)

    upsample_layer = UpsampleLayer(stride=stride)

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

    layer_name = "YoloLayer"

    yolo_layer = None

    mask = find_option_list(block, "mask", int, default=None)
    anchors = find_option_list(block, "anchors", int, default=None)
    classes = find_option(block, "classes", int, YOLO_NCLS_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    num = find_option(block, "num", int, YOLO_NUM_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    jitter = find_option(block, "jitter", float, YOLO_JITTER_DEF)
    resize = find_option(block, "resize", float, YOLO_RESIZE_DEF)
    ignore_thresh = find_option(block, "ignore_thresh", float, YOLO_IGNORE_DEF, loud=True, layer_name=layer_name, layer_idx=layer_idx)
    truth_thresh = find_option(block, "truth_thresh", float, YOLO_TRUTH_DEF)
    random = find_option(block, "random", float, YOLO_RANDOM_DEF)
    scale_xy = find_option(block, "scale_x_y", float, YOLO_SCALEXY_DEF)
    iou_thresh = find_option(block, "iou_thresh", float, YOLO_IOU_THRESH_DEF)
    cls_norm = find_option(block, "cls_normalizer", float, YOLO_CLS_NORM_DEF)
    iou_norm = find_option(block, "iou_normalizer", float, YOLO_IOU_NORM_DEF)
    iou_loss = find_option(block, "iou_loss", str, YOLO_IOU_LOSS_DEF)
    nms_kind = find_option(block, "nms_kind", str, YOLO_NMS_KIND_DEF)
    beta_nms = find_option(block, "beta_nms", float, YOLO_BETA_NMS_DEF)
    max_delta = find_option(block, "max_delta", float, YOLO_MAX_DELTA_DEF)

    error_state = False

    # Validation
    if(mask is None):
        print("yolo: Error on layer_idx", layer_idx, ": Missing 'mask' entry")
        error_state = True
    if(anchors is None):
        print("yolo: Error on layer_idx", layer_idx, ": Missing 'anchors' entry")
        error_state = True

    # Validating anchors
    if(not error_state):
        if(len(anchors) % 2 != 0):
            print("yolo: Error on layer_idx", layer_idx, ": All anchors must have width and height")
            error_state = True
        else:
            i = 0
            anchors_temp = anchors
            anchors = []
            while i < len(anchors_temp):
                anchor = (anchors_temp[i], anchors_temp[i+1])
                anchors.append(anchor)
                i += 2

            # Verifying anchor and mask integrity
            if(num != len(anchors)):
                print("yolo: Error on layer_idx", layer_idx, ": 'num' entry must equal number of anchors")
                error_state = True
            for m in mask:
                if((m < 0) or (m >= num)):
                    print("yolo: Error on layer_idx", layer_idx, ": Mask index", m, "is invalid for given num entry")
                    error_state = True

    # end if

    # Setting up YoloLayer
    if(not error_state):
        yolo_layer = YoloLayer(
            anchors, mask, n_classes=classes, ignore_thresh=ignore_thresh, truth_thresh=truth_thresh,
            random=random, jitter=jitter, resize=resize, scale_xy=scale_xy, iou_thresh=iou_thresh,
            cls_norm=cls_norm, iou_norm=iou_norm, iou_loss=iou_loss, nms_kind=nms_kind,
            beta_nms=beta_nms, max_delta=max_delta
        )
    else:
        yolo_layer = None

    return yolo_layer
