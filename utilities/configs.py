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

# parse_config
def parse_config(config_path):
    """
    ----------
    Author: Damon Gwinn
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

    model.cuda()

    return model

# parse_lines_into_blocks
def parse_lines_into_blocks(lines):
    """
    ----------
    Author: Damon Gwinn
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
    Author: Damon Gwinn
    ----------
    - Parses a list of blocks and returns the darknet model (present on CPU)
    ----------
    """

    net_block = None
    darknet_model = None
    module_list = nn.ModuleList()

    cur_channel_count = START_CHANNEL_COUNT

    for block in blocks:
        block_type = block[DARKNET_CONFIG_BLOCK_TYPE]

        # net block
        if(block_type == "net"):
            if(net_block is not None):
                print("parse_blocks_into_model: WARNING: Multiple [net] blocks found. Ignoring repeat blocks.")
            else:
                net_block = block

        elif(block_type == "convolutional"):
            conv_layer, cur_channel_count = parse_convolutional_block(block, cur_channel_count)
            module_list.append(conv_layer)
        elif(block_type == "maxpool"):
            module_list.append( parse_maxpool_block(block) )
        elif(block_type == "route"):
            module_list.append( parse_route_block(block) )
        elif(block_type == "shortcut"):
            module_list.append( parse_shortcut_block(block) )
        elif(block_type == "upsample"):
            module_list.append( parse_upsample_block(block) )
        elif(block_type == "yolo"):
            module_list.append( parse_yolo_block(block) )

        # Unrecognized
        else:
            print("parse_blocks_into_model: WARNING: Ignoring unrecognized block type:", block_type)

    # end for

    # Checking for error state
    if(None not in module_list):
        darknet_model = Darknet(module_list, net_block)

    return darknet_model


# parse_convolutional_block
def parse_convolutional_block(block, in_channels):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses convolutional block into a ConvolutionalLayer
    ----------
    """

    conv_layer = None

    batch_normalize = CONV_DEFAULT_BN
    filters = None
    size = None
    stride = None
    pad = CONV_DEFAULT_PAD
    activation = CONV_DEFAULT_ACTIV

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
            print("parse_convolutional_block: WARNING: Ignoring unrecognized block key:", k)

    # Validation
    error_state = False
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
def parse_maxpool_block(block):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses maxpool block into a MaxpoolLayer
    ----------
    """

    maxpool_layer = None

    size = None
    stride = None

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "size"):
            size = int(v)
        elif(k == "stride"):
            stride = int(v)
        else:
            print("parse_maxpool_block: WARNING: Ignoring unrecognized block key:", k)

    # end for

    # Validation
    error_state = False
    if(size is None):
        print("parse_maxpool_block: Error: Missing 'size' entry")
        error_state = True
    if(stride is None):
        print("parse_maxpool_block: Error: Missing 'stride' entry")
        error_state = True

    # Setting up MaxpoolLayer
    if(not error_state):
        maxpool_layer = MaxpoolLayer(size, stride)

    return maxpool_layer

# parse_route_block
def parse_route_block(block):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses route block into a RouteLayer
    ----------
    """

    route_layer = None

    layers = None

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "layers"):
            layers = [int(l.strip()) for l in v.split(",")]
        else:
            print("parse_route_block: WARNING: Ignoring unrecognized block key:", k)

    # end for

    # Validation
    error_state = False
    if(layers is None):
        print("parse_route_block: Error: Missing 'layers' entry")
        error_state = True

    # Setting up RouteLayer
    if(not error_state):
        route_layer = RouteLayer(layers)

    return route_layer

# parse_shortcut_block
def parse_shortcut_block(block):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses shortcut block into a ShortcutLayer
    ----------
    """

    shortcut_layer = None

    from_entry = None
    activation = SHCT_DEFAULT_ACTIV

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "from"):
            from_entry = int(v)
        elif(k == "activation"):
            activation = v
        else:
            print("parse_shortcut_block: WARNING: Ignoring unrecognized block key:", k)

    # end for

    # Validation
    error_state = False
    if(from_entry is None):
        print("parse_shortcut_block: Error: Missing 'from' entry")
        error_state = True

    # Setting up ShortcutLayer
    if(not error_state):
        shortcut_layer = ShortcutLayer(from_entry, activation)

    return shortcut_layer

# parse_upsample_block
def parse_upsample_block(block):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses upsample block into a UpsampleLayer
    ----------
    """

    upsample_layer = None

    stride = None

    # Filling in values
    for k, v in block.items():
        if(k == DARKNET_CONFIG_BLOCK_TYPE):
            continue # Ignore this key
        elif(k == "stride"):
            stride = int(v)
        else:
            print("parse_upsample_block: WARNING: Ignoring unrecognized block key:", k)

    # end for

    # Validation
    error_state = False
    if(stride is None):
        print("parse_upsample_block: Error: Missing 'stride' entry")
        error_state = True

    # Setting up UpsampleLayer
    if(not error_state):
        upsample_layer = UpsampleLayer(stride)

    return upsample_layer

# parse_yolo_block
def parse_yolo_block(block):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Parses yolo block into a YoloLayer
    ----------
    """

    yolo_layer = None

    mask = None
    anchors = None
    classes = None
    num = None
    jitter = None
    ignore_thresh = None
    truth_thresh = None
    random = None

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
        elif(k == "anchors"):
            anchors_temp = [int(m) for m in v.split(",")]
            if(len(anchors_temp) % 2 != 0):
                print("parse_yolo_block: Error: All anchors must have width and height")
                return None

            i = 0
            anchors = []
            while i < len(anchors_temp):
                anchor = (anchors_temp[i], anchors_temp[i+1])
                anchors.append(anchor)

                i += 2

        else:
            print("parse_yolo_block: WARNING: Ignoring unrecognized block key:", k)

    # end for

    # Validation (NOTE: Ignoring truth_thresh since it is unused in actual yolo)
    error_state = False
    if(mask is None):
        print("parse_yolo_block: Error: Missing 'mask' entry")
        error_state = True
    if(anchors is None):
        print("parse_yolo_block: Error: Missing 'anchors' entry")
        error_state = True
    if(classes is None):
        print("parse_yolo_block: Error: Missing 'classes' entry")
        error_state = True
    if(num is None):
        print("parse_yolo_block: Error: Missing 'num' entry")
        error_state = True
    if(jitter is None):
        print("parse_yolo_block: Error: Missing 'jitter' entry")
        error_state = True
    if(ignore_thresh is None):
        print("parse_yolo_block: Error: Missing 'ignore_thresh' entry")
        error_state = True
    if(random is None):
        print("parse_yolo_block: Error: Missing 'random' entry")
        error_state = True

    # More validation and extraction
    anchors_masked = []
    if(not error_state):
        if(num != len(anchors)):
            print("parse_yolo_block: Error: 'num' entry must equal number of anchors")
            error_state = True
        else:
            for m in mask:
                anchors_masked.append(anchors[m])


    # Setting up YoloLayer
    if(not error_state):
        yolo_layer = YoloLayer(anchors_masked, classes, ignore_thresh, truth_thresh)

    return yolo_layer
