##### TWEAKABLE CONSTANTS #####
START_CHANNEL_COUNT = 3 # How many channels in the input image
YOLO_LEAKY_SLOPE = 0.1 # Slope for leaky relu
UPSAMPLE_MODE = "nearest" # Type of interpolation for upsampling

##### CONVOLUTIONAL DEFAULT CONSTANTS #####
CONV_BN = False
CONV_PAD = False
CONV_ACTIV = "linear"

##### SHORTCUT CONSTANTS #####
SHCT_ACTIV = "linear"

##### YOLO CONSTANTS #####
YOLO_RANDOM = False
YOLO_JITTER = 0
YOLO_SCALEXY = 1
YOLO_IOU_THRESH = 0.25
YOLO_CLS_NORM = 1
YOLO_IOU_NORM = 1
YOLO_IOU_LOSS = "ciou"
YOLO_NMS_KIND = "greedynms"
YOLO_BETA_NMS = 0.6
YOLO_MAX_DELTA = 5


##### CONFIGURATION FILE CONSTANTS #####
DARKNET_CONFIG_BLOCK_TYPE = "DARKNET_CONFIG_BLOCK_TYPE"

##### MISC #####
SEPARATOR = "========================="
CHANNEL_DIM = 1
