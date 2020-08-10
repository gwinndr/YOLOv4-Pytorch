import cv2
import sys

##### ACTIVATION SPECIFIC CONSTANTS #####
LEAKY_RELU_SLOPE = 0.1 # Slope for leaky relu
MISH_BETA = 1 # The beta for each mish activation
MISH_THRESHOLD = 20 # The threshold for each mish activation


##### DETECTION EXTRACTION THRESHOLDING #####
NMS_THRESHOLD = 0.45 # Threshold for considering two bounding boxes overlapping (IOU) for NMS
OBJ_THRESH_DEF = 0.25


##### AUGMENTATION CONSTANTS #####
# Offset forces mosaic to not place the cut point more than 20% into any side
MOSAIC_MIN_OFFSET = 0.2

LETTERBOX_COLOR = 0.5
CV2_INTERPOLATION = cv2.INTER_LINEAR # Interpolation method for resizing imagesS
### END AUGMENTATION CONSTANTS ###


##### SUPPORTED POLICIES #####
POLICY_CONSTANT = "constant"
POLICY_STEPS = "steps"


##### SUPPORTED NMS TYPES #####
GREEDY_NMS = "greedynms"


##### SUPPORTED BBOX LOSS #####
BBOX_MSE_LOSS = "mse"
BBOX_CIOU_LOSS = "ciou"


##### RANDOM RESIZING CONSTANTS #####
N_BATCH_TO_RANDOM_RESIZE = 10
NET_RAND_COEF_IF_1 = 1.4


##### INPUT CONSTANTS #####
IMG_CHANNEL_COUNT = 3

INPUT_BATCH_DIM = 0
INPUT_CHANNEL_DIM = 1
INPUT_H_DIM = 2
INPUT_W_DIM = 3

TENSOR_C_DIM = 0
TENSOR_H_DIM = 1
TENSOR_W_DIM = 2

BBOX_X1 = 0
BBOX_Y1 = 1
BBOX_X2 = 2
BBOX_Y2 = 3
BBOX_N_ELEMS = 4
### END INPUT CONSTANTS ###


##### CV2 CONSTANTS #####
CV2_H_DIM = 0
CV2_W_DIM = 1
CV2_C_DIM = 2
CV2_N_DIMS = 3 # Number of cv2 dimensions

# If using an image in float32 form (0-1), then the max h value is 360 <-- we do this
# If using an image in uint8 form (0-255), then the max h value is 179
CV2_HSV_H_MAX = 360.0

# Flipping magic
CV2_FLIP_VERTICAL = 0
CV2_FLIP_HORIZONTAL = 1
CV2_FLIP_BOTH = -1
### END CV2 Constants ###


##### BENCHMARKING #####
# Number of times to run the model on a random image as a warmup when benchmarking
BENCHMARK_N_WARMUPS = 25
# Benchmarking (MODEL_ONLY is equivalent to official darknet benchmark fps)
NO_BENCHMARK = 0 # No benchmarking
MODEL_ONLY = 1 # Fps running the model only (recommended)
MODEL_WITH_PP = 2 # Fps running pre/post-processing + MODEL_ONLY
MODEL_WITH_IO = 3 # Fps running file io + MODEL_WITH_PP


##### CONFIGURATION FILE CONSTANTS #####
DARKNET_CONFIG_BLOCK_TYPE = "DARKNET_CONFIG_BLOCK_TYPE"


##### NET BLOCK #####
NET_BATCH_DEF = 1
NET_SUBDIV_DEF = 1
NET_W_DEF = 416
NET_H_DEF = 416
NET_CH_DEF = 3
NET_MOMEN_DEF = 0.9
NET_DECAY_DEF = 0.0001
NET_ANG_DEF = 0
NET_SATUR_DEF = 1
NET_EXPOS_DEF = 1
NET_HUE_DEF = 0
NET_FLIP_DEF = 1
NET_LR_DEF = 0.001
NET_BURN_DEF = 0
NET_POW_DEF = 4
NET_MAX_BAT_DEF = 500
NET_POL_DEF = POLICY_CONSTANT
NET_STEP_DEF = (1,)
NET_SCALE_DEF = (1,)
NET_MOSAIC_DEF = 0
NET_RESIZE_STEP_DEF = 32
# Note: present on yolo_layer but moved to net_block when parsing configs
NET_JITTER_DEF = 0.2
NET_RAND_DEF = 0
NET_RESIZE_DEF = 1.0
NET_NMS_DEF = GREEDY_NMS
### END NET BLOCK ###


##### CONVOLUTIONAL CONSTANTS #####
# Layer init defaults
CONV_FILT_DEF = 1
CONV_SIZE_DEF = 1
CONV_STRIDE_DEF = 1
CONV_BN_DEF = 0
CONV_PAD_DEF = 0
CONV_ACTIV_DEF = "linear"


##### SHORTCUT CONSTANTS #####
# Layer init defaults
SHCT_ACTIV_DEF = "linear"


##### MAXPOOL CONSTANTS #####
MAXPL_SIZE_DEF = 1
MAXPL_STRIDE_DEF = 1
MAXPL_PAD_DEF = MAXPL_SIZE_DEF - 1


##### UPSAMPLE CONSTANTS #####
UPSAMP_STRIDE_DEF = 2
UPSAMP_MODE_DEF = "nearest" # Type of interpolation for upsampling


##### YOLO CONSTANTS #####
# Layer init defaults
YOLO_NCLS_DEF = 20
YOLO_NUM_DEF = 1
YOLO_IGNORE_DEF = 0.5
YOLO_TRUTH_DEF = 1.0
YOLO_RANDOM_DEF = NET_RAND_DEF
YOLO_JITTER_DEF = NET_JITTER_DEF
YOLO_RESIZE_DEF = NET_RESIZE_DEF
YOLO_SCALEXY_DEF = 1.0
YOLO_IOU_THRESH_DEF = 1.0
YOLO_CLS_NORM_DEF = 1.0
YOLO_IOU_NORM_DEF = 1.0
YOLO_IOU_LOSS_DEF = BBOX_MSE_LOSS
YOLO_NMS_KIND_DEF = NET_NMS_DEF
YOLO_BETA_NMS_DEF = 0.6
YOLO_MAX_DELTA_DEF = sys.float_info.max

# YOLO BBOX-specific attributes
YOLO_TX = 0
YOLO_TY = 1
YOLO_TW = 2
YOLO_TH = 3
YOLO_OBJ = 4
YOLO_N_BBOX_ATTRS = 5 # x,y,w,h,obj
YOLO_CLASS_START = 5 # start of class confidences
### END YOLO CONSTANTS ###


##### OUTPUT CONSTANTS #####
# Raw output dimensions from darknet model when in eval mode
PREDS_BATCH_DIM = 0
PREDS_N_PREDS_DIM = 1
PREDS_ATTRS_DIM = 2

# Output order for each extracted darknet detection
DETECTION_X1 = 0
DETECTION_Y1 = 1
DETECTION_X2 = 2
DETECTION_Y2 = 3
DETECTION_CLASS_START = 4 # Class scores for the n classes
### END OUTPUT CONSTANTS ###


##### ANNOTATION CONSTANTS
# Darknet bbox annotation format
ANN_BBOX_X1 = 0
ANN_BBOX_Y1 = 1
ANN_BBOX_X2 = 2
ANN_BBOX_Y2 = 3
ANN_BBOX_CLASS = 4
ANN_BBOX_N_ELEMS = 5 # Number of attributes for each annotation

# Value for padding annotations for batched annotations
ANN_PAD_VAL = -1

# Coco bbox annotation format
COCO_ANN_BBOX_X = 0
COCO_ANN_BBOX_Y = 1
COCO_ANN_BBOX_W = 2
COCO_ANN_BBOX_H = 3
### END ANNOTATION CONSTANTS ###


##### COCO EVAL #####
COCO_ANN_TYPE_SEGM = "segm"
COCO_ANN_TYPE_BBOX = "bbox"
COCO_ANN_TYPE_KEYP = "keypoints"

# Index with yolo's 80 class id to get the corresponding coco 91 class id
COCO_80_TO_91 = ( 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 )

# Returned coco stats
COCO_STAT_LAYOUT = "type--IOU--area--maxDets"

COCO_STAT_0_NAME = "AP--0.50:0.95--all--100"
COCO_STAT_1_NAME = "AP--0.50--all--100"
COCO_STAT_2_NAME = "AP--0.75--all--100"
COCO_STAT_3_NAME = "AP--0.50:0.95--small--100"
COCO_STAT_4_NAME = "AP--0.50:0.95--medium--100"
COCO_STAT_5_NAME = "AP--0.50:0.95--large--100"

COCO_STAT_6_NAME = "AR--0.50:0.95--all--1"
COCO_STAT_7_NAME = "AR--0.50:0.95--all--10"
COCO_STAT_8_NAME = "AR--0.50:0.95--all--100"
COCO_STAT_9_NAME = "AR--0.50:0.95--small--100"
COCO_STAT_10_NAME = "AR--0.50:0.95--medium--100"
COCO_STAT_11_NAME = "AR--0.50:0.95--large--100"

ALL_COCO_STAT_NAMES = (
    COCO_STAT_0_NAME, COCO_STAT_1_NAME, COCO_STAT_2_NAME, COCO_STAT_3_NAME, COCO_STAT_4_NAME,
    COCO_STAT_5_NAME, COCO_STAT_6_NAME, COCO_STAT_7_NAME, COCO_STAT_8_NAME, COCO_STAT_9_NAME,
    COCO_STAT_10_NAME, COCO_STAT_11_NAME
)
N_COCO_STATS = len(ALL_COCO_STAT_NAMES)
### END COCO EVAL ###


##### BBOX DRAWING #####
# Specific to draw_detections
BBOX_INCLUDE_CLASS_CONF = True

BBOX_COLORS = ( (255,0,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,100,100) )
COLOR_BLACK = (0,0,0)

BBOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_RECT_THICKNESS = 1
BBOX_FONT_SCALE = 0.8
BBOX_FONT_THICKNESS = 1
CV2_RECT_FILL = -1

BBOX_TEXT_LEFT_PAD = 2
BBOX_TEXT_RIGHT_PAD = 2
BBOX_TEXT_TOP_PAD = 5
BBOX_TEXT_BOT_PAD = 6

CV2_TEXT_SIZE_W = 0
CV2_TEXT_SIZE_H = 1
### END BBOX DRAWING ###


##### MISC #####
SEPARATOR = "========================="
CARRIAGE_RETURN = "\r"
