import cv2

##### TWEAKABLE CONSTANTS #####
# NN Constants
YOLO_LEAKY_SLOPE = 0.1 # Slope for leaky relu
UPSAMPLE_MODE = "nearest" # Type of interpolation for upsampling
NMS_THRESHOLD = 0.45 # Threshold for considering two bounding boxes overlapping (IOU) for NMS
CV2_INTERPOLATION = cv2.INTER_LINEAR # Interpolation method for resizing imagesS

MISH_BETA = 1 # The beta for each mish activation
MISH_THRESHOLD = 20 # The threshold for each mish activation

OBJ_THRESH_DEFAULT = 0.25

LETTERBOX_DEFAULT = True
INPUT_DIM_DEFAULT = 608

# Number of times to run the model on a random image as a warmup when benchmarking
BENCHMARK_N_WARMUPS = 25

##### BENCHMARKING #####
# Benchmarking (MODEL_ONLY is equivalent to official darknet benchmark fps)
NO_BENCHMARK = 0 # No benchmarking
MODEL_ONLY = 1 # Fps running the model only (recommended)
MODEL_WITH_PP = 2 # Fps running pre/post-processing + MODEL_ONLY
MODEL_WITH_IO = 3 # Fps running file io + MODEL_WITH_PP

##### BBOX DRAWING #####
# Specific to draw_detections
BBOX_INCLUDE_CLASS_CONF = True

# Specific to both draw_detections and draw_annotations
BBOX_COLORS = ( (255,0,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,100,100) )
COLOR_BLACK = (0,0,0)

# Rest are specific to draw_bbox_inplace
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

##### COCO EVAL #####
COCO_ANN_TYPE_SEGM = "segm"
COCO_ANN_TYPE_BBOX = "bbox"
COCO_ANN_TYPE_KEYP = "keypoints"

# Index with yolo's 80 class id to get the corresponding coco 91 class id
COCO_80_TO_91 = ( 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 )

##### SUPPORTED NMS TYPES #####
GREEDY_NMS = "greedynms"

# Supported Bbox loss
MSE = "mse"
CIOU = "ciou"

##### CONVOLUTIONAL DEFAULT CONSTANTS #####
CONV_BN = False
CONV_PAD = False
CONV_ACTIV = "linear"

##### SHORTCUT CONSTANTS #####
SHCT_ACTIV = "linear"

##### MAXPOOL CONSTANTS #####
POOL_SPECIAL_PAD_MODE = "replicate"

##### YOLO CONSTANTS #####
# Layer init defaults
YOLO_RANDOM = False
YOLO_JITTER = 0
YOLO_SCALEXY = 1
YOLO_IOU_THRESH = 0.25
YOLO_CLS_NORM = 1
YOLO_IOU_NORM = 1
YOLO_IOU_LOSS = MSE
YOLO_NMS_KIND = GREEDY_NMS
YOLO_BETA_NMS = 0.6
YOLO_MAX_DELTA = 5

# YOLO BBOX-specific attributes
YOLO_TX = 0
YOLO_TY = 1
YOLO_TW = 2
YOLO_TH = 3

# YOLO confidence scores
YOLO_OBJ = 4
YOLO_CLASS_START = 5

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

# Other
YOLO_N_BBOX_ATTRS = 5 # x,y,w,h,obj

##### CONFIGURATION FILE CONSTANTS #####
DARKNET_CONFIG_BLOCK_TYPE = "DARKNET_CONFIG_BLOCK_TYPE"

##### PREPROCESSING #####
LETTERBOX_COLOR = 0.5

##### MISC #####
SEPARATOR = "========================="

IMG_CHANNEL_COUNT = 3

INPUT_BATCH_DIM = 0
INPUT_CHANNEL_DIM = 1
INPUT_H_DIM = 2
INPUT_W_DIM = 3

CV2_H_DIM = 0
CV2_W_DIM = 1
CV2_C_DIM = 2
CV2_N_DIMS = 3 # Number of cv2 dimensions

TENSOR_C_DIM = 0
TENSOR_H_DIM = 1
TENSOR_W_DIM = 2

BBOX_X1 = 0
BBOX_Y1 = 1
BBOX_X2 = 2
BBOX_Y2 = 3
BBOX_N_ELEMS = 4

# For printing on the same line
CARRIAGE_RETURN = "\r"
