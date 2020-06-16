import torch
import cv2

##### TWEAKABLE CONSTANTS #####
# NN Constants
YOLO_LEAKY_SLOPE = 0.1 # Slope for leaky relu
UPSAMPLE_MODE = "nearest" # Type of interpolation for upsampling
NMS_THRESHOLD = 0.45 # Threshold for considering two bounding boxes overlapping (IOU) for NM
CV2_INTERPOLATION = cv2.INTER_LINEAR # Interpolation method for resizing imagesS

MISH_BETA = 1 # The beta for each mish activation
MISH_THRESHOLD = 20 # The threshold for each mish activation

OBJ_THRESH_DEFAULT = 0.25

LETTERBOX_DEFAULT = True
INPUT_DIM_DEFAULT = 608

# May experiment with precision stuff in future
TORCH_FLOAT = torch.float32

IMG_CHANNEL_COUNT = 3

##### BBOX DRAWING #####
BBOX_INCLUDE_CLASS_CONF = True

BBOX_COLORS = ( (255,0,255),(0,0,255),(0,255,255),(0,255,0),(255,255,0),(255,100,100) )
COLOR_BLACK = (0,0,0)
BBOX_FONT = cv2.FONT_HERSHEY_SIMPLEX
BBOX_RECT_THICKNESS = 2
BBOX_FONT_SCALE = 0.8
BBOX_FONT_THICKNESS = 1
CV2_RECT_FILL = -1

BBOX_TEXT_LEFT_PAD = 2
BBOX_TEXT_RIGHT_PAD = 2
BBOX_TEXT_TOP_PAD = 5
BBOX_TEXT_BOT_PAD = 6

CV2_TEXT_SIZE_W = 0
CV2_TEXT_SIZE_H = 1

##### SUPPORTED NMS TYPES #####
GREEDY_NMS = "greedynms"

##### CONVOLUTIONAL DEFAULT CONSTANTS #####
CONV_BN = False
CONV_PAD = False
CONV_ACTIV = "linear"

##### SHORTCUT CONSTANTS #####
SHCT_ACTIV = "linear"

##### YOLO CONSTANTS #####
# Layer init defaults
YOLO_RANDOM = False
YOLO_JITTER = 0
YOLO_SCALEXY = 1
YOLO_IOU_THRESH = 0.25
YOLO_CLS_NORM = 1
YOLO_IOU_NORM = 1
YOLO_IOU_LOSS = "ciou"
YOLO_NMS_KIND = GREEDY_NMS
YOLO_BETA_NMS = 0.6
YOLO_MAX_DELTA = 5

# Object threshold default
YOLO_OBJ_THRESH = 0.25

# BBOX attributes
YOLO_TX = 0
YOLO_TY = 1
YOLO_TW = 2
YOLO_TH = 3

# YOLO confidence scores
YOLO_OBJ = 4
YOLO_CLASS_START = 5

# Raw output dimensions from yolo model when in eval mode
YOLO_OUT_BATCH_DIM = 0
YOLO_OUT_N_PREDS_DIM = 1
YOLO_OUT_ATTRS_DIM = 2

# Output order for each extracted yolo detection
DETECTION_X1 = 0
DETECTION_Y1 = 1
DETECTION_X2 = 2
DETECTION_Y2 = 3
DETECTION_CLASS_IDX = 4
DETECTION_CLASS_PROB = 5

DETECTION_N_ELEMS = 6


# Other
YOLO_N_BBOX_ATTRS = 5 # x,y,w,h,obj

##### CONFIGURATION FILE CONSTANTS #####
DARKNET_CONFIG_BLOCK_TYPE = "DARKNET_CONFIG_BLOCK_TYPE"

##### PREPROCESSING #####
LETTERBOX_COLOR = 0.5

##### MISC #####
SEPARATOR = "========================="

BATCH_DIM = 0
CHANNEL_DIM = 1
X_DIM = 2
Y_DIM = 3

CV2_H_DIM = 0
CV2_W_DIM = 1
CV2_C_DIM = 2
CV2_N_DIMS = 3

INPUT_C_DIM = 0
INPUT_H_DIM = 1
INPUT_W_DIM = 2

BBOX_X1 = 0
BBOX_Y1 = 1
BBOX_X2 = 2
BBOX_Y2 = 3
BBOX_N_ELEMS = 4
