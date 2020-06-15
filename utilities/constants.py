import torch
import cv2

##### TWEAKABLE CONSTANTS #####
# NN Constants
START_CHANNEL_COUNT = 3 # How many channels in the input image
YOLO_LEAKY_SLOPE = 0.1 # Slope for leaky relu
UPSAMPLE_MODE = "nearest" # Type of interpolation for upsampling
NMS_THRESHOLD = 0.45 # Threshold for considering two bounding boxes overlapping (IOU) for NMS
# NMS_THRESHOLD = 1.5

LETTERBOX_DEFAULT = True
INPUT_DIM_DEFAULT = 608

# CV2 Constants
CV2_INTERPOLATION = cv2.INTER_LINEAR # Interpolation method for resizing images


# May experiment with precision stuff in future
TORCH_FLOAT = torch.float32

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
YOLO_OBJ = 4
YOLO_CLASS_START = 5

# Raw output dimensions from yolo model when in detection mode
YOLO_OUT_BATCH_DIM = 0
YOLO_OUT_N_PREDS_DIM = 1
YOLO_OUT_ATTRS_DIM = 2

# Output from detections extracted from Yolo
# Output is a list of tensors representing each batch
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

CV2_W_DIM = 1
CV2_H_DIM = 0
CV2_C_DIM = 2
CV2_N_IMG_DIM = 3

INPUT_C_DIM = 0
INPUT_H_DIM = 1
INPUT_W_DIM = 2

BBOX_X1 = 0
BBOX_Y1 = 1
BBOX_X2 = 2
BBOX_Y2 = 3
BBOX_N_ELEMS = 4
