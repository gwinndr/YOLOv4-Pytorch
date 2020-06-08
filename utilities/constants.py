import cv2

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
# Layer init defaults
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

# Other
YOLO_N_BBOX_ATTRS = 5 # x,y,w,h,obj

##### CONFIGURATION FILE CONSTANTS #####
DARKNET_CONFIG_BLOCK_TYPE = "DARKNET_CONFIG_BLOCK_TYPE"

##### PREPROCESSING #####
LETTERBOX_COLOR = 128

##### MISC #####
SEPARATOR = "========================="
BATCH_DIM = 0
CHANNEL_DIM = 1
X_DIM = 2
Y_DIM = 3

YOLO_TX = 0
YOLO_TY = 1
YOLO_TW = 2
YOLO_TH = 3
YOLO_OBJ = 4
YOLO_CLASS_START = 5

CV2_W_DIM = 1
CV2_H_DIM = 0
CV2_C_DIM = 2
CV2_N_IMG_DIM = 3
CV2_INTERPOLATION = cv2.INTER_AREA
