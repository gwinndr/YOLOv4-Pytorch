import cv2

from utilities.constants import *

# load_image
def load_image(image_f):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Loads in an image from file using cv2
    - Converts from grayscale to color if applicable
    - Returns None if invalid
    ----------
    """

    image = cv2.imread(image_f)
    if(image is None):
        print("load_image: Error: Could not read image file:", image_f)
        return None

    fixed_image = fix_image_channels(image)
    if(fixed_image is None):
        print("load_image: Error: With image file:", image_f)
        print("    Expected 1 or 3 color channels, got:", image.shape[CV2_C_DIM])
        return None

    return fixed_image

def load_frame(video_capture):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Loads in a frame from a cv2 VideoCapture object
    - Converts from grayscale to color if applicable
    - Returns None if error or end of capture
    ----------
    """

    _, frame = cap.read()
    if(frame is None):
        print("load_frame: Hit end of video capture")
        return None
    else:
        fixed_frame = fix_image_channels(frame)
        if(fixed_image is None):
            print("load_frame: Error: Expected 1 or 3 color channels, got:", frame.shape[CV2_C_DIM])
            return None

        frame = fixed_frame

    return frame

# fix_image_channels
def fix_image_channels(image):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Fixes image to have 3 channels if grayscale (1 channel)
    - If image already has 3 channels, does nothing
    - Returns None if channel dims are not 1 or 3
    ----------
    """

    if(image.shape[CV2_C_DIM] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif(image.shape[CV2_C_DIM] != 3):
        image = None

    return image
