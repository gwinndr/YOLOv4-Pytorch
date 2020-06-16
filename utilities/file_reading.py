import cv2

from utilities.constants import *

# load_image
def load_image(image_f):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Loads in an image from file using cv2
    - Image is always a 3-Channel color image
    - Returns None if invalid
    ----------
    """

    image = cv2.imread(image_f)
    if(image is None):
        print("load_image: Error: Could not read file:", image_f)
    elif(image.shape[CV2_C_DIM] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif(image.shape[CV2_C_DIM] != 3):
        print("load_image: Error: With image file:", image_f)
        print("    Expected 1 or 3 color channels, got:", image.shape[CV2_C_DIM])
        image = None

    return image


# # read_image_dir
# def load_image_dir(image_dir, sort=True):
#     image_fs = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
#     image_fs = [f for f in image_fs if os.path.isfile(f)]
#
#     if(sort):
#         image_fs.sort()
#
#     images = [load_image(f) for f in image_fs]
#     images = [img for img in images if img is not None]
#
#     return images
#
# # load_video
# def load_video(video_file):
#     capture = cv2.VideoCapture(video_file)
#     if(capture is None):
#         print("load_video: Error: Could not load video file:", video_file)
#
#     return capture
