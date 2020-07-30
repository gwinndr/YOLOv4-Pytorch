import csv
import os

from utilities.constants import *

DETECTOR_BATCH_HEADER = ("Batch", "Learn Rate", "Loss")
DETECTOR_BATCH_TB_GROUP = "Batches"
DETECTOR_EPOCH_COCO_HEADER = ("Epoch", *ALL_COCO_STAT_NAMES)
DETECTOR_EPOCH_COCO_TB_GROUP = "Epochs"

# make_log_file
def make_log_file(log_f, header=None):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Creates a csv log file with the given header
    - Warning: Creates an empty csv file regardless if log_f already exists
    ----------
    """

    p, _ = os.path.split(log_f)
    os.makedirs(p, exist_ok=True)

    with open(log_f, "w", newline="") as log_stream:
        if(header is not None):
            writer = csv.writer(log_stream)
            writer.writerow(header)

    return

# log_detector_batch
def log_detector_batch(log_f, batch, learn_rate, loss):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Logs loss to csv file for a single batch on the detector
    - Creates log_f if it does not exist already, otherwise appends to it
    ----------
    """

    # Making the file and path if it doesn't exist already
    if(not os.path.isfile(log_f)):
        make_log_file(log_f, header=DETECTOR_BATCH_HEADER)

    # Writing batch and loss
    row = (batch, learn_rate, loss)
    with open(log_f, "a", newline="") as log_stream:
        writer = csv.writer(log_stream)
        writer.writerow(row)

    return

# log_detector_batch_tb
def log_detector_batch_tb(tensorboard_summary, batch, learn_rate, loss):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Logs loss to Tensorboard for a single batch on the detector
    ----------
    """

    name = "%s/Learn Rate" % DETECTOR_BATCH_TB_GROUP
    tensorboard_summary.add_scalar(name, learn_rate, global_step=batch)

    name = "%s/Loss" % DETECTOR_BATCH_TB_GROUP
    tensorboard_summary.add_scalar(name, loss, global_step=batch)

    tensorboard_summary.flush()

    return

# log_detector_epoch_coco
def log_detector_epoch_coco(log_f, epoch, coco_stats):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Logs coco statistics (AP and AR) to file for an epoch on the detector
    ----------
    """

    # Making the file and path if it doesn't exist already
    if(not os.path.isfile(log_f)):
        make_log_file(log_f, DETECTOR_EPOCH_COCO_HEADER)

    # Writing batch and loss
    row = (epoch, *coco_stats)
    with open(log_f, "a", newline="") as log_stream:
        writer = csv.writer(log_stream)
        writer.writerow(row)

    return

# log_detector_epoch_coco_tb
def log_detector_epoch_coco_tb(tensorboard_summary, epoch, coco_stats):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Logs coco statistics (AP and AR) to Tensorboard for an epoch on the detector
    ----------
    """

    for i in range(N_COCO_STATS):
        name = "%s/%s" % (DETECTOR_EPOCH_COCO_TB_GROUP, ALL_COCO_STAT_NAMES[i])
        tensorboard_summary.add_scalar(name, coco_stats[i], global_step=epoch)

    tensorboard_summary.flush()

    return
