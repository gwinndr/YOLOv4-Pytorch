import torch
import traceback
import sys
import math

from datasets.coco import coco_evaluate_bbox

from utilities.constants import *
from utilities.devices import get_device
from utilities.logging import log_detector_epoch_coco, log_detector_epoch_coco_tb

# train_batch_batchloader
def train_batch_batchloader(model, minibatch_loader, optim, scheduler, print_shape=False):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Trains a batch using the BatchLoader class (utilities.loaders)
    - minibatch_loader should be a BatchLoader class with its batch size set to the mini-batch (batch / subdivisions)
    - Returns total_loss for the batch
    - NOTE: Does not track if this batch finishes an epoch
    ----------
    """

    net = model.net_block

    model.train()
    optim.zero_grad()

    total_loss = 0.0
    for subdiv in range(net.subdivisions):
        x, anns, img_ids = minibatch_loader.next_batch()
        try:
            x = x.to(get_device())
            anns = anns.to(get_device())

            # Sanity check
            if((subdiv == 0) and print_shape):
                print("Input shape:", list(x.shape))

            loss = model(x, anns=anns)
            loss.backward()

            total_loss += loss.item()

            print("Subdivisions: %d / %d" % (subdiv+1, net.subdivisions), end=CARRIAGE_RETURN)

            # Just making sure memory is freed up
            del loss

        except:
            img_ids = ", ".join([str(id.item()) for id in img_ids])

            print("")
            print("----- Exception occured on image ids:", img_ids, "-----")
            traceback.print_exc()
            sys.exit(1)
    # end for

    print("")

    optim.step()
    scheduler.step()

    # Average loss with respect to the batch size (yolo loss is already averaged by the minibatch size)
    avg_loss = total_loss / net.subdivisions

    return avg_loss

# evaluate_and_log_epoch_coco
def evaluate_and_log_epoch_coco(model, epoch, val_set, obj_thresh, epoch_log=None, tb_summary=None, max_imgs=0):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Evaluates a model epoch on MS-COCO and logs it
    - If epoch_log is None, does not log to csv
    - If tb_summary is None, does not log to tb_summary
    ----------
    """

    coco_stats = coco_evaluate_bbox(val_set, model, obj_thresh, max_imgs=max_imgs)

    # Logging
    if(epoch_log is not None):
        print("Logging to csv...")
        log_detector_epoch_coco(epoch_log, epoch, coco_stats)
    if(tb_summary is not None):
        print("Logging to Tensorboard...")
        # Doesn't work when epoch is a float so we'll do the ceiling
        epoch_adj = math.ceil(epoch)
        log_detector_epoch_coco_tb(tb_summary, epoch_adj, coco_stats)

    return
