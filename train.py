import torch
import random
import os
import shutil
import time

from utilities.constants import *
from utilities.devices import get_device
from utilities.arguments import parse_train_args
from utilities.configs import parse_config
from utilities.weights import load_weights, write_weights

from datasets.coco import CocoDataset

from utilities.loaders import BatchLoader
from utilities.logging import log_detector_batch, log_detector_batch_tb
from utilities.training import train_batch_batchloader, evaluate_and_log_epoch_coco
from utilities.optimizer import build_optimizer, get_optimizer_lr
from utilities.augmentations import possible_image_sizings

def main():
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Entry point for training a darknet model
    ----------
    """

    args = parse_train_args()

    train_imgs = os.path.realpath(args.train_imgs)
    train_anns = os.path.realpath(args.train_anns)

    val_imgs = os.path.realpath(args.val_imgs)
    val_anns = os.path.realpath(args.val_anns)

    print("Parsing config into model...")
    model = parse_config(args.cfg)
    if(model is None):
        return

    model = model.to(get_device())
    model.train()

    net = model.net_block

    # Network input dim
    if(net.width != net.height):
        print("Error: Width and height must match in [net]")
        return

    mini_batch = net.batch // net.subdivisions

    print("Loading weights...")
    load_weights(model, args.weights)

    # For testing purposes
    # model.imgs_seen = 0
    # net.max_batches = 5000

    print("Building optimizer...")
    optim, scheduler = build_optimizer(model)

    ##### Setting up results #####
    print("Setting up results folder:", args.results)
    weights_dir = os.path.join(args.results, "weights")
    info_f = os.path.join(args.results, "info.txt")

    # Sanity check so you don't lose all your hard work
    if(os.path.isdir(args.results)):
        print("")
        print("---- WARNING: Results folder already exists: %s ----" % args.results)
        print("This could really mess things up if you're not meaning to continue from a checkpoint")
        if(not args.no_ask):
            print("")
            while(True):
                user_in = input("Are you absolutely positively sure you want to continue?(y/n): ")
                if(user_in == "y"):
                    break
                elif(user_in == "n"):
                    print("Crisis averted, don't forget to leave a tip :-)")
                    return
        print("")

    os.makedirs(weights_dir, exist_ok=True)

    # Writing the info.txt with hyperparam information
    print("Writing info.txt...")
    if(os.path.isfile(info_f)):
        info_mode = "a"
    else:
        info_mode = "w"
    with open(info_f, info_mode, newline="") as info_stream:
        print("Detector trained and evaluated on MS-COCO", file=info_stream)
        print("cfg:", args.cfg, file=info_stream)
        print("weights:", args.weights, file=info_stream)
        print("obj_thresh:", args.obj_thresh, file=info_stream)
        print("max_imgs:", args.max_imgs, file=info_stream)
        print(net.to_string(), file=info_stream)
        print(SEPARATOR, file=info_stream)

    # Copying config file to results which acts as a nice little sanity check :-)
    print("Copying config...")
    shutil.copy(args.cfg, args.results)

    # Log csv setup
    if(args.batch_csv):
        batch_log = os.path.join(args.results, "batch_log.csv")
    else:
        batch_log = None
    if(args.epoch_csv):
        epoch_log = os.path.join(args.results, "epoch_log.csv")
    else:
        epoch_log = None

    # Tensorboard setup
    if(args.tensorboard):
        print("Setting up tensorboard...")
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = os.path.join(args.results, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboard_dir)
    else:
        tensorboard_dir = None
        tensorboard_summary = None

    print("")

    print("Building datasets:")
    init_dim = net.width
    train_set = CocoDataset(train_imgs, input_dim=init_dim, letterbox=True, annotation_file=train_anns)
    val_set = CocoDataset(val_imgs, input_dim=init_dim, letterbox=True, annotation_file=val_anns)
    print("")

    print("Building BatchLoader...")
    train_loader = BatchLoader(train_set, mini_batch, shuffle=True, drop_last=True)
    print("")

    ##### Train setup #####
    # Building random resizing list
    if(net.random != 0.0):
        rand_coef = net.random if (net.random != 1.0) else NET_RAND_COEF_IF_1
        resizings = possible_image_sizings(init_dim, rand_coef, net.resize_step)

        # First resize will be the max so we test if there's enough memory
        max_dim = resizings[-1]
        train_set.resize(max_dim)

        init_dim = max_dim

        # Uncomment to test out different random coefficients
        # print(rand_coef)
        # print(resizings)
        # import sys
        # sys.exit()
    else:
        resizings = None

    # Getting position in training
    batches_trained = model.imgs_seen // net.batch
    epochs_trained = model.imgs_seen // len(train_set)

    ### Too unstable and slow ###
    # # Evaluate on epoch 0 for a baseline error
    # if(epochs_trained == 0):
    #     print("Evaluating epoch 0 as a baseline mAP and mAR...")
    #     evaluate_and_log_epoch_coco(
    #         model, epochs_trained, val_set, args.obj_thresh,
    #         epoch_log=epoch_log, tb_summary=tensorboard_summary, max_imgs=args.max_imgs
    #     )
    #     print("")

    # For tracking when we finished a epoch
    progress_epoch = 0

    # Filenames for printing
    cfg_fname = os.path.basename(args.cfg)
    weights_fname = os.path.basename(args.weights)

    print("Resizings:", resizings)
    print("----- Initial input dim: %d x %d -----" % (init_dim, init_dim))

    ##### Training Loop #####
    while(batches_trained < net.max_batches):
        cur_lr = get_optimizer_lr(optim)

        print(SEPARATOR)
        print("Batches: %d / %d  Epochs: %d" % (batches_trained, net.max_batches, epochs_trained))
        print("Model: %s  Weights: %s" % (cfg_fname, weights_fname))
        print("Learn Rate:", cur_lr)
        print("")

        # Training ze batch
        before = time.time()
        loss = train_batch_batchloader(model, train_loader, optim, scheduler, print_shape=True)
        after = time.time()

        print("")
        print("Loss: %.4f" % loss)
        print("")
        print("Time taken: %.2f seconds" % (after - before))

        # Logging
        if(batch_log is not None):
            print("Logging to csv...")
            log_detector_batch(batch_log, batches_trained, cur_lr, loss)
        if(tensorboard_summary is not None):
            print("Logging to Tensorboard...")
            log_detector_batch_tb(tensorboard_summary, batches_trained, cur_lr, loss)

        print(SEPARATOR)
        print("")

        batches_trained += 1
        model.imgs_seen += net.batch
        progress_epoch += net.batch

        # After n batches, random resize (if applicable)
        if((resizings is not None) and (batches_trained % N_BATCH_TO_RANDOM_RESIZE == 0)):
            new_dim = random.choice(resizings)
            print("----- Resizing input to %d x %d -----" % (new_dim, new_dim))
            train_set.resize(new_dim)

        # Finished an epoch, evaluate mAP and mAR
        if(progress_epoch >= len(train_set)):
            epochs_trained += 1
            progress_epoch = 0

            print("----- Finished epoch -----")
            print("Num epochs trained:", epochs_trained)
            print("")

            # Write weights
            if((epochs_trained % args.epoch_mod == 0) and (not args.only_save_last)):
                print("Writing weights...")
                cur_weights = os.path.join(weights_dir, "weights_epoch_%d.weights" % epochs_trained)
                write_weights(model, cur_weights)
                print("")

            # Log that epoch
            print("Evaluating epoch...")
            evaluate_and_log_epoch_coco(
                model, epochs_trained, val_set, args.obj_thresh,
                epoch_log=epoch_log, tb_summary=tensorboard_summary, max_imgs=args.max_imgs
            )
            print("")

        # end if
    # end while

    print("")
    print("---- Finished Training! ----")
    print("")

    # Will log if it's a partial epoch
    final_epoch = round((progress_epoch / len(train_set)) + epochs_trained, 4)
    if(final_epoch != epochs_trained):
        print("Part way through an epoch, evaluating and logging")
        evaluate_and_log_epoch_coco(
            model, final_epoch, val_set, args.obj_thresh,
            epoch_log=epoch_log, tb_summary=tensorboard_summary, max_imgs=args.max_imgs
        )
        print("")

    print("Saving final weights...")
    final_weights = os.path.join(weights_dir, "final_weights.weights")
    write_weights(model, final_weights)
    print("")

    # Sanity check just to make sure everything is gone
    if(tensorboard_summary is not None):
        tensorboard_summary.flush()

    print("Goodbye!")
    print("")

    return


if __name__ == "__main__":
    main()
