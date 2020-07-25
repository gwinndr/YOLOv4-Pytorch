import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from utilities.constants import *
from utilities.lr import LearnRateConstant, LearnRateSteps

# build_optimizer
def build_optimizer(model):
    """
    ----------
    Author: Damon Gwinn (gwinndr)
    ----------
    - Builds an optimizer and corresponding scheduler based on model parameters
    - Returns the optimizer and scheduler (scheduler should step every batch)
    ----------
    """

    net = model.net_block

    # Learn rate is divided by the batch size in darknet
    lr = net.lr / net.batch
    batches_seen = model.imgs_seen // net.batch

    sgd = SGD(model.parameters(), lr=lr, momentum=net.momentum, weight_decay=net.decay)

    # Learn rate scheduler
    if(net.policy == POLICY_CONSTANT):
        scaler = LearnRateConstant(batches_seen, net.burn_in, net.power)
    elif(net.policy == POLICY_STEPS):
        scaler = LearnRateSteps(batches_seen, net.burn_in, net.power, net.steps, net.scales)
    else:
        print("----- Warning: Optimizer policy '%s' is not supported. Defaulting to '%s' -----" % (net.policy, POLICY_CONSTANT))
        scaler = LearnRateConstant(batches_seen, net.burn_in, net.power)

    scheduler = LambdaLR(sgd, scaler.step)

    return sgd, scheduler

# get_optimizer_lr
def get_optimizer_lr(optimizer):
    """
    ----------
    Author: Damon Gwinn
    ----------
    - Given an optimizer, returns the learn rate
    - A little hacky tacky, but it gets the job done :-)
    ----------
    """

    for param_group in optimizer.param_groups:
        return param_group["lr"]
