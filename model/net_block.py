from utilities.constants import *

class NetBlock:
    def __init__(self,
        batch=NET_BATCH_DEF, subdivisions=NET_SUBDIV_DEF, width=NET_W_DEF, height=NET_H_DEF,
        channels=NET_CH_DEF, momentum=NET_MOMEN_DEF, decay=NET_DECAY_DEF, angle=NET_ANG_DEF,
        saturation=NET_SATUR_DEF, exposure=NET_EXPOS_DEF, hue=NET_HUE_DEF, flip=NET_FLIP_DEF, lr=NET_LR_DEF,
        burn_in=NET_BURN_DEF, power=NET_POW_DEF, max_batches=NET_MAX_BAT_DEF, policy=NET_POL_DEF,
        steps=NET_STEP_DEF, scales=NET_SCALE_DEF, mosaic=NET_MOSAIC_DEF, resize_step=NET_RESIZE_STEP_DEF,
        jitter=NET_JITTER_DEF, random=NET_RAND_DEF, resize=NET_RESIZE_DEF, nms_kind=NET_NMS_DEF):

        self.batch = batch
        self.subdivisions = subdivisions
        self.width = width
        self.height = height
        self.channels = channels
        self.momentum = momentum
        self.decay = decay
        self.angle = angle
        self.saturation = saturation
        self.exposure = exposure
        self.hue = hue
        self.flip = flip
        self.lr = lr
        self.burn_in = burn_in
        self.power = power
        self.max_batches = max_batches
        self.policy = policy
        self.steps = steps
        self.scales = scales
        self.mosaic = mosaic
        self.resize_step = resize_step
        self.jitter = jitter
        self.random = random
        self.resize = resize
        self.nms_kind = nms_kind

    def to_string(self):
        ret = "NETWORK: batch: %d  subdivs: %d  width: %d  height: %d  channels: %d  resize_step: %d\n" \
              "         lr: %f  momentum: %f  decay: %f  burn_in: %d  power: %.2f  max_batches: %d\n" \
              "         angle: %f  saturation: %f  exposure: %f  hue: %f  flip: %d  mosaic: %d\n" \
              "         jitter: %f  random: %f  resize: %f  nms_kind: '%s'\n" \
              "         policy: '%s'" % \
              (self.batch, self.subdivisions, self.width, self.height, self.channels, self.resize_step, \
               self.lr, self.momentum, self.decay, self.burn_in, self.power, self.max_batches, \
               self.angle, self.saturation, self.exposure, self.hue, self.flip, self.mosaic, \
               self.jitter, self.random, self.resize, self.nms_kind, \
               self.policy)

        if(self.policy == POLICY_STEPS):
            steps = ",".join([str(s) for s in self.steps])
            scales = ",".join([str(s) for s in self.scales])

            ret += "  steps: %s  scales: %s" % (steps, scales)

        return ret
