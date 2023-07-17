# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MAL/blob/main/LICENSE

"""Cosine learning rate scheduler."""

import math


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.train.lr * (epoch / cfg.train.warmup_epochs)
    else:
        lr = cfg.train.min_lr + (cfg.train.lr - cfg.train.min_lr) * 0.5 * \
            (1. + math.cos(
                math.pi * (epoch - cfg.train.warmup_epochs) /
                (cfg.train.num_epochs - cfg.train.warmup_epochs) * cfg.train.num_wave))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
