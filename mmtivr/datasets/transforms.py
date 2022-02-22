#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Transform."""


import numpy as np
from PIL import Image

import torch
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from mmtivr.core.config import cfg


_MEAN = (0.48145466, 0.4578275, 0.40821073)
_SD = (0.26862954, 0.26130258, 0.27577711)


def get_transform(_split):
    """Frames train(test) transformation."""
    if "test" in _split or "val" in _split: 
        transform_fn = transforms.Compose([
            transforms.Resize(cfg.TEST.IM_SIZE, interpolation=BICUBIC),
            transforms.CenterCrop(cfg.TRAIN.IM_SIZE),
            transforms.Normalize(mean=_MEAN, std=_SD),
        ])
    elif "train" in _split: 
        transform_fn = transforms.Compose([
            transforms.RandomResizedCrop(cfg.TRAIN.IM_SIZE, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.Normalize(mean=_MEAN, std=_SD),
        ])
    return transform_fn
