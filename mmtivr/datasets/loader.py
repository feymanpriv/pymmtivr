#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""


import os

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from mmtivr.core.config import cfg
from mmtivr.datasets.dataset import VideoTextDataset
from mmtivr.datasets.dataset import FramesTextDataset

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _construct_loader(dataset_name, split, batch_size, shuffle, collect, drop_last):
    """Constructs the data loader for the given dataset."""
    data_path = os.path.join(_DATA_DIR, dataset_name)
    # Construct the dataset
    if cfg.DATA_LOADER.TYPE == "VideoTextLoader":
    	dataset = VideoTextDataset(data_path, split)
    elif cfg.DATA_LOADER.TYPE == "FramesTextLoader":
    	dataset = FramesTextDataset(data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        collect = collect,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        collect=collate_fn,
        drop_last=True,
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), err_str
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)


def collate_fn(batch_data):
	"""Collect data."""
    frames = torch.stack([item[0] for item in batch_data])
    captions = [cap for item in batch_data for cap in item[1]]
    return frames, captions
