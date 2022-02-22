#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Video(frames)-text dataset."""


import os
import random
import numpy as np
from PIL import Image

import torch
import decord
decord.bridge.set_bridge("torch")

import mmtivr.core.logging as logging
from mmtivr.utils.file_io import read_json
from mmtivr.core.config import cfg
from mmtivr.dataset.transforms import get_transform

logger = logging.get_logger(__name__)


class VideoTextDataset(torch.utils.data.Dataset):
    """Video text dataset."""
    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing dataset ...")
        self._data_path = data_path
        self._split = split
        self.video_setting = cfg.DATA_LOADER.VIDEO_SETTING
        self.text_setting = cfg.DATA_LOADER.TEXT_SETTING
        self.transform = get_transform(self._split)
        self._construct_vtdb()
    
    def _construct_vtdb(self):
        """Constructs the db."""
        self.data = read_json(os.path.join(self._data_path, self._split))
        logger.info("Number of pairs: {}".format(len(self._data)))

    def __getitem__(self, index):
        """Load data"""
        video_name, total_frames, captions = self.data[index]
        frames, frame_idxs = self._get_video(video_name)
        # preprocess frames
        frames = prepare_frames(frames, self.transform)
        captions = self._get_text(captions)
        return frames, captions
    
    def _get_video(self, name):
        """Video loader"""
        frames, frame_idxs = read_frames_from_video(
            os.path.join(self.video_setting.VIDEO_PATH, name),
            self.video_setting.NUM_FRAMES,
            self.video_setting.SAMPLE,
            self.video_setting.FIX_START
            )
        return frames, frame_idxs
    
    def _get_text(self, captions):
        """Text loader"""
        if self.text_setting.SAMPLE == 'random':
            sampled_captions = [random.choice(captions)]
        elif self.text_setting.SAMPLE == 'all':
            sampled_captions = captions
        elif self.text_setting.SAMPLE.startswith('mt-rand'):
            sampled_captions = random.sample(captions, 
                int(self.text_setting.SAMPLE.split('_')[1]))
        return sampled_captions
    
    def __len__(self):
        """__len__"""
        return len(self.data)


class FramesTextDataset(torch.utils.data.Dataset):
    """Frames text dataset."""
    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing dataset ...")
        self._data_path = data_path
        self._split = split
        self.video_setting = cfg.DATA_LOADER.VIDEO_SETTING
        self.text_setting = cfg.DATA_LOADER.TEXT_SETTING
        self.transform = get_transform(self._split)
        self._construct_vtdb()
    
    def _construct_vtdb(self):
        """Constructs the db."""
        self.data = read_json(os.path.join(self._data_path, self._split))
        logger.info("Number of pairs: {}".format(len(self._data)))

    def __getitem__(self, index):
        """Load data"""
        video_name, total_frames, captions = self.data[index]
        frames, frame_idxs = self._get_video(video_name)
        # preprocess frames
        frames = prepare_frames(frames, self.transform)
        captions = self._get_text(captions)
        return frames, captions
    
    def _get_video(self, name, total_frames):
        """Frames loader"""
        frames, frame_idxs = read_frames_from_folder(
            os.path.join(self.video_setting.VIDEO_PATH, name),
            total_frames,
            self.video_setting.NUM_FRAMES,
            self.video_setting.SAMPLE,
            self.video_setting.FIX_START
            )
        return frames, frame_idxs
    
    def _get_text(self, captions):
        """Text loader"""
        if self.text_setting.SAMPLE == 'random':
            sampled_captions = [random.choice(captions)]
        elif self.text_setting.SAMPLE == 'all':
            sampled_captions = captions
        elif self.text_setting.SAMPLE.startswith('mt-rand'):
            sampled_captions = random.sample(captions, 
                int(self.text_setting.SAMPLE.split('_')[1]))
        return sampled_captions
    
    def __len__(self):
        """_len"""
        return len(self.data)


def prepare_frames(frames, transform):
    """Preprocess wrapper."""
    return transform(frames.float().div_(255).permute(0, 3, 1, 2))


def read_frames_from_video(video_path, num_frames, sample='rand', fix_start=None):
    """Video decoder by decord."""
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = video_reader.get_batch(frame_idxs)
    return frames, frame_idxs


def read_frames_from_folder(video_path, total_frames, sample_num, sample='rand', fix_start=None):
    """Load frames from extracted folder."""
    frame_idxs = sample_frames(sample_num, total_frames, sample=sample, fix_start=fix_start)
    frames = []
    for idx in frame_idxs:
        frames.append(torch.from_numpy(np.asarray(
            Image.open(os.path.join(video_path, f'{idx}.jpg')))))
    frames = torch.stack(frames)
    return frames, frame_idxs


def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    """Sample frames from https://github.com/m-bain/frozen-in-time."""
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs
