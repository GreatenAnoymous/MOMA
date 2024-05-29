# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from zoedepth.utils.config import change_dataset
from .ddad import get_ddad_loader
from .diml_indoor_test import get_diml_indoor_loader
from .diml_outdoor_test import get_diml_outdoor_loader
from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .ibims import get_ibims_loader
from .sun_rgbd_loader import get_sunrgbd_loader
from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader
from .dataloadpreprocess import DataLoadPreprocess
from .preprocess import CropParams, get_white_border, get_black_border
from .cleargrasp import get_cleargrasp_loader,ClearGraspSynthetic
from .omniverse import get_omniverse_loader,OmniverseObject
from .depth_complete import *
from .data_preparation import ToTensor
from .clearpose import ClearPoseDataset


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """
        self.config = config
        print("num workers", config.workers)
        if config.dataset=='depth_complete':
            if mode == 'train':
                pin_memory = True
                num_workers = config.workers
                batch_size = config.batch_size
            else:
                pin_memory = False
                num_workers=1
                batch_size=1
            self.data = DataLoader(DepthCompleteData(config, mode, transform=transform), batch_size=batch_size,  num_workers=num_workers, pin_memory=pin_memory)
            return
        if config.dataset == 'clearpose':
            if mode=="train":
                self.data = DataLoader(ClearPoseDataset(config, 'train', device=device), batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)
            else:
                self.data = DataLoader(ClearPoseDataset(config, 'online_eval', device=device), 1, num_workers=1, pin_memory=False)
            return
        
        if config.dataset == 'arcl':
            if mode=="train":
                self.data = DataLoader(ClearPoseDataset(config, "train", device=device), batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)
            else:
                self.data = DataLoader(ClearPoseDataset(config, 'online_eval', device=device), 1, num_workers=1, pin_memory=False)  
            return
        
        if config.dataset == 'transcg':
            if mode=="train":
                self.data = DataLoader(ClearPoseDataset(config, "train", device=device), batch_size=config.batch_size, num_workers=config.workers, pin_memory=False)
            else:
                self.data = DataLoader(ClearPoseDataset(config, 'online_eval', device=device), 1, num_workers=1, pin_memory=False)  
            return
        if config.dataset=='cleargrasp':
            self.data= get_cleargrasp_loader(config, mode, batch_size=1, num_workers=1)
            return 
        if config.dataset=="omniverse":
            self.data= get_omniverse_loader(config, mode, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':
            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=False,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            print("Using online eval sampler")
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


class TransMixDataloader(object):
    def __init__(self, config, mode, device="cpu", **kwargs ) -> None:
        config=edict(config)
        config.workers=config.workers
        self.config=config
        cleargrasp_conf=change_dataset(edict(config), 'cleargrasp')
        omniverse_conf=change_dataset(edict(config), 'omniverse')
        transcg_conf=change_dataset(edict(config), 'transcg')
        clearpose_conf=change_dataset(edict(config), 'clearpose')
        img_size = self.config.get("img_size", None)
        if mode=="train":
            # cleargrasp_loader=DepthDataLoader(cleargrasp_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # omniverse_loader=DepthDataLoader(omniverse_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # transcg_loader=DepthDataLoader(transcg_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # clearpose_loader=DepthDataLoader(clearpose_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data   
            # self.data=RepetitiveRoundRobinDataLoader(  transcg_loader, clearpose_loader,cleargrasp_loader,omniverse_loader)
            cleargrasp_data=ClearGraspSynthetic(cleargrasp_conf.data_root, split='train',**cleargrasp_conf)
            omniverse_data=OmniverseObject(omniverse_conf.data_root, split='train',**omniverse_conf)
            transcg_data=ClearPoseDataset(transcg_conf, 'train', device=device)
            clearpose_data=ClearPoseDataset(clearpose_conf, 'train', device=device)
            self.data=DataLoader(ConcatDataset([cleargrasp_data, omniverse_data, transcg_data, clearpose_data]), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=False)
        else:
            self.data=DepthDataLoader(transcg_conf, mode, device=device).data



class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data




