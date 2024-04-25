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

import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import Normalize
from zoedepth.models.base_models.dpt_dinov2.dpt import DPT_DINOv2
from zoedepth.models.base_models.depth_anything import *
from zoedepth.models.base_models.dpt_dinov2.dpt_lora import DPT_DINOv2_Lora
from .depth_anything import DepthAnythingCore

class DepthAnythingLoraCore(DepthAnythingCore):
    def __init__(self, midas, trainable=False, fetch_features=True, layer_names=('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1'), freeze_bn=False, keep_aspect_ratio=True,
                 img_size=384, **kwargs):
        """Midas Base model used for multi-scale feature extraction.

        Args:
            midas (torch.nn.Module): Midas model.
            trainable (bool, optional): Train midas model. Defaults to False.
            fetch_features (bool, optional): Extract multi-scale features. Defaults to True.
            layer_names (tuple, optional): Layers used for feature extraction. Order = (head output features, last layer features, ...decoder features). Defaults to ('out_conv', 'l4_rn', 'r4', 'r3', 'r2', 'r1').
            freeze_bn (bool, optional): Freeze BatchNorm. Generally results in better finetuning performance. Defaults to False.
            keep_aspect_ratio (bool, optional): Keep the aspect ratio of input images while resizing. Defaults to True.
            img_size (int, tuple, optional): Input resolution. Defaults to 384.
        """
        super().__init__(midas, trainable, fetch_features, layer_names, freeze_bn, keep_aspect_ratio, img_size, **kwargs)

    @staticmethod
    def build(midas_model_type="dinov2_large", train_midas=False, use_pretrained_midas=True, fetch_features=False, freeze_bn=True, force_keep_ar=False, force_reload=False, **kwargs):
        if "img_size" in kwargs:
            kwargs = DepthAnythingCore.parse_img_size(kwargs)
        img_size = kwargs.pop("img_size", [384, 384])
        
        depth_anything = DPT_DINOv2(out_channels=[256, 512, 1024, 1024], use_clstoken=False)
        
        state_dict = torch.load('./checkpoints/depth_anything_vitl14.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict)
        
        kwargs.update({'keep_aspect_ratio': force_keep_ar})
        
        depth_anything_lora=DPT_DINOv2_Lora(dinov2=depth_anything, out_channels=[256, 512, 1024, 1024], use_clstoken=False)        
        depth_anything_core = DepthAnythingLoraCore(depth_anything_lora, trainable=train_midas, fetch_features=fetch_features,
                               freeze_bn=freeze_bn, img_size=img_size, **kwargs)

        depth_anything_core.set_output_channels()
        print("Using Lora for fine-tuning Depth-Anything model.")
        return depth_anything_core


nchannels2models = {
    tuple([256]*5): ["DPT_BEiT_L_384", "DPT_BEiT_L_512", "DPT_BEiT_B_384", "DPT_SwinV2_L_384", "DPT_SwinV2_B_384", "DPT_SwinV2_T_256", "DPT_Large", "DPT_Hybrid"],
    (512, 256, 128, 64, 64): ["MiDaS_small"]
}

# Model name to number of output channels
MIDAS_SETTINGS = {m: k for k, v in nchannels2models.items()
                  for m in v
                  }
