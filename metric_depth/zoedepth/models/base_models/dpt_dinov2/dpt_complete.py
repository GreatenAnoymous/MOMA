import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import numpy as np
import torch.nn as nn

from .fusion import ConcatConv, AFF
from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F

from .dpt import DPTHead,_make_fusion_block, DPT_DINOv2

from .dpt_lora import _LoRA_qkv


class DCM_DINOv2(DPT_DINOv2):
    def __init__(self, dinov2:DPT_DINOv2, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, lora_layer=None):
        super(DCM_DINOv2, self).__init__()
        rank=256
        # fusion network
        for param in dinov2.pretrained.parameters():
            param.requires_grad = False

        self.lora_layer=list(range(len(dinov2.pretrained.blocks)))
        self.w_As=[]
        self.w_Bs=[]    
        for t_layer_i, blk in enumerate(dinov2.pretrained.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
            w_b_linear_v = nn.Linear(rank, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        self.pretrained=dinov2.pretrained
        self.depth_head=dinov2.depth_head
        self.fusionnet=nn.ModuleList([ConcatConv(in_channels=1) for i in range(4)]) 
        

    
    def forward(self, x, y):
        """_summary_

        Args:
            x (RGB): _description_
            y (raw depth): _description_

        Returns:
            _type_: _description_
        """
        h, w = x.shape[-2:]
        features_x = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        features_y = self.pretrained.get_intermediate_layers(y, 4, return_class_token=True)
        features=[]
        for i in range(4):

            fused_i = self.fusionnet[i](features_x[i][0], features_y[i][0])
            features.append([fused_i,1])
        patch_h, patch_w = h // 14, w // 14
        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
    
if __name__ == "__main__":
    pass