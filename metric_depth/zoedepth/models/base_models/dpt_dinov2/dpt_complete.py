import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import numpy as np
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from .fusion import AFF
from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F

from .dpt import DPTHead,_make_fusion_block
from .dpt_lora import _LoRA_qkv, DPT_DINOv2_Lora


    


class DCM_DINOv2_Lora(DPT_DINOv2_Lora):
    def __init__(self, dinov2=None, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, lora_layer=None):
        rank=1024
        print("LORA rank=",rank)
        super(DCM_DINOv2_Lora, self).__init__()
        self.fusionnet=AFF()
        self.depth_head = dinov2.depth_head
        print(sum(p.numel() for p in dinov2.pretrained.parameters() if p.requires_grad == True)/1e6, sum(p.numel() for p in dinov2.depth_head.parameters())/1e6, "number of parameters")

    
    def forward(self, x, y):
        h, w = x.shape[-2:]
        

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        features_y = self.pretrained.get_intermediate_layers(y, 4, return_class_token=True)
        features = self.fusionnet(features, features_y)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
    
if __name__ == "__main__":
    pass