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

from .blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F

from .dpt import DPTHead,_make_fusion_block



class _LoRA_qkv(nn.Module):
    """In Dinov2 it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    


class DPT_DINOv2_Lora(nn.Module):
    def __init__(self, dinov2=None, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False, lora_layer=None):
        rank=1024
        print("LORA rank=",rank)
        super(DPT_DINOv2_Lora, self).__init__()

        torch.manual_seed(1)

        for param in dinov2.pretrained.parameters():
            param.requires_grad = False
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer=list(range(len(dinov2.pretrained.blocks)))
        self.w_As=[]
        self.w_Bs=[]
        
        
        dim = dinov2.pretrained.blocks[0].attn.qkv.in_features
    
        
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
        
        self.depth_head = dinov2.depth_head
        self.reset_parameters()
    

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """
        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)
            
    
    def forward(self, x):

        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
    
    def load_lora_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        # assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        print('loaded lora parameters from %s.' % filename)


if __name__ == "__main__":
    pass