
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from exr_utils import exr_loader
from zoedepth.utils.config import get_config
from importlib import import_module
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
from zoedepth.models.model_io import load_wts
import create_pc

import matplotlib.pyplot as plt
from zoedepth.data.data_preparation import process_depth
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align


import numpy as np
from scipy.optimize import curve_fit


def model_function(xy, xc, yc, d, alpha, beta, fc):
    x, y,z  = xy
   # assuming z is the third element of xy
    return np.cos(beta)*np.cos(alpha)* z -np.sin(beta) * (x - xc) * z*fc  + np.sin(alpha)*np.cos(beta) * (y - yc) * z *fc + d

class CameraIntrinsic:
    def __init__(self,fx=900,fy=900,ppx=321.8606872558594,ppy=239.07879638671875) -> None:
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy


def build_model(config) -> DepthModel:
    """Builds a model from a config. The model is specified by the model name and version in the config. The model is then constructed using the build_from_config function of the model interface.
    This function should be used to construct models for training and evaluation.

    Args:
        config (dict): Config dict. Config is constructed in utils/config.py. Each model has its own config file(s) saved in its root model folder.

    Returns:
        torch.nn.Module: Model corresponding to name and version as specified in config
    """
    module_name = f"zoedepth.models.{config.model}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as e:
        # print the original error message
        print(e)
        raise ValueError(
            f"Model {config.model} not found. Refer above error for details.") from e
    try:
        get_version = getattr(module, "get_version")
    except AttributeError as e:
        raise ValueError(
            f"Model {config.model} has no get_version function.") from e
    return get_version(config.version_name).build_from_config(config)




def load_ckpt(config, model, checkpoint_dir="./checkpoints", ckpt_type="best"):
    import glob
    import os

    from zoedepth.models.model_io import load_wts

    if hasattr(config, "checkpoint"):
        checkpoint = config.checkpoint
    elif hasattr(config, "ckpt_pattern"):
        pattern = config.ckpt_pattern
        matches = glob.glob(os.path.join(
            checkpoint_dir, f"*{pattern}*{ckpt_type}*"))
        if not (len(matches) > 0):
            raise ValueError(f"No matches found for the pattern {pattern}")

        checkpoint = matches[0]

    else:
        return model
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model


class DAM(object):
    def __init__(self) -> None:
        pass
    
    def predictDepth(self, image, depth_raw, depth_gt, object_mask=None , DEVICE="cuda:4", scale=1):
        if object_mask is not None:
            object_mask=np.array(object_mask)
            object_mask=object_mask>0
        checkpoint="./depth_anything_finetune/DAMC.pt"
        # checkpoint="./checkpoints/depth_anything_metric_depth_indoor.pt"
        config=get_config("damc", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)
        
        depth_anything = depth_anything.to(DEVICE)

        from PIL import Image
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)/scale
        depth_raw = np.asarray(depth_raw, dtype=np.float32)/scale
        plt.imshow(depth_raw, cmap='jet', vmin=0.3, vmax=1)
        plt.colorbar()
        plt.savefig('raw_depth.png')
        plt.close()
        depth_mu=depth_raw[depth_raw>0].mean()
        depth_raw[depth_raw>1]=1
        depth_std=depth_raw[depth_raw>0].std()
        depth_min=depth_raw[depth_raw>0].min()-0.5*depth_std-1e-6
        depth_max=depth_raw[depth_raw>0].max()+0.5*depth_std+1e-6
        print("depth min", depth_raw[depth_raw>0].min(), "depth max", depth_raw[depth_raw>0].max(), "depth std",depth_std)
        depth_raw = process_depth(depth_raw)
        plt.imshow(depth_raw, cmap='jet', vmin=0.3, vmax=1)
        plt.colorbar()
        plt.savefig('raw_depth_processed.png')
        plt.close()
        rgb = torch.FloatTensor(image).to(device=DEVICE).unsqueeze(0)
        depth = torch.FloatTensor(depth_raw).to(device=DEVICE).unsqueeze(0)
        rgb=rgb.permute(0,3,1,2)

        with torch.no_grad():
            depth_res=depth_anything(rgb, depth)
        depth_res=depth_res["metric_depth"]
        depth_res=depth_res.squeeze().cpu().numpy()
        depth_res = depth_res * (depth_max - depth_min) + depth_min
        mask = np.logical_and((depth_gt> 0),(depth_gt<2))
        if object_mask is not None:
            mask=mask & object_mask[:,:,0]
        depth_res = cv2.resize(depth_res, (depth_gt.shape[1], depth_gt.shape[0]))
        # Visualize the fitted depth
        plt.imshow(depth_res, cmap='jet', vmin=0.3, vmax=1)
        plt.colorbar()
        plt.savefig('completed_depth.png')
        plt.close()
        
        plt.imshow(image)
        plt.savefig('rgb.png')
        plt.close()
        
        plt.imshow(depth_gt, cmap='jet', vmin=0.3, vmax=1)
        plt.colorbar()
        plt.savefig("gt_depth.png")
        
        plt.close()







from PIL import Image

    
dam =DAM()
# depth_gt=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000069-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# depth_raw=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000069-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000069-transparent-rgb-img.jpg")
# mask_image=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000069-mask.png")


image=cv2.imread("./data/nyu/transcg/scene1/217/rgb2.png")
depth_raw =np.array(Image.open("./data/nyu/transcg/scene1/217/depth2.png"))
depth_gt =np.array(Image.open("./data/nyu/transcg/scene1/217/depth2-gt.png"))
mask_image=cv2.imread("./data/nyu/transcg/scene1/217/depth2-gt-mask.png")


# image=cv2.imread("./data/nyu/arcl/001/0_color.png")
# depth_raw =np.array(Image.open("./data/nyu/arcl/001/0_raw_depth.png"))
# depth_gt =np.array(Image.open("./data/nyu/arcl/001/0_gt_depth.png"))
# mask_image=cv2.imread("./data/nyu/arcl/001/0_color.png")
scale=4000

dam.predictDepth(image, depth_raw, depth_gt, object_mask=mask_image, scale=scale)
