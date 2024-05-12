
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

from zoedepth.models.base_models.depth_anything_complete import DepthCompleteCore
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align


import numpy as np
from scipy.optimize import curve_fit




model=DepthCompleteCore.build(img_size=[518,518])
depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
depth_raw=depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-transparent-rgb-img.jpg")

mask_image=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-mask.png")
depth_raw=np.array(depth_raw)

depth=np.array(depth)
depth[np.isnan(depth)]=0
image=np.array(image,dtype=np.float32)/255.0
image = np.transpose(image, (2, 0, 1))
image=torch.tensor(image)
depth=torch.tensor(depth)
depth = depth.repeat(3, 1, 1)
print(image.shape, depth.shape)
image=image.unsqueeze(0)
depth=depth.unsqueeze(0)
output=model(image, depth, denorm=False, return_rel_depth=True)

