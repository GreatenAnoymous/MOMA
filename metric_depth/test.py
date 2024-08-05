
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


from zoedepth.models.base_models.depth_anything_complete import DepthCompleteCore
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align

import h5py
import cv2
import numpy as np
from scipy.optimize import curve_fit

h5path="/mnt/ssd_990/teng/BinPicking/DPT_transparent_objects/metric_depth/data/nyu/omniverse_v3/train/20200910/output.1600130896.7054846.h5"
f = h5py.File(h5path, 'r')
rgb = cv2.cvtColor(f['rgb_glass'][:], cv2.COLOR_RGB2BGR)
disparity = f['depth'][:]
depth_gt = 1. / (disparity + 1e-8) * 0.01
depth_gt = np.clip(depth_gt, 0, 10)
import matplotlib.pyplot as plt

# Plot RGB image
plt.subplot(1, 2, 1)
plt.imshow(rgb)
plt.title('RGB Image')

# Plot Depth image
plt.subplot(1, 2, 2)
plt.imshow(depth_gt, cmap='jet')
plt.title('Depth Image')
plt.savefig("show.png")
# plt.show()

# model=DepthCompleteCore.build(img_size=[518,518])
# depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# depth_raw=depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-transparent-rgb-img.jpg")

# mask_image=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000089-mask.png")
# depth_raw=np.array(depth_raw)

# depth=np.array(depth)
# depth[np.isnan(depth)]=0
# image=np.array(image,dtype=np.float32)/255.0
# image = np.transpose(image, (2, 0, 1))
# image=torch.tensor(image)
# depth=torch.tensor(depth)
# depth = depth.repeat(3, 1, 1)
# print(image.shape, depth.shape)
# image=image.unsqueeze(0)
# depth=depth.unsqueeze(0)
# output=model(image, depth, denorm=False, return_rel_depth=True)

