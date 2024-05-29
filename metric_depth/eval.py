
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
from PIL import Image
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align

from zoedepth.models.base_models.depth_anything_lora import DepthAnythingLoraCore
from zoedepth.models.base_models.depth_anything import DepthAnythingCore
import numpy as np
from scipy.optimize import curve_fit


def model_function(xy, xc, yc, d, alpha, beta, fc):
    x, y,z  = xy
   # assuming z is the third element of xy
    return np.cos(beta)*np.cos(alpha)* z -np.sin(beta) * (x - xc) * z*fc  + np.sin(alpha)*np.cos(beta) * (y - yc) * z *fc + d



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

def load_dam_model(DEVICE="cuda"):
    checkpoint="./depth_anything_finetune/mixed.pt"
    # checkpoint="./checkpoints/depth_anything_metric_depth_indoor.pt"
    config=get_config("zoedepth", "train", "nyu")
    depth_anything = build_model(config)
    depth_anything= load_wts(depth_anything, checkpoint)
    
    depth_anything = depth_anything.to(DEVICE)
    return depth_anything


def get_grids_sample(w, h, sample_interval):
    # Generate a grid of size (w, h) with num_samples samples
    x = np.linspace(0, w-1, sample_interval)
    y = np.linspace(0, h-1, sample_interval)
    grid = np.meshgrid(x, y)
    return grid

def get_random_samples(w, h, num_samples):
    # Generate random samples in the range of (w, h)
    samples = np.random.rand(num_samples, 2) * np.array([w, h])
    return samples



def eval_for_one_grid_with_samples(depth_anything, image, depth, object_mask, depth_raw):
    if object_mask is not None:
        object_mask=np.array(object_mask)
        object_mask=object_mask>0
    depth_numpy = depth_anything.infer_pil(image)

    depth_raw=np.array(depth_raw)
    depth_raw[np.isnan(depth_raw)] = 0
    mask = np.logical_and((depth_raw> 0),(depth_raw<1))
    # if object_mask is not None:
    #     mask=mask & object_mask[:,:,0]
    
            
    # Flatten the depth_numpy and depth arrays
    depth_numpy_flat = depth_numpy[mask].flatten()
    depth_flat = depth_raw[mask].flatten()

    # Fit a linear function to the data
    coefficients = np.polyfit(depth_numpy_flat, depth_flat, 1)

    # Extract the slope and intercept from the coefficients
    slope = coefficients[0]
    intercept = coefficients[1]

    # Apply the linear function to depth_numpy
    # print(slope,"slope", intercept, "intercept")
    scaled_fitted_depths = depth_numpy * slope + intercept
    
    # nonzero_indices = np.nonzero(depth_raw)
    interval=5
    nonzero_indices = get_grids_sample(depth_raw.shape[0], depth_raw.shape[1], interval)
    nonzero_indices = nonzero_indices[:, depth_raw[nonzero_indices] != 0]
    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    initial_guess = [0, 0, 0, 0, 0, 1]
    popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)


    xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt = popt

    fitted_depths=np.zeros_like(depth_raw)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]), xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
    
    # print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
    # print([xc_opt, yc_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt])
    # # Calculate absolute differences between original and fitted depth maps
    
    gt_mask= np.logical_and(depth>0, depth<1)
    # print(object_mask[:,:,0].min(), object_mask[:,:,0].max())

    # gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])

    metrics=compute_errors(depth[gt_mask], fitted_depths[gt_mask])
    absolute_diff = np.abs(depth[gt_mask] - fitted_depths[gt_mask])


    # # Calculate mean absolute error
    mean_absolute_error = np.mean(absolute_diff)

    metrics_linear=compute_errors(depth[gt_mask], scaled_fitted_depths[gt_mask])
    absolute_diff_linear = np.abs(depth[gt_mask] - scaled_fitted_depths[gt_mask])
    mean_absolute_error_linear = np.mean(absolute_diff_linear)

    metrics_zoedepth=compute_errors(depth[gt_mask], depth_numpy[gt_mask])
    absolute_diff_zoedepth = np.abs(depth[gt_mask] - depth_numpy[gt_mask])
    mean_absolute_error_zoedepth = np.mean(absolute_diff_zoedepth)
    
    return metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth, mean_absolute_error_zoedepth

def eval_for_one(depth_anything, image, depth, object_mask, depth_raw):
    if object_mask is not None:
        object_mask=np.array(object_mask)
        object_mask=object_mask>0
    depth_numpy = depth_anything.infer_pil(image)

    depth_raw=np.array(depth_raw)
    depth_raw[np.isnan(depth_raw)] = 0
    mask = np.logical_and((depth_raw> 0),(depth_raw<1))
    # if object_mask is not None:
    #     mask=mask & object_mask[:,:,0]
    
            
    # Flatten the depth_numpy and depth arrays
    depth_numpy_flat = depth_numpy[mask].flatten()
    depth_flat = depth_raw[mask].flatten()

    # Fit a linear function to the data
    coefficients = np.polyfit(depth_numpy_flat, depth_flat, 1)

    # Extract the slope and intercept from the coefficients
    slope = coefficients[0]
    intercept = coefficients[1]

    # Apply the linear function to depth_numpy
    # print(slope,"slope", intercept, "intercept")
    scaled_fitted_depths = depth_numpy * slope + intercept
    
    nonzero_indices = np.nonzero(depth_raw)
    nonzero_indices = np.nonzero(depth_raw)
    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    initial_guess = [0, 0, 0, 0, 0, 1]
    popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)


    xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt = popt

    fitted_depths=np.zeros_like(depth_raw)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]), xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
    
    # print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
    # print([xc_opt, yc_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt])
    # # Calculate absolute differences between original and fitted depth maps
    
    gt_mask= np.logical_and(depth>0, depth<1)
    # print(object_mask[:,:,0].min(), object_mask[:,:,0].max())

    # gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])

    metrics=compute_errors(depth[gt_mask], fitted_depths[gt_mask])
    absolute_diff = np.abs(depth[gt_mask] - fitted_depths[gt_mask])


    # # Calculate mean absolute error
    mean_absolute_error = np.mean(absolute_diff)

    metrics_linear=compute_errors(depth[gt_mask], scaled_fitted_depths[gt_mask])
    absolute_diff_linear = np.abs(depth[gt_mask] - scaled_fitted_depths[gt_mask])
    mean_absolute_error_linear = np.mean(absolute_diff_linear)

    metrics_zoedepth=compute_errors(depth[gt_mask], depth_numpy[gt_mask])
    absolute_diff_zoedepth = np.abs(depth[gt_mask] - depth_numpy[gt_mask])
    mean_absolute_error_zoedepth = np.mean(absolute_diff_zoedepth)
    
    return metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth, mean_absolute_error_zoedepth


def eval_cleargrasp():
    model=load_dam_model()
    metrics1={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics2={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics3={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    count=0
    for i in range(1,90):
        try:
            print(i)
            depth=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
            depth_raw=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
            image = cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-transparent-rgb-img.jpg")
            mask_image=cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-mask.png")
            metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth ,mean_absolute_error_zoedepth=eval_for_one(model, image, depth, mask_image, depth_raw)
            metrics1["a1"]+=metrics["a4"]
            metrics1["a2"]+=metrics["a5"]
            metrics1["a3"]+=metrics["a1"]
            metrics1["abs_rel"]+=metrics["abs_rel"]
            metrics1["rmse"]+=metrics["rmse"]
            metrics1["mae"]+=mean_absolute_error
            metrics2["a1"]+=metrics_linear["a4"]
            metrics2["a2"]+=metrics_linear["a5"]
            metrics2["a3"]+=metrics_linear["a1"]
            metrics2["abs_rel"]+=metrics_linear["abs_rel"]
            metrics2["rmse"]+=metrics_linear["rmse"]
            metrics2["mae"]+=mean_absolute_error_linear
            metrics3["a1"]+=metrics_zoedepth["a4"]
            metrics3["a2"]+=metrics_zoedepth["a5"]
            metrics3["a3"]+=metrics_zoedepth["a1"]
            metrics3["abs_rel"]+=metrics_zoedepth["abs_rel"]
            metrics3["rmse"]+=metrics_zoedepth["rmse"]
            metrics3["mae"]+=mean_absolute_error_zoedepth
            count+=1
        except Exception as e:
            print(e)
    metrics2["a1"]/=count
    metrics2["a2"]/=count
    metrics2["a3"]/=count
    metrics2["abs_rel"]/=count
    metrics2["rmse"]/=count
    metrics2["mae"]/=count
    metrics1["a1"]/=count
    metrics1["a2"]/=count
    metrics1["a3"]/=count
    metrics1["abs_rel"]/=count
    metrics1["rmse"]/=count
    metrics1["mae"]/=count
    metrics3["a1"]/=count
    metrics3["a2"]/=count
    metrics3["a3"]/=count
    metrics3["abs_rel"]/=count
    metrics3["rmse"]/=count
    metrics3["mae"]/=count

    print(metrics1)
    print(metrics2)
    print(metrics3)

eval_cleargrasp()