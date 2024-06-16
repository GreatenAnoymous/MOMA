
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
import json
import pandas as pd
from scipy import stats

def model_function(xy, xc, yc, d, alpha, beta, fc):
    x, y,z  = xy
   # assuming z is the third element of xy
    return np.cos(beta)*np.cos(alpha)* z -np.sin(beta) * (x - xc) * z*fc  + np.sin(alpha)*np.cos(beta) * (y - yc) * z *fc + d


def compute_local_alignment(u, v, depths, depths_gt, M, N, b=100, alpha=1):
    # Compute distance matrix
    ui, vi = np.indices((M, N))

    dist = (u[:, None, None] - ui)**2 + (v[:, None, None] - vi)**2

    # Compute weight matrix
    w =  np.exp(-alpha * dist / (2 * b**2))

    W = np.zeros([len(u), len(u),M,N])
    W[:, :, ui, vi] = 1
    idx= np.arange(len(u))
    W[idx, idx, :, :] = dist
    # print(W[0][0], u.shape, v.shape, dist.shape, W.shape)
    W=np.transpose(W, (2,3,0,1))

    # Construct X matrix
    X = np.vstack((depths, np.ones_like(depths)))

    
    X = X.T

    # Compute intermediate matrices

    XTWX  =np.einsum('mk,ijkl,ln->ijmn',X.T, W,  X)


    XTWX =np.linalg.pinv(XTWX)
    
    depths_gt = depths_gt.reshape(-1, 1)
    XTWy=np.einsum('mk,ijkl,ln->ijmn',  X.T,W, depths_gt)

    # Compute beta using linear algebra
    beta = np.einsum('ijkl,ijlm->ijkm', XTWX, XTWy)

    # Reshape beta to match the required output shape
    # print(beta)
    return beta

def normalize_depth_robust(target, mask):
    medium=np.median(target[mask])
    target=target-medium
    s=np.mean(np.abs(target[mask]-medium))
    return target/s


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



def eval_for_one_with_random_samples(depth_anything, image, depth, object_mask, depth_raw, N):
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
    
    array1, array2 = nonzero_indices
    
    # Ensure both arrays are of the same size
    assert len(array1) == len(array2), "Arrays in the tuple must be of the same size"
    
    # Total number of available indices
    total_indices = len(array1)
    
    # Randomly sample N indices from the available indices
    sampled_indices = np.random.choice(total_indices, N, replace=False)
    
    
    # Use the sampled indices to create new arrays
    sampled_array1 = array1[sampled_indices]
    sampled_array2 = array2[sampled_indices]

    assert len(sampled_array1)==N, "Sampled array1 must have N elements"

    nonzero_indices = (sampled_array1, sampled_array2)

    # print(len(nonzero_indices),nonzero_indices)
    
    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    initial_guess=np.random.rand(6)
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

    gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])

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
    depth_zoe = depth_anything.infer_pil(image)
    depth_numpy = depth_zoe
    
    depth_raw=np.array(depth_raw)
    depth_raw[np.isnan(depth_raw)] = 0
    mask = np.logical_and((depth_raw> 0),(depth_raw<2))

    
            
    # Flatten the depth_numpy and depth arrays
    depth_numpy_flat = depth_numpy[mask].flatten()
    depth_flat = depth_raw[mask].flatten()

    # Fit a linear function to the data
    # coefficients = np.polyfit(depth_numpy_flat, depth_flat, 1)

    # # Extract the slope and intercept from the coefficients
    # slope = coefficients[0]
    # intercept = coefficients[1]
    slope, intercept, r_value, p_value, std_err = stats.linregress(depth_numpy_flat, depth_flat)

    # Apply the linear function to depth_numpy
    # print(slope,"slope", intercept, "intercept")
    scaled_fitted_depths = depth_numpy * slope + intercept
    

    nonzero_indices = np.nonzero(depth_raw)
    x_data = nonzero_indices[0]
    y_data = nonzero_indices[1]
    z_data = depth_numpy[nonzero_indices]
    zt_data = depth_raw[nonzero_indices]

    initial_guess=np.random.rand(6)
    popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)

    # xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt =[774.6724988288931, -893.5787484362147, 0.5205030005998124, 1.0907786593742994, 2.172754788378457, -0.0022836359312404853]
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

    gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])

    metrics=compute_errors(depth[gt_mask], fitted_depths[gt_mask])
    absolute_diff = np.abs(depth[gt_mask] - fitted_depths[gt_mask])



    # # Calculate mean absolute error
    mean_absolute_error = np.mean(absolute_diff)

    metrics_linear=compute_errors(depth[gt_mask], scaled_fitted_depths[gt_mask])
    absolute_diff_linear = np.abs(depth[gt_mask] - scaled_fitted_depths[gt_mask])
    mean_absolute_error_linear = np.mean(absolute_diff_linear)

    metrics_zoedepth=compute_errors(depth[gt_mask], depth_zoe[gt_mask])
    absolute_diff_zoedepth = np.abs(depth[gt_mask] - depth_zoe[gt_mask])
    mean_absolute_error_zoedepth = np.mean(absolute_diff_zoedepth)

    w,h=depth.shape[0], depth.shape[1]
    depth_gt_1d=depth.flatten()
    depth_1d=depth_numpy.flatten()
    valid=depth_gt_1d>0
    depth_gt_1d=depth_gt_1d[valid]
    depth_1d=depth_1d[valid]
    u,v=np.meshgrid(np.arange(h), np.arange(w))
    u=u.flatten()
    v=v.flatten()
    u=u[valid]
    v=v[valid]  
    indices = np.random.choice(len(depth_1d), size=25, replace=False)
    u_selected = u[indices]
    v_selected = v[indices]
    depth_gt_1d_selected = depth_gt_1d[indices]
    depth_1d_selected = depth_1d[indices]
    beta=compute_local_alignment(u_selected, v_selected, depth_1d_selected, depth_gt_1d_selected, w, h)
    depth_local_aligned = beta[..., 0, 0] * depth_numpy + beta[..., 1, 0]
    metrics_local_aligned=compute_errors(depth[gt_mask], depth_local_aligned[gt_mask])
    absolute_diff_local_aligned = np.abs(depth[gt_mask] - depth_local_aligned[gt_mask])
    mean_absolute_error_local_aligned = np.mean(absolute_diff_local_aligned)
    return metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth, mean_absolute_error_zoedepth, metrics_local_aligned, mean_absolute_error_local_aligned


def eval_cleargrasp():
    model=load_dam_model()
    metrics1={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics2={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics3={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics4={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    count=0
    for i in range(1,90):
        try:
            print(i)
            depth=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
            depth_raw=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
            image = cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-transparent-rgb-img.jpg")
            mask_image=cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{i:02}-mask.png")
            metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth ,mean_absolute_error_zoedepth, metrics_local, mean_absolute_error_local=eval_for_one(model, image, depth, mask_image, depth_raw)
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
            metrics4["a1"]+=metrics_local["a4"]
            metrics4["a2"]+=metrics_local["a5"]
            metrics4["a3"]+=metrics_local["a1"]
            metrics4["abs_rel"]+=metrics_local["abs_rel"]
            metrics4["rmse"]+=metrics_local["rmse"]
            metrics4["mae"]+=mean_absolute_error_local
            
            count+=1
            save_tmp(metrics1, metrics2, metrics3, metrics4, count, filename="cleargrasp.json")
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
    metrics4["a1"]/=count
    metrics4["a2"]/=count
    metrics4["a3"]/=count
    metrics4["abs_rel"]/=count
    metrics4["rmse"]/=count
    metrics4["mae"]/=count


    print(metrics1)
    print(metrics2)
    print(metrics3)
    print(metrics4)
    

def save_tmp(metrics1, metrics2, metrics3, metrics4 ,count, filename="tmp.json"):
    tmp_metrics1=metrics1.copy()
    tmp_metrics2=metrics2.copy()
    tmp_metrics3=metrics3.copy()
    tmp_metrics4=metrics4.copy()
    tmp_metrics1["a1"]/=count
    tmp_metrics1["a2"]/=count
    tmp_metrics1["a3"]/=count
    tmp_metrics1["abs_rel"]/=count
    tmp_metrics1["rmse"]/=count
    tmp_metrics1["mae"]/=count
    tmp_metrics2["a1"]/=count
    tmp_metrics2["a2"]/=count
    tmp_metrics2["a3"]/=count
    tmp_metrics2["abs_rel"]/=count
    tmp_metrics2["rmse"]/=count
    tmp_metrics2["mae"]/=count
    tmp_metrics3["a1"]/=count
    tmp_metrics3["a2"]/=count
    tmp_metrics3["a3"]/=count
    tmp_metrics3["abs_rel"]/=count
    tmp_metrics3["rmse"]/=count
    tmp_metrics3["mae"]/=count
    tmp_metrics4["a1"]/=count
    tmp_metrics4["a2"]/=count
    tmp_metrics4["a3"]/=count
    tmp_metrics4["abs_rel"]/=count
    tmp_metrics4["rmse"]/=count
    tmp_metrics4["mae"]/=count

    
    with open(filename, "w") as f:
        json.dump({"metrics1":tmp_metrics1, "metrics2":tmp_metrics2, "metrics3": tmp_metrics3, "metrics4": tmp_metrics4}, f)


def eval_transcg():
    model=load_dam_model()
    metrics1={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics2={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics3={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics4={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}

    count=0
    txt_file="./train_test_inputs/transcg_test.txt"
    dataset_path="./data/nyu/"
    with open(txt_file, "r") as f:
        for line in f.readlines():
    
            rgb_path, depth_gt_path,focal=line.split(" ")
        
            image=cv2.imread(dataset_path+rgb_path)
            
            if "depth1" in depth_gt_path:
                depth=np.array(Image.open(dataset_path+depth_gt_path))
                depth_raw_path=depth_gt_path.replace("depth1-gt", "depth1")
                depth_raw=np.array(Image.open(dataset_path+depth_raw_path))
                depth=depth/1000
                depth_raw=depth_raw/1000
                mask_path=depth_gt_path.replace("depth1-gt", "depth1-gt-mask")
                mask_image=cv2.imread(dataset_path+mask_path)

            else:
                # continue
                depth=np.array(Image.open(dataset_path+depth_gt_path))
                depth_raw_path=depth_gt_path.replace("depth2-gt", "depth2")
                depth=depth/1000
                depth_raw=np.array(Image.open(dataset_path+depth_raw_path))
                depth_raw=depth_raw/1000
                mask_path=depth_gt_path.replace("depth2-gt-converted", "depth2-gt-mask")
                mask_image=cv2.imread(dataset_path+mask_path)
            print(depth_raw_path, depth_gt_path, mask_path)
            try:
                metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth ,mean_absolute_error_zoedepth, metrics_local, mean_absolute_error_local =eval_for_one(model, image, depth, mask_image, depth_raw)
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
                metrics4["a1"]+=metrics_local["a4"]
                metrics4["a2"]+=metrics_local["a5"]
                metrics4["a3"]+=metrics_local["a1"]
                metrics4["abs_rel"]+=metrics_local["abs_rel"]
                metrics4["rmse"]+=metrics_local["rmse"]
                metrics4["mae"]+=mean_absolute_error_local

                count+=1
                save_tmp(metrics1, metrics2, metrics3, metrics4, count, filename="transcg.json")
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
    metrics4["a1"]/=count
    metrics4["a2"]/=count
    metrics4["a3"]/=count
    metrics4["abs_rel"]/=count
    metrics4["rmse"]/=count
    metrics4["mae"]/=count


    print(metrics1)
    print(metrics2)
    print(metrics3)
    print(metrics4)



def eval_arcl(data_id=5):
    model=load_dam_model()
    metrics1={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics2={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics3={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    metrics4={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
    count=0
    
    for i in range(0,100):
        try:
            print(i)
            image=cv2.imread(f"./data/nyu/test/00{data_id}/{i}_opaque_color.png")
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            depth =np.array(Image.open(f"./data/nyu/test/00{data_id}/{i}_gt_depth.png"))/1000.
            depth_raw=np.array(Image.open(f"./data/nyu/test/00{data_id}/{i}_gt_depth.png"))/1000.
            mask_image=cv2.imread(f"./data/nyu/test/00{data_id}/{i}_mask.png")
           
            metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth ,mean_absolute_error_zoedepth, metrics_local, mean_absolute_error_local=eval_for_one(model, image, depth, mask_image, depth_raw)
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
            metrics4["a1"]+=metrics_local["a4"]
            metrics4["a2"]+=metrics_local["a5"]
            metrics4["a3"]+=metrics_local["a1"]
            metrics4["abs_rel"]+=metrics_local["abs_rel"]
            metrics4["rmse"]+=metrics_local["rmse"]
            metrics4["mae"]+=mean_absolute_error_local

            count+=1
        

            save_tmp(metrics1, metrics2, metrics3, metrics4, count, f"./arcl_opaque_{data_id}.json")
            
        except Exception as e:
            pass
            # raise e
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
    metrics4["a1"]/=count
    metrics4["a2"]/=count
    metrics4["a3"]/=count
    metrics4["abs_rel"]/=count
    metrics4["rmse"]/=count
    metrics4["mae"]/=count


    print(metrics1)
    print(metrics2)
    print(metrics3)
    print(metrics4)
    

def eval_samples():
    model=load_dam_model()
    
    data_id=3
    id=72
    df=pd.DataFrame(columns=["N","a1", "a2", "a3", "abs_rel", "rmse", "mae"])
    samples=np.arange(50, 3100, 100)
    # image=cv2.imread(f"./data/nyu/test/00{data_id}/{id}_opaque_color.png")
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # depth =np.array(Image.open(f"./data/nyu/test/00{data_id}/{id}_gt_depth.png"))/1000.
    # depth_raw=np.array(Image.open(f"./data/nyu/test/00{data_id}/{id}_gt_depth.png"))/1000.
    # mask_image=cv2.imread(f"./data/nyu/test/00{data_id}/{id}_mask.png")

    depth=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{data_id:02}-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
    depth_raw=exr_loader(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{data_id:02}-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
    image = cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{data_id:02}-transparent-rgb-img.jpg")
    mask_image=cv2.imread(f"../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/0000000{data_id:02}-mask.png")
    for N in samples:
    

        count=0
        metrics1={"a1":0, "a2":0, "a3":0, "abs_rel":0, "rmse":0, "mae":0}
        
        for k in range(10):
            try:
                metrics, mean_absolute_error, metrics_linear, mean_absolute_error_linear, metrics_zoedepth ,mean_absolute_error_zoedepth=eval_for_one_with_random_samples(model, image, depth, mask_image, depth_raw, N)
                count+=1
                metrics1["a1"]+=metrics["a4"]
                metrics1["a2"]+=metrics["a5"]
                metrics1["a3"]+=metrics["a1"]
                metrics1["abs_rel"]+=metrics["abs_rel"]
                metrics1["rmse"]+=metrics["rmse"]
                metrics1["mae"]+=mean_absolute_error
            except Exception as e:
                pass
        
        metrics1["a1"]/=count
        metrics1["a2"]/=count
        metrics1["a3"]/=count
        metrics1["abs_rel"]/=count
        metrics1["rmse"]/=count
        metrics1["mae"]/=count
        df.loc[len(df)]=[N, metrics1["a1"], metrics1["a2"], metrics1["a3"], metrics1["abs_rel"], metrics1["rmse"], metrics1["mae"]]
    
        df.to_csv('sample_eval.csv', index=False)
# eval_cleargrasp()
eval_transcg()
# eval_arcl(data_id=5)
# eval_arcl(data_id=6)
# eval_arcl(data_id=7)
# eval_arcl(data_id=8)
# eval_samples()