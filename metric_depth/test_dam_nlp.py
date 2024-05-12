
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

from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import curve_fit
import nlopt

class CameraIntrinsic:
    def __init__(self,fx=900,fy=900,ppx=321.8606872558594,ppy=239.07879638671875) -> None:
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy



def model_function(xy, xc, yc, theta1, theta2, d, lambda1, lambda2, lambda3):
    x, y,z  = xy
   # assuming z is the third element of xy
    return lambda3* z + lambda1 * (x - xc) * np.cos(theta1) + lambda2 * (y - yc) * np.cos(theta2) + d



def rotation_matrix(roll, pitch, yaw):
    """
    Compute rotation matrix from roll, pitch, and yaw angles.
    """
    # Convert angles to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Compute rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    return np.dot(R_z, np.dot(R_y, R_x))

def forward_transform(params, u, v, zc):
    """
    Compute forward transform X = R * Xc + T.
    """
    cxc, cyc, fc, roll, pitch, yaw, dx, dy, dz = params
    R = rotation_matrix(roll, pitch, yaw)
    xc = zc * (u - cxc) / fc
    yc = zc * (v - cyc) / fc
    Xc = np.array([xc, yc, zc])
    T = np.array([dx, dy, dz])
    X = np.dot(R, Xc) + T
    return X
import numpy as np

def objective_function(params, inputs, outputs):
    """
    Objective function to minimize the difference between expected and actual values of X.
    """
    # Unpack parameters
    cxc, cyc, fc, roll, pitch, yaw, dx, dy, dz = params
    
    # Extract inputs
    u, v, zc = np.array(inputs).T
    
    # Calculate expected X using vectorized operations
    CI = CameraIntrinsic()
    x = (u - CI.ppx) * zc / CI.fx
    y = (v - CI.ppy) * zc / CI.fy
    z = zc
    Xc = np.array([x, y, z])
    R = rotation_matrix(roll, pitch, yaw)
    T = np.array([dx, dy, dz])
    T = np.tile(T, (Xc.shape[1], 1)).T
    # print(Xc.shape, R.shape, T.shape)
    expected_X = np.dot(R, Xc) + T
    
    # Extract outputs
    u_out, v_out, zc_out = np.array(outputs).T
    actual_X = np.array([(u - CI.ppx) * zc_out / CI.fx, (v - CI.ppy) * zc_out / CI.fy, zc_out]).T
    # print(expected_X.shape, actual_X.shape)
    # Calculate error using vectorized operations
    error = np.sum(np.linalg.norm(expected_X.T - actual_X, axis=1))/len(inputs)
    print(error)
    
    return error

def one_shot_predict(optimized_params, i,j, zc):
    cxc, cyc, fc, roll, pitch, yaw, dx, dy, dz = optimized_params
    R = rotation_matrix(roll, pitch, yaw)
    xc = zc * (i - cxc) / fc
    yc = zc * (j - cyc) / fc
    Xc = np.array([xc, yc, zc])
    T = np.array([dx, dy, dz])
    X = np.dot(R, Xc) + T
    return X[2]


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
    
    
    
    def testDAM(self, image, depth=None, DEVICE="cuda"):
        
        from zoedepth.models.base_models.depth_anything_lora import DepthAnythingLoraCore
        from zoedepth.models.base_models.depth_anything import DepthAnythingCore
        checkpoint="./depth_anything_finetune/transcg_relative.pt"
        rgb=image
        image=torch.tensor(np.array(image))
        image=image.permute(2,0,1)
        image=image.unsqueeze(0).float()
        image=image/255.0
        b,c,w,h=image.shape
        depth_anything= DepthAnythingCore.build(img_size=[518,518])
        depth_anything=load_wts(depth_anything, checkpoint)

        output=depth_anything(image, denorm=False, return_rel_depth=True)
  
        depth_numpy=F.interpolate(output.unsqueeze(0), size=(w, h), mode='bilinear', align_corners=False)
        
        depth_numpy=depth_numpy.to(torch.float32)

        # print(depth)
        if depth is not None:
            depth=torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            depth[depth==0]=1e-3
            mask=(depth>1e-3)
            
            # tp_mask=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000002-mask.png")
            # tp_mask = torch.tensor(tp_mask, dtype=torch.bool)
            # rgb_mask=tp_mask
            # tp_mask=tp_mask.permute(2,0,1)[0].unsqueeze(0)
            # mask=mask & tp_mask
            loss=ScaleAndShiftInvariantLoss()
            print(loss(depth_numpy, depth,  mask))
            # print(loss(1/depth, depth,  mask ))
            res=compute_ssi_metrics(depth, depth_numpy)
            import matplotlib.pyplot as plt
            # Visualize depth_numpy
            plt.imshow(depth_numpy.squeeze(), cmap='jet')
            plt.colorbar()
            plt.savefig('depth_zoe.png')
            plt.close()
            
            depth_aligned=compute_align(depth, depth_numpy)
            plt.imshow(depth_aligned.squeeze(), cmap='jet',vmin=0, vmax=1.5)
            plt.colorbar()
            plt.savefig('depth_aligned.png')
            plt.close()
            # Visualize the fitted depth
        
            
            
            plt.imshow(depth.squeeze(), cmap='jet',vmin=0, vmax=1.5)
            plt.colorbar()
            plt.savefig('depth_gt.png')
            plt.close()


            plt.imshow(rgb)
            plt.savefig('rgb.png')
            plt.close()

            
 
        
    
    def predictDepth(self, image, depth=None, DEVICE="cuda"):
        
        checkpoint="./depth_anything_finetune/transcg_dam.pt"
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)

        depth_anything = depth_anything.to(DEVICE)

        from PIL import Image
        # depth_numpy=Image.open("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000000-transparent-rgb-img.jpg").convert("RGB")  # load
        # image = Image.open("/common/home/gt286/BinPicking/Depth-Anything/metric_depth/data/nyu/transcg/scene18/0/rgb1.png").convert("RGB")  # load
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        
        print(depth_numpy)
        # print(depth)
        if depth is not None:
            depth=np.array(depth)
            mask = np.logical_and((depth> 0),(depth<1))
   
            nonzero_indices = np.nonzero(depth)
            x_data = nonzero_indices[0]
            y_data = nonzero_indices[1]
            #predicted from models
            z_data = depth_numpy[nonzero_indices]
            # gt    
            zt_data=depth[nonzero_indices]
            inputs = np.concatenate((x_data[:, np.newaxis], y_data[:, np.newaxis], z_data[:, np.newaxis]), axis=1)
            outputs=np.concatenate((x_data[:, np.newaxis], y_data[:, np.newaxis], zt_data[:, np.newaxis]), axis=1)
            

            initial_params = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # optimized_params = minimize(one_shot_error_function, initial_guess, args=(pixel_depth,), method="CG", options={"maxiter": 1,"disp": True})
            opt = nlopt.opt(nlopt.LN_BOBYQA, len(initial_params))
            opt.set_min_objective(lambda params, grad: objective_function(params, inputs, outputs))
            opt.set_lower_bounds([-1000] * len(initial_params))  # No lower bounds for parameters
            opt.set_upper_bounds([1000] * len(initial_params))   # No upper bounds for parameters
            opt.set_stopval(0.03) 
            # opt.set_ftol_rel(1e-2)
            result = opt.optimize(initial_params)
            final_error = opt.last_optimum_value()

            print("Final error:", final_error)
            print(result)
        

            fitted_depths=np.zeros_like(depth)
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    # fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]), xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
                    fitted_depths[i,j]=one_shot_predict(result, i,j,depth_numpy[i,j])
            
            # print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
            # print(xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
            # # Calculate absolute differences between original and fitted depth maps
            absolute_diff = np.abs(depth[mask] - fitted_depths[mask])

            # # Calculate mean absolute error
            mean_absolute_error = np.mean(absolute_diff)

            print("Mean Absolute Error (MAE):", mean_absolute_error)
            import matplotlib.pyplot as plt
            # Visualize depth_numpy
            plt.imshow(depth_numpy.squeeze(), cmap='jet',vmin=0.3, vmax=1)
            plt.colorbar()
            plt.savefig('depth_zoe.png')
            plt.close()
            
            # Visualize the fitted depth
            plt.imshow(fitted_depths.squeeze(), cmap='jet',vmin=0.3, vmax=1)
            plt.colorbar()
            plt.savefig('fitted_depth.png')
            plt.close()
            
            
            plt.imshow(depth.squeeze(), cmap='jet',vmin=0.3, vmax=1)
            plt.colorbar()
            plt.savefig('depth_gt.png')
            plt.close()


            plt.imshow(image)
            plt.savefig('rgb.png')
            plt.close()

        # depth_pil = depth_anything.infer_pil(image, output_type="pil")  # as 16-bit PIL Image


    def dump_to_pointcloud(self, image,  depth_scale=0.3, clip_distance_max=1, intrinsics=CameraIntrinsic()):
        DEVICE="cuda"
        from zoedepth.models.model_io import load_wts
        checkpoint="./depth_anything_finetune/mydata.pt"
        
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)

        depth_anything = depth_anything.to(DEVICE)
    
        
        # Local file
        from PIL import Image
        # image = Image.open("./frame_color.png").convert("RGB")  # load
        rgb = np.array(image)
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        depth_numpy=depth_numpy
        print(depth_numpy)

        points=create_pc.depth2PointCloudFakeDepth(depth_numpy, rgb, depth_scale, clip_distance_max, intrinsics)
        create_pc.create_point_cloud_file2(points, "pc.ply")
        # depth_tensor = depth_anything.infer_pil(image, output_type="tensor")  # as torch tensor


def generate_fake_depth(input_folder, output_folder):
    DEVICE="cuda"
    from zoedepth.models.model_io import load_wts
    from PIL import Image
    from zoedepth.utils.misc import colorize
    checkpoint="./depth_anything_finetune/transcg_dam.pt"
    
    config=get_config("zoedepth", "train", "nyu")
    depth_anything = build_model(config)
    depth_anything= load_wts(depth_anything, checkpoint)
    
    depth_anything = depth_anything.to(DEVICE)
    for filename in os.listdir(input_folder):
        if not filename.endswith(('.png', '.jpg')):
            continue

        if "depth" not in filename:
            print(filename)
            image = Image.open(input_folder+"/"+filename).convert("RGB")  # load
            depth_numpy = depth_anything.infer_pil(image)
            colored = colorize(depth_numpy)
            np.savez_compressed(output_folder+"/"+filename.split(".")[0]+"_fakedepth.npz", depth=depth_numpy)
            fpath_colored=output_folder+"/"+filename.split(".")[0]+"_fakedepth.png"
            Image.fromarray(colored).save(fpath_colored)
  



    
dam =DAM()
depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000017-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000017-transparent-rgb-img.jpg")

from PIL import Image
# image=cv2.imread("./data/nyu/transcg/scene2/70/rgb2.png")
# depth =np.array(Image.open("./data/nyu/transcg/scene2/70/depth2-gt-converted.png"))

# image=cv2.imread("./data/nyu/pose_test/001/5_opaque_color.png")
# depth =np.array(Image.open("./data/nyu/pose_test/001/5_gt_depth.png"))

# image=cv2.imread("./data/nyu/clearpose_downsample_100/set1/scene1/010100-color.png")
# depth =np.array(Image.open("./data/nyu/clearpose_downsample_100/set1/scene1/010100-depth.png"))

scale=1
# depth=dam.testDAM(image, depth/scale)
dam.predictDepth(image, depth/scale)
# dam.dump_to_pointcloud(image)

# generate_fake_depth("/common/home/gt286/BinPicking/objet_dataset/object_dataset_6/", "./fakedepth/")


