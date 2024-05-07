
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


import numpy as np
from scipy.optimize import curve_fit


def model_function(xy, xc, yc, theta1, theta2, d, lambda1, lambda2, lambda3):
    x, y,z  = xy
   # assuming z is the third element of xy
    return lambda3* z + lambda1 * (x - xc) * np.cos(theta1) + lambda2 * (y - yc) * np.cos(theta2) + d


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
    
    
    
    def testDAM(self, image, depth=None, DEVICE="cuda"):
        
        from zoedepth.models.base_models.depth_anything_lora import DepthAnythingLoraCore
        from zoedepth.models.base_models.depth_anything import DepthAnythingCore
        checkpoint="./depth_anything_finetune/transcg_relative.pt"

        image=torch.tensor(np.array(image))
        image=image.permute(2,0,1)
        image=image.unsqueeze(0).float()
        image=image/255.0
        b,c,w,h=image.shape
        depth_anything= DepthAnythingCore.build(img_size=[518,518])
        # depth_anything=load_wts(depth_anything, checkpoint)

        output=depth_anything(image, denorm=False, return_rel_depth=True)
  
        depth_numpy=F.interpolate(output.unsqueeze(0), size=(w, h), mode='bilinear', align_corners=False)
        
        depth_numpy=depth_numpy.to(torch.float32)

        # print(depth)
        if depth is not None:
            depth=torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            depth[depth==0]=1.5
            mask=(depth>1e-3)
            depth[depth>1.5]=1.5
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
            depth_numpy=compute_align(depth,depth_numpy)
        
            # Visualize depth_numpy
            plt.imshow(depth_numpy.squeeze(), cmap='jet')
            plt.colorbar()
            plt.savefig('depth_numpy.png')
            plt.close()
            print(res)
            
            # visualize the depth also
            plt.imshow(depth.squeeze(), cmap='jet')
            plt.colorbar()
            plt.savefig('depth_gt.png')
            plt.close()
            
            # rgb_mask=rgb_mask.permute(2,0,1).unsqueeze(0)
            # image[~rgb_mask]=0
            plt.imshow(image.squeeze().permute(1,2,0))
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
            mask = np.logical_and((depth> 0),(depth<2))
            # Flatten the depth_numpy and depth arrays
            # depth_numpy_flat = depth_numpy[mask].flatten()
            # depth_flat = depth[mask].flatten()

            # # Fit a linear function to the data
            # coefficients = np.polyfit(1/depth_numpy_flat, 1/depth_flat, 1)

            # # Extract the slope and intercept from the coefficients
            # slope = coefficients[0]
            # intercept = coefficients[1]

            # # Apply the linear function to depth_numpy
            # fitted_depths = 1/(1/depth_numpy * slope + intercept)
            
            nonzero_indices = np.nonzero(depth)
            x_data = nonzero_indices[0]
            y_data = nonzero_indices[1]
            z_data = depth_numpy[nonzero_indices]
            zt_data=depth[nonzero_indices]
            initial_guess = [1, 0, 0, 0, 0, 0,0,0]
            # popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)
            # xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt = popt
            xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt = [-200.71216423933848, 0.4964572185666514, 0.0014144118823155461, -4.266939590415664e-06, 0.030943094561458162, -0.0017814284461063947, 0.0002689114813523485, 2.6810426203350337]
            fitted_depths=np.zeros_like(depth)
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]), xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
            
            print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
            print(xc_opt, yc_opt, theta1_opt, theta2_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
            # # Calculate absolute differences between original and fitted depth maps
            absolute_diff = np.abs(depth[mask] - fitted_depths[mask])

            # # Calculate mean absolute error
            mean_absolute_error = np.mean(absolute_diff)

            print("Mean Absolute Error (MAE):", mean_absolute_error)
            import matplotlib.pyplot as plt
            # Visualize depth_numpy
            plt.imshow(depth_numpy.squeeze(), cmap='jet',vmin=0, vmax=1.5)
            plt.colorbar()
            plt.savefig('depth_zoe.png')
            plt.close()
            
            # Visualize the fitted depth
            plt.imshow(fitted_depths.squeeze(), cmap='jet',vmin=0, vmax=1.5)
            plt.colorbar()
            plt.savefig('fitted_depth.png')
            plt.close()
            
            
            plt.imshow(depth.squeeze(), cmap='jet',vmin=0, vmax=1.5)
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

# image=cv2.imread("./data/nyu/arcl/001/1_color.png")
# depth =np.array(Image.open("./data/nyu/arcl/001/1_gt_depth.png"))

# image=cv2.imread("./data/nyu/clearpose_downsample_100/set1/scene1/010100-color.png")
# depth =np.array(Image.open("./data/nyu/clearpose_downsample_100/set1/scene1/010100-depth.png"))

scale=1
# depth=dam.testDAM(image, depth/scale)
dam.predictDepth(image, depth/scale)
# dam.dump_to_pointcloud(image)

# generate_fake_depth("/common/home/gt286/BinPicking/objet_dataset/object_dataset_6/", "./fakedepth/")


