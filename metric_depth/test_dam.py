
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


def model_function(xy, xc, yc, d, alpha, beta, fc):
    x, y,z  = xy
   # assuming z is the third element of xy
    return np.cos(beta)*np.cos(alpha)* z -np.sin(beta) * (x - xc) * z*fc  + np.sin(alpha)*np.cos(beta) * (y - yc) * z *fc + d


# def local_alignment(ui,vi, u, v, depths, depths_gt,alpha=0.01 , b=1280):
#     m= u.shape[0]

#     W = np.zeros((m, m))  # Initialize weight matrix
    
#     for k in range(m):
#         dist_k = np.sqrt((u[k] - ui)**2 + (v[k] - vi)**2)
#         wk = 1 / np.sqrt(2 * np.pi) * np.exp(-alpha* dist_k**2 / (2 * b**2))
#         W[k, k] = wk
#     X = np.vstack((depths, np.ones_like(depths)))  # Construct X
#     X = X.T

#     depths_gt = depths_gt.reshape(-1, 1)
#     XTWX = np.dot(np.dot(X.T, W), X)
#     print(XTWX)
#     XTWy = np.dot(np.dot(X.T, W), depths_gt) 
#     beta = np.dot(np.linalg.inv(XTWX), XTWy)
#     return beta

# def compute_local_alignment(u, v, depths, depths_gt, M, N, b=2):
#     m = u.shape[0]
#     beta = np.zeros((2, M, N))
#     for i in range(M):
#         for j in range(N):
#             beta[:, i, j] = local_alignment(i, j, u, v, depths, depths_gt, b).flatten()
#     return beta

def compute_local_alignment(u, v, depths, depths_gt, M, N, b=640, alpha=1e-7):
    # Compute distance matrix
    ui, vi = np.indices((M, N))
    dist = np.sqrt((u[:, None, None] - ui)**2 + (v[:, None, None] - vi)**2)
    print(dist.shape, depths.shape, depths_gt.shape)
    # Compute weight matrix
    w = 1 / np.sqrt(2 * np.pi) * np.exp(-alpha * dist**2 / (2 * b**2))
    W = np.zeros([len(u), len(u),M,N])
    W[:, :, ui, vi] = w
    W=np.transpose(W, (2,3,0,1))

    # Construct X matrix
    X = np.vstack((depths, np.ones_like(depths)))
    X = X.T

    # Compute intermediate matrices
    # XTWX = np.matmul(np.matmul(X.T, W), X)
    XTWX  =np.einsum('mk,ijkl,ln->ijmn',X.T, W,  X)

    XTWX =np.linalg.pinv(XTWX)

    # exit()
    # XTWy = np.matmul(np.matmul(X.T, W), depths_gt)
    depths_gt = depths_gt.reshape(-1, 1)
    XTWy=np.einsum('mk,ijkl,ln->ijmn',  X.T,W, depths_gt)

    # Compute beta using linear algebra
    beta = np.einsum('ijkl,ijlm->ijkm', XTWX, XTWy)

    # Reshape beta to match the required output shape
    print(beta.shape)
    return beta

def test_function():
    W,H=640, 480
    depth = np.random.rand(W, H)
    depth_gt = np.random.rand(W, H)

    depth_1d = depth.flatten()
    depth_gt_1d = depth_gt.flatten()
    

    u, v = np.meshgrid(np.arange(H), np.arange(W))
    
    u = u.flatten()
    v = v.flatten()
    
    # Randomly choose 200 points
    indices = np.random.choice(len(depth_1d), size=50, replace=False)

    # Get the corresponding u, v, depth_gt_1d
    u_selected = u[indices]
    v_selected = v[indices]
    depth_gt_1d_selected = depth_gt_1d[indices]
    depth_1d_selected = depth_1d[indices]
    beta=compute_local_alignment(u_selected, v_selected, depth_1d_selected, depth_gt_1d_selected, 640, 480)
    print(beta.shape)

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

           
            
 
        
    
    def predictDepth(self, image, depth=None, object_mask=None ,  depth_raw=None,DEVICE="cuda"):
        if object_mask is not None:
            object_mask=np.array(object_mask)
            object_mask=object_mask>0
        checkpoint="./depth_anything_finetune/mixed.pt"
        # checkpoint="./checkpoints/depth_anything_metric_depth_indoor.pt"
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)
        
        depth_anything = depth_anything.to(DEVICE)

        from PIL import Image
        # depth_numpy=Image.open("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000000-transparent-rgb-img.jpg").convert("RGB")  # load
        # image = Image.open("/common/home/gt286/BinPicking/Depth-Anything/metric_depth/data/nyu/transcg/scene18/0/rgb1.png").convert("RGB")  # load
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        
        # print(depth_numpy)
        

        if depth_raw is not None:
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
            # x_data = nonzero_indices[0]
            # y_data = nonzero_indices[1]
            # z_data = depth_numpy[nonzero_indices]
            # zt_data=depth[nonzero_indices]
            initial_guess = [0, 0, 0, 0, 0, 1]
            popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)

            xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt =[534.2324844480863, -190.42289893222687, -0.4183163449642485, 0.0001494418507248443, 0.0011123054000330792, 2.994101666824724]
            # xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt =[-12876.598818650644, 3879.3807488025254, 0.09036849843330247, 1.7720131641397066e-05, 7.030278082064835e-05, 0.4987274794212721]
            xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt = popt

            fitted_depths=np.zeros_like(depth_raw)
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]), xc_opt, yc_opt,  d_opt, lambda1_opt, lambda2_opt, lambda3_opt)
            
            # print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
            print([xc_opt, yc_opt, d_opt, lambda1_opt, lambda2_opt, lambda3_opt])
            # # Calculate absolute differences between original and fitted depth maps
            
            gt_mask= np.logical_and(depth>0, depth<1)
            # print(object_mask[:,:,0].min(), object_mask[:,:,0].max())
            num_nonzero = np.count_nonzero(gt_mask)
            # print("Number of Nonzero Values in object_mask:", num_nonzero)

        
            gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])
            num_nonzero = np.count_nonzero(gt_mask)
            print("Number of Nonzero Values in object_mask:", num_nonzero)

            metrics=compute_errors(depth[gt_mask], fitted_depths[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - fitted_depths[gt_mask])


            # # Calculate mean absolute error
            mean_absolute_error = np.mean(absolute_diff)
            
            print(metrics)

            print("Mean Absolute Error (MAE):", mean_absolute_error)

            ## linear fitting
            metrics=compute_errors(depth[gt_mask], scaled_fitted_depths[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - scaled_fitted_depths[gt_mask])
            mean_absolute_error = np.mean(absolute_diff)
            print(metrics)
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
            

            plt.imshow(scaled_fitted_depths.squeeze(), cmap='jet',vmin=0.3, vmax=1)
            plt.colorbar()
            plt.savefig('scaled_fitted_depth.png')
            plt.close()
            
            plt.imshow(depth.squeeze(), cmap='jet',vmin=0.3, vmax=1)
            plt.colorbar()
            plt.savefig('depth_gt.png')
            plt.close()

            if object_mask is not None:
                image[~object_mask]=0
            plt.imshow(image)
            plt.savefig('rgb.png')
            plt.close()

            # print(depth.shape)
            # w,h=depth.shape[0], depth.shape[1]
            # depth_gt_1d=depth.flatten()
            # depth_1d=depth_numpy.flatten()
            # valid=depth_gt_1d>0
            # depth_gt_1d=depth_gt_1d[valid]
            # depth_1d=depth_1d[valid]
            # u,v=np.meshgrid(np.arange(h), np.arange(w))
            # u=u.flatten()
            # v=v.flatten()
            # u=u[valid]
            # v=v[valid]  
            # indices = np.random.choice(len(depth_1d), size=50, replace=False)
            # u_selected = u[indices]
            # v_selected = v[indices]
            # depth_gt_1d_selected = depth_gt_1d[indices]
            # depth_1d_selected = depth_1d[indices]
            # beta=compute_local_alignment(u_selected, v_selected, depth_1d_selected, depth_gt_1d_selected, w, h)
            # print(beta.shape)
            # # expanded_depth = np.expand_dims(depth, axis=-1)

            # # Perform the element-wise multiplication and addition
            # recovered_depth = beta[..., 0, 0] * depth_numpy + beta[..., 1, 0]
            
            # plt.imshow(recovered_depth.reshape(w,h), cmap='jet',vmin=0, vmax=1.5)
            # plt.colorbar()
            # plt.savefig('depth_recovered.png')
            # plt.close()




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
# depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000019-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# depth_raw=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000019-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000019-transparent-rgb-img.jpg")
# mask_image=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000019-mask.png")



from PIL import Image
image=cv2.imread("./data/nyu/transcg/scene97/81/rgb1.png")
depth =np.array(Image.open("./data/nyu/transcg/scene97/81/depth1-gt.png"))
depth_raw=np.array(Image.open("./data/nyu/transcg/scene97/81/depth1-gt.png"))
mask_image=cv2.imread("./data/nyu/transcg/scene97/81/depth1-gt-mask.png")

# # image=cv2.imread("./data/nyu/pose_test/001/15_opaque_color.png")
# # depth =np.array(Image.open("./data/nyu/pose_test/001/15_gt_depth.png"))

# # image=cv2.imread("./data/nyu/clearpose_downsample_100/set1/scene1/010100-color.png")
# # depth =np.array(Image.open("./data/nyu/clearpose_downsample_100/set1/scene1/010100-depth.png"))

scale=1000.0
# # depth=dam.testDAM(image, depth/scale)
dam.predictDepth(image, depth/scale,object_mask=mask_image, depth_raw=depth_raw/scale)
# # dam.dump_to_pointcloud(image)

# # generate_fake_depth("/common/home/gt286/BinPicking/objet_dataset/object_dataset_6/", "./fakedepth/")


# test_function()