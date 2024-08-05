
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from exr_utils import exr_loader
from sklearn import datasets, linear_model
from importlib import import_module
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

from PIL import Image
import open3d as o3d
import create_pc
from zoedepth.utils.config import get_config
from zoedepth.models.model_io import load_wts
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics, compute_align
from metric3d import Metric3D

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

import h5py
from camera import Camera


def read_from_h5(h5path="/mnt/ssd_990/teng/BinPicking/DPT_transparent_objects/metric_depth/data/nyu/omniverse_v3/train/20200910/output.1599770731.9668603.h5"):
    f = h5py.File(h5path, 'r')
    rgb = cv2.cvtColor(f['rgb_glass'][:], cv2.COLOR_RGB2BGR)
    disparity = f['depth'][:]
    depth_gt = 1. / (disparity + 1e-8) * 0.01
    depth_gt = np.clip(depth_gt, 0, 10)
    return rgb, depth_gt
    

def model_function(xy, scale,xc, yc, d, alpha, beta, fc):
    x, y,z  = xy
   # assuming z is the third element of xy
    return (np.cos(beta)*np.cos(alpha)* z -np.sin(beta) * (x - xc) * z*fc  + np.sin(alpha)*np.cos(beta) * (y - yc) * z *fc)*scale + d


def compute_local_alignment(u, v, depths, depths_gt, M, N, b=200, alpha=1):
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


def compute_local_alignment_bruteforce(u, v, depths, depths_gt, M, N, b=300, alpha=1):
    beta=np.zeros((M, N, 2, 1))
    for i in range(M):
        for j in range(N):
            w= np.exp(-alpha * ((u-i)**2 + (v-j)**2) / (2 * b**2))
            coefficients = np.polyfit(depths, depths_gt, 1, w=w)
            beta[i, j, 0, 0] = coefficients[0]
            beta[i, j, 1, 0] = coefficients[1]
    print(beta)
    return beta

    


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
    
    
    def run(self, image, DEVICE="cuda"):
        checkpoint="./depth_anything_finetune/mixed.pt"
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)
        
        depth_anything = depth_anything.to(DEVICE)
        depth_numpy = depth_anything.infer_pil(image)
        return depth_numpy
        
    
    
 
        
    
    def predictDepth(self, image, depth=None, object_mask=None ,  depth_raw=None,DEVICE="cuda"):
        if object_mask is not None:
            object_mask=np.array(object_mask)
            object_mask=object_mask>0
        checkpoint="./depth_anything_finetune/mixed.pt"
        # checkpoint="./depth_anything_finetune/transcg_dam.pt"
        # checkpoint="./depth_anything_finetune/metrics1.pt"
    
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)
        
        depth_anything = depth_anything.to(DEVICE)
        # depth_anything = Metric3D()

        from PIL import Image
        # depth_numpy=Image.open("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000000-transparent-rgb-img.jpg").convert("RGB")  # load
        # image = Image.open("/common/home/gt286/BinPicking/Depth-Anything/metric_depth/data/nyu/transcg/scene18/0/rgb1.png").convert("RGB")  # load
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        
        # print(depth_numpy)

        def visualize_point_cloud(point_cloud):
            # Create a point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd])

        # Convert depth_numpy to point cloud
        
        # Visualize the point cloud
        # visualize_point_cloud(point_cloud)
        
        

        if depth_raw is not None:
            depth_raw=np.array(depth_raw)
            depth_raw[np.isnan(depth_raw)] = 0
            mask = np.logical_and((depth_raw> 0),(depth_raw<1))
    
            
            
            N=100
            # Flatten the depth_numpy and depth arrays
            depth_numpy_flat = depth_numpy[mask].flatten()
            depth_flat = depth_raw[mask].flatten()
            indexes = np.arange(len(depth_numpy_flat))
            np.random.shuffle(indexes)
            depth_numpy_flat=depth_numpy_flat[indexes][0:N]
            depth_flat=depth_flat[indexes][0:N]





            # # Fit a linear function to the data
            # coefficients = np.polyfit(depth_numpy_flat, depth_flat, 1)

            # # Extract the slope and intercept from the coefficients
            # slope = coefficients[0]
            # intercept = coefficients[1]

            slope, intercept, r_value, p_value, std_err = stats.linregress(depth_numpy_flat, depth_flat)
            
            # Apply the linear function to depth_numpy
            # print(slope,"slope", intercept, "intercept")
            scaled_fitted_depths = depth_numpy * slope + intercept


            ## ransac fitting

            ransac=linear_model.RANSACRegressor()
            ransac.fit(depth_numpy_flat.reshape(-1, 1), depth_flat)
            inlier_mask = ransac.inlier_mask_
            outlier_mask = np.logical_not(inlier_mask)
            slope=ransac.estimator_.coef_[0]
            intercept=ransac.estimator_.intercept_
            ransac_fitted_depths = depth_numpy * slope + intercept
            ransac_fitted_depths[ransac_fitted_depths<0]=1e-3
            # ransac_fitted_depths = ransac.predict(depth_numpy.reshape(-1, 1))


            import matplotlib.pyplot as plt

            # Plot the linear fitting result
            plt.scatter(depth_numpy_flat, depth_flat, color='blue', label='Original Data')
            plt.plot(depth_numpy_flat, depth_numpy_flat * slope + intercept, color='red', label='Linear Fit')
            plt.xlabel('Depth Anything Depth')
            plt.ylabel('Ground Truth Depth')
            plt.legend()
            plt.savefig('linear_fit.png')
            plt.close()

            
            nonzero_indices = np.nonzero((depth_raw > 0) & (depth_raw < 1))

            array1, array2 = nonzero_indices

            total_indices = len(array1)
            N=100
    
            # Randomly sample N indices from the available indices
            sampled_indices = np.random.choice(total_indices, N, replace=False)
            
            
            # Use the sampled indices to create new arrays
            sampled_array1 = array1[sampled_indices]
            sampled_array2 = array2[sampled_indices]

            assert len(sampled_array1)==N, "Sampled array1 must have N elements"

            nonzero_indices = (sampled_array1, sampled_array2)


            x_data = nonzero_indices[0]
            y_data = nonzero_indices[1]
            z_data = depth_numpy[nonzero_indices]
            zt_data = depth_raw[nonzero_indices]
        
            initial_guess =  np.random.rand(7)
            popt, pcov = curve_fit(model_function, (x_data, y_data, z_data), zt_data, p0=initial_guess)
            print(popt)


            scale,xc, yc, d, alpha, beta, fc = popt

            fitted_depths=np.zeros_like(depth_raw)
            for i in range(depth.shape[0]):
                for j in range(depth.shape[1]):
                    fitted_depths[i,j]=model_function((i,j,depth_numpy[i,j]),scale, xc, yc, d, alpha, beta, fc)
            
            # print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
            point_cloud = create_pc.depth2PointCloudFakeDepth(fitted_depths, image, 0.3, 1, CameraIntrinsic())
            create_pc.create_point_cloud_file2(point_cloud, "pc.ply")
            
            
            gt_mask= np.logical_and(depth>0, depth<1)
            # print(object_mask[:,:,0].min(), object_mask[:,:,0].max())
            num_nonzero = np.count_nonzero(gt_mask)
            # print("Number of Nonzero Values in object_mask:", num_nonzero)
            assert object_mask is not None, "Object mask must be provided"
            if object_mask is not None:
                gt_mask=np.logical_and(gt_mask, object_mask[:,:,0])
            num_nonzero = np.count_nonzero(gt_mask)
            print("Number of Nonzero Values in object_mask:", num_nonzero)

            metrics=compute_errors(depth[gt_mask], fitted_depths[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - fitted_depths[gt_mask])


            # # Calculate mean absolute error
            mean_absolute_error = np.mean(absolute_diff)
            
            print(metrics)

            print("Mean Absolute Error SCALE-SHIFT-ROTATION (MAE):", mean_absolute_error)
            scaled_fitted_depths[scaled_fitted_depths<0]=1e-3
            ## linear fitting
            metrics=compute_errors(depth[gt_mask], scaled_fitted_depths[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - scaled_fitted_depths[gt_mask])
            mean_absolute_error = np.mean(absolute_diff)
            print(metrics)
            print("Mean Absolute Error Global LR (MAE):", mean_absolute_error)


            ## ransac fitting
            metrics=compute_errors(depth[gt_mask], ransac_fitted_depths[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - ransac_fitted_depths[gt_mask])
            mean_absolute_error = np.mean(absolute_diff)
            print(metrics)
            print("Mean Absolute Error RANSAC (MAE):", mean_absolute_error)


            ## zoedepth
            metrics=compute_errors(depth[gt_mask], depth_numpy[gt_mask])
            absolute_diff = np.abs(depth[gt_mask] - depth_numpy[gt_mask])
            mean_absolute_error = np.mean(absolute_diff)
            print(metrics)
            print("Mean Absolute Error (MAE):", mean_absolute_error)

            import matplotlib.pyplot as plt
            # Visualize depth_numpy
            plt.figure(figsize=(10,5))
            plt.subplot(2, 3, 1)
            plt.imshow(image)
            
            plt.axis('off')
            
            
            plt.subplot(2, 3, 2)
            plt.imshow(depth.squeeze(), cmap='jet',vmin=0.4, vmax=0.7)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.subplot(2, 3, 3)
            plt.imshow(depth_numpy.squeeze(), cmap='jet',vmin=0.4, vmax=0.7)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            # plt.savefig('depth_zoe.svg')
            # plt.close()
            
            
            
            # Visualize the fitted depth
            plt.subplot(2, 3, 4)
            plt.imshow(fitted_depths.squeeze(), cmap='jet',vmin=0.4, vmax=0.7)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(scaled_fitted_depths.squeeze(), cmap='jet',vmin=0.4, vmax=0.7)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
    


            # plt.imshow(ransac_fitted_depths.squeeze(), cmap='jet',vmin=0.4, vmax=0.7)
            # plt.colorbar()

            
            



            if object_mask is not None:
                image[~object_mask]=0
            


            print(depth.shape)
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
            indices = np.random.choice(len(depth_1d), size=20, replace=False)
            u_selected = u[indices]
            v_selected = v[indices]
            depth_gt_1d_selected = depth_gt_1d[indices]
            depth_1d_selected = depth_1d[indices]
            beta=compute_local_alignment(u_selected, v_selected, depth_1d_selected, depth_gt_1d_selected, w, h)
            print(beta.shape)
            # expanded_depth = np.expand_dims(depth, axis=-1)

            # Perform the element-wise multiplication and addition
            recovered_depth = beta[..., 0, 0] * depth_numpy + beta[..., 1, 0]
            plt.subplot(2, 3, 6)
            plt.imshow(recovered_depth.reshape(w,h), cmap='jet',vmin=0.4, vmax=0.7)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.savefig('result_example.pdf', bbox_inches='tight')
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
  


def generate_vibrate():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    dam = DAM()
    id=78
    data_id=6
    image=cv2.imread(f"./data/nyu/test/{data_id:03}/{id}_opaque_color.png")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth =np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
    depth_raw=np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
    # mask_image=cv2.imread(f"./data/nyu/test/00{data_id}/{id}_mask.png")
    mask_image=None
    scale=1000
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2,2,2)
    depth_pred=dam.run(image)
    plt.imshow(depth_pred, cmap='jet',vmin=0.4, vmax=0.8)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    
    id=33
    data_id=6
    image=cv2.imread(f"./data/nyu/test/{data_id:03}/{id}_opaque_color.png")
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth =np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
    depth_raw=np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
    plt.subplot(2,2,3)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(2,2,4)
    depth_pred=dam.run(image)
    plt.imshow(depth_pred, cmap='jet',vmin=0.4, vmax=0.8)
    plt.axis('off')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig("example_fluctuate.pdf", bbox_inches='tight')
    
    

    
dam =DAM()
# depth=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000011-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# depth_raw=exr_loader("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000011-opaque-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# image = cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000011-transparent-rgb-img.jpg")
# mask_image=cv2.imread("../../cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000011-mask.png")
# scale=1


# image=cv2.imread("./data/nyu/transcg/scene1/0/rgb2.png")
# depth =np.array(Image.open("./data/nyu/transcg/scene1/0/depth2-gt.png"))
# depth_raw=np.array(Image.open("./data/nyu/transcg/scene1/0/depth2-gt.png"))
# # mask_image=cv2.imread("./data/nyu/transcg/scene1/0/depth2-gt-mask.png")
# mask_image=None
# scale=4000


# id=1413
# image=cv2.imread(f"./data/nyu/fake/rgb_{id}.png")
# depth =np.array(Image.open(f"./data/nyu/fake/depth_{id}.png"))
# depth_raw=np.array(Image.open(f"./data/nyu/fake/depth_{id}.png"))
# mask_image=None
# scale=1000

# image, depth = read_from_h5()
# depth_raw=depth
# mask_image=None



# id=91
# image=cv2.imread(f"./data/nyu/test/010/{id}_color.png")
# depth =np.array(Image.open(f"./data/nyu/test/010/{id}_gt_depth.png"))
# depth_raw=np.array(Image.open(f"./data/nyu/test/010/{id}_gt_depth.png"))
# mask_image=np.array(Image.open(f"./data/nyu/test/010/{id}_mask.png"))
# scale=1000

# image, depth = read_from_h5()
# depth_raw=depth
# mask_image=None


# generate_vibrate()
# exit(0)
id=28
# id=2
data_id=7
image=cv2.imread(f"./data/nyu/test/{data_id:03}/{id}_opaque_color.png")
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth =np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
depth_raw=np.array(Image.open(f"./data/nyu/test/{data_id:03}/{id}_gt_depth.png"))
mask_image=cv2.imread(f"./data/nyu/test/00{data_id}/{id}_mask.png")
# mask_image=None
scale=1000




# camera = Camera()
# image, depth = camera.get_data()
# mask_image=None
# scale=1
# depth_raw=depth
# # depth=dam.testDAM(image, depth/scale)
dam.predictDepth(image, depth/scale,object_mask=mask_image, depth_raw=depth_raw/scale)
# # dam.dump_to_pointcloud(image)

# # generate_fake_depth("/common/home/gt286/BinPicking/objet_dataset/object_dataset_6/", "./fakedepth/")


# test_function()