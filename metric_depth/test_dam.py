
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
# print(project_path)
import create_pc
# sys.path.append(project_path+"/torchhub")
# print(project_path+"/torchhub/facebookresearch_dinov2_main")
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model
from zoedepth.trainers.loss import ScaleAndShiftInvariantLoss, compute_scale_and_shift
from zoedepth.utils.misc import compute_errors, compute_ssi_metrics

# class CameraIntrinsic:
#     def __init__(self,fx=595.429443359375,fy=595.7514038085938,ppx=321.8606872558594,ppy=239.07879638671875) -> None:
#         self.fx = fx
#         self.fy = fy
#         self.ppx = ppx
#         self.ppy = ppy

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
    
        image=torch.tensor(np.array(image))
        image=image.permute(2,0,1)
        image=image.unsqueeze(0).float()
        image=image/255.0
        b,c,w,h=image.shape
        depth_anything= DepthAnythingLoraCore.build()

        depth_anything = depth_anything
        output=depth_anything(image, denorm=False, return_rel_depth=True)
  
        depth_numpy=F.interpolate(output.unsqueeze(0), size=(w, h), mode='bilinear', align_corners=False)
        
        depth_numpy=depth_numpy.to(torch.float32)
    
        # print(depth)
        if depth is not None:
            # mask = np.logical_and((depth> 0),(depth<2))
            depth=torch.tensor(depth, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            print(depth_numpy.shape,torch.tensor(depth).unsqueeze(0).unsqueeze(0).shape )
            loss=ScaleAndShiftInvariantLoss()
            
            res=compute_ssi_metrics( depth, depth_numpy)
            # disparity=1/depth
            # disparity[depth==0]=0
            # # 
            # print("???",depth_numpy.shape, torch.tensor(disparity).unsqueeze(0).shape,torch.tensor(mask).unsqueeze(0).shape)
            # scale,shift=compute_scale_and_shift(depth_numpy, torch.tensor(disparity).unsqueeze(0),torch.tensor(mask).unsqueeze(0))
            # print(scale.shape)
            # scaled_disparity=  scale.view(-1, 1, 1) * depth_numpy + shift.view(-1, 1, 1)
            # scaled_depth=1/scaled_disparity
            # scaled_depth[scaled_depth<0]=1e-3
            # scaled_depth[scaled_depth>2]=2
            # scaled_depth=scaled_depth.numpy()
            # scaled_depth=scaled_depth.squeeze(0)
            # print(scaled_depth.shape, depth.shape, mask.shape)
            # res=compute_errors(scaled_depth[mask], depth[mask])
            
            print(res)
            # scaled_prediction = scale.view(-1, 1, 1) * torch.tensor(depth_numpy).unsqueeze(0), + shift.view(-1, 1, 1)
            # print(scaled_prediction)
            
            # err=loss(depth_numpy.unsqueeze(0).unsqueeze(0), torch.tensor(depth).unsqueeze(0).unsqueeze(0),torch.tensor(mask).unsqueeze(0).unsqueeze(0))
            # print("the loss function is",err)
        depth_numpy=torch.tensor(depth_numpy, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # print(depth_numpy.shape,torch.tensor(depth).unsqueeze(0).unsqueeze(0).shape )
        depth_numpy = (depth_numpy - depth_numpy.min()) / (depth_numpy.max() - depth_numpy.min()) * 255.0
    
        depth_numpy = depth_numpy.cpu().numpy().astype(np.uint8)
        depth_numpy=depth_numpy.squeeze()
        depth_numpy = np.repeat(depth_numpy[..., np.newaxis], 3, axis=-1)
        print(depth.shape)
        cv2.imwrite("output_colored.png", depth_numpy)
        
        
    
    def predictDepth(self, image, depth=None, DEVICE="cuda"):
        
        from zoedepth.models.model_io import load_wts
    
        # checkpoint="./checkpoints/depth_anything_vitl14.pth"
        checkpoint="./depth_anything_finetune/ZoeDepthv1_24-Apr_15-33-0b38ad832d3c_best.pt"
        
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
            mask = np.logical_and((depth> 0),(depth<2))
            
            print(depth_numpy.shape, depth.shape, torch.tensor(mask).unsqueeze(0).unsqueeze(0).shape)
            loss=ScaleAndShiftInvariantLoss()
            # scale,shift=compute_scale_and_shift(torch.tensor(depth_numpy).unsqueeze(0).unsqueeze(0), torch.tensor(depth).unsqueeze(0).unsqueeze(0),torch.tensor(mask).unsqueeze(0).unsqueeze(0))
            # scaled_prediction = scale.view(-1, 1, 1) * torch.tensor(depth_numpy).unsqueeze(0), + shift.view(-1, 1, 1)
            # print(scaled_prediction)
            
            # err=loss(torch.tensor(depth_numpy).unsqueeze(0).unsqueeze(0), torch.tensor(depth).unsqueeze(0).unsqueeze(0),torch.tensor(mask).unsqueeze(0).unsqueeze(0))
            # print(err)
        
        
        from zoedepth.utils.misc import colorize

        colored = colorize(depth_numpy)
        from PIL import Image
        fpath_colored = "./output_colored.png"
        Image.fromarray(colored).save(fpath_colored)
        # depth_pil = depth_anything.infer_pil(image, output_type="pil")  # as 16-bit PIL Image


    def dump_to_pointcloud(self, image,  depth_scale=0.3, clip_distance_max=1, intrinsics=CameraIntrinsic()):
        DEVICE="cuda"
        from zoedepth.models.model_io import load_wts
        checkpoint="./depth_anything_finetune/transcg_dam.pt"
        
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
# depth=exr_loader("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000000-transparent-depth-img.exr", ndim = 1, ndim_representation = ['R'])
# image = cv2.imread("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000001-transparent-rgb-img.jpg")
from PIL import Image
image=cv2.imread("./data/nyu/transcg/scene1/0/rgb1.png")
depth =np.array(Image.open("./data/nyu/transcg/scene1/0/depth1-gt.png"))

# image= cv2.imread("../../object_dataset/object_dataset_14/17_color.png")
# depth=np.array(Image.open("../../object_dataset/object_dataset_14/17_gt_depth.png"))

# depth = dam.predictDepth(image, depth/1000)
depth=dam.testDAM(image, depth/1000)
# dam.dump_to_pointcloud(image)

# generate_fake_depth("/common/home/gt286/BinPicking/objet_dataset/object_dataset_6/", "./fakedepth/")


# import cv2
# import numpy as np

# # Read depth image
# depth_image = cv2.imread('output_colored.png', cv2.IMREAD_UNCHANGED)

# # Intrinsic parameters
# fx = 1000  # example focal length
# fy = 1000  # example focal length
# cx = depth_image.shape[1] / 2  # example principal point
# cy = depth_image.shape[0] / 2  # example principal point

# # Convert depth image to point cloud
# points = []
# for y in range(depth_image.shape[0]):
#     for x in range(depth_image.shape[1]):
#         depth = depth_image[y, x, 0]
#         if depth > 0:  # ignore invalid depth values
#             X = (x - cx) * depth / fx
#             Y = (y - cy) * depth / fy
#             Z = depth
#             points.append([X, Y, Z])
# import open3d as o3d
# # Convert list of points to numpy array
# point_cloud = np.array(points)

# # Create Open3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud)

# # Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])

