
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from zoedepth.utils.config import get_config
from importlib import import_module
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# print(project_path)
import create_pc
# sys.path.append(project_path+"/torchhub")
# print(project_path+"/torchhub/facebookresearch_dinov2_main")
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.builder import build_model

class CameraIntrinsic:
    def __init__(self,fx=595.429443359375,fy=595.7514038085938,ppx=321.8606872558594,ppy=239.07879638671875) -> None:
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
    
    
    def predictDepth(self, image, DEVICE="cuda"):
        from zoedepth.models.model_io import load_wts
        checkpoint="./model_weights.pth"
        # checkpoint="./checkpoints/depth_anything_vitl14.pth"
        # checkpoint="./depth_anything_finetune/ZoeDepthv1_03-Apr_15-44-85d899fe7050_best.pt"
        
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)

        depth_anything = depth_anything.to(DEVICE)

        from PIL import Image
        image = Image.open("/common/home/gt286/BinPicking/Depth-Anything/metric_depth/data/nyu/transcg/scene18/0/rgb1.png").convert("RGB")  # load
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        print(depth_numpy)
        from zoedepth.utils.misc import colorize

        colored = colorize(depth_numpy)
        from PIL import Image
        fpath_colored = "./output_colored.png"
        Image.fromarray(colored).save(fpath_colored)
        # depth_pil = depth_anything.infer_pil(image, output_type="pil")  # as 16-bit PIL Image


    def dump_to_pointcloud(self,  depth_scale=0.3, clip_distance_max=0.5, intrinsics=CameraIntrinsic()):
        DEVICE="cuda"
        from zoedepth.models.model_io import load_wts
        checkpoint="./model_weights.pth"
        
        config=get_config("zoedepth", "train", "nyu")
        depth_anything = build_model(config)
        depth_anything= load_wts(depth_anything, checkpoint)
      
        depth_anything = depth_anything.to(DEVICE)
    
        
        # Local file
        from PIL import Image
        image = Image.open("./frame_color.png").convert("RGB")  # load
        rgb = np.array(image)
        depth_numpy = depth_anything.infer_pil(image)  # as numpy
        depth_numpy=depth_numpy*depth_scale
        print(depth_numpy)

        points=create_pc.depth2PointCloudFakeDepth(depth_numpy, rgb, depth_scale, clip_distance_max, intrinsics)
        create_pc.create_point_cloud_file2(points, "pc.ply")
        # depth_tensor = depth_anything.infer_pil(image, output_type="tensor")  # as torch tensor


def generate_fake_depth(input_folder, output_folder):
    DEVICE="cuda"
    from zoedepth.models.model_io import load_wts
    from PIL import Image
    from zoedepth.utils.misc import colorize
    checkpoint="./model_weights.pth"
    
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
# image = cv2.imread("/common/home/gt286/BinPicking/objet_dataset/object_dataset_5/0_color.png")


depth = dam.predictDepth(None)
# dam.dump_to_pointcloud()

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

