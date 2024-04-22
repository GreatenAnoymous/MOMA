
import numpy as np
import cv2
from matplotlib import pyplot as plt

def depth2PointCloud(depth, rgb, depth_scale, clip_distance_max, intrinsics=None):
    
    # intrinsics = depth.profile.as_video_stream_profile().intrinsics
    # print(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, "intrinsics")
        # 595.429443359375 595.7514038085938 321.8606872558594 239.07879638671875 intrinsics

    depth = np.asanyarray(depth.get_data()) * depth_scale # 1000 mm => 0.001 meters
    
    print(depth,"depth_scale")
    rgb = np.asanyarray(rgb.get_data())
    rows,cols  = depth.shape

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)

    valid = (depth > 0) & (depth < clip_distance_max) #remove from the depth image all values above a given value (meters).
    valid = np.ravel(valid)
    z = depth 
    x =  z * (c - intrinsics.ppx) / intrinsics.fx
    y =  z * (r - intrinsics.ppy) / intrinsics.fy
   
    z = np.ravel(z)[valid]
    x = np.ravel(x)[valid]
    y = np.ravel(y)[valid]
    
    r = np.ravel(rgb[:,:,0])[valid]
    g = np.ravel(rgb[:,:,1])[valid]
    b = np.ravel(rgb[:,:,2])[valid]
    
    pointsxyzrgb = np.dstack((x, y, z, r, g, b))
    pointsxyzrgb = pointsxyzrgb.reshape(-1,6)

    return pointsxyzrgb      

def depth2PointCloudFakeDepth(depth, rgb, depth_scale, clip_distance_max, intrinsics=None):
    
    # intrinsics = depth.profile.as_video_stream_profile().intrinsics
    # print(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy, "intrinsics")
        # 595.429443359375 595.7514038085938 321.8606872558594 239.07879638671875 intrinsics

    print(depth,"depth_scale")

    rows,cols  = depth.shape
    print(rows,cols,"rows,cols")
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)

    valid = (depth > 0) & (depth < clip_distance_max) #remove from the depth image all values above a given value (meters).
    valid = np.ravel(valid)
    z = depth 
    x =  z * (c - intrinsics.ppx) / intrinsics.fx
    y =  z * (r - intrinsics.ppy) / intrinsics.fy
   
    z = np.ravel(z)[valid]
    x = np.ravel(x)[valid]
    y = np.ravel(y)[valid]
    
    r = np.ravel(rgb[:,:,0])[valid]
    g = np.ravel(rgb[:,:,1])[valid]
    b = np.ravel(rgb[:,:,2])[valid]
    
    pointsxyzrgb = np.dstack((x, y, z, r, g, b))
    pointsxyzrgb = pointsxyzrgb.reshape(-1,6)

    return pointsxyzrgb  

def create_point_cloud_file2(vertices, filename):
    ply_header = '''ply
  format ascii 1.0
  element vertex %(vert_num)d
  property float x
  property float y
  property float z
  property uchar red
  property uchar green
  property uchar blue
  end_header
  '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')
