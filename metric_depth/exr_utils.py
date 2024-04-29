import OpenEXR

import cv2
import torch
import Imath
import random
import OpenEXR
import numpy as np

def exr_loader(exr_path, ndim = 3, ndim_representation = ['R', 'G', 'B']):
    """
    Loads a .exr file as a numpy array.

    This is adapted from implicit-depth repository, ref: https://github.com/NVlabs/implicit_depth/blob/main/src/utils/data_augmentation.py.

    Parameters
    ----------

    exr_path: path to the exr file
    
    ndim: number of channels that should be in returned array. Valid values are 1 and 3.
        - if ndim=1, only the 'R' channel is taken from exr file;
        - if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file. The exr file must have 3 channels in this case.
    
    depth_representation: list of str, the representation of channels, default = ['R', 'G', 'B'].
    
    Returns
    -------

    numpy.ndarray (dtype=np.float32).
        - If ndim=1, shape is (height x width);
        - If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(exr_path)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    assert ndim == len(ndim_representation), "ndim should match ndim_representation."

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ndim_representation:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel(ndim_representation[0], pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr