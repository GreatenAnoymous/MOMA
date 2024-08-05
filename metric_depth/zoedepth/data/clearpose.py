import os
import yaml
import json
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_preparation import process_data, remove_leading_slash, process_data_light
import random
from PIL import Image

class ClearPoseDataset(Dataset):
    """
    ClearGrasp real-world dataset.
    """
    def __init__(self, config, mode, transform=None, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;
        
        split: str in ['train', 'test'], optional, default: 'test', the dataset split option.
        """
        super(ClearPoseDataset, self).__init__()
      
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.image_size= kwargs.get('image_size', (640, 480))
        self.depth_min = kwargs.get('depth_min', 1e-3)
        self.depth_max = kwargs.get('depth_max', 2)
        self.depth_norm = kwargs.get('depth_norm', 1.0)


    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)
        return image, depth_gt
    
    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    
    
    def __getitem__(self, id):
        sample_path = self.filenames[id]

        focal = float(sample_path.split()[2])
        if self.mode == 'train':
            image_path = os.path.join(
                self.config.data_path, remove_leading_slash(sample_path.split()[0]))
            depth_path = os.path.join(
                self.config.gt_path, remove_leading_slash(sample_path.split()[1]))
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
        
            random_angle = (random.random() - 0.5) * 2 * self.config.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(
                depth_gt, random_angle, flag=Image.NEAREST)
        
            image = np.asarray(image, dtype=np.float32) 
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            image=image/255.0
            depth_gt = depth_gt / 1000.0
            image, depth_gt=self.train_preprocess(image, depth_gt)
            return process_data_light(image, depth_gt, self.depth_min, self.depth_max)
        
        elif self.mode == 'online_eval':
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path
            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            
            image = Image.open(image_path)
            gt_path = self.config.gt_path_eval
            depth_path = os.path.join(
                gt_path, remove_leading_slash(sample_path.split()[1]))
            has_valid_depth = False
            try:
                depth_gt = Image.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = False
                print('Missing gt for {}'.format(image_path))
                assert False 
            image=np.asarray(image, dtype=np.float32)
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            image=image/255.0
            depth_gt = depth_gt / 1000.0
            return  process_data_light(image, depth_gt, self.depth_min, self.depth_max)

    def __len__(self):
        return len(self.filenames)

