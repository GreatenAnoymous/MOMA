
import os
import yaml
import json
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_preparation import process_data, exr_loader, remove_leading_slash, process_depth, CachedReader
import cv2
import random



    
    
    


class DepthCompleteData(Dataset):
    """
    depth completion dataset
    """
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()
        self.reader=CachedReader()
        self.mode=mode
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval


    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[4])
        sample = {}

        if self.mode == 'train':
            image_path = os.path.join(
                self.config.data_path, remove_leading_slash(sample_path.split()[0]))
            depth_path = os.path.join(
                self.config.gt_path, remove_leading_slash(sample_path.split()[1]))
            raw_depth_path=os.path.join(self.config.gt_path, remove_leading_slash(sample_path.split()[2]))
            mask_path=os.path.join(self.config.gt_path, remove_leading_slash(sample_path.split()[3]))
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            depth_raw = Image.open(raw_depth_path)
            w, h = image.size
            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (np.random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                depth_raw = self.rotate_image(depth_raw, random_angle, flag=Image.NEAREST)
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_raw = np.asarray(depth_raw, dtype=np.float32)
            depth_gt = depth_gt / 1000.0
            depth_raw = depth_raw / 1000.0
      

            depth_gt = process_depth(depth_gt)
            depth_raw = process_depth(depth_raw)

            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_raw = np.expand_dims(depth_raw, axis=2)
            
  
            
                
            
                
            image, depth_gt, depth_raw = self.train_preprocess(image, depth_gt, depth_raw)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            
            if not np.any(mask):
                print(depth_gt.shape, depth_gt.min(), depth_gt.max())
                raise ValueError("Mask does not have any True values.")
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, "depth_raw": depth_raw,**sample}

        else:
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0]))
            image = np.asarray(self.reader.open(image_path),
                               dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.config.gt_path_eval
                depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[1]))
                raw_depth_path = os.path.join(gt_path, remove_leading_slash(sample_path.split()[2]))
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    depth_raw = self.reader.open(raw_depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt for {}'.format(image_path))
                    assert False 

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_raw = np.asarray(depth_raw, dtype=np.float32)
                    depth_gt = depth_gt / 1000.0
                    depth_raw = depth_raw / 1000.0
                    depth_raw=process_depth(depth_raw)
                    depth_gt = process_depth(depth_gt)

                    
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    depth_raw = np.expand_dims(depth_raw, axis=2)
                    

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                else:
                    mask = False
            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask, "depth_raw": depth_raw}
            else:
                sample = {'image': image, 'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt, depth_raw):
        if self.config.aug:
            # Random flipping
            do_flip = np.random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()
                depth_raw = (depth_raw[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = np.random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt, depth_raw

    def augment_image(self, image):
        # gamma augmentation
        gamma = np.random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
 
        brightness = np.random.uniform(0.75, 1.25)

        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)
    



    




if __name__=="__main__":
    depth_gt = exr_loader("/mnt/ssd_990/teng/BinPicking/cleargrasp/cleargrasp-dataset-train/cup-with-waves-train/depth-imgs-rectified/000000000-depth-rectified.exr", ndim = 1, ndim_representation = ['R'])
    print(depth_gt)