import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
import random
from zoedepth.utils.config import change_dataset

from .data_preparation import ImReader, ToTensor, CachedReader, remove_leading_slash
from .preprocess import CropParams, get_white_border, get_black_border


###################################################################################

class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval

        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}

        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
            depth_gt = self.reader.open(depth_path)
            w, h = image.size

            if self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.dataset == 'nyu' and self.config.avoid_boundary:
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)


            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.config.dataset == 'nyu' or self.config.dataset == 'clearpose' or self.config.dataset == 'transcg':
                depth_gt = depth_gt / 1000.0
                # depth_gt = depth_gt / 4000.0
            else:
                depth_gt = depth_gt / 256.0

            if self.config.aug and (self.config.random_crop):
                print("use augmentation!")
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)
            
            if self.config.aug and self.config.random_translate:
                print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            image, depth_gt = self.train_preprocess(image, depth_gt)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, **sample}

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
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print('Missing gt for {}'.format(image_path))
                    assert False 

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.config.dataset == 'nyu' or self.config.dataset == 'clearpose' or self.config.dataset == 'transcg':
                        depth_gt = depth_gt / 1000.0
                        # depth_gt = depth_gt / 4000.0
                    else:
                        depth_gt = depth_gt / 256.0

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                else:
                    mask = False

            if self.config.do_kb_crop:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask}
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

    def __len__(self):
        return len(self.filenames)
    

class DataLoadPreprocessWithMask(DataLoadPreprocess):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        super().__init__(config, mode, transform, is_for_online_eval, **kwargs)

    def train_preprocess(self, image, depth_gt, depth_mask):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()
                depth_mask = (depth_mask[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt, depth_mask

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}

        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[0]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[1]))

            image = self.reader.open(image_path)
            depth_gt = self.reader.open(depth_path)
            
            # depth_mask_path = os.path.splitext(depth_path)[0] + "-mask.png"  # for transCG
            depth_mask_path=depth_path.split("-")[0]+"-label.png" # for clearpose
            print(depth_mask_path)

            depth_mask = self.reader.open(depth_mask_path)
            w, h = image.size

            if self.config.do_kb_crop:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                depth_mask = depth_mask.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.dataset == 'nyu' and self.config.avoid_boundary:
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_mask = depth_mask.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)

                depth_mask= np.array(depth_mask)
                depth_mask = np.pad(depth_mask, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)   
                depth_mask = Image.fromarray(image)


            if self.config.do_random_rotate and (self.config.aug):
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)
                depth_mask= self.rotate_image(
                    depth_mask, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_mask=np.expand_dims(depth_mask, axis=2)

            if self.config.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
                # depth_gt = depth_gt / 4000.0
            else:
                depth_gt = depth_gt / 256.0

            if self.config.aug and (self.config.random_crop):
                image, depth_gt = self.random_crop(
                    image, depth_gt, self.config.input_height, self.config.input_width)
            
            if self.config.aug and self.config.random_translate:
                # print("Random Translation!")
                image, depth_gt = self.random_translate(image, depth_gt, self.config.max_translation)

            image, depth_gt = self.train_preprocess(image, depth_gt, depth_mask)
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth).squeeze()[None, ...]
            sample = {'image': image, 'depth': depth_gt, 'focal': focal,
                      'mask': mask, **sample}

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
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1]))
                depth_mask_path = os.path.splitext(depth_path)[0] + "-mask.png"
                has_valid_depth = False
                try:
                    depth_gt = self.reader.open(depth_path)
                    depth_mask= self.reader.open(depth_mask_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    depth_mask = np.asarray(depth_mask, dtype=np.float32)
                    depth_mask=np.expand_dims(depth_mask, axis=2)
                    if self.config.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                        # depth_gt = depth_gt / 4000.0
                    else:
                        depth_gt = depth_gt / 256.0

                    mask = np.logical_and(
                        depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth)
                    # print(mask.shape, depth_mask.shape)
                    mask=np.logical_and(mask, depth_mask).squeeze()[None, ...]
                    
                else:
                    mask = False

            if self.config.do_kb_crop:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask}
            else:
                sample = {'image': image, 'focal': focal}

        if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
            mask = np.logical_and(depth_gt > self.config.min_depth,
                                  depth_gt < self.config.max_depth)
            mask=np.logical_and(mask, depth_mask).squeeze()[None, ...]
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}

        return sample

