import os
from re import L
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

from PIL import Image
from skimage import io
import scipy.io as scio
from dataset_utils.weight_transforms import RandCrop, RandHorizontalFlip, RandRotate, RandVerticalFlip, ToTensor, Normalize, RandCrop_cell_pred, ToTensor_R
import numpy as np
import random
import pdb
from typing import List
from collections import namedtuple


# image transformation
TRANSFORM = {
    'train': T.Compose([
        RandCrop((320, 320)),
        RandHorizontalFlip(0.5),
        RandVerticalFlip(0.5),
        RandRotate(),
        ToTensor(),
    ]),
    'test': T.Compose([
        RandCrop((320, 320)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

TRANSFORM_CELL_PRED = T.Compose([
    ToTensor()
])

TRANSFORM_CELL_PRED_R = T.Compose([
    RandCrop_cell_pred((320, 320)),
    ToTensor_R()
])

PADREGION = namedtuple('PADREGION', ['h_start', 'h_end', 'w_start', 'w_end'])


class cell_dataset(Dataset):
    """cell dataset

    """
    def __init__(self, images_dir, bin_masks_dir, ins_masks_dir, weight_maps_dir, data_list, phase='train', transform = TRANSFORM) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.bin_masks_dir = bin_masks_dir
        self.ins_masks_dir = ins_masks_dir
        self.weight_maps_dir = weight_maps_dir
        self.phase = phase
        self.transform = transform
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.images_dir, self.data_list[i]+'.png'))
        binary_mask = Image.open(os.path.join(self.bin_masks_dir, self.data_list[i]+'.png'))
        ins_mask = np.load(os.path.join(self.ins_masks_dir, self.data_list[i]+'.npy'))
        weight_map = np.load(os.path.join(self.weight_maps_dir, self.data_list[i]+'.png'))
        if self.phase == 'train':
            ins_mask = ins_mask[:320, :320]
        if self.phase == 'check_test':
            sample = {'img': image, 'bin_mask': binary_mask, 'ins_mask': ins_mask}
            sample = TRANSFORM_CELL_PRED_R(sample)
            return sample['img'], sample['bin_mask'], sample['ins_mask']
        sample = {'img': image, 'bin_mask': binary_mask, 'weight': weight_map}
        sample = self.transform[self.phase](sample)
        return sample['img'], sample['bin_mask'], ins_mask, sample['weight']


class pred_cell_dataset(Dataset):
    def __init__(self, img_list, bin_list, ins_list, weight_list transform = TRANSFORM_CELL_PRED) -> None:
        super().__init__()
        self.img_list = img_list
        self.bin_list = bin_list
        self.ins_list = ins_list
        self.weight_list = weight_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, i):
        image = self.img_list[i]
        bin_mask = self.bin_list[i]
        ins_mask = self.ins_list[i]
        weight_map = self.weight_list[i]
        
        sample = self.transform({'img': image, 'bin_mask': bin_mask, 'weight': weight_map})
        return sample['img'], sample['bin_mask'], ins_mask, sample['weight']


class CellWeight:
    def __init__(self, imgs_dir, bin_dir, ins_dir, weight_dir, train_list, test_list, target_size=320) -> None:
        self.img_dir = imgs_dir
        self.bin_dir = bin_dir
        self.ins_dir = ins_dir
        self.weight_dir = weight_dir
        self.train_list = train_list
        self.test_list = test_list
        self.target_size = target_size
        

    def get_train_dataset(self):
        train_dataset = cell_dataset(self.img_dir, self.bin_dir, self.ins_dir, self.weight_dir, self.train_list, phase='train')
        return train_dataset
    
    def get_test_dataset(self):
        test_dataset = cell_dataset(self.img_dir, self.bin_dir, self.ins_dir, self.test_list, phase='check_test')
        return test_dataset
    
    def get_cell_predict_dataset(self):
        self.clip_test()
        test_dataset = pred_cell_dataset(self.augmented_test_img_list, self.augmented_test_bin_list, self.augmented_test_ins_list)
        return test_dataset

    def clip_test(self):
        self.augmented_test_img_list = []
        self.augmented_test_bin_list = []
        self.augmented_test_ins_list = []
        self.pad_point = []
        for name in self.test_list:
            # img
            img = Image.open(os.path.join(self.img_dir, name+'.png'))
            pad_img = self._augment_test_data(np.array(img), add_point=True)
            self.augmented_test_img_list.extend(self._get_subimgs_list(pad_img))
            # bin_mask
            bin_ = Image.open(os.path.join(self.bin_dir, name+'.png'))
            pad_bin = self._augment_test_data(np.array(bin_))
            self.augmented_test_bin_list.extend(self._get_subimgs_list(pad_bin))
            # ins_mask
            ins_ = np.load(os.path.join(self.ins_dir, name+'.npy'))
            pad_ins = self._augment_test_data(ins_)
            self.augmented_test_ins_list.extend(self._get_subimgs_list(pad_ins))

        
    def _get_subimgs_list(self, img):
        divided_imgs = []
        for i in range(3):
            for j in range(3):
                divided_imgs.append(img[i*self.target_size:(i+1)*self.target_size, j*self.target_size:(j+1)*self.target_size])
        return divided_imgs

    def _augment_test_data(self, img, aug_h=960, aug_w=960, add_point=False):
        h, w = img.shape[0], img.shape[1]
        if len(img.shape)==2 or img.shape[2]==1:
            canvas = np.full((aug_h, aug_w), 255, dtype=np.uint8)
        else:
            canvas = np.full((aug_h, aug_w, img.shape[2]), [255, 255, 255], dtype=np.uint8)

        # x_center = (aug_w-w) // 2
        x_center = 0
        # y_center = (aug_h-h) // 2
        y_center = 0
        canvas[y_center:y_center+h, x_center:x_center+w] = img
        if add_point:
            self.pad_point.append(PADREGION(y_center, y_center+h, x_center, x_center+w))
        return canvas

    def _merge_back_and_cut_pad(self, img_aray) -> List:
        # img_aray: np.ndarray [N*9, , 960, 3/-]
        demo_shape = img_aray[0].shape
        if len(demo_shape) == 3:
            canvas_demo = np.zeros((960, 960, 3), dtype=np.uint8)
        else:
            canvas_demo = np.zeros((960,960), dtype=np.uint8)
        origin_imgs = []
        for i in range(len(img_aray)//9):
            canvas = np.zeros_like(canvas_demo)
            for j in range(3):
                for k in range(3):
                    canvas[j*320:(j+1)*320, k*320:(k+1)*320] = img_aray[i*9+j*3+k]
            pad_reg = self.pad_point[i]

            origin_imgs.append(canvas[pad_reg.h_start:pad_reg.h_end, pad_reg.w_start:pad_reg.w_end])

        return origin_imgs


# monuseg dataset class 
class Monuseg_Dataset(Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        segs_dir = path of segmentation masks
    Output:
        a dictionary"""

    def __init__(self, images_dir, bin_masks_dir, ins_masks_dir, phase = 'train', transform = TRANSFORM):
        self.images_dir = images_dir
        self.bin_masks_dir = bin_masks_dir
        self.ins_masks_dir = ins_masks_dir
        self.phase = phase
        self.transform = transform
        
        self.images = sorted(os.listdir(self.images_dir))
        self.bin_masks = sorted(os.listdir(self.bin_masks_dir))
        self.ins_masks = sorted(os.listdir(self.ins_masks_dir))
        

    def __len__(self):
        return len(self.images)


    def __getitem__(self, i):
        image = Image.open(os.path.join(self.images_dir, self.images[i]))
        binary_mask = Image.open(os.path.join(self.bin_masks_dir, self.bin_masks[i]))
        ins_mask = np.load(os.path.join(self.ins_masks_dir, self.ins_masks[i]))
        sample = {'img': image, 'bin_mask': binary_mask}
        transform = self.transform[self.phase]
        sample = transform(sample)

        return sample['img'], sample['bin_mask'], ins_mask


if __name__ == '__main__':
    train_images_dir = '/home/krystal/workspace/dataset/MoNuSeg/train/H_rgb/512'
    train_bin_masks_dir = '/home/krystal/workspace/dataset/MoNuSeg/train/GT_bin/512'
    test_images_dir = '/home/krystal/workspace/dataset/MoNuSeg/test/H_rgb/512'
    test_bin_masks_dir = '/home/krystal/workspace/dataset/MoNuSeg/test/GT_bin/512'

    train_dataset = Monuseg_Dataset(train_images_dir, train_bin_masks_dir, phase = 'train', transform = TRANSFORM)
    test_dataset = Monuseg_Dataset(test_images_dir, test_bin_masks_dir, phase = 'test', transform = TRANSFORM)

    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset[0])
