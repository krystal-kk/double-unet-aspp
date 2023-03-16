import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numbers
import random


class RandCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img, bin_mask = sample['img'], sample['bin_mask']
        i, j, h, w = self.get_params(img, self.size)
        img1 = F.crop(img, i, j, h, w)
        bin_mask1 = F.crop(bin_mask, i, j, h, w)
        return {'img': img1, 'bin_mask': bin_mask1}

class RandCrop_cell_pred(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img, bin_mask, ins_mask = sample['img'], sample['bin_mask'], sample['ins_mask']
        i, j, h, w = self.get_params(img, self.size)
        img1 = F.crop(img, i, j, h, w)
        bin_mask1 = F.crop(bin_mask, i, j, h, w)
        ins_mask1 = ins_mask[i:i+h, j:j+w]
        return {'img': img1, 'bin_mask': bin_mask1, 'ins_mask': ins_mask1}


class RandHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, bin_mask = sample['img'], sample['bin_mask']
            img1 = F.hflip(img)
            bin_mask1 = F.hflip(bin_mask)
            return {'img': img1, 'bin_mask': bin_mask1}
        return sample


class RandVerticalFlip(T.RandomVerticalFlip):
    def forward(self, sample):
        if torch.rand(1) < self.p:
            img, bin_mask = sample['img'], sample['bin_mask']
            img1 = F.vflip(img)
            bin_mask1 = F.vflip(bin_mask)
            return {'img': img1, 'bin_mask': bin_mask1}
        return sample



class RandRotate(object):
    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0,3)
        angle = idx * 90
        return angle

    def __call__(self, sample):
        img, bin_mask = sample['img'], sample['bin_mask']
        angle = self.get_params()
        img1 = F.rotate(img, angle, self.resample, self.expand, self.center)
        bin_mask1 = F.rotate(bin_mask, angle, self.resample, self.expand, self.center)
        return {'img': img1, 'bin_mask': bin_mask1}


class ToTensor(object):
    def __call__(self, sample):
        img, bin_mask = sample['img'], sample['bin_mask']
        img1 = F.to_tensor(img)
        bin_mask1 = F.to_tensor(bin_mask)
        return {'img': img1, 'bin_mask': bin_mask1}

class ToTensor_R(object):
    def __call__(self, sample):
        img, bin_mask, ins_mask = sample['img'], sample['bin_mask'], sample['ins_mask']
        img1 = F.to_tensor(img)
        bin_mask1 = F.to_tensor(bin_mask)
        return {'img': img1, 'bin_mask': bin_mask1, 'ins_mask':ins_mask}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, bin_mask = sample['img'], sample['bin_mask']
        img1 = F.normalize(img, self.mean, self.std)
        return {'img': img1, 'bin_mask': bin_mask}
