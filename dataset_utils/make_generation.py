import numpy as np
from typing import List
import cv2
import os

def read_npys(dir: str) -> List:
    npys = os.listdir(dir)
    return npys

def convert_to_binary_mask(p_dir, out_dir, npy_list):
    for name in npy_list:
        npy = np.load(os.path.join(p_dir, name))
        npy[np.where(npy>0)] = 255
        cv2.imwrite(f"{out_dir}/{name.split('.')[0]}.png", npy)


def convert_to_instance_mask(p_dir, out_dir, npy_list):
    for name in npy_list:
        npy = np.load(os.path.join(p_dir, name))
        instance_img = np.zeros((npy.shape[0], npy.shape[1], 3), dtype=np.uint8)
        for val in np.unique(npy)[1:]:
            instance_img[np.where(npy==val)] = list(np.random.choice(range(256), size=3))
        cv2.imwrite(f"{out_dir}/{name.split('.')[0]}.png", instance_img)


def main():
    mask_dir = "/home/krystal/workspace/dataset/monuseg-3/test"
    size_mask = "ins_npy/256"
    npys_list = read_npys(os.path.join(mask_dir, size_mask))
    binary_save = os.path.join(mask_dir, 'bin_masks', '256')
    if not os.path.exists(binary_save):
        os.makedirs(binary_save, exist_ok=True)
    convert_to_binary_mask(os.path.join(mask_dir, size_mask), binary_save, npys_list)
    
    instance_save = os.path.join(mask_dir, 'ins_masks'+'256')
    if not os.path.exists(instance_save):
        os.makedirs(instance_save, exist_ok=True)
    convert_to_instance_mask(os.path.join(mask_dir, size_mask), instance_save, npys_list)


if __name__ == "__main__":
    main()
