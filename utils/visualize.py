from curses import noecho
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os


SAVE_DIR = '/home/krystal/workspace/experiment/monuseg/nuclei-segmentation/viz'

def get_color_ins(ins_pred):
    ins_img = np.zeros((ins_pred.shape[0], ins_pred.shape[1], 3), dtype=np.uint8)
    for col_index in np.unique(ins_pred)[1:]:
        color = list(np.random.choice(range(256), size=3))
        ins_img[np.where(ins_pred==col_index)] = color

    return ins_img
    
def visualize_instance_mask(ins_pred, ins_mask=None, img=None, save=True, name="test", savedir=None):
    if savedir is None:
        save_dir = os.path.join(SAVE_DIR, 'instance')
    else:
        save_dir = savedir
    os.makedirs(save_dir, exist_ok=True)
    color_pred = get_color_ins(ins_pred)
    if ins_mask is not None:
        color_mask = get_color_ins(ins_mask)
    
    if ins_mask is None:
        plt.figure(figsize=(6,6))
        plt.imshow(color_pred)
        plt.savefig(f"{save_dir}/{name}")
    if img is None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(color_mask)
        plt.subplot(122)
        plt.imshow(color_pred)
        plt.savefig(f"{save_dir}/{name}")
    else:
        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(color_mask)
        plt.subplot(133)
        plt.imshow(color_pred)
        plt.savefig(f"{save_dir}/{name}")

    
def visualize_bin(bin_pred, bin_mask=None, img=None, save=True, name="bin_test", savedir=None):
    if savedir is None:
        save_dir = os.path.join(SAVE_DIR, 'bin')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = savedir
    
    if bin_mask is None:
        plt.figure(figsize=(6,6))
        plt.imshow(bin_pred)
        plt.savefig(f"{save_dir}/{name}")
    if img is None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(bin_mask)
        plt.subplot(122)
        plt.imshow(bin_pred)
        plt.savefig(f"{save_dir}/{name}")
    else:
        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(bin_mask)
        plt.subplot(133)
        plt.imshow(bin_pred)
        plt.savefig(f"{save_dir}/{name}")
