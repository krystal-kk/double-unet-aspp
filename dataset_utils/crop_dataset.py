import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm


def crop_512(name, base_path, save_path):
    im = Image.open(os.path.join(base_path, name))
    base_name = name.split('.')[0]
    
    lt = (0, 0, 512, 512)
    rt = (488, 0, 1000, 512)
    lb = (0, 488, 512, 1000)
    rb = (488, 488, 1000, 1000)
    
    img1 = im.crop(lt)
    img2 = im.crop(rt)
    img3 = im.crop(lb)
    img4 = im.crop(rb)
    
    img1.save(os.path.join(save_path, base_name+'_1'+'.png'))
    img2.save(os.path.join(save_path, base_name+'_2'+'.png'))
    img3.save(os.path.join(save_path, base_name+'_3'+'.png'))
    img4.save(os.path.join(save_path, base_name+'_4'+'.png'))


def crop_512_mask(name, base_path, save_path):
    im = np.load(os.path.join(base_path, name))
    base_name = name.split('.')[0]
    
    lt = (0, 0, 512, 512)
    rt = (488, 0, 1000, 512)
    lb = (0, 488, 512, 1000)
    rb = (488, 488, 1000, 1000)
    
    img1 = im[lt[0]:lt[2], lt[1]:lt[3]]
    img2 = im[rt[0]:rt[2], rt[1]:rt[3]]
    img3 = im[lb[0]:lb[2], lb[1]:lb[3]]
    img4 = im[rb[0]:rb[2], rb[1]:rb[3]]
    
    np.save(os.path.join(save_path, base_name+'_1'+'.npy'), img1)
    np.save(os.path.join(save_path, base_name+'_2'+'.npy'), img2)
    np.save(os.path.join(save_path, base_name+'_3'+'.npy'), img3)
    np.save(os.path.join(save_path, base_name+'_4'+'.npy'), img4)


# 256 (512->256)
def crop_256(name, base_path, save_path):
    im = Image.open(os.path.join(base_path, name))
    base_name = name.split('.')[0]
    
    lt = (0, 0, 256, 256)
    rt = (256, 0, 512, 256)
    lb = (0, 256, 256, 512)
    rb = (256, 256, 512, 512)
    
    img1 = im.crop(lt)
    img2 = im.crop(rt)
    img3 = im.crop(lb)
    img4 = im.crop(rb)
    
    img1.save(os.path.join(save_path, base_name+'1'+'.png'))
    img2.save(os.path.join(save_path, base_name+'2'+'.png'))
    img3.save(os.path.join(save_path, base_name+'3'+'.png'))
    img4.save(os.path.join(save_path, base_name+'4'+'.png'))


def crop_256_mask(name, base_path, save_path):
    im = np.load(os.path.join(base_path, name))
    base_name = name.split('.')[0]
    
    lt = (0, 0, 256, 256)
    rt = (256, 0, 512, 256)
    lb = (0, 256, 256, 512)
    rb = (256, 256, 512, 512)
    
    img1 = im[lt[0]:lt[2], lt[1]:lt[3]]
    img2 = im[rt[0]:rt[2], rt[1]:rt[3]]
    img3 = im[lb[0]:lb[2], lb[1]:lb[3]]
    img4 = im[rb[0]:rb[2], rb[1]:rb[3]]
    
    np.save(os.path.join(save_path, base_name+'_1'+'.npy'), img1)
    np.save(os.path.join(save_path, base_name+'_2'+'.npy'), img2)
    np.save(os.path.join(save_path, base_name+'_3'+'.npy'), img3)
    np.save(os.path.join(save_path, base_name+'_4'+'.npy'), img4)


def main():
    dataset_dir = '/home/xuexi/workspace/datasets/monuseg/test'
    img_1000 = os.path.join(dataset_dir, "Tissue_Images")
    mask_1000 = os.path.join(dataset_dir, "1000_masks")
    img_512 = os.path.join(dataset_dir, "512_imgs")
    mask_512 = os.path.join(dataset_dir, "512_masks")
    img_256 = os.path.join(dataset_dir, "256_imgs")
    mask_256 = os.path.join(dataset_dir, "256_masks")
    # 1000 -> 512
    for name in tqdm(os.listdir(img_1000)):
        crop_512(name, img_1000, img_512)
    for name in tqdm(os.listdir(mask_1000)):
        crop_512_mask(name, mask_1000, mask_512)
    
    # 512 -> 256
    for name in tqdm(os.listdir(img_512)):
        crop_256(name, img_512, img_256)
    for name in tqdm(os.listdir(mask_512)):
        crop_256_mask(name, mask_512, mask_256)

if __name__ == "__main__":
    main()
