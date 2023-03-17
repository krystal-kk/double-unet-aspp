# image info
img_h = 320
img_w = 320
in_channel = 3
out_channel = 1

# model info
model_name = 'Unet'
model_detail = '200'
log_detail = "train"
loss_type = 'BEC'

# training setting
pretrained_1st = False    # pretrained_weight in the frist stage
pretrained_2nd = False    # pretrained_weight in the second stage
num_epoch = 200
load_epoch = 'best'
num_epochs_decay = 70
batch_size = 8
num_workers = 0

use_weight = False
init_lr = 1.25e-4
beta1 = 0.5
beta2 = 0.999
ngpus = 0

# monuseg_data
base_dir = '/home/krystal/workspace/dataset'
save_model = '/home/krystal/workspace/experiment/checkpoints'
save_result = '/home/krystal/workspace/experiment/results'

train_images_dir='/home/krystal/workspace/dataset/monuseg-3/train/ori_img/256'
train_bin_masks_dir='/home/krystal/workspace/dataset/monuseg-3/train/bin_masks/256'
train_ins_masks_dir='/home/krystal/workspace/dataset/monuseg-3/train/ins_masks/256'
train_ins_npy_dir='/home/krystal/workspace/dataset/monuseg-3/train/ins_npy/256'
train_weight_map_dir = ''

test_images_dir='/home/krystal/workspace/dataset/monuseg-3/test/ori_img/256'
test_bin_masks_dir='/home/krystal/workspace/dataset/monuseg-3/test/bin_masks/256'
test_ins_masks_dir='/home/krystal/workspace/dataset/monuseg-3/test/ins_masks/256'
test_ins_npy_dir='/home/krystal/workspace/dataset/monuseg-3/test/ins_npy/256'
visualize = True

# erihc_data
cell_img_dir = '/home/krystal/workspace/dataset/er-ihc/images'
cell_bin_dir = '/home/krystal/workspace/dataset/er-ihc/binary_masks'
cell_ins_dir = '/home/krystal/workspace/dataset/er-ihc/ins_npy'
train_csv = '/home/krystal/workspace/dataset/er-ihc/train.csv'
test_csv = '/home/krystal/workspace/dataset/er-ihc/test.csv'
