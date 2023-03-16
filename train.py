import os
import pdb
import time
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn
import torchvision
import torchsummary
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from utils.util import display
from dataset_utils import Monuseg_Dataset, Cell
from dataset_utils import CellWeight
from utils.loss import dice_loss, smooth_truncated_loss, compare_loss, dist_loss
from models.models import U_Net, NestedUNet, AttU_Net, R2U_Net, DoubleUnet, UnetA
from models.unetaspp import UnetASPP, AttUnetAspp, NestedUnetASPP
from utils.vis_loss import draw_loss
from utils.save_load_weight import save_weight
import wandb
import cfg

import warnings
warnings.filterwarnings("ignore")


def get_train_dataloader(train_dataset, pin_memory):
    # train_data = Monuseg_Dataset(cfgs.train_images_dir, 
    #                             cfgs.train_bin_masks_dir,
    #                             cfgs.train_ins_npy_dir,
    #                             phase='train')

    train_loader = DataLoader(train_dataset,
                            batch_size=cfg.batch_size, 
                            shuffle=True, 
                            num_workers=cfg.num_workers, 
                            pin_memory=pin_memory)
    return train_loader

def get_test_dataloader(test_datset, pin_memory):
    # test_data = Monuseg_Dataset(cfgs.test_images_dir,
    #                             cfgs.test_bin_masks_dir,
    #                             cfgs.test_ins_npy_dir,
    #                             phase='test')
    test_loader = DataLoader(test_datset, 
                            batch_size=1, 
                            shuffle=False,
                            num_workers=1, 
                            pin_memory=pin_memory)
    return test_loader


def run_one_epoch(model, phase, data_loader, optimizer, device):
    if phase == 'train':
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    length = 0

    if phase == 'train':
        for img, bin_mask, _ in data_loader:
            img, bin_mask = img.to(device), bin_mask.to(device)

            optimizer.zero_grad()
            bin_pred = model(img)
            # Todo modify loss
            if cfg.use_weight == False:
                loss = compare_loss(bin_pred, bin_mask)
            elif cfg.use_weight == True:
                pass
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            length += img.size(0)
        epoch_loss = running_loss / length
        print('train loss: {}'.format(epoch_loss))
    else:
        with torch.no_grad():
            for img, bin_mask, _ in data_loader:
                img, bin_mask = img.to(device), bin_mask.to(device)
                pred_pred = model(img)
                # Todo modify loss
                if cfg.use_weight == False:
                    loss = compare_loss(bin_pred, bin_mask)
                elif cfg.use_weight == True:
                    pass
                running_loss += loss.item()
                length += img.size(0)
            epoch_loss = running_loss / length
            print('test loss: {}'.format(epoch_loss))
    return epoch_loss


def get_name_list(train_df, test_df):
    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    train_list = train_df['train'].tolist()
    test_list = test_df['test'].tolist()
    return train_list, test_list


def main():
    wandb.init(project="nuclei_segmentation", entity="study-sync")
    # GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = False
    if torch.cuda.is_available():
        print('Training on GPU!')
        pin_memory = True
    else:
        print('Training on CPU!')

    # load data
    train_list, test_list = get_name_list(cfg.train_csv, cfg.test_csv)
    cell_data = Cell(cfg.cell_img_dir, cfg.cell_bin_dir, cfg.cell_ins_dir, train_list, test_list)
    train_loader = get_train_dataloader(cell_data.get_train_dataset(), pin_memory)
    test_loader = get_test_dataloader(cell_data.get_test_dataset(), pin_memory)

    # build model
    if cfg.model_name == 'Unet':
        model = U_Net(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == "NestedUnet":
        model = NestedUNet(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'AttentionUnet':
        model = AttU_Net(cfg.in_channel, cfg.out_channel)
    # --- add aspp ---
    elif cfg.model_name == 'UnetAspp':
        model = UnetASPP(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'AttentionUnetAspp':
        model = AttUnetAspp(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'NestedUnetASPP':
        model = NestedUnetASPP(cfg.in_channel, cfg.out_channel)

    model = model.to(device)
    torchsummary.summary(model, input_size=(cfg.in_channel, cfg.img_h, cfg.img_w))

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.init_lr, [cfg.beta1, cfg.beta2])
    #set criterion

    # create model weight if not exist
    if cfg.model_detail:
        weight_dir = os.path.join(cfg.save_model, cfg.model_name+'_'+cfg.model_detail+'_'+cfg.loss_type)
    else:
        weight_dir = os.path.join(cfg.save_model, cfg.model_name+'_'+cfg.loss_type)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    wandb.log({"model_detail": cfg.model_detail})

    #* start training model
    print('Starting training...')
    train_loss = []
    test_loss = []
    for epoch in range(1, cfg.num_epoch+1):
        print('-'*10)
        print('Epoch: {}/{}'.format(epoch, cfg.num_epoch))
        # train
        train_epoch_loss = run_one_epoch(model, phase='train', data_loader=train_loader, optimizer=optimizer, device=device)
        train_loss.append(train_epoch_loss)
        wandb.log({"train_loss": train_epoch_loss})
        # test
        test_epoch_loss = run_one_epoch(model, phase='test', data_loader=test_ader, optimizer=optimizer, device=device)
        test_loss.append(test_epoch_loss)
        wandb.log({"test_loss": test_epoch_loss})

        np.savetxt(os.path.join(weight_dir, 'train_loss.txt'), train_loss, fmt='%.6f')
        np.savetxt(os.path.join(weight_dir, 'test_loss.txt'), test_loss, fmt='%.6f')

        save_weight(os.path.join(weight_dir, 'model_last.pth'), epoch, model)
        if test_epoch_loss == min(test_loss):
            save_weight(os.path.join(weight_dir, 'model_best.pth'), epoch, model)
        if epoch % 10 == 0 or epoch == 1:
            save_weight(os.path.join(weight_dir, 'model_{}.pth'.format(epoch)), epoch, model)


# draw train and test loss
# draw_loss(train_loss, test_loss, weight_dir)
if __name__ == "__main__":
    main()
