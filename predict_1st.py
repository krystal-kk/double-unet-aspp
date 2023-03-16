import os
import pdb
import time
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from tensorboard import summary

import torch
import torch.nn
import torchvision
import torchsummary
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset_utils import Monuseg_Dataset, Cell
from utils.loss import dice_loss, smooth_truncated_loss, compare_loss, dist_loss
from models.models import U_Net, NestedUNet, AttU_Net, R2U_Net, DoubleUnet, UnetA
from models.unetaspp import UnetASPP, AttUnetAspp, UnetAsppA

from utils.vis_loss import draw_loss
from utils.save_load_weight import save_weight
from utils.util import get_name_list
from utils.visualize import visualize_instance_mask, visualize_bin
from tqdm import tqdm
import skimage.morphology as morph
from utils.metrics import compute_nmi, compute_iou, bin_metric, get_fast_aji

import cfg
from datetime import datetime

import warnings
import wandb
warnings.filterwarnings("ignore")


def load_dataset(dataset, pin_memory):
    test_loader = DataLoader(dataset,
                            batch_size=1, 
                            shuffle=False, 
                            num_workers=cfg.num_workers, 
                            pin_memory=pin_memory)
    return test_loader


def main():
    # GPU setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = False
    if torch.cuda.is_available():
        print('Training on GPU!')
        pin_memory = True
    else:
        print('Training on CPU!')

    wandb.init(project="nuclei_segmentation", entity="study-sync")
    wandb.log({'phase': 'test'})

    # set config
    train_list, test_list = get_name_list(cfg.train_csv, cfg.test_csv)
    cell_data = Cell(cfg.cell_img_dir, cfg.cell_bin_dir, cfg.cell_ins_dir, train_list, test_list)
    test_loader = load_dataset(cell_data.get_cell_predict_dataset(), pin_memory)
    # test_loader = load_dataset(cell_data.get_test_dataset(), pin_memory)

    # build model
    if cfg.model_name == 'Unet':
        model = U_Net(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == "NestedUnet":
        model = NestedUNet(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'AttentionUnet':
        model = AttU_Net(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'UnetAspp':
        model = UnetASPP(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'AttentionUnetAspp':
        model = AttUnetAspp(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'DoubleUnet':
        model = DoubleUnet()
    elif cfg.model_name == 'UnetA':
        model = UnetA(cfg.in_channel, cfg.out_channel)
    elif cfg.model_name == 'UnetAsppA':
        model = UnetAsppA(cfg.in_channel, cfg.out_channel)

    model = model.to(device)

    if cfg.model_detail:
        weight_path = os.path.join(cfg.save_model, cfg.model_name+'_'+cfg.model_detail+'_'+cfg.loss_type, 'model_{}.pth').format(cfg.load_epoch)
    else:
        weight_path = os.path.join(cfg.save_model, cfg.model_name+'_'+cfg.loss_type, 'model_{}.pth').format(cfg.load_epoch)

    assert os.path.exists(weight_path), "file: '{}' does not exist.".format(weight_path)
    model.load_state_dict(torch.load(weight_path, map_location=device)['state_dict'], strict=False)

    model.eval()

    img_list = []
    bin_mask_list = []
    bin_pred_list = []
    ins_mask_list = []
    ins_pred_list = []
    for i, (img, bin_mask, ins_mask) in tqdm(enumerate(test_loader)):
        img = img.to(device)
        # pred = model(img).cpu()
        x, pred, _ = model(img)
        pred = pred.cpu()



        # pred = torch.sigmoid(pred)
        pred = (pred > 0.5)
        # squeeze the dimesion
        bin_pred = pred.squeeze().detach().numpy().astype(np.uint8)
        # TODO watershed

        ins_pred = morph.label(bin_pred, connectivity=1)

        bin_mask = bin_mask.squeeze().detach().numpy().astype(np.uint8)
        ins_mask = ins_mask.squeeze().detach().numpy().astype(int)

        # in cell-dataset_predict, we should collect all imgs, bin_masks, ins_masks
        # combine them and remove the paded area
        img_list.append(img.permute(0,2,3,1).squeeze().cpu().detach().numpy()*255)
        bin_mask_list.append(bin_mask)
        ins_mask_list.append(ins_mask)
        bin_pred_list.append(bin_pred)
        ins_pred_list.append(ins_pred)
    

    """normal tests
    """
    # show_result(img_list, bin_mask_list, ins_mask_list, bin_pred_list, ins_pred_list, length=len(test_loader))


    """re_merge the images
    """

    ori_img_list = cell_data._merge_back_and_cut_pad(img_list)

    ori_bin_list = cell_data._merge_back_and_cut_pad(bin_mask_list)

    ori_ins_list = cell_data._merge_back_and_cut_pad(ins_mask_list)

    pred_bin_list = cell_data._merge_back_and_cut_pad(bin_pred_list)

    # pred_ins_list = cell_data._merge_back_and_cut_pad(ins_pred_list)
    pred_ins_list = [bin_to_ins(pred_bin_list[x]) for x in range(len(pred_bin_list))]


    show_result(ori_img_list, ori_bin_list, ori_ins_list, pred_bin_list, pred_ins_list, length=len(ori_img_list))


def bin_to_ins(bin_pred):
    ins_pred = morph.label(bin_pred, connectivity=1)
    return ins_pred

def show_result(ori_img_list, ori_bin_list, ori_ins_list, pred_bin_list, pred_ins_list, length):
    i = 0
    ious = []
    ajis = []
    f1s = []
    accs = []
    precs = []
    recalls = []
    nmis = []
    wandb.define_metric("predict_step")
    wandb.define_metric("acc", summary="mean", step_metric='predict_step')
    wandb.define_metric("precision", summary="mean", step_metric='predict_step')
    wandb.define_metric("recall", summary="mean", step_metric='predict_step')
    wandb.define_metric("f1", summary="mean", step_metric='predict_step')
    wandb.define_metric("iou", summary="mean", step_metric='predict_step')
    wandb.define_metric("nmi", summary="mean", step_metric='predict_step')
    wandb.define_metric("aji", summary="mean", step_metric='predict_step')

    if cfg.model_detail:
        result_dir = os.path.join(cfg.save_result, cfg.model_name+'_'+cfg.model_detail+'_'+cfg.loss_type, cfg.load_epoch)
    else:
        result_dir = os.path.join(cfg.save_result, cfg.model_name+'_'+cfg.loss_type, cfg.load_epoch)

    # Add time stamp
    result_dir += "-"+datetime.now().strftime("%m-%d-%H-%M-%S")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print('Successfully created results directory {}'.format(result_dir))

    for img, mask_bin, mask_ins, pred_bin, pred_ins in zip(ori_img_list, ori_bin_list, ori_ins_list, pred_bin_list, pred_ins_list):
        # cal bin metric
        bm = bin_metric(mask_bin, pred_bin)
        accs.append(bm.acc)
        precs.append(bm.precision)
        recalls.append(bm.recall)
        f1s.append(bm.f1)
        
        iou = compute_iou(pred_bin, mask_bin)
        ious.append(iou)
        
        # cal ins_metric
        nmi = compute_nmi(mask_ins.flatten(), pred_ins.flatten())
        nmis.append(nmi)

        aji = get_fast_aji(mask_ins, pred_ins)
        ajis.append(aji)
        log_dict = {
            'predict_step': i,
            'acc': bm.acc,
            'precision': bm.precision,
            'recall': bm.recall,
            'f1': bm.f1,
            'iou': iou,
            # 'nmi': nmi,
            # 'aji': aji
        }
        wandb.log(log_dict)

        if cfg.visualize:
            ins_save_dir = os.path.join(result_dir, 'instance')
            os.makedirs(ins_save_dir, exist_ok=True)
            bin_save_dir = os.path.join(result_dir, 'bins')
            os.makedirs(bin_save_dir, exist_ok=True)
            # visualize_instance_mask(pred_ins, mask_ins, img, name=str(i), savedir=ins_save_dir)
            visualize_bin(pred_bin, mask_bin, img, name=str(i), savedir=bin_save_dir)
        i += 1

    print('*** result ***')
    print('mean aji: ', np.sum(np.array(ajis)) / length)
    print('mean iou: ', np.sum(np.array(ious)) / length)
    print('mean nmi: ', np.sum(np.array(nmis)) / length)
    print('mean acc: ', np.sum(np.array(accs)) / length)
    print('mean precision: ', np.sum(np.array(precs)) / length)
    print('mean recall: ', np.sum(np.array(recalls)) / length)
    print('mean f1: ', np.sum(np.array(f1s)) / length)


if __name__ == "__main__":
    main()
