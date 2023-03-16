from __future__ import print_function, division
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy

import matplotlib.pyplot as plt
import numpy as np

# cfg = parse_args()
# img_h = cfg.img_h
# img_w = cfg.img_w

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda else "cpu")


# new
def dice_loss(input, target, eps=1e-7, if_sigmoid=True):
    if if_sigmoid:
        input = F.sigmoid(input)
    b = input.shape[0]
    iflat = input.contiguous().view(b, -1)
    tflat = target.float().contiguous().view(b, -1)
    intersection = (iflat * tflat).sum(dim=1)
    L = (1 - ((2. * intersection + eps) / (iflat.pow(2).sum(dim=1) + tflat.pow(2).sum(dim=1) + eps))).mean()
    return L

def smooth_truncated_loss(p, t, ths=0.06, if_reduction=True, if_balance=True):
    n_log_pt = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    pt = (-n_log_pt).exp()
    L = torch.where(pt>=ths, n_log_pt, -math.log(ths)+0.5*(1-pt.pow(2)/(ths**2)))
    if if_reduction:
        if if_balance:
            return 0.5*((L*t).sum()/t.sum().clamp(1) + (L*(1-t)).sum()/(1-t).sum().clamp(1))
        else:
            return L.mean()
    else:
        return L

def balance_bce_loss(input, target):
    L0 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return 0.5*((L0*target).sum()/target.sum().clamp(1)+(L0*(1-target)).sum()/(1-target).sum().clamp(1))

def compute_loss_list(loss_func, pred=[], target=[], **kwargs):
    losses = []
    for ipred, itarget in zip(pred, target):
        losses.append(loss_func(ipred, itarget, **kwargs))
    return losses





################################################
# original
# def dice_loss(prediction, target):
#     """Calculating the dice loss
#     Args:
#         prediction = predicted image
#         target = Targeted image
#     Output:
#         dice_loss"""

#     smooth = 1.0

#     i_flat = prediction.view(-1)
#     t_flat = target.view(-1)

#     intersection = (i_flat * t_flat).sum()

#     return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss




def my_stain_loss(raw, prediction, target):
    #bce = bce.cpu()
    #b = bce.detach().numpy()
    h0 = np.array(raw.cpu()[0, 0, :, :])
    g0 = np.array(target.cpu()[0, 0, :, :])
    h1 = 1 - h0
    h2 = g0 - h1
    h3 = 1 - g0
    h4 = h0 * h3

    myw=np.sqrt(np.power(h2,2)+np.power(h4,2))
    imbalance_w = imbalance_weight(target)
    w = myw + imbalance_w

    w = from_numpy(w)
    w = w.to(device)

    bce = F.binary_cross_entropy_with_logits(prediction, target, weight=w)
    loss = bce
    return loss

def imbalance_weight(mask):
    ibw = np.zeros([img_h, img_w])
    mask=np.array(mask.cpu()[0, 0, :, :])
    fore_count=np.count_nonzero(mask)
    back_count=img_h*img_w-fore_count
    fw = 1/fore_count
    bw = 5/back_count
    norm=max(fw,bw)
    fw=fw/norm
    bw=bw/norm
    ibw[mask==0] = bw
    ibw[mask==1] = fw
    return ibw

def bound_loss(prediction,target,bce_weight=0.5):
    merge=target.cpu()
    mask=np.array(merge[0][0])
    thre=0.2
    mask[mask>thre]=1
    mask[mask<=thre]=0
    ibw=imbalance_weight(mask)
    mask=mask[np.newaxis,np.newaxis,:]
    c_map_1=np.array(merge[0][1],dtype='float32')
    c_map=np.sqrt(np.power(c_map_1,2)+np.power(ibw,2))*255
    mask=from_numpy(mask)
    c_map=from_numpy(c_map)
    mask=mask.to(device)
    c_map=c_map.to(device)

    bce = F.binary_cross_entropy_with_logits(prediction, mask,weight=c_map)
    loss=bce

    return loss


def emap_loss(raw, prediction, target, mask, likelihood):

    likelihood = np.array(likelihood.cpu()[0, 0, :, :])
    weights = likelihood + imbalance_weight(mask) + 1
    weights = from_numpy(weights)
    weights = weights.to(device)

    loss = nn.BCEWithLogitsLoss(reduction='none')
    out = loss(prediction, target)
    result = (out * weights).sum()
    result = result/(512*512)

    return result

def compare_loss(prediction, target):

    bce = F.binary_cross_entropy_with_logits(prediction, target)
    loss = bce

    return loss

def dist_loss(predict, target):
    #L = F.mse_loss(predict,target)
    L = torch.mean(torch.pow((predict - target), 2))
    return L



