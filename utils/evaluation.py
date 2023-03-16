import os
import pdb
import numpy as np
import pandas as pd
from PIL import Image
from config import parse_args


def dice_coeff(im1, im2, empty_score=1.0):
    """Calculates the dice coefficient for the images"""

    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im1 = im1 > 0.5
    im2 = im2 > 0.5

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print(im_sum)

    return 2. * intersection.sum() / im_sum


def numeric_score(prediction, groundtruth):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = float(np.sum((prediction == 0) & (groundtruth == 0)))

    return FP, FN, TP, TN



'''comput f1 score'''
def compute_f1(name_list, pred_path, mask_path):
    f1 = []
    for name in name_list:
        pred_i = os.path.join(pred_path, name+'.png')
        mask_i = os.path.join(mask_path, name+'.png')
        pred = np.asarray(Image.open(pred_i).convert('L'), dtype=bool)
        mask = np.asarray(Image.open(mask_i).convert('L'), dtype=bool)
        FP, FN, TP, TN = numeric_score(pred, mask)
        f1_score = TP / (TP + 0.5 * (FP + FN))
        f1.append(f1_score)
    mean_f1 = sum(f1) / len(f1)
    print('mean_f1: ', mean_f1 * 100.0)

    content = name_list.copy()
    content.append('mean_f1')
    f1.append(mean_f1)
    df = pd.DataFrame({'img_id':content, 'f1':f1})
    df.to_csv(pred_path + '/scores/f1.csv', index=False)



'''compute accuracy score'''
def compute_accuracy(name_list, pred_path, mask_path):
    acc_scores = []
    for name in name_list:
        pred_i = os.path.join(pred_path, name+'.png')
        mask_i = os.path.join(mask_path, name+'.png')
        pred = np.asarray(Image.open(pred_i).convert('L'), dtype=bool)
        mask = np.asarray(Image.open(mask_i).convert('L'), dtype=bool)
        FP, FN, TP, TN = numeric_score(pred, mask)
        N = FP + FN + TP + TN
        accuracy = np.divide(TP + TN, N)
        acc_scores.append(accuracy)
    mean_acc = sum(acc_scores) / len(acc_scores)
    print('mean_acc: ', mean_acc * 100.0)

    content = name_list.copy()
    content.append('mean_acc')
    acc_scores.append(mean_acc)
    df = pd.DataFrame({'img_id':content, 'accuracy':acc_scores})
    df.to_csv(pred_path + '/scores/accuracy.csv', index=False)    



'''compute iou score'''
def compute_iou(name_list, pred_path, mask_path):
    """
    compute IOU between two combined masks, this does not follow kaggle's evaluation
    :return: IOU, between 0 and 1
    """
    ious = []
    for name in name_list:
        pred_i = os.path.join(pred_path, name+'.png')
        mask_i = os.path.join(mask_path, name+'.png')
        pred = np.asarray(Image.open(pred_i).convert('L'), dtype=bool)
        mask = np.asarray(Image.open(mask_i).convert('L'), dtype=bool)
        union = np.sum(np.logical_or(mask, pred))
        intersection = np.sum(np.logical_and(mask, pred))
        iou = intersection/union
        ious.append(iou)
    mean_iou = sum(ious) / len(ious)
    print('mean_iou: ', mean_iou * 100.0)

    content = name_list.copy()
    content.append('mean_iou')
    ious.append(mean_iou)
    df = pd.DataFrame({'img_id':content, 'iou':ious})
    df.to_csv(pred_path + '/scores/iou.csv', index=False)



'''compute precision score'''
def compute_precision(name_list, pred_path, mask_path):
    precisions = []
    for name in name_list:
        pred_i = os.path.join(pred_path, name+'.png')
        mask_i = os.path.join(mask_path, name+'.png')
        pred = np.asarray(Image.open(pred_i).convert('L'), dtype=bool)
        mask = np.asarray(Image.open(mask_i).convert('L'), dtype=bool)
        FP, FN, TP, TN = numeric_score(pred, mask)
        precision_score = TP / (TP + FP)
        precisions.append(precision_score)
    mean_precision = sum(precisions) / len(precisions)
    print('mean_precision: ', mean_precision * 100.0)

    content = name_list.copy()
    content.append('mean_precision')
    precisions.append(mean_precision)
    df = pd.DataFrame({'img_id':content, 'precision':precisions})
    df.to_csv(pred_path + '/scores/precision.csv', index=False)
    


'''compute recall score'''
def compute_recall(name_list, pred_path, mask_path):
    recalls = []
    for name in name_list:
        pred_i = os.path.join(pred_path, name+'.png')
        mask_i = os.path.join(mask_path, name+'.png')
        pred = np.asarray(Image.open(pred_i).convert('L'), dtype=bool)
        mask = np.asarray(Image.open(mask_i).convert('L'), dtype=bool)
        FP, FN, TP, TN = numeric_score(pred, mask)
        recall_score = TP / (TP + FN)
        recalls.append(recall_score)
    mean_recall = sum(recalls) / len(recalls)
    print('mean_recall: ', mean_recall * 100.0)

    content = name_list.copy()
    content.append('mean_recall')
    recalls.append(mean_recall)
    df = pd.DataFrame({'img_id':content, 'recall':recalls})
    df.to_csv(pred_path + '/scores/recall.csv', index=False)





if __name__ == '__main__':
    args = parse_args()
    epoch = 'best'
    
    if args.model_detail:
        pred_path = os.path.join(args.save_result, args.model_name+'_'+args.model_detail+'_'+args.loss_type, epoch)
    else:
        pred_path = os.path.join(args.save_result, args.model_name+'_'+args.loss_type, epoch)

    mask_path = args.test_bin_masks_dir

    name_list = []
    allname = os.listdir(mask_path)
    for name in allname:
        name_list.append(name.split('.')[0])

    if not os.path.exists(os.path.join(pred_path, 'scores')):
        os.makedirs(os.path.join(pred_path, 'scores'))


    compute_f1(name_list, pred_path, mask_path)
    compute_accuracy(name_list, pred_path, mask_path)
    compute_iou(name_list, pred_path, mask_path)
    compute_precision(name_list, pred_path, mask_path)
    compute_recall(name_list, pred_path, mask_path)




