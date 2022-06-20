import torch
import numpy as np


def get_sensitivity(SR, GT, threshold=0.5):
    """
    cal each img sensitivity
    """
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)

    return SE


def meansensitivity_seg(pred, gt, sens):
    """
    :return save img' sensitivity values in sens
    """
    gt_tensor = gt
    gt_tensor = gt_tensor.cpu()
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    pred = pred.type(torch.LongTensor)
    # TO CPU
    # pred_np = pred.data.cpu().numpy()
    # gt = gt.data.cpu().numpy()
    for x in range(pred.size()[0]):
        sen = get_sensitivity(pred[x], gt_tensor[x])
        sens.append(sen)
    return sens