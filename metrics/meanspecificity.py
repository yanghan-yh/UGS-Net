import torch
import numpy as np


def get_specificity(SR, GT, threshold=0.5):
    """
    cal each img specificity
    所有负例中被分对的概率
    结节在在输入图片中所占比例较少 所以该指标的值很高
    """
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    # TP = ((SR == 1) + (GT == 1)) == 2
    # FN = ((SR == 0) + (GT == 1)) == 2
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2

    SE = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SE


def meanspecificity_seg(pred, gt, spes):
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
        spe = get_specificity(pred[x], gt_tensor[x])
        spes.append(spe)
    return spes