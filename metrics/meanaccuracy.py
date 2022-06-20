import torch
import numpy as np


def get_accuracy(SR, GT, threshold=0.5):
    """
    cal each img accuracy
    """
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # TN : True Negative
    # FP : False Positive
    # FN : False Negative
    TP = ((SR == 1) + (GT == 1)) == 2
    TN = ((SR == 0) + (GT == 0)) == 2
    FP = ((SR == 1) + (GT == 0)) == 2
    FN = ((SR == 0) + (GT == 1)) == 2

    AC = float(torch.sum(TP + TN)) / (float(torch.sum(TP + TN + FP + FN)) + 1e-6)

    return AC


def meanaccuracy_seg(pred, gt, accs):
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
        acc = get_accuracy(pred[x], gt_tensor[x])
        accs.append(acc)
    return accs