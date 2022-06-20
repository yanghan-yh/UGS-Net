import numpy as np
import torch


def meandice(pred, gt, dices):
    """
    :return save img' dice value in IoUs
    """
    # dices = []
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1
    pred = pred.type(torch.LongTensor)
    pred_np = pred.data.cpu().numpy()
    gt = gt.data.cpu().numpy()
    for x in range(pred.size()[0]):
        dice = np.sum(pred_np[x][gt[x] == 1]) * 2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
        dices.append(dice)
    return dices
