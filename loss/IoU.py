import torch
from .utils import *
import numpy as np


def IoU_loss(input, target, threshold=0.5):
    """
    2d dice loss
    :param input: predict tensor
    :param target: target tensor
    :return: scalar loss value
    """
    
    input = input > 0.5
    target = target == torch.max(target)

    input = to_float_and_cuda(input)
    target = to_float_and_cuda(target)
    num = input * target
    num = torch.sum(num, dim=2)
    num = torch.sum(num, dim=2)

    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    iou =  num / (den1 + den2 - num) + 1e-6
    iou_total = 1 - 1 * torch.sum(iou) / iou.size(0)  # divide by batchsize
    return iou_total
