import torch
from .utils import *


def dice_loss(input, target):
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
    num = torch.sum(num, dim=2)  # 在dim维度上求和 维度减1 如果想要保留原始维度 使用keepdim=True
    num = torch.sum(num, dim=2)

    den1 = input * input
    den1 = torch.sum(den1, dim=2)
    den1 = torch.sum(den1, dim=2)

    den2 = target * target
    den2 = torch.sum(den2, dim=2)
    den2 = torch.sum(den2, dim=2)

    dice = 2 * (num / (den1 + den2)) + 1e-6
    dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batchsize

    return dice_total
