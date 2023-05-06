import torch.nn as nn
import torch

loss_function = torch.nn.BCELoss()

def BCE_loss(input, target):
    # input = input.cuda()
    # target = target.cuda()
    return loss_function(input, target.float().cuda())