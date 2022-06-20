import torch.nn.functional as F
from torch import nn
import torch

def ContrastiveLoss(output1, output2):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
    return loss_contrastive