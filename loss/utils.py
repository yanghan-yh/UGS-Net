import torch


def to_float_and_cuda(input):
    # input = input.type(torch.FloatTensor)
    # input = input.type(torch.LongTensor)
    input = input.float()
    input = input.cuda()
    return input


def to_long_and_cuda(input):
    # input = input.type(torch.FloatTensor)
    # input = input.type(torch.LongTensor)
    input = input.long()
    input = input.cuda()
    return input