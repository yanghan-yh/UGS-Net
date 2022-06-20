import os

from numpy.core.numeric import False_

class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        self.arch = 'mymodel_fold1'
        # training settings
        self.epochs = 150
        self.learning_rate = 0.00001
        self.gpu = 0
        self.evaluate = False # test or train
        self.resume = False
        self.num_classes = 3
        self.in_dim = 3
        self.out_dim = 1
        self.lr_type = 'SGDR'
        self.milestones = [80, 160, 240]
        self.sgdr_t = 50
        self.weight_seg1 = 1
        self.weight_seg2 = 0.5
        self.weight_con = 0.1
        self.weight_im = 0.9
        self.weight_kd = 0.5
        self.weight_edge = 0.5
        self.batch_size = 8

        # cross validation settings
        self.fold = 1
        self.fold_num = 5

        self.training_fold_index = []
        for i in range(self.fold_num + 1):
            if i != self.fold and i != 0:
                self.training_fold_index.append(i)

        self.test_fold_index = [self.fold]

        # paths 
        self.maskPath1 = './data/row_image/' # 对比实验所用数据集
        self.csvPath = './dataprocess/split_csv/' # fold_csv 
        self.maskPath2 = './data/mid_image/'
        self.midPath = './data/mid_image_npy/'
        self.lungPath = './data/lung_image/'
        self.mediaPath = './data/media_image/'
        self.figurePath = './result/figure/'


