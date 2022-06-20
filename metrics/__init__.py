from .meandice import *
from .meanIoU import *
from .meansensitivity import *
from .meanspecificity import *
from .meanaccuracy import *
from .meanNSD import *

"""
因为在实验中可能会有需求需要每张图片各个指标的评价值，所有每个batch的结果都保存在一个list中返回。
"""