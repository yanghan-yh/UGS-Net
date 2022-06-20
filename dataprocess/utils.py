import numpy as np
import random
import copy


def data_resample4seg(train_datas, train_masks,index=-2, splitvalue=3, resample_rate=1, is_big=True):
    print('Before Resample datas:', len(train_datas))
    res_datas = copy.deepcopy(train_datas)
    res_masks = copy.deepcopy(train_masks)
    for train_data, train_mask in zip(train_datas,train_masks):
        if (is_big is True and int(train_data.split('_')[index]) >= splitvalue) or (is_big is False and int(train_data.split('_')[index]) < splitvalue):
            for i in range(resample_rate):
                res_datas.append(train_data)
                res_masks.append(train_mask)
    sorted(res_datas)
    sorted(res_masks)
    temp = list(zip(res_datas, res_masks))
    random.shuffle(temp)
    res_datas, res_masks = zip(*temp)
    print('After Resample datas:', len(res_datas))
    return res_datas, res_masks


def data_resample4cls(train_datas, index=-2, splitvalue=3, resample_rate=1, is_big=True):
    print('Before Resample datas:',len(train_datas))
    resample_datas = copy.deepcopy(train_datas)
    for train_data in train_datas:
        # print(train_data)
        value = int(train_data.split('_')[index])
        if (is_big is True and value >= splitvalue) or (is_big is False and value <= splitvalue):
            for i in range(resample_rate):
                resample_datas.append(train_data)
                # print(train_data)
    print('After Resample datas:', len(resample_datas))
    random.shuffle(resample_datas)
    return resample_datas
