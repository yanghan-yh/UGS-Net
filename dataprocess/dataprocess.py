import random
from .segdataloader import *
from .utils import *
import csv
import glob 
import cv2

fold = 1

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line[0])
    return lines


def get_dataloader(config, mode='train', batchsize=64, width=64, height=64):

    train_datas = []
    train_masks = []
    for index in config.training_fold_index:
        tempdata = readCSV(os.path.join(config.csvPath, 'data_fold' + str(index) + '.csv'))
        tempmask = readCSV(os.path.join(config.csvPath, 'mask_fold' + str(index) + '.csv'))

        train_datas += tempdata
        train_masks += tempmask

    test_datas = readCSV(os.path.join(config.csvPath, 'data_fold' + str(config.test_fold_index[0]) + '.csv'))
    test_masks = readCSV(os.path.join(config.csvPath, 'mask_fold' + str(config.test_fold_index[0]) + '.csv'))

    
    if mode=='train':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_train_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp_test_datas = []
        for one in test_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_test_datas.append(one_list[0] + '_' + one_list[1] + '_' + one_list[2])

        mid_files = os.listdir(config.maskPath2)


        temp2_train_inter = []
        temp2_train_union = []
        temp2_train_data = []
        temp2_train_lung = []
        temp2_train_media = []
        temp2_train_mask = []

        for one_train_data in temp_train_datas:
            imagename = one_train_data + '.png'

            if imagename in mid_files:
                innertemp0 = config.midPath + one_train_data + '.npy'
                innertemp1 = config.lungPath + one_train_data + '_lung.npy'
                innertemp2 = config.mediaPath + one_train_data + '_mediastinal.npy'
                innertemp3 = config.maskPath2 + one_train_data + '_red.png'
                innertemp4 = config.maskPath2 + one_train_data + '_blue.png'
                innertemp5 = config.maskPath1 + 'mid_' + one_train_data + '_mask.png'
                temp2_train_data.append(innertemp0)
                temp2_train_lung.append(innertemp1)
                temp2_train_media.append(innertemp2)
                temp2_train_union.append(innertemp3) 
                temp2_train_inter.append(innertemp4)
                temp2_train_mask.append(innertemp5)


        temp2_test_data = []
        temp2_test_lung = []
        temp2_test_media = []
        temp2_test_inter = []
        temp2_test_union = []
        temp2_test_mask = []
        for one_test_data in temp_test_datas:
            imagename = one_test_data + '.png'       
            if imagename in mid_files:
                innertemp0 = config.midPath + one_test_data + '.npy' 
                innertemp1 = config.lungPath + one_test_data + '_lung.npy'
                innertemp2 = config.mediaPath + one_test_data + '_mediastinal.npy'
                innertemp3 = config.maskPath2 + one_test_data + '_red.png'
                innertemp4 = config.maskPath2 + one_test_data + '_blue.png'
                innertemp5 = config.maskPath1 + 'mid_' + one_test_data + '_mask.png'
                temp2_test_data.append(innertemp0)
                temp2_test_lung.append(innertemp1)
                temp2_test_media.append(innertemp2)
                temp2_test_union.append(innertemp3) 
                temp2_test_inter.append(innertemp4)
                temp2_test_mask.append(innertemp5)


        print('***********')
        print('the length of train data: ', len(temp2_train_data))
        print('the length of test data: ', len(temp2_test_data))
        print('-----------')
        dataloader = loader(Dataset(temp2_train_data, temp2_train_lung, temp2_train_media, temp2_train_inter, temp2_train_union, temp2_train_mask,  width=width, height=height), batchsize)
        dataloader_val = loader(Dataset(temp2_test_data, temp2_test_lung, temp2_test_media, temp2_test_inter, temp2_test_union, temp2_test_mask, width=width, height=height), batchsize)
        return dataloader, dataloader_val

    if mode=='row':
        # remove features labels
        temp_train_datas = []
        for one in train_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_train_datas.append('mid_' + one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp_test_datas = []
        for one in test_datas:
            one_temp = one.split('/')[-1]
            one_list = one_temp.split('_')
            temp_test_datas.append('mid_' + one_list[0] + '_' + one_list[1] + '_' + one_list[2])
        temp2_train_datas = []
        temp2_train_masks = []
        temp2_test_datas = []
        temp2_test_masks = []
        row_files = os.listdir(config.rowPath)
        for one_train_data in temp_train_datas:
            imagename = one_train_data + '.png'
            if imagename in row_files:
                innertemp0 = config.rowPath + one_train_data + '.png'
                innertemp1 = config.rowPath + one_train_data + '_mask.png'
                temp2_train_datas.append(innertemp0)
                temp2_train_masks.append(innertemp1)
        for one_test_data in temp_test_datas:
            imagename = one_test_data + '.png'
            if imagename in row_files:
                innertemp0 = config.rowPath + one_test_data + '.png'
                innertemp1 = config.rowPath + one_test_data + '_mask.png'
                temp2_test_datas.append(innertemp0)
                temp2_test_masks.append(innertemp1)

        dataloader = loader(RowDataset(temp2_train_datas, temp2_train_masks, width=width, height=height), batchsize)
        dataloader_val = loader(RowDataset(temp2_test_datas, temp2_test_masks, width=width, height=height), batchsize)

        return dataloader, dataloader_val
        
