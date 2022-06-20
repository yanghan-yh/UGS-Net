import csv 
import platform 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import otsu 
from FeatureExtract import *
from skimage.feature import hog, haar_like_feature,local_binary_pattern,multiblock_lbp,daisy
import pysift
import time 

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines 

ori_path = 'C:/Users/RL/Desktop/可解释性的特征学习分类/nodules/ori_hu/'

def get_feature_name(temp, index, value):
    name = []
    for t in temp:
        if float(t[index]) == value:
            if len(t[1]) == 1:
                name.append('LIDC-IDRI-000' + t[1] + '_' + t[3] + '_' + t[2] + '.npy')
            elif len(t[1]) == 2:
                name.append('LIDC-IDRI-00' + t[1] + '_' + t[3] + '_' + t[2] + '.npy')
            elif len(t[1]) == 3:
                name.append('LIDC-IDRI-0' + t[1] + '_' + t[3] + '_' + t[2] + '.npy')
            else:
                name.append('LIDC-IDRI-' + t[1] + '_' + t[3] + '_' + t[2] + '.npy')
    return name 

def read_name():
    labelCSV = readCSV('C:/Users/RL/Desktop/可解释性的特征学习分类/特征图片/malignancy.csv')
    Max = []
    Min = []
    temp = [labelCSV[i] for i in range(len(labelCSV))]
    final = np.array([[float(temp[i][j + 21]) for j in range(9)]  for i in range(len(temp)) if '0' not in temp[i]])
    Max = final.max(axis=0)
    Min = final.min(axis=0)
    # prob_map = [[final[i][j] / (Max[j] + Min[j] + 1) if j != 8 else final[i][j] >= 3.5 for j in range(final.shape[1])] for i in range(final.shape[0])]
    # 29 28 27毛刺 26分叶 25 24 23 22内在的 21
    # print(prob_map[0])  
    name_maoci= get_feature_name(temp, 27, Max[6])
    name_fenye = get_feature_name(temp, 26, Max[5])
    name_solid = get_feature_name(temp, 28, Max[7])
    name_non_solid = get_feature_name(temp, 28, 3)
    name_moboli = get_feature_name(temp, 28, 1)
    # with open('C:/Users/RL/Desktop/可解释性的特征学习分类/特征图片/name.txt',"w") as f:
    #     f.write("*" * 10 + "maoci:"+ '\n')
    #     for name in name_maoci:
    #         f.write(name + '\n')
    #     f.write("*" * 10 + "fenye:"+ '\n')
    #     for name in name_fenye:
    #         f.write(name+ '\n')
    #     f.write("*" * 10 + "shixing:"+ '\n')
    #     for name in name_solid:
    #         f.write(name+ '\n')
    #     f.write("*" * 10 + "yashixing:"+ '\n')
    #     for name in name_non_solid:
    #         f.write(name+ '\n')
    #     f.write("*" * 10 + "moboli:"+ '\n')
    #     for name in name_moboli:
    #         f.write(name+ '\n')
    return name_maoci, name_fenye, name_solid, name_non_solid, name_moboli

def read_lidc(filename):
    image = np.load(ori_path + filename)
    return image 
 
def image_feature_extract(image):
    features = [image]
    # otsu_image = otsu.helper(image.copy())
    features.append(otsu.helper(image.copy()))
    features.append(gabor(image.copy()))


    # 添加新的方式
    # lbp = local_binary_pattern(image.copy(), 3, 3, method='var')
    # features.append(otsu._otsu(lbp))
    # features.append(edge_detection(image.copy()))
    _, hog_image = hog(gabor(image.copy()), orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=False)
    features.append(hog_image) 


    features.append(np.transpose(super_pixel(image.copy()),(2,0,1))[0])
    features.append(local_binary_pattern(image.copy(), 4, 4, method='var')) # 会产生无穷大或者是无穷小
    fd, hog_image = hog(image.copy(), orientations=9, pixels_per_cell=(4, 4),cells_per_block=(2, 2), visualize=True, multichannel=False)

    _, descs_img = daisy(image.copy(), step=180, radius=58, rings = 2,  histograms=10,orientations=8,visualize=True)
    features.append(hog_image)
    features.append(np.transpose(descs_img.copy(),(2,0,1))[0])
    features.append(descs_img)
    kps, _ = pysift.computeKeypointsAndDescriptors(image.copy())
    x = []
    y = []
    for kp in kps:
        x.append(kp.pt[0])
        y.append(kp.pt[1])
    return features, x, y

# 接下里就是 统计每个属性的图像特征输出
if __name__ == "__main__":
    a1, a2, a3, a4, a5 = read_name()
    ori_image = []
    # 不清楚normalize的影响
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0007_2_3000631.npy')))])# 毛刺
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0060_1_3000575.npy')))])# 分叶
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0003_4_3000611.npy')))]) # 磨玻璃
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0003_2_3000611.npy')))]) # 实性
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0008_1_3000549.npy')))]) # 亚实性 
    ori_image.append([normalization(truncate_hu(read_lidc('LIDC-IDRI-0132_1_5418.npy')))]) # 空洞
    X = []
    Y = []
    for i in range(6):
        features, x, y = image_feature_extract(ori_image[i][0])
        for feature in features:
            ori_image[i].append(feature)
        # print(len(ori_image[i]))
        X.append(x)
        Y.append(y)
    name = ["origial","sift-key-point","otsu","gabor","new","super-pixel","lbp","hog","daisy-three-dim","daisy-gray"]
    plt.figure()
    # 对实性结节 对亚实性结节 对分叶 对毛玻璃 对空洞 对毛刺
    numRows = 6
    numCols = 10
    font2 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 7,
            }
    for i in range(numRows):
        for j in range(numCols):
            ax = plt.subplot(numRows,numCols,1 + i * numCols + j)
            if i == 0:
                ax.set_title(name[j],font2)
            if j != numCols - 1:
                plt.imshow(ori_image[i][j], cmap="gray")
            else:
                plt.imshow(ori_image[i][j])
            if j == 1:
                plt.scatter(X[i], Y[i], color='red', s=3, alpha=0.5)
            plt.xticks([])
            plt.yticks([])
    # plt.savefig("C:/Users/RL/Desktop/可解释性的特征学习分类/特征图片/pictures/test" + str(time.time()) + ".png",dpi=1500)
    plt.show()
