from .otsu import *
#from FeatureExtract import *
import cv2
import numpy as np
from skimage import io
import scipy.misc

def truncate_hu(image_array, max = 400, min = -900):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    return image
'''
edge不用这个函数了，比较慢
'''
def edge_detection(img):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度图
    # img = truncate_hu(gray)
    img = truncate_hu(img)

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    #z = cv2.Sobel(img, cv2.CV_16S, 1, 1)

    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    #absz = cv2.convertScaleAbs(z)

    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    return dst
    
'''
otsu还没找到代替的
'''
def otsu(img):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 转换为灰度图
    # otsu_img = otsu_helper(gray, upper=118, down = 45,categories=1)
    otsu_img = otsu_helper(img, categories=1)
    return otsu_img
