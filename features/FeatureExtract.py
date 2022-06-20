'''
Created by Wang Qiuli

2020/5/23
'''

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import features.slic
import features.rgb_gr
from PIL import Image
from skimage import io, color
from skimage import data, filters


def truncate_hu(image_array, max = 400, min = -900):
    image = image_array.copy()
    image[image > max] = max
    image[image < min] = min
    return image

def hist(img):
    '''
    return histgram values
    1 * 128
    '''
    img = truncate_hu(img)
    hist = cv2.calcHist([img],[0],None,[128],[-900,400])
    # print(hist.shape)
    # plt.subplot(121)
    # plt.imshow(img,'gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title("Original")
    # plt.subplot(122)
    # plt.hist(img.ravel(),128,[-900,400])
    # plt.show()    
    return hist

def gray2rgb(rgb,imggray):
   R = rgb[:,:,0]
   G = rgb[:,:,1]
   B = ((imggray) - 0.299 * R - 0.587 * G) / 0.114
 
   grayRgb = np.zeros((rgb.shape))
   grayRgb[:, :, 2] = B
   grayRgb[:, :, 0] = R
   grayRgb[:, :, 1] = G
 
   return grayRgb

def super_pixel(img):
    '''
    return super_pixel images
    img w * h
    '''
    img = truncate_hu(img)
    # io.imsave('ori.png', img)
    img = np.expand_dims(img, 2)
    # # print(img.shape)
    rgb = np.concatenate((img, img, img), 2)
    # io.imsave('ori2.png', rgb)
    obj = slic.SLICProcessor(rgb, 4096, 5)
    res = obj.iterate_10times()
    return res

def standard_deviation(img):
    hist_value = hist(img)
    std = np.std(hist_value)
    # print(std)

    return std

def edge_detection(img):
    '''
    edge detection
    '''
    img = truncate_hu(img)
    # io.imsave('ori.png', img)
    # img = np.expand_dims(img, 2)
    # # # print(img.shape)
    # rgb = np.concatenate((img, img, img), 2)

    # gray= cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    z = cv2.Sobel(img, cv2.CV_16S, 1, 1)

    return x
    # io.imsave('canny1.png', x)
    # io.imsave('canny2.png', y)
    # io.imsave('canny3.png', z)


def gabor(img):
    filt_real, filt_imag = filters.gabor(img,frequency=0.6)   
    # io.imsave('filt_imag.png', filt_imag)
    return filt_imag


def threshold_void(img):
    void = truncate_hu(img, -600, -900)
    # io.imsave('void.png', void)
    return void

def normalization(image_array):
    image_array = image_array + 900

    max = 1300
    min = 0
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    # avg = image_array.mean()
    # image_array = image_array-avg
    image_array = image_array * 255
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def toGrey(img):
    '''
    get grey-level images
    0-256
    '''
    img = truncate_hu(img)

    img_nor = normalization(img)
    # io.imsave('img_nor.png', img_nor)
    return img_nor

def OTSU(img):
    gray = toGrey(img)
    print(gray)

    gray.dtype="int16"
    print('int16')
    print(gray)

    retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    io.imsave('OTSU.png', dst)
