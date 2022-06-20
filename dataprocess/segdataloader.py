import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from Transforms import Scale
import cv2

# width, height = 96, 96


def att_compare(a, b=3):
    if a > b:
        return np.array([1])
    else:
        return np.array([0])


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datas, lungs, medias, inters, unions, masks, label=None, width=64, height=64):
        self.size = (width, height)
        self.datas = datas
        self.lungs = lungs
        self.medias = medias
        self.inters = inters
        self.unions = unions
        self.masks = masks
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.datas
        self.lung_paths = self.lungs
        self.media_paths = self.medias

        self.inter_path = self.inters
        self.union_path = self.unions
        self.mask_path = self.masks

        print('Training data:')
        print(len(self.datas))

    def __getitem__(self, index):
        input = np.load(self.input_paths[index])
        lung = np.load(self.lung_paths[index])
        media = np.load(self.media_paths[index])
        lung = normalazation(lung)
        media = normalazation(media)

        inter = Image.open(self.inter_path[index])
        union = Image.open(self.union_path[index])
        mask = Image.open(self.mask_path[index])

        input = cv2.resize(input, self.size)
        lung = cv2.resize(lung, self.size)
        media = cv2.resize(media, self.size)
        inter = self.img_resize(inter)
        union = self.img_resize(union)
        mask = self.img_resize(mask)

        torch.from_numpy(input)
        torch.from_numpy(lung)
        torch.from_numpy(media)

        # 归一化
        input = self.img_transform_gray(input)
        lung = self.img_transform_gray(lung)
        media = self.img_transform_gray(media)
        inter = self.img_transform_gray(inter)
        union = self.img_transform_gray(union)
        mask = self.img_transform_gray(mask)

        # 二值化
        inter[inter > 0.5] = 1
        inter[inter < 0.5] = 0
        union[union > 0.5] = 1
        union[union < 0.5] = 0
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0

        return input, lung, media, union, inter, mask

    def __len__(self):
        return len(self.input_paths)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, datas, medias, inters, unions, masks, label=None, width=64, height=64):
        self.size = (width, height)
        self.data = datas
        self.medias = medias
        self.inters = inters
        self.unions = unions
        self.mask = masks
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.data
        self.media_paths = self.medias
        self.inter_path = self.inters
        self.union_path = self.unions
        self.mask_paths = self.mask

        print('Testing data:')
        print(len(self.input_paths))

    def __getitem__(self, index):
        input = np.load(self.input_paths[index])
        media = np.load(self.media_paths[index])
        inter = Image.open(self.inter_path[index])
        union = Image.open(self.union_path[index])
        mask = Image.open(self.mask_paths[index])

        input = cv2.resize(input, self.size)
        media = cv2.resize(media, self.size)
        inter = self.img_resize(inter)
        union = self.img_resize(union)
        mask = self.img_resize(mask)

        torch.from_numpy(input)
        torch.from_numpy(media)

        input = self.img_transform_gray(input)
        media = self.img_transform_gray(media)
        inter = self.img_transform_gray(inter)
        union = self.img_transform_gray(union)
        mask = self.img_transform_gray(mask)

        # 二值化
        inter[inter > 0.5] = 1
        inter[inter < 0.5] = 0
        union[union > 0.5] = 1
        union[union < 0.5] = 0
        mask[mask > 0.5] = 1
        mask[mask < 0.5] = 0
        

        return input, media, union, inter, mask

    def __len__(self):
        return len(self.input_paths)


class RowDataset(torch.utils.data.Dataset):
    def __init__(self, data, mask, label=None, width=64, height=64):
        self.size = (width, height)
        self.data = data
        self.mask = mask
        self.label = label
        self.img_resize = Compose([
            Scale(self.size, Image.BILINEAR),

        ])
        self.label_resize = Compose([
            Scale(self.size, Image.NEAREST),
        ])
        self.img_transform_gray = Compose([
            ToTensor(),
            Normalize(mean=[0.448749], std=[0.399953])  # 这个归一化方式可以提升近1个点的dice
        ])

        self.img_transform_rgb = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_paths = self.data
        self.mask_paths = self.mask
        self.label_paths = self.label

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index])
        mask = Image.open(self.mask_paths[index])
        # mask.save('image1.jpg')

        image = self.img_resize(image)
        mask = self.img_resize(mask)
        # mask.save('image2.jpg')

        image = self.img_transform_gray(image)
        mask = self.img_transform_gray(mask)

        # mask = mask.squeeze()
        # mask = np.transpose(mask.cpu().detach().numpy(), (0,1))
        # cv2.imwrite('image3.jpg', mask)
        return image, mask

    def __len__(self):
        return len(self.input_paths)


def loader(dataset, batch_size, num_workers=8, shuffle=False):
    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers)

    return input_loader
