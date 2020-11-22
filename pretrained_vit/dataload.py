import os, sys, json
import random

import cv2
import numpy as np
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



def letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
    # resize a rectangular image to a padded rectangular
    shape = img.shape[:2] # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]

    dw = (width - new_shape[0]) / 2     # width padding
    dh = (height - new_shape[1]) / 2    # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dh + 0.1)

    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakerBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=(512, 512), transforms=None):
        # ToDo: augmentation
        self.path = path
        self.img_files, self.labels = self._get_name_list()

        self.nF = len(self.img_files)
        self.width = img_size[0]
        self.height = img_size[1]

        self.transforms = transforms

    def _get_name_list(self):
        assert os.path.isdir(self.path) == True

        label_file = os.path.join(self.path, 'labellist.json')
        img_folders_path = os.path.join(self.path, 'train')

        assert os.path.exists(label_file) == True
        assert os.path.isdir(img_folders_path) == True

        with open(label_file, 'r') as f:
            label_list = json.load(f)
        img_folders_list = sorted(os.listdir(img_folders_path))

        assert len(img_folders_list) == len(label_list)

        label_idx = {}
        i = 0 
        for label in label_list:
            label_idx[label] = i
            i += 1

        img_files = []
        labels = []

        for imgs_folder_name, label in zip(img_folders_list, label_list):
            imgs_path = os.path.join(img_folders_path, imgs_folder_name)
            imgs_list = os.listdir(imgs_path)
            one_hot = np.zeros(len(label_list))
            one_hot[label_idx[label]] = 1

            for img_file in imgs_list:
                img_files.append(img_file)
                labels.append(one_hot)

        return img_files, labels

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label = self.labels[files_index]
        img, img_path, (h, w) = self._get_data(img_path=img_path)
        return {'image': img, 'label': label}

    def _get_data(self, img_path):
        height = self.height
        width = self.width

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        # augmentation ? 
        # hsv ?
        # random_affine ? 
        # rotate ?
        # gaussian ?
        
        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        img = np.ascontiguousarray(img[:, :, ::-1])     # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, img_path, (h, w)

    def __len__(self):
        return self.nF
