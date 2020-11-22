import os, sys, json
import cv2
import numpy as np

from pprint import pprint

def _label_json_read(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
    return data

def _img_list(img_path):
    assert os.path.isdir(img_path) == True
    return os.listdir(img_path)


def main():
    label_file_path = './../data/imagenet/labellist.json'
    img_folders_path = './../data/imagenet/train'

    label_list = _label_json_read(label_file_path)
    img_folders_list = sorted(_img_list(img_folders_path))

    i = 0
    j = 0
    for imgs_folder_name, label in zip(img_folders_list, label_list):
        if i > 4:
            break
        imgs_path = os.path.join(img_folders_path, imgs_folder_name)
        imgs_list = os.listdir(imgs_path)
        
        for img_name in imgs_list:
            if j > 3:
                break
            img = cv2.imread(os.path.join(imgs_path, img_name))
            print(label)
            cv2.imshow('?', img)
            cv2.waitKey(0)
            j += 1
        j = 0
        i += 1


if __name__ == '__main__':
    main()
