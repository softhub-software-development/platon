#!/usr/bin/python3

import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os

CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'Ä', 'Ö', 'Ü', ' '
]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):

    def __init__(self, img_dir, imgSize, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        height, width, _ = image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            image = cv2.resize(image, self.img_size)
        image = self.PreprocFun(image)
        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        #imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            if c == '_':
                c = ' '
            label.append(CHARS_DICT[c])
        return image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)
    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

if __name__ == "__main__":
    dataset = LPRDataLoader(['train/data/lpr/trn'], (94, 24))   
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print('data length is {}'.format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print('image batch shape is', imgs.shape)
        print('label batch shape is', labels.shape)
        print('label length is', len(lengths))      
        break

