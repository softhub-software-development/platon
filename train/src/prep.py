#!/usr/bin/env python3

import os
import sys
from imutils import paths
import numpy as np
import cv2
import argparse
import random

base_path = os.path.dirname(os.path.abspath(__file__))
data_dir = base_path + '/../data'
parser = argparse.ArgumentParser(description='crop the licence plate from original image')
parser.add_argument("-image", help='image path', default = data_dir + '/raw/images', type=str)
parser.add_argument("-dir_train", help='save directory', default = data_dir + '/raw/train', type=str)
parser.add_argument("-dir_val", help='save directory', default = data_dir + '/raw/val', type=str)
parser.add_argument("-size", help='the number of images to be saved', default=5000, type=int)
args = parser.parse_args()

img_paths = []
img_paths += [el for el in paths.list_images(args.image)]
random.shuffle(img_paths)

save_dir_train = args.dir_train
save_dir_val = args.dir_val

if not os.path.exists(save_dir_train):
    os.makedirs(save_dir_train)
if not os.path.exists(save_dir_val):
    os.makedirs(save_dir_val)

print('image data processing is kicked off...')
print("%d images in total" % len(img_paths))

idx = 0
idx_train = 0
idx_val = 0
for i in range(len(img_paths)):
    filename = img_paths[i]  
    basename = os.path.basename(filename)   
    img = cv2.imread(filename)
    idx += 1 
    if idx % 100 == 0:
        print("%d images done" % idx)
    if idx % 4 == 0:
        save = save_dir_val + '/' + basename
        print("save val " + save)
        cv2.imwrite(save, img)
        idx_val += 1
    else:
        save = save_dir_train + '/' + basename
        print("save train " + save)
        cv2.imwrite(save, img)
        idx_train += 1
    if idx == args.size:
        break   
        
print('image data processing done, write %d training images, %d val images' % (idx_train, idx_val))

