#!/usr/bin/python3

import os
import sys
import numpy as np

sys.path.append(os.getcwd())

def parse_filename(path):
    basename = os.path.basename(path)
    filename, suffix = os.path.splitext(basename)
    filename_split = filename.split('-')
    xy1 = filename_split[2].split('_')[0].split('&')
    xy2 = filename_split[2].split('_')[1].split('&')
    x1, y1, x2, y2 = int(xy1[0]), int(xy1[1]), int(xy2[0]), int(xy2[1])
    boxes = np.zeros((1, 4), dtype = np.int32)
    boxes[0,0], boxes[0,1], boxes[0,2], boxes[0,3] = x1, y1, x2, y2
    return [filename, suffix, boxes]

image_path_1 = '../xxx/0020-0_3-334&519_404&544-402&544_334&543_336&519_404&520-0_0_8_33_17_24_32-55-14.jpg'
image_path_2 = '../xxx/0020-0_3-334&519_404&544-bla.jpg'

if __name__ == '__main__':
    filename, suffix, boxes = parse_filename(image_path_1)
    print("filename: " + filename + " suffix: " + suffix)
    print("boxes: " + str(boxes))
    filename, suffix, boxes = parse_filename(image_path_2)
    print("filename: " + filename + " suffix: " + suffix)
    print("boxes: " + str(boxes))

