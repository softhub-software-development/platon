#!/usr/bin/python3

import os
import sys
sys.path.append(os.getcwd())
import bld

onet_postive_file = 'data/pos_24_val.txt'
onet_part_file = 'data/part_24_val.txt'
onet_neg_file = 'data/neg_24_val.txt'
# pnet_landmark_file = './data/landmark_12.txt'
imglist_filename = 'data/imglist_anno_24_val.txt'

if __name__ == '__main__':
    anno_list = []
    anno_list.append(onet_postive_file)
    anno_list.append(onet_part_file)
    anno_list.append(onet_neg_file)
    chose_count = bld.assemble_data(imglist_filename, anno_list)
    print("RNet train annotation result file path:%s" % imglist_filename)

