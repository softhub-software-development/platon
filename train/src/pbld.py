#!/usr/bin/python3

import os
import sys
sys.path.append(os.getcwd())
import bld

pnet_postive_file = 'data/pos_12_val.txt'
pnet_part_file = 'data/part_12_val.txt'
pnet_neg_file = 'data/neg_12_val.txt'
# pnet_landmark_file = './data/landmark_12.txt'
imglist_filename = 'data/imglist_anno_12_val.txt'

if __name__ == '__main__':
    anno_list = []
    anno_list.append(pnet_postive_file)
    anno_list.append(pnet_part_file)
    anno_list.append(pnet_neg_file)
    # anno_list.append(pnet_landmark_file)
    chose_count = bld.assemble_data(imglist_filename, anno_list)
    print("PNet train annotation result file path:%s" % imglist_filename)

