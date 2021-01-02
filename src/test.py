#!/usr/bin/python3

import sys
from PIL import Image, ImageDraw, ImageFont
from net import *
from util import *
import numpy as np
import argparse
import torch
import time
import cv2
import os
from imutils import paths

def center_in_rect(bbox, rect):
    cx = (bbox[0] + bbox[2]) / 2;
    cy = (bbox[1] + bbox[3]) / 2;
    return rect[0] <= cx and cx <= rect[2] and rect[1] <= cy and cy <= rect[3]

def test(img_path):
    basename = os.path.basename(img_path)
    imgname, suffix = os.path.splitext(basename)
    imgname_split = imgname.split('-')
    if len(imgname_split) > 1:
        rec_x1y1 = imgname_split[2].split('_')[0].split('&')
        rec_x2y2 = imgname_split[2].split('_')[1].split('&')
        x1, y1, x2, y2 = int(rec_x1y1[0]), int(rec_x1y1[1]), int(rec_x2y2[0]), int(rec_x2y2[1])
        boxes = np.zeros((1,4), dtype=np.int32)
        boxes[0,0], boxes[0,1], boxes[0,2], boxes[0,3] = x1, y1, x2, y2
        img = cv2.imread(img_path)
        (h, w) = img.shape[:2]
        sx = 600 / w;
        sy = 800 / h;
        log_info(basename)
        log_info("sx1: " + str(sx) + " sy1: " + str(sy) + " " + str(x1) + " " + str(y1))
        x1, y1, x2, y2 = x1 * sx, y1 * sx, x2 * sx, y2 * sx
        log_info("sx2: " + str(sx) + " sy2: " + str(sy) + " " + str(x1) + " " + str(y1))
        img = cv2.resize(img, (600, round(h * sx)))
        output = np.copy(img)
        model_postfix = args.model
        p_model_path = 'weights/pnet_' + model_postfix
        o_model_path = 'weights/onet_' + model_postfix
        log_info('using models: ' + p_model_path + ' ' + o_model_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pboxes = create_pnet(output, (50, 15), device, p_model_path=p_model_path)
        bboxes = create_onet(output, device, pboxes, o_model_path=o_model_path)
        success = False
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            success += center_in_rect(bbox, (x1, y1, x2, y2))
            output = draw_ellipse(output, bbox, cv2.FILLED)
        log_info("success: " + str(success))
        if not success:
            output = draw_rectangle(output, (x1, y1, x2, y2), 1, color=(0, 0, 255))
            output = draw_text(output, 'truth', (x1, y1), textColor=(255, 0, 0), textSize=18)
            output = draw_text(output, 'FAIL', (12, 12), textColor=(255, 0, 0), textSize=72)
        output_old_name = 'www/tmp/test/' + imgname + '-old' + suffix;
        output_ori_name = 'www/tmp/test/' + imgname + '-ori' + suffix;
        output_img_name = 'www/tmp/test/' + imgname + suffix;
        print("write: " + output_img_name)
        try:
            os.replace(output_img_name, output_old_name);
        except OSError:
            pass
        if not cv2.imwrite(output_ori_name, img):
            log_info("failed to write " + output_ori_name);
        if not cv2.imwrite(output_img_name, output):
            log_info("failed to write " + output_img_name);

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Platon')
    parser.add_argument("-model", help='model postfix', default='eu', type=str)
    parser.add_argument("-debug", help='debug mode', default='no debug', type=str)
    parser.add_argument("-dir", help='test directory', default='test', type=str)
    args = parser.parse_args()
    debug = args.debug == "debug"
    base_path = os.path.dirname(os.path.abspath(__file__))
    img_dir = base_path + '/../' + args.dir
    print("use images from: " + img_dir)
    img_paths = [el for el in paths.list_images(img_dir)]
    num = len(img_paths)
    print("%d pics total" % num)
    for img_path in img_paths:
        print("process: " + img_path)
        test(img_path)

