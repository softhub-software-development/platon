#!/usr/bin/python3

import sys
from PIL import Image, ImageDraw, ImageFont
from net import *
from lpr_eval import decode
from util import *
import numpy as np
import argparse
import torch
import time
import cv2

def recognize_license(image, bbox, lpr_net, st_net, filename):
    x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
    w = int(x2 - x1 + 1.0)
    h = int(y2 - y1 + 1.0)
    img_box = np.zeros((h, w, 3))
    img_box = image[y1:y2+1, x1:x2+1, :]
    im = cv2.resize(img_box, (94, 24), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filename, im)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94])
    transfer = st_net(data)
    preds = lpr_net(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 68, 18)
    labels, pred_labels = decode(preds, CHARS)
    #return labels[0]
    if len(labels[0]) > 0:
        return labels[0]
    else:
        return '???'

def mark_candidates(pboxes, bboxes, output):
    for i in range(pboxes.shape[0]):
        bbox = pboxes[i, :4]
        output = draw_rectangle(output, bbox, 2)
    output = draw_text(output, str(len(pboxes)) + ' pboxes', (12, 12), textColor=(128, 0, 0), textSize=18)
    output = draw_text(output, str(len(bboxes)) + ' license plates', (12, 28), textColor=(0, 0, 255), textSize=18)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Platon')
    parser.add_argument("-image", help='image path', default='test/0.jpg', type=str)
    parser.add_argument("-output", help='image output', default='/var/www/ai/tmp', type=str)
    parser.add_argument("-prefix", help='image prefix', default='lic-', type=str)
    parser.add_argument("-model", help='model postfix', default='eu', type=str)
    parser.add_argument("-license", help='recognize license', default='no', type=str)
    parser.add_argument("-debug", help='debug mode', default='no debug', type=str)
    args = parser.parse_args()
    debug = args.debug == "debug" or args.debug == "yes"
    basename = args.output + '/' + args.prefix
    rel_path = 'tmp/' + args.prefix
    with open(basename + 'desc.txt', 'w') as descriptor:
        descriptor.write("src:" + rel_path + 'in.png' + "::\n")
        descriptor.write("dst:" + rel_path + 'out.png' + "::\n")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_info("read image " + args.image + " " + str(args.debug));
        image = cv2.imread(args.image)
        (h, w) = image.shape[:2]
        image = cv2.resize(image, (600, round(h * 600 / w)))
        output = np.copy(image)
        model_postfix = args.model
        p_model_path = 'weights/pnet_' + model_postfix
        o_model_path = 'weights/onet_' + model_postfix
        log_info('using models: ' + p_model_path + ' ' + o_model_path)
        pboxes = create_pnet(output, (50, 15), device, p_model_path=p_model_path)
        bboxes = create_onet(output, device, pboxes, o_model_path=o_model_path)
        if debug:
            output = mark_candidates(pboxes, bboxes, output)
        lpr_net = create_lpr_net(device)
        st_net = create_st_net(device)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :4]
            output = draw_ellipse(output, bbox, cv2.FILLED)
            bbox_img_name = "box-" + str(i) + ".png"
            descriptor.write("lic:" + rel_path + bbox_img_name + ":" + str(bbox.astype(int)))
            print("debug: " + str(debug))
            if args.license == "yes" or debug:
                label = recognize_license(image, bbox, lpr_net, st_net, basename + bbox_img_name)
                descriptor.write(":" + label)
                x1, y1, x2, y2 = [int(bbox[j]) for j in range(4)]
                output = draw_rectangle(output, (x1, y1, x2, y2), 1)
                output = draw_text(output, label, (x1, y1-12), textColor=(228, 228, 228), textSize=18)
            descriptor.write("\n")
        input_img_name = basename + 'in.png';
        output_img_name = basename + 'out.png';
        if not cv2.imwrite(input_img_name, image):
            log_info("failed to write " + input_img_name);
        if not cv2.imwrite(output_img_name, output):
            log_info("failed to write " + output_img_name);
    descriptor.close()

