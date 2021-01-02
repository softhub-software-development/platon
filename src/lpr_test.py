#!/usr/bin/env python3

import sys
import os
sys.path.append(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
from lpr_net import LPRNet
from lpr_eval import decode
from load_data import CHARS
from stn import STNet
import numpy as np
import argparse
import torch
import time
import cv2

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    return inp

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("arial.ttf", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Demo')
    parser.add_argument("-image", help='image path', default='train/data/lpr/val/DE_BI_AE_608.jpeg', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('weights/lpr_net', map_location=lambda storage, loc: storage))
    lprnet.eval()

    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('weights/st_net', map_location=lambda storage, loc: storage))
    STN.eval()

    print("Successful to build network!")

    since = time.time()
    image = cv2.imread(args.image)
    im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
    transfer = STN(data)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 68, 18)

    labels, pred_labels = decode(preds, CHARS)
    print("model inference in {:2.3f} seconds".format(time.time() - since))

    img = cv2ImgAddText(image, labels[0], (0, 0))

    transformed_img = convert_image(transfer)
#   cv2.imshow('transformed', transformed_img)
    cv2.imwrite('www/tmp/lpr-out.jpg', img)

#   cv2.imshow("test", img)
    cv2.imwrite("www/tmp/lpr-test.jpg", img)
#   cv2.waitKey()
#   cv2.destroyAllWindows()

