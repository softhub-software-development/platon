#!/usr/bin/env python3

from lpr_net import LPRNet
from stn import STNet
from load_data import LPRDataLoader, collate_fn, CHARS
from util import *
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 
    inp = inp[:,:,::-1]
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        dataset = LPRDataLoader([args.img_dirs], args.img_size)   
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn) 
        imgs, labels, lengths = next(iter(dataloader))
        input_tensor = imgs.cpu()
        transformed_input_tensor = stn(imgs.to(device)).cpu()
        in_grid = convert_image(torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image(torchvision.utils.make_grid(transformed_input_tensor))
        # Plot the results side-by-side
        #f, axarr = plt.subplots(1,2)
        #axarr[0].imshow(in_grid)
        cv2.imwrite('www/tmp/lpr-in.jpg', in_grid)
        #axarr[0].set_title('Dataset Images')
        #axarr[1].imshow(out_grid)
        cv2.imwrite('www/tmp/lpr-out.jpg', out_grid)
        #axarr[1].set_title('Transformed Images')

def decode(preds, CHARS):
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :, :]
        pred_label = list()
        for j in range(pred.shape[1]):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
#   print("labels: " + str(labels))
    return labels, pred_labels

def eval(lprnet, stn, dataloader, num_samples, device):
    lprnet = lprnet.to(device)
    stn = stn.to(device)
    tp = 0
    for imgs, labels, lengths in dataloader:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = stn(imgs)
        logits = lprnet(transfer) # torch.Size([batch_size, CHARS length, output length ])
        preds = logits.cpu().detach().numpy()  # (batch size, 40, 18)
        _, pred_labels = decode(preds, CHARS)  # list of predict output
        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                tp += 1
    acc = tp / num_samples
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPR Evaluation')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs', default="train/data/lpr/trn", help='the images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--batch_size', default=2, help='batch size.')
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('weights/lpr_net', map_location=lambda storage, loc: storage))
    lprnet.eval() 
    print("LPRNet loaded")

    stn = STNet().to(device)
    stn.load_state_dict(torch.load('weights/st_net', map_location=lambda storage, loc: storage))
    stn.eval()
    print("STN loaded")

    dataset = LPRDataLoader([args.img_dirs], args.img_size)   
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn) 
    print('dataset loaded with length : {}'.format(len(dataset)))

    num_samples = len(dataset)
    acc = eval(lprnet, stn, dataloader, num_samples, device)
    print('the accuracy is {:.2f} %'.format(acc * 100))
    visualize_stn()

