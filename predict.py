import argparse
import logging

import os, sys
from os.path import *
sys.path.append(join(dirname(abspath(__file__)), "trapped_ball"))

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from PIL import Image
from torchvision import transforms as T

from unet import UNet

from run import region_get_map


def to_tensor(img):

    transforms = T.Compose(
                    [
                        # to tensor will change the channel order and divide 255 if necessary
                        T.ToTensor(),
                        T.Normalize(0.5, 0.5, inplace = True)
                    ]
                )

    return transforms(img)

def denormalize(img):
    # denormalize
    inv_normalize = T.Normalize( mean=-1, std=2)

    img_np = inv_normalize(img.repeat(3,1,1)).clamp(0, 1)
    img_np = img_np * 255
    
    # to numpy
    img_np = img_np.cpu().numpy().transpose((1,2,0))
    
    return Image.fromarray(img_np.astype(np.uint8))

def to_numpy(f, size):
    
    if type(f) == str:
        img = np.array(Image.open(f).convert("L"))
        
    else:
        img = np.array(f.convert("L"))

    h, w = img.shape
    if w > size or h > size:
        ratio = size/w if w > h else size/h
    else:
        ratio = 1

    return cv2.resize(img, (int(w*ratio), int(h*ratio)), interpolation=cv2.INTER_AREA)

def predict_img(net,
                full_img,
                device,
                size):
    net.eval()

    # read image
    print("Log:\tpredict image at size %d"%size)
    img = to_tensor(to_numpy(full_img, size))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

    return denormalize(output[0])


def get_args():
    parser = argparse.ArgumentParser(description='Predict edge from line art',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model', '-m', default='./checkpoints/exp1/CP_epoch2001.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    
    parser.add_argument('--input', '-i', type=str,
                        help='filename of single input image, include path')

    parser.add_argument('--output', '-o', type=str,
                        help='filename of single ouput image, include path')

    parser.add_argument('--input-path', type=str, default="./flatting/validation",
                        help='path to input images')

    parser.add_argument('--output-path', type=str, default="./results/val",
                        help='path to ouput images')

    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_args()

    in_files = args.input

    net = UNet(in_channels=1, out_channels=1, bilinear=True)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for f in os.listdir(args.input_path):
        name, _ = splitext(f)

        logging.info("\nPredicting image {} ...".format(join(args.input_path, f)))

        
        # predict edge and save image
        edge = predict_img(net=net,
                           full_img=join(args.input_path, f),
                           device=device,
                           size = 1024)

        edge.save(join(args.output_path, name + "_pred.png"))
        
        # trapped ball fill and save image
        region_get_map(join(args.output_path, name + "_pred.png"), args.output_path,
                            radius_set=[1], percentiles=[0],
                            path_to_line = join(args.input_path, f),
                            save_org_size = True)
        