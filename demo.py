import os, sys
import argparse
from os.path import *
sys.path.append(join(dirname(abspath(__file__)), "trapped_ball"))


import gradio as gr
import numpy as np
import torch
import random

from PIL import Image
from torchvision import transforms as T
from torchvision import utils

# import model
from unet import UNet
from predict import predict_img

# import trapped ball filling func
from run import region_get_map

from functools import partial
from zipfile import ZipFile

def to_t(array):
    return torch.Tensor(array).cuda().unsqueeze(0)


def to_tensor(img):
    
    img_t = (
            torch.from_numpy(img).unsqueeze(-1)
            .to(torch.float32)
            .div(255)
            .add_(-0.5)
            .mul_(2)
            .permute(2, 0, 1)
        )
    return img_t.unsqueeze(0).cuda()

def denormalize(img):
    # denormalize
    inv_normalize = T.Normalize( mean=[-1, -1, -1], std=[2, 2, 2])

    img_np = inv_normalize(img.repeat(3,1,1))
    img_np = (img_np * 255).clamp(0, 255)
    
    # to numpy
    img_np = img_np.cpu().numpy().transpose((1,2,0))
    
    return Image.fromarray(img_np.astype(np.uint8))

def zip_files(files):
    with ZipFile("./flatting/gradio/all.zip", 'w') as zipObj:
        for f in files:
            zipObj.write(f)
    return "./flatting/gradio/all.zip"

def split_to_4(img):

    # now I just write a simple code to split images into 4 evenly
    w, h = img.size
    h1 = h // 2
    w1 = w // 2
    img = np.array(img)

    # top left
    img1 = Image.fromarray(img[:h1, :w1])
    
    # top right
    img2 = Image.fromarray(img[:h1, w1:])

    # bottom left
    img3 = Image.fromarray(img[h1:, :w1])

    # bottom right
    img4 = Image.fromarray(img[h1:, w1:])

    return img1, img2, img3, img4

def merge_to_1(imgs):

    img1, img2, img3, img4 = imgs
    img_top = np.concatenate((img1, img2), axis = 1)
    img_bottom  = np.concatenate((img3, img4), axis = 1)

    return np.concatenate((img_top, img_bottom), axis = 0)

def pred_and_fill(img, op, radius, patch, nets, outputs="./flatting/gradio"):
    
    # initail out files
    outs = []
    outs.append(join(outputs, "%s_input.png"%op))
    outs.append(join(outputs, "%s_fill.png"%op))
    outs.append(join(outputs, "%s_fill_edge.png"%op))
    outs.append(join(outputs, "%s_fill_line.png"%op))
    outs.append(join(outputs, "%s_fill_line_full.png"%op))
    

    # predict full image
    # img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
    if patch == "False":
        # img_w = Image.new("RGBA", img.size, "WHITE")
        # try:
        #     img_w.paste(img, None, img)
        #     img = img_w.convert("L")
        # except:
        #     print("Log:\tfailed to add white background")
        
        edge = predict_img(net=nets[op][0],
               full_img=img,
               device=nets[op][1],
               size = int(op.replace("_rand", "")))
    else:
        print("Log:\tsplit input into 4 patch with model %s"%(op))
        # cut image into non-overlapping patches
        imgs = split_to_4(img)

        edges = []
        for patch in imgs:
            edge = predict_img(net=nets[op][0],
               full_img=patch,
               device=nets[op][1],
               size = int(op))

            edges.append(np.array(edge))

        edge = Image.fromarray(merge_to_1(edges))

    # img.save(outs[0])

    print("Log:\ttrapping ball filling with radius %s"%radius)
    fill, fill_pred, fill_line = region_get_map(edge.convert("L"),
                                # output_png=(outputs,op),
                                output_png=None,
                                radius_set=[int(radius)], percentiles=[0],
                                save_org_size = True,
                                path_to_line=img,
                                return_numpy=True)
    
    # zips = zip_files(outs)

    return edge, fill, fill_pred, fill_line
    # return edge, fill, fill_pred, fill_line, zips

def initial_models(path_to_ckpt):

    # find the lastest model
    ckpt_list = []
    
    if ".pth" not in path_to_ckpt:
        for c in os.listdir(path_to_ckpt):
            if ".pth" in c:
                ckpt_list.append(c)
        ckpt_list.sort()
        path_to_ckpt = join(path_to_ckpt, ckpt_list[-1])

    assert exists(path_to_ckpt)
    
    # init model
    net = UNet(in_channels=1, out_channels=1, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    # load model
    print("Log\tload %s"%path_to_ckpt)
    net.load_state_dict(torch.load(path_to_ckpt, map_location=device))
    net.eval()

    return net, device

def initial_flatting_input():

    # inputs
    img = gr.inputs.Image(image_mode='L', 
        invert_colors=False, source="upload", label="Input Image",
        type = "pil")
    # resize = gr.inputs.Radio(choices=["1024", "512", "256"], label="Resize")
    model = gr.inputs.Radio(choices=["1024", "1024_rand", "512", "512_rand","256","256_rand"], label="Model")
    # split = gr.inputs.Radio(choices=["True", "False"], label="Split")
    radius = gr.inputs.Slider(minimum=1, maximum=10, step=1, default=7, label="kernel radius")
    
    # outputs
    out1 = gr.outputs.Image(type='pil', label='line prediction')
    out2 = gr.outputs.Image(type='pil', label='fill')
    out3 = gr.outputs.Image(type='pil', label='fill with pred')
    out4 = gr.outputs.Image(type='pil', label='fill with line')
    # out5 = gr.outputs.File(label="all results")

    return [img, model, radius], [out1, out2, out3, out4]
    # return [img, resize], [out1, out2, out3, out4, out5]

def start_demo(fn, inputs, outputs, examples):
    iface = gr.Interface(fn = fn, inputs = inputs, outputs = outputs, examples = examples, layout = "unaligned")
    iface.launch()

def main():

    # get base tcode number
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-1024", type=str, default = "./checkpoints/base_1024/")
    parser.add_argument("--ckpt-512", type=str, default = "./checkpoints/base_512/CP_epoch2501.pth")
    parser.add_argument("--ckpt-256", type=str, default = "./checkpoints/base_256/CP_epoch5001.pth")
    parser.add_argument("--ckpt-512-rand", type=str, default = "./checkpoints/rand_512/")
    parser.add_argument("--ckpt-256-rand", type=str, default = "./checkpoints/rand_256/")
    parser.add_argument("--ckpt-1024-rand", type=str, default = "./checkpoints/rand_1024/")
    
    args = parser.parse_args()

    # initailize
    nets = {}
    nets["1024"] = initial_models(args.ckpt_1024)
    nets["1024_rand"] = initial_models(args.ckpt_512_rand)
    nets["512"] = initial_models(args.ckpt_512)
    nets["512_rand"] = initial_models(args.ckpt_512_rand)
    nets["256"] = initial_models(args.ckpt_256)
    nets["256_rand"] = initial_models(args.ckpt_512_rand)

    
    # construct exmaples
    example_path = "./flatting/validation"
    example_list = os.listdir(example_path)
    example_list.sort()
    
    examples = []
    
    for file in example_list:
        print("find %s"%file)
        img = os.path.join(example_path, file)
        model = random.choice(["1024", "512"])
        radius = 1
        examples.append((img, model, radius))

    # initial pred func
    fn = partial(pred_and_fill, nets=nets, patch="False", outputs="./flatting/gradio")

    # bug fix
    fn.__name__ = fn.func.__name__

    # start
    inputs, outputs = initial_flatting_input()
    start_demo(fn=fn, inputs=inputs, outputs=outputs, examples=examples)

def debug():
    # get base tcode number
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-1024", type=str, default = "./checkpoints/base_1024/")
    parser.add_argument("--ckpt-512", type=str, default = "./checkpoints/base_512/CP_epoch2501.pth")
    parser.add_argument("--ckpt-256", type=str, default = "./checkpoints/base_256/CP_epoch5001.pth")
    parser.add_argument("--ckpt-512-rand", type=str, default = "./checkpoints/rand_512/")
    parser.add_argument("--ckpt-256-rand", type=str, default = "./checkpoints/rand_256/")
    parser.add_argument("--ckpt-1024-rand", type=str, default = "./checkpoints/rand_1024/")
    args = parser.parse_args()

    # initailize
    nets = {}
    nets["1024"] = initial_models(args.ckpt_1024)
    nets["1024_rand"] = initial_models(args.ckpt_512_rand)
    nets["512"] = initial_models(args.ckpt_512)
    nets["512_rand"] = initial_models(args.ckpt_512_rand)
    nets["256"] = initial_models(args.ckpt_256)
    nets["256_rand"] = initial_models(args.ckpt_512_rand)
    
  
    
    img = Image.open("./flatting/validation/0120.png").convert("L")
    pred_and_fill(img, radius=1, op='256', size=256, patch="False", nets=nets, outputs="./flatting/gradio")

if __name__ == '__main__':
    main()
    # debug()
