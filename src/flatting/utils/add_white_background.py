import os
import numpy as np

from os.path import *
from PIL import Image
from tqdm import tqdm

source = "L:\\2.Research_project\\3.flatting\\Pytorch-UNet\\flatting\\validation"
target = "L:\\2.Research_project\\3.flatting\\Pytorch-UNet\\flatting\\validation"

for img in tqdm(os.listdir(source)):
    
    if ".png" not in img: continue

    # open image
    img_a = Image.open(join(source, img))
    
    # prepare white backgournd
    img_w = Image.new("RGBA", img_a.size, "WHITE")
    try:
        img_w.paste(img_a, None, img_a)
        img_w.convert("RGB").save(join(target, img))
    except:
        print("Error:\tfailed on %s"%img)
    