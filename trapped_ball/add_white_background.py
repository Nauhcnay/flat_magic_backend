import os
import numpy as np

from os.path import *
from PIL import Image

source = "L:\\0.Data_set\\Flatting\\line"
target = "L:\\0.Data_set\\Flatting\\line_white_background"

for img in os.listdir(source):
    
    if ".png" not in img: continue

    # open image
    img_a = Image.open(join(source, img))
    
    # prepare white backgournd
    img_w = Image.new("RGBA", img_a.size, "WHITE")
    img_w.paste(img_a, None, img_a)

    img_w.convert("RGB").save(join(target, img))
    