import os
import webp
from os.path import *
from PIL import Image

flat = './flat'
line = './line'
obj = './objects'

flats = os.listdir(flat)
lines = os.listdir(line)
objs = os.listdir(obj)

flats.sort()
lines.sort()
objs.sort()

miss_flat = []
miss_line = []

for i in lines:
    if i.replace("line", "flat") not in flats:
        miss_flat.append(i)

for i in flats:
    if i.replace("flat", "line") not in lines:
        miss_line.append(i)

print("Missing flats:")
print(miss_flat)

print("Missing lines:")
print(miss_line)


# rename files
# count = 0
# # start rename 
# for img_l in lines:
#     img_f = img_l.replace("line", "flat")
#     img_o = img_l.replace("line", "objects")
#     assert exists(join(flat, img_f))
#     os.rename(join(line, img_l), join(line, "%05d.png"%count))
#     os.rename(join(flat, img_f), join(flat, "%05d.png"%count))
#     if img_o in objs:
#         os.rename(join(obj, img_o), join(obj, "%05d.png"%count))
#     count += 1

# convert pngs to webp
for img in lines:
    l = Image.open(join(line, img))
    f = Image.open(join(flat, img))
    webp.save_image(l, join(line, img.replace(".png", ".webp")))
    webp.save_image(f, join(flat, img.replace(".png", ".webp")))