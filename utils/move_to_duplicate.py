import os
import shutil
from os.path import *

flat = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\flat"
line = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\line"
duplicate = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\duplicate"


# with open("moving log.txt", "r") as f:
#     move_list = f.readlines()

# for img in move_list:
#     img = img.replace("\n", "").replace("Log:    moving ", "")
#     if "flat" in img:
#         shutil.move(join(duplicate, img), join(flat, img))
#     if "line" in img:
#         shutil.move(join(duplicate, img), join(line, img))

flats = os.listdir(flat)
lines = os.listdir(line)

for img in os.listdir(line):
    if img.replace("line", "flat") not in flats:
        print("Log:\tmoving %s"%img)
        shutil.move(join(line, img), join(duplicate, img))

for img in os.listdir(flat):
    if img.replace("flat", "line") not in lines:
        print("Log:\tmoving %s"%img)
        shutil.move(join(flat, img), join(duplicate, img))