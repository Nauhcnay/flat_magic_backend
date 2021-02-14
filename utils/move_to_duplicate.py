import os
import shutil
from os.path import *

flat = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\flat"
line = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\line"
duplicate = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\duplicate"
test = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_org\\test"


line_croped = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_1024\\line_croped"
line_detection_croped = "L:\\2.Research_project\\3.flatting\\flatting_trapped_ball\\flatting\\size_1024\\line_detection_croped"

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

lines_croped = os.listdir(line_croped)
lines_croped.sort()
lines_detection_croped = os.listdir(line_detection_croped)
lines_detection_croped.sort()

assert len(lines_croped) == len(lines_detection_croped)
# for img in os.listdir(line):
#     if img.replace("line", "flat") not in flats:
#         print("Log:\tmoving %s"%img)
#         shutil.move(join(line, img), join(test, img.replace(".png", "_line.png")))

# for img in os.listdir(flat):
#     if img.replace("flat", "line") not in lines:
#         print("Log:\tmoving %s"%img)
#         shutil.move(join(flat, img), join(test, img.replace(".png", "_flat.png")))

count = 0
for i in range(len(lines_croped)):
    assert lines_croped[i] == lines_detection_croped[i]
    img = lines_croped[i]
    os.rename(join(line_croped, img), join(line_croped, "%04d.png"%count))
    os.rename(join(line_detection_croped, img), join(line_detection_croped, "%04d.png"%count))
    count += 1