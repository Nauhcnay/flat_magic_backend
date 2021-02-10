import os
import cv2
import numpy as np

from os.path import *

size = [256, 512, 1024, 2048]
target_size = size[-1]

out_path = "./size_all_compare"
line_path = "./size_2048/line_croped"

for s in size:
    size_path = "./size_%d/trapping_fill_croped"%s
    for f in os.listdir(size_path):
        
        name, _ = splitext(f)
        
        if "edge" in f:
            continue
        elif "line" in f and s != target_size:
            continue
        elif "line" in f and s == target_size:
            img_up = cv2.imread(join(size_path, f), cv2.COLOR_BGR2RGB)
            cv2.imwrite(join(out_path, name.replace("_line", "")+"_%d.png"%s), img_up)    
            continue
        
        # find line image
        assert exists(join(line_path, f.replace("_fill", "")))

        # read image
        print("Log:\topen %s"%join(size_path, f))
        img = cv2.imread(join(size_path, f), cv2.COLOR_BGR2RGB)

        print("Log:\topen %s"%join(line_path, f.replace("_fill", "")))
        line = cv2.imread(join(line_path, f.replace("_fill", "")), 0)
        
        # resize image
        img_up = cv2.resize(img, (target_size, target_size), interpolation = cv2.INTER_NEAREST)

        # combine image
        assert img_up.shape[:2] == line.shape[:2]
        img_up[np.where(line == 0)] = 0
        cv2.imwrite(join(out_path, name+"_%d.png"%s), img_up)
