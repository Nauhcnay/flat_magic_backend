import cv2
import numpy as np
import os
from PIL import Image
from os.path import *

def extract_gt(input_line, input_flat, out_path):
    # initialize
    if exists(out_path) is False:
        os.makedirs(out_path)

    # canny edge detection
    img = cv2.imread(input_flat, 0)
    edges = cv2.Canny(img,0,25, L2gradient=True)

    line = cv2.imread(input_line, 0)
    _, line_name = split(input_line)
    # Image.fromarray(edges).show()

    # sobel kernel
    # not good, canny is better
    # grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    # grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # print("Showing result")
    # Image.fromarray(edges).show()

    # remove nosiy shapes
    # let do this later
    # 1.get points of white pixels
    # points_list = np.where(edges == 255)
    # points = np.array(points_list).T
    # 2. generate connectivaty groups for each point
    # connectivity = {}
    # idx = 0
    # while len(points_list) != 0
    #     start = points_list.pop(0)

    # generate mask
    kernel = np.ones((20,20),np.uint8)
    edges_mask = cv2.dilate(edges,kernel,iterations = 1)
    assert edges_mask.shape == line.shape
    # Image.fromarray(edges).show()
    line[np.where(edges == 255)] = 0
    line[np.where(edges_mask != 255)] = 255
    cv2.imwrite(join(out_path, "line_detection", line_name), 255-edges)
    # cv2.imwrite(join(out_path, "line_simplified", line_name), line)

def main():
  
    input_line_path = "./flatting/size_512/line_croped/"
    input_flat_path = "./flatting/size_512/flat_croped/"
    out_path = "./flatting/size_512/line_detection_croped/"

    for img in os.listdir(input_line_path):
        
        input_line = join(input_line_path, img)
        input_flat = join(input_flat_path, img)
        extract_gt(input_line, input_flat, out_path)



if __name__=="__main__":
    main()