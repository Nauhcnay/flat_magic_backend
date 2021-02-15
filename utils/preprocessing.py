# remove all white regions in image, and do the sample crop to groud turth
# then resize all images by calling magick
import os
import cv2
import numpy as np

from os.path import *
from PIL import Image
from tqdm import tqdm

def to_np(img_path, th = False):
    
    if th:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img,220,255,cv2.THRESH_BINARY)
        # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    return img

def to_point_list(img_np):
    p = np.where(img_np < 220)
    return p

def find_bbox(p):
    t = p[0].min()
    l = p[1].min()
    b = p[0].max()
    r = p[1].max()
    return t,l,b,r

def crop_img(bbox, img_np):
    t,l,b,r = bbox
    return img_np[t:b, l:r]   

def center_crop_resize(img_np, size, crop=False, th=False):
    # if the image is a very long or wide image, then split it before cropping
    if len(img_np.shape) == 2:
        h, w = img_np.shape
    else:
        h, w, _ = img_np.shape
    
    short_side = w if w < h else h
    r = size / short_side * 1.2
    target_w = int(w*r+0.5)
    target_h = int(h*r+0.5)
    img_np = cv2.resize(img_np, (target_w, target_h), interpolation = cv2.INTER_AREA)
    if th:
         _, img_np = cv2.threshold(img_np,250,255,cv2.THRESH_BINARY)
    # center crop image
    if crop:
        l = (target_w - size)//2
        t = (target_h - size)//2
        r = (target_w + size)//2
        b = (target_h + size)//2
        img_np = img_np[t:b, l:r]
    return img_np

def try_split(img_np):
    
    img_list = []
    
    h, w = img_np.shape[:2]
    if h >= 2*w:
        splition = h // w
        for i in range(0, h-h//splition, h//splition):
            img_list.append(img_np[i:i+h//splition])
    elif w >= 2*h:
        splition = w // h
        for i in range(0, w-w//splition, w//splition):
            img_list.append(img_np[:,i:i+w//splition])
    else:
        img_list.append(img_np)

    return img_list

def main():
    path_root = "../flatting"
    org = "size_org"
    
    crop_size = 1024
    size = "size_%d"%crop_size

    path_to_img = join(path_root, org, "line")
    path_to_mask = join(path_root, org, "line_detection")
    out_path_img = join(path_root, size, "line_croped")
    out_path_mask = join(path_root, size, "line_detection_croped")

    counter = 0
    for img_name in tqdm(os.listdir(path_to_img)):
        
        mask_name = img_name
        assert exists(join(path_to_mask, mask_name))
        
        img_np_th = to_np(join(path_to_img, img_name), th=True)
        img_np = to_np(join(path_to_img, img_name))
        mask_np = to_np(join(path_to_mask, mask_name))

        # remove addtional blank area
        bbox = find_bbox(to_point_list(img_np_th))    
        img_crop = crop_img(bbox, img_np)
        mask_crop = crop_img(bbox, mask_np)

        # detect if a image need split
        if False:
            img_crop_list = try_split(img_crop)
            mask_crop_list = try_split(mask_crop)
            assert len(img_crop_list) == len(mask_crop_list)
        else:
            img_crop_list = [img_crop]
            mask_crop_list = [mask_crop]

        # crop and resize each image
        for i in range(len(img_crop_list)):

            img = center_crop_resize(img_crop_list[i], crop_size)
            mask = center_crop_resize(mask_crop_list[i], crop_size, th=True)

            assert img.shape[:2] == mask.shape[:2]

            if False:
                cv2.imwrite(join(out_path_img, "%05d.png"%counter), img)
                cv2.imwrite(join(out_path_mask, "%05d.png"%counter), mask)
            else:
                cv2.imwrite(join(out_path_img, img_name), img)
                cv2.imwrite(join(out_path_mask, img_name), mask)
            counter += 1



if __name__=="__main__":
    main()
