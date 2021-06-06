import cv2
import numpy as np
import os, sys
sys.path.append("./line")
import torch

from PIL import Image
from os.path import *
from tqdm import tqdm
from hed.run import estimate
from line.thin import Thinner
from skimage.morphology import skeletonize, thin


# let's use a advanced edge detection algorithm
def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).
    
    # Arguments
        radius: radius of ball shape.
             
    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def to_tensor(path_to_img):
    
    # img = Image.open(path_to_img)
    img_np = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
    img_np =  np.ascontiguousarray(img_np[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))

    return torch.FloatTensor(img_np)

def to_numpy(edge_tensor):
    
    edge_np = edge_tensor.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0
    edge_np = edge_np.astype(np.uint8)

    return edge_np

def extract_skeleton(img):

    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while done is False:
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel

def extract_gt_hed(input_line, input_flat, out_path):

    _, line_name = split(input_line)

    # extract edge by HED
    tenInput = to_tensor(input_flat)
    tenOutput = estimate(tenInput)

    # threshold the output
    edge = to_numpy(tenOutput)
    edge_thresh = cv2.adaptiveThreshold(255-edge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # values = np.unique(edge)
    # lower_bound = np.percentile(values, 30)
    # _, edge_thresh = cv2.threshold(edge, lower_bound, 255, cv2.THRESH_BINARY)

    # get skeleton 
    # thin = Thinner()
    # edge_thin = thin(Image.fromarray(edge)).detach().cpu().numpy().transpose(1,2,0)*255
    # edge_thin = edge_thin.astype(np.uint8).repeat(3, axis=-1)

    # all of these not work
    # edge_thin = cv2.ximgproc.thinning(edge)
    # edge_thin = extract_skeleton(255 - edge_thresh)
    # edge_thin = skeletonize(edge_thresh)

    # Image.fromarray(edge_thresh).save(join(out_path, line_name))
    cv2.imwrite(join(out_path, line_name), edge_thresh)
    

def extract_gt(input_line, input_flat, out_path):
    # initialize

    print("Log:\topen %s"%input_flat)
    if exists(out_path) is False:
        os.makedirs(out_path)

    # canny edge detection
    img = cv2.imread(input_flat)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.blur(img,(5,5))
    
    # analyaze the gradient of flat image
    grad = cv2.Laplacian(img,cv2.CV_64F)
    grad = abs(grad).sum(axis = -1)
    grad_v, grad_c = np.unique(grad, return_counts=True)

    # remove the majority grad, which is 0
    assert np.where(grad_v==0) == np.where(grad_c==grad_c.max())
    grad_v = np.delete(grad_v, np.where(grad_v==0))
    grad_c = np.delete(grad_c, np.where(grad_c==grad_c.max()))
    print("Log:\tlen of grad_v %d"%len(grad_v))
    grad_c_cum = np.cumsum(grad_c)

    # if grad number is greater than 100, then this probably means the current
    # image exists pretty similar colors, then we should apply 
    # another set of parameter to detect edge
    # this could be better if we can find the realtion between them
    if len(grad_v) < 100:
        min_val = grad_v[np.where(grad_c_cum<=np.percentile(grad_c_cum, 25))[0].max()]
        max_val = grad_v[np.where(grad_c_cum<=np.percentile(grad_c_cum, 40))[0].max()]
    else:
        min_val = grad_v[np.where(grad_c_cum<=np.percentile(grad_c_cum, 1))[0].max()]
        max_val = grad_v[np.where(grad_c_cum<=np.percentile(grad_c_cum, 10))[0].max()]

    edges = cv2.Canny(img, min_val, max_val, L2gradient=True)

    # write result
    _, line_name = split(input_line)
    cv2.imwrite(join(out_path, line_name.replace("webp", "png")), 255-edges)

def main():
    
    input_line_path = "../flatting/size_org/line/"
    input_flat_path = "../flatting/size_org/flat/"
    out_path = "../flatting/size_org/line_detection/"    

    for img in tqdm(os.listdir(input_line_path)):
        input_line = join(input_line_path, img)
        input_flat = join(input_flat_path, img.replace("line", "flat"))
        
        # neural net base edge detection
        # extract_gt_hed(input_line, input_flat, out_path)
        
        # canny edge detection
        extract_gt(input_line, input_flat, out_path)



if __name__=="__main__":
    main()