
import numpy as np
import torch
import logging
import cv2
import webp

from os.path import *
from os import listdir

from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.nn import Threshold
from io import BytesIO


class BasicDataset(Dataset):
    # let's try to make this all work in memory
    '''
    The original version, which read image from disk
    uncomment to enable
    '''
    # def __init__(self, line_dir, edge_dir, radius = 2, crop_size = 0):
    #     self.line_dir = line_dir
    #     self.edge_dir = edge_dir
    #     self.kernel = self.get_ball_structuring_element(radius)
        
    #     self.crop_size = crop_size if crop_size != 0 else 1024
    #     assert self.crop_size > 0
        
    #     self.ids = listdir(line_dir)
    #     self.length = len(self.ids)
    #     assert self.length == len(listdir(edge_dir))
        
    #     logging.info(f'Creating dataset with {len(self.ids)} examples')

    '''
    The modified version, read the whole data set in numpy array
    '''
    def __init__(self, lines_bytes, edges_bytes, radius = 2, crop_size = 0):
        
        self.lines_bytes = lines_bytes
        self.edges_bytes = edges_bytes
        
        self.kernel = self.get_ball_structuring_element(radius)
        
        self.crop_size = crop_size if crop_size != 0 else 1024
        assert self.crop_size > 0
        
        
        
        self.length = len(lines_bytes)
        # self.length = len(self.ids)

        assert self.length == len(edges_bytes)
        
        logging.info(f'Creating dataset with {self.length} examples')

    def __len__(self):
        return self.length

    def get_ball_structuring_element(self, radius):
        """Get a ball shape structuring element with specific radius for morphology operation.
        The radius of ball usually equals to (leaking_gap_size / 2).
        
        # Arguments
            radius: radius of ball shape.
                 
        # Returns
            an array of ball structuring element.
        """
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    def __getitem__(self, i):

        
        '''
        The original version, uncomment to enable
        '''
        # idx = self.ids[i]
        # edge_path = join(self.edge_dir, idx.replace("webp", "png"))
        # line_path = join(self.line_dir, idx)

        # assert exists(edge_path), \
        #     f'No edge map found for the ID {idx}: {edge_path}'
        # assert exists(line_path), \
        #     f'No line art found for the ID {idx}: {line_path}'
        
        # remove white 
        # if ".webp" in line_path:
        #     line_np = np.array(webp.load_image(line_path, "RGB").convert("L"))
        # else:
        #     line_np = np.array(Image.open(line_path))

        # if ".webp" in edge_path:
        #     edge_np = np.array(webp.load_image(edge_path, "RGB"))
        # else:
        #     edge_np = np.array(Image.open(edge_path))
        '''
        the end of orignal version
        '''

        '''
        The modified version
        '''
        line_bytes = self.lines_bytes[i]
        edge_bytes = self.edges_bytes[i]

        
        buffer = BytesIO(line_bytes)
        line_np = np.array(Image.open(buffer).convert("L"))

        buffer = BytesIO(edge_bytes)
        edge_np = np.array(Image.open(buffer).convert("L"))
        '''
        end of modified version
        '''

        '''
        The following part should be fine
        '''

        # crop_bbox = self.find_bbox(self.to_point_list(line_np))
        # line_np = self.crop_img(crop_bbox, line_np)
        # edge_np = self.crop_img(crop_bbox, edge_np)
        
        # line_np, edge_np = self.random_resize([line_np, edge_np])
        
        # or threshold by opencv? 
        _, mask1_np = cv2.threshold(line_np, 125, 255, cv2.THRESH_BINARY)
        _, mask2_np = cv2.threshold(edge_np, 125, 255, cv2.THRESH_BINARY)

        # convert to tensor, and the following process should all be done by cuda
        line = self.to_tensor(line_np)
        edge = self.to_tensor(edge_np)
        
        mask1 = self.to_tensor(mask1_np, normalize = False)
        mask2 = self.to_tensor(mask2_np, normalize = False)

        assert line.shape == line.shape, \
            f'Line art and edge map {i} should be the same size, but are {line.shape} and {edge.shape}'

        

        imgs = self.augment(torch.cat((line, edge, mask1, mask2), dim=0))
        
        # it returns tensor at last
        return torch.chunk(imgs, 4, dim=0)

    def to_point_list(self, img_np):
        p = np.where(img_np < 220)
        return p

    def find_bbox(self, p):
        t = p[0].min()
        l = p[1].min()
        b = p[0].max()
        r = p[1].max()
        return t,l,b,r

    def crop_img(self, bbox, img_np):
        t,l,b,r = bbox
        return img_np[t:b, l:r]

    def random_resize(self, img_np_list):
        '''
        Experiment shows that random resize is not working well, so this function is obsoleted and just be left here 
        as a record. 
        Don't try random resize in this way, it will not work!
        Much slower converging speed and not obvious better generalizetion ability
        '''        
        size = self.crop_size * (1 + np.random.rand()/5)
        
        # if the image is a very long or wide image, then split it before cropping
        img_np_resize_list = []
        for img_np in img_np_list:
            if len(img_np.shape) == 2:
                h, w = img_np.shape
            else:
                h, w, _ = img_np.shape
            
            short_side = w if w < h else h
            r = size / short_side
            target_w = int(w*r+0.5)
            target_h = int(h*r+0.5)
            img_np = cv2.resize(img_np, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img_np_resize_list.append(img_np)

        return img_np_resize_list
    
    def to_tensor(self, pil_img, normalize = True):

        # assume the input is always grayscal
        if normalize:
            transforms = T.Compose(
                    [
                        # to tensor will change the channel order and divide 255 if necessary
                        T.ToTensor(),
                        T.Normalize(0.5, 0.5, inplace = True)
                    ]
                )
        else:
            transforms = T.Compose(
                    [
                        # to tensor will change the channel order and divide 255 if necessary
                        T.ToTensor(),
                    ]
                )

        return transforms(pil_img)
    
    def augment(self, tensors):
        transforms = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomCrop(size = self.crop_size)

                ]
            )
        return transforms(tensors)
