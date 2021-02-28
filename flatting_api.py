# global variable as palette
import os, sys
import numpy as np
import cv2

from os.path import *
sys.path.append(join(dirname(abspath(__file__)), "trapped_ball"))
from run import region_get_map, merge_to_ref, verify_reigon
from predict import predict_img
from demo import initial_models
from PIL import Image

# global variables shared by all api functions
color_palette_auto = None
color_palette_manual = None
nets = {}

def initail_nets():
    '''
        Load trained U-Net models to nets
    '''
    path_1024 = "./checkpoints/rand_1024/"
    path_1024_base = "./checkpoints/base_1024/"
    path_512_base = "./checkpoints/base_512/"
    path_512 = "./checkpoints/rand_512/CP_epoch1201.pth"
    nets["1024"] = initial_models(path_1024)
    nets["1024_base"] = initial_models(path_1024_base)

    nets["512"] = initial_models(path_512)
    nets["512_base"] = initial_models(path_512_base)

    return None

def show_fillmap_manual(fill_map):
    '''
        Given:
            fill_map, the labeled region map
        return:
            color_map, the random colorized map
    '''
    if len(color_palette_manual) is None:
        raise ValueError("user color palette is not initailized, can't continue process")

    return color_palette_auto[fill_map]

def show_fillmap_auto(fill_map):
    '''
        Given:
            fill_map, the labeled region map
        return:
            color_map, the random colorized map
    '''
    region_num = np.max(fill_map) + 1
    if color_palette_auto is None:
        init_palette()

    if region_num > len(color_palette_auto):
        print("Warning:\tgot region numbers greater than color palette size, which is unusual, please check your if filling result is correct")
        init_palette(region_num)

    return color_palette_auto[fill_map]

def drop_small_regions(fill_map, th=0.00002):
    h, w = fill_map.shape
    th = int(h*w*th)
    labels = np.unique(fill_map)

    counter = 0
    for r in labels:
        mask = fill_map == r
        if mask.sum() < th:
            fill_map[mask] = 0
            counter += 1
    print("Log:\t%d region that smaller than %d pixels are removed"%(counter, th))

def get_layers(fill_map):
    '''
    Given:
        fill_map, the labeled region map
    Return:
        layers, a list of numpy arrays, each array is a single region
    '''
    h, w = fill_map.shape
    layers = []

    labels = np.unique(fill_map)
    assert np.max(fill_map) + 1 == len(labels)
    assert len(labels) <= len(color_palette_auto)

    for region in labels:

        layer = np.ones((h, w, 3), dtype=np.uint8) * 255
        mask = fill_map == region
        layer[mask] = color_palette_auto[region]
        layers.append(layer)

    return layers

def init_palette(color_num = 100):
    '''
    Initialize the auto color palette, assume there will be no more than 500 regions
    If we get a fill map more than 100 regions, then we need to re-initialize this with larger number
    '''
    global color_palette_auto
    color_palette_auto = np.random.randint(0, 255, (color_num, 3), dtype=np.uint8) 
    color_palette_auto[0] = [0, 0, 0]

    return None


def run_single(line_artist, net, radius, preview=False):
    '''
    Given:
        line_artst, the line art drawn by aritst, 
            it should be a binary image, contains paint storkes only
        nets, the trained U-net to simplify aritst line
        radius, the trapped ball filling radius
        preview, return the coarse preview result, 
            this option will be much faster than getting the full results
    Return:
        fill_map, the final fill map, each region is given a number label
            it comes from the combination o ftrapped ball filling on 
            simplified line and connected component filling on artist line
        fill, the colored fill_map
        fill_map_artist, the connect component fill map on aritst line
            it usually contain much more splited regions than fill_map
        fill_artist, the colored fill_map_artist
    '''
    # simplify artist line
    if str(nets[net][1]) == 'cpu':
        print("Warning:\tno gpu found, using cpu mode, the inference may be slow")
    
    print("Log:\tsimplify aritst line")
    size = int(net.split("_")[0])
    line_simplify = predict_img(net=nets[net][0],
                       full_img=line_artist,
                       device=nets[net][1],
                       size = int(size))

    # filling and refine
    print("Log:\ttrapping ball filling with radius %s"%str(radius))
    if preview:
        return region_get_map(line_simplify,
                path_to_line_artist=line_artist,  
                radius_set=[int(radius)], percentiles=[0],
                preview=preview
                )
    else:
        fill_map, fill_map_neural, fill_map_artist = region_get_map(line_simplify,
                                                    path_to_line_artist=line_artist,  
                                                    output_path = "./", # need to comment later
                                                    radius_set=[int(radius)], percentiles=[0],
                                                    )

    # color fill map for visualize
    fill = show_fillmap(fill_map)
    drop_small_regions(fill_map_artist)
    fill_map_artist = verify_reigon(fill_map_artist, True)
    fill_artist = show_fillmap(fill_map_artist)

    # get layers
    layers = get_layers(fill_map)
    layers_artist = get_layers(fill_map_artist)

    return fill, fill_map, layers, fill_artist, fill_map_artist, layers_artist


def run_multiple(line_artist_list, net_list, radius_list, preview=False):
    '''
    Given:
        line_artist_list, a list of artist line art, now we assume it will always be numpy array
        net, the name of model
        radius, the radius for trapped ball filling
    Return:
        result lists
    '''
    assert len(line_artist_list) == len(net_list)
    assert len(net_list) == len(radius_list)

    fill_list = []
    fill_map_list = []
    layers_list = []
    fill_artist_list = []
    fill_map_artist_list = []
    layers_artist_list = []

    for i in range(len(line_artist_list)):
        if preview:
            result = run_single(line_artist_list[i], net_list[i], radius_list[i], True)
            fill_list.append(result)

        else:
            results = run_single(line_artist_list[i], net_list[i], radius_list[i])

            fill_list.append(results[0])
            fill_map_list.append(results[1])
            layers_list.append(results[2])
            fill_artist_list.append(results[3])
            fill_map_artist_list.append(results[4])
            layers_artist_list.append(results[5])

    return fill_list, fill_map_list, layers_list, fill_artist_list, fill_map_artist_list, layers_artist_list

def stroke_to_label(fill_map, stroke_map):
    '''
    Given:
        fill_map, labeled region map
        merge_map, a image contains stroke only, assume it is image stored as numpy array
    Return:
        the region index of which are selected by stroke map
    '''
    # threshold the stroke map
    if len(stroke_map.shape) == 3:
        stroke_map = cv2.cvtColor(stroke_map, cv2.COLOR_BGR2GRAY)
    stroke_map[stroke_map < 125] = False
    stroke_map[stroke_map >= 125] = True

    # get the labels selected by stroke map
    labels = np.unique(fill_map[stroke_map])
    labels = labels[labels != 0]

    return labels

def merge(fill_map, merge_map):
    '''
    Given:
        fill_map, labeled region map
        merge_map, a image contains merge stroke only
    Action:
        merge all selected region to the largest one
    '''
    # get labels selected by merge stroke
    merge_labels = stroke_to_label(fill_map, merge_map)

    # find max region
    max_label = find_max_region(fill_map, merge_labels)

    # merge all rest regions to max region
    for r in merge_labels:
        if r == max_label: continue
        mask = fill_map == r
        fill_map[mask] = max_label

    return None

def find_max_region(fill_map, selected_labels):
    '''
    A helper function to find the label of max region amout selected regions
    '''
    # find the size of selected regions
    labels, counts = np.unique(fill_map, return_counts=True)
    labels_sorted = labels[np.argsort(counts)]

    max_idx = -1
    for r in selected_labels:
        if max_idx == -1:
            max_idx = np.where(labels_sorted == r)
        else:
            cur_idx = np.where(labels_sorted == r)
            if max_idx < cur_idx:
                max_idx = cur_idx
    max_label = labels_sorted[max_idx]

    return max_label

def is_split_safe(fill_map, mask, th=0.9):
    '''
    A helper function that to test if the given region could be splited safely
        if the major region in the given mask less than th% of mask area (size)
        then it should not be splited.
    '''
    labels, counts = np.unique(fill_map[mask], return_counts=True)
    if counts.max() / counts.sum() < th:
        return False
    else:
        return True

def split_auto(fill_map, fill_map_artist, split_map_auto):
    '''
    Given:
        fill_map, labeled final region map 
        fill_map_artist, labeled region on aritst line only, 
            this will contain much more regions than the final fill map
        split_map_auto, a image contains split stroke only
    Action:
        split new regions into fill_map
    '''
    # select regions that user want to split
    split_labels_artist = stroke_to_label(fill_map_artist, split_map_auto)
    split_label = stroke_to_label(fill_map, split_map_auto)
    # everytime we will just split one region
    if len(split_label) > 1:
        # select the region with max areas
        split_label = find_max_region(fill_map, split_label)

    # find out the region that don't needs to be split
    assert len(split_labels_artist) > 1
    fixed_region = find_max_region(fill_map_artist, split_labels_artist)

    # split rest regions
    next_label = np.max(fill_map) + 1
    for r in split_labels_artist:
        if r == fixed_region: continue
        mask = fill_map_artist == r
        fill_map[mask] = next_label
        next_label += 1

    return None

def merge_points(points_list):
    '''
    A helper function for merge masks
    '''
    for i in range(len(points_list)):
        points_list[i] = np.array(points_list[i])

    points_merge = np.concatenate(points_list, axis = -1)

    return points_merge

def split_manual(fill_map, fill_map_artist, artist_line, split_map_manual):
    '''
    Given:
        fill_map, labeled final region map 
        fill_map_artist, labeled region on aritst line
        artist_line, the artist line art
        split_map_manual, a image contains manual split stroke only, it should be precise
    Action:
        split new regions into fill_map
    '''
    # convert split map to grayscale
    if len(split_map_manual.shape) == 3:
        split_map_manual = cv2.cvtColor(split_map_manual, cv2.COLOR_BGR2GRAY)

    # find the region need to be split on artist fill map
    split_labels = stroke_to_label(fill_map_artist, split_map_manual)
    
    # create new artist line
    split_artist_line = artist_line[split_map_manual < 125] = 0
    
    # merge all involved regions
    p_list = []
    for r in split_labels:
        p_list.append(np.where(fill_map == r))
    merged_mask = merge_points(p_list)

    # split  
    masks = []
    split_single_region = np.ones(fill_map.shape, dtype=np.uint8)
    split_single_region[fill_map_artist != r] = 0
    _, split_regions = cv2.connectedComponents(split_single_region, connectivity=8)
    regions = np.unique(split_regions)
    regions = regions[regions != 0]
    if len(regions) > 1:
        max_region = find_max_region(split_single_region, regions)
        for s in regions:
            if s == max_region: continue
            masks.append(split_single_region == s)

    # split regions
    next_label = fill_map.max() + 1
    for mask in masks:
        if is_split_safe(fill_map, mask):
            fill_map[mask] = next_label
            next_label += 1

    return None

def merge_test():
    pass

def split_auto_test():
    pass

def split_manual_test():
    pass

# for debug
if __name__ == "__main__":
    line_path = "./trapped_ball/examples/02.png"
    line_artist = Image.open(line_path).convert("L")

    initail_nets()

    results = run_single(line_artist, "512_base", 1)
    print("Done")