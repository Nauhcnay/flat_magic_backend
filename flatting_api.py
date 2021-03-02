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
nets = {}

def initail_nets(force_refresh=False):
    '''
        Load trained U-Net models to nets
    '''
    global nets

    try:
        if len(nets) == 0 or force_refresh:
            path_1024 = "./checkpoints/rand_1024/"
            path_1024_base = "./checkpoints/base_1024/"
            path_512_base = "./checkpoints/base_512/"
            path_512 = "./checkpoints/rand_512/"
            nets["1024"] = initial_models(path_1024)
            nets["1024_base"] = initial_models(path_1024_base)

            nets["512"] = initial_models(path_512)
            nets["512_base"] = initial_models(path_512_base)

        return True
    except:
        return False

def run_single(line_artist, net, radius, preview=False):
    '''
    Given:
        line_artst, the line art drawn by artist, 
            it should be a binary image, contains paint storkes only
        nets, the trained U-net to simplify artist line
        radius, the trapped ball filling radius
        preview, return the coarse preview result, 
            this option will be much faster than getting the full results
    Return:
        fill_map, the final fill map, each region is given a number label
            it comes from the combination o ftrapped ball filling on 
            simplified line and connected component filling on artist line
        fill, the colored fill_map
        fill_map_artist, the connect component fill map on artist line
            it usually contain much more splited regions than fill_map
        fill_artist, the colored fill_map_artist
    '''
    assert initail_nets()
    global nets

    # simplify artist line
    if str(nets[net][1]) == 'cpu':
        print("Warning:\tno gpu found, using cpu mode, the inference may be slow")
    
    print("Log:\tsimplify artist line")
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
    drop_small_regions(fill_map_artist)
    fill_map_artist = verify_reigon(fill_map_artist, True)
    fill_artist, palette = show_fillmap_auto(fill_map_artist)

    fill, palette = show_fillmap_auto(fill_map, palette)
    # get layers
    layers = get_layers(fill_map)
    layers_artist = get_layers(fill_map_artist)

    ## np.array([[0,1], [1,2]]).tolist() -> [[0,1], [1,2]]
    ## base64.encode( Image.fromarray( fill ).save( format = PNG, io.ByteIO ) )

    # return {
    #     'fill_color': fill.tolist(),
    #     'fill_integer': fill_map.tolist(),
    #     'layers': layers.tolist(),
    #     'components_color': fill_artist.tolist(),
    #     'components_integer': fill_map_artist.tolist(),
    #     'components_layers': layers_artist.tolist(),
    #     'palette': palette.tolist()
    #     }

    return {
        'fill_color': fill,
        'fill_integer': fill_map,
        'layers': layers,
        'components_color': fill_artist,
        'components_integer': fill_map_artist,
        'components_layers': layers_artist,
        'palette': palette
        }

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
    palette_list = []

    for i in range(len(line_artist_list)):
        if preview:
            result = run_single(line_artist_list[i], net_list[i], radius_list[i], preview=True)
            fill_list.append(result)

        else:
            results = run_single(line_artist_list[i], net_list[i], radius_list[i])

            fill_list.append(results['fill_color'])
            fill_map_list.append(results['fill_integer'])
            layers_list.append(results['layers'])
            fill_artist_list.append(results['components_color'])
            fill_map_artist_list.append(results['components_integer'])
            layers_artist_list.append(results['components_layers'])
            palette_list.append(results['palette'])
    
    return {
        'fill_color': fill_list,
        'fill_integer': fill_map_list,
        'layers': layers_list,
        'components_color': fill_artist_list,
        'components_integer': fill_map_artist_list,
        'components_layers': layers_artist_list
        }

def merge(fill_map, merge_map, palette):
    '''
    Given:
        fill_map, labeled region map
        merge_map, a image contains merge stroke only
    Action:
        merge all selected region to the largest one
    '''
    # get labels selected by merge stroke
    merge_labels = stroke_to_label(fill_map, merge_map)
    if len(merge_labels) <= 1:
        print("Log:\t(probably) inaccurate input, skip merge")
        return fill_map

    # find max region
    max_label = find_max_region(fill_map, merge_labels)

    # merge all rest regions to max region
    for r in merge_labels:
        if r == max_label: continue
        mask = fill_map == r
        fill_map[mask] = max_label

    # visualize fill_map
    fill, palette = show_fillmap_auto(fill_map, palette)
    
    return {"fill_color": fill,
            "fill_integer": fill_map,
            "palette": palette}

def split_auto(fill_map, fill_map_artist, split_map_auto, palette):
    '''
    Given:
        fill_map, labeled final region map 
        fill_map_artist, labeled region on artist line only, 
            this will contain much more regions than the final fill map
        split_map_auto, a image contains split stroke only
    Action:
        split new regions into fill_map
    '''
    # select regions that user want to split
    split_labels_artist = stroke_to_label(fill_map_artist, split_map_auto)
    if len(split_labels_artist) <= 1:
        print("Log:\t(probably) inaccurate input, skip split auto")
        return fill_map

    # find out the region that don't needs to be extract
    neural_to_artist = {} # map from fill_map to fill_map_artist
    for r in split_labels_artist:
        rn = np.unique(fill_map[fill_map_artist == r])
        for s in rn:
            if s in neural_to_artist:
                neural_to_artist[s].append(r)
            else:
                neural_to_artist[s] = [r]
    fixed_region = []
    for s in neural_to_artist.values():
        if len(s) > 1:
            fixed_region.append(find_max_region(fill_map_artist, s))

    # split rest regions
    next_label = np.max(fill_map) + 1
    for r in split_labels_artist:
        if r in fixed_region: continue
        mask = fill_map_artist == r
        fill_map[mask] = next_label
        next_label += 1

    # visualize fill_map
    fill, palette = show_fillmap_auto(fill_map, palette)
    
    return {"fill_color": fill,
            "fill_integer": fill_map,
            "palette": palette}

def split_manual(fill_map, fill_map_artist, artist_line, split_map_manual, palette):
    '''
    Given:
        fill_map, labeled final region map 
        fill_map_artist, labeled region on artist line
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
    if len(split_labels) == 0:
        print("Log:\t(probably) inaccurate input, skip split manual")
        return fill_map

    # merge user modify to artistline
    artist_line_new = artist_line.copy()
    artist_line_new[split_map_manual < 240] = 0
    
    # merge all involved regions
    p_list = []
    for r in split_labels:
        p_list.append(np.where(fill_map_artist == r))
    merged_mask = merge_points(p_list)
    label_old = find_region(fill_map, merged_mask)

    # split  
    masks = []
    split_single_region = np.zeros(fill_map.shape, dtype=np.uint8)
    split_single_region[merged_mask] = 1
    split_single_region[artist_line_new < 240] = 0
    _, split_regions = cv2.connectedComponents(split_single_region, connectivity=8)
    regions = np.unique(split_regions)
    regions = regions[regions != 0]
    if len(regions) > 1:
        max_region = find_max_region(split_regions, regions)
        for s in regions:
            if s == max_region: 
                mask_old = split_regions == s
            else:
                masks.append(split_regions == s)

    # split regions
    next_label = fill_map.max() + 1
    for mask in masks:
        if is_split_safe(fill_map, mask):
            fill_map[mask] = next_label
            next_label += 1
    fill_map[mask_old] = label_old
    fill_map[split_map_manual < 240] = label_old
    fill_map[artist_line < 240] = 0

    # visualize fill_map
    fill, palette = show_fillmap_auto(fill_map, palette)
    
    return {"fill_color": fill,
            "fill_integer": fill_map,
            "palette": palette}

def show_fillmap_manual(fill_map, palette):
    '''
        Given:
            fill_map, the labeled region map
        return:
            color_map, the random colorized map
    '''
    region_num = np.max(fill_map) + 1
    
    if region_num > len(palette):
        print("Warning:\tfind unmatched fill map and color palette, please stop and debug the server")
        p_increment = np.random.randint(0, 255, (region_num - len(palette), 3), dtype=np.uint8)
        palette = np.concatenate((palette, p_increment), axis = 0)

    return palette[fill_map], palette

def show_fillmap_auto(fill_map, palette=None):
    '''
    Given:
        fill_map, the labeled region map
        palette, the color palette
    return:
        color_map, the random colorized map
    '''
    region_num = np.max(fill_map) + 1
    
    if palette is None:
        palette = init_palette(region_num + 50)

    if region_num > len(palette):
        print("Warning:\tgot region numbers greater than color palette size, which is unusual, please check your if filling result is correct")
        palette = init_palette(region_num)

    return palette[fill_map], palette

def drop_small_regions(fill_map, th=0.000005):
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
    
    palette = np.random.randint(0, 255, (color_num, 3), dtype=np.uint8) 
    palette[0] = [0, 0, 0]

    return palette

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
    stroke_map = stroke_map.copy()
    stroke_map[stroke_map < 240] = 0
    stroke_map[stroke_map >= 240] = 1

    # get the labels selected by stroke map
    labels = np.unique(fill_map[stroke_map == 0])
    labels = labels[labels != 0]

    return labels

def display_region(fill_map, region):
    '''
    A helper function that visualize a given region
    '''
    r = (fill_map == region).astype(np.uint8)*255
    Image.fromarray(r).show()

def find_region(fill_map, mask):
    '''
    A helper function to find the most possible region in mask
    '''
    labels, count = np.unique(fill_map[mask], return_counts=True)
    labels = labels[labels != 0]

    return labels[np.argsort(count)[-1]]

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
        if the major region in the given have less than th% of mask area (size)
        then it should not be splited.
    '''
    labels, counts = np.unique(fill_map[mask], return_counts=True)
    if counts.max() / counts.sum() < th:
        return False
    else:
        return True

def merge_points(points_list):
    '''
    A helper function for merge masks
    '''
    for i in range(len(points_list)):
        points_list[i] = np.array(points_list[i])

    points_merge = np.concatenate(points_list, axis = -1)

    return tuple(points_merge)

def save_np(array, path):
    '''
    A helper function to save numpy array, for debug purpose
    '''
    with open(path, 'wb') as f:
        np.save(f, array)

def load_np(path):
    '''
    A helper function to load numpy array, for debug purpose
    '''
    print('Log:\tload %s'%path)
    with open(path, 'rb') as f:
        result = np.load(f)
    return result

def test_case1():
    import os
    
    fill_map = load_np("./trapped_ball/examples/01_fill_integer.npy")
    fill_map_artist = load_np("./trapped_ball/examples/01_components_integer.npy")
    palette = init_palette(100)

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test01.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test02.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split_auto
    split_map_auto = np.array(Image.open("./trapped_ball/examples/01_split_auto_test01.png").convert("L"))
    result = split_auto(fill_map, fill_map_artist, split_map_auto, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test03.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test04.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    print("Done")

def test_case2():
    import os
    
    fill_map = load_np("./trapped_ball/examples/02_fill_integer.npy")
    fill_map_artist = load_np("./trapped_ball/examples/02_components_integer.npy")
    artist_line = np.array(Image.open("./trapped_ball/examples/02.png").convert("L"))
    palette = init_palette(100)

    # split manual
    split_map_manual = np.array(Image.open("./trapped_ball/examples/02_split_manual_test01.png").convert("L"))
    result = split_manual(fill_map, fill_map_artist, artist_line, split_map_manual, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split manual
    split_map_manual = np.array(Image.open("./trapped_ball/examples/02_split_manual_test02.png").convert("L"))
    result = split_manual(fill_map, fill_map_artist, artist_line, split_map_manual, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split auto
    split_map_auto = np.array(Image.open("./trapped_ball/examples/02_split_auto_test01.png").convert("L"))
    result = split_auto(fill_map, fill_map_artist, split_map_auto, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split auto
    split_map_auto = np.array(Image.open("./trapped_ball/examples/02_split_auto_test02.png").convert("L"))
    result = split_auto(fill_map, fill_map_artist, split_map_auto, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")    

    # split auto
    split_map_auto = np.array(Image.open("./trapped_ball/examples/02_split_auto_test03.png").convert("L"))
    result = split_auto(fill_map, fill_map_artist, split_map_auto, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")    

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/02_merge_test01.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    merge_map = np.array(Image.open("./trapped_ball/examples/02_merge_test02.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    merge_map = np.array(Image.open("./trapped_ball/examples/02_merge_test03.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    merge_map = np.array(Image.open("./trapped_ball/examples/02_merge_test04.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    print("Done")

# for debug
if __name__ == "__main__":
    
    '''
    test for initial work flow
    '''
    # line_path = "./trapped_ball/examples/02.png"
    # line_artist = Image.open(line_path).convert("L")
    # initail_nets()
    # results = run_single(line_artist, "512", 1)
    
    '''
    test for merge
    '''
    test_case1()
    
    '''
    test for auto split and manual split
    '''
    test_case2()
