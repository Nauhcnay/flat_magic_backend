import os, sys
import numpy as np
import cv2
import torch

from pathlib import Path
# from os.path import *
# sys.path.append(join(dirname(abspath(__file__)), "trapped_ball"))
# sys.path.append(dirname(abspath(__file__)))
from os.path import exists, join
from .trapped_ball.run import region_get_map, merge_to_ref, verify_region
from .trapped_ball.thinning import thinning
from PIL import Image
from .predict import predict_img
from .unet import UNet

# global variables shared by all api functions
nets = {}

def initial_models(path_to_ckpt):
    path_to_ckpt = str(path_to_ckpt)

    # find the lastest model
    ckpt_list = []
    
    if ".pth" not in path_to_ckpt:
        for c in os.listdir(path_to_ckpt):
            if ".pth" in c:
                ckpt_list.append(c)
        ckpt_list.sort()
        path_to_ckpt = join(path_to_ckpt, ckpt_list[-1])

    assert exists(path_to_ckpt)
    
    # init model
    net = UNet(in_channels=1, out_channels=1, bilinear=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    # load model
    print("Log:\tload %s"%path_to_ckpt)
    try:
        net.load_state_dict(torch.load(path_to_ckpt, map_location=device))
    except:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(path_to_ckpt, map_location=device))
    net.eval()

    return net, device

def initial_nets(force_refresh=False):
    '''
        Load trained U-Net models to nets
    '''
    global nets
    # print(os.listdir("./"))
    if getattr(sys, 'frozen', False):
        # We are running in a bundle.
        checkpoints = Path(sys._MEIPASS)/'checkpoints'
    else:
        # We are running in a normal Python environment.
        checkpoints = Path(__file__).parent.parent/'checkpoints'

    try:
        if len(nets) == 0 or force_refresh:
            # we currently load the baseline mode for test
            # will update new model later
            # path_1024 = "./src/flatting/checkpoints/rand_1024/"
            # path_1024_base = "./src/flatting/checkpoints/base_1024/"
            # path_512_base = "./src/flatting/checkpoints/base_512/"
            # path_512 = "./src/flatting/checkpoints/rc_512/"
            # nets["1024"] = initial_models(path_1024)
            # nets["1024_base"] = initial_models(path_1024_base)
            # nets["512"] = initial_models(path_512)
            # nets["512_base"] = initial_models(path_512_base)
            path_512 = checkpoints/"base_512/"
            if "512" not in nets:
                nets["512"] = initial_models(path_512)
            else:
                print("Log:\tFound model exsits, skip initialization")

        return True
    except:
        return False

def add_white(img, return_numpy = False, grayscale=True):
    img = np.array(img)
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            mask = img[:,:,3] == 0
            # expand mask the the same shape of img
            mask = np.expand_dims(mask, -1)
            mask = np.repeat(mask, 3, -1)
            # remove the alpha channel
            img = img[:,:,0:3]
            # apply alpha channel mask
            img[mask] = 255
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if return_numpy:
        return img
    else:
        return Image.fromarray(img)

def add_alpha(img, line_color = None, opacity = 1):
    if img.mode != "RGBA":
        img = img.convert("RGB")
    img = np.array(img)
    
    if img.shape[2] == 4:
        return Image.fromarray(img)
    
    h, w, _ = img.shape
    img_alpha = np.zeros((h,w,4), dtype = np.uint8)
    img_alpha[:,:,:3] = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    img_alpha[:,:,3] = ((255 - img) * opacity).astype(np.uint8)

    if isinstance(line_color, str):
        assert len(line_color) == 6
        r = int(line_color[0:2], 16)
        g = int(line_color[2:4], 16)
        b = int(line_color[4:6], 16)
        img_alpha[:,:,0] = r
        img_alpha[:,:,1] = g
        img_alpha[:,:,2] = b

    return img_alpha

def fillmap_masked_line(fill_map, line_input=None, one_pixel=False):
    '''
    A helper function to extract fill region boundary as lines
    '''
    # get the one-pixel width edge
    edges = cv2.Canny(fill_map.astype(np.uint8), 0, 0)
    
    if line_input is not None:
        # increase the width of edge line
        kernel = np.ones((5,5),np.uint8)
        edges = cv2.dilate(edges, kernel, iterations = 2)
        result = np.array(line_input).copy()
        # mask line input with bolded edge line
        result[edges == 0] = 255
    else:
        # reverse the edge map
        if one_pixel==False:
            kernel = np.ones((2,2),np.uint8)
            edges = cv2.dilate(edges, kernel, anchor=(1,1), iterations = 1)
        result = np.zeros(edges.shape, np.uint8)
        result[edges == 0] = 255

    return Image.fromarray(result)

# add cropped region back
def add_cropped_back(img_pil, bbox, img_size):
    t, l, b, r = bbox
    w1, h1 = img_size # original size
    w2 = r - l # original croped size
    h2 = b - t
    w3, h3 = img_pil.size # resized croped size
    ratio = w3/w2 if w2 > h2 else h3/h2

    # resize org size to neural output size (the size if we didn't crop the input)
    w1 = int(w1 * ratio + 0.5)
    h1 = int(h1 * ratio + 0.5)
    
    # resize the bbox to the size level same as the input 
    t = int(t * ratio + 0.5)
    l = int(l * ratio + 0.5)
    b = int(b * ratio + 0.5)
    r = int(r * ratio + 0.5)

    # pad the croped image back to w1 by h1
    if (r-l != w3):
        offset_w = r - l - w3
        if offset_w < 0 and r - offset_w >= w1:
            l = l + offset_w
        elif offset_w > 0 and r - offset_w >= w1:
            l = l + offset_w - (r - offset_w - w1)
        else:
            r = r - offset_w
    if (b-t != h3):
        offset_h = b - t - h3
        if offset_h < 0 and b - offset_h >= h1:
            t = t + offset_h
        elif offset_h > 0 and b - offset_h >= h1:
            t = t + offset_h - (b - offset_h - h1)
        else:
            b = b - offset_h
    result =  np.ones((h1, w1), dtype=np.uint8) * 255
    # result[t:b, l:r] = np.array(img_pil)[:b-t, :r-l]
    result[t:b, l:r] = np.array(img_pil)

    return Image.fromarray(result)

def run_single(line_artist, net, radius, resize, w_new=None, h_new=None):
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
    loaded_initial_nets = initial_nets()
    assert loaded_initial_nets
    global nets
    line_input = add_white(line_artist)
    
    # resize the input if resize flag is on
    if resize:
        h_new = int(h_new)
        w_new = int(w_new)
        print("Log:\rresize image to size %d x %d (h x w)"%(h_new, w_new))
        line_input = line_input.resize((w_new, h_new))   
    
    # simplify artist line
    if str(nets[net][1]) == 'cpu':
        print("Warning:\tno gpu found, using cpu mode, the inference may be slow")
    print("Log:\tsimplify artist line")
    size = int(net.split("_")[0])
    line_neural, bbox = predict_img(net=nets[net][0],
                           full_img=line_input,
                           device=nets[net][1],
                           size = int(size))  
    del nets[net]
    torch.cuda.empty_cache()
    line_neural = add_cropped_back(line_neural, bbox, line_input.size)

    # refine filling result
    print("Log:\ttrapping ball filling with radius %s"%str(radius))
    fill_map, fill_map_artist, fill_color_final, \
    fill_color_neural,  fill_color_artist= region_get_map(line_neural,
                                        path_to_line_artist=line_input,  
                                        radius_set=[int(radius)], percentiles=[0],
                                        )

    # colorize fill maps with gray scale
    drop_small_regions(fill_map_artist)
    fill_map_artist = verify_region(fill_map_artist, True)
    fill, _ = show_fillmap_auto(fill_map)
    fill_artist = fill_color_artist

    # generate line hint layers
    line_simplify = fillmap_masked_line(fill_map)
    line_hint = fillmap_masked_line(fill_map_artist)
    line_simplify = add_alpha(line_simplify, line_color = "9ae42c", opacity = 0.7)
    line_hint = add_alpha(line_hint, line_color = "ec91d8", opacity = 0.7)
    line_artist = add_alpha(line_input)

    return {
        'line_artist': line_artist,
        'line_hint': line_hint,
        'line_simplified': line_simplify, 
        'fill_color': fill,
        'components_color': fill_artist,
        "fill_color_neural": fill_color_neural,
        "line_neural": line_neural
        }

def to_fillmap(image):
    '''
    Convert input image to fill map, this will make the api support undo feature
    Given:
        image, the numpy array of input image
    '''
    # we need to consider the alpha channel
    h, w, c = image.shape
    assert c == 3 or c == 4
    # get labels of the color filled result
    if c == 3:
        color_map = (1e6*image[:,:,0] + 1e3*image[:,:,1] + image[:,:,2]).astype(np.int64)
        colors = np.unique(color_map)
        color_to_label = {}
        palette = []
        for i, cl in enumerate(colors):
            # convert cl back to color
            color = [cl//(10**(3*i))%1000 for i in range(2, -1, -1)]
            palette.append(color)
            color_to_label[cl] = i

        # map color map to fill map
        fill_map = np.copy(color_map)
        for k, v in color_to_label.items():
            fill_map[color_map == k] = v
    else:
        color_map = (1e9*image[:,:,0] + 1e6*image[:,:,1] + 1e3*image[:,:,2] + image[:,:,3]).astype(np.int64)
        colors = np.unique(color_map)
        color_to_label = {}
        palette = []
        i = 0 # we need to control the counter i manually
        for cl in colors:
            color = [cl//(10**(3*i))%1000 for i in range(3, -1, -1)]
            if color[3] == 0:
                # discard this region, we use -1 to indicate all regions that is transparent
                color_to_label[cl] = -1
            else:
                color_to_label[cl] = i
                i += 1
                palette.append(color[0:3])
        # map color map to fill map
        fill_map = np.copy(color_map)
        for k, v in color_to_label.items():
            fill_map[color_map == k] = v

    return fill_map, np.array(palette)

def is_grayscale(color):
    color = color.squeeze()
    assert len(color) == 3
    r = color[0]
    g = color[1]
    b = color[2]
    return r==g and g==b

def select_labels(fill_map, stroke_mask, stroke_color, fill_palette, for_split=False):
    '''
    Given
        fill_map, fill map that want to choose regions from, it could be FF map or neural map
        stroke_mask, boolean mask of user stroke
        stroke_color, stroke color
        fill_palette, the reference fill map and its palette, it should be a nerual map
        for_split, switch for using different criterias, will be more likely to select 
            FF regions if ON
    Return
        split_labels, the labels finally selected by the user stroke
    Notes
        1. The color should also be an important factor to consider which region should be selected
            generally, we should tend to split (choose region in fill map) if we found the region 
            is colorized (not grayscale color)   
        2. User will always be inaccurate, their stroke could always go across the region boundary to a region
            they don't really want to select. we should consider all possible those cases
            2.1 Larger region should need larger stroke to select, in other word, a tiny user stroke 
                should not be able to select a huge region in fill map
            2.2 stroke should evenly distributed in all regions that he really wants
            2.3 if a region is tiny and a stroke has cover most of its region, then the user definitely
                want to select
            2.4 there would be a ambiguous case that can't really have a good standard to infer user's intention
                which is if a stroke if prefectly locate into both a FF region and neural region, it is really 
                hard to tell which region to choose

    '''
    
    fill, palette = fill_palette
    split_labels, split_labels_count = np.unique(fill_map[stroke_mask], return_counts=True)
    color_weights = []

    # criteria 5: whenever the bursh selected the region in fill_map is completely covered by the same region in 
    # fill_map_artist, we should NOT select the larger region
    criteria5 = []

    for sl in split_labels:
        sp = np.unique(fill[fill_map==sl])
        # if the region in fill_map is covered by the concurrent region in fill
        # find the largest region's color as the selected color
        # generally, we need to build up a relation between fill map and fill
        froce_false = False
        if len(sp) > 1:
            # find the reigon in fill that have the largest overlay on sl in fill_map
            overlay_size = []
            for so in sp:
                # this will have a problem: we don't know who contains who
                overlay = np.logical_and((fill_map==sl), (fill==so)).sum()
                overlay_size.append(overlay)
                # if any region in fill is contained by region in fill_map, we should always NOT split
                if overlay/(fill==so).sum() > 0.95:
                    froce_false = True
            idx_max = np.argsort(overlay_size)[-1]
            sp_max = sp[idx_max]
            so_max = overlay_size[idx_max]
        else:   
            assert len(sp) == 1
            sp_max = sp
            so_max = np.logical_and((fill_map==sl), (fill==sp)).sum()
        if so_max/(fill==sp_max).sum() > 0.9 and\
            (fill_map==sl).sum()>(fill==sp_max).sum() or froce_false:
            criteria5.append(False)
        else:
            criteria5.append(True)
        if is_grayscale(palette[sp_max]):
            color_weights.append(1) # if first time colorize, give the highest weight
        elif (palette[sp_max] == stroke_color).all():
            # if the region has be colorized and have the same color as the current stroke
            # igonre that region
            color_weights.append(0) 
        else:
            color_weights.append(0.8)
    assert len(color_weights) == len(split_labels)
    color_weights = np.array(color_weights)
    criteria5 = np.array(criteria5)

    DEBUG = False
    if DEBUG and for_split:
        print("Log:\tselect labels for spliting")
    elif DEBUG and for_split == False:
        print("Log:\tselect labels for merging")
    # criteria 2: the releative size of merge stroke in each region should be ballance, region which
    # covered by small merge stroke size will be discard
    criteria2 = split_labels_count.copy() * color_weights
    if criteria2.max() != 0:
        criteria2 = criteria2 / criteria2.max()
    if DEBUG:
        print("Log\tgot stroke region ratio size as %s, it should greater than %f"%(str(criteria2), 0.25))
    # we have to set this large, since the cost of wrong selected is much greater than wrong unselected
    criteria2 = criteria2 > 0.25

    # criteria 3: still, we need a overall threshold to tell should we split it or not, if the stroke is really large,
    # then the user might want to colorize the whole region.
    # this criteria is FOR SPLIT ONLY!
    criteria3 = []

    # criteria 4: if the region has been covered by more than 1/3, always select it
    criteria4 = []

    # criteria 1: the size of the selected region in fill_map should be balance, we should
    # exclude extreme cases, for example, the largest region is 1000 time larger than the smallest region
    criteria1 = []
    criteria1_thre = []
    
    for i, sl in enumerate(split_labels):
        if color_weights[i] < 1:
            criteria3.append(True) # always split for re-colorize
            criteria1_thre.append(100) # if the region is greater than 100 times of smallest region, exclude it
        else:
            if split_labels_count.sum() < 45 ** 2:
                criteria3.append(True)
            else:
                criteria3.append(False)
            criteria1_thre.append(1000) # if this is the first time colorize, the threshold could be loose
        criteria1.append((fill_map==sl).sum())               
        criteria4.append(split_labels_count[i] / (fill_map==sl).sum())
    
    criteria3 = np.array(criteria3)
    criteria4 = np.array(criteria4) * color_weights
    
    criteria1 = np.array(criteria1)
    criteria1 = (criteria1 / criteria1.min()) < criteria1_thre
    if DEBUG:
        print("Log\tgot stroke pixel size as %s, it should less than %d"%(str(split_labels_count), 45 ** 2))

    if DEBUG:
        print("Log:\tgot stroke/region as: %s, it should greater than 0.3"%str(criteria4))
    criteria4 = criteria4 > 0.3

    # select region that really need to be splited
    if for_split:
        split_labels = split_labels[np.logical_or(
                            np.logical_and(
                            np.logical_and(
                            np.logical_and(criteria1, criteria2), criteria3), criteria5), criteria4)]
    else:    
        split_labels = split_labels[np.logical_or(np.logical_and(criteria1, criteria2), criteria4)]
    if DEBUG:
        print("Log:\tselected regions %s"%str(split_labels))
        import pdb
        pdb.set_trace()
    return split_labels

def merge(fill_neural, fill_artist, merge_map, line_artist):
    '''
    Given:
        fill_neural, neural fill result as a image, same content as the "result_neural" layer
        fill_artist, connectcomponent fill result, not shown in Photoshop
        merge_map, color merge strokes, same as the "merge-hint" layer
        line_artist, line art of artist, same as the "line_artist" layer
    Action:
        1. try to split the selected region, upate the fill map
        2. colorize the new fill map accroding to the merge stroke and its colors.
    '''
    print("Log:\tmerging")
    
    # pixel map to label
    fill_map, palette = to_fillmap(fill_neural)
    fill_map_fix = fill_map.copy()
    fill_map_artist, _ = to_fillmap(fill_artist)
    line_artist = add_white(line_artist)
    
    # group merge stroke by color
    strokes, stroke_palette = to_fillmap(np.array(merge_map))
    
    # split any possible region for each merge stroke group
    for i in range(len(stroke_palette)):
        
        # get stroke mask and its color
        color = stroke_palette[i]
        if (strokes == i).sum() < 4: continue # ignore the stroke if it is too small (user will not notice it and unexpect results will happen)
        stroke = (1 - (strokes == i).astype(np.int)) * 255
        
        # update fill map if neccessary, new_labels is the collection of new added labels
        def split_in_merge(stroke, color, fill_map, fill_map_fix, fill_map_artist):
            '''
            Given
                stroke, merge stroke as BW image
                color, stroke color
                fill_map, neural fill map, it could be updated during the splitting
                fill_map_fix, the first version fill map, it will be fixed during the whole merge function
                fill_map_artist, flood fill (FF for short) fill map
            Return
                fill_map, updated fill map
                new_labels, new added region labels
            Action
                if the merge stroke is locating in side both a FF and neural region, check if 
                the user want to split this FF region, if yes, update the fill map
            '''
            new_labels = []
            
            # break up stroke group to single strokes
            _, stroke_map = cv2.connectedComponents((255 - stroke).astype(np.uint8), connectivity=8)
            stroke_label = np.unique(stroke_map)
            skip_region = np.array([]) # what skip region used for?
            
            # check split and split
            for l in stroke_label:
                # check split
                if l == 0: continue # skip background
                stroke_mask = stroke_map == l
                split_labels_artist = select_labels(fill_map_artist, stroke_mask, color, 
                                                (fill_map_fix, palette), for_split=True)
                # split
                if len(split_labels_artist) > 0:
                    fill_map, new_label = split_by_labels(split_labels_artist, 
                                    fill_map, fill_map_artist, skip_region)
                    new_labels += new_label

                # add to skip list after split, so we need to
                # find all artist fill map region coverd by the final selection
                skip_labels_nerual = select_labels(fill_map, stroke_mask, color, 
                                                (fill_map_fix, palette))
                if len(skip_labels_nerual) > 0:
                    skip_mask = fill_map == skip_labels_nerual[0]
                    skip_labels_nerual = np.delete(skip_labels_nerual, 0)
                    
                    for sl in skip_labels_nerual:
                        skip_mask[fill_map==sl] = True

                    skip_region = np.unique(np.append(skip_region, np.unique(fill_map_artist[skip_mask])))

            # refine new_labels
            for r in new_labels:
                if r not in fill_map:
                    new_labels.remove(r)
            return fill_map, np.array(new_labels)
        fill_map, new_labels = split_in_merge(stroke, color, fill_map, fill_map_fix, fill_map_artist)

        # re-select regions in new fill map with the same region
        merge_labels = stroke_to_label(fill_map, stroke, color, (fill_map_fix, palette))
        merge_labels = np.unique(np.concatenate((merge_labels, new_labels)))
        
        # update palette if it can't cover all regions in the new fill map
        # this is very likely to happen cause the split may creat many new regions
        if fill_map.max() >= len(palette):
            palette = init_palette(fill_map.max()+1, palette)

        # find max region among the re-selected regions, then merge and re-colorize
        max_label = find_max_region(fill_map, merge_labels)
        if max_label != -1:
            # merge
            for r in merge_labels:
                if r == max_label: continue
                mask = fill_map == r
                fill_map[mask] = max_label
            # re-colorize
            palette[max_label] = color

    # generate feedback results
    fill, _ = show_fillmap_auto(fill_map, palette)
    fill_map, _ = to_fillmap(fill)
    line_neural = fillmap_masked_line(fill_map)
    line_neural = add_alpha(line_neural, line_color = "9ae42c", opacity = 0.7)
    print("Log:\tmerge finished")

    return {"line_simplified": line_neural,
            "fill_color": fill
            }

def split_auto(fill_neural, fill_artist, split_map_auto, line_artist):
    '''
    Given:
        fill_map, labeled final region map 
        fill_map_artist, labeled region on artist line only, 
            this will contain much more regions than the final fill map
        split_map_auto, a image contains split stroke only
    Action:
        split new regions into fill_map
    '''
    # get fill map from input
    fill_map, palette = to_fillmap(fill_neural)
    fill_map_artist, _ = to_fillmap(fill_artist)

    # preprocessing lines
    line_artist = add_white(line_artist, return_numpy = True)
    split_map_auto = add_white(split_map_auto, return_numpy = True)

    # select regions that user want to split
    print("Log:\tcoarse splitting")
    split_labels_artist = stroke_to_label(fill_map_artist, split_map_auto)
    
    

    fill_map, _ = split_by_labels(split_labels_artist, fill_map, fill_map_artist)

    # update neural line
    neural_line = fillmap_masked_line(fill_map, line_artist)

    # visualize fill_map and lines
    fill, _ = show_fillmap_auto(fill_map, palette)
    neural_line = add_alpha(neural_line, line_color = "9ae42c", opacity = 0.7)

    return {"line_neural": neural_line,
            "fill_color": fill,
            }

def split_by_labels(split_labels_artist, fill_map, fill_map_artist, skip_region):
    '''
    A helper function for corase split
    '''

    # find out the region that don't needs to be extract
    neural_to_artist = {} # map from fill_map to fill_map_artist, seems we don't use that mapping...
    for r in split_labels_artist: # r is the region in fine split regions selected by stroke
        rn = np.unique(fill_map[fill_map_artist == r]) # find regions that locate at the same position in neural fill map
        for s in rn: # create the mapping from neural region to artist region
            if s in neural_to_artist:
                neural_to_artist[s].append(r)
            else:
                neural_to_artist[s] = [r]
    fixed_region = []
    # only if the regions in fill_map_artist covers the same region as the region in fill_map, 
    # we need to keep the largest regions label unchanged. otherwise
    # this will cause color blasting because we didn't split region that should be splited out
    for s in neural_to_artist:
        fill_map_a = fill_map_artist.copy()
        combined_region = neural_to_artist[s][0]
        for k in neural_to_artist[s]: fill_map_a[fill_map_a==k] = combined_region
        if len(neural_to_artist[s]) > 1 and ((fill_map==s) == (fill_map_a==combined_region)).all():
            fixed_region.append(find_max_region(fill_map_artist, neural_to_artist[s]))

    # split rest regions
    next_label = np.max(fill_map) + 1
    new_labels = [] # the new splited region means it will definitely be selected
    for r in split_labels_artist:
        if r in fixed_region: continue
        if r in skip_region: continue
        mask = fill_map_artist == r
        # if the mask selected more than one region, then we should refine this mask first
        # the split mask should always only select single region in fill map
        selected_regions, selected_regions_counts = np.unique(fill_map[mask], return_counts=True)
        if len(selected_regions) > 1:
            for sr in selected_regions:
                # if the selected region completly inside the mask, then we should exclude it
                mask[fill_map == sr] = False
            # if the artist region is totally coverd by the neural reigons and its size is less than the neural regions
            # this probably means we should restore the small artist region
            if mask.sum() < 5 and (fill_map == sr).sum() >= selected_regions_counts.sum() * 1.1:
                mask = fill_map_artist == r
        elif len(selected_regions) == 1:
            # if this region has been splitted, we also need to skip it
            if (mask == fill_map[selected_regions[0]]).all():
                continue
        else: 
            continue
        if mask.sum() == 0: continue
        fill_map[mask] = next_label
        new_labels.append(next_label)
        next_label += 1

    return fill_map, new_labels

def split_manual(fill_neural, fill_artist, split_map_manual, line_artist, add_only=True):
    '''
    Given:
        fill_neural, pixel map (colorized) from neural fill (transmitting png is faster than numpy array)
        fill_artist, pixel map (colorized) from opencv connect components fill (flood fill)
        split_map_manual, user split stroke
        line_artist, input line art (could be modified by split brush) 
    Action:
        split new region and assign a random gray scale color, while keep other regions the same
    '''
    print("Log:\tfine splitting")
    
    # remove alpha channel if neccessary
    line_artist = add_white(line_artist, return_numpy = True)
    split_map_manual = add_white(split_map_manual, return_numpy = True)

    # convert split map to grayscale
    if len(split_map_manual.shape) == 3:
        split_map_manual = cv2.cvtColor(split_map_manual, cv2.COLOR_BGR2GRAY)

    # generate label fill map
    fill_map, palette = to_fillmap(fill_neural)
    
    # here we need a special artist fill map, which need to be generated on the fly (I don't like stroke and tranmit the same data repeatly)
    if add_only:
        line_artist_copy = line_artist.copy()
        neural_line = np.array(fillmap_masked_line(fill_map))
        line_artist_copy[line_artist_copy >= 250] = 255
        line_artist_copy[line_artist_copy < 250] = 0
        line_artist_copy[split_map_manual < 250] = 0
        line_artist_copy[neural_line < 250] = 0
        _, fill_map_artist = cv2.connectedComponents(line_artist_copy, connectivity=8)
        fill_map_artist = thinning(fill_map_artist)
    else:
        fill_map_artist, _ = to_fillmap(fill_artist)

    # merge user modify to lines
    line_artist[split_map_manual < 240] = 0

    # find the region need to be split on artist fill map
    split_labels = stroke_to_label(fill_map_artist, split_map_manual, None, (None, None), True)
    split_labels_neural = stroke_to_label(fill_map, split_map_manual, None, (None, None), True)
    if len(split_labels) == 0:
        print("Log:\t(probably) inaccurate input, skip split manual")
        line_artist = add_alpha(Image.fromarray(line_artist))
        neural_line = fillmap_masked_line(fill_map, line_artist)
        neural_line = add_alpha(neural_line, line_color = "9ae42c", opacity = 0.7)
        fill, palette = show_fillmap_auto(fill_map, palette)
        return {"line_artist": line_artist,
                "line_neural": neural_line,
                "fill_color": fill,
                }

    # merge all involved regions
    p_list = []
    for r in split_labels:
        p_list.append(np.where(fill_map_artist == r))
    merged_mask = merge_points(p_list)
    # find the largest seleted region in fill map
    # label_old = find_region(fill_map, merged_mask)
    split_single_region = np.zeros(fill_map.shape, dtype=np.uint8)
    split_single_region[merged_mask] = 1
    
    if add_only:
        split_single_region[line_artist_copy < 240] = 0
    else:
        split_single_region[line_artist < 240] = 0
    
    _, split_regions = cv2.connectedComponents(split_single_region, connectivity=8)
    regions = np.unique(split_regions)
    regions = regions[regions != 0]
    if len(regions) > 1:
        max_region = find_max_region(split_regions, regions)
        old_label = find_max_region(fill_map, split_labels_neural)
        next_label = fill_map.max() + 1
        # split regions
        for s in regions:
            mask = split_regions == s
            # remove all regions in neural fill map that is contained by this mask
            # becasue if the old region in fill map could still covered by the new 
            # mask, that means this region is not modified at all, and we should not 
            # change it.
            remove_inside_regions(fill_map, mask, split_labels_neural, split_map_manual)
            if s == max_region:
                fill_map[mask] = old_label
            else:
                fill_map[mask] = next_label
                next_label += 1
        
    new_temp_label = fill_map.max() + 1
    fill_map[fill_map==0] = new_temp_label

    if add_only:
        fill_map[line_artist_copy<240] = 0
    else:
        fill_map[line_artist<240] = 0

    fill_map = thinning(fill_map)
    fill_map[fill_map==new_temp_label] = 0

    # update neural line
    neural_line = fillmap_masked_line(fill_map)
    neural_line = add_alpha(neural_line, line_color = "9ae42c", opacity = 0.7)
    
    # update palette if necssary
    if fill_map.max() >= len(palette):
        palette = init_palette(fill_map.max()+1, palette)

    # visualize fill_map
    fill, palette = show_fillmap_auto(fill_map, palette)

    # update the fill_artist map
    if add_only:
        _, fill_map_artist_new = cv2.connectedComponents((line_artist_copy > 250).astype(np.uint8), connectivity=8)
    else:
        _, fill_map_artist_new = cv2.connectedComponents((line_artist > 250).astype(np.uint8), connectivity=8)
    
    # clean up regions that less than 0.5% image area
    fill_new, fill_new_count = np.unique(fill_map_artist_new, return_counts=True)
    tiny_regions = fill_new[fill_new_count<0.00005*fill_map_artist_new.size]
    tiny_mask = np.zeros(fill_map_artist_new.shape).astype(np.bool)
    for r in tiny_regions:
        tiny_mask = np.logical_or(tiny_mask, fill_map_artist_new==r)
    fill_map_artist_new[tiny_mask] = 0
    fill_map_artist_new = thinning(fill_map_artist_new)
    
    # update to new fill artist
    fill_artist, _ = show_fillmap_auto(fill_map_artist_new)
    # layers = get_layers(fill_map, palette)

    line_hint = fillmap_masked_line(fill_map_artist_new)
    # the line hint should also be updated
    line_hint = add_alpha(line_hint, line_color = "ec91d8", opacity = 0.7)
    line_artist = add_alpha(Image.fromarray(line_artist))
    print("Log:\tdone")
    return {"line_artist": line_artist,
            "line_neural": neural_line,
            "fill_color": fill,
            "fill_artist": fill_artist,
            "line_hint": line_hint
            }
def export_layers(fill_color):
    print("Log:\textracting flat result to layers")
    fill_map, palette = to_fillmap(fill_color)
    flat_layers = get_layers(fill_map, palette)
    print("Log:\tdone")
    return {
        "layers": flat_layers
    }

def remove_inside_regions(fill_map, mask, region_list, split_map_manual, remove_selected=False):
    '''
    A helper function to remove regions that should not be splited in each given split mask
    '''
    if remove_selected:
        for r in region_list:
            mask[fill_map==r] = False
    else:
        for r in np.unique(fill_map[mask]):
            if r in region_list: continue
            mask[fill_map==r] = False

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

    return {"fill_color": palette[fill_map], 
            "palette": palette}

def show_fillmap_auto(fill_map, palette=None):
    '''
    Given:
        fill_map, the labeled region map
        palette, the color palette
    return:
        color_map, the random colorized map
    '''
    region_num = np.max(fill_map)
    
    if palette is None and region_num < 256:
        palette = init_palette(region_num, grayscale=True)
    
    elif palette is None and region_num >= 256:
        palette = init_palette(region_num)

    if region_num >= len(palette):
        # print("Warning:\tgot region numbers greater than color palette size, which is unusual, please check your if filling result is correct")
        palette = init_palette(region_num, palette)

    return palette[fill_map].astype(np.uint8), palette

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

def get_layers(fill_map, palette):
    '''
    Given:
        fill_map, the labeled region map
    Return:
        layers, a list of numpy arrays, each array is a single region
    '''
    h, w = fill_map.shape
    layers = []

    labels = np.unique(fill_map)
    # assert np.max(fill_map) + 1 == len(labels)
    assert len(labels) <= len(palette)

    for region in labels:
        layer = np.ones((h, w, 4), dtype=np.uint8) * 255
        mask = fill_map == region
        layer[:,:,0:3][mask] = palette[region]
        layer[:,:,3][(1-mask).astype(np.bool)] = 0
        layers.append(Image.fromarray(layer))

    return layers

def init_palette(color_num = 100, old_palette=None, grayscale=False):
    '''
    Initialize the auto color palette, assume there will be no more than 500 regions
    If we get a fill map more than 100 regions, then we need to re-initialize this with larger number
    '''
    # if the old palette is provide, initialize it based on the old palette
    if old_palette is not None:
        p_size = len(old_palette)
        color_num = color_num if color_num > p_size else p_size+20
        if color_num > 255:
            palette = np.random.randint(0, 255, (color_num, 3), dtype=np.uint8) 
        else:
            # we always generate grayscale color
            # find grayscale color in the palette
            skip_grayscale = []
            for i in range(len(old_palette)):
                if old_palette[i][0] == old_palette[i][1] and old_palette[i][1] == old_palette[i][2]:
                    skip_grayscale.append(old_palette[i][0])

            palette = old_palette.copy()
            for i in range(p_size, color_num):
                new_color = np.random.randint(0, 255, (1, 1), dtype=np.uint8)
                # keep generate new color that is not used yet
                while new_color not in skip_grayscale:
                    new_color = np.random.randint(0, 255, (1, 1), dtype=np.uint8)
                    skip_grayscale.append(int(new_color))
                # extend its demension size to 3 channels
                new_color = np.repeat(new_color, 3, axis=-1)
                # append to the end of new palette
                palette = np.append(palette, new_color, axis=0)
        # always reset the first part of new palette to the old palette, no matter what we have done
        palette[0 : p_size] = old_palette

    else:
        # generally, we need to generate a palette which size is eq or greater than the fill map size
        if grayscale:
            # according to the five color theroem, this palette should be enough
            # but only use five colors will make the extraction of fill map from color image impossible or non-trivial
            # fixed = ["#EEEEEE", "#CCCCCC", "#AAAAAA", "#999999", "#666666"]
            assert color_num < 256
            step = 255 // color_num
            palette = np.array([[i, i, i] for i in range(255, -1, -step)])
        else:
            # we probably will not use this branch anymore, but just keep it here in case some day we need a random
            # color map again
            fixed = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", 
                    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                    "#17becf", "#9edae5"]
            fixed, p_size = to_digital_palette(fixed)
            palette = np.random.randint(0, 255, (color_num, 3), dtype=np.uint8) 
            palette[0 : p_size] = fixed

    return palette

def to_digital_palette(text_palette):
    '''
    A helper function to convert text palette to digital palette
    '''
    for i in range(len(text_palette)):
        assert len(text_palette[i]) == 7
        text_palette[i] = text_palette[i].replace("#", "")
        color = []
        for j in range(0, len(text_palette[i]), 2):
            color.append(int(text_palette[i][j:j+2], 16))
        text_palette[i] = color
    digital_palette = np.array(text_palette, dtype = np.uint8)
    p_size = len(digital_palette)
    return digital_palette, p_size

def stroke_to_label(fill_map, stroke_map, stroke_color, fill_palette, precise=False):
    '''
    Given:
        fill_map, labeled region map
        stroke_map, a numpy array that contains select strokes with the same color
    Return:
        the region index of which are selected by stroke map
    '''
    # threshold the stroke map
    fill, palette = fill_palette
    if len(stroke_map.shape) == 3:
        stroke_map = cv2.cvtColor(stroke_map, cv2.COLOR_BGR2GRAY)
    stroke = stroke_map.copy()
    stroke[stroke_map < 250] = 0
    stroke[stroke_map >= 250] = 1

    
    labels, labels_count = np.unique(fill_map[stroke == 0], return_counts = True)
    if precise == False:
        _, smap = cv2.connectedComponents((255 - stroke*255).astype(np.uint8), connectivity=8)
        stroke_label = np.unique(smap)
        labels_new = []
        for l in stroke_label:
            if l == 0: continue
            mask = smap == l
            labels_new.append(select_labels(fill_map, mask, stroke_color, (fill, palette)))
        labels = np.unique(np.concatenate(labels_new))
    return labels

    '''
    Old codes of the label seletcion
    '''
    # # get the labels selected by stroke map
    # # definitely, this rule need to be imporved
    # labels, labels_count = np.unique(fill_map[stroke == 0], return_counts = True)
    # if precise == False:
    #     labels_ratio = labels_count / labels_count.max()
    #     criteria1 = labels_ratio > 0.1
    #     criteria2 = labels_count > 256
    #     if (stroke == 0).sum() > 64**2:
    #         labels = labels[np.logical_or(criteria1, criteria2)]
    #     else:
    #         labels = labels[np.logical_and(criteria1, criteria2)]
    # # labels = labels[labels != 0]

    # return labels

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
    A helper function to find the label with largest region size among selected regions
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
    if max_idx == -1:
        # if max_idx is -1, that means no label is selected, there must be something wrong
        print("Warning:\tno region is selected, please debug the code!")
        max_label = -1
    else:
        max_label = labels_sorted[max_idx]

    return max_label

def is_split_safe(mask, stroke):
    '''
    A helper function that to test if the given region could be splited safely
        if adding the stroke will not change the connectivity of fill_map, then it is safe to split
    '''
    # merge stroke to mask
    mask = mask.copy().astype(np.uint8)
    mask[stroke <= 240] = 1
    _, mask_regions = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if len(np.unique(mask_regions)) > 2:
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
    fill_map = thinning(fill_map)
    fill_map_artist = thinning(fill_map_artist)

    line_artist = Image.open("./trapped_ball/examples/01.png")
    # line_neural = Image.open("./trapped_ball/examples/01_sim.png")
    # line_neural = line_neural.resize(line_artist.size)
    # line_neural = fillmap_masked_line(fill_map, line_artist)
    # line_neural = add_alpha(line_neural, line_color = "9ae42c")
    # Image.fromarray(line_neural).save("11.png")

    palette = init_palette(100)

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test01.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    print("Done")
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # merge
    merge_map = np.array(Image.open("./trapped_ball/examples/01_merge_test02.png").convert("L"))
    result = merge(fill_map, merge_map, palette)
    print("Done")
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split_auto
    split_map_auto = Image.open("./trapped_ball/examples/01_split_auto_test01.png")
    split_map_auto = add_white(split_map_auto).convert("L")
    split_map_auto = np.array(split_map_auto)

    result = split_auto(fill_map, fill_map_artist, split_map_auto, line_artist, palette)
    print("Done")
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
    # artist_line = np.array(Image.open("./trapped_ball/examples/02.png").convert("L"))
    palette = init_palette(100)

    # split manual
    split_map_manual = np.array(Image.open("./trapped_ball/examples/02_split_manual_test01.png").convert("L"))
    result = split_manual(fill_map, fill_map_artist, split_map_manual, palette)
    fill = result['fill_color']
    Image.fromarray(fill).show()
    os.system("pause")

    # split manual
    split_map_manual = np.array(Image.open("./trapped_ball/examples/02_split_manual_test02.png").convert("L"))
    result = split_manual(fill_map, fill_map_artist, split_map_manual, palette)
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
    # initial_nets()
    # results = run_single(line_artist, "512", 1)
    
    '''
    test for merge
    '''
    test_case1()
    
    '''
    test for auto split and manual split
    '''
    # test_case2()
