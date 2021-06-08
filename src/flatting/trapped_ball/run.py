
from trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, show_fill_map, merger_fill_2nd
from trappedball_fill import get_ball_structuring_element, extract_line, to_masked_line
from thinning import thinning
from skimage.morphology import skeletonize
from PIL import Image
from tqdm import tqdm

import argparse
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from os.path import *

# use cython adjacency matrix
try:
    import adjacency_matrix
## If it's not already compiled, compile it.
except:
    import pyximport
    pyximport.install()
    import adjacency_matrix

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

def generate_masked_line(line_simplify, line_artist, line_artist_fullsize):
        line_masked = to_masked_line(line_simplify, line_artist, rk1=1, rk2=1, tn=1)

        # remove isolate points
        # it is not safe to do that at down scaled size
        # _, result = cv2.connectedComponents(255 - line_masked, connectivity=8)

        # up scale masked line to full size
        line_masked_fullsize_t = cv2.resize(line_masked.astype(np.uint8), 
                                (line_artist_fullsize.shape[1], line_artist_fullsize.shape[0]), 
                                interpolation = cv2.INTER_NEAREST)

        # maske with fullsize artist line again
        line_masked_fullsize = to_masked_line(line_masked_fullsize_t, line_artist_fullsize, rk1=7, rk2=1, tn=2)

        # remove isolate points
        _, temp = cv2.connectedComponents(255 - line_masked_fullsize, connectivity=4)
        
        def remove_stray_points(fillmap, drop_thres = 32):
            ids = np.unique(fillmap)
            result = np.ones(fillmap.shape) * 255

            for i in tqdm(ids):
                if i == 0: continue
                if len(np.where(fillmap == i)[0]) < drop_thres:
                    # set them as background
                    result[fillmap == i] = 255
                else:
                    # set them as line
                    result[fillmap == i] = 0
            
            return result

        line_masked_fullsize = remove_stray_points(temp, 16)

        return line_masked_fullsize


def region_get_map(path_to_line_sim, 
                path_to_line_artist=None,
                output_path=None, 
                radius_set=[3,2,1],
                percentiles=[90, 0, 0], 
                visualize_steps=False,
                return_numpy=False, 
                preview=False):
    '''
    Given:
        the path to input png file
    Return:
        the initial region map as a numpy matrix
    '''
    def read_png(path_to_png, to_grayscale=True):
        '''
        Given:
            path_to_png, it accept be any type of input, path, numpy array or PIL Image
        Return:
            the numpy array of a image
        '''

        # if it is png file, open it
        if isinstance(path_to_png, str):
            # get file name
            _, file = os.path.split(path_to_png)
            name, _ = os.path.splitext(file)

            print("Log:\topen %s"%path_to_png)
            img_org = cv2.imread(path_to_png, cv2.IMREAD_COLOR)
            if to_grayscale:
                img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            else:
                img = img_org
        
        elif isinstance(path_to_png, Image.Image):
            if to_grayscale:
                path_to_png = path_to_png.convert("L")
            img = np.array(path_to_png)
            name = "result"

        elif isinstance(path_to_png, np.ndarray):
            img = path_to_png
            if len(img.shape) > 2 and to_grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            name = "result"

        else:
            raise ValueError("The input data type %s is not supported"%str(type(path_to_png)))

        return img, name

    # read files
    img, name = read_png(path_to_line_sim)
    line_artist_fullsize, _ = read_png(path_to_line_artist)
    line_artist_fullsize = cv2.adaptiveThreshold(line_artist_fullsize, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    
    print("Log:\ttrapped ball filling")
    # line_simplify = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    _, line_simplify = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    fills = []
    result = line_simplify # this should be line_simplify numpu array
    
    if len(line_artist_fullsize.shape) == 3:
        line = cv2.cvtColor(line_artist_fullsize, cv2.COLOR_BGR2GRAY)
    else:
        line = line_artist_fullsize
    
    # may be resize the original line is not a good idea
    if line.shape[:2] != line_simplify.shape[:2]:
        line = cv2.resize(line, (line_simplify.shape[1],line_simplify.shape[0]), 
                        interpolation = cv2.INTER_AREA)
    _, line_artist = cv2.threshold(line, 125, 255, cv2.THRESH_BINARY)
    assert len(radius_set) == len(percentiles)

    for i in range(len(radius_set)):
        fill = trapped_ball_fill_multi(result, radius_set[i], percentile=percentiles[i])    
        fills += fill
        result = mark_fill(result, fill)
        if visualize_steps:
            cv2.imwrite("%d.r%d_per%.2f.png"%(i+1, radius_set[i], percentiles[i]), 
                show_fill_map(build_fill_map(result, fills)))            

    # maybe we should add a early stop
    fill = flood_fill_multi(result)
    fills += fill

    # give each region a identify number
    # could we do something to imporve the fill map?
    fillmap_neural = build_fill_map(result, fills)
    if visualize_steps:
        i+=1
        cv2.imwrite("%d.final_fills.png"%i, show_fill_map(fillmap_neural))
    
    # merge small pieces into large region, but what is the mergeing stradegy?
    fillmap_neural = merge_fill(fillmap_neural)
    
    if visualize_steps:
        i+=1
        cv2.imwrite("%d.merged.png"%i, show_fill_map(fillmap_neural))
    
    fillmap_neural = thinning(fillmap_neural)
    
    if visualize_steps:
        i+=1
        cv2.imwrite("%d.fills_final.png"%i, show_fill_map(fillmap_neural))

    fillmap_neural_fullsize = cv2.resize(fillmap_neural.astype(np.uint8), 
                                (line_artist_fullsize.shape[1], line_artist_fullsize.shape[0]), 
                                interpolation = cv2.INTER_NEAREST)
    fillmap_neural_fullsize = fillmap_neural_fullsize.astype(np.int32)
    
    if preview:
        fill_neural_fullsize = show_fill_map(fillmap_neural_fullsize)
        fill_neural_fullsize[line_artist_fullsize < 125] = 0
        return Image.fromarray(fill_neural_fullsize.astype(np.uint8))


    fill_neural = show_fill_map(fillmap_neural)

    fill_neural_line = fill_neural.copy()
    fill_neural_line[line_simplify == 0] = 0

    fillmap_artist_fullsize = np.ones(fillmap_neural_fullsize.shape, dtype=np.uint8) * 255
    fillmap_artist_fullsize[line_artist_fullsize < 125] = 0
    
    _, fillmap_artist_fullsize_c = cv2.connectedComponents(fillmap_artist_fullsize, connectivity=8)

    print("Log:\tcompute cartesian product")
    fillmap_neural_fullsize_c = fillmap_neural_fullsize.copy()

    fillmap_neural_fullsize[line_artist_fullsize < 125] = 0
    fillmap_neural_fullsize = verify_region(fillmap_neural_fullsize)

    fillmap_artist_fullsize = fillmap_cartesian_product(fillmap_artist_fullsize_c, fillmap_neural_fullsize)
    fillmap_artist_fullsize[line_artist_fullsize < 125] = 0
    
    # re-order both fillmaps
    fillmap_artist_fullsize = verify_region(fillmap_artist_fullsize, True)
    # fillmap_neural_fullsize = verify_region(fillmap_neural_fullsize, True)
    
    fillmap_neural_fullsize = bleeding_removal_yotam(fillmap_neural_fullsize_c, fillmap_artist_fullsize, th=0.0002)
    fillmap_neural_fullsize[line_artist_fullsize < 125] = 0
    fillmap_neural_fullsize = verify_region(fillmap_neural_fullsize, True)
    
    # convert final result to graph
    # we have adjacency matrix, we have fillmap, do we really need another graph for it?
    fillmap_artist_fullsize_c = thinning(fillmap_artist_fullsize_c)
    fillmap_neural_fullsize = thinning(fillmap_neural_fullsize)

    fill_artist_fullsize = show_fill_map(fillmap_artist_fullsize_c)
    fill_neural_fullsize = show_fill_map(fillmap_neural_fullsize)
    # fill_neural_fullsize[line_artist_fullsize < 125] = 0

    if output_path is not None:

        print("Log:\tsave final fill at %s"%os.path.join(output_path, str(name)+"_fill.png"))        
        cv2.imwrite(os.path.join(output_path, str(name)+"_fill.png"), fill_neural_fullsize)
        
        print("Log:\tsave neural fill at %s"%os.path.join(output_path, str(name)+"_neural.png"))
        cv2.imwrite(os.path.join(output_path, str(name)+"_neural.png"), fill_neural)
        
        print("Log:\tsave fine fill at %s"%os.path.join(output_path, str(name)+"_fine.png"))
        cv2.imwrite(os.path.join(output_path, str(name)+"_fine.png"), 
            show_fill_map(fillmap_artist_fullsize_c))
    
    print("Log:\tdone")
    if return_numpy:
        return fill_neural, fill_neural_line, fill_artist_fullsize, fill_neural_fullsize
    else:
        return fillmap_neural_fullsize, fillmap_neural_fullsize_c, fillmap_artist_fullsize_c

def fillmap_cartesian_product(fill1, fill2):
    '''
    Given:
        fill1, fillmap 1
        fill2, fillmap 2
    Return:
        A new fillmap based on its cartesian_product
    '''
    assert fill1.shape == fill2.shape

    if len(fill1.shape)==2:
        fill1 = np.expand_dims(fill1, axis=-1)
    
    if len(fill2.shape)==2:
        fill2 = np.expand_dims(fill2, axis=-1)
    
    # cat along channel
    fill_c = np.concatenate((fill1, fill2), axis=-1)

    # regnerate all region labels
    labels, inv = np.unique(fill_c.reshape(-1, 2), return_inverse=True, axis=0)
    labels = tuple(map(tuple, labels)) # convert array to tuple
    
    # assign a number lable to each cartesian product tuple
    l_to_r = {}
    for i in range(len(labels)):
        l_to_r[labels[i]] = i+1

    # assign new labels back to fillmap
    # https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key        
    fill_c = np.array(list(map(l_to_r.get, labels)))[inv]
    fill_c = fill_c.reshape(fill1.shape[0:2])

    return fill_c


# verify if there is no isolate sub-region in each region, if yes, split it and assign a new region id    
# Yotam: Can this function be replaced with a single call to cv2.connectedComponents()?
# Chuan: I think no, to find the bleeding regions on the bounderay, iteratively flood fill each region is necessary
def verify_region(fillmap, reorder_only=False):
    fillmap = fillmap.copy().astype(np.int32)
    labels = np.unique(fillmap)
    h, w = fillmap.shape
    # split region
    # is this really necessary？
    # yes, without this snippet, the result will be bad at line boundary
    # intuitively, this is like an "alingment" of smaller neural fill map to the large original line art
    # is it possible to crop the image before connectedComponents filling?
    next_label = labels.max() + 1
    if reorder_only == False:
        print("Log:\tsplit isolate regions in fillmap")
        for r in tqdm(labels):
            if r == 0: continue
            # inital input fill map
            region = np.ones(fillmap.shape, dtype=np.uint8)
            region[fillmap != r] = 0
            '''
            seems this get the speed even slower, sad
            need to find a better way
            '''
            # # try to split region
            # def find_bounding_box(region):
            #     # find the pixel coordination of this region       
            #     points = np.array(np.where(region == 1)).T
            #     t = points[:,0].min() # top
            #     l = points[:,1].min() # left
            #     b = points[:,0].max() # bottom
            #     r = points[:,1].max() # right
            #     return t, l, b, r
            # t, l, b, r = find_bounding_box(region)
            # region_cropped = region[t:b+1, l:r+1]
            # # fill_map_corpped = fill_map[t:b+1, l:r+1]

            _, region_verify = cv2.connectedComponents(region, connectivity=8)
            
            '''
            seems this get the speed even slower, sad
            '''
            # padding 0 back to the region
            # region_padded = cv2.copyMakeBorder(region_verify, t, h-b-1, l, w-r-1, cv2.BORDER_CONSTANT, 0)
            # assert region_padded.shape == fillmap.shape
            # region_verify = region_padded


            # split region if necessary
            label_verify = np.unique(region_verify)
            if len(label_verify) > 2: # skip 0 and the first region
                for j in range(2, len(label_verify)):
                    fillmap[region_verify == label_verify[j]] = next_label
                    next_label += 1

    # re-order regions
    assert np.unique(fillmap).max() == next_label - 1
    old_to_new = [0] * next_label
    idx = 1
    l = len(old_to_new)
    labels = np.unique(fillmap)
    for i in range(l): 
        if i in labels and i != 0:
            old_to_new[i] = idx
            idx += 1
        else:
            old_to_new[i] = 0
    old_to_new = np.array(old_to_new)
    fillmap_out = old_to_new[fillmap]
    
    # assert np.unique(fillmap_out).max()+1 == len(np.unique(fillmap_out))
    return fillmap_out

def update_adj_matrix(A, source, target):
        
    # update A, region s and max is not neigbor any more
    # assert A[source, target] == 1
    A[source, target] = 0

    # assert A[target, source] == 1
    A[target, source] = 0        

    # neighbors of s should become neighbor of max
    s_neighbors_x = np.where(A[source,:] == 1)
    s_neighbors_y = np.where(A[:,source] == 1)
    A[source, s_neighbors_x] = 0
    A[s_neighbors_y, source] = 0

    # neighbor of neighbors of s should use max instead of s
    A[s_neighbors_x, target] = 1
    A[target, s_neighbors_y] = 1
    
    return A

def merge_to_ref(fill_map_ref, fill_map_source, r_idx, result):
    
    # this could be imporved as well
    # r_idx is the region labels
    F = {} #mapping of large region to ref region
    for i in range(len(r_idx)):
        r = r_idx[i]

        if r == 0: continue
        label_mask = fill_map_source == r
        idx, count = np.unique(fill_map_ref[label_mask], return_counts=True)
        most_common = idx[np.argmax(count)]
        F[r] = most_common

    for r in r_idx:
        if r == 0: continue
        label_mask = fill_map_source == r
        result[label_mask] = F[r]

    return result

def merge_small_fast(fill_map_ref, fill_map_source, th):
    '''
    OK let's understand the improved version
        
    '''
    
    fill_map_source = fill_map_source.copy()
    fill_map_ref = fill_map_ref.copy()

    num_regions = len(np.unique(fill_map_source))
    
    # the definition of long int is different on windows and linux
    try:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int32), num_regions)
    except:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int64), num_regions)

    r_idx_source, r_count_source = np.unique(fill_map_source, return_counts=True)
    
    
    
    ## Labels should be contiguous.
    assert len(r_idx_source) == max(r_idx_source)+1
    ## A should have the same dimensions as number of labels.
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len( r_idx_source )
    ARTIST_LINE_LABEL = 0
    def get_small_region(r_idx_source, r_count_source, th):
        return set(
            # 1. size less that threshold
            r_idx_source[ r_count_source < th ]
            ) | set(
            # 2. not the neighbor of artist line
            ## Which of `r_idx_source` have a 0 in the adjacency position for `ARTIST_LINE_LABEL`?
            r_idx_source[ A[r_idx_source,ARTIST_LINE_LABEL] == 0 ]
            )
    
    r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)
    
    stop = False
    
    while len(r_idx_source_small) > 0 and stop == False:
        
        stop = True

        for s in r_idx_source_small:
            if s == ARTIST_LINE_LABEL: continue

            neighbors = np.where(A[s,:] == 1)[0]
            
            # remove line regions
            neighbors = neighbors[neighbors != ARTIST_LINE_LABEL]

            # skip if this region doesn't have neighbors
            if len(neighbors) == 0: continue
            
            # find region size 
            # sizes = np.array([get_size(r_idx_source, r_count_source, n) for n in neighbors]).flatten()
            sizes = r_count_source[ neighbors ]

            # merge regions if necessary
            largest_index = np.argmax(sizes)
            if neighbors[largest_index] == ARTIST_LINE_LABEL and len(neighbors) > 1:
                # if its largest neighbor is line skip it
                del neighbors[ largest_index ]
                del sizes[ largest_index ]
            
            if len(neighbors) >= 1:
                label_mask = fill_map_source == s
                max_neighbor = neighbors[np.argmax(sizes)]
                A = update_adj_matrix(A, s, max_neighbor)
                fill_map_source[label_mask] = max_neighbor
                stop = False
            else:
                continue
        
        r_idx_source, r_count_source = np.unique(fill_map_source, return_counts=True)
        r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)

        '''
        for debug
        after the first for loop, these 3 variable should have exactly same value compare to the merge_small_fast2's result
        '''
        # return r_idx_source_small, r_idx_source, r_count_source
        # return fill_map_source

    return fill_map_source

def merge_small_fast2(fill_map_ref, fill_map_source, th):
    '''

    '''
    
    fill_map_source = fill_map_source.copy()
    fill_map_ref = fill_map_ref.copy()

    num_regions = len(np.unique(fill_map_source))
    
    # the definition of long int is different on windows and linux
    try:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int32), num_regions)
    except:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int64), num_regions)

    r_idx_source, r_count_source = np.unique(fill_map_source, return_counts=True)
    ## Convert them to masked arrays
    # why?
    r_idx_source = np.ma.masked_array( r_idx_source )
    r_count_source = np.ma.masked_array( r_count_source )
    
    
    ## Labels should be contiguous.
    assert len(r_idx_source) == max(r_idx_source)+1
    ## A should have the same dimensions as number of labels.
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len( r_idx_source )
    ARTIST_LINE_LABEL = 0
    def get_small_region(r_idx_source, r_count_source, th):
        return set(
            # 1. size less that threshold
            r_idx_source[ r_count_source < th ].compressed()
            ) | set(
            # 2. not the neighbor of artist line
            ## Which of `r_idx_source` have a 0 in the adjacency position for `ARTIST_LINE_LABEL`?
            r_idx_source[ A[r_idx_source,ARTIST_LINE_LABEL] == 0 ].compressed()
            )
    
    r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)
    
    # since the region labels are always continous numbers, so it is safe to create a remap array like this
    # in other word, r_idx_source.max() + 1 == len(r_idx_source)
    remap = np.arange(len(r_idx_source))
    
    stop = False
    
    while len(r_idx_source_small) > 0 and stop == False:
        
        stop = True

        for s in r_idx_source_small:
            if s == ARTIST_LINE_LABEL: continue

            neighbors = np.where(A[s,:] == 1)[0]
            
            # remove line regions
            neighbors = neighbors[neighbors != ARTIST_LINE_LABEL]

            # skip if this region doesn't have neighbors
            if len(neighbors) == 0: continue
            
            # find region size 
            # sizes = np.array([get_size(r_idx_source, r_count_source, n) for n in neighbors]).flatten()
            sizes = r_count_source[ neighbors ]

            # merge regions if necessary
            largest_index = np.argmax(sizes)
            if neighbors[largest_index] == ARTIST_LINE_LABEL and len(neighbors) > 1:
                # if its largest neighbor is line skip it
                del neighbors[ largest_index ]
                del sizes[ largest_index ]
            
            if len(neighbors) >= 1:
                max_neighbor = neighbors[np.argmax(sizes)]
                A = update_adj_matrix(A, s, max_neighbor)
                # record the operation of merge
                remap[s] = max_neighbor
                # update the region size
                r_count_source[max_neighbor] = r_count_source[max_neighbor] + r_count_source[s]
                # remove the merged region, however, we should keep the index unchanged
                r_count_source[s] = np.ma.masked
                r_idx_source[s] = np.ma.masked
                stop = False
            else:
                continue
        
        r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)

        '''
        for debug
        after the first for loop, these 3 variable should have exactly same value compare to the merge_small_fast2's result
        '''
        # return r_idx_source_small, r_idx_source, r_count_source
        # adjacency_matrix.remap_labels( fill_map_source, remap )
        # return fill_map_source
    
    fill_map_source = adjacency_matrix.remap_labels( fill_map_source, remap )
    
    return fill_map_source

def merge_small(fill_map_ref, fill_map_source, th):
    '''
    Given:
        fill_map_ref: 2D numpy array as neural fill map on neural line
        fill_map_source: Connected commponent fill map on artist line
        th: A threshold to identify small regions 
    Returns:
        
    '''
    
    # result_fast1 = merge_small_fast(fill_map_ref, fill_map_source, th)
    # result_fast2 = merge_small_fast2(fill_map_ref, fill_map_source, th)
    # assert ( result_fast1 == result_fast2 ).all()
    # r1, r2, r3 = merge_small_fast(fill_map_ref, fill_map_source, th)
    # s1, s2, s3 = merge_small_fast2(fill_map_ref, fill_map_source, th)
    # return result_fast1
    
    # make a copy of input, we don't want to affect the array outside of this function
    fill_map_source = fill_map_source.copy()
    fill_map_ref = fill_map_ref.copy()

    num_regions = len(np.unique(fill_map_source))
    
    # the definition of long int is different on windows and linux
    try:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int32), num_regions)
    except:
        A = adjacency_matrix.adjacency_matrix(fill_map_source.astype(np.int64), num_regions)

    # find the label and size of each region
    r_idx_source, r_count_source = np.unique(fill_map_source, return_counts=True)
    
    def get_small_region(r_idx_source, r_count_source, th):
        '''
        Find the 'small' region that need to be merged to its neighbor
        '''
        r_idx_source_small = []
        for i in range(len(r_idx_source)):
        # there are two kinds of region should be identified as small region:
            # 1. size less the threshold
            if r_count_source[i] < th:
                r_idx_source_small.append(r_idx_source[i])
            # 2. not the neighbor of artist line, this type of region is not adjecent to any stroke lines, 
            # so it need to be merged to a neighbor which touch the strokes no matter how big it is
            n = np.where(A[r_idx_source[i],:] == 1)[0]
            if 0 not in n:
                r_idx_source_small.append(r_idx_source[i])
        return r_idx_source_small

    # find the small regions that need to be merged
    r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)
    
    # early stop sign
    stop = False
    
    # main loop to iteratively merge all small regions into its largest neighbor
    while len(r_idx_source_small) > 0 and stop == False:
        
        stop = True
        # each time process small regions in the list sequentially
        for s in r_idx_source_small:
            if s == 0: continue

            # get the pixel mask of region s
            label_mask = fill_map_source == s
            
            # find all neighbors of region s
            neighbors = np.where(A[s,:] == 1)[0]
            
            # remove line regions
            neighbors = neighbors[neighbors != 0]

            # skip if this region doesn't have neighbors
            if len(neighbors) == 0: continue
            
            # find region size of s's neighbors
            sizes = np.array([get_size(r_idx_source, r_count_source, n) for n in neighbors]).flatten()

            # merge regions s to its largest neighbor
            if neighbors[np.argsort(sizes)[-1]] == 0 and len(neighbors) > 1:
                # if its largest neighbor is line skip it
                max_neighbor = neighbors[np.argsort(sizes)[-2]]
                A = update_adj_matrix(A, s, max_neighbor)
                fill_map_source[label_mask] = max_neighbor
                stop = False
            elif len(neighbors) >= 1:
                # esle return its largest nerighbor
                max_neighbor = neighbors[np.argsort(sizes)[-1]]
                A = update_adj_matrix(A, s, max_neighbor)
                fill_map_source[label_mask] = max_neighbor
                stop = False
            else:
                continue
        
        # re-search the small regions for next loop
        r_idx_source, r_count_source = np.unique(fill_map_source, return_counts=True)
        r_idx_source_small = get_small_region(r_idx_source, r_count_source, th)

    # assert ( fill_map_source == result_fast2 ).all()
    return fill_map_source

def get_size(idx, count, r):
    assert r in idx
    assert r != 0

    return count[np.where(idx==r)]

def bleeding_removal_yotam(fill_map_ref, fill_map_source, th):

    fill_map_ref = fill_map_ref.copy() # connected compoenent fill map
    fill_map_source = fill_map_source.copy() # the cartesian product of connected component and neural fill map

    w, h = fill_map_ref.shape
    th = int(w * h * th)
    
    result = np.zeros(fill_map_ref.shape, dtype=np.int32)
    # 1. merge small regions which has neighbors
    # the int64 means long on linux but long long on windows, sad
    print("Log:\tmerge small regions")
    fill_map_source = merge_small_fast2(fill_map_ref, fill_map_source, th)
    
    # 2. merge large regions
    # now the fill_map_source is clean, no bleeding. but it still contains many "broken" pieces which 
    # should belong to the same semantical regions. So, we can merge these "large but still broken" region
    # together by the neural fill map.
    print("Log:\tmerge large regions")
    r_idx_source= np.unique(fill_map_source)
    result = merge_to_ref(fill_map_ref, fill_map_source, r_idx_source, result)
    
    return result

def sweep_line_merge(fillmap_neural_fullsize, fillmap_artist_fullsize, add_th, keep_th):
        
        assert fillmap_neural_fullsize.shape == fillmap_artist_fullsize.shape

        result = np.zeros(fillmap_neural_fullsize.shape)

        def to_sweep_list(fillmap):
            sweep_dict = {}
            sweep_ml = [] # most left position, which is also the sweep line's anchor
            sweep_list, sweep_count = np.unique(fillmap, return_counts=True)
            for i in range(len(sweep_list)):
                idx = sweep_list[i]
                if idx == 0: continue
                # 1. point sets 2. if have been merged 3. region area
                points = np.where(fillmap == idx)
                sweep_dict[idx] = [points, False, sweep_count[i]]
                sweep_ml.append(points[0].min())

            sweep_list = sweep_list[np.argsort(np.array(sweep_ml))]

            return sweep_list, sweep_dict

        # turn fill map to sweep list
        r_idx_ref, r_dict_ref = to_sweep_list(fillmap_neural_fullsize)
        r_idx_source, r_dict_artist = to_sweep_list(fillmap_artist_fullsize)

        skip = []
        for rn in tqdm(r_idx_ref):
            
            if rn == 0: continue

            r1 = np.zeros(fillmap_neural_fullsize.shape)
            r1[fillmap_neural_fullsize == rn] = 1

            for ra in r_idx_source:
                if ra == 0: continue

                # skip if this region has been merged
                if r_dict_artist[ra][1]: continue

                # compute iou of this two regions
                r2 = np.zeros(r1.shape)
                r2[fillmap_artist_fullsize == ra] = 1
                iou = (r1 * r2).sum()

                # compute the precentage of iou/region area
                c1 = iou/r_dict_ref[rn][2]
                c2 = iou/r_dict_artist[ra][2]

                # merge
                # r1 and r2 are quite similar, then use r2 instead of r1
                if c1 > 0.9 and c2 > 0.9:
                    result[r_dict_artist[ra][0]] = rn
                    r_dict_artist[ra][1] = True
                    continue
                
                # # r1 is almost contained by r2, the keep r1
                # elif c1 > 0.9 and c2 < 0.6:
                #     result[r_dict_ref[rn][0]] = rn
                #     # todo:
                #     # then we need refinement!

                # r2 is almost covered by r1, then merge r2 into r1
                elif c1 < 0.6 and c2 > 0.9:
                    result[r_dict_artist[ra][0]] = rn
                    r_dict_artist[ra][1] = True
                
                # r1 and r2 are not close, do nothing then
                else:
                    # we probably could record the c1 and c2, see what the parameter looks like
                    if c1 != 0 and c2 != 0:
                        skip.append((c1,c2))

        return result.astype(np.uint8), skip


def show_region(region_bit):
    plt.imshow(show_fill_map(region_bit))
    plt.show()

def get_figsize(img_num, row_num, img_size, dpi = 100):
    # inches = resolution / dpi
    # assume all image have the same resolution
    width = row_num * (img_size[0] + 200)
    height = round(img_num / row_num + 0.5) * (img_size[1] + 300)
    return width / dpi, height / dpi

def visualize(test_folder, row_num = 3):
    '''
    
    '''
    
    img_list = []
    for img in os.listdir(test_folder):
        img_list.append(img)

    img = Image.open(os.path.join(test_folder, img_list[0]))


    # visualize collected centers
    plt.rcParams["figure.figsize"] = get_figsize(len(img_list), row_num, img.size)   

    
    i = 0
    for i in range((len(img_list)//row_num + 1 if len(img_list)%row_num != 0 else len(img_list)//row_num)):
        for j in range(row_num):
            plt.subplot(len(img_list)//row_num + 1 , row_num , i*row_num+j+1)
            if i*row_num+j < len(img_list):
                # the whole function contains two steps
                # 1. get region map
                img = region_get_map(os.path.join(test_folder, img_list[i*row_num+j]))
                # 2. fill the region map
                plt.imshow(show_fill_map(img))
                plt.title(img_list[i])
    return plt

def radius_percentile_explor(radius_set, method_set, input, output):
    for radius in radius_set:
        for method in method_set:
            print("Log:\ttrying radius %d with percentile %s"%(radius, method))

            # get file name
            _, file = os.path.split(input)
            name, _ = os.path.splitext(file)

            # open image
            img_org = cv2.imread(input, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

            ret, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
            fills = []
            result = binary # this should be binary numpu array
            
            # save result
            fill = trapped_ball_fill_multi(result, radius, percentile=method)
            
            outpath = os.path.join(output, name+"_%d"%radius+"_percentail %s.png"%str(method))
            out_map = show_fill_map(build_fill_map(result, fill))
            out_map[np.where(binary == 0)]=0
            cv2.imwrite(outpath, out_map)
            
            
            # outpath = os.path.join(output, name+"_%d"%radius+"_percentail%s_final.png"%str(method))
            # out_map = show_fill_map(thinning(build_fill_map(result, fill)))
            # cv2.imwrite(outpath, out_map)
def radius_percentile_explor_repeat(radius_set, input, output, percentile_set = [100], repeat = 20):
    for r in radius_set:
        for p in percentile_set:
            _, file = os.path.split(input)
            name, _ = os.path.splitext(file)

            # open image
            img_org = cv2.imread(input, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

            ret, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
            fills = []
            result = binary # this should be binary numpu array

            for i in range(1, repeat+1):
                print("Log:\ttrying radius %d with percentile %s, No.%d"%(r, str(p), i))

                # get file name
                
                
                # save result
                fill = trapped_ball_fill_multi(result, r, percentile=p)
                fills+=fill

                outpath = os.path.join(output, name+"_%d"%r+"_percentail %s_No %d.png"%(str(p), i))
                out_map = show_fill_map(build_fill_map(result, fills))
                out_map[np.where(binary == 0)]=0
                
                cv2.imwrite(outpath, out_map)
                result = mark_fill(result, fill)

def trappedball_2pass_exp(path_line, path_line_sim, save_file=False):
    
    # open image
    line = cv2.imread(path_line, cv2.IMREAD_COLOR)
    line = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    line_sim = cv2.imread(path_line_sim, cv2.IMREAD_COLOR)
    line_sim = cv2.cvtColor(line_sim, cv2.COLOR_BGR2GRAY)

    _, line = cv2.threshold(line, 220, 255, cv2.THRESH_BINARY)
    _, binary = cv2.threshold(line_sim, 220, 255, cv2.THRESH_BINARY)
    
    result = binary
    fills = []

    # filling
    fill = trapped_ball_fill_multi(result, 1, percentile=0)    
    fills += fill
    result = mark_fill(result, fill)

    # fill rest region
    fill = flood_fill_multi(result)
    fills += fill
    
    # merge
    fillmap = build_fill_map(result, fills)
    fillmap = merge_fill(fillmap)

    # thin
    fillmap = thinning(fillmap)

    # let's do 2nd pass merge!
    fillmap_full = cv2.resize(fillmap.astype(np.uint8), 
                        (line.shape[1], line.shape[0]), 
                        interpolation = cv2.INTER_NEAREST)

    # construct a full mask
    line_sim_scaled = cv2.resize(line_sim.astype(np.uint8), 
                        (line.shape[1], line.shape[0]), 
                        interpolation = cv2.INTER_NEAREST)
    # line_full =  cv2.bitwise_and(line, line_sim_scaled)
    line_full = line

    # fillmap_full[np.where(line_full<220)] = 0
    fillmap_full[line_full<220] = 0
    fillmap_full = merger_fill_2nd(fillmap_full)[0]
    # fillmap_full = thinning(fillmap_full)

    '''
    save results
    '''
    if save_file:
        # show fill map
        fill_scaled = show_fill_map(fillmap)
        fill_scaled_v1 = show_fill_map(fillmap_full)
        fill_full = cv2.resize(fill_scaled.astype(np.uint8), 
                            (line.shape[1], line.shape[0]), 
                            interpolation = cv2.INTER_NEAREST)
        line_scaled = cv2.resize(line.astype(np.uint8), 
                            (line_sim.shape[1], line_sim.shape[0]), 
                            interpolation = cv2.INTER_NEAREST)
        
        # overlay strokes
        fill_scaled[np.where(line_scaled<220)] = 0
        fill_scaled_v1[np.where(line<220)] = 0
        fill_full[np.where(line<220)]=0

        # save result
        cv2.imwrite("fill_sacled.png", fill_scaled)
        cv2.imwrite("fill_scaled_v1.png", fill_scaled_v1)
        cv2.imwrite("fill_full.png", fill_full)

    return fillmap_full
if __name__ == '__main__':

    __spec__ = None
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--single", action = 'store_true', help="process and save a single image to output")
    parser.add_argument("--show-intermediate", action = 'store_true', help="save intermediate results")
    parser.add_argument("--visualize", action = 'store_true')
    parser.add_argument("--exp1", action = 'store_true', help="experiment of exploring the parameter")    
    parser.add_argument("--exp3", action = 'store_true', help="experiment of exploring the parameter")
    parser.add_argument("--exp4", action = 'store_true', help="experiment of exploring the parameter")    
    parser.add_argument("--exp5", action = 'store_true', help="experiment of exploring the parameter")
    parser.add_argument('--input', type=str, default="./flatting/line_white_background/image0001_line.png",
                        help = "path to input image, support png file only")
    parser.add_argument('--output', type=str, default="./exp1",
                        help = "the path to result saving folder")

    args = parser.parse_args()

    if args.single:
        bit_map = region_get_map(args.input, args.output)  
        if args.visualize:
            show_region(bit_map)
    elif args.exp1:
        # define the range of parameters
        radius_set1 = list(range(7, 15))
        method_set = list(range(0, 101, 5)) + ["mean"]
        radius_percentile_explor(radius_set1, method_set, args.input, args.output)
    elif args.exp3:
        radius_set2 = list(range(1, 15))
        radius_percentile_explor_repeat(radius_set2, args.input, "./exp3")
    elif args.exp4:
        # let's test 2 pass merge
        line = "./examples/01.png"
        line_sim = "./examples/01_sim.png"
        # trappedball_2pass_exp(line, line_sim)
        region_get_map(line_sim, 
                path_to_line_artist=line,
                output_path='./', 
                radius_set=[1],
                percentiles=[0], 
                visualize_steps=False,
                return_numpy=False)
    elif args.exp5:
        # let's test 2 pass merge
        line = "./examples/tiny.png"
        line_sim = "./examples/tiny_sim.png"
        # trappedball_2pass_exp(line, line_sim)
        region_get_map(line_sim, 
                path_to_line_artist=line,
                output_path='./', 
                radius_set=[1],
                percentiles=[0], 
                visualize_steps=False,
                return_numpy=False)
    else:
        in_path = "./flatting/size_2048/line_detection_croped"
        out_path = "./exp4"
        for img in os.listdir(in_path):
            region_get_map(join(in_path, img), out_path, radius_set=[1], percentiles=[0])          
