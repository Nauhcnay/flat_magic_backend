
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
import pyximport; pyximport.install()
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

def region_get_map(path_to_png, 
                output_png=None, 
                radius_set=[3,2,1],
                percentiles=[90, 0, 0], 
                visualize_steps=False,
                path_to_line=None,
                save_org_size=False,
                return_numpy=False):
    '''
    Given:
        the path to input svg file
    Return:
        the initial region map as a numpy matrix
    '''
    if type(path_to_png) == str:
        # get file name
        _, file = os.path.split(path_to_png)
        name, _ = os.path.splitext(file)

        # open image
        print("Log:\topen %s"%path_to_png)
        img_org = cv2.imread(path_to_png, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    else:
        img = np.array(path_to_png)
        if output_png != None:
            name = output_png[1]
            output_png = output_png[0]
        else:
            name = 0

    print("Log:\ttrapped ball filling")
    _, line_simplify = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    fills = []
    result = line_simplify # this should be line_simplify numpu array
    
    if path_to_line == None and type(path_to_png) == str:
        path_to_line = path_to_png.replace("_detection", "")
    
    if type(path_to_line) == str:
        assert exists(path_to_line)
        print("Log:\topen %s"%path_to_line)
        line_artist_fullsize = cv2.imread(path_to_line, cv2.IMREAD_COLOR)
        
    elif path_to_line != None:
        line_artist_fullsize = np.array(path_to_line)
    else:
        line_artist_fullsize = img

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

    # version1, pure filling result
    fill_neural = show_fill_map(fillmap_neural)

    # version2, filling result overlay simpified line
    fill_neural_line = fill_neural.copy()
    fill_neural_line[line_simplify == 0] = 0

    # version3, filling result overlay down scaled artist line
    fill_artist_line = fill_neural.copy()
    fill_artist_line[line_artist == 0] = 0

    # version4, up scaled filling result overlay full size artist line
    # this could work, but it is non-trivial
    # line_masked_fullsize = generate_masked_line(line_simplify, line_artist, line_artist_fullsize)

    # fillmap_neural_fullsize[line_masked_fullsize < 125] = 0
    # _, fillmap_neural_fullsize = cv2.connectedComponents(fillmap_neural_fullsize, connectivity=8)

    # version5, correct initail fillmap by maybe sweep line?
    # let's do it!
    fillmap_artist_fullsize = np.ones(fillmap_neural_fullsize.shape, dtype=np.uint8) * 255
    fillmap_artist_fullsize[line_artist_fullsize < 125] = 0
    _, fillmap_artist_fullsize = cv2.connectedComponents(fillmap_artist_fullsize, connectivity=8)

    print("Log:\trefine filling results")

    # # this also not fully worked
    # fillmap_neural_fullsize, skip = sweep_line_merge(fillmap_neural_fullsize, fillmap_artist_fullsize, add_th=0.4, keep_th=0.001)
    
    fillmap_neural_fullsize[line_artist_fullsize < 125] = 0
    fillmap_neural_fullsize = bleeding_removal_yotam(fillmap_neural_fullsize, fillmap_artist_fullsize, th=0.001)
    
    # convert final result to graph
    # we have adjacency matrix, we have fillmap, do we really need another graph for it?
    fill_artist_fullsize = show_fill_map(fillmap_artist_fullsize)
    fill_neural_fullsize = show_fill_map(fillmap_neural_fullsize)
    fill_neural_fullsize[line_artist_fullsize < 125] = 0

    if output_png is not None:

        print("Log:\tsave at %s"%os.path.join(output_png, str(name)+"_fill.png"))        
        cv2.imwrite(os.path.join(output_png, str(name)+"_fill.png"), fill_neural)
        cv2.imwrite(os.path.join(output_png, str(name)+"_fill_edge.png"), fill_neural_line)
        cv2.imwrite(os.path.join(output_png, str(name)+"_fill_line.png"), fill_artist_line)

        if save_org_size:
            cv2.imwrite(os.path.join(output_png, str(name)+"_fill_line_full.png"), fill_neural_fullsize)
    
    if return_numpy:
        return fill_neural, fill_neural_line, fill_artist_fullsize, fill_neural_fullsize
    else:
        return fillmap_neural

def bleeding_removal_yotam(fillmap_neural_fullsize, fillmap_artist_fullsize, th):
        
        w, h = fillmap_neural_fullsize.shape

        th = int(w * h * th)    

        num_regions = len(np.unique(fillmap_artist_fullsize))
        A = adjacency_matrix.adjacency_matrix(fillmap_artist_fullsize.astype(np.int64), num_regions)

        r_idx_neural, r_count_neural = np.unique(fillmap_neural_fullsize, return_counts=True)
        r_idx_artist, r_count_artist = np.unique(fillmap_artist_fullsize, return_counts=True)
        
        def get_size(idx, count, r):
            assert r in idx
            assert r != 0

            return count[np.where(idx==r)]

        F = {}
        for i in range(len(r_idx_artist)):
            r = r_idx_artist[i]

            if r == 0: continue
            label_mask = fillmap_artist_fullsize == r
            idx, count = np.unique(fillmap_neural_fullsize[label_mask], return_counts=True)
            most_common = idx[np.argmax(count)]
            F[r] = most_common

        small_regions = r_idx_artist[r_count_artist < th]

        result = np.zeros(fillmap_neural_fullsize.shape, dtype=np.int)

        for r in r_idx_artist:
            if r == 0: continue
            label_mask = fillmap_artist_fullsize == r
            result[label_mask] = F[r]

        # this may not be neccessary for result, but is necessary for further editing
        # todo
        # for s in small_regions:
        #     label_mask = fillmap_artist_fullsize == s
        #     neighbors = np.where(A[s,:] == 1)[0]
            
        #     # remove line regions, we don't need to consider that
        #     neighbors = neighbors[np.where(neighbors != 0)]

        #     if len(neighbors) == 0: continue
            
        #     sizes = np.array([get_size(r_idx_artist, r_count_artist, n) for n in neighbors])

        #     if neighbors[np.argsort(sizes)[-1]] == 0 and len(neighbors) > 1:
        #         max_neighbor = neighbors[np.argsort(sizes)[-2]]

        #     elif len(neighbors) > 1:
        #         max_neighbor = neighbors[np.argsort(sizes)[-1]]

        #     else:
        #         continue
            
        #     result[label_mask] = max_neighbor

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
        r_idx_neural, r_dict_neural = to_sweep_list(fillmap_neural_fullsize)
        r_idx_artist, r_dict_artist = to_sweep_list(fillmap_artist_fullsize)

        skip = []
        for rn in tqdm(r_idx_neural):
            
            if rn == 0: continue

            r1 = np.zeros(fillmap_neural_fullsize.shape)
            r1[fillmap_neural_fullsize == rn] = 1

            for ra in r_idx_artist:
                if ra == 0: continue

                # skip if this region has been merged
                if r_dict_artist[ra][1]: continue

                # compute iou of this two regions
                r2 = np.zeros(r1.shape)
                r2[fillmap_artist_fullsize == ra] = 1
                iou = (r1 * r2).sum()

                # compute the precentage of iou/region area
                c1 = iou/r_dict_neural[rn][2]
                c2 = iou/r_dict_artist[ra][2]

                # merge
                # r1 and r2 are quite similar, then use r2 instead of r1
                if c1 > 0.9 and c2 > 0.9:
                    result[r_dict_artist[ra][0]] = rn
                    r_dict_artist[ra][1] = True
                    continue
                
                # # r1 is almost contained by r2, the keep r1
                # elif c1 > 0.9 and c2 < 0.6:
                #     result[r_dict_neural[rn][0]] = rn
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
        line = "./examples/line.png"
        line_sim = "./examples/line_simplify.png"
        trappedball_2pass_exp(line, line_sim)
    else:
        in_path = "./flatting/size_2048/line_detection_croped"
        out_path = "./exp4"
        for img in os.listdir(in_path):
            region_get_map(join(in_path, img), out_path, radius_set=[1], percentiles=[0])          
