import cv2
import numpy as np
import pdb
import pickle
import time
import pyximport; pyximport.install()
import adjacency_matrix
# it seemed that multi thread will not help to reduce running time
# https://medium.com/python-experiments/parallelising-in-python-mutithreading-and-mutiprocessing-with-practical-templates-c81d593c1c49
from multiprocessing import Pool
from multiprocessing import freeze_support
from functools import partial
from skimage.morphology import skeletonize, thin



from tqdm import tqdm
from PIL import Image

def save_obj(fill_graph, save_path='fill_map.pickle'):
    
    with open(save_path, 'wb') as f:
        pickle.dump(fill_graph, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(load_path='fill_map.pickle'):
    
    with open(load_path, 'rb') as f:
        fill_graph = pickle.load(f)

    return fill_graph

def extract_line(fills_result):
    
    img = cv2.blur(fills_result,(5,5))
    
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
    return 255-edges

def to_masked_line(line_sim, line_artist, rk1=None, rk2=None, ak=None, tn=1):
    '''
    Given:
        line_sim, simplified line, which is also the neural networks output
        line_artist, artist line, the original input
        rk, remove kernel, thicken kernel. if the neural network ouput too thin line, use this option
        ak, add kernel, thinning kernel. if the neural network output too thick line, use this option
    Return:
        the masked line for filling
    '''
    # 1. generate lines removed unecessary strokes
    if rk1 != None:
        kernel_remove1 = get_ball_structuring_element(rk1)
        # make the simplified line to cover the artist's line
        mask_remove = cv2.morphologyEx(line_sim, cv2.MORPH_ERODE, kernel_remove1)
    else:
        mask_remove = line_sim

    mask_remove = np.logical_and(line_artist==0, mask_remove==0)
    
    # 2. generate lines that added by line_sim
    if ak != None:
        kernel_add = get_ball_structuring_element(ak)
        # try to make the artist's line cover the simplified line
        mask_add = cv2.morphologyEx(line_sim, cv2.MORPH_DILATE, kernel_add)
    else:
        mask_add = line_sim

    # may be we don't need that skeleton
    # mask_add = 255 - skeletonize((255 - mask_add)/255, method='lee')
    
    # let's try just thin it
    mask_add = 255 - thin(255 - mask_add, max_iter=tn).astype(np.uint8)*255

    if rk2 != None:
        kernel_remove2 = get_ball_structuring_element(rk2)
        line_artist = cv2.morphologyEx(line_artist, cv2.MORPH_ERODE, kernel_remove2)
    mask_add = np.logical_and(mask_add==0, np.logical_xor(mask_add==0, line_artist==0))
    
    # 3. combine and return the result
    mask = np.logical_or(mask_remove, mask_add).astype(np.uint8)*255

    # # 4. connect dot lines if exists
    # if connect != None:
    #     kernel_con = get_ball_structuring_element(1)
    #     for _ in range(connect):
    #         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_con)

    return 255 - mask


def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).
    
    # Arguments
        radius: radius of ball shape.
             
    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def get_unfilled_point(image):
    """Get points belong to unfilled(value==255) area.

    # Arguments
        image: an image.

    # Returns
        an array of points.
    """
    y, x = np.where(image == 255)

    return np.stack((x.astype(int), y.astype(int)), axis=-1)


def exclude_area(image, radius):
    """Perform erosion on image to exclude points near the boundary.
    We want to pick part using floodfill from the seed point after dilation. 
    When the seed point is near boundary, it might not stay in the fill, and would
    not be a valid point for next floodfill operation. So we ignore these points with erosion.

    # Arguments
        image: an image.
        radius: radius of ball shape.

    # Returns
        an image after dilation.
    """
    # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)


def trapped_ball_fill_single(image, seed_point, radius):
    """Perform a single trapped ball fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
        radius: radius of ball shape.
    # Returns
        an image after filling.
    """

    ball = get_ball_structuring_element(radius)

    pass1 = np.full(image.shape, 255, np.uint8)
    pass2 = np.full(image.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(image) # why inverse image?

    # Floodfill the image
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    
    # retval, image, mask, rect = cv.floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]])
    # fill back pixles, Flood-filling cannot go across non-zero pixels in the input mask.
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4) #seed point is the first unfilled point

    # Perform dilation on image. The fill areas between gaps became disconnected.
    # close any possible gaps that could be coverd by the ball
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

    # Floodfill with seed point again to select one fill area.
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    
    # Perform erosion on the fill result leaking-proof fill.

    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)

    return pass2


def trapped_ball_fill_multi(image, radius, percentile='mean', max_iter=1000, verbo=False):
    """Perform multi trapped ball fill operations until all valid areas are filled.

    # Arguments
        image: an image. The image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        radius: radius of ball shape.
        method: method for filtering the fills. 
               'max' is usually with large radius for select large area such as background.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    if verbo:
        print('trapped-ball ' + str(radius))

    unfill_area = image # so unfill_area is the binary numpy array (but contain 0, 255 only), 0 means filled region I guess
    
    h, w = image.shape
    
    filled_area, filled_area_size, result = [], [], []
 
    for _ in range(max_iter):

        # get the point list of unfilled regions
        points = get_unfilled_point(exclude_area(unfill_area, radius))
        # points = get_unfilled_point(unfill_area)

        # terminate if all points have been filled
        if not len(points) > 0:
            break

        # perform a single flood fill
        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        
        # update filled region
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        # record filled region of each iter
        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size)

    # a filter to remove the "half" filed regions
    if percentile == "mean":
        area_size_filter = np.mean(filled_area_size)
    
    elif type(percentile)==int:
        assert percentile>=0 and percentile<=100
        area_size_filter = np.percentile(filled_area_size, percentile)
    else:
        print("wrong percentile %s"%percentile)
        raise ValueError

    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    # filter out all region that is less than the area_size_filter
    for i in result_idx:
        result.append(filled_area[i])

    # result is a list of point list for each filled region
    return result


def flood_fill_single(im, seed_point):
    """Perform a single flood fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
    # Returns
        an image after filling.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def flood_fill_multi(image, max_iter=20000, verbo=False):

    """Perform multi flood fill operations until all valid areas are filled.
    This operation will fill all rest areas, which may result large amount of fills.

    # Arguments
        image: an image. the image should contain white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    if verbo:
        print('floodfill')

    unfill_area = image
    filled_area = []

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)

        if not len(points) > 0:
            break

        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))

    return filled_area


def mark_fill(image, fills):
    """Mark filled areas with 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result


def build_fill_map(image, fills):
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an array.
    """
    result = np.zeros(image.shape[:2], np.int)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap):
    """Mark filled areas with colors. It is useful for visualization.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3), dtype=np.uint8) 
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def get_bounding_rect(points):
    """Get a bounding rect of points.

    # Arguments
        points: array of points.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2


def get_border_bounding_rect(h, w, p1, p2, r):
    """Get a valid bounding rect in the image with border of specific size.

    # Arguments
        h: image max height.
        w: image max width.
        p1: start point of rect.
        p2: end point of rect.
        r: border radius.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if 0 < x1 - r else 0
    y1 = y1 - r if 0 < y1 - r else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w # why here plus 1?
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return x1, y1, x2, y2


def get_border_point(points, rect, max_height, max_width):
    """Get border points of a fill area

    # Arguments
        points: points of fill .
        rect: bounding rect of fill.
        max_height: image max height.
        max_width: image max width.
    # Returns
        points , convex shape of points
    """
    
    # Get a local bounding rect.
    # what this function used for?
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect, all 0s
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    
    # Move points to the rect.
    # offset points into the fill
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    # pdb.set_trace()
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    # Structuring element in cross shape is used instead of box to get 4-connected border.
    '''
    # Cross-shaped Kernel
    >>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    array([[0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [1, 1, 1, 1, 1],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0]], dtype=uint8)
    '''
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) # this is a ball shape kernel
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill
    border_pixel_points = np.where(border_pixel_mask == 255)

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap, max_iter=10, verbo=False):
    """Merge fill areas.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for i in range(max_iter):
        if verbo:
            print('merge ' + str(i + 1))

        # set stroke as black
        result[np.where(fillmap == 0)] = 0

        # get list of fill id
        fill_id = np.unique(result.flatten())
        fills = []

        for j in fill_id:

            # select one region each time
            point = np.where(result == j)

            fills.append({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point)
            })

        for j, f in enumerate(fills):
            
            # ignore lines
            if f['id'] == 0:
                continue

            # get border shape of a region, but that may contains many nosiy segementation?
            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points] # pixel values or seg index of that region
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            # remove id that equal 0
            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:
                # if f['area'] < 32:
                    new_id = 0
            else:
                # region id may be set to region with largest contact
                new_id = ids[0]

            # a point, because the convex shape only contains 1 point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            # so this means 
            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:
            # if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 10000:
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:
            # if f['area'] < 5000 and len(ids) == 1:
                result[f['point']] = new_id

            if f['area'] < 50:
            # if f['area'] < 100:
                result[f['point']] = new_id

        # if no merge happen, stop this process
        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result

def search_point(points, point):

    idx = np.where((points == point).all(axis = 1))[0]

    return idx

def extract_region_obsolete(points, point, width, height):
    
    # unfortunately, this function is too costly to run

    # get 8-connectivity neighbors
    point_list = []

    # search top left
    # point[0] is height
    # point[1] is width
    if point[0] > 0 and point[1] > 0:
        tl = np.array([point[0]-1, point[1]-1])
        idx = search_point(points, tl)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(tl)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, tl, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")
    # search top
    if point[0] > 0:
        t = np.array([point[0], point[1]-1])
        idx = search_point(points, t)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(t)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, t, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search top right
    if point[0] > 0 and point[1] < width:
        tr = np.array([point[0]-1, point[1]+1])
        idx = search_point(points, tr)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(tr)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, tr, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search mid left
    if point[1] > 0:
        ml = np.array([point[0], point[1]-1])
        idx = search_point(points, ml)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(ml)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, ml, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search mid right
    if point[1] < width:
        mr = np.array([point[0], point[1]+1])
        idx = search_point(points, mr)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(mr)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, mr, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search bottom left
    if point[0] < height and point[1] > 0:
        bl = np.array([point[0]+1, point[1]-1])
        idx = search_point(points, bl)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(bl)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, bl, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search bottom
    if point[0] < height:
        b = np.array([point[0]+1, point[1]])
        idx = search_point(points, b)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(b)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, b, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    # search bottom right
    if point[0] < height and point[1] < width:
        br = np.array([point[0]+1, point[1]+1])
        idx = search_point(points, br)
        if len(idx) == 0:
            pass
        elif len(idx) == 1:
            point_list.append(br)
            # pop out current point
            points = np.delete(points, idx, axis=0)
            point_list += extract_region(points, br, width, height)
        else:
            raise ValueError("There should not exist two identical points in the list!")

    return point_list

def extract_region():
    
    # let's try flood fill

    flood_fill_multi(image, max_iter=20000)

def to_graph(fillmap, fillid):

    # how to speed up this part?
    # use another graph data structure
    # or maybe use list instead of dict

    fills = {}
    for j in tqdm(fillid):
        
        # select one region each time
        point = np.where(fillmap == j)

        fills[j] = {"point":point,
                    "area": len(point[0]),
                    "rect": get_bounding_rect(point),
                    "neighbor":[]}
    return fills

def to_fillmap(fillmap, fills):
    
    for j in fills:
        if fills[j] == None:
            continue
        fillmap[fills[j]['point']] = j

    return fillmap

def merge_list_ordered(list1, list2, idx):
    
    for value in list2:
        if value not in list1 and value != idx and value != None:
            list1.append(value)

    return list1

def merge_region(fills, source_idx, target_idx, result):
    
    # merge from source to target
    assert fills[source_idx] != None
    assert fills[target_idx] != None

    # update target region
    fills[target_idx]['point'] = (np.concatenate((fills[target_idx]['point'][0], fills[source_idx]['point'][0])),
                                np.concatenate((fills[target_idx]['point'][1], fills[source_idx]['point'][1])))

    fills[target_idx]['area'] += fills[source_idx]['area']
    assert len(fills[target_idx]['point'][0]) + len(fills[source_idx]['point'][0]) == fills[target_idx]['area'] + fills[source_idx]['area']
    
    fills[target_idx]['neighbor'] = merge_list_ordered(fills[target_idx]['neighbor'], 
                                    fills[source_idx]['neighbor'], target_idx)

    # update source's neighbor
    for n in fills[source_idx]['neighbor']:
        if n != None:
            if source_idx in fills[n]['neighbor']:
                t = fills[n]['neighbor'].index(source_idx)
                fills[n]['neighbor'][t] = None
            else:
                print("find one side neighbor")

            if target_idx not in fills[n]['neighbor'] and target_idx != n:
                fills[n]['neighbor'].append(target_idx)

    # remove source region
    fills[source_idx] = None

    return fills

def split():
    # this might be a different function
    pass

def list_region(fill_graph, th = None, verbo = True):

    regions = 0
    small_regions = []
    for key in fill_graph:
        if fill_graph[key] != None:
            
            if verbo:
                print("Log:\tregion %d with size %d"%(key, fill_graph[key]['area']))
            regions += 1

            if th == None:
                continue

            # collect small regions
            if fill_graph[key]['area'] < th:
                small_regions.append(key)

    if verbo:
        print("Log:\ttotal regions %d"%regions)

    return small_regions

def visualize_graph(fills_graph, result, region=None):
    if region == None:
        Image.fromarray(show_fill_map(to_fillmap(result, fills_graph)).astype(np.uint8)).show()
    else:
        assert region in fills_graph
        show_map = np.zeros(result.shape, np.uint8)
        show_map[fills_graph[region]['point']] = 255
        Image.fromarray(show_map).show()

def visualize_result(result, region=None):
    if region == None:
        Image.fromarray(show_fill_map(result).astype(np.uint8)).show()
    else:
        assert region in result
        show_map = np.zeros(result.shape, np.uint8)
        show_map[np.where(result == region)] = 255
        Image.fromarray(show_map).show()

def graph_self_check(fill_graph):

    for key in fill_graph:
        if fill_graph[key] != None:
            if len(fill_graph[key]['neighbor']) > 0:
                if len(fill_graph[key]['neighbor']) != len(set(fill_graph[key]['neighbor'])):
                    print("Log:\tfind duplicate neighbor!")
                for n in fill_graph[key]['neighbor']:
                    if key not in fill_graph[n]['neighbor']:
                        print("Log:\tfind missing neighbor")
            # print("Log:\tregion %d has %d points"%(key, fill_graph[key]['area']))  

def flood_fill_single_proc(region_id, img):
    
    # construct fill region
    fill_region = np.full(img.shape, 0, np.uint8)
    fill_region[np.where(img == region_id)] = 255
    return flood_fill_multi(fill_region, verbo=False)

def flood_fill_multi_proc(func, fill_id, result, n_proc):
    print("Log:\tmulti process spliting bleeding regions")
    with Pool(processes=n_proc) as p:
        return p.map(partial(func, img=result), fill_id)

def split_region(result, multi_proc=False):

    # # get list of fill id
    # fill_id = np.unique(result.flatten()).tolist()
    # fill_id.remove(0)
    # assert 0 not in fill_id

    # _, result = cv2.connectedComponents(result, connectivity=4)
    # # there will left some small regions, we can merge them into region 0 in the following step

    # # result = build_fill_map(result, fill_points)
    
    # fill_id_new = np.unique(result)

    # generate thershold of merging region
    w, h = result.shape
    th = int(w*h*0.09)
    # get list of fill id
    fill_id = np.unique(result.flatten()).tolist()
    fill_id.remove(0)
    assert 0 not in fill_id
    
    fill_points = []

    # get each region ready to be filled
    if multi_proc:
        n_proc = 8
        start = time.process_time()
        
        fill_points_multi_proc = flood_fill_multi_proc(flood_fill_single_proc, fill_id, result, n_proc)
        for f in fill_points_multi_proc:
            fill_points += f

        print("Mutiprocessing time: {}secs\n".format((time.process_time()-start)))

    else:
        # split each region if it is splited by ink region
        start = time.process_time()
        for j in tqdm(fill_id):

            # skip strokes
            if j == 0:
                continue

            # generate fill mask of that region
            fill_region = np.full(result.shape, 0, np.uint8)
            fill_region[np.where(result == j)] = 255

            # corp to a smaller region that only cover the current filling region to speed up
            # todo

            # assign new id to
            fills = flood_fill_multi(fill_region, verbo=False)

            merge = []
            merge_idx = [] 
            for i in range(len(fills)):
                if len(fills[i][0]) > th:
                    merge_idx.append(i)

            for i in range(len(merge_idx)):
                merge.append(fills[merge_idx[i]])

            for i in merge_idx:
                fills.pop(i)

            if len(merge) > 0:
                region_merged = merge.pop(0)
                for p in merge:
                    region_merged = (np.concatenate((region_merged[0], p[0])), np.concatenate((region_merged[1], p[1])))
                fills.append(region_merged)
            
            fill_points += fills
        print("Single-processing time: {}secs\n".format((time.process_time()-start)))

    result = build_fill_map(result, fill_points)
    fill_id_new = np.unique(result)

    return result, fill_id_new

def find_neighbor(result, fills_graph, max_height, max_width):
    
    fill_id_new = np.unique(result)

    for j in tqdm(fill_id_new):
        
        if j == 0:
            continue

        fill_region = np.zeros(result.shape, np.uint8)
        fill_region[np.where(result == j)] = 255
        
        # find boundary of each region
        # sometimes this function is not multually correct, why?
        border_points, _ = get_border_point(fills_graph[j]['point'], fills_graph[j]['rect'], max_height, max_width)
        
        # construct a graph map of all regions
        neighbor_id = np.unique(result[border_points])

        # record neighbor information
        for k in neighbor_id:
            if k != 0:
                if k not in fills_graph[j]["neighbor"]:
                    fills_graph[j]["neighbor"].append(k)
                if j not in fills_graph[k]["neighbor"]:
                    fills_graph[k]["neighbor"].append(j)

    return fills_graph

def find_min_neighbor(fills_graph, idx):

    neighbors = []
    neighbor_sizes = []
    for n in fills_graph[idx]["neighbor"]:
        if n != None:
            neighbors.append(n)
            neighbor_sizes.append(fills_graph[n]['area'])
        else:
            neighbors.append(n)
            neighbor_sizes.append(-1)

    # we need to sort the index
    sort_idx = sorted(range(len(neighbors)), key=lambda k: neighbor_sizes[k])
    
    # for i in sort_idx:
    #     print("Log:\tregion %d with size %d"%(neighbors[i], neighbor_sizes[i]))

    return sort_idx, neighbor_sizes

def merge_all_neighbor(fills_graph, idx, result):

    for n in fills_graph[idx]['neighbor']:
        result[fills_graph[n]['point']] = idx

    return result

def check_all_neighbor(fills_graph, j, low_th, max_th):

    min_neighbors = find_min_neighbor(fills_graph, j)
    for k in min_neighbors:
        nb = fills_graph[j]['neighbor'][k]

        if nb != None:
            if fills_graph[nb] != None and fills_graph[nb]['area'] <= low_th and nb != j: 
                continue
            else: 
                print("Log:\texclude region %d"%nb)

def remove_bleeding(fills_graph, fill_id_new, max_iter, result, low_th, max_th):
    
    count = 0
    really_low_th = 100
    # max region absorb small neighbors
    for i in range(max_iter):
        # print('merge 2nd ' + str(i + 1))    
        for j in tqdm(fill_id_new):
            if j == 0:
                continue
            if fills_graph[j] == None: # this region has been removed
                continue
            if fills_graph[j]['area'] < max_th:
                continue

            min_neighbors, min_neighbor_sizes = find_min_neighbor(fills_graph, j)
            # print("Log:\tfound region %d have %d neighbors"%(j, len(min_neighbors)))
            
            for k in min_neighbors:
                
                nb = fills_graph[j]['neighbor'][k]
                
                if min_neighbor_sizes[k] == -1:
                    continue

                if nb != None:
                    if fills_graph[nb] != None and fills_graph[nb]['area'] <= low_th and nb != j: 
                        fills_graph = merge_region(fills_graph, nb, j, result)
                        count += 1
                    else: 
                        fills_graph[j]['neighbor'][k] = None
        
    # small region join its largest neighbor
    small_regions = list_region(fills_graph, low_th, False) 
    first_loop = True
    num_samll_before = len(small_regions)
    num_samll_after = len(small_regions)
    
    while first_loop or num_samll_before - num_samll_after > 0:
        first_loop = False
        for s in small_regions:

            min_neighbors, min_neighbor_sizes = find_min_neighbor(fills_graph, s)

            if len(min_neighbors) == 0 or min_neighbors == None or min_neighbor_sizes[min_neighbors[-1]] == -1:
                if fills_graph[s]['area'] < really_low_th:
                    fills_graph = merge_region(fills_graph, s, 0, result)
                continue

            t = fills_graph[s]['neighbor'][min_neighbors[-1]]
            fills_graph = merge_region(fills_graph, s, t, result)
            count += 1

        small_regions = list_region(fills_graph, low_th, False)
        
        num_samll_before = num_samll_after
        num_samll_after = len(small_regions) 

    print("Log:\t %d neighbors merged"%count)
    
    return fills_graph


def merger_fill_2nd(fillmap, max_iter=10, low_th=0.001, max_th=0.01, debug=False):
    
    """
    next step should be using multi threading in each step
    get the function as fast as I can
    """

    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()
    low_th = int(max_height*max_width*low_th)
    max_th = int(max_height*max_width*max_th)

    # 1. convert filling map to graphs
    # this step take 99% of running time, need optimaization a lot
    if debug:
        print("Log:\tload fill_map.pickle")
        result = load_obj("fill_map.pickle")
        fill_id_new = np.unique(result)
    else:
        print("Log:\tsplit bleeding regions")
        result, fill_id_new = split_region(result)
    
    # initailize the graph of regions
    if debug:
        print("Log:\tload fills_graph.pickle")
        fills_graph_init = load_obj("fills_graph.pickle")
        fills_graph = load_obj("fills_graph.pickle")
    else:
        print("Log:\tinitialize region graph")
        fills_graph = to_graph(result, fill_id_new)

    # find neighbor
    if debug:
        print("Log:\tload fills_graph_n.pickle")
        fills_graph = load_obj("fills_graph_n.pickle")
    else:
        print("Log:\tfind region neighbors")
        fills_graph = find_neighbor(result, fills_graph, max_height, max_width)

    # self check if the graph is constructed correctly 
    graph_self_check(fills_graph)              

    # 2. merge all small region to its largest neighbor
    # this step seems fast, it only takes around 20s to finish
    print("Log:\tremove leaking color")
    fills_graph = remove_bleeding(fills_graph, fill_id_new, max_iter, result, low_th, max_th)
    
    # 3. show the refined the result
    visualize_graph(fills_graph, result, region=None)
    
    # 4. map region graph back to fillmaps
    result = to_fillmap(result, fills_graph)
    return result, fills_graph
