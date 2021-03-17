# need to write some test case
import requests
import base64
import io
import json

from os.path import *
from server import to_pil
from io import BytesIO
from PIL import Image

url = "http://jixuanzhi.asuscomm.com:8080/"
image = "./trapped_ball/examples/01.png"
run_single_test = "flatsingle"
run_multi_test = "flatmultiple"
merge_test = "merge"
split_auto_test = "splitauto"
split_manual_test = "splitmanual"
show_fillmap_test = "showfillmap"

def png_to_base64(path_to_img):
    with open(path_to_img, 'rb') as f:
        return base64.encodebytes(f.read()).decode("utf-8")

def test_case1():
    # case for run single test
    data = {}
    data['image'] = png_to_base64(image)
    data['net'] = '1024_base'
    data['radius'] = 1
    data['preview'] = False

    # convert to json
    result = requests.post(url+run_single_test, json = json.dumps(data))
    if result.status_code == 200:
        result = result.json()
    line_sim = to_pil(result['line_simplified'])
    line_sim.show()


    print("Done")

def to_pil(byte):
    '''
    A helper function to convert byte png to PIL.Image
    '''
    byte = base64.b64decode(byte)
    return Image.open(BytesIO(byte))

test_case1()