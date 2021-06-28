# we need to import these modules in the first level, otherwise pyinstaller will not be able to import them
# import sys, os
# import pathlib
# sys.path.append(pathlib.Path(__file__).parent.absolute()/"flatting")
# sys.path.append(pathlib.Path(__file__).parent.absolute()/"flatting"/"trapped_ball")
# from aiohttp import web
# from PIL import Image
# from io import BytesIO
# import numpy as np
# import flatting_api
# import flatting_api_async
# import base64
# import io
# import json
# import asyncio
# import multiprocessing
# import cv2
# import torch
# from pathlib import Path
# from os.path import *
# from run import region_get_map, merge_to_ref, verify_region
# from thinning import thinning
# from predict import predict_img
# from unet import UNet
# import asyncio
# from concurrent.futures import ProcessPoolExecutor
# import functools

if __name__ == '__main__':
    from flatting import app
    app.main()
