# Flatting

This project is based on [U-net](https://github.com/milesial/Pytorch-UNet)

## Install
To run color filling, you need the following module installed:

- numpy
- opencv-python
- tqdm
- pillow
- cython

And you will need additonal these modules to trian and test line simplification network:

- pytorch

## Train

## Test

## Demo

## Color filling APIs
### Color leaking remove
To repeat the color leaking remove example, please go to ./trapped_ball

`cd trapped_ball`

Then run:

`python run.py --exp4`

### Interactive color filling test case
To repeat this test case, please download the [test input](https://drive.google.com/file/d/1wVB4zPOWiVXmSwItq1Dq1px2zdobZsfB/view?usp=sharing) and put it into ./trapped_ball, then:

`cd trapped_ball`

`unzip examples.zip`

 `python faltting_api.py`
