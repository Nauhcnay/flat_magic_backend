# Flatting
This project is based on [U-net](https://github.com/milesial/Pytorch-UNet)

## Install
### 1. Install required packages
To run color filling, you need the following module installed:

- numpy
- opencv-python
- tqdm
- pillow
- cython
- aiohttp
- skimage

And you will need the following modules to train and test line simplification network:

- pytorch

You can install these dependencies via [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Miniconda is faster to install. (On Windows, choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to this directory.)
Then:

    conda env create -f environment.yml
    conda activate flatting

To update an already created environment if the `environment.yml` file changes or to change environments, activate and then run `conda env update --file environment.yml --prune`.

### 2. Download pertrained models
Download the [pertrained network model](https://drive.google.com/file/d/15l3wPO4WbMk0DmqR7reSU1mHzOaaBCfQ/view?usp=sharing) and put it into `./`, then:

`unzip checkpoints.zip`

## Color filling APIs (continues updating)
### 1. Refresh nerual models
Send a empty post request to `ServAdd/refreshnet`. It will reload all neural network models and return True (in json) if success, False if failed.
A client could call this to reload the up-to-date models.

### 2. Get initail filling result

#### **Input**

post request should be a python dictionary `data` in JSON send to `ServAdd/flatsingle`, it should contain:

>  `data["image"]`, a base64 encoded png format image as the input artist line art

>  `data["net"]`, a string as `512` to indicate the name of neural model

>  `data["radius"]`, a int as 1

>  `data["preview"]`, a bool variable as False

#### **Return**

Then the API will return a ptyhon dictionary `result` in JSON, it will contain:

>  `result['line_artist']`, a base64 encoded png format image as the normalized artist line (alpha channel added)

>  `result['line_simplified']`, a base64 encoded png format image as the normalized simplified line art (alpha channel added, line color set to 9ae42c)

>  `result['image']`, a base64 encoded png format image as the output filling result

>  `result['fillmap']`, a 2D list as labelled fill map 

>  `result['fillmap_c']`, a 2D list as labelled component fill map 

>  `result['palette']`, a 2D (n by 3) list as the mapping from fill map label to pixel color


### 3. Merge filling regions

#### **Input**
post request should be a python dictionary `data` in JSON send to `ServAdd/merge`

>  `data["line_artist"]`, a base64 encoded png format image as the artist line

>  `data["fillmap"]`, a 2D list as the fill map which user want to work with

>  `data["stroke"]`, a base64 encoded png format image as the user input merge stroke (the stroke could be any color except 255)

>  `data["palette"]`, a 2D (n by 3) list as the mapping from fill map label to pixel color

#### **Output**
Then the API will return a ptyhon dictionary `result` in JSON, it will contain:

>  `result['line_simplified']`, a base64 encoded png format image as the **updated** simplified line
  
>  `result['image']`, a base64 encoded png format image as the output filling result after merge

>  `result['fillmap']`, a 2D list as labelled fill map after merge

>  `result['layers']`, a list which contains base64 encoded png image for each individual region in the fill map after merge

>  `result['palette']`, a 2D (n by 3) list as the mapping from fill map label to pixel color

### 4. Split filling regions

#### 4.1 **Coarse split**

#### **Input**
post request should be a python dictionary `data` in JSON send to `ServAdd/splitauto`, it should contain:

>  `data["line_artist"]`, a base64 encoded png format image as the artist line

>  `data["fillmap"]`, a 2D list as the fill map which user want to work with
  
>  `data["fillmap_artist"]`, a 2D list as the corresponding component fill map

>  `data["stroke"]`, a base64 encoded png format image as the user input split stroke

>  `data["palette"]`, a 2D (n by 3) list as the mapping from fill map label to pixel color

#### **Output**
Then the API will return a ptyhon dictionary `result` in JSON, it will contain:

>  `result['image']`, a base64 encoded png format image as the output filling result after split
  
>  `result['line_simplified']`, a base64 encoded png format image as the **modified** simplified line

>  `result['fillmap']`, a 2D list as labelled fill map after split

>  `result['layers']`, a list which contains base64 encoded png image for each individual region in the fill map after split

>  `result['palette']`, a 2D (n by 3) list as the mapping from fill map label to pixel color
  
#### 4.2 **Fine split**

#### **Input**

post request should be a python dictionary `data` in JSON send to `ServAdd/splitmanual`, it should contain:

>  `data["line_artist"]`, a base64 encoded png format image as the artist line

>  `data["line_simplified"]`, a base64 encoded png format image as the simplied line

>  `data["fillmap"]`, a 2D list as labelled fill map as the fill map which user want to work with
  
>  `data["fillmap_artist"]`, a 2D list as labelled fill map as the corresponding component fill map

>  `data["stroke"]`, a base64 encoded png format image as the user input split stroke

>  `data["palette"]`, a 2D (n by 3) list as the mapping from fill map label to pixel color


#### **Output**
Then the API will return a ptyhon dictionary `result` in JSON, it will contain:

>  `result['line_artist']`, a base64 encoded png format image as the **modified** artist line

>  `result['line_simplified']`, a base64 encoded png format image as the **modified** simplified line

>  `result['image']`, a base64 encoded png format image as the output filling result after split

>  `result['fillmap']`, a 2D list as labelled fill map after split

>  `result['layers']`, a list which contains base64 encoded png image for each individual region in the fill map after split

>  `result['palette']`, a 2D (n by 3) list as the mapping from fill map label to pixel color


## Test case
### Merging and spliting test
Here are some test case that run locally to show how merge and split function works. To repeat those examples, please:

Download the [test input](https://drive.google.com/file/d/1wVB4zPOWiVXmSwItq1Dq1px2zdobZsfB/view?usp=sharing) and put it into `./trapped_ball`, then:

`cd trapped_ball`

`unzip examples.zip`

`cd ..`

`python flatting_api.py`

### Trapped ball filling and bleeding removal test
Simply run the command below:

`cd trapped_ball`

`python run.py --exp4`

## Train line simplification network

## Test the whole filling pipline locally in a web demo
