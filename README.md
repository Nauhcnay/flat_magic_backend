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

And you will need the following modules to trian and test line simplification network:

- pytorch


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

post request should be a python dictionary `data` in JSON send to `ServAdd/split_manual`, it should contain:

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
Here are some test case that run locally to show how merge and split function works. To repeat those examples, please:

Download the [test input](https://drive.google.com/file/d/1wVB4zPOWiVXmSwItq1Dq1px2zdobZsfB/view?usp=sharing) and put it into `./trapped_ball`, then:

`cd trapped_ball`

`unzip examples.zip`

`cd ..`

`python flatting_api.py`

## Train line simplification network

## Test the whole filling pipline locally in a web demo
