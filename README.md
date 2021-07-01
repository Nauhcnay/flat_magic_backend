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
- scikit-image
- torch
- torchvision

You can install these dependencies via [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Miniconda is faster to install. (On Windows, choose the 64-bit Python 3.x version. Launch the Anaconda shell from the Start menu and navigate to this directory.)
Then:

    conda env create -f environment.yml
    conda activate flatting

To update an already created environment if the `environment.yml` file changes or to change environments, activate and then run `conda env update --file environment.yml --prune`.

### 2. Download pretrained models
Download the [pretrained network model](https://drive.google.com/file/d/15l3wPO4WbMk0DmqR7reSU1mHzOaaBCfQ/view?usp=sharing) and unzip `checkpoints.zip` into `./src/flatting/`.

### 3. Run

    cd src
    python -m flatting

### 4. Package

Use `briefcase` [commands](https://docs.beeware.org/en/latest/tutorial/tutorial-1.html) for packaging. Briefcase can't compile Cython modules, so you must first do that. There is only one. Compile it via `cythonize -i src/flatting/trapped_ball/adjacency_matrix.pyx`.

To start the process, run:

    briefcase create
    briefcase build

To run the standalone program:

    briefcase run

To create an installer:

    briefcase package

To update the standalone program when your code or dependencies change:

    briefcase update -r -d

You can also simply run `briefcase run -u`.

To debug this process, you can run your code from the entrypoint briefcase uses:

    briefcase dev

This reveals some issues important to debug. It doesn't reveal dependency issues, because it's not using briefcase's python installation.

On my setup, I have to manually edit `edit macOS/app/Flatting/Flatting.app/Contents/Resources/app_packages/torch/distributed/rpc/api.py` to insert a line `if docstring is None: continue` after line 443:

    assert docstring is not None, "RRef user-facing methods should all have docstrings."

### 4b. Packaging (without briefcase)

If briefcase doesn't work, you can use [pyinstaller](https://www.pyinstaller.org/):

    pyinstaller --noconfirm flatting_server.spec

PyInstaller is not listed as a conda dependency, because it's optional. Manually install it via: `conda install -c conda-forge pyinstaller`.

### 5. Install Photoshop plugin
Download the [flatting plugin](https://drive.google.com/file/d/1HivdqU2Z2dIL2MvqzEYmCLO2_nDL2Cnk/view?usp=sharing) and unzip it to any place. 
Download the backend server by following the instructions inside the "flatting plugin.zip"
