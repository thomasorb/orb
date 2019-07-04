# Introduction

[ORB](https://github.com/thomasorb/orb) is the kernel module for the
whole suite of data reduction and analysis tools for
[SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle):
[ORBS](https://github.com/thomasorb/orbs),
[ORCS](https://github.com/thomasorb/orcs).

It provides basic access to the data cubes as long as the fitting
engine of ORCS and numerous utilitary functions for the analysis of
interferometric and spectral data, imaging data, astrometry,
photometry.


You can find the up-to-date documentation here:

https://orb.readthedocs.io


## installation instructions with Anaconda (should work on Linux, Mac OSX, Windows)

### 1. download Miniconda for your OS and python 2.7.

**If you already have [Anaconda](https://www.anaconda.com/) installed go to step 2**

instructions are here: [Miniconda — Conda](https://conda.io/miniconda.html)
1. place the downloaded file on your home directory
2. install it (use the real file name instead of `Miniconda*.sh`)
```bash
bash Miniconda*.sh
```
### 2. install `conda-build` tools
```bash
conda install conda-build
```

### 3. create orb environment
#### automatic procedure if you have a `*.yml*` file:
create an environment automatically
```bash
conda env create -f orb-env.yml
```

#### manual installation of the librairies
create an environment and install needed modules manually
```bash
conda create -n orb python=2.7 
conda install -n orb numpy scipy bottleneck matplotlib astropy cython h5py dill pandas
conda install -n orb -c conda-forge pyregion
conda install -n orb -c astropy photutils
```
if your shell is bash, which is now generally the case for OSX and most linux distributions please do
```
conda init bash
```
If you have a different shell replace bash with the name of your shell.

Now if you have Scisoft installed (which does not goes well with Anaconda/Python... but still you may want to have it)
then please add `export PYTHONPATH=''` at the end of your profile file where some environment variables are defined (e.g. `.bashrc`, `.profile`, `.bash_profile`).

You may then activate your environment with
```
conda activate orb
```
now your prompt should be something like `(orb) $`. If `conda activate orb` does not work. Please do the following `conda init bash` (if bash is your shell

note: you may now have to install `gcc` on Mac OSX or linux if you want to be able to execute the following with no error (see https://discussions.apple.com/thread/8336714). To check that you have gcc installed type `gcc --help` in a terminal. You should see the usage informations displayed.

e.g. to install `gcc` under ubuntu (or debian based distributions):

```bash
sudo apt update
sudo apt install build-essential
```

Then you can install the last python modules
```bash
pip install gvar --no-deps
pip install lsqfit --no-deps
pip install pp --no-deps
```

### 4. add orb module

clone [ORB](https://github.com/thomasorb/orb)
```bash
mkdir orb-stable # do it where you want to put orb files
cd orb-stable
git clone https://github.com/thomasorb/orb.git
```

in the downloaded folder
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
cd path/to/orb-stable/orb
python setup.py build_ext --inplace
python setup.py install
```

Test it:
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
python -c 'import orb.core'
```

### 5. Install jupyter

```bash
conda install -n orb -c conda-forge jupyterlab
```
Run it

```bash
conda activate orb # you don't need to do it if you are already in the orb environment
jupyter lab
```
You should now have your web browser opened and showing the jupyter lab interface !

	  
