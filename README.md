# ORB

[ORB](https://github.com/thomasorb/orb) is the kernel module for the
whole suite of data reduction and analysis tools for
[SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle):
[ORBS](https://github.com/thomasorb/orbs),
[ORCS](https://github.com/thomasorb/orcs).

It provides basic access to the data cubes as long as the fitting
engine of ORCS and numerous utilitary functions for the analysis of
interferometric and spectral data, imaging data, astrometry,
photometry.


## Documentation

You can find the up-to-date documentation here:

http://celeste.phy.ulaval.ca/orb-doc/index.html



## installation instructions with Anaconda (should work on Linux, Mac OSX, Windows)

### 1. download Miniconda for your OS and python 3.7.

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

### 3. create orb3 environment

create an environment and install needed modules manually
```bash
conda create -n orb3 python=3.7 
conda install -n orb3 numpy scipy bottleneck matplotlib astropy cython h5py dill pandas
conda install -n orb3 -c conda-forge pyregion
conda install -n orb3 -c astropy photutils astroquery
conda activate orb3
```
now your prompt should be something like `(orb3) $`.
```bash
pip install gvar --no-deps
pip install lsqfit --no-deps
pip install fpdf --no-deps
```

### 4. add orb3 module

clone [ORB](https://github.com/thomasorb/orb)
```bash
mkdir orb-stable # do it where you want to put orb files
cd orb-stable
git clone https://github.com/thomasorb/orb.git
```

in the downloaded folder
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
cd path/to/orb-stable
python setup.py build_ext --inplace
python setup.py install
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orb.core'
```
### 5. add orcs module

clone [ORCS](https://github.com/thomasorb/orcs)
```bash
mkdir orcs-stable # do it where you want to put orcs files
cd orcs-stable
git clone https://github.com/thomasorb/orcs.git
```

in the downloaded folder
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
cd path/to/orcs-stable
python setup.py install
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orcs.process'
```

### 6. Install jupyter

```bash
conda install -n orb3 -c conda-forge jupyterlab
```
Run it

```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
jupyter lab
```
You should now have your web browser opened and showing the jupyter lab interface !

	  
