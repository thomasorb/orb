# ORB

[ORB](https://github.com/thomasorb/orb) is the kernel module for the
whole suite of data reduction and analysis tools for
[SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle):
[ORBS](https://github.com/thomasorb/orbs),

It provides basic access to the data cubes as long as the fitting
engine of ORCS and numerous utilitary functions for the analysis of
interferometric and spectral data, imaging data, astrometry,
photometry.


## Documentation

You can find the up-to-date documentation here:

https://orb.readthedocs.io/en/latest/



## installation instructions with Anaconda (should work on Linux, Mac OSX, Windows)

### 1. download Miniconda for your OS and python 3.7.

**If you already have [Anaconda](https://www.anaconda.com/) installed go to step 2**

instructions are here: [Miniconda — Conda](https://conda.io/miniconda.html)
1. place the downloaded file on your home directory
2. install it (use the real file name instead of `Miniconda*.sh`)
```bash
bash Miniconda*.sh
```
You may have to run
```bash
conda init bash
```

### 2. install `conda-build` tools
```bash
conda install conda-build
```

### 3. create orb3 environment

create an environment and install needed modules manually
```bash
conda create -n orb3 python=3.7 
conda install -n orb3 numpy scipy matplotlib astropy cython h5py dill pandas pytables
conda install -n orb3 -c conda-forge pyregion
conda install -n orb3 -c astropy photutils astroquery
conda install -n orb3 -c anaconda gitpython
conda activate orb3
```
now your prompt should be something like `(orb3) $`.
```bash
pip install gvar==9.2 --no-deps
pip install lsqfit==11.2 --no-deps
pip install fpdf --no-deps
pip install gitpython --no-deps
```

If you encounter a problem building gvar you may want to install build-essential (for Ubuntu users)
```
sudo apt install build-essential
```

### 4. add ORB module

clone [ORB](https://github.com/thomasorb/orb)
```bash
cd
mkdir orb-stable # this is an example and the folder can be the one you wish (but the following lines must be changed accordingly)
cd orb-stable
git clone https://github.com/thomasorb/orb.git
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
cd orb
python setup.py build_ext --inplace
python setup.py install # not for developer
```
**(developer only)**
```bash
cd
echo '/absolute/path/to/orb-stable/orb' > miniconda3/envs/orb3/lib/python3.7/site-packages/conda.pth
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orb.core'
```

### 5. install jupyter

```bash
conda install -n orb3 -c conda-forge jupyterlab
```
Run it

```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
jupyter lab
```
You should now have your web browser opened and showing the jupyter lab interface !


## Troubleshooting

### Check the version of the packages

- [Here](docs/orb3-ver.txt) is a list of all the versions of the packages on a working installation ([docs/orbs3/ver.txt](docs/orb3-ver.txt)). Higher versions generally work but not in some cases.

- [Here](docs/orb3-env.txt) is an environment file that can be used directly with conda to install the correct versions of the packages ([docs/orbs3/env.txt](docs/orb3-env.txt)). 



	  
