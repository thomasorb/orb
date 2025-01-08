![header](docs/images/svg/orb_header.jpg?raw=true "Title")

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


## installation instructions with Anaconda (should work on Linux and Mac OSX, Windows is not supported anymore)

### 1. download Miniconda for your OS and python 3.10.

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
### 3. create environment from a conda environment file.

Copy the [orb3.yml](docs/orb3.yml) file anywhere on your system then run:

```bash
conda env create -f orb3.yml
```

**note:** If you want to change the name of the created environment you can modify the first line of the file ```name: orb3```.

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
echo '/absolute/path/to/orb-stable/orb' > ~/miniconda3/envs/orb3/lib/python3.10/site-packages/conda.pth
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orb.core'
```

### 5. Run jupyter

```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
jupyter lab
```
You should now have your web browser opened and showing the jupyter lab interface !


## Troubleshooting

### Create an environment manually

If the environment cannot be built from the [orb3.yml](docs/orb3.yml) file, you can create an environment and install the needed modules manually.

```bash
conda create -n orb3  -c conda-forge -c astropy -c anaconda python=3.10.9 numpy=1.24.2 scipy=1.10.1 matplotlib astropy=5.2.1 cython=3.0.0 h5py dill pandas=1.5.3 pytables=3.7.0 jupyterlab photutils=1.5.0 astroquery reproject gitpython pyregion=2.1.1

conda activate orb3

pip install gvar==11.11.1 --no-deps
pip install lsqfit==13.0 --no-deps
pip install fpdf --no-deps
```

### Building GVAR
If you encounter a problem building gvar you may want to install build-essential (for Ubuntu users)
```
sudo apt install build-essential
```

### Check the version of the packages

- [Here](docs/orb3-ver.txt) is a list of all the versions of the packages on a working installation ([docs/orbs3/ver.txt](docs/orb3-ver.txt)). Higher versions generally work but not in some cases.

- These packages are known to raise exceptions when using the latest version. After uninstalling them you can install them again in a working version using the following:
```bash
conda install astropy=4.2.1
conda install photutils=1.0.1
```
### Save your environment file

The orb3.yml environment file was created with the following command in a working environment

```bash
conda env export -n orb3 > orb3.yml
```

	  
