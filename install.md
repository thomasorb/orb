# installation instructions with Anaconda (should work on Linux, Mac OSX, Windows)

## 1. download Miniconda for your OS and python 2.7.

**If you already have [Anaconda](https://www.anaconda.com/) installed go to step 2**

instructions are here: [Miniconda — Conda](https://conda.io/miniconda.html)
1. place the downloaded file on your home directory
2. install it (use the real file name instead of `Miniconda*.sh`)
```bash
bash Miniconda*.sh
```
## 2. install `conda-build` tools
```bash
conda install conda-build
```

## 3. create orb environment
### automatic procedure if you have a `*.yml*` file:
create an environment automatically
```bash
conda env create -f orb-env.yml
```

### manual installation of the librairies
create an environment and install needed modules manually
```bash
conda create -n orb python=2.7 
conda install -n orb numpy scipy bottleneck matplotlib astropy cython h5py dill pandas
conda install -n orb -c conda-forge pyregion
conda install -n orb -c astropy photutils
conda activate orb
```
now your prompt should be something like `(orb) $`.
```bash
pip install gvar --no-deps
pip install lsqfit --no-deps
pip install pp --no-deps
```

## 4. add orb module

clone [ORB](https://github.com/thomasorb/orb)
```bash
mkdir orb-stable # do it where you want to put orb files
cd orb-stable
git clone https://github.com/thomasorb/orb.git
```

in the downloaded folder
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
cd path/to/orb-stable
python setup.py build_ext --inplace
python setup.py install
```

Test it:
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
python -c 'import orb.core'
```
## 5. add orcs module

clone [ORCS](https://github.com/thomasorb/orcs)
```bash
mkdir orcs-stable # do it where you want to put orcs files
cd orcs-stable
git clone https://github.com/thomasorb/orcs.git
```

in the downloaded folder
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
cd path/to/orcs-stable
python setup.py install
```

Test it:
```bash
conda activate orb # you don't need to do it if you are already in the orb environment
python -c 'import orcs.process'
```

## 6. Install jupyter

```bash
conda install -n orb -c conda-forge jupyterlab
```
Run it

```bash
conda activate orb # you don't need to do it if you are already in the orb environment
jupyter lab
```
You should now have your web browser opened and showing the jupyter lab interface !
