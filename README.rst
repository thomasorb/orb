
ORB
===

ORB is the kernel module for the whole suite of data reduction and
analysis tools for SITELLE: ORBS, ORCS, ...

It provides basic access to the data cubes as long as the fitting
engine of ORCS and numerous utilitary functions for the analysis of
interferometric and spectral data, imaging data, astrometry,
photometry.



Installation for Ubuntu/Debian users
------------------------------------

You can install the required packages with::
  
  apt-get install python-pip python-dev build-essential
  apt-get install libatlas-base-dev gfortran
  apt-get install libhdf5-dev libhdf5-8 hdf5-tools
  
Some modules are also required to be installed before orb can be installed::
  
  pip install -r requirements.txt
		    
Finally you can install orb with::
  
  python setup.py install


Compile Cutils
==============

If you have an error with orb.cutils or orb.cgvar you might have to
recompile it. You can do it with::

  python setup.py build_ext --inplace

You might need to specify the path to numpy with the following command
(the included path might change)::

  python setup.py build_ext --inplace -I /usr/local/lib/python2.7/sit-packages/numpy/core/include/
