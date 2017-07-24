
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

.. note:: For Debian Stretch the hdf5 library that must be used is: libhdf5-100

Some modules are also required to be installed before orb can be installed::
  
  pip install -r requirements.txt
		    
Finally you can install orb with::

  python setup.py build_ext
  python setup.py install


Uninstall
---------

If you want to uninstall you must do it manually. A good way to do
this is to add the option --record in the installation line and use
the generated file to remove all the created files::

  python setup.py install --record uninstall.txt
  rm -rf `cat uninstall.txt`

