
ORB
===

ORB is the kernel module for the whole suite of data reduction and
analysis tools for SITELLE_: ORBS_, ORCS_, ...

It provides basic access to the data cubes as long as the fitting
engine of ORCS and numerous utilitary functions for the analysis of
interferometric and spectral data, imaging data, astrometry,
photometry.


Documentation
=============

You can find the up-to-date documentation here:

http://celeste.phy.ulaval.ca/orb-doc/index.html


Python
======

First of all you must have Python 2.7 installed.

If you already have Python you can check the version with the
following command::

  python --version

If the version you have is not the good one you can go on this `page
<http://www.python.org/download/releases/>`_, download the last 2.7.x
release as a tar ball file then build it from sources.

.. warning:: Orbs has been written for Python 2. You must not use
     Python 3 !


	  
Installation for Ubuntu/Debian users
====================================

.. note:: Instructions are generally given for Ubuntu. It should work
	  also for any Debian based operating system. All the
	  installation can be used with pip or setup.py. Its mostly
	  the same for Mac (see below).

.. note:: For Ubuntu users just add sudo in front of all commands

Required packages
-----------------

You can install the required packages with::
  
  apt-get install python-pip python-dev build-essential
  apt-get install libatlas-base-dev gfortran
  apt-get install libhdf5-dev libhdf5-8 hdf5-tools
  
.. note:: For Debian Stretch the hdf5 library that must be used is: libhdf5-100

Download and install ORB
------------------------

The archive and the installation instructions for ORB_ can be found on github::
  
  https://github.com/thomasorb/orb

Once the archive has been downloaded (from github just click on the
green button `clone or download` and click on `Download ZIP`) you may
extract it in a temporary folder (try to avoid path containing
non-ASCII characters, a good one may be ``~/temp/``). Then cd into the
extracted folder and type the following to install all the required
python modules via pip::
  
  pip install cython
  pip install numpy
  pip install gvar
  pip install -r requirements.txt

The actual requirements file is (it can be found at the root of the
archive):

.. literalinclude:: ../requirements.txt

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
  
.. _ORB: https://github.com/thomasorb/orb
.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
.. _SITELLE: http://www.cfht.hawaii.edu/Instruments/Sitelle
.. _ORBS: https://github.com/thomasorb/orbs
.. _ORCS: https://github.com/thomasorb/orcs
