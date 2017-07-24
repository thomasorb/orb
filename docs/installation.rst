ORB installation
################

.. contents::

.. note:: Instructions are generally given for Ubuntu. It should work
	  also for any Debian based operating system. All the
	  installation can be used with pip or setup.py. Its mostly
	  the same for Mac (see below).


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

.. note:: For Ubuntu users just add sudo in front of all commands

Required packages
-----------------

You can install the required packages with::
  
  apt-get install python-pip python-dev build-essential
  apt-get install libatlas-base-dev gfortran
  apt-get install libhdf5-dev libhdf5-8 hdf5-tools
  
Some modules are also required to be installed before orb can be installed::
  
  pip install -r requirements.txt

The actual requirements file is:

.. literalinclude:: ../requirements.txt

Finally you can install orb with::
  
  pip install orb --upgrade
  
  
Compile Cutils
==============

If you have an error with orb.cutils or orb.cgvar you might have to
recompile it. You can do it with::

  python setup.py build_ext --inplace

You might need to specify the path to numpy with the following command
(the included path might change)::

  python setup.py build_ext --inplace -I /usr/local/lib/python2.7/sit-packages/numpy/core/include/

  
