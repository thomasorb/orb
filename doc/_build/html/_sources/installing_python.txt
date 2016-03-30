Installing Python
#################

.. contents::

.. note:: Instructions are generally given for Ubuntu |Ubuntu|. It
     might work for any Debian based operating system. Most of the
     installation can be used with pip or setup.py and its the same
     for Mac |Mac| (see below).

Fast installation sequence for MAC users
========================================

Most of the installation process can be done with pip::

  sudo easy_install pip
  sudo pip install pip --upgrade
  sudo pip install numpy --upgrade
  sudo pip install scipy --upgrade
  sudo pip install cython --upgrade
  sudo pip install astropy --upgrade
  sudo pip install h5py --upgrade
  
For the libraries that are not accessible via pip (e.g. pp) you can
install them using python. Download the archive, unzip it and, in the
unzipped folder type::

  sudo python setup.py install

You must also install fftw3 libraries (see `here <https://pypi.python.org/pypi/h5py/2.5.0>`_) and then install the pyfftw wrapper with pip::

  sudo pip install pyfftw --upgrade




Python_
=======

First of all you must have Python_ 2.7 installed.

If you already have Python you can check the version with the
following command::

  python --version

If the version you have is not the good one you can go on this `page
<http://www.python.org/download/releases/>`_, download the last 2.7.x
release as a tar ball file then build it from sources.

.. warning:: Orbs has been written for Python 2. You must not use
     Python 3 !



Modules
=======

The following modules are needed for the reduction pipeline, minimum
version required is in parenthesis (in general always try to use use
the **latest version**) :

  * Scipy_ (v0.13.3)
  * Numpy_ (v1.8.1)
  * PyFITS_ (v3.1.1) now astropy.io.fits
  * Parallel_ Python (v1.6.4)
  * PyWCS_ (v1.11) 
  * Bottleneck_ (v0.8.0)
  * h5py_ (v2.5.0)
  * Cython_ : Needed to compile Cython_ functions in cutils.pyx
  * pyFFTW_ (v0.10.1)

The following modules are optional. 

Check if a module is already installed
--------------------------------------

To check if a module is installed just try the following::

  python
  >> import module_name
 


To use the Viewers (orb-viewer and orb-viewer3d)
------------------------------------------------
1. Install pygtk

  sudo pip install pygtk

2. Install pycairo

  sudo apt-get install python-cairo

3. Install development libraries for gtk2

  sudo apt-get install python-gtk2-dev

4. Rebuild matlplotlib

  sudo pip install --upgrade matplotlib

5. Install Vispy (for the 3d viewer)

  sudo pip install vispy

6. Install PySide (for the 3d viewer)

  sudo apt-get install python-pyside


Scipy_ and Numpy_
-----------------

Simply run::

  sudo apt-get install python-scipy
  sudo apt-get install python-numpy

In order to get an updated version of Numpy you can then run::

  sudo pip install numpy --upgrade

In order to get an updated version of Scipy you can then run::

  sudo apt-get install libatlas-base-dev gfortran
  sudo pip install scipy --upgrade

If you need to install PIP_::

  sudo apt-get install python-pip python-dev build-essential
  sudo pip install --upgrade pip 
  sudo pip install --upgrade virtualenv 

PyFITS_
-------

You might need the latest version of PyFITS_ (v3.1.2 or better)


PyFITS_ is now part of the package astropy.io.fits. Astropy package
must then be installed. See the installation steps at
http://docs.astropy.org/en/stable/install.html.

  sudo pip install --no-deps astropy


If you have an old packaged tar.gz version you can go through the
steps below.

You must first have 'distutils' installed. You can install it using
the command::

  sudo apt-get install python-setuptools

Download the latest version `here
<http://www.stsci.edu/institute/software_hardware/pyfits/Download>`_ and
untar it using this command::

  tar -xzvf tar_name.tar

You can then install PyFITS_ by running this command in the
uncompressed directory::

  sudo python setup.py install

Parallel_ Python
----------------

Download the latest version (1.6.4 or better) `here
<http://www.parallelpython.com/content/view/18/32/>`_
(pp-1.x.x.tar.gz), untar it and once in the uncompressed directory
run::

  sudo python setup.py install

PyWCS
-----

PyWCS is a set of routines for handling the FITS World Coordinate
System (WCS) standard. It can be downloaded `here <https://pypi.python.org/pypi/pywcs>`_ (pywcs-1.xx.tar.gz). Once downloaded you must untar it and run::

  sudo python setup.py install


Bottleneck_
-----------

Download the latest version (0.8.0 or better) `here
<https://pypi.python.org/pypi/Bottleneck>`_, untar it and once in the
uncompressed directory run::

  sudo python setup.py install

h5py
----

Installation must be manual because the SZIP library must be installed
and linked to hdf5 which can finally be linked to h5py.


Install SZIP
~~~~~~~~~~~~

You can find SZIP `here
<http://www.hdfgroup.org/ftp/lib-external/szip/2.1/src/szip-2.1.tar.gz>`_
and info on SZIP compression in HDF5 `here
<https://www.hdfgroup.org/doc_resource/SZIP/>`_. Then after the
extraction you can go in the extracted folder and do::

  sudo ./configure --prefix=/usr/local/lib/szip
  sudo make
  sudo make check
  sudo make install

.. note:: folder :file:`/usr/local/lib/szip` can be changed as long as
          you also change it in the following installation steps.

Install HDF5
~~~~~~~~~~~~

You can find HDF5 sources `here
<https://www.hdfgroup.org/HDF5/release/obtainsrc.html>`_. Then extract the
sources and jump into the extracted folder before typing::

  sudo ./configure --prefix=/usr/local/lib/hdf5 --with-szlib=/usr/local/lib/szip
  sudo make
  sudo make check
  sudo make install

Install H5PY (pip cannot be used directly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can find h5py sources `here <https://pypi.python.org/pypi/h5py/2.5.0>`_. After extraction, just
run the following into the extracted folder::

  sudo python setup.py configure --hdf5=/usr/local/lib/hdf5
  sudo python setup.py build
  sudo python setup.py install


Cython_
-------

To install Cython_::

  sudo pip install cython --upgrade


Install PyFFTW
--------------

FFTW3 library must be installed (see `here
<https://pypi.python.org/pypi/pyFFTW>`_)::

  sudo apt-get install libfftw3-dev

then the package can be installed via pip::

  sudo pip install pyfftw



.. |Ubuntu| image:: os_linux.png
            :height: 40
   	    :width: 40
            :scale: 70

.. |Mac| image:: os_apple.png
            :height: 40
   	    :width: 40
            :scale: 70

.. _Python: http://www.python.org/
.. _Scipy: http://www.scipy.org/
.. _Numpy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _Parallel: http://www.parallelpython.com/
.. _Cython: http://cython.org/
.. _PyWCS: http://stsdas.stsci.edu/astrolib/pywcs/
.. _Bottleneck: https://pypi.python.org/pypi/Bottleneck
.. _PIP: https://pypi.python.org/pypi/pip
.. _h5py: https://pypi.python.org/pypi/h5py/2.5.0
.. _pyFFTW: https://pypi.python.org/pypi/pyFFTW
