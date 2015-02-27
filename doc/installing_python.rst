Installing Python
#################

.. contents::

.. note:: All instruction are given for Ubuntu |Ubuntu|. It might work
     for any Debian based operating system.


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

The following modules are optional. 

  * Tkinter_ : You will need it to use the script ``orbs-optcreator``
    which can help you creating an option file. But you don't need it
    to run the reduction.

  * Cython_ : Needed to compile Cython_ functions in cutils.pyx

To use the Viewer (orb-viewer)
------------------------------

`Ginga <http://ejeschke.github.io/ginga/>`_ must be installed and the GTK backend must be compiled inside matplotlib:

1 - Install Ginga

  sudo pip install ginga

2 - Install development libraries for gtk2

  sudo apt-get install python-gtk2-dev

3 - Rebuild matlplotlib

  sudo pip install --upgrade matplotlib


To check if a module is installed just try the following::

  python
  >> import module_name
 

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

Tkinter_
--------

Simply run::

  sudo apt-get install python-tk
  sudo apt-get install python-imaging-tk


Cython_
-------

To install Cython_::

    sudo pip install cython --upgrade



.. |Ubuntu| image:: ubuntu-icon.png
            :height: 40
   	    :width: 40
            :scale: 70

.. _Python: http://www.python.org/
.. _Scipy: http://www.scipy.org/
.. _Numpy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _Parallel: http://www.parallelpython.com/
.. _Tkinter: http://docs.python.org/2/library/tkinter.html
.. _Cython: http://cython.org/
.. _PyWCS: http://stsdas.stsci.edu/astrolib/pywcs/
.. _Bottleneck: https://pypi.python.org/pypi/Bottleneck
.. _PIP: https://pypi.python.org/pypi/pip
