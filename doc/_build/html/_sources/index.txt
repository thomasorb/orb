.. Orb documentation master file, created by
   sphinx-quickstart on Sat May 26 01:02:35 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


ORB Documentation
##################

.. image:: orb.png
   :width: 40%
   :align: center

.. topic:: Welcome to ORB documentation !

   ORB is the kernel module for the whole suite of data reduction and
   analysis tools for SpIOMM_ and SITELLE_: ORBS, ORCS, OACS, ORUS and IRIS.

   .. _SpIOMM: 

   **SpIOMM** (*Spectromètre Imageur de l'Observatoire du Mont Mégantic*) is an astronomical instrument operating at Mont Mégantic_ (Québec, CANADA) designed to obtain the visible spectra of all the objects in a 12 arc minutes field of view.

   .. _SITELLE: 

   **SITELLE** (Spectromètre-Imageur pour l’Étude en Long et en Large des raie d’Émissions) is a larger version of SpIOMM operating at the CFHT_ (Canada-France-Hawaii Telescope, Hawaii, USA).

Table of contents
-----------------

.. contents::

Installation
------------

You'll maybe need to install Python_ or some modules before installing
ORB. You'll find here some informations on how to install Python_ on
Ubuntu.

.. toctree::
   :maxdepth: 2

   installing_orb
   installing_python


Python for ORB users
---------------------

.. toctree::
   :maxdepth: 2

   python_for_orb

Documentation
-------------

.. toctree::
   :maxdepth: 2

   core_module
   utils_module
   cutils_module
   astrometry_module

Changelog
---------

.. toctree::
   :maxdepth: 2

   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _Mégantic: http://omm.craq-astro.ca/
.. _CFHT: http://www.cfht.hawaii.edu/
.. _Python: http://www.python.org/
.. _Scipy: http://www.scipy.org/
.. _Numpy: http://numpy.scipy.org/
.. _PyFITS: http://www.stsci.edu/resources/software_hardware/pyfits
.. _Parallel: http://www.parallelpython.com/
