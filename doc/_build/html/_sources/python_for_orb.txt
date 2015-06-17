Python for ORB users
####################

.. contents::


The basics
==========

An excellent tutorial can be found `here <http://docs.python.org/2/tutorial/>`_.

In this part we will just give some small examples, but a lot more
information can be found in the Python tutorial.


Hello world
-----------

You can use any text editor for Python (e.g. Emacs, Vi). A simple hello world program like this one,

.. code-block:: python
  :linenos:

  print 'hello world'

can be saved in a file named 'hello.py' and run this way:

.. code-block:: console

  python hello.py

The result will be:

.. code-block:: console
  
  hello world


Handling arrays
---------------
.. code-block:: python
  :linenos:

  import numpy as np # load the Numerical module

  a = np.arange(25) # create a 1d indexed array
  a = range(25) # another possibility without using Numpy
  b = np.ones(25) # create a 1d array of ones
  b *= 2 # multiply b by 2
  c = a * b # multiply a by b
  print c
  

Note that there's a difference between np.arange() and range(). The
first one creates an array, the second one creates a `list <http://docs.python.org/2/tutorial/introduction.html#lists>`_ .

A simple loop
-------------

Indentation is the key to the structure of a Python program (no begin, end, or {} statements just indentation.)

.. code-block:: python
  :linenos:

  import numpy as np # load the Numerical module

  a = np.empty((5,5), dtype=float) # an empty 2D array of floats of shape (5,5) 

  for i in range(5):
    for j in range(5):
      if j != 0:
        a[i,j] = i/j
      else:
        a[i,j] = np.inf # replace a division by 0 by an 'inf'

  print a


Plotting data
-------------

You must have matplolib_ installed.

.. code-block:: python
  :linenos:

  import pylab as pl # import plot module
  import math # import math module

  x = np.linspace(-math.pi, math.pi, 1000)
  y = np.cos(x)

  pl.plot(x, y, linestyle='--', color='green') # plot as a green dotted line
  pl.plot(x, -y, linestyle='-', color='red') # plot as a red line
  pl.show() # show the plotting result (don't forget it or you will see nothing)




Importing and using ORB
========================

Importing ORB
--------------

ORB is aimed not only to reduce cubes but also to handle them. 

.. code-block:: python
  :linenos:

  from orb.core import Tools, Cube # Import core classes
  import orb.utils # Import useful functions


Reading and writing FITS
------------------------

The Tools() class provides reading and writing functions.

.. note:: All the basic functions have a lot of useful options. Please
  read the documentation, it might be of a certain help ;) !

.. code-block:: python
  :linenos:

  to = Tools() # instanciate Tools class
  
  data = to.read_fits('cube.fits') # Load a cube
  data, hdr = to.read_fits('cube.fits', return_header=True) # Load a cube and its header
  
  print hdr['OBJECT'] # Prints the object name
  hdr['CALCOEFF'] = (25, 'calibration coeff') # Add a header key
  
  calibration_coeff = 1e-5 
  data *= calibration_coeff # recalibrate whole data

  to.write_fits('recalibrated_cube.fits', data, fits_header=hdr)
  
  import pylab as pl # import plot module

  step = hdr['STEP'] # Get step
  order = hdr['ORDER'] # Get folding order
  axis = create_nm_axis(data.shape[2], step, order) # create axis
  spectrum = data[84,58,:] # get spectrum at the pixel [84,58]

  pl.plot(axis, spectrum) # plot spectrum
  pl.show() # show plot


Loading large cubes or reduction data
-------------------------------------

A large cube (30 Go) might be too large to be handled easily with ds9
or loaded entirely in memory. That's why those cubes are stored in a
frame-divided way i.e. as a set of independant frames in a folder and
a list of the paths to the frames in the correct order. ORB has been
designed to handle large data cubes.

.. code-block:: python
  :linenos:

  cube = Cube('liste') # A simple list is enough to initialize a Cube instance
  quadrant = Cube[25:50, 25:50, :] # Here you just load a small quadrant
  spectrum = Cube[84,58,:] # load spectrum at pixel [84,58]

  
.. _matplolib: http://matplotlib.org/
