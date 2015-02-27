Changelog
#########

	
v1.0 Creation of ORB
********************

Major modification of ORBS architecture. All the shared content
originally in the ORBS module have been moved to ORB. This way, ORBS,
ORCS, OACS, IRIS and ORUS can share the same core module without
importing ORBS entirely each time. Conceptually ORBS, like the other
softwares, just wraps around ORB module and is not any more the
central part of the whole suite of softwares.

v1.1
****

Multi fit of stars
==================

* :py:meth:`~astrometry.fit_stars_in_frame` has been updated to fit
  multiple stars at the same time (see:
  :py:meth:`~cutils.multi_fit_stars`). The star fit is now way more
  robust.

USNO-B1 based star detection
============================

* :py:meth:`~astrometry.Astrometry.query_vizier` and
  :py:meth:`~astrometry.Astrometry.register` have been moved from
  :py:class:`orbs.process.Spectrum` so that the registration is part of
  the Astrometry module (which makes more sense). This way it is now
  possible to use a star catalogue like USNO-B1 to detect stars in the
  cube. It is not a default behaviour because extended emission region
  contains virtually no catalogued stars. This option can be useful for
  galaxies to avoid the confision of HII regions and stars.

v1.2
****

* :py:meth:`~cutils.multi_fit_stars` noise estimation
  enhanced. The initial estimation of the shift has also been updated.

* Minor bugs fix. This version is considered as a nearly stable
  version ready for release.

* :py:meth:`~cutils.multi_fit_stars` initial estimation enhanced (more
  robust and precise)
    
v1.2.1
======

* :py:meth:`~utils.transform_spectrum` and
  :py:meth:`~utils.transform_interferogram` adjusted to lose no energy
  in the transformation process.  They are able to treat wavenumber
  transformation (useful to avoid the mutiple interpolation nescessary
  to move from a regular wavenumber space to an iregular wavelength
  space back and forth)'
    
* new keywords in config.orb: FIELD_OF_VIEW_2, EXT_ILLUMINATION
    
* doc updated
    
* bug fix

v1.2.2
======

* :py:meth:`~utils.spectrum_mean_energy` and
  :py:meth:`~utils.interf_mean_energy` Cythonised to
  :py:meth:`~cutils.spectrum_mean_energy` and
  :py:meth:`~cutils.interf_mean_energy`.

* :py:class:`~core.OptionFile` enhanced to be used by
  :py:meth:`orbs.orbs.Orbs.__init__`.

ORCS integration
----------------

* new keywords in config.orb: OBS_LAT, OBS_LON, OBS_ALT for ORCS.

* new general keyword in OptionFile: INCLUDE, used to include the
  parameters of another option file.

* Warning messages are not displayed anymore when using the silent
  option with :py:class:`~core.Tools`

* move :py:meth:`orbs.orbs.Orbs._create_list_from_dir` to
  :py:meth:`~core.Tools._create_list_from_dir` to make this useful
  method accessible to ORCS.

* doc updated


ORB's scripts
-------------

* move ORB's scripts (dstack, combine, rollxz, rollyz, reduce) from
  orbs/scripts to orb/scripts so that only ORBS specific scripts are
  in orbs/scripts.

* create **unstack** script to unstack a cube into a set of frames

v1.2.3
======

* add :py:meth:`~utils.flambda2ABmag`

* change file globals.py for constants.py

* add :py:meth:`~core.Tools._get_basic_spectrum_header` to return a
  header for a 1D spectrum.

* :py:meth:`~core.Tools.write_fits` updated to create ds9 readable 1D
  FITS files.

* :py:meth:`~utils.fit_lines_in_vector` accepts a tuple for the
  parameter cov_pos. This tuple gives the lines that are
  covarying. This way, [NII] and Halpha can have different velocities,
  but the [NII] lines will share the same velocity, improving a lot
  the precision on their estimated velocity without being biased by
  the Halpha velocity.

* :py:meth:`~utils.fit_map` created. This function is a generalization
  of the old :py:meth:`orbs.process.Phase.fit_phase_map` which now use
  this general function also. The fitting process has been enhanced
  and is now more robust and use NaNs instead of zeros.

v1.2.4
======

Miscellaneous
-------------

* all scripts have been renamed to orb-*

* --nostar and --flat bug fixed. Cosmic ray detection will not be done
  if those options are given.

SITELLE data
------------

* new command: **orb-conf**. Its general purpose is to help the
  administrator to quickly change ORB configuration. Its first use is
  to change the configuration file depending on the used
  instrument. To change the configration file from spiomm to sitelle
  just type::

    orb-conf -i sitelle

  This command avoid the painful manual change of the config file. At
  each new version this command can be run to quickly (and safely)
  reconfigure ORB. Note that this function requires write rights on
  the ORB installation folder.

Sitelle image mode
~~~~~~~~~~~~~~~~~~

* if ORBS is in **sitelle mode** (if the configuration file points to
  config.sitelle.orb), SITELLE's data frames are handled at the core
  level. :py:meth:`~core.Tools.read_fits` accepts two new options:
  image_mode and chip_index. If image_mode is set to 'sitelle' and the
  chip index is 1 or 2, then the read_fits function will return only
  of the 2 chips (depending on the chip index). **Chip slicing** is
  handled by
  :py:meth:`~core.Tools._read_sitelle_chip`. :py:meth:`~core.Cube.__getitem__`
  has also been modified in the same way with the same new options. A
  parameter line can now be added to the very first line of the image
  list passed to the :py:class:`~core.Cube`. This line must be
  something like::
    
    # sitelle 1

  If the first keyword is sitelle, the second keyword is understood as
  the chip index to read. This way, :py:class:`~core.Cube` understand
  that the data is SITELLE's data and what chip has to be read.

* :py:meth:`~core.Tools._create_list_from_dir` now accepts the options
  image_mode and chip_index and creates the parameter line at the very
  beginning of the output file list.

* **overscan** :py:meth:`~core.Tools._read_sitelle_chip` automatically
  substract the bias level given by the overscan areas of the returned
  image. This default behaviour can be canceled in the future.

Prebinning
~~~~~~~~~~

Used for faster computation of big data set. It
can also be useful if the user simply wants binned data. At the user
level only one option must be passed to the option file::

  PREBINNING 2 # Data is prebinned by 2

.. warning:: The real binning of the original data must be kept to the
   same values. The user must no modify the the values of BINCAM1 and
   BINCAM2.

* if this option is set :py:meth:`~core.Tools._create_list_from_dir`
  just adds the following directive at the beginning of the image list
  file::

    # prebinning 2

* :py:meth:`~core.Tools.read_fits` accepts the option
  'binning'. :py:meth:`~core.Tools._image_binning` has been created to
  bin 2D data efficiently. :py:meth:`~core.Cube.__getitem__` has been
  modified to read and treat transparently the new prebinning
  directive that is added at the beginning of an image list file.


v1.2.4.1
========

* Enhanced frame
  registration. :py:meth:`~astrometry.Astrometry.register` now takes
  full advantage of the multi fit of stars and filters the best stars
  by SNR. A double fit is also done at the beginning to ensure that
  the positions pattern is the best possible.

* bug fix, minimum number of good fitted pixels in a column for a
  phase fit lowered to 1/3 of the column length instead of 1/2.

v1.2.4.2
========

Astropy
-------

Astropy (http://www.astropy.org/) is definitly needed, pyfits and
pywcs standalone modules are not needed anymore by ORBS (but they
still can be used by other modules ;) even modules imported by ORBS so
becarefull before removing them)

* PYFITS: now imported from astropy.io.fits
* PYWCS: now imported from astropy.wcs


Better Star fit
---------------

* :py:meth:`~cutils.multi_fit_stars`: tilted background added to the model

* detected stars are selected not too far from the center of the
  frame

* star box coeff set to 10 instead of 7 to get a better sky statistic
  around stars.


SpIOMM bias overscan for camera 2
---------------------------------

When it exists, the bias overscan created with each frame of the
camera 2 is used to remove automatically the bias. Note that in this
case **the path to the bias frames must not be given to ORBS** because
ORBS will try to create a master bias and remove it at step 3. In
fact, the mean of the master bias will be near 0 because the overscan
is removed from the bias frames also. The impact of giving the path to
the bias frame is thus not dramatic. But it is better not to give it.

Miscellaneous
-------------
* :py:meth:`~astrometry.Astrometry.register` optimization routine is
  based on a least square fit instead of a powell algorithm.

* transfered :py:meth:`~cutils.part_value` from OACS cutils.

* :py:meth:`~astrometry.Astrometry.get_alignment_vectors` simplified
  because the multi fit mode is now robust enough to remove all which
  was written for the preceding individual fit mode.

* :py:meth:`~utils.indft`, :py:meth:`~cutils.indft` added to compute
  Inverse Non-uniform Discret Fourier Transform (INDFT). New option
  **sampling_vector** in :py:meth:`~utils.transform_spectrum` to give
  the possibility to compute an INDFT by giving a non-uniform sampling
  vector.

v1.3 Start of CFHT integration
******************************

v1.3.0 ORB-Viewer
=================

A viewer based on Ginga (https://github.com/ejeschke/ginga) has been
added to ORB (scripts/**orb-viewer**). It can be used to analyse
reduced data cube (spectral cube) or raw interferometric cubes. Basic
functionalities (fft, spectrum fit, image operations ...)  have been
implemented.

v1.3.1
======

orb-header
----------

script **orb-header** added to display and manipulate headers of FITS
files.


Miscellaneous
-------------

* All classes which inherit from :py:class:`~core.Tools` can be passed
  all Tools arguments even if the __init__ method has been
  reimplemented (a new cofiguration file path can thus be defined
  easily)

* :py:meth:`~astrometry.Astrometry.register` enhanced to compute scale
  only at the center of the frame. This function can now be used to
  compute the optical distorsion pattern of an image.

* :py:meth:`~astrometry.fit_star`: 'saturation' option added to avoid
  saturated pixels during a the fit of a star. Allows for saturated
  star reconstruction of the real flux.


* :py:meth:`~utils.compute_line_fwhm`,
  py:meth:`~utils.compute_line_shift`
  py:meth:`~utils.compute_radial_velocity` transfered from ORCS to
  ORB.


* script **orb-dstack** can be given a directory instead of a file
  list. It is now able to filter SITELLE files to get only the
  'object', 'dark' or 'flat' type files.

