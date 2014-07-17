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
------------------

* :py:meth:`~astrometry.fit_stars_in_frame` has been updated to fit
  multiple stars at the same time (see:
  :py:meth:`~cutils.multi_fit_stars`). The star fit is now way more
  robust.

USNO-B1 based star detection
----------------------------

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

* :py:meth:`~cutils.spectrum_mean_energy` and
  :py:meth:`~cutils.interf_mean_energy` Cythonised.

* :py:class:`~core.OptionFile` enhanced to be used by
  :py:meth:`orbs.Orbs.__init__`.

ORCS integration
----------------

* new keywords in config.orb: OBS_LAT, OBS_LON, OBS_ALT for ORCS.

* new general keyword in OptionFile: INCLUDE, used to include the
  parameters of another option file.

* Warning messages are not displayed anymore when using the silent
  option with :py:class:`~core.Tools`

* move :py:meth:`orbs.Orbs._create_list_from_dir` to
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

* add py:meth:`~utils.flambda2ABmag`

* change file globals.py for constants.py

* add py:meth:`~core.Tools._get_basic_spectrum_header` to return a
  header for a 1D spectrum.

* py:meth:`~core.Tools.write_fits` updated to create ds9 readable 1D
  FITS files.

* py:meth:`~utils.fit_lines_in_vector` accepts a tuple for the
  parameter cov_pos. This tuple gives the lines that are
  covarying. This way, [NII] and Halpha can have different velocities,
  but the [NII] lines will share the same velocity, improving a lot
  the precision on their estimated velocity without being biased by
  the Halpha velocity.

* py:meth:`~utils.fit_map` created. This function is a generalization
  of the old py:meth:`orbs.process.Phase.fit_phase_map` which now use
  this general function also. The fitting process has been enhanced
  and is now more robust and use NaNs instead of zeros.
