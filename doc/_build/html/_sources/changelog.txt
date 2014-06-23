Changelog
#########

	
v1.0 Creation of ORB
********************

Architecture change of ORBS. All the shared content originally in the
ORBS module have been moved to ORB.

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
******

* :py:meth:`~utils.transform_spectrum` and
  :py:meth:`~utils.transform_interferogram` adjusted to lose no energy
  in the transformation process.  They are able to treat wavenumber
  transformation (useful to avoid the mutiple interpolation nescessary
  to move from a regular wavenumber space to an iregular wavelength
  space back and forth)'
    
* new keywords in config.orb: FIELD_OF_VIEW_2, EXT_ILLUMINATION
    
* new keyword in the option file: TRYCAT to use a USNO-B1 catalogue
  for star detection
    
* doc updated
    
* bug fix
