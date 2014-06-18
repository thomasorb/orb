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
  :py:meth:`~cutils.multi_fit_stars`). The star fit is now a lot more
  robust.

