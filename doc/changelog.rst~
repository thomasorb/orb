Changelog
#########

	
v3.1
****

v3.1.0
======

Alignment
---------

* :py:meth:`~process.InterferogramMerger.find_alignment` has been
  completly modified. There's no more need for a list of detected
  stars in the cube B (or camera 2). Stars found in cube A are
  searched in cube B. The process is now faster and must be more
  robust.

* :py:meth:`~orbs.Orbs.transform_cube_B` has also been changed to
  avoid looking for stars in the cube B.


Cosmic rays detection
---------------------

* :py:meth:`~process.RawData.create_cosmic_ray_map` remove a low order
  polynomial on each tested interferogram vector before checking for
  cosmic rays. It helps a lot in avoiding too much detected cosmic
  rays on stars.


Overwriting FITS files
----------------------

* :py:meth:`~core.Tools.write_fits` has a new option to overwrite a
  FITS file.
* 'orbs' script can be passed the option -o to enable overwriting over
  the existing FITS files.
* :py:class:`~orbs.Orbs`, :py:class:`~core.Cube` and all processing
  classes have been modified to use this new option and pass it to any
  self._write_fits() function.

Transmission function
---------------------

* :py:meth:`~process.InterferogramMerger.merge` has been changed to
  eliminate the stars with not enough flux (set to 0.25 times the
  maximum flux of the detected stars)

v3.1.1
======

* :py:meth:`~process.RawData.add_missing_frames` : correction of a
  minor bug when adding zeros frames for a single cube interferogram
* :py:meth:`~process.RawData.add_missing_frames` and
  :py:meth:`~process.InterferogramMerger.add_missing_frames` have been
  corrected to add the correct header to the created zeros frames.

v3.1.2
======

* :py:meth:`~process.InterferogramMerger.find_alignment` has been
  modified to use a larger box_size to detect stars during the rough
  alignment step. A warning is printed if less than 80 % of the stars
  can be fitted after the first optimization pass and an error is
  raised if less than 50 % of the stars can be fitted.

* **orbs** script can be passed the option -p or --phase to enter the
  number of points to use for space correction. If 0 is entered no
  phase correction will be done.

v3.2
****

Dark and Bias correction for SpIOMM CAM2
========================================

* :py:meth:`~process.RawData.create_interferogram` : Dark and
  Bias are now removed using their precise temperature and a
  calibrated function. The temperature of the frames (bias frames,
  dark frames and frames to be corrected) must be recorded in the
  header of the files with the keyword CCD-TEMP.

* **orbs-tempreader** script: This script has been created to read the
  temperature files created during an observing run and write the
  temperature of the files in the headers of the frames. It must be
  used in each folder where the temperature are needed (bias, dark and
  cam2 folders)

* **config.orb** : new keywords (DARK_CALIB_PARAM_A, DARK_CALIB_PARAM_B,
  DARK_CALIB_PARAM_C, BIAS_CALIB_PARAM_A, BIAS_CALIB_PARAM_B) have
  been added for the calibration coefficients of the bias and the
  dark. They have been computed from calibrated bias and dark curves :
  I(T) = f(T).

Frames transformation
=====================

* :py:meth:`~utils.transform_frame` has been completely changed and
  optimized using scipy.ndimage fast routines for image
  transformation. The time consumption of the transformation step has
  been dramatically decreased : this process is now more than 10 times
  faster.

Orbs script
===========

* **orbs** script option -r --raw has been removed and replaced by the
  option --nostar.Using the option -s (single reduction : only the
  cam1 cube is reduced) it is now possible to reduce one or both cubes
  without stars. The alignment steps are skipped and the default
  alignment parameters are used during the merging process.


v3.3
****

Cosmic ray detection and correction
===================================

* :py:meth:`~process.RawData.create_cosmic_ray_map` is now capable of
  detecting ver faint cosmic rays without overdetecting cosmic rays in
  stars. Planes and satellites are also detected.

* Cosmic rays corrected by
  :py:meth:`~process.RawData.create_interferogram` are now replaced by
  a weighted average of the neighbourhood. Weights are defined using a
  gaussian kernel. The kernel degree (i.e. neighbourhood radius) can
  be choosen.

v3.4
****

v 3.4.0:  Phase correction
==========================

* :py:meth:`~process.InterferogramMerger.merge` has been modified to
  create a small data cube containing the interferograms of choosen
  stars. Those interferograms can be used by
  :py:meth:`~process.Interferogram.compute_spectrum` to recover the
  first order coefficient of the phase. This way the phase is not
  computed for each pixel but a general correction is made.

* :py:meth:`~process.Interferogram.compute_spectrum` has been modified
  to compute the mean first order coefficient given a cube of stars
  interferogram.

* :py:meth:`~utils.transform_frame` has been modified to compute the
  phase fo each pixel given a phase map which gives the zeroth order
  of the polynomial function of the phase for each pixel and the first
  order coefficient. Both parameters (the phase map and the first
  order coefficient) must be given to avoid a pixel by pixel phase
  computation which can be unreliable.

* :py:class:`~process.Phase`: A new class has been created to manage
  phase data cubes. Those data cubes are useful to recover the phase
  maps. Three methods have been created:
  :py:meth:`~process.Phase.create_phase_maps` which create the phase
  maps from a phase cube, :py:meth:`~process.Phase.smooth_phase_map`
  which smooth the values of the phase map of 0th order (remember that
  phase values are defined modulo PI) and
  :py:meth:`~process.Phase.fit_phase_map` which fit the created
  smoothed phase map of 0th order to remove noisy data.

* :py:meth:`~orbs.Orbs.full_reduction` and
  :py:meth:`~orbs.Orbs.single_reduction` use computed phase maps by
  default. An external phase map of order 0 can be given if it has
  been computed (e.g. from a flat cube).

* **orbs** script option -s replaced by the options -1 or -2 in order
    to reduce only the camera 1 cube (-1) or the camera 2 cube
    (-2). --flat option added to reduce flat cubes and obtain only
    their phase map (spectrum is not computed)


v3.4.1
======

Correct for strange phase with calibration stars
------------------------------------------------

* :py:meth:`~process.Interferogram.compute_spectrum` no longer use
  stars interferogram to recover the first order coefficient but use
  the mean of the first order phase map. The precision is far better.

Better stars interferogram at merging step
------------------------------------------
* :py:meth:`~process.InterferogramMerger.merge` compute better stars
  interferograms. :py:meth:`~utils.fit_stars_in_frame` and
  :py:meth:`~utils.fit_gaussian2d` have been modified to give better
  fitted photometry points and retry a fit if it fails because the
  stars is not centered in the box.

v3.4.2
======

Bad frames vectors management
-----------------------------

The idea is to use the different processes to detect bad frames and
collect their bad frames vectors to suppress bad frames prior to the
transformation of the interferograms.

* :py:meth:`~process.InterferogramMerger.merge` creates a bad frames
  vector using a threshold of transmission (70%).

* :py:meth:`~orbs.create_bad_frames_vector` has been created to
  collect the bad frames vector created by various processes and
  create a full bad frames vector which can be passed
  :py:meth:`~process.Interferogram.compute_spectrum` and
  :py:meth:`~utils.transform_interferogram` in order to remove all the
  detected bad frames prior to transform the interferograms.

Zeros smoothing
---------------

* :py:meth:`~utils.transform_interferogram` has been modified to do
  what we call the zeros smoothing. The objective is to reduce ringing
  due to steep transition between 'normal' points and zeros. The
  interferogram is multiplied by a function which smoothes the
  transition between zeros parts and 'good parts' of the
  interferogram. The good parts symmetrical to the zeros parts (The
  ZPD is the center of symmetry) are multiplied by 2. And the same
  transition is applied from parts multpilied by 2 to parts
  multplied by 1. This way the same weight is given to each and every
  point of the interferogram (points multiplied by zero have their
  symmetrical point multplied by 2). The degree of smoothing can be
  choosen (smoothing_deg option). A higher degree means a smoother
  transition between one part to another but may reduce the SNR.

* :py:meth:`~utils.smooth` can now smooth a vector using a gaussian
  kernel convolution (much faster).

v3.4.3
======

* Minor bugs corrections

Better fit of stars
-------------------

* :py:meth:`~utils.fit_stars_in_frame` and
  :py:meth:`~utils.fit_gaussian2d` modified to give better fitting
  results. Especially for the method
  :py:meth:`~process.InterferogramMerger.merge` which depends a lot on
  a good fit of all the detected stars.


v3.4.4
======

Various phase fit degree
------------------------

* :py:meth:`~process.Phase.create_phase_maps` and
  :py:meth:`~orbs.Orbs.compute_spectrum` modified to use any order of
  the polynomial fit to the phase

* **config.orb**: New keyword PHASE_FIT_DEG to configure the desired
    degree of the polynomial fit o the
    phase. :py:meth:`~orbs.Orbs.__init__` modified to use this
    keyword.

v3.4.5 (stable)
===============

* Minor modifications of :py:meth:`~process.InterferogramMerger.merge`
  to make it more stable

* PHASE_FIT_DEG in **config.orb** set to 1

This version is considered as stable.

v3.4.6
======

Enhanced phase determination
----------------------------

* :py:meth:`~utils.get_lr_phase` window changed to a NORTON_BEER 2.0
  to get a phase with much less artefacts: give a much more precise
  phase and thus much more precise phase maps.

* :py:meth:`~process.Interferogram.compute_phase_coeffs_vector` use a
  cleaner way to get the median phase coefficient for each phase map:
  used points are choosen from the residual map created by
  :py:meth:`~process.Phase.create_phase_maps` and sigma-clipped before the mean
  is taken from a well defined gaussian-like distribution of phase
  coefficients.


v3.4.7
======

Reversed spectrum corection
---------------------------

.. note:: The problem comes from the 0th order phase map which is
  defined modulo PI. An addition of PI on the phase vector (thus on
  the 0th order of the polynomial) reverses the returned spectrum.

* :py:meth:`~process.Interferogram.compute_spectrum` modified to avoid the
  spectrum to be reversed (values are negative instead of positive)
  after phase correction. Spectrum polarity is checked using a mean
  interferogram over the whole cube. If the resulting spectrum is
  reversed the whole 0th order phase map is added PI.

  

Sky transmission correction in single-camera mode
-------------------------------------------------

* :py:meth:`~process.Interferogram.create_correction_vectors` created to get
  the correction vectors (sky transmission and added light) and
  correct interferograms in single camera-mode. Now **phase
  correction** and **sky transmission correction** are available in
  single camera-mode (but less precise than in binocular mode).

.. note:: The sky transmission vector is computed from star
  photometry. Its precision is good but it must be corrected for ZPD
  because with only one camera and near the ZPD stars interferograms
  are not 'flat' anymore. The 'added light' vector is computed from a
  median 'interferogram' It has also to be corrected near the ZPD.

* minor bugs correction and enhancements


Passing alignment parameters to orbs command
--------------------------------------------

* **orbs** script: new option : **--align** to pass precomputed alignement
  parameters. Useful in the case of the computation of a FLAT cube
  (with no possible alignment) if the alignment parameters are already
  knowm from the reduction of an object taken during the same mission.


v3.4.8
======

Master combination algorithms
-----------------------------

* :py:meth:`~process.RawData._create_master_frame` created to use
  better combination algorithms for the creation of master bias, master
  dark and master flat. Some pixels are rejected using a rejection
  algorithm before the images are combined using a median or an
  average function. The rejection algorithms proposed are:

    * Min-Max rejection
    * Sigma-Clipping
    * Average Sigma-Clipping (default)
    
  Master frames are also written to the disk for checking
  purpose. Note that those rejection algorithm have been inspired by
  the IRAF function combine.

* :py:meth:`~process.RawData.detect_stars` also uses
  :py:meth:`~process.RawData._create_master_frame` to combine frames.
  
Minor modifications
-------------------

* DATA_FRAMES in :py:meth:`~process.RawData.detect_stars` changed from
  10 to 30. Help in finding more stars in some cubes.

* :py:meth:`~orbs.Orbs._create_list_from_dir` now check if all the FITS
  files in the directory have the same shape.

* :py:meth:`~process.RawData.correct_frame` and
  :py:meth:`~process.RawData.create_interferogram` modified to
  correct for bias, flat and dark even if one of them are not given
  (before, without biases no correction at all would have been made)

* :py:meth:`~process.Spectrum.correct_filter` modified when filter min
  or max are outside the spectrum.

* :py:class:`~orbs.Orbs.__init__` prints modules versions

* :py:class:`~orbs.Orbs.__init__` modified. It is now possible to
  change configuration options for a particular reduction using the
  option file. Keywords are the same.

* :py:meth:`~process.RawData.correct_frame` modified to avoid strange
  behaviour when dark level is too low.


v3.5
****

3.5.0
=====

Alignment and photometry
-------------------------

:py:class:`~astrometry.Astrometry` class created with a whole new
astrometry module. This module is used for all astrometrical processes
(star position detection for alignment and star
photometry). Astrometry and photometry precision are now a lot better.


Merging process
---------------

* Single camera reduction: A new step of reduction has been added to
  better correct single camera interferograms for variations of
  transmission and light refracted on clouds.

* 2-camera reduction without merging frames (optional):
  :py:meth:`~process.InterferogramMerger.merge` has a better way of
  correct interferograms without merging frames. Camera 2 frames are
  used to create correction vectors but are not merged to the frames
  of the camera 1.


Cosmic Ray Detection
--------------------

:py:meth:`~process.RawData.create_cosmic_ray_map`. Completly changed
and upgraded using ORUS simulated cubes. Faster and far more
efficient. 95% of good detection over CR's with an energy higher than
the median CR energy. Small number of false detections. Less problems
with stars and ZPD.

FFT
---

'Zeros smoothing' step in :py:meth:`~utils.transform_interferogram`
modified to avoid correcting very small zeros parts (CR, bad frames)
which was creating noise.


3.5.1
=====

Minor bugs correction

3.5.2
=====

Alternative Merging Process
---------------------------

Addition of an alternative merging process
(:py:meth:`~process.InterferogramMerger.alternative_merge`): in fact,
this is the basic merging process which makes no use of star
photometry. This alternative way of merging, somehow more noisy than
the regular way, is more robust and might be the best guess if there's
not enough good stars in the field or when all the fiel is covered
with intense emission lines. It is recommanded to always do the
reduction this way and the regular way to take what seems the best
cube.

The new option in the orbs launch command is::

 --alt_merge

3.5.3
=====

Aperture Photometry
-------------------

An aperture photometry function
(:py:meth:`~astrometry.aperture_photometry`) has been designed to get
a far more robust and precise photometry of the stars during the
'normal' merging process. Sky transmission vector precision is now a
lot better and do not need any more smoothing.


Cosmic Ray Detection
--------------------

New step frame check added (removed a long time ago but added once
again) to get rid of star detection in some frames due to disalignment
and the size of interferometric fringes. Avoid getting bad photometry
on stars.

Mask
----

Frames created by ORBS are coupled with a mask frame. Mask frames are
used to get the exact position of all the pixels affected by the
cosmic rays correction. Cosmic rays correction in stars creates bad
pixels that have to be taken into account during the photometry
process t avoid too deviant values.


Tuning parameters
-----------------

It is now possible to tune the parameters of some methods externally
(in the option file). To tune a 'tunable' parameter you must use the
keyword TUNE, give the full name of the method parameter
(class.method.parameter_name) and its new value::
  
  TUNE InterferogramMerger.find_alignement.BOX_SIZE_COEFF 7.

.. note:: All the parameters are not tunable: this option has to be
  implemented in the method itself with the method
  :py:meth:`~core.Tools._get_tuning_parameter`.

.. warning:: This possibility is intented be used only for the
  reduction of some particular cubes. If the default value of a
  paramater has to be changed it is better to do it in the method
  itself.

3.5.4
=====

Astrometry & Photometry
-----------------------

Astrometry and photometry processes (fit and aperture) upgraded. They
know meet the theoretical error and their returned reduced-chi-square
is far better. All dependant processes in the process module have been
updated to use this better information and filter bad fitted stars.


v3.6
****

3.6.0
=====

Flux Calibration
----------------

Flux calibration has been added. :py:meth:`~orbs.Orbs.calibrate_spectrum`
replace the old function :py:meth:`~orbs.Orbs.correct_spectrum`. The path
to a standard spectrum reduced by ORBS must be given (STDPATH). This
can be achieved by reducing a standard cube using the option
--standard. The standard name must also be given in the option file
(STDNAME). This must be recorded in the standard table
(orbs/data/std_table). Standard spectra form MASSEY et al. 1988 and
CALSPEC have been added to the data of ORBS so that they can be used
to do a flux calibration. To do a flux calibration the steps are thus:

  1. Reduce standard cube with option --standard

  2. Give the path to the standard spectrum (STDPATH) and the name of
     the standard (STDNAME) in the option file of the cube yo want to
     calibrate

  3. Reduce the cube you want to calibrate (or only redo the last step)
  
.. seealso:: :py:meth:`~process.Spectrum.get_flux_calibration_vector`

.. note:: The standard cube must be reduced with the same number of
     camera as the cube you want to reduce.

.. note:: A new class has been created to manage standard spectra:
     :py:class:`~process.Standard`

.. note:: Final spectrum cube is now rescaled pixel to pixel in order
     to keep the same energy at the input and at the output of the
     reduction process. With 2 cubes we use the scale map. The scale
     map is the sum of the deep frame of both cubes ; The deep frame
     of cube A is scaled by the modulation coefficient which comes
     from the difference of gain between both cameras. For a single
     cube reduction, its own deep frame is used.

WCS correction
--------------

If the ra/dec (TARGETR/TARGETD) and x/y (TARGETX/TARGETY)
corresponding position of a target near the center of the frame is
given, WCS coords of the cube are updated at the last step (Calibration step
step): 

.. seealso:: :py:meth:`~process.Spectrum.get_corrected_wcs`

.. note:: An **internet connection** must be available to correct WCS
     because the USNO database is used to get precise astrometric
     coordinates of the stars in the field.

.. warning:: A new module is now required to launch ORBS: PyWCS (see
    http://stsdas.stsci.edu/astrolib/pywcs/)



Simplification
--------------

  * :py:class:`core.Indexer` created to index reduction files and get
    their location easily in :py:class:`orbs.Orbs`.

  * No more quadrants: Reduction of big data cubes has been simplified
    and do not save reduced files in quadrants any more. Big data
    cubes are thus handled as small data cubes. Reduced by quadrants
    but saved as one set of frames.

3.6.2
=====

Cython & speed optimization
---------------------------

  * :py:meth:`utils.transform_frame` has been modified to do only one
    geometrical transformation instead of a set of transformations
    (tip-tilt then translations then rotation etc.). Coordinates
    transformation function (:meth:`cutils.transform_A_to_B`)
    written in Cython to optimize processing speed .

  * core functions for fitting stars
    (:meth:`cutils.gaussian_array2d` and
    :meth:`cutils.surface_value`) have been transcripted to Cython
    for faster processing.

  * lots of functions have been cythonized to improve the overall
    speed.

v3.7
****

3.7.0
=====

ORB: A new core module
----------------------

ORBS core classes and functions (core.py, utils.py and cutils.pyx)
have been moved to a module of shared core libraries: ORB. This way,
ORBS, ORCS, OACS, IRIS and ORUS can share the same core module without
importing ORBS entirely each time. Conceptually ORBS, like the others,
just wraps around a core module and is not any more a central part ot
the whole suite of softwares.
