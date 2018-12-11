#!/usr/bin/env python
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: astrometry.py

## Copyright (c) 2010-2017 Thomas Martin <thomas.martin.1@ulaval.ca>
## 
## This file is part of ORB
##
## ORB is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ORB is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

"""
The Astrometry module is aimed to manage all astrometry processes:
Fitting star position and photometry for alignement and cubes merging.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'

import os
import math
import warnings
import time
import logging
import warnings

import numpy as np
from scipy import optimize, interpolate, signal
import astropy.wcs as pywcs
import bottleneck as bn

import core
__version__ = core.__version__
from core import Tools, ProgressBar

import utils.astrometry
import utils.image
import utils.stats
import utils.vector
import utils.web
import utils.misc
import utils.io

import pyximport; pyximport.install(
    setup_args={"include_dirs":np.get_include()})


##################################################
#### CLASS Astrometry ############################
##################################################

class Astrometry(Tools):
    """Manage all astrometrical processes that can be made on a single
    frame of a whole cube.

    This class can detect stars in the given data and return fit
    parameters. Fit parameters are returned as a
    :py:class:`~astrometry.StarsParams` object used to store and access
    fit parameters.

    Possible fitting profiles are at least Gaussian and Moffat. Other
    profiles can be created by expanding :py:class:`~astrometry.PSF`
    class.
    """
    
    def __init__(self, data, profile_name=None,
                 fwhm_arc=None,
                 detect_stack=None, fit_tol=1e-2, moffat_beta=None,
                 star_list_path=None, box_size_coeff=7.,
                 reduced_chi_square_limit=1.5,
                 target_radec=None, target_xy=None, wcs_rotation=None,
                 sip=None, **kwargs):

        """
        Init astrometry class.

        :param data: Can be an 2D or 3D Numpy array or an instance of
          core.Cube class. Note that the frames must not be too
          disaligned (a few pixels in both directions).

        :param fwhm_arc: (Optional) Rough FWHM of the stars in arcsec
          (default None). If None, the instrument configuration is
          used.

        :param profile_name: (Optional) Name of the PSF profile to use
          for fitting. Can be 'moffat' or 'gaussian' (default
          None, set to config value).

        :param detect_stack: (Optional) Number of frames to stack
          before detecting stars (default None, set to config value).

        :param fit_tol: (Optional) Tolerance on the paramaters fit
          (the lower the better but the longer too) (default 1e-2).

        :param moffat_beta: (Optional) Default value of the beta
          parameter for the moffat profile (default None, set to
          config value).

        :param star_list_path: (Optional) Path to a file containing a
          list of star positions (default None).

        :param box_size_coeff: (Optional) Coefficient giving the size
          of the box created around each star before a fit is
          done. BOX_SIZE = box_size_coeff * STAR_FWHM. (default 10.).
          Note that this coeff is divided by 3 if the moffat profile
          is used (helps to avoid bad fit).
        
        :param reduced_chi_square_limit: (Optional) Coefficient on the
          reduced chi square for bad quality fits rejection (default
          1.5)

        :param target_radec: (Optional) [RA, DEC] in degrees of a
          target near the center of the field. If the options
          target_xy (for the same target) and wcs rotation are also
          given , star detection will use a catalogue to get star
          positions in the field and WCS registration of an image or a
          cube is possible (default None).

        :param target_xy: (Optional) [X, Y] of a target near the
          center of the field. If the options target_radec (for the
          same target) and wcs rotation are also given , star
          detection will use a catalogue to get star positions in the
          field and WCS registration of an image or a cube is possible
          (default None).

        :param wcs_rotation: (Optional) Initial rotation angle of the
          field relatively to the North direction. Useful if the
          options target_radec and target_xy are also given. In this
          case, WCS registration and catalogued star detection are
          possible (default None).

        :param sip: (Optional) An astropy.wcs.WCS instance containing
          the SIP parameters of the distortion map (default None).

        :param kwargs: :py:class:`~orb.core.Tools` kwargs.
        """
        Tools.__init__(self, **kwargs)

        self.master_frame = None # Master frame created from a combination of
                                 # the frames of the cube
        self.box_size_coeff = None # Coefficient used to define the size of the
                                   # box from FWHM
        self.box_size = None # Size of the fitting box in pixel
        self.profile = None # A PSF class (e.g. Moffat or Gaussian)
        self.star_list = None # List of stars
        self.star_nb = None # Number of stars in the star list
        self._silent = False # If True only warnings and error message
                             # will be printed
        self.deep_frame = None # computed deep frame
        self.wcs = None # When data is registered this pywcs.WCS instance gives
                        # the corrected WCS.
        self.fit_results = None # Array containing all the resulting parameters
                                # (as dictionaries) of the fit of each
                                # star in each frame

        
        # load data and init parameters
        if isinstance(data, Cube):
            self.data = data
            self.dimx = self.data.dimx
            self.dimy = self.data.dimy
            self.dimz = self.data.dimz
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 2 or len(data.shape) == 3:
                self.data = data
                self.dimx = self.data.shape[0]
                self.dimy = self.data.shape[1]
                if len(data.shape) == 3:
                    self.dimz = self.data.shape[2]
                else:
                    self.dimz = 1
            else:
               raise StandardError("Data array must have 2 or 3 dimensions") 
        else:
            raise StandardError("Cube must be an instance of Cube class or a Numpy array")

                
    

    def _get_combined_frame(self, use_deep_frame=False, realign=False):
        """Return a combined frame to work on.

        :param use_deep_frame: (Optional) If True returned frame is a
          deep frame instead of a combination of the first frames only
          (default False)

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.
        """
        # shortcut
        if use_deep_frame and self.deep_frame is not None:
            return np.copy(self.deep_frame)

        _cube = None
        
        # realignment of the frames if necessary
        if realign and self.dimz > 1:
            _cube = self.data[:,:,:]
            _cube = utils.astrometry.realign_images(_cube)
                
        # If we have 3D data we work on a combined image of the first
        # frames
        if self.dimz > 1:
            _cube = None
                        
            if use_deep_frame:
                if _cube is None:
                    self.deep_frame = self.data.get_mean_image().astype(float)
                else:
                    self.deep_frame = np.nanmedian(_cube, axis=2)
                return np.copy(self.deep_frame)
            
            
            stack_nb = self.detect_stack
            if stack_nb + self.DETECT_INDEX > self.frame_nb:
                stack_nb = self.frame_nb - self.DETECT_INDEX

            dat = self.data[
                :,:, int(self.DETECT_INDEX):
                int(self.DETECT_INDEX+stack_nb)]
                
            if not self.config.BIG_DATA:
                im = utils.image.create_master_frame(dat)
            else:
                im = utils.image.pp_create_master_frame(dat)
                
        # else we just return the only frame we have
        else:
            im = np.copy(self.data)
            
        return im.astype(float)


    def set_deep_frame(self, deep_frame_path):
        deep_frame = utils.io.read_fits(deep_frame_path)
        if deep_frame.shape == (self.dimx, self.dimy):
            self.deep_frame = deep_frame
        else:
            raise StandardError('Deep frame must have the same shape')


            


    def fit_stars_in_cube(self, correct_alignment=False, save=False,
                          add_cube=None, hpfilter=False,
                          fix_height=True, fix_beta=True,
                          fix_fwhm=False,
                          fwhm_min=0.5, local_background=True,
                          no_aperture_photometry=False,
                          fix_aperture_size=False, precise_guess=False,
                          aper_coeff=3., blur=False, 
                          no_fit=False,
                          estimate_local_noise=True, multi_fit=False,
                          enable_zoom=False, enable_rotation=False,
                          saturation=None):
        
        """Fit stars in the cube.

        Frames must not be too disaligned. Disalignment can be
        corrected as the cube is fitted by setting correct_alignment
        option to True.

        .. note:: 2 fitting modes are possible::
        
            * Individual fit mode [multi_fit=False]
            * Multi fit mode [multi_fit=True]

            see :meth:`astrometry.utils.fit_stars_in_frame` for more
            information.
    
        :param correct_alignment: (Optional) If True, the initial star
          positions from the star list are corrected by their last
          recorded deviation. Useful when the cube is smoothly
          disaligned from one frame to the next.

        :param save: (Optional) If True save the fit results in a file
          (default True).

        :param add_cube: (Optional) A tuple [Cube instance,
          coeff]. This cube is added to the data before the fit so
          that the fitted data is self.data[] + coeff * Cube[].

        :param hpfilter: (Optional) If True, frames are HP filtered
          before fitting stars. Useful for alignment purpose if there
          are too much nebulosities in the frames. This option must
          not be used for photometry (default False).
    
        :param fix_height: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.fix_height`
          (default True)

        :param fix_beta: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.fix_beta` (default
          True).

        :param fix_fwhm: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.fix_fwhm`
          (default False)

        :param fwhm_min: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.fix_min` (default
          0.5)

        :param local_background: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.local_background`
          (default True).

        :param no_aperture_photometry: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.no_aperture_photometry`
          (default False).

        :param fix_aperture_size: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.fix_aperture_size`
          (default False).

        :param precise_guess: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.precise_guess`
          (default False).

        :param aper_coeff: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.aper_coeff`
          (default 3).

        :param blur: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.blur` (default
          False).

        :param no_fit: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.no_fit` (default
          False).

        :param estimate_local_noise: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.estimate_local_noise`
          (default True).

        :param multi_fit: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.multi_fit` (default
          False).

        :param enable_zoom: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.enable_zoom`
          (default False).

        :param enable_rotation: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.enable_rotation`
          (default False).
  
        :param saturation: (Optional) see
          :paramref:`utils.astrometry.fit_stars_in_frame.saturation`
          (default None).
        """
        def get_index_mean_dev(index):
            dx = utils.stats.robust_mean(utils.stats.sigmacut(
                self.fit_results[:,index,'dx']))
            dy = utils.stats.robust_mean(utils.stats.sigmacut(
                self.fit_results[:,index,'dy']))
            return dx, dy


        FOLLOW_NB = 5 # Number of deviation value to get to follow the
                      # stars
        
        logging.info("Fitting stars in cube")

        if self.data is None: raise StandardError(
            "Some data must be loaded first")
        
        if self.star_list is None: raise StandardError(
            "A star list must be loaded or created first")

        if self.dimz < 2: raise StandardError(
            "Data must have 3 dimensions. Use fit_stars_in_frame method instead")
        if fix_aperture_size:
            fix_aperture_fwhm_pix = self.fwhm_pix
        else:
            fix_aperture_fwhm_pix = None

        if add_cube is not None:
            if np.size(add_cube) >= 2:
                added_cube = add_cube[0]
                added_cube_scale = add_cube[1]
                if not isinstance(added_cube, Cube):
                    raise StandardError('Added cube must be a Cube instance. Check add_cube option')
                if np.size(added_cube_scale) != 1:
                    raise StandardError('Bad added cube scale. Check add_cube option.')
                
        self.fit_results = StarsParams(self.star_nb, self.frame_nb,
                                       silent=self._silent,
                                       instrument=self.instrument)

        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()

        frames = np.empty((self.dimx, self.dimy, ncpus), dtype=float)
        
        progress = ProgressBar(int(self.frame_nb), silent=self._silent)
        x_corr = None
        y_corr = None
        for ik in range(0, self.frame_nb, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.frame_nb):
                ncpus = self.frame_nb - ik
    
            if correct_alignment:
                if ik > 0:
                    old_x_corr = float(x_corr)
                    old_y_corr = float(y_corr)

                    if ik > FOLLOW_NB - 1:
                        # try to get the mean deviation over the
                        # last fitted frames
                        x_mean_dev = [get_index_mean_dev(ik-ifol-1)[0]
                                      for ifol in np.arange(FOLLOW_NB)]
                        y_mean_dev = [get_index_mean_dev(ik-ifol-1)[1]
                                      for ifol in np.arange(FOLLOW_NB)]
                        x_corr = utils.stats.robust_median(x_mean_dev)
                        y_corr = utils.stats.robust_median(y_mean_dev)
                    else:
                        x_corr, y_corr = get_index_mean_dev(ik-1)

                    if np.isnan(x_corr):
                        x_corr = float(old_x_corr)
                    if np.isnan(y_corr):
                        y_corr = float(old_y_corr)
                    
                else:
                    x_corr = 0.
                    y_corr = 0.

                star_list = np.copy(self.star_list)
                star_list[:,0] += x_corr
                star_list[:,1] += y_corr

            else:
                star_list = self.star_list

            # follow FWHM variations
            if ik > FOLLOW_NB - 1 and not no_fit:
                fwhm_mean = utils.stats.robust_median(
                    [utils.stats.robust_mean(utils.stats.sigmacut(
                        self.fit_results[:,ik-ifol-1,'fwhm_pix']))
                     for ifol in np.arange(FOLLOW_NB)])
                
                if np.isnan(fwhm_mean):
                    fwhm_mean = self.fwhm_pix
            else:
                fwhm_mean = self.fwhm_pix
          

            for ijob in range(ncpus):
                frame = np.copy(self.data[:,:,ik+ijob])
                
                # add cube
                if add_cube is not None:
                    frame += added_cube[:,:,ik+ijob] * added_cube_scale
        
                if hpfilter:
                    frame = utils.image.high_pass_diff_image_filter(
                        frame, deg=2)
                    
                frames[:,:,ijob] = np.copy(frame)

            # get stars photometry for each frame
            jobs = [(ijob, job_server.submit(
                utils.astrometry.fit_stars_in_frame,
                args=(frames[:,:,ijob], star_list, self.box_size,
                      self.profile_name, self.scale, fwhm_mean,
                      self.default_beta, self.fit_tol, fwhm_min,
                      fix_height, fix_aperture_fwhm_pix, fix_beta, fix_fwhm,
                      local_background, no_aperture_photometry,
                      precise_guess,
                      aper_coeff, blur, no_fit, estimate_local_noise,
                      multi_fit, enable_zoom, enable_rotation, saturation),
                modules=("import logging",
                         "import orb.utils.stats",
                         "import orb.utils.image",
                         "import numpy as np",
                         "import math",
                         "import orb.cutils",
                         "import bottleneck as bn",
                         "import warnings",
                         "from orb.utils.astrometry import *")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                res = job()
                if res is not None:
                    for istar in range(len(star_list)):
                        self.fit_results[istar, ik+ijob] = res[istar]
                
            progress.update(ik, info="frame : " + str(ik))
            
        self._close_pp_server(job_server)
        
        progress.end()

        if save:
            self.fit_results.save_stars_params(self._get_fit_results_path())

        # print reduced chi square
        mean_red_chi_square = utils.stats.robust_mean(utils.stats.sigmacut(
            self.fit_results[:, 'reduced-chi-square']))
        
        logging.info("Mean reduced chi-square: %f"%mean_red_chi_square)
        
        return self.fit_results




    def get_alignment_vectors(self, fit_cube=False, min_coeff=0.2):
        """Return alignement vectors

        :param fit_cube: (Optional) If True, the cube is fitted before
          using the fit results to create the alignement vectors. Else
          the vectors are created using the fit results already in
          memory (default False).

        :param min_coeff: The minimum proportion of stars correctly
            fitted to assume a good enough calculated disalignment
            (default 0.2).
        """
        # Filter frames before alignment
        HPFILTER = int(self._get_tuning_parameter('HPFILTER', 0))
        
        if self.data is None: raise StandardError(
            "Some data must be loaded first")
        
        if self.star_list is None: raise StandardError(
            "A star list must be loaded or created first")
    
        if fit_cube:
            self.fit_stars_in_cube(correct_alignment=True,
                                   no_aperture_photometry=True,
                                   hpfilter=HPFILTER, multi_fit=False,
                                   fix_height=False, save=False)
      
        if self.star_nb < 4: 
            raise StandardError("Not enough stars to align properly : %d (must be >= 3)"%self.star_nb)
            
        fit_x = self.fit_results[:,:,'x']
        fit_y = self.fit_results[:,:,'y']
        rcs = self.fit_results[:,:,'reduced-chi-square']
        fit_x_err = self.fit_results[:,:,'x_err']
        fit_y_err = self.fit_results[:,:,'y_err']
    
        start_x = np.squeeze(np.copy(fit_x[:, 0]))
        start_y = np.squeeze(np.copy(fit_y[:, 0]))

        # Check if enough stars have been fitted in the first frame
        good_nb = len(np.nonzero(~np.isnan(start_x))[0])
        
        if good_nb < 4 or good_nb < min_coeff * self.star_nb:
            raise StandardError("Not enough detected stars (%d) in the first frame"%good_nb)

        ## Create alignment vectors from fitted positions
        alignment_vector_x = ((fit_x.T - start_x.T).T)[0,:]
        alignment_vector_y = ((fit_y.T - start_y.T).T)[0,:]
        alignment_error = np.sqrt(fit_x_err[0,:]**2. + fit_y_err[0,:]**2.)

        # correct alignment vectors for NaN values
        alignment_vector_x = utils.vector.correct_vector(
            alignment_vector_x, polyfit=True, deg=3)
        alignment_vector_y = utils.vector.correct_vector(
            alignment_vector_y, polyfit=True, deg=3)

        # print some info
        logging.info(
            'Alignment vectors median error: %f pixel'%utils.stats.robust_median(alignment_error))
                
        return alignment_vector_x, alignment_vector_y, alignment_error




##################################################
#### CLASS Aligner ###############################
##################################################
            
class Aligner(Tools):
    """This class is aimed to align two images of the same field of
    stars and correct for optical distortions.

    Primarily designed to align the cube of the camera 2 onto the cube
    of the camera 1 it can be used to align any other kind of images
    containing stars.
    """
 
    saturation_threshold = None # saturation threshold

    image1 = None # 1st image
    image2 = None # 2nd image
    bin1 = None # binning of the 1st image
    bin2 = None # binning of the 2nd image
    pix_size1 = None # pixel size of the 1st image im um
    pix_size2 = None # pixel size of the 2nd image in um
    

    sip1 = None # pywcs.WCS() instance of the 1st image
    sip2 = None # pywcs.WCS() instance of the 2nd image
    
    astro1 = None # Astrometry instance of the 1st image
    astro2 = None # Astrometry instance of the 2nd image

    search_size_coeff = None # Define the range of pixels around the
                             # initial shift values where the correct
                             # shift parameters have to be found
                             # (default 0.01).
          
    # transformation parameters
    dx = None
    dy = None
    dr = None
    da = None
    db = None
    rc = None
    zoom_factor = None
            
    def __init__(self, image1, image2, fwhm_arc, fov1, fov2,
                 bin1, bin2, pix_size1, pix_size2,
                 init_angle, init_dx, init_dy,
                 sip1=None, sip2=None,
                 saturation_threshold=60000,
                 project_header=list(), overwrite=False,
                 **kwargs):
        """Aligner init   

        :param image1: Image 1.

        :param image2: Image 2.

        :param fwhm_arc: rough FWHM of the stars in arcseconds.

        :param fov1: Field of view of the image 1.

        :param fov2: Field of view of the image 2.
        
        :param bin1: Binning of the image 1.
        
        :param bin2: Binning of the image 2.

        :param init_angle: Initial guess on the angle between the two
          images.

        :param init_dx: Initial guess of the translation along the X
          axis (in binned pixels).

        :param init_dy: Initial guess of the translation along the Y
          axis (in binned pixels).

        :param sip1: (Optional) A pywcs.WCS() instance containing the
          SIP parameters of the image 1 (default None).

        :param sip2: (Optional) A pywcs.WCS() instance containing the
          SIP parameters of the image 2 (default None).

        :param saturation_threshold: (Optional) Saturation threshold
          of the detectors in the intensity unit of the images
          (default 60000, for images in counts).    

        :param project_header: (Optional) header section to be added
          to each output files based on merged data (an empty list by
          default).

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        self.overwrite = overwrite
        self._project_header = project_header
        
        self.range_coeff = float(self._get_tuning_parameter(
            'RANGE_COEFF', self.config.ALIGNER_RANGE_COEFF))
        
        self.saturation_threshold = saturation_threshold
        
        self.image1 = image1
        self.image2 = image2

        self.bin1 = bin1
        self.bin2 = bin2
        self.pix_size1 = pix_size1
        self.pix_size2 = pix_size2
        self.rc = [self.image2.shape[0] / 2.,
                   self.image2.shape[1] / 2.]

        self.sip1 = sip1
        self.sip2 = sip2

        self.zoom_factor = ((float(self.pix_size2) * float(self.bin2)) / 
                            (float(self.pix_size1) * float(self.bin1)))
        
        self.astro1 = Astrometry(self.image1, profile_name='gaussian', instrument=self.instrument)
        self.astro2 = Astrometry(self.image2, profile_name='gaussian', instrument=self.instrument)

        self.dx = init_dx
        self.dy = init_dy
        self.dr = init_angle
        self.da = 0.
        self.db = 0.


    def _get_guess_matrix_path(self):
        """Return path to the guess matrix"""
        return self._data_path_hdr + "guess_matrix.fits"

    def _get_guess_matrix_header(self):
        """Return path to the guess matrix"""
        return (self._get_basic_header('Alignment guess matrix') +
                self._project_header)
    
    def print_alignment_coeffs(self):
        """Print the alignement coefficients."""
        logging.info("\n> dx : " + str(self.dx) + "\n" +
                        "> dy : " + str(self.dy) + "\n" +
                        "> dr : " + str(self.dr) + "\n" +
                        "> da : " + str(self.da) + "\n" +
                        "> db : " + str(self.db))
    
    def compute_alignment_parameters(self, correct_distortion=False,
                                     star_list_path1=None, fwhm_arc=None,
                                     brute_force=True):
        """Return the alignment coefficients that match the stars of the
        frame 2 to the stars of the frame 1.

        :param correct_distortion: (Optional) If True, a SIP is computed to
          match stars from frame 2 onto the stars from frame 1. But it
          needs a lot of stars to run correctly (default False).

        :param star_list_path1: (Optional) Path to a list of stars in
          the image 1. If given the fwhm_arc must also be set (default None).

        :param fwhm_arc: (Optional) mean FWHM of the stars in
          arcseconds. Must be given if star_list_path1 is not None
          (default None).

        :param brute_force: (Optional) If True the first step is a
          brute force guess. This is very useful if the initial
          parameters are not well known (default True).

        .. note:: The alignement coefficients are:
        
          * dx : shift along x axis in pixels
          
          * dy : shift along y axis in pixels
          
          * dr : rotation angle between images (the center of rotation
            is the center of the images of the camera 1) in degrees
            
          * da : tip angle between cameras (along x axis) in degrees
          
          * db : tilt angle between cameras (along y axis) in degrees

        .. note:: The process tries to find the stars detected in the camera A in the frame of the camera B. It goes through 2 steps:

           1. Rough alignment (brute force style) only looking over
              dx, dy. dr is kept to its initial value (init_angle), da
              and db are set to 0.

           2. Fine alignment pass.

        .. warning:: This alignment process do not work if the initial
          parameters are too far from the real value. The angle must
          be known within a few degrees. The shift must be known
          within 4 % of the frame size (The latter can be changed
          using the SIZE_COEFF constant)
        """

        def match_star_lists(p, slin, slout, rc, zf, sip1, sip2):
            """return the transformation parameters given two list of
            star positions.
            """
            def diff(p, slin, slout, rc, zf, sip1, sip2):
                slin_t = utils.astrometry.transform_star_position_A_to_B(
                    slin, p, rc, zf,
                    sip_A=sip1, sip_B=sip2)
                result = (slin_t - slout).flatten()
                return result[np.nonzero(~np.isnan(result))]

            try:
                fit = optimize.leastsq(diff, p,
                                       args=(slin, slout, rc, zf, sip1, sip2),
                                       full_output=True, xtol=1e-6)
            except Exception, e:
                raise Exception('No matching parameters found: {}'.format(e))
            
            if fit[-1] <= 4:
                match = np.sqrt(np.mean(fit[2]['fvec']**2.))
                if match > 1e-3:
                    warnings.warn('Star lists not perfectly matched (residual {} > 1e-3)'.format(match))
                return fit[0]
            
            else:
                raise Exception('No matching parameters found')

        def brute_force_alignment(xystep_size, angle_range, angle_steps, range_coeff):
            # define the ranges in x and y for the rough optimization
            x_range_len = range_coeff * float(self.astro2.dimx)
            y_range_len = range_coeff * float(self.astro2.dimy)

            x_hrange = np.arange(xystep_size, x_range_len/2, xystep_size)
            x_range = np.hstack((-x_hrange[::-1], 0, x_hrange)) + self.dx
            
            y_hrange = np.arange(xystep_size, y_range_len/2, xystep_size)
            y_range = np.hstack((-y_hrange[::-1], 0, y_hrange)) + self.dy
            
          
            r_range = np.linspace(-angle_range/2.,
                                  angle_range/2.,
                                  angle_steps) + self.dr

            (self.dx, self.dy, self.dr, guess_matrix) = (
                utils.astrometry.brute_force_guess(
                    self.image2, self.astro1.star_list,
                    x_range, y_range, r_range,
                    self.rc, self.zoom_factor,
                    self.astro2.fwhm_pix * 3.))
            self.da = 0.
            self.db = 0.

            # Save guess matrix
            utils.io.write_fits(self._get_guess_matrix_path(),
                                guess_matrix,
                                fits_header=self._get_guess_matrix_header(),
                                overwrite=self.overwrite)

        
            
        ERROR_RATIO = 0.2 # Minimum ratio of fitted stars once the
                          # optimization pass has been done. If
                          # the ratio of fitted stars is less than
                          # this ratio an error is raised.

        WARNING_RATIO = 0.5 # If there's less than this ratio of
                            # fitted stars after the 
                            # optimization pass a warning is printed.

        WARNING_DIST = .3 # Max optimized distance in arcsec before a
                          # warning is raised
                          
        ERROR_DIST = 2.* WARNING_DIST # Max optimized distance in
                                      # arcsec before an error is
                                      # raised
        
        MIN_STAR_NB = 30 # Target number of star to detect to find the
                         # transformation parameters

        XYSTEP_SIZE = 0.5 # Pixel step size of the search range

        ANGLE_STEPS = 10 # Angle steps for brute force guess
        ANGLE_RANGE = 1. # Angle range for brute force guess
        
        # Skip fit checking
        SKIP_CHECK = bool(int(self._get_tuning_parameter('SKIP_CHECK', 0)))

        if star_list_path1 is None:
            star_list_path1, fwhm_arc = self.astro1.detect_stars(
                min_star_number=MIN_STAR_NB,
                saturation_threshold=self.saturation_threshold,
                no_save=True)
        elif fwhm_arc is not None:
            self.astro1.load_star_list(star_list_path1)
        else:
            raise StandardError('If the path to a list of stars is given (star_list_path1) the fwhm in arcsec(fwhm_arc) must also be given.')

        self.astro2.reset_fwhm_arc(fwhm_arc)
        self.astro1.reset_fwhm_arc(fwhm_arc)


        ##########################################
        ### BRUTE FORCE GUESS (only dx and dy) ###
        ##########################################
        if brute_force:
            logging.info("Brute force guess on large field")
            brute_force_alignment(4*XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS/2, self.range_coeff*10)
            logging.info("Brute force guess:") 
            self.print_alignment_coeffs()

            logging.info("Finer brute force guess")
            brute_force_alignment(XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS, self.range_coeff)
            logging.info("Brute force guess:") 
            self.print_alignment_coeffs()


        guess = [self.dx, self.dy, self.dr, self.da, self.db]
        
        ##########################
        ## FINE ALIGNMENT STEP ###
        ##########################
        
        # create sip corrected and transformed list
        star_list2 = utils.astrometry.transform_star_position_A_to_B(
            np.copy(self.astro1.star_list), guess, self.rc, self.zoom_factor,
            sip_A=self.sip1)

        self.astro2.reset_star_list(star_list2)

        fit_results = self.astro2.fit_stars_in_frame(
            0, no_aperture_photometry=True,
            multi_fit=True, enable_zoom=False,
            enable_rotation=True, fix_fwhm=True,
            sip=self.sip2, save=False)
        
        [self.dx, self.dy, self.dr, self.da, self.db] = match_star_lists(
            guess, np.copy(self.astro1.star_list), fit_results.get_star_list(
                all_params=True),
            self.rc, self.zoom_factor, sip1=self.sip1, sip2=self.sip2)

        logging.info("Fine alignment parameters:") 
        self.print_alignment_coeffs()
                

        #####################################
        ### COMPUTE DISTORTION CORRECTION ###
        #####################################

        if correct_distortion:
            logging.info('Computing distortion correction polynomial (SIP)')
            raise Exception('Must be checked. using transformation parameters with sip may not be implemented properly.')
            # try to detect a maximum number of stars in frame 1
            star_list1_path1, fwhm_arc = self.astro1.detect_stars(
                min_star_number=400,
                saturation_threshold=self.saturation_threshold,
                no_save=True, r_max_coeff=2.)

            ############################
            ### plot stars positions ###
            ############################
            ## import pylab as pl
            ## im = pl.imshow(self.image1.T, vmin=0, vmax=1000)
            ## im.set_cmap('gray')
            ## pl.scatter(self.astro1.star_list[:,0], self.astro1.star_list[:,1],
            ##            edgecolor='red', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            ## pl.show()

            star_list2 = utils.astrometry.transform_star_position_A_to_B(
                np.copy(self.astro1.star_list),
                [self.dx, self.dy, self.dr, self.da, self.db],
                self.rc, self.zoom_factor,
                sip_A=self.sip1, sip_B=self.sip2)
            self.astro2.reset_star_list(star_list2)

            # fit stars
            fit_results = self.astro2.fit_stars_in_frame(
                0, no_aperture_photometry=True,
                multi_fit=False, enable_zoom=False,
                enable_rotation=False, 
                fix_fwhm=False, sip=None, save=False)
            err = fit_results[:,'x_err']


            ## FIT SIP 
            ## SIP 1 and SIP 2 are replaced by only one SIP that matches the
            ## stars of the frame 2 onto the stars of the frame 1
            self.sip1 = self.astro1.fit_sip(
                np.copy(self.astro1.star_list),
                fit_results.get_star_list(all_params=True),
                params=[self.dx, self.dy, self.dr, self.da, self.db,
                        self.rc[0], self.rc[1], self.zoom_factor],
                init_sip=None, err=None, crpix=self.sip1.wcs.crpix,
                crval=self.sip1.wcs.crval)
            self.sip2 = None


        else:
            # fit stars
            fit_results = self.astro2.fit_stars_in_frame(
                0, no_aperture_photometry=True,
                multi_fit=False, enable_zoom=False,
                enable_rotation=False, 
                fix_fwhm=False, sip=None, save=False)

            fitted_star_nb = float(np.sum(~np.isnan(
                fit_results.get_star_list(all_params=True)[:,0])))
            
            if (fitted_star_nb < ERROR_RATIO * MIN_STAR_NB):
                raise StandardError("Not enough fitted stars in both cubes (%d%%). Alignment parameters might be wrong."%int(fitted_star_nb / MIN_STAR_NB * 100.))
                
            if (fitted_star_nb < WARNING_RATIO * MIN_STAR_NB):
                warnings.warn("Poor ratio of fitted stars in both cubes (%d%%). Check alignment parameters."%int(fitted_star_nb / MIN_STAR_NB * 100.))

            
            err = fit_results[:,'x_err']
            
        star_list2 = utils.astrometry.transform_star_position_A_to_B(
        np.copy(self.astro1.star_list),
            [self.dx, self.dy, self.dr, self.da, self.db],
            self.rc, self.zoom_factor,
            sip_A=self.sip1, sip_B=self.sip2)
        self.astro2.reset_star_list(star_list2)

        fwhm_arc2 = utils.stats.robust_mean(
            utils.stats.sigmacut(fit_results[:, 'fwhm_arc']))
        
        dx_fit = (star_list2[:,0]
                  - fit_results.get_star_list(all_params=True)[:,0])
        dy_fit = (star_list2[:,1]
                  - fit_results.get_star_list(all_params=True)[:,1])
        dr_fit = np.sqrt(dx_fit**2. + dy_fit**2.)
        final_err = np.mean(utils.stats.sigmacut(dr_fit))

        if not SKIP_CHECK:
            if final_err < self.astro1.arc2pix(WARNING_DIST):
                logging.info('Mean difference on star positions: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
            elif final_err < self.astro1.arc2pix(ERROR_DIST):
                warnings.warn('Mean difference on star positions is bad: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
            else:
                raise StandardError('Mean difference on star positions is too bad: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
        

        ### PLOT ERROR ON STAR POSITIONS ###
        ## import pylab as pl
        ## scale = self.astro1.scale
        ## pl.errorbar(dx_fit*scale, dy_fit*scale, xerr=err*scale,
        ##             yerr=err*scale, linestyle='None')
        ## circ = pl.Circle([0.,0.], radius=fwhm_arc/2.,
        ##                  fill=False, color='g', linewidth=2.)
        ## pl.gca().add_patch(circ)
        ## pl.axes().set_aspect('equal')
        ## pl.grid()
        ## pl.xlim([-fwhm_arc/2.,fwhm_arc/2.])
        ## pl.ylim([-fwhm_arc/2.,fwhm_arc/2.])
        ## pl.show()

        return {'coeffs':[self.dx, self.dy, self.dr, self.da, self.db],
                'rc': self.rc,
                'zoom_factor': self.zoom_factor,
                'sip1': self.sip1,
                'sip2': self.sip2,
                'star_list1': self.astro1.star_list,
                'star_list2': self.astro2.star_list,
                'fwhm_arc1': fwhm_arc,
                'fwhm_arc2': fwhm_arc2}


