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

import pandas

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

    def fwhm(self, x):
        """Return fwhm from width

        :param x: width
        """
        return x * abs(2.*np.sqrt(2. * np.log(2.)))

    def width(self, x):
        """Return width from fwhm

        :param x: fwhm.
        """
        return x / abs(2.*np.sqrt(2. * np.log(2.)))

    def detect_stars(self, min_star_number=4, no_save=False,
                     saturation_threshold=35000, try_catalogue=False,
                     r_max_coeff=0.6, filter_image=True):
        """Detect star positions in data.

        :param index: Minimum index of the images used for star detection.
        
        :param min_star_number: Minimum number of stars to
          detect. Must be greater or equal to 4 (minimum star number
          for any alignment process).

        :param no_save: (Optional) If True do not save the list of
          detected stars in a file, only return a list (default
          False).


        :param try_catalogue: (Optional) If True, try to use a star
          catalogue (e.g. USNO-B1) to detect stars if target_ra,
          target_dec, target_x, target_y and wcs_rotation parameters
          have been given (see
          :py:meth:`astrometry.Astrometry.query_vizier`, default
          False).

        :param r_max_coeff: (Optional) Coefficient that sets the limit
          radius of the stars (default 0.6).

        :param filter_image: (Optional) If True, image is filtered
          before detection to remove nebulosities (default True).

        :return: (star_list_path, mean_fwhm_arc) : (a path to a list
          of the dected stars, the mean FWHM of the stars in arcsec)

        .. note:: Star detection walks through 2 steps:
        
           1. Preselection of 4 times the minimum number of stars to
              detect using a variable threshold with a filter for hot
              pixels and stars near the border of the image.

           2. Stars are fitted to test if they are 'real' stars. The
              most luminous stars (that do not saturate) are
              eventually taken.
        """
        def define_box(x,y,box_size,ima):
            minx = x - int(box_size/2.)
            if minx < 0: minx = 0
            maxx = x + int(box_size/2.) + 1
            if maxx > ima.shape[0]: maxx = ima.shape[0]
            miny = y - int(box_size/2.)
            if miny < 0: miny = 0
            maxy = y + int(box_size/2.) + 1
            if maxy > ima.shape[1] : maxy = ima.shape[1]
            return ima[minx:maxx, miny:maxy], [minx, miny]

        def test_fit(box, mins, profile_name, fwhm_pix,
                     default_beta, fit_tol, min_fwhm_coeff,
                     saturation_threshold, profile):

            height = np.median(box)

            params = fit_star(
                box, profile_name=profile_name,
                fwhm_pix=fwhm_pix,
                height=height,
                beta=default_beta, fix_height=True,
                fit_tol=fit_tol,
                fix_beta=True,
                fwhm_min=min_fwhm_coeff * fwhm_pix)
            if params != []:
                # eliminate possible saturated star
                if (params['height'] + params['amplitude']
                    < saturation_threshold):
                    # keep only a star far enough from another star or
                    # another bright point
                    
                    # 1 - remove fitted star from box
                    box -= profile(params).array2d(box.shape[0],
                                                   box.shape[1])
                    
                    # 2 - check pixels around
                    if np.max(box) > params['amplitude'] / 3.:
                        return []
                  
                    params['x'] += float(mins[0])
                    params['y'] += float(mins[1])
                else:
                    return []

            return params
           
        THRESHOLD_COEFF = 0.1
        """Starting threshold coefficient"""
        
        PRE_DETECT_COEFF = float(
            self._get_tuning_parameter('PRE_DETECT_COEFF', 8))
        """Ratio of the number of pre-detected stars over the minimum
        number of stars"""
        
        MIN_FWHM_COEFF = 0.5
        """Coefficient used to determine the minimum FWHM given the
        Rough stars FWHM. """

        
        # TRY catalogue
        if try_catalogue:
            if (self.target_ra is not None and self.target_dec is not None
                and self.target_x is not None and self.target_y is not None
                and self.wcs_rotation is not None):
                return self.detect_stars_from_catalogue(
                    min_star_number=min_star_number, no_save=no_save,
                    saturation_threshold=saturation_threshold)
         

        logging.info("Detecting stars")

        # high pass filtering of the image to remove nebulosities
        if filter_image:
            start_time = time.time()
            logging.info("Filtering master image")
            hp_im = utils.image.high_pass_diff_image_filter(self.data, deg=1)
            logging.info("Master image filtered in {} s".format(
                time.time() - start_time))
        else:
            hp_im = np.copy(self.data)

        # preselection
        logging.info("Stars preselection")
        mean_hp_im = np.nanmean(hp_im)
        std_hp_im = np.nanstd(hp_im)
        max_im = np.nanmax(self.data)
        # +1 is just here to make sure we enter the loop
        star_number = PRE_DETECT_COEFF * min_star_number + 1 
        
        old_star_list = []
        while(star_number > PRE_DETECT_COEFF * min_star_number):
            pre_star_list = np.array(np.nonzero(
                (hp_im > mean_hp_im + THRESHOLD_COEFF * std_hp_im)
                * (self.data < saturation_threshold)))
            star_list = list()
            for istar in range(pre_star_list.shape[1]):
                ix = pre_star_list[0, istar]
                iy = pre_star_list[1, istar]
                (box, mins)  = define_box(ix, iy, self.box_size, self.data)
                ilevel = self.data[ix, iy]
                if (ilevel == np.max(box)) and (ilevel <= max_im):
                    # filter stars too far from the center
                    cx, cy = self.dimx/2., self.dimy/2.
                    r_max = np.sqrt(cx**2. + cy**2.) * r_max_coeff
                    if np.sqrt((ix - cx)**2. + (iy - cy)**2.) <= r_max:
                        star_list.append([ix, iy])
                    
            star_number = np.array(star_list).shape[0]
            if star_number > PRE_DETECT_COEFF * min_star_number:
                old_star_list = star_list
            THRESHOLD_COEFF += 0.1
        if old_star_list != []:
            star_list = old_star_list
        else: 
            if star_number < min_star_number:
                warnings.warn(
                    "Not enough detected stars in the image : %d"%star_number)
        
        ### FIT POSSIBLE STARS ############
                
        # first fit test to eliminate "bad" stars
        
        logging.info("Bad stars rejection based on fitting")
        
        params_list = list()
        
        job_server, ncpus = self._init_pp_server()
        
        progress = core.ProgressBar(len(star_list), silent=False)
        for istar in range(0, len(star_list), ncpus):
            
            if istar + ncpus >= len(star_list):
                ncpus = len(star_list) - istar

            jobs = [(ijob, job_server.submit(
                test_fit, 
                args=(define_box(star_list[istar+ijob][0],
                                 star_list[istar+ijob][1],
                                 self.box_size, self.data)[0],
                      define_box(star_list[istar+ijob][0],
                                 star_list[istar+ijob][1],
                                 self.box_size, self.data)[1],
                      self.profile_name, self.fwhm_pix,
                      self.default_beta, self.fit_tol, MIN_FWHM_COEFF,
                      saturation_threshold, self.profile),
                modules=("import logging",
                         "import numpy as np",
                         "from orb.utils.astrometry import fit_star")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                params = job()
                if params != []:
                    params_list.append(params)
            progress.update(
                istar,
                info="Fitting star %d/%d"%(istar, len(star_list)))

        self._close_pp_server(job_server)
        progress.end()

        ### FIT CHECK ##############
        logging.info("Fit check")
        
        fitted_star_list = list()
        fwhm_list = list()
        snr_list = list()
        for params in params_list:
            fitted_star_list.append((params['x'],
                                     params['y'], 
                                     params['amplitude']))
            fwhm_list.append((params['fwhm_pix']))
            snr_list.append((params['snr']))

        if len(fwhm_list) == 0:
            raise StandardError("All detected stars have been rejected !")

        # check FWHM value to ensure that it is a star and reject too
        # large or too narrow structures (e.g. galaxies and hot pixels)
        median_fwhm = utils.stats.robust_median(utils.stats.sigmacut(
            fwhm_list, sigma=3.))
        std_fwhm = utils.stats.robust_std(utils.stats.sigmacut(
            fwhm_list, sigma=3.))
      
        istar = 0
        while istar < len(fwhm_list):
            if ((fwhm_list[istar] > median_fwhm + 3. * std_fwhm)
                or (fwhm_list[istar] < median_fwhm - 2. * std_fwhm)
                or fwhm_list[istar] < 1.):
                fitted_star_list.pop(istar)
                fwhm_list.pop(istar)
                snr_list.pop(istar)
            else:
                istar += 1

        # keep the brightest stars only
        fitted_star_list.sort(key=lambda star: star[2], reverse=True)
        star_list = fitted_star_list[:min_star_number]

        # write down detected stars
        mean_fwhm = np.mean(np.array(fwhm_list))

        star_list_file = utils.io.open_file(self._get_star_list_path(), 'w')
        for istar in star_list:
            star_list_file.write(str(istar[0]) + " " + str(istar[1]) + "\n")

        # Print some comments and check number of detected stars    
        logging.info("%d stars detected" %(len(star_list)))
        logging.info("Detected stars FWHM : %f pixels, %f arc-seconds"%(
            mean_fwhm, self.pix2arc(mean_fwhm)))
        snr_list = np.array(snr_list)
        logging.info("SNR Min: %.1e, Max:%.1e, Median:%.1e"%(
            np.min(snr_list), np.max(snr_list), np.median(snr_list)))
        
        if len(star_list) < min_star_number:
            warnings.warn(
                "Not enough detected stars in the image : %d/%d"%(
                    len(star_list), min_star_number))
        if len(star_list) < 4:
            raise StandardError(
                "Not enough detected stars: %d < 4"%len(star_list))

        self.reset_star_list(np.array(star_list)[:,:2])
        self.reset_fwhm_pix(mean_fwhm)
        
        return self._get_star_list_path(), self.pix2arc(mean_fwhm)
            






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




    def detect_stars_from_catalogue(self, min_star_number=4, no_save=False,
                                    saturation_threshold=35000):
        """Detect star positions in data from a catalogue.

        :param index: Minimum index of the images used for star detection.
        
        :param min_star_number: Minimum number of stars to
          detect. Must be greater or equal to 4 (minimum star number
          for any alignment process).

        :param no_save: if True do not save the list of detected stars
          in a file, only return a list (default False).

        :param saturation_threshold: Number of counts above which the
          star can be considered as saturated. Very low by default
          because at the ZPD the intensity of a star can be twice the
          intensity far from it (default 35000).

        :return: (star_list_path, mean_fwhm_arc) : (a path to a list
          of the dected stars, the mean FWHM of the stars in arcsec)
        """

        LIMIT_RADIUS_RATIO = 1.0 # radius ratio around the center of
                                 # the frame where the stars are kept
  
        logging.info("Detecting stars from catalogue")
        # during registration a star list compted from the catalogue
        # is created.
        self.register()

        fit_params = self.fit_stars(multi_fit=False,
                                    local_background=True,
                                    save=False)
        
        fitted_star_list = [[istar['x'], istar['y'],
                             istar['flux'], istar['snr']]
                            for istar in fit_params
                            if (istar is not None
                                and istar['amplitude'] < saturation_threshold)]
        snr_list = np.array(fitted_star_list)[:,3]

        # remove stars in the corners of the frame
        rcx = self.dimx / 2.
        rcy = self.dimy / 2.
        fitted_star_list = [
            istar for istar in fitted_star_list
            if (np.sqrt((istar[0] - rcx)**2. + (istar[1] - rcy)**2.)
                < LIMIT_RADIUS_RATIO * min(rcx, rcy))]

        # keep the brightest stars only
        fitted_star_list.sort(key=lambda star: star[3], reverse=True)
        
        star_list = np.array(fitted_star_list)[:min_star_number,:2]
        snr_list = snr_list[:min_star_number]
        
        # write down detected stars
        mean_fwhm = self.fwhm_arc
        star_list_file = utils.io.open_file(self._get_star_list_path())
        for istar in star_list:
            star_list_file.write(str(istar[0]) + " " + str(istar[1]) + "\n")

        # Print some comments and check number of detected stars    
        logging.info("%d stars detected" %(len(star_list)))
        logging.info("Detected stars FWHM : %f pixels, %f arc-seconds"%(
            mean_fwhm, self.pix2arc(mean_fwhm)))
        snr_list = np.array(snr_list)
        logging.info("SNR Min: %.1e, Max:%.1e, Median:%.1e"%(
            np.min(snr_list), np.max(snr_list), np.median(snr_list)))
        
        if len(star_list) < min_star_number:
            warnings.warn(
                "Not enough detected stars in the image : %d/%d"%(
                    len(star_list), min_star_number))
        if len(star_list) < 4:
            raise StandardError(
                "Not enough detected stars: %d < 4"%len(star_list))

        self.reset_star_list(star_list)
        
        return self._get_star_list_path(), self.pix2arc(mean_fwhm)


    def detect_all_sources(self):
        """Detect all point sources in the cube regardless of there FWHM.

        Galaxies, HII regions, filamentary knots and stars might be
        detected.
        """

        SOURCE_SIZE = 2

        def aggregate(init_source_list, source_size):
            
            px = list(init_source_list[0])
            py = list(init_source_list[1])
            
            sources = list()
            while len(px) > 0:
                source = list()
                source.append((px[0], py[0]))
                px.pop(0)
                py.pop(0)
                
                ii = 0
                while ii < len(px):
                    if ((abs(px[ii] - source[0][0]) <= source_size)
                        and (abs(py[ii] - source[0][1]) <= source_size)):
                        source.append((px[ii], py[ii]))
                        px.pop(ii), py.pop(ii)
                    else:
                        ii += 1

                if len(source) > source_size:
                    xmean = 0.
                    ymean = 0.
                    for ipoint in source:
                        xmean += float(ipoint[0])
                        ymean += float(ipoint[1])
                    xmean /= float(len(source))
                    ymean /= float(len(source))

                    sources.append((xmean, ymean))
                    
            return sources
        
        logging.info("Detecting all point sources in the cube")
        
        start_time = time.time()
        logging.info("Filtering master image")
        hp_im = utils.image.high_pass_diff_image_filter(self.data, deg=1)
        logging.info("Master image filtered in {} s".format(
            time.time() - start_time))


        # image is binned to help detection
        binning = int(self.fwhm_pix) + 1        
        hp_im = utils.image.nanbin_image(hp_im, binning)

        # detect all pixels above the sky theshold
        detected_pixels = np.nonzero(
            hp_im > 4. * np.nanstd(utils.stats.sigmacut(
                hp_im)))
        
        logging.info('{} detected pixels'.format(len(detected_pixels[0])))
        
        # pixels aggregation in sources
        logging.info('aggregating detected pixels')
        sources = aggregate(detected_pixels, SOURCE_SIZE)
            
        
        logging.info('{} sources detected'.format(len(sources)))
        
        star_list_file = utils.io.open_file(self._get_star_list_path())
        for isource in sources:
            star_list_file.write('{} {}\n'.format(
                isource[0]*binning, isource[1]*binning))
            
        self.reset_star_list(sources)
        
        return self._get_star_list_path(), self.fwhm_arc


