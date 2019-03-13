#!/usr/bin/env python
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: image.py

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
The Image module is aimed to manage image class
"""

import numpy as np
import logging
import time
import warnings
import os

import astropy.wcs as pywcs
import astropy.stats
import astropy.nddata
from  astropy.coordinates.name_resolve import NameResolveError
from astropy.io.fits.verify import VerifyWarning, VerifyError, AstropyUserWarning

import photutils

from scipy import optimize, interpolate, signal
import pandas

import core
import cutils

import utils.astrometry
import utils.image
import utils.stats
import utils.vector
import utils.web
import utils.misc
import utils.io


#################################################
#### CLASS Frame2D ##############################
#################################################
class Frame2D(core.WCSData):

    def __init__(self, *args, **kwargs):
        
        core.WCSData.__init__(self, *args, **kwargs)

        # checking
        if self.data.ndim != 2:
            raise TypeError('input image has {} dims but must have exactly 2 dimensions'.format(self.data.ndim))

    def get_stats(self, fast=True):
        """Return image stats
        
        :param fast: If fast, only a random fraction of the image is
          used to make stats.

        :return: mean, median, std
        """
        FAST_FRAC = 0.02

        if fast:
            pix = np.random.randint(0, high=self.data.size,
                                    size=int(FAST_FRAC*self.data.size))
            _data = np.copy(self.data).flatten()[pix]
        else:
            _data = self.data
            
        return astropy.stats.sigma_clipped_stats(_data, sigma=3.0)

    def crop(self, cx, cy, size):
        """Return a portion of the image as another Frame2D instance.

        :param cx: X center position

        :param cy: Y center position

        :param size: Size of the cropped rectangle. A tuple (sz,
          sy). Can be single int in which case the cropped data is a
          box.

        .. warning:: size of the returned box is not guaranteed if cx
          and cy are on the border of the image.

        """
        size = np.array(size)
        if size.size == 2:
            size = size[::-1]
        elif size.size == 1:
            size = [size, size]
        else:
            raise TypeError('size must be a single number or a tuple (sx, sy)')
        
        cutout = astropy.nddata.Cutout2D(
            self.data.T, position=[cx, cy],
            size=size, wcs=self.get_wcs())
        
        newim = self.copy(data=cutout.data.T)
        newim.update_params(cutout.wcs.to_header())
        ((ymin, ymax), (xmin, xmax)) = cutout.bbox_original
        newim.params['cropped_bbox'] = (xmin, xmax+1, ymin, ymax+1)
        return newim
        

#################################################
#### CLASS Image ################################
#################################################

class Image(Frame2D):

    BOX_SIZE_COEFF = 7
    FIT_TOL = 1e-2
    REDUCED_CHISQ_LIMIT = 1.5
    DETECT_INDEX = 0
    
    profiles = ['moffat', 'gaussian']
    
    def __init__(self, data, **kwargs):
        
        Frame2D.__init__(self, data, **kwargs)

        if 'box_size_coeff' in self.params:
            self.box_size_coeff = self.params.box_size_coeff
        else:
            self.box_size_coeff = self.BOX_SIZE_COEFF

        if 'profile_name' not in self.params:
            self.params['profile_name'] = self.config.PSF_PROFILE

        if self.params.profile_name == 'moffat':
            self.box_size_coeff /= 3.
                    
        # define profile
        self.reset_profile_name(self.params.profile_name)

        self.reduced_chi_square_limit = self.REDUCED_CHISQ_LIMIT

        
        if 'detect_stack' in self.params:
            self.detect_stack = self.params.detect_stack
        else:
            self.detect_stack = self.config.DETECT_STACK

        if 'moffat_beta' in self.params:
            self.default_beta = self.params.moffat_beta
        else:
            self.default_beta = self.config.MOFFAT_BETA
        
        self.fit_tol = self.FIT_TOL

        # define astrometry parameters        
        if 'fwhm_arc' in self.params:
            self.fwhm_arc = self.params.fwhm_arc
        else:
            self.fwhm_arc = self.config.INIT_FWHM

        if 'scale' in self.params:
            self.scale = self.params.scale
        else:
            self.scale = self.config.FIELD_OF_VIEW_1 / self.config.CAM1_DETECTOR_SIZE_X * 60.
            
        self.reset_scale(self.scale)

        self.target_ra = self.params.target_ra
        self.target_dec = self.params.target_dec
        self.target_x = self.params.target_x
        self.target_y = self.params.target_y
        self.wcs_rotation = self.params.wcs_rotation

    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results.hdf5"

    def _get_guess_matrix_path(self):
        """Return path to the guess matrix"""
        return self._data_path_hdr + "guess_matrix.fits"

    def reset_scale(self, scale):
        """Reset scale attribute.
        
        :param scale: Frame scale in arcsec/pixel
        """
        self.scale = float(scale)
        self.fov = self.dimx * self.scale / 60.
        self.reset_fwhm_arc(self.fwhm_arc)

    def reset_fwhm_arc(self, fwhm_arc):
        """Reset FWHM of stars in arcsec

        :param fwhm_arc: FWHM of stars in arcsec
        """
        self.fwhm_arc = float(fwhm_arc)
        self.reset_fwhm_pix(self.arc2pix(self.fwhm_arc))

    def reset_fwhm_pix(self, fwhm_pix):
        """Reset FWHM of stars in pixels

        :param fwhm_arc: FWHM of stars in pixels
        """
        self.fwhm_pix = float(fwhm_pix)
        self.reset_box_size()

    def reset_box_size(self):
        """Reset box size attribute. Useful if FWHM or scale has been
        modified after class init.
        """
        self.box_size = int(np.ceil(self.box_size_coeff *  self.fwhm_pix))
        self.box_size += int(~self.box_size%2) # make it odd

    
    def copy(self, data=None):
        """Return a copy of the instance

        :param data: (Optional) can be used to change data
        """
        return Frame2D.copy(self, data=data, instrument=self.instrument,
                            config=self.config, data_prefix=self._data_prefix,
                            sip=self.sip)
    

    def detrend(self, bias=None, dark=None, flat=None, shift=None, cr_map=None):
        """Return a detrended image

        :param bias: Master bias

        :param dark: Master dark (must be in counts/s)

        :param flat: Master flat

        :param shift: shift correction (dx, dy)

        :param cr_map: A map of cosmic rays with boolean type: 1 = CR, 0 = NO_CR
        """
        frame = np.copy(self.data)
        
        if bias is not None:
            if bias.shape != self.shape: raise TypeError('Bias must have shape {}'.format(self.shape))
            frame -= bias
            
        if dark is not None:
            if dark.shape != self.shape: raise TypeError('Dark must have shape {}'.format(self.shape))
            frame -= dark * self.params.exposure_time
            
        if flat is not None:
            if flat.shape != self.shape: raise TypeError('Flat must have shape {}'.format(self.shape))
            flat = np.copy(flat)
            flat[np.nonzero(flat == 0)] = np.nan
            flat /= np.nanpercentile(flat, 99.9)
            frame /= flat
                
        if cr_map is not None:
            if cr_map.shape != self.shape:
                raise TypeError('cr_map must have shape {}'.format(self.shape))
    
            if cr_map.dtype != np.bool: raise TypeError('cr_map must have type bool')

            frame = utils.image.correct_cosmic_rays(frame, cr_map)
            
        if shift is not None:
            utils.validate.has_len(shift, 2, object_name='shift')
            dx, dy = shift
            if (dx != 0.) and (dy != 0.):
                frame = utils.image.shift_frame(frame, dx, dy, 
                                                0, self.dimx, 
                                                0, self.dimy, 1)
            
        return frame
        
            

    def find_object(self, is_standard=False, return_radec=False):
        """Try to find the object given the name in the header

        :param is_standard: if object is a standard, do not try to
          resolve the name.

        :param return_radec: if true, radec coordinates are retruned
          instead of image coordinates (pixel position).
        """
        object_found = False
        if not is_standard:
            logging.info('resolving coordinates of {}'.format(self.params.OBJECT))
            try:
                print astropy.coordinates.get_icrs_coordinates(self.params.OBJECT)
                object_found = True
            except NameResolveError:
                logging.debug('object name could not be resolved')
                
        logging.info('looking in the standard table for {}'.format(self.params.OBJECT))
        try:
            std_name = ''.join(self.params.OBJECT.strip().split()).upper()
            std_ra, std_dec, std_pm_ra, std_pm_dec = self._get_standard_radec(
                std_name, return_pm=True)
            object_found = True
        except StandardError:
            warnings.warn('object name not found in the standard table')

        if not object_found:
            raise StandardError('object coordinates could not be resolved')

        std_yr_obs = float(self.params['DATE-OBS'].split('-')[0])
        pm_orig_yr = 2000 # radec are considered to be J2000
        # compute ra/dec with proper motion
        std_ra, std_dec = utils.astrometry.compute_radec_pm(
            std_ra, std_dec, std_pm_ra, std_pm_dec,
            std_yr_obs - pm_orig_yr)
        std_ra_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *utils.astrometry.deg2ra(std_ra))
        std_dec_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *utils.astrometry.deg2dec(std_dec))
        logging.info('Object {} RA/DEC: {} ({:.3f}) {} ({:.3f}) (corrected for proper motion)'.format(self.params.OBJECT, std_ra_str, std_ra, std_dec_str, std_dec))
        if not return_radec:
            return np.squeeze(self.world2pix([std_ra, std_dec]))
        else:
            return [std_ra, std_dec]
            

    

    def load_fit_results(self, fit_results_path=None):
        """Load a file containing the fit results"""
        if fit_results_path is None:
            fit_results_path = self._get_fit_results_path()
        self.fit_results.load_stars_params(fit_results_path)
    
    def reset_profile_name(self, profile_name):
        """Reset the name of the profile used.

        :param profile_name: Name of the PSF profile to use for
          fitting. Can be 'moffat' or 'gaussian'.
        """
        if profile_name in self.profiles:
            self.profile_name = profile_name
            self.profile = utils.astrometry.get_profile(self.profile_name)
        else:
            raise StandardError(
                "Bad profile name (%s) please choose it in: %s"%(
                    profile_name, str(self.profiles)))
        

    def detect_stars(self, min_star_number=30, path=None):
        """Detect star positions in data.

        :param min_star_number: Minimum number of stars to
          detect. Must be greater or equal to 4 (minimum star number
          for any alignment process). this is also the number of stars
          returned (sorted by intensity)

        :param path: (Optional) Path to the output star list file. If
          None, nothing is written.

        """
        DETECT_THRESHOLD = 5
        MAX_ROUNDNESS = 0.1
        FWHM_STARS_NB = 30
        
        mean, median, std = self.get_stats()
        daofind = photutils.DAOStarFinder(fwhm=self.fwhm_pix,
                                          threshold=DETECT_THRESHOLD * std)
        sources = daofind(self.data.T - median).to_pandas()
        if len(sources) == 0: raise StandardError('no star detected, check input image')
        # this 0.45 on the saturation threshold ensures that stars at ZPD won't saturate
        saturation_threshold = np.nanmax(self.data) * 0.45 
        sources = sources[sources.peak < saturation_threshold]
        # filter by roundness
        sources = sources[np.abs(sources.roundness2) < MAX_ROUNDNESS]
        # filter by radius from center to avoid using distorded stars
        sources['radius'] = np.sqrt((sources.xcentroid - self.dimx/2.)**2
                                    + (sources.ycentroid - self.dimy/2.)**2)
        sources = sources[sources.radius < min(self.dimx, self.dimy) / 2.]
        sources = sources.sort_values(by=['flux'], ascending=False)
        logging.info("%d stars detected" %(len(sources)))
        sources = sources[:min_star_number]
        logging.info("star list reduced to %d stars" %(len(sources)))
        sources = utils.astrometry.df2list(sources)
        sources = self.fit_stars(sources, no_aperture_photometry=True)
        mean_fwhm, mean_fwhm_err = self.detect_fwhm(sources[:FWHM_STARS_NB])

        if path is not None:         
            utils.io.open_file(path, 'w') # used to create the folder tree
            sources.to_hdf(path, 'data', mode='w')
            logging.info('sources written to {}'.format(path))
        return sources, mean_fwhm

    def detect_fwhm(self, star_list):
        """Return fwhm of a list of stars

        :param star_list: list of stars (can be an np.ndarray or a path
          to a star list).
        """
        star_list = utils.astrometry.load_star_list(star_list)
        
        mean_fwhm, mean_fwhm_err = utils.astrometry.detect_fwhm_in_frame(
            self.data, star_list,
            self.fwhm_pix)
        
        logging.info("Detected stars FWHM : {:.2f}({:.2f}) pixels, {:.2f}({:.2f}) arc-seconds".format(mean_fwhm[0], mean_fwhm_err[0], self.pix2arc(mean_fwhm[0]), self.pix2arc(mean_fwhm_err[0])))

        self.reset_fwhm_pix(mean_fwhm[0])
        return mean_fwhm[0], mean_fwhm_err[0]

    def aperture_photometry(self, star_list, aper_coeff=3., silent=False):
        """Perform aperture photometry.

        Based on photutils.

        :param star_list: May be an array (n, 2) or a path to a star
        list file created with detect_stars()

        :param aper_coeff: (Optional) Aperture coefficient (default
          3.) Aperture = aper_coeff * fwhm.
        """
        C_AP = aper_coeff # Aperture coefficient
        C_IN = C_AP + 1. # Inner radius coefficient of the bckg annulus
        MIN_BACK_COEFF = 5. # Minimum percentage of the pixels in the
                            # annulus to estimate the background
        C_OUT = np.sqrt((MIN_BACK_COEFF * 1.5 * C_AP**2.) + C_IN**2.)

        star_list = utils.astrometry.load_star_list(star_list)
        aper = photutils.CircularAperture(star_list,
                                          r=C_AP * self.fwhm_pix)
        aper_ann = photutils.CircularAnnulus(star_list,
                                             r_in=C_IN * self.fwhm_pix,
                                             r_out=C_OUT * self.fwhm_pix)

        ann_masks = aper_ann.to_mask(method='center')
        ann_medians = list()
        for imask in ann_masks:
            ivalues = imask.multiply(self.data)            
            ivalues = ivalues[ivalues > 0]
            if ivalues.size > aper.area() * MIN_BACK_COEFF:
                ann_medians.append(utils.astrometry.sky_background_level(
                    ivalues, return_error=True))
            else:
                ann_medians.append((np.nan, np.nan))
        
        ann_medians = np.array(ann_medians)
        phot_table = photutils.aperture_photometry(self.data, aper,
                                                   method='subpixel',
                                                   subpixels=10)
        phot_table['background'] = ann_medians[:,0] * aper.area()
        phot_table['photometry'] = phot_table['aperture_sum'] - phot_table['background']
        phot_table['background_err'] = ann_medians[:,1] * aper.area()
        phot_table['photometry_err'] = np.sqrt(np.abs(phot_table['photometry'])) + phot_table['background_err']
        if 'EXPTIME' in self.params:
            phot_table['flux'] = phot_table['photometry'] / self.params.EXPTIME
            phot_table['flux_err'] = phot_table['photometry_err'] / self.params.EXPTIME
        return phot_table
    
    def register(self, max_stars_detect=60,
                 return_fit_params=False, rscale_coeff=1.,
                 compute_precision=True, compute_distortion=False,
                 return_error_maps=False,
                 return_error_spl=False):
        """Register data and return a corrected pywcs.WCS
        object.

        Optionally (if return_error_maps set to True or
        return_error_spl set to True) 2 distortion maps used to refine
        a calculated SIP distortion model are returned.
        
        Precise RA/DEC positions of the stars in the field are
        recorded from a catalog of the VIZIER server.

        Using the real position of the same stars in the frame, WCS
        transformation parameters are optimized va a SIP model.
        
        :param max_stars_detect: (Optional) Number of detected stars
          in the frame for the initial wcs parameters (default 60).
          
        :param return_fit_params: (Optional) If True return final fit
          parameters instead of wcs (default False).

        :param rscale_coeff: (Optional) Coefficient on the maximum
          radius of the fitted stars to compute scale. When rscale_coeff
          = 1, rmax is half the longest side of the image (default 1).

        :param compute_distortion: (Optional) If True, optical
          distortion (SIP) are computed. Note that a frame with a lot
          of stars is better for this purpose (default False).

        :param compute_precision: (Optional) If True, astrometrical
          precision is computed (default True).

        :param return_error_maps: (Optional) If True, error maps
          (200x200 pixels) on the registration are returned (default
          False).

        :param return_error_spl: (Optional) If True, error maps on the
          registration are returned as
          scipy.interpolate.RectBivariateSpline instances (default
          False).
        """
        def get_transformation_error(guess, deg_list, fit_list,
                                     target_ra, target_dec):
            _wcs = pywcs.WCS(naxis=2)
            _wcs.wcs.crpix = [guess[1], guess[2]]
            _wcs.wcs.cdelt = [-guess[3], guess[4]]
            _wcs.wcs.crval = [target_ra, target_dec]
            _wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            _wcs.wcs.crota = [guess[0], guess[0]]
            
            trans_list = list()
            for istar in deg_list:
                pos = _wcs.wcs_world2pix(istar[0], istar[1], 0)
                trans_list.append((pos[0], pos[1]))

            result = np.array(np.array(trans_list) - np.array(fit_list),
                              dtype=float).flatten()
            return result[np.nonzero(~np.isnan(result))]

        def radius_filter(star_list, rmax, borders=None):
            star_list = np.array(star_list)
            star_list = [[star_list[i,0], star_list[i,1]]
                         for i in range(star_list.shape[0])]
            final_star_list = list()
            for istar in star_list:
                posx = istar[0] ; posy = istar[1]
                r = np.sqrt((posx - self.dimx/2.)**2.
                              + (posy - self.dimy/2)**2.)
                if r <= rmax:
                    if borders is None:
                        final_star_list.append((posx, posy))
                    else:
                        if (posx > borders[0] and posx < borders[1]
                            and posy > borders[2] and posy < borders[3]):
                            final_star_list.append((posx, posy))
                        else:
                            final_star_list.append((np.nan, np.nan))
                else:
                    final_star_list.append((np.nan, np.nan))
            return np.array(final_star_list)

        def world2pix(wcs, star_list):
            star_list = np.array(star_list)
            return np.array(wcs.all_world2pix(
                star_list[:,0],
                star_list[:,1], 0, quiet=True)).T

        
        def get_filtered_params(fit_params, snr_min=None,
                                dist_min=1e9,
                                param='star_list',
                                return_index=False):

            if param == 'star_list':
                param_list = utils.astrometry.df2list(fit_params)
            else:
                param_list = fit_params[param]
            snr = fit_params['snr']
            
            if snr_min is None:
                snr_min = max(utils.stats.robust_median(snr), 3.)
            
            if return_index:
                index = np.zeros(param_list.shape[0])
                
            param_list_f = list()
            for istar in range(param_list.shape[0]):
                if not np.isnan(snr[istar]):
                    dist = np.sqrt(fit_params['dx'][istar]**2
                                   + fit_params['dy'][istar]**2)
                    if snr[istar] > snr_min and dist < dist_min:
                        if param == 'star_list':
                            param_list_f.append(param_list[istar,:])
                        else:
                            param_list_f.append(param_list[istar])
                        if return_index:
                            index[istar] = 1
            if not return_index:
                return np.array(param_list_f)
            else:
                return np.array(param_list_f), index
        
      
        MIN_STAR_NB = 4 # Minimum number of stars to get a correct WCS

        XYRANGE_STEP_NB = 20 # Define the number of steps for the
                             # brute force guess
        XY_HIST_BINS = 200 # Define the number of steps for the
                           # histogram registration
                           
        # warning: too much steps is not good. a good value is 40
        # steps for 12 degrees (i.e. 20 steps for 6 degrees etc.).
        ANGLE_STEPS = 40 
        ANGLE_RANGE = 12
        ZOOM_RANGE_COEFF = 0.015

        if not (self.target_ra is not None and self.target_dec is not None
                and self.target_x is not None and self.target_y is not None
                and self.wcs_rotation is not None):
            raise StandardError("Not enough parameters to register data. Please set target_xy, target_radec and wcs_rotation parameters at Astrometry init")

        if return_error_maps and return_error_spl: raise StandardError('return_error_maps and return_error_spl cannot be both set to True, choose one of them')
        
        logging.info('Computing WCS')

        logging.info("Initial scale: {} arcsec/pixel".format(self.scale))
        logging.info("Initial rotation: {} degrees".format(self.wcs_rotation))
        logging.info("Initial target position in the image (X, Y): {} {}".format(
            self.target_x, self.target_y))
        logging.info("Initial target position in the image (RA, DEC): {} {}".format(
            self.target_ra, self.target_dec))
        deltax = self.scale / 3600. # arcdeg per pixel
        deltay = float(deltax)

        # get FWHM
        star_list_fit_init_path, fwhm_arc = self.detect_stars(
            min_star_number=max_stars_detect)
        star_list_fit_init = utils.astrometry.load_star_list(
            star_list_fit_init_path, remove_nans=True)
        ## star_list_fit_init = self.load_star_list(
        ##     './temp/data.Astrometry.star_list)'
        ## fwhm_arc= 1.
        
        self.box_size_coeff = 5.
        self.reset_fwhm_arc(fwhm_arc)

        # clean deep frame by keeping only the pixels around the
        # detected stars to avoid strong saturated stars.
        deep_frame_corr = np.empty_like(self.data)
        deep_frame_corr.fill(np.nan)
        for istar in range(star_list_fit_init.shape[0]):
            x_min, x_max, y_min, y_max = utils.image.get_box_coords(
                star_list_fit_init[istar, 0],
                star_list_fit_init[istar, 1],
                self.fwhm_pix*7,
                0, self.dimx,
                0, self.dimy)
            deep_frame_corr[x_min:x_max,
                            y_min:y_max] = self.data[x_min:x_max,
                                                     y_min:y_max]
        
        # Query to get reference star positions in degrees
        star_list_query = self.query_vizier(max_stars=100 * max_stars_detect)
        ## utils.io.write_fits('star_list_query.fits', star_list_query, overwrite=True)
        ## star_list_query = utils.io.read_fits('star_list_query.fits')
        
        if len(star_list_query) < MIN_STAR_NB:
            raise StandardError("Not enough stars found in the field (%d < %d)"%(len(star_list_query), MIN_STAR_NB))
            
        # reference star position list in degrees
        star_list_deg = star_list_query[:max_stars_detect*20]
        
        ## Define a basic WCS        
        wcs = utils.astrometry.create_wcs(
            self.target_x, self.target_y,
            deltax, deltay, self.target_ra, self.target_dec,
            self.wcs_rotation, sip=self.sip)
       
        # Compute initial star positions from initial transformation
        # parameters
        rmax = max(self.dimx, self.dimy) / np.sqrt(2)
        star_list_pix = radius_filter(
            world2pix(wcs, star_list_deg), rmax)

        ## Plot star lists #####
        ## import pylab as pl
        ## pl.imshow(
        ##     deep_frame.T,
        ##     vmin=cutils.part_value(deep_frame.flatten(), 0.02),
        ##     vmax=cutils.part_value(deep_frame.flatten(), 0.995),
        ##     cmap=pl.gray())
        ## pl.scatter(star_list_fit_init[:,0], star_list_fit_init[:,1])
        ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1], c='red')
        ## pl.show()
        ## quit()

        # fast histogram determination of the inital parameters
        max_list = list()
        for iangle in np.linspace(-ANGLE_RANGE/2., ANGLE_RANGE/2., ANGLE_STEPS):
            iwcs = utils.astrometry.create_wcs(
                self.target_x, self.target_y,
                deltax, deltay, self.target_ra, self.target_dec,
                self.wcs_rotation + iangle, sip=self.sip)
            istar_list_pix = radius_filter(
                world2pix(iwcs, star_list_deg), rmax)

            max_corr, max_dx, max_dy = utils.astrometry.histogram_registration(
                star_list_fit_init, istar_list_pix,
                self.dimx, self.dimy, XY_HIST_BINS)
            
            max_list.append((max_corr, iangle, max_dx, max_dy))
            logging.info('histogram check: correlation level {}, angle {}, dx {}, dy {}'.format(
                *max_list[-1]))
        max_list = sorted(max_list, key = lambda imax: imax[0], reverse=True)
        max_list = np.array(max_list)
        if np.max(max_list[:,0]) < 2 * np.median(max_list[:,0]):
            raise StandardError('maximum correlation is not high enough, check target_ra, target_dec')
        
        self.target_x += max_list[0, 2]
        self.target_y += max_list[0, 3]
        self.wcs_rotation = max_list[0, 1]

        # update wcs
        wcs = utils.astrometry.create_wcs(
            self.target_x, self.target_y,
            deltax, deltay, self.target_ra, self.target_dec,
            self.wcs_rotation, sip=self.sip)

        star_list_pix = radius_filter(
            world2pix(wcs, star_list_deg), rmax)

        logging.info(
            "Histogram guess of the parameters:\n"
            + "> Rotation angle [in degree]: {:.3f}\n".format(self.wcs_rotation)
            + "> Target position [in pixel]: ({:.3f}, {:.3f})\n".format(
                self.target_x, self.target_y))

        ## brute force guess ####
        x_range_len = max(self.dimx, self.dimy) / float(XY_HIST_BINS) * 4
        y_range_len = x_range_len
        r_range_len = ANGLE_RANGE / float(ANGLE_STEPS) * 8
        x_range = np.linspace(-x_range_len/2, x_range_len/2,
                              XYRANGE_STEP_NB)
        y_range = np.linspace(-y_range_len/2, y_range_len/2,
                              XYRANGE_STEP_NB)
        r_range = np.linspace(-r_range_len, r_range_len,
                              ANGLE_STEPS)

        dx, dy, dr, guess_matrix = utils.astrometry.brute_force_guess(
            deep_frame_corr,
            star_list_deg, x_range, y_range, r_range,
            None, 1., self.fwhm_pix * 3., init_wcs=wcs)
            
        # refined brute force guess
        x_range_len = max(np.diff(x_range)[0] * 4, self.fwhm_pix * 3) # 3 FWHM min
        y_range_len = x_range_len
        finer_angle_range = np.diff(r_range)[0] * 4.
        finer_xy_step = min(XYRANGE_STEP_NB / 4,
                            int(x_range_len) + 1) # avoid xystep < 1 pixel

        
        x_range = np.linspace(dx-x_range_len/2, dx+x_range_len/2,
                              finer_xy_step)
        y_range = np.linspace(dy-y_range_len/2, dy+y_range_len/2,
                              finer_xy_step)
        r_range = np.linspace(dr-finer_angle_range/2., dr+finer_angle_range/2.,
                              ANGLE_STEPS * 2)

        zoom_range = np.linspace(1.-ZOOM_RANGE_COEFF/2.,
                                 1.+ZOOM_RANGE_COEFF/2., 20)
        zoom_guesses = list()
        for izoom in zoom_range:
            dx, dy, dr, guess_matrix = utils.astrometry.brute_force_guess(
                self.data,
                star_list_deg, x_range, y_range, r_range,
                None, izoom, self.fwhm_pix * 3.,
                verbose=False, init_wcs=wcs, raise_border_error=False)

            zoom_guesses.append((izoom, dx, dy, dr, np.nanmax(guess_matrix)))
            logging.info('Checking with zoom {}: dx={}, dy={}, dr={}, score={}'.format(*zoom_guesses[-1]))

        # sort brute force guesses to get the best one
        best_guess = sorted(zoom_guesses, key=lambda zoomp: zoomp[4])[-1]

        self.wcs_rotation -= best_guess[3]
        self.target_x -= best_guess[1]
        self.target_y -= best_guess[2]
    
        deltax *= best_guess[0]
        deltay *= best_guess[0]
        
        logging.info(
            "Brute force guess of the parameters:\n"
            + "> Rotation angle [in degree]: {:.3f}\n".format(self.wcs_rotation)
            + "> Target position [in pixel]: ({:.3f}, {:.3f})\n".format(
                self.target_x, self.target_y)
            + "> Scale X (arcsec/pixel): {:.5f}\n".format(
                deltax * 3600.)
            + "> Scale Y (arcsec/pixel): {:.5f}".format(
                deltay * 3600.))

        # update wcs
        wcs = utils.astrometry.create_wcs(
            self.target_x, self.target_y,
            deltax, deltay, self.target_ra, self.target_dec,
            self.wcs_rotation, sip=self.sip)
        
        ############################
        ### plot stars positions ###
        ############################
        ## import pylab as pl
        ## im = pl.imshow(deep_frame.T,
        ##                vmin=np.nanmedian(deep_frame),
        ##                vmax=np.nanmedian(deep_frame)+50)
        ## im.set_cmap('gray')
        ## star_list_pix = radius_filter(
        ##     world2pix(wcs, star_list_deg), rmax)
        ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1],
        ##            edgecolor='blue', linewidth=2., alpha=1.,
        ##            facecolor=(0,0,0,0))
        ## pl.show()
                    

        ## COMPUTE SIP
        if compute_distortion:
            logging.info('Computing SIP coefficients')
            if self.sip is None:
                warnings.warn('As no prior SIP has been given, this initial SIP is computed over the field inner circle. To cover the whole field the result of this registration must be passed at the definition of the class')
                r_coeff = 0.5
            else:
                r_coeff = 1./np.sqrt(2)
                
            # compute optical distortion with a greater list of stars
            rmax = max(self.dimx, self.dimy) * r_coeff

            star_list_pix = radius_filter(
                world2pix(wcs, star_list_query), rmax,
                borders=[0, self.dimx, 0, self.dimy])
            
            fit_params = self.fit_stars(
                local_background=True,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True)
            
            ## SNR and DIST filter
            star_list_fit, index = get_filtered_params(
                fit_params, param='star_list', dist_min=15.,
                return_index=True)
            
            star_list_pix = star_list_pix[np.nonzero(index)]
            
            ############################
            ### plot stars positions ###
            ############################
            ## import pylab as pl
            ## pl.imshow(self.data.T, vmin=30, vmax=279, cmap='gray',
            ##           interpolation='None')
            ## pl.scatter(star_list_fit[:,0], star_list_fit[:,1],
            ##            edgecolor='blue', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))

            ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1],
            ##            edgecolor='red', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            ## pl.show()
            
            wcs = self.fit_sip(star_list_pix,
                               star_list_fit,
                               params=None, init_sip=wcs,
                               err=None, sip_order=4)

            ## star_list_pix = wcs.all_world2pix(star_list_query[:,:2], 0)
            ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1],
            ##            edgecolor='green', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            
        # computing distortion maps
        logging.info('Computing distortion maps')
        r_coeff = 1./np.sqrt(2)
        rmax = max(self.dimx, self.dimy) * r_coeff

        star_list_pix = radius_filter(
            world2pix(wcs, star_list_query), rmax, borders=[
                0, self.dimx, 0, self.dimy])

        # fit based on SIP corrected parameters
        fit_params = self.fit_stars(
            star_list_pix,
            local_background=False,
            multi_fit=False, fix_fwhm=True,
            no_aperture_photometry=True)

        _x = fit_params['x'].values
        _y = fit_params['y'].values
        _r = np.sqrt((_x - self.dimx / 2.)**2.
                     + (_y - self.dimy / 2.)**2.)

        _dx = fit_params['dx'].values
        _dy = fit_params['dy'].values

        # filtering badly fitted stars (jumping stars)
        ## _x[np.nonzero(np.abs(_dx) > 5.)] = np.nan
        ## _x[np.nonzero(np.abs(_dy) > 5.)] = np.nan
        _x[np.nonzero(_r > rmax)] = np.nan

        # avoids duplicate of the same star (singular matrix error
        # with RBF)
        for ix in range(_x.size):
            if np.nansum(_x == _x[ix]) > 1:
                 _x[ix] = np.nan

        nonans = np.nonzero(~np.isnan(_x))
        _w = 1./fit_params['x_err'].values[nonans]
        _x = fit_params['x'].values[nonans]
        _y = fit_params['y'].values[nonans]
        _dx = fit_params['dx'].values[nonans]
        _dy = fit_params['dy'].values[nonans]

        dxrbf = interpolate.Rbf(_x, _y, _dx, epsilon=1, function='linear')
        dyrbf = interpolate.Rbf(_x, _y, _dy, epsilon=1, function='linear')

        # RBF models are converted to pixel maps and fitted with
        # Zernike polynomials
        X, Y = np.mgrid[:self.dimx:200j,:self.dimy:200j]
        dxmap = dxrbf(X, Y)
        dymap = dyrbf(X, Y)

        dxmap_fit, dxmap_res, fit_error = utils.image.fit_map_zernike(
            dxmap, np.ones_like(dxmap), 20)
        dymap_fit, dymap_res, fit_error = utils.image.fit_map_zernike(
            dymap, np.ones_like(dymap), 20)

        # error maps are converted to a RectBivariateSpline instance
        dxspl = interpolate.RectBivariateSpline(
            np.linspace(0, self.dimx, dxmap.shape[0]),
            np.linspace(0, self.dimy, dxmap.shape[1]),
            dxmap_fit, kx=3, ky=3)
        
        dyspl = interpolate.RectBivariateSpline(
            np.linspace(0, self.dimx, dymap.shape[0]),
            np.linspace(0, self.dimy, dymap.shape[1]),
            dymap_fit, kx=3, ky=3)

        ## import pylab as pl
        ## pl.figure(1)
        ## pl.imshow(dxmap.T, interpolation='none')
        ## pl.colorbar()
        ## pl.scatter(_x/10., _y/10.,
        ##            edgecolor='red', linewidth=2., alpha=1.,
        ##            facecolor=(0,0,0,0))
        ## pl.figure(2)
        ## pl.imshow(dymap.T, interpolation='none')
        ## pl.colorbar()            
        ## pl.scatter(_x/10., _y/10.,
        ##            edgecolor='red', linewidth=2., alpha=1.,
        ##            facecolor=(0,0,0,0))
        ## pl.show()

            
        ## COMPUTE PRECISION
        if compute_precision:
            logging.info('Computing astrometrical precision')

            rmax = max(self.dimx, self.dimy) / 2.

            # compute astrometrical precision with a greater list of stars
            star_list_pix = radius_filter(
                world2pix(wcs, star_list_query), rmax, borders=[
                    0, self.dimx, 0, self.dimy])

            ## for checking purpose: ############################
            ## # refine position with calculated dxmap and dymap
            ## if dxspl is not None:
            ##     star_list_pix_old = np.copy(star_list_pix)
            ##     star_list_pix[:,0] += dxspl.ev(star_list_pix_old[:,0],
            ##                                    star_list_pix_old[:,1])
            ##     star_list_pix[:,1] += dyspl.ev(star_list_pix_old[:,0],
            ##                                    star_list_pix_old[:,1])
                

            fit_params = self.fit_stars(
                star_list_pix,
                local_background=False,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True)
            
            ############################
            ### plot stars positions ###
            ############################
            ## import pylab as pl
            ## pl.imshow(self.data.T, vmin=30, vmax=279, cmap='gray',
            ##           interpolation='None')
            ## star_list_fit, index = get_filtered_params(
            ##     fit_params, param='star_list', dist_min=15.,
            ##     return_index=True)
            ## star_list_pix = star_list_pix[np.nonzero(index)]
            ## pl.scatter(star_list_fit[:,0], star_list_fit[:,1],
            ##            edgecolor='blue', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1],
            ##            edgecolor='red', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            ## pl.scatter(star_list_pix_old[:,0], star_list_pix_old[:,1],
            ##            edgecolor='green', linewidth=2., alpha=1.,
            ##            facecolor=(0,0,0,0))
            ## pl.show()

            # results must be filtered for 'jumping' stars
            # when fitted independantly the most brilliant stars in
            # the fit box gets fitted instead of the star at the
            # center of it. 
            dx = get_filtered_params(
                fit_params, param='dx',
                dist_min=self.fwhm_pix*2.)
            dy = get_filtered_params(
                fit_params, param='dy',
                dist_min=self.fwhm_pix*2.)
            x_err = get_filtered_params(
                fit_params, param='x_err',
                dist_min=self.fwhm_pix*2.)
            y_err = get_filtered_params(
                fit_params, param='y_err',
                dist_min=self.fwhm_pix*2.)

            precision = np.sqrt(dx**2. + dy**2.)
            precision_mean = np.sqrt(np.nanmedian(np.abs(dx))**2.
                                     + np.nanmedian(np.abs(dy))**2.)
            precision_mean_err = np.sqrt(
                (np.nanpercentile(dx, 84) - precision_mean)**2.
                 + (np.nanpercentile(dy, 84) - precision_mean)**2.)

            logging.info(
                "Astrometrical precision [in arcsec]: {:.3f} [+/-{:.3f}] computed over {} stars".format(
                    precision_mean * deltax * 3600.,
                    precision_mean_err * deltax * 3600., np.size(dx)))

            ### PLOT ERROR ON STAR POSITIONS ###
            ## import pylab as pl
            ## pl.errorbar(dx * deltax * 3600.,
            ##             dy * deltay * 3600.,
            ##             xerr= x_err * deltax * 3600.,
            ##             yerr= y_err * deltay * 3600.,
            ##             linestyle='None')
            ## circ = pl.Circle([0.,0.], radius=fwhm_arc/2.,
            ##                  fill=False, color='g', linewidth=2.)
            ## pl.gca().add_patch(circ)
            ## pl.axes().set_aspect('equal')
            ## pl.grid()
            ## pl.xlim([-fwhm_arc/2.,fwhm_arc/2.])
            ## pl.ylim([-fwhm_arc/2.,fwhm_arc/2.])
            ## pl.show()
       
        logging.info(
            "Optimization parameters:\n"
            + "> Rotation angle [in degree]: {:.3f}\n".format(self.wcs_rotation)
            + "> Target position [in pixel]: ({:.3f}, {:.3f})\n".format(
                self.target_x, self.target_y)
            + "> Scale X (arcsec/pixel): {:.5f}\n".format(
                deltax * 3600.)
            + "> Scale Y (arcsec/pixel): {:.5f}".format(
                deltay * 3600.))

        self.reset_scale(np.mean((deltax, deltay)) * 3600.)
        
        logging.info('corrected WCS computed')
        self.set_wcs(wcs)
        logging.info('internal WCS updated (to update the file use self.writeto function)')
        logging.info(str(self.get_wcs()))
        
        if not return_fit_params:
            if return_error_maps:
                return wcs, dxmap_fit, dymap_fit
            elif return_error_spl:
                return wcs, dxspl, dyspl
            else:
                return wcs
        else:
            return fit_params


    def fit_sip(self, star_list1, star_list2, params=None, init_sip=None,
                err=None, sip_order=2, crpix=None, crval=None):
        """FIT the distortion correction polynomial to match two lists
        of stars (the list of stars 2 is distorded to match the list
        of stars 1).

        :param star_list1: list of stars 1
        
        :param star_list2: list of stars 2
        
        :param params: (Optional) Transformation parameter to go from
          the list of stars 1 to the list of stars 2. Must be a tuple
          [dx, dy, dr, da, db, rcx, rcy, zoom_factor] (default None).

        :param init_sip: (Optional) Initial SIP (an astropy.wcs.WCS object,
          default None)

        :param err: (Optional) error on the star positions of the star
          list 2 (default None).
          
        :param sip_order: (Optional) SIP order (default 3).

        :param crpix: (Optional) If an initial wcs is not given (init_sip
          set to None) this header value must be given.

        :param crval: (Optional) If an initial wcs is not given (init_sip
          set to None) this header value must be given.

        """
        return utils.astrometry.fit_sip(
            self.dimx, self.dimy, self.scale, star_list1, star_list2,
            params=params, init_sip=init_sip, err=err, sip_order=sip_order,
            crpix=crpix, crval=crval)



    def fit_stars(self, star_list, **kwargs):
        """
        Fit stars in one frame.

        This function is basically a wrapper around
        :meth:`utils.astrometry.fit_stars_in_frame`.

        .. note:: 2 fitting modes are possible::
        
            * Individual fit mode [multi_fit=False]
            * Multi fit mode [multi_fit=True]

            see :meth:`utils.astrometry.fit_stars_in_frame` for more
            information.

        :param star_list: Path to a list of stars
          
        :param kwargs: Same optional arguments as for
          :meth:`utils.astrometry.fit_stars_in_frame`.

        .. warning:: Some optional arguments are taken directly from the
          values computed at the init of the Class. The following
          optional arguments thus cannot be passed::
          
            * profile_name
            * scale
            * fwhm_pix
            * beta
            * fit_tol          
        """        
        frame = np.copy(self.data)
        star_list = utils.astrometry.load_star_list(star_list)
        protected_kwargs =['profile_name', 'scale', 'fwhm_pix', 'beta', 'fit_tol']
        for ik in protected_kwargs:
            if ik in kwargs:
                raise StandardError('{} should not be passed in kwargs'.format(ik))
        
        kwargs['profile_name'] = self.profile_name
        kwargs['scale'] = self.scale
        kwargs['fwhm_pix'] = self.fwhm_pix
        kwargs['beta'] = self.default_beta
        kwargs['fit_tol'] = self.fit_tol

        # fit
        fit_results = utils.astrometry.fit_stars_in_frame(
            frame, star_list, self.box_size, **kwargs)
        
        return utils.astrometry.fit2df(fit_results)
    
    def compute_alignment_parameters(self, image2, correct_distortion=False,
                                     star_list1=None, fwhm_arc=None,
                                     brute_force=True):
        """Return the alignment coefficients that match the stars of this
        image to the stars of the image 2.

        :param image2: Image instance.

        :param correct_distortion: (Optional) If True, a SIP is computed to
          match stars from frame 2 onto the stars from frame 1. But it
          needs a lot of stars to run correctly (default False).

        :param star_list1: (Optional) Path to a list of stars in
          the image 1. If given the fwhm_arc must also be set (default None).

        :param fwhm_arc: (Optional) mean FWHM of the stars in
          arcseconds. Must be given if star_list1 is not None
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
        def print_alignment_coeffs():
            """Print the alignement coefficients."""
            logging.info("\n> dx : " + str(coeffs.dx) + "\n" +
                         "> dy : " + str(coeffs.dy) + "\n" +
                         "> dr : " + str(coeffs.dr) + "\n" +
                         "> da : " + str(coeffs.da) + "\n" +
                         "> db : " + str(coeffs.db))

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
            x_range_len = range_coeff * float(image2.dimx)
            y_range_len = range_coeff * float(image2.dimy)

            x_hrange = np.arange(xystep_size, x_range_len/2, xystep_size)
            x_range = np.hstack((-x_hrange[::-1], 0, x_hrange)) + coeffs.dx
            
            y_hrange = np.arange(xystep_size, y_range_len/2, xystep_size)
            y_range = np.hstack((-y_hrange[::-1], 0, y_hrange)) + coeffs.dy
            
          
            r_range = np.linspace(-angle_range/2.,
                                  angle_range/2.,
                                  angle_steps) + coeffs.dr

            (coeffs.dx, coeffs.dy, coeffs.dr, guess_matrix) = (
                utils.astrometry.brute_force_guess(
                    image2.data.astype(np.float64), star_list1,
                    x_range, y_range, r_range,
                    coeffs.rc, coeffs.zoom,
                    image2.fwhm_pix * 3.))
            coeffs.da = 0.
            coeffs.db = 0.

            # Save guess matrix
            utils.io.write_fits(self._get_guess_matrix_path(),
                                guess_matrix)

        
            
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

        RANGE_COEFF = float(self._get_tuning_parameter(
            'RANGE_COEFF', self.config.ALIGNER_RANGE_COEFF))

        coeffs = self.get_initial_alignment_parameters()
        
        if star_list1 is None:
            star_list1, fwhm_arc = self.detect_stars(
                min_star_number=MIN_STAR_NB)
        elif fwhm_arc is None:
            raise StandardError('If the path to a list of stars is given (star_list1) the fwhm in arcsec(fwhm_arc) must also be given.')

        star_list1 = utils.astrometry.load_star_list(star_list1)
        
        image2.reset_fwhm_arc(fwhm_arc)
        self.reset_fwhm_arc(fwhm_arc)

        ##########################################
        ### BRUTE FORCE GUESS (only dx and dy) ###
        ##########################################
        if brute_force:
            logging.info("Brute force guess on large field")
            brute_force_alignment(4*XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS/2, RANGE_COEFF*10)
            logging.info("Brute force guess:") 
            print_alignment_coeffs()

            logging.info("Finer brute force guess")
            brute_force_alignment(XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS, RANGE_COEFF)
            logging.info("Brute force guess:") 
            print_alignment_coeffs()


        guess = [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db]
        
        ##########################
        ## FINE ALIGNMENT STEP ###
        ##########################
        
        # create sip corrected and transformed list
        star_list2 = utils.astrometry.transform_star_position_A_to_B(
            np.copy(star_list1), guess, coeffs.rc, coeffs.zoom,
            sip_A=self.sip)

        fit_results = image2.fit_stars(
            star_list2, no_aperture_photometry=True,
            multi_fit=True, enable_zoom=False,
            enable_rotation=True, fix_fwhm=True)
        
        if fit_results.empty:
            raise StandardError('registration failed. check INIT_ANGLE.')
        
        [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db] = match_star_lists(
            guess, np.copy(star_list1), utils.astrometry.df2list(fit_results),
            coeffs.rc, coeffs.zoom, sip1=self.sip, sip2=image2.sip)

        logging.info("Fine alignment parameters:") 
        print_alignment_coeffs()
                

        #####################################
        ### COMPUTE DISTORTION CORRECTION ###
        #####################################

        if correct_distortion:
            logging.info('Computing distortion correction polynomial (SIP)')
            raise Exception('Must be checked. using transformation parameters with sip may not be implemented properly.')
            # try to detect a maximum number of stars in frame 1
            star_list1_path1, fwhm_arc = self.detect_stars(
                min_star_number=400)

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
                np.copy(star_list1),
                [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db],
                coeffs.rc, coeffs.zoom,
                sip_A=self.sip, sip_B=image2.sip)

            # fit stars
            fit_results = image2.fit_stars(
                star_list2, no_aperture_photometry=True,
                multi_fit=False, enable_zoom=False,
                enable_rotation=False, 
                fix_fwhm=False)
            err = fit_results['x_err'].values


            ## FIT SIP 
            ## SIP 1 and SIP 2 are replaced by only one SIP that matches the
            ## stars of the frame 2 onto the stars of the frame 1
            self.sip = self.fit_sip(
                np.copy(star_list1),
                utils.astrometry.df2list(fit_results),
                params=[coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db,
                        coeffs.rc[0], coeffs.rc[1], coeffs.zoom],
                init_sip=None, err=None, crpix=self.sip.wcs.crpix,
                crval=self.sip.wcs.crval)
            image2.sip = None


        else:
            # fit stars
            fit_results = image2.fit_stars(
                star_list2, no_aperture_photometry=True,
                multi_fit=False, enable_zoom=False,
                enable_rotation=False, 
                fix_fwhm=False)

            fitted_star_nb = float(np.sum(~np.isnan(
                utils.astrometry.df2list(fit_results)[:,0])))
            
            if (fitted_star_nb < ERROR_RATIO * MIN_STAR_NB):
                raise StandardError("Not enough fitted stars in both cubes (%d%%). Alignment parameters might be wrong."%int(fitted_star_nb / MIN_STAR_NB * 100.))
                
            if (fitted_star_nb < WARNING_RATIO * MIN_STAR_NB):
                warnings.warn("Poor ratio of fitted stars in both cubes (%d%%). Check alignment parameters."%int(fitted_star_nb / MIN_STAR_NB * 100.))

            
            err = fit_results['x_err'].values
            
        star_list2 = utils.astrometry.transform_star_position_A_to_B(
        np.copy(star_list1),
            [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db],
            coeffs.rc, coeffs.zoom,
            sip_A=self.sip, sip_B=image2.sip)

        fwhm_arc2 = utils.stats.robust_mean(
            utils.stats.sigmacut(fit_results['fwhm_arc'].values))
        
        dx_fit = (star_list2[:,0]
                  - utils.astrometry.df2list(fit_results)[:,0])
        dy_fit = (star_list2[:,1]
                  - utils.astrometry.df2list(fit_results)[:,1])
        dr_fit = np.sqrt(dx_fit**2. + dy_fit**2.)
        final_err = np.mean(utils.stats.sigmacut(dr_fit))

        if not SKIP_CHECK:
            if final_err < self.arc2pix(WARNING_DIST):
                logging.info('Mean difference on star positions: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
            elif final_err < self.arc2pix(ERROR_DIST):
                warnings.warn('Mean difference on star positions is bad: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
            else:
                raise StandardError('Mean difference on star positions is too bad: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
        

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

        return {'coeffs':[coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db],
                'rc': coeffs.rc,
                'zoom_factor': coeffs.zoom,
                'sip1': self.sip,
                'sip2': image2.sip,
                'star_list1': star_list1,
                'star_list2': star_list2,
                'fwhm_arc1': fwhm_arc,
                'fwhm_arc2': fwhm_arc2}


