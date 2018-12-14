0#!/usr/bin/env python
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

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import astropy.stats
import photutils

from scipy import optimize, interpolate, signal
import pandas

import core

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
class Frame2D(core.Data):

    def __init__(self, *args, **kwargs):

        core.Data.__init__(self, *args, **kwargs)

        # checking
        if self.data.ndim != 2:
            raise TypeError('input image has {} dims but must have exactly 2 dimension'.format(self.data.ndim))

    def get_stats(self, fast=True):
        """Return image stats
        
        :param fast: If fast, only a random fraction of the image is
          used to make stats.

        :return: mean, median, std
        """
        FAST_FRAC = 0.02

        if fast:
            pix = np.random.randint(0, high=self.data.size,
                                    size=FAST_FRAC*self.data.size)
            _data = np.copy(self.data).flatten()[pix]
        else:
            _data = self.data
            
        return astropy.stats.sigma_clipped_stats(_data, sigma=3.0)


#################################################
#### CLASS Image ################################
#################################################

class Image(Frame2D, core.Tools):

    BOX_SIZE_COEFF = 7
    FIT_TOL = 1e-2
    REDUCED_CHISQ_LIMIT = 1.5
    DETECT_INDEX = 0
    
    profiles = ['moffat', 'gaussian']
    wcs_params = ('target_ra', 'target_dec', 'target_x', 'target_y')
    
    def __init__(self, data, instrument=None, config=None,
                 data_prefix="./", sip=None, **kwargs):

        # try to read instrument parameter from file
        if isinstance(data, str):
            if 'hdf' in data:
                with utils.io.open_hdf5(data, 'r') as f:
                    if instrument is None:
                        if 'instrument' not in f.attrs:
                            raise ValueError("instrument could not be read from the file attributes. Please set it to 'sitelle' or 'spiomm'")                
                        instrument = f.attrs['instrument']
            elif 'fit' in data:
                hdu = utils.io.read_fits(data, return_hdu_only=True)
                _hdr = hdu[0].header
                if 'INSTRUME' in _hdr:
                    instrument = _hdr['INSTRUME'].lower()
                            
        
        core.Tools.__init__(self, instrument=instrument,
                            data_prefix=data_prefix,
                            config=config)
        
        Frame2D.__init__(self, data, **kwargs)

        self.params['instrument'] = instrument

        # load old orb file header
        if 'target_x' not in self.params:
            if 'TARGETX' in self.params:
                self.params['target_x'] = float(self.params['TARGETX'])
        if 'target_y' not in self.params:
            if 'TARGETY' in self.params:
                self.params['target_y'] = float(self.params['TARGETY'])
        if 'target_ra' not in self.params:
            if 'TARGETR' in self.params:
                self.params['target_ra'] = utils.astrometry.ra2deg(
                    self.params['TARGETR'].split(':'))
        if 'target_dec' not in self.params:
            if 'TARGETD' in self.params:
                self.params['target_dec'] = utils.astrometry.dec2deg(
                    self.params['TARGETD'].split(':'))

        # check if all needed parameters are present
        for iparam in self.wcs_params:
            if iparam not in self.params:
                raise StandardError('param {} must be set'.format(iparam))

        # load astrometry params
        if 'profile_name' not in self.params:
            self.params['profile_name'] = self.config.PSF_PROFILE

        if ('target_ra' in self.params and 'target_dec' in self.params):
            if not isinstance(self.params.target_ra, float):
                raise TypeError('target_ra must be a float')
            if not isinstance(self.params.target_dec, float):
                raise TypeError('target_dec must be a float')

            target_radec = (self.params.target_ra, self.params.target_dec)
        else:
            target_radec = None
            
        if ('target_x' in self.params and 'target_y' in self.params):
            target_xy = (self.params.target_x, self.params.target_y)               
        else:
            target_xy = None
          
        if 'data_prefix' not in kwargs:
            kwargs['data_prefix'] = self._data_prefix
            
        if 'wcs_rotation' in self.params:
            wcs_rotation = self.params.wcs_rotation
        else:
            wcs_rotation=self.config.INIT_ANGLE
        

        # define astrometry parameters
        if 'box_size_coeff' in self.params:
            self.box_size_coeff = self.params.box_size_coeff
        else:
            self.box_size_coeff = self.BOX_SIZE_COEFF
        
        
        if self.params.profile_name == 'moffat':
            self.box_size_coeff /= 3.
            
        if 'fwhm_arc' in self.params:
            self.fwhm_arc = self.params.fwhm_arc
        else:
            self.fwhm_arc = self.config.INIT_FWHM

        self.fov = self.config.FIELD_OF_VIEW_1
        self.reset_scale(float(self.fov) * 60. / self.dimx)

        if 'detect_stack' in self.params:
            self.detect_stack = self.params.detect_stack
        else:
            self.detect_stack = self.config.DETECT_STACK

        if 'moffat_beta' in self.params:
            self.default_beta = self.params.moffat_beta
        else:
            self.default_beta = self.config.MOFFAT_BETA
        
        self.fit_tol = self.FIT_TOL
    
        # define profile
        self.reset_profile_name(self.params.profile_name)

        self.reduced_chi_square_limit = self.REDUCED_CHISQ_LIMIT

        self.target_ra = self.params.target_ra
        self.target_dec = self.params.target_dec
        self.target_x = self.params.target_x
        self.target_y = self.params.target_y
        
        if 'wcs_rotation' in self.params:
            self.wcs_rotation = self.params.wcs_rotation
        else:
            self.wcs_rotation = self.config.INIT_ANGLE
        
        self.sip = None
        if sip is not None:
            if isinstance(sip, pywcs.WCS):
                self.sip = sip
            else:
                raise StandardError('sip must be an astropy.wcs.WCS instance')

    def _get_star_list_path(self):
        """Return the default path to the star list file."""
        return self._data_path_hdr + "star_list.hdf5"

    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results.hdf5"

    def set_wcs(self, wcs):
        """Set WCS from w WCS instance or a FITS image

        :param wcs: Must be an astropy.wcs.WCS instance or a path to a FITS image
        """
        if isinstance(wcs, str):
            wcs = pywcs.WCS(
                orb.utils.io.read_fits(wcs_path, return_hdu_only=True)[0].header,
                naxis=2, relax=True)
        self.update_params(wcs.to_header(relax=True))

    def get_wcs(self):
        """Return the WCS of the cube as a astropy.wcs.WCS instance """
        return pywcs.WCS(self.get_header(), naxis=2, relax=True)

    def get_wcs_header(self):
        """Return the WCS of the cube as a astropy.wcs.WCS instance """
        return self.get_wcs().to_header(relax=True)

        
    def pix2world(self, xy, deg=True):
        """Convert pixel coordinates to celestial coordinates

        :param xy: A tuple (x,y) of pixel coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...)

        :param deg: (Optional) If true, celestial coordinates are
          returned in sexagesimal format (default False).

        .. note:: it is much more effficient to pass a list of
          coordinates than run the function for each couple of
          coordinates you want to transform.
        """
        xy = np.squeeze(xy).astype(float)
        if np.size(xy) == 2:
            x = [xy[0]]
            y = [xy[1]]
        elif np.size(xy) > 2 and len(xy.shape) == 2:
            if xy.shape[0] < xy.shape[1]:
                xy = np.copy(xy.T)
            x = xy[:,0]
            y = xy[:,1]
        else:
            raise StandardError('xy must be a tuple (x,y) of coordinates or a list of tuples ((x0,y0), (x1,y1), ...)')

        if not self.has_param('dxmap') or not self.has_param('dymap'):
            coords = np.array(
                self.get_wcs().all_pix2world(
                    x, y, 0)).T
        else:
            if np.size(x) == 1:
                xyarr = np.atleast_2d([x, y]).T
            else:
                xyarr = xy
            coords = orb.utils.astrometry.pix2world(
                self.get_wcs_header(), self.dimx, self.dimy, xyarr,
                self.params.dxmap, self.params.dymap)
        if deg:
            return coords
        else: return np.array(
            [orb.utils.astrometry.deg2ra(coords[:,0]),
             orb.utils.astrometry.deg2dec(coords[:,1])])


    def world2pix(self, radec, deg=True):
        """Convert celestial coordinates to pixel coordinates

        :param xy: A tuple (x,y) of celestial coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...). Must be in degrees.

        .. note:: it is much more effficient to pass a list of
          coordinates than run the function for each couple of
          coordinates you want to transform.
        """
        radec = np.squeeze(radec)
        if np.size(radec) == 2:
            ra = [radec[0]]
            dec = [radec[1]]
        elif np.size(radec) > 2 and len(radec.shape) == 2:
            if radec.shape[0] < radec.shape[1]:
                radec = np.copy(radec.T)
            ra = radec[:,0]
            dec = radec[:,1]
        else:
            raise StandardError('radec must be a tuple (ra,dec) of coordinates or a list of tuples ((ra0,dec0), (ra1,dec1), ...)')

        if not self.has_param('dxmap') or not self.has_param('dymap'):
            coords = np.array(
                self.get_wcs().all_world2pix(
                    ra, dec, 0,
                    detect_divergence=False,
                    quiet=True)).T
        else:
            radecarr = np.atleast_2d([ra, dec]).T
            coords = orb.utils.astrometry.world2pix(
                self.get_wcs_header(), self.dimx, self.dimy, radecarr,
                self.params.dxmap, self.params.dymap)

        return coords

    def load_fit_results(self, fit_results_path=None):
        """Load a file containing the fit results"""
        if fit_results_path is None:
            fit_results_path = self._get_fit_results_path()
        self.fit_results.load_stars_params(fit_results_path)
    
    def load_star_list(self, star_list_path):
        """Load a list of stars coordinates

        :param star_list_path: The path to the star list file.

        .. seealso:: :py:meth:`astrometry.load_star_list`
        """
        star_list = utils.astrometry.load_star_list(
            star_list_path, silent=False)
        
        self.reset_star_list(np.array(star_list, dtype=float))
        
        return self.star_list

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
        
    def reset_star_list(self, star_list):
        """Reset the list of stars
        
        :param star_list: An array of shape (star_nb, 2) giving the
          positions in x and y of the stars.
        """
        if isinstance(star_list, list):
            star_list = np.array(star_list)
            
        if len(star_list.shape) == 2:
            if star_list.shape[1] != 2:
                raise StandardError('Incorrect star list shape. The star list must be an array of shape (star_nb, 2)')
        else:
            raise StandardError('Incorrect star list shape. The star list must be an array of shape (star_nb, 2)')
            
        self.star_list = star_list
        self.star_nb = self.star_list.shape[0]
        # create an empty StarsParams array
        self.fit_results = core.StarsParams(self.star_nb, 1,
                                            instrument=self.instrument)
        return self.star_list
    
    def reset_scale(self, scale):
        """Reset scale attribute.
        
        :param scale: Frame scale in arcsec/pixel
        """
        self.scale = float(scale)
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
    
    def arc2pix(self, x):
        """Convert pixels to arcseconds

        :param x: a value or a vector in pixel
        """
        if self.scale is not None:
            return np.array(x).astype(float) / self.scale
        else:
            raise StandardError("Scale not defined")

    def pix2arc(self, x):
        """Convert arcseconds to pixels

        :param x: a value or a vector in arcsec
        """
        if self.scale is not None:
            return np.array(x).astype(float) * self.scale
        else:
            raise StandardError("Scale not defined")

    def query_vizier(self, catalog='gaia', max_stars=100):
        """Return a list of star coordinates around an object in a
        given radius based on a query to VizieR Services
        (http://vizier.u-strasbg.fr/viz-bin/VizieR)    

        :param catalog: (Optional) Catalog to ask on the VizieR
          database (see notes) (default 'gaia')

        :param max_stars: (Optional) Maximum number of row to retrieve
          (default 100)

        .. seealso:: :py:meth:`orb.utils.web.query_vizier`
        """
        radius = self.fov / np.sqrt(2)
        if self.target_ra is None or self.target_dec is None:
            raise StandardError('No catalogue query can be done. Please make sure to give target_radec and target_xy parameter at class init')
        
        return utils.web.query_vizier(
            radius, self.target_ra, self.target_dec,
            catalog=catalog, max_stars=max_stars)


    def detect_stars(self, min_star_number=4, saturation_threshold=35000):
        """Detect star positions in data.

        :param min_star_number: Minimum number of stars to
          detect. Must be greater or equal to 4 (minimum star number
          for any alignment process). this is also the number of stars
          returned (sorted by intensity)

        :param saturation_threshold: (Optional) Number of counts above
          which the star can be considered as saturated. Very low by
          default because at the ZPD the intensity of a star can be
          twice the intensity far from it (default 35000).

        """
        DETECT_THRESHOLD = 5
        MAX_ROUNDNESS = 0.1
        FWHM_STARS_NB = 30
        
        mean, median, std = self.get_stats()
        daofind = photutils.DAOStarFinder(fwhm=self.fwhm_pix,
                                          threshold=DETECT_THRESHOLD * std)
        sources = daofind(self.data.T - median).to_pandas()
        sources = sources[sources.peak < saturation_threshold]
        sources = sources[np.abs(sources.roundness2) < MAX_ROUNDNESS]
        sources = sources.sort_values(by=['flux'], ascending=False)
        logging.info("%d stars detected" %(len(sources)))

        mean_fwhm, mean_fwhm_err = self.detect_fwhm(sources[:FWHM_STARS_NB])

        sources.to_hdf(self._get_star_list_path(), 'data', mode='w')
        logging.info('sources written to {}'.format(self._get_star_list_path()))
        return self._get_star_list_path(), mean_fwhm

    def load_star_list(self, star_list):
        """Load a star list from different sources.

        :star_list: can be a np.ndarray of shape (n, 2) or a path to a star list
        """
        if isinstance(star_list, str):
            sources = pandas.read_hdf(self._get_star_list_path(), key='data')
            star_list = utils.astrometry.sources2list(sources)

        elif isinstance(star_list, pandas.DataFrame):
            star_list = utils.astrometry.sources2list(star_list)

        else:
            if not isinstance(np.ndarray):
                raise TypeError('star_list must be an instance of numpy.ndarray or a path to a star list')
        if star_list.ndim != 2: raise TypeError('star list must have 2 dimensions')
        if star_list.shape[1] != 2: raise TypeError('badly formatted star list. must have shape (n, 2)')
        self.star_list = np.copy(star_list)
        return star_list

    def detect_fwhm(self, star_list):
        """Return fwhm of a list of stars

        :param star_list: list of stars (can be an np.ndarray or a path
          ot a star list).
        """
        self.load_star_list(star_list)
        
        mean_fwhm, mean_fwhm_err = utils.astrometry.detect_fwhm_in_frame(
            self.data, utils.astrometry.sources2list(star_list),
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
        C_OUT = np.sqrt((MIN_BACK_COEFF*C_AP**2.) + C_IN**2.)


        star_list = self.load_star_list(star_list)
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
                param_list = fit_params.get_star_list(all_params=True)
            else:
                param_list = fit_params[:,param]
            snr = fit_params[:,'snr']
            
            if snr_min is None:
                snr_min = max(utils.stats.robust_median(snr), 3.)
            
            if return_index:
                index = np.zeros(param_list.shape[0])
                
            param_list_f = list()
            for istar in range(param_list.shape[0]):
                if not np.isnan(snr[istar]):
                    dist = np.sqrt(fit_params[istar,'dx']**2
                                   + fit_params[istar,'dy']**2)
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
        logging.info("Initial target position in the image (X,Y): {} {}".format(
            self.target_x, self.target_y))

        deltax = self.scale / 3600. # arcdeg per pixel
        deltay = float(deltax)

        # get FWHM
        star_list_fit_init_path, fwhm_arc = self.detect_stars(
            min_star_number=max_stars_detect)
        star_list_fit_init = self.load_star_list(star_list_fit_init_path)
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
        self.reset_star_list(star_list_pix)

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
            logging.info('histogram check: correlation level {}, angle {}, dx {}, dy {}'.format(*max_list[-1]))
        max_list = sorted(max_list, key = lambda imax: imax[0], reverse=True)
        self.target_x += max_list[0][2]
        self.target_y += max_list[0][3]
        self.wcs_rotation = max_list[0][1]

        # update wcs
        wcs = utils.astrometry.create_wcs(
            self.target_x, self.target_y,
            deltax, deltay, self.target_ra, self.target_dec,
            self.wcs_rotation, sip=self.sip)

        star_list_pix = radius_filter(
            world2pix(wcs, star_list_deg), rmax)
        self.reset_star_list(star_list_pix)

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
        ## self.reset_star_list(star_list_pix)
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
            self.reset_star_list(star_list_pix)
            
            fit_params = self.fit_stars(
                local_background=True,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True, save=False)
            
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
        self.reset_star_list(star_list_pix)

        # fit based on SIP corrected parameters
        fit_params = self.fit_stars(
            local_background=False,
            multi_fit=False, fix_fwhm=True,
            no_aperture_photometry=True,
            save=False)

        _x = fit_params[:,'x']
        _y = fit_params[:,'y']
        _r = np.sqrt((_x - self.dimx / 2.)**2.
                     + (_y - self.dimy / 2.)**2.)

        _dx = fit_params[:, 'dx']
        _dy = fit_params[:, 'dy']

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
        _w = 1./fit_params[:, 'x_err'][nonans]
        _x = fit_params[:, 'x'][nonans]
        _y = fit_params[:, 'y'][nonans]
        _dx = fit_params[:, 'dx'][nonans]
        _dy = fit_params[:, 'dy'][nonans]

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
                
            self.reset_star_list(star_list_pix)

            fit_params = self.fit_stars(
                local_background=False,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True,
                save=False)
            
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
