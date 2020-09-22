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
import astropy.time
import astropy.coordinates
import astropy.units

import photutils

from scipy import optimize, interpolate, signal
import pandas

import orb.core
import orb.photometry

import orb.utils.astrometry
import orb.utils.image
import orb.utils.stats
import orb.utils.vector
import orb.utils.web
import orb.utils.misc
import orb.utils.io

import pylab as pl
import matplotlib.cm
import matplotlib.colors

import copy

#################################################
#### CLASS Frame2D ##############################
#################################################
class Frame2D(orb.core.WCSData):

    def __init__(self, *args, **kwargs):
        
        orb.core.WCSData.__init__(self, *args, **kwargs)

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
        newim.update_params(cutout.wcs.to_header(relax=True))
        ((ymin, ymax), (xmin, xmax)) = cutout.bbox_original
        newim.params['cropped_bbox'] = (xmin, xmax+1, ymin, ymax+1)
        newim.set_wcs(cutout.wcs)
        return newim

    def imshow(self, figsize=(15,15), perc=99, cmap='viridis', wcs=True, alpha=1, ncolors=None,
               vmin=None, vmax=None):
        """Convenient image plotting function

        :param figsize: size of the figure (same as pyplot.figure's figsize keyword)

        :param perc: percentile of the data distribution used to scale
          the colorbar. Can be a tuple (min, max) or a scalar in which
          case the min percentile will be 100-perc.

        :param cmap: colormap

        :param wcs: if True, display wcs coordinates. Else, pixel coordinates are shown.

        :param alpha: image opacity (if another image is displayed above)

        :param ncolors: if an integer is passed, the colorbar is
          discretized to this number of colors.
        
        :param vmin: min value used to scale the colorbar. If set the
          perc parameter is not used.

        :param vmax: max value used to scale the colorbar. If set the
          perc parameter is not used.
        """
        try:
            iter(perc)
        except Exception:
            perc = np.clip(float(perc), 50, 100)
            perc = 100-perc, perc

        else:
            if len(list(perc)) != 2:
                raise Exception('perc should be a tuple of len 2 or a single float')

        if vmin is None: vmin = np.nanpercentile(self.data, perc[0])
        if vmax is None: vmax = np.nanpercentile(self.data, perc[1])
        
        if ncolors is not None:
            cmap = getattr(matplotlib.cm, cmap)
            norm = matplotlib.colors.BoundaryNorm(np.linspace(vmin, vmax, ncolors),
                                                  cmap.N, clip=True)
        else:
            norm = None
            
        fig = pl.figure(figsize=figsize)
        if wcs:
            ax = fig.add_subplot(111, projection=self.get_wcs())
            ax.coords[0].set_major_formatter('d.dd')
            ax.coords[1].set_major_formatter('d.dd')
        pl.imshow(self.data.T, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', alpha=alpha,norm=norm)
        

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
        if not self.has_param('fwhm_arc'):
            self.params['fwhm_arc'] = self.config.INIT_FWHM


    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results.hdf5"
        
    def reset_fwhm_arc(self, fwhm_arc):
        """Reset FWHM of stars in arcsec

        :param fwhm_arc: FWHM of stars in arcsec
        """
        self.params['fwhm_arc'] = float(fwhm_arc)
        logging.debug('fwhm reset to {} arcseconds, i.e. {} pixels'.format(self.params.fwhm_arc, self.get_fwhm_pix()))

    def get_fwhm_pix(self):
        return self.arc2pix(self.params.fwhm_arc)

    def get_box_size(self):
        box_size = int(np.ceil(self.box_size_coeff *  self.get_fwhm_pix()))
        box_size += int(~box_size%2) # make it odd
        return box_size
    
    def copy(self, data=None):
        """Return a copy of the instance

        :param data: (Optional) can be used to change data
        """
        return Frame2D.copy(self, data=data, instrument=self.instrument,
                            config=self.config, data_prefix=self._data_prefix,
                            sip=self.get_wcs())
    

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

            frame = orb.utils.image.correct_cosmic_rays(frame, cr_map)
            
        if shift is not None:
            orb.utils.validate.has_len(shift, 2, object_name='shift')
            dx, dy = shift
            if (dx != 0.) and (dy != 0.):
                frame = orb.utils.image.shift_frame(frame, dx, dy, 
                                                0, self.dimx, 
                                                0, self.dimy, 1)
            
        return frame

    def reset_wcs(self, target_ra, target_dec, fov=None, wcs_rotation=None):
        """Reset WCS to a raw WCS that can be registered without error with
        register().
        
        :param target_ra: RA at the center of the field

        :param target_dec: DEC at the center of the field

        :param fov: (Optional) If the instrument is unknown (not
          sitelle/spiomm) the field of view must be set in arcminutes
          (default None).

        :param fov: (Optional) If the instrument is unknown (not
          sitelle/spiomm) the wcs_rotation must be set in degrees
          (default None).
        
        """
        if fov is None:
            if 'FIELD_OF_VIEW_1' not in self.config:
                raise Exception('instrument config not loaded. Please set fov.')
            if self.is_cam1():
                fov = self.config['FIELD_OF_VIEW_1']
            else:
                fov = self.config['FIELD_OF_VIEW_2']
        if wcs_rotation is None:
            if 'WCS_ROTATION' not in self.config:
                raise Exception('instrument config not loaded. Please set wcs_rotation.')
            wcs_rotation = self.config['WCS_ROTATION']
            
        wcs = orb.utils.astrometry.create_wcs(
            self.dimx/2, self.dimy/2, fov/self.dimx/60., fov/self.dimx/60.,
            target_ra, target_dec, wcs_rotation)

        self.set_wcs(wcs)

    def reset_sip(self):
        """Remove sip informations"""
        wcs = self.get_wcs()
        wcs.sip = None
        self.set_wcs(wcs)
        self.params['CTYPE1'] = 'RA---TAN'
        self.params['CTYPE2'] = 'DEC--TAN'
            
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
                print(astropy.coordinates.get_icrs_coordinates(self.params.OBJECT))
                object_found = True
            except NameResolveError:
                logging.debug('object name could not be resolved')
                
        logging.info('looking in the standard table for {}'.format(self.params.OBJECT))
        try:
            std_name = ''.join(self.params.OBJECT.strip().split()).upper()
            std_ra, std_dec, std_pm_ra, std_pm_dec = self._get_standard_radec(
                std_name, return_pm=True)
            object_found = True
        except Exception:
            logging.warning('object name not found in the standard table')

        if not object_found:
            raise Exception('object coordinates could not be resolved')

        std_yr_obs = float(self.params['DATE-OBS'].split('-')[0])
        pm_orig_yr = 2000 # radec are considered to be J2000
        # compute ra/dec with proper motion
        std_ra, std_dec = orb.utils.astrometry.compute_radec_pm(
            std_ra, std_dec, std_pm_ra, std_pm_dec,
            std_yr_obs - pm_orig_yr)
        std_ra_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *orb.utils.astrometry.deg2ra(std_ra))
        std_dec_str = '{:.0f}:{:.0f}:{:.3f}'.format(
            *orb.utils.astrometry.deg2dec(std_dec))
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
            self.params['profile_name'] = profile_name
        else:
            raise Exception(
                "Bad profile name (%s) please choose it in: %s"%(
                    profile_name, str(self.profiles)))
        
    def get_stars_from_catalog(self, max_stars=5000):
        """Return star positions, corrected for proper motion from Gaia DR2
        """

        cat = self.query_vizier(as_pandas=True, max_stars=max_stars, catalog='gaia2')

        date_key = 'DATE-OBS'
        if date_key not in self.params:
            date_key = 'DATE'
            if date_key not in self.params:
                raise Exception('No DATE or DATE-OBS in header')
            
        obsdate = astropy.time.Time(self.params[date_key])
            
        epoch = astropy.time.Time(cat.Epoch.values[0], format='decimalyear')
        pm_unit = astropy.units.mas/astropy.units.yr

        # everything below was adapted from
        # astropy.coordinates.SkyCoord.apply_space_motion. When ORB
        # will be adapted to python 3 this function should be directly
        # used
        cat_frame = astropy.coordinates.ICRS(
            cat.ra.values * astropy.units.deg, 
            cat.dec.values * astropy.units.deg, 
            pm_ra_cosdec=cat.pmRA.values*pm_unit, 
            pm_dec=cat.pmRA.values*pm_unit)

        t1 = epoch.tdb
        t2 = obsdate.tdb

        icrsrep = cat_frame.represent_as(
            astropy.coordinates.SphericalRepresentation,
            astropy.coordinates.SphericalDifferential)
        icrsvel = icrsrep.differentials['s']

        starpm = astropy._erfa.starpm(
            icrsrep.lon.radian, icrsrep.lat.radian,
            icrsvel.d_lon.to_value(astropy.units.radian/astropy.units.yr),
            icrsvel.d_lat.to_value(astropy.units.radian/astropy.units.yr),
            0, 0, t1.jd1, t1.jd2, t2.jd1, t2.jd2)

        icrs2 = astropy.coordinates.ICRS(
            starpm[0]* astropy.units.radian, 
            starpm[1]* astropy.units.radian, 
            pm_ra_cosdec=starpm[2]* astropy.units.radian/astropy.units.yr * np.cos(starpm[1]),
            pm_dec=starpm[3]* astropy.units.radian/astropy.units.yr)

        sld = np.array([icrs2.ra.to('deg').value, icrs2.dec.to('deg').value])
        slp = self.world2pix(sld)
        slp = np.array([istar for istar in slp 
                        if istar[0] > 0 
                        and istar[0] < self.dimx 
                        and istar[1] > 0 
                        and istar[1] < self.dimy])

        sld = self.pix2world(slp)
        sources = pandas.DataFrame({
            'x':slp[:,0],
            'y':slp[:,1],
            'ra':sld[:,0],
            'dec':sld[:,1]
        })
        
        return sources

        
    def detect_stars(self, min_star_number=30, max_roundness=0.1,
                     max_radius_coeff=1., path=None, saturation_threshold=None):
        """Detect star positions in data.

        :param min_star_number: Minimum number of stars to
          detect. Must be greater or equal to 4 (minimum star number
          for any alignment process). this is also the number of stars
          returned (sorted by intensity)

        :param path: (Optional) Path to the output star list file. If
          None, nothing is written.

        """
        DETECT_THRESHOLD = 5
        FWHM_STARS_NB = 30
        
        mean, median, std = self.get_stats()
        daofind = photutils.DAOStarFinder(fwhm=self.get_fwhm_pix(),
                                          threshold=DETECT_THRESHOLD * std)
        sources = daofind(self.data.T - median).to_pandas()
        if len(sources) == 0: raise Exception('no star detected, check input image')
        logging.debug('initial number of stars: {}'.format(len(sources)))
        
        # this 0.45 on the saturation threshold ensures that stars at ZPD won't saturate
        if saturation_threshold is None:
            saturation_threshold = np.nanmax(self.data) * 0.45 
        sources = sources[sources.peak < saturation_threshold]
        logging.debug('number of stars after peak filter: {}'.format(len(sources)))
            
        # filter by roundness
        sources = sources[np.abs(sources.roundness2) < max_roundness]
        logging.debug('number of stars after roundness filter: {}'.format(len(sources)))
        
        # filter by radius from center to avoid using distorded stars
        sources['radius'] = np.sqrt((sources.xcentroid - self.dimx/2.)**2
                                    + (sources.ycentroid - self.dimy/2.)**2)
        sources = sources[sources.radius < max_radius_coeff * min(self.dimx, self.dimy) / 2.]
        logging.debug('number of stars after radius filter: {}'.format(len(sources)))
        # sort by flux
        sources = sources.sort_values(by=['flux'], ascending=False)
        logging.info("%d stars detected" %(len(sources)))
        sources = sources[:min_star_number]
        logging.info("star list reduced to %d stars" %(len(sources)))
        sources = orb.utils.astrometry.df2list(sources)
        sources = self.fit_stars(sources, no_aperture_photometry=True)
        mean_fwhm, mean_fwhm_err = self.detect_fwhm(sources[:FWHM_STARS_NB])    

        if path is not None:         
            orb.utils.io.open_file(path, 'w') # used to create the folder tree
            sources.to_hdf(path, 'data', mode='w')
            logging.info('sources written to {}'.format(path))
        return sources, mean_fwhm # be careful this is returned in pixels

    def detect_fwhm(self, star_list):
        """Return fwhm of a list of stars (in pixels)

        can be converted to arcseconds with self.pix2arc()

        :param star_list: list of stars (can be an np.ndarray or a path
          to a star list).
        """
        stars = self.fit_stars(star_list, no_aperture_photometry=True)        
        fwhms = orb.utils.stats.sigmacut(stars['fwhm'], sigma=3.5)
        mean_fwhm = np.nanmedian(stars['fwhm'])
        mean_fwhm_err = orb.utils.stats.unbiased_std(fwhms)
        
        if np.isnan(mean_fwhm):
            logging.warning('detected FWHM is nan')
            return self.get_fwhm_pix(), 0
        else:
            logging.info("Detected stars FWHM : {:.2f}({:.2f}) pixels, {:.2f}({:.2f}) arc-seconds".format(mean_fwhm, mean_fwhm_err, self.pix2arc(mean_fwhm), self.pix2arc(mean_fwhm_err)))
            self.reset_fwhm_arc(self.pix2arc(mean_fwhm))
            return mean_fwhm, mean_fwhm_err # return in pixels

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

        star_list = orb.utils.astrometry.load_star_list(star_list)
        aper = photutils.CircularAperture(star_list,
                                          r=C_AP * self.get_fwhm_pix())
        aper_ann = photutils.CircularAnnulus(star_list,
                                             r_in=C_IN * self.get_fwhm_pix(),
                                             r_out=C_OUT * self.get_fwhm_pix())

        ann_masks = aper_ann.to_mask(method='center')
        ann_medians = list()
        for imask in ann_masks:
            ivalues = imask.multiply(self.data.T)            
            ivalues = ivalues[ivalues > 0]
            if ivalues.size > aper.area() * MIN_BACK_COEFF:
                ann_medians.append(orb.utils.astrometry.sky_background_level(
                    ivalues, return_error=True))
            else:
                ann_medians.append((np.nan, np.nan))
        
        ann_medians = np.array(ann_medians)
        phot_table = photutils.aperture_photometry(self.data.T, aper,
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
                 max_roundness=0.2,
                 max_radius_coeff=1.,
                 saturation_threshold=None,
                 sip_order=3,
                 rrange=None, xyrange=None,
                 nsteps=7,
                 return_fit_params=False, 
                 skip_registration=False,
                 compute_precision=True, compute_distortion=False,
                 return_error_maps=False,
                 return_error_spl=False, star_list_query=None, fwhm_arc=None):
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

        :param max_radius_coeff: (Optional) Maximum radius to detect stars.

        :param nsteps: (Optional) Number of refinement steps (default 7).

        :param sip_order: (Optional) SIP order (default 3)

        :param return_fit_params: (Optional) If True return final fit
          parameters instead of wcs (default False).

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

        :param star_list_query: (Optional) A list of star positions in
          degree to perform registration. fwhm_arc must also be set.

        :param fwhm_arc: (Optional) must be set if star_list_query is
          not None.

        """        
        def get_filtered_params(fit_params, snr_min=None,
                                dist_min=1e9,
                                param='star_list',
                                return_index=False):

            if param == 'star_list':
                param_list = orb.utils.astrometry.df2list(fit_params)
            else:
                param_list = fit_params[param]
            snr = fit_params['snr']
            
            if snr_min is None:
                snr_min = max(orb.utils.stats.robust_median(snr), 5.)
                snr_min = min(snr_min, 10)
                logging.debug('star filter snr min: {}'.format(snr_min))
            
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
        
      
        MIN_STAR_NB = 10 # Minimum number of stars to get a correct WCS

        XYMAX = 500
        RMAX = 6
        ZMAX = 0.03

        if not (self.params.target_ra is not None and self.params.target_dec is not None
                and self.params.target_x is not None and self.params.target_y is not None
                and self.params.wcs_rotation is not None):
            raise Exception("Not enough parameters to register data. Please set target_xy, target_radec and wcs_rotation parameters at Astrometry init")

        if return_error_maps and return_error_spl: raise Exception('return_error_maps and return_error_spl cannot be both set to True, choose only one of them')
        
        logging.info('Computing WCS')
        logging.info('initial wcs')
        logging.info(str(self.get_wcs()))
        logging.info('initial sip: {}'.format(self.get_wcs().sip))
        
        # Query to get reference star positions in degrees
        if star_list_query is None:
            #star_list_query = self.query_vizier(max_stars=100 * max_stars_detect)
            star_list_query = self.get_stars_from_catalog(max_stars=100 * max_stars_detect)
            star_list_query = np.array([star_list_query.ra.values, star_list_query.dec.values]).T
        else:
            star_list_query = np.copy(star_list_query)
            if fwhm_arc is None:
                logging.warning('fwhm_arc is kept to its default value: {}'.format(self.params.fwhm_arc))
            else:
                self.reset_fwhm_arc(fwhm_arc)
                
        if len(star_list_query) < MIN_STAR_NB:
            raise Exception("Not enough stars found in the field (%d < %d)"%(
                len(star_list_query), MIN_STAR_NB))
            
        # reference star position list in degrees
        sl_cat_deg = star_list_query[:max_stars_detect*20]
            
        # ## Define a basic WCS        
        # wcs = orb.utils.astrometry.create_wcs(
        #     self.params.target_x, self.params.target_y,
        #     deltax, deltay, self.params.target_ra, self.params.target_dec,
        #     self.params.wcs_rotation, sip=self.get_wcs())
        
        # Compute initial star positions from initial transformation
        # parameters
        # rmax = max(self.dimx, self.dimy) / np.sqrt(2)
        
        
        # get FWHM
        sl_im_pix_path, fwhm_pix = self.detect_stars(
            min_star_number=max_stars_detect, max_roundness=max_roundness,
            max_radius_coeff=max_radius_coeff, saturation_threshold=saturation_threshold)
        sl_im_pix = orb.utils.astrometry.load_star_list(
            sl_im_pix_path, remove_nans=True)

        self.box_size_coeff = 5.

        if not skip_registration:
            # match lists
            if rrange is None:
                rrange = (RMAX, RMAX/6)

            if xyrange is None:
                xyrange = (XYMAX, XYMAX/10.)

            wcs, sl_cat_matched, sl_im_matched = orb.utils.astrometry.match_star_lists(
                self.get_wcs(),
                sl_cat_deg[:,:2],
                sl_im_pix, [self.params.target_x, self.params.target_y],
                xyrange=xyrange,
                rrange=rrange,
                zrange=(ZMAX, ZMAX/1),
                nsteps=nsteps)

            self.set_wcs(wcs)

            logging.info('wcs after lists matching')
            logging.info(str(self.get_wcs()))

            try:
                # refine registration
                wcs = orb.utils.astrometry.fit_wcs(
                    sl_im_pix[sl_im_matched],
                    sl_cat_deg[sl_cat_matched][:,:2],
                    self.get_wcs())
            except Exception as e:
                logging.warning('registration could not be fitted: {}'.format(e))
                
            else:
                # update wcs
                self.set_wcs(wcs)
                logging.info('wcs after fit')
                logging.info(str(self.get_wcs()))                    

        ## COMPUTE SIP
        if compute_distortion:
            logging.info('Computing SIP coefficients')
            logging.info('sip before sip fit (A matrix)')
            if self.get_wcs().sip is not None:
                logging.info(str(self.get_wcs().sip.a))
            else:
                logging.info('no sip')

            sl_cat_pix = self.world2pix(sl_cat_deg[:,:2])
        
            logging.info('sip computed with {} stars'.format(sl_cat_pix.shape[0]))
            
            fit = self.fit_stars(
                sl_cat_pix,
                local_background=True,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True)
            
            ## remove bad fits
            
            fit[np.abs(fit.dx) > self.get_fwhm_pix() * 3] = np.nan
            fit[np.abs(fit.dy) > self.get_fwhm_pix() * 3] = np.nan

            logging.info('{} stars fitted'.format(len(fit.dropna())))

            wcs = self.fit_sip(sl_cat_pix,
                               orb.utils.astrometry.df2list(fit),
                               params=None, init_sip=self.get_wcs(),
                               err=None, sip_order=sip_order)

            # update wcs
            self.set_wcs(wcs)
            logging.info('sip after sip fit (A matrix)')
            logging.info(str(self.get_wcs().sip.a))
            
        # computing distortion maps
        if return_error_maps or return_error_spl:
            logging.info('Computing distortion maps')
            
            sl_cat_pix = self.world2pix(sl_cat_deg[:,:2])
            logging.info('distortion maps computed with {} stars'.format(sl_cat_pix.shape[0]))
            
            # fit based on SIP corrected parameters
            fit = self.fit_stars(
                sl_cat_pix,
                local_background=True,
                multi_fit=False, fix_fwhm=True,
                no_aperture_photometry=True)

            ## remove bad fits            
            fit[np.abs(fit.dx) > self.get_fwhm_pix()] = np.nan
            fit[np.abs(fit.dy) > self.get_fwhm_pix()] = np.nan
            fit = fit.dropna()

            # avoids duplicate of the same star (singular matrix error
            # with RBF)
            for ix in range(len(fit)):
                if np.nansum(np.isclose(fit.x - fit.iloc[ix].x, 0)) > 1:
                    fit.iloc[ix] = np.nan
                     
            fit = fit.dropna()
            _w = 1./fit.x_err.values
            _x = fit.x.values
            _y = fit.y.values
            _dx = fit.dx.values
            _dy = fit.dy.values

            logging.info('distortion maps computed with {} good stars'.format(_x.shape[0]))
            
            dxrbf = interpolate.Rbf(_x, _y, _dx, epsilon=1, function='linear')
            dyrbf = interpolate.Rbf(_x, _y, _dy, epsilon=1, function='linear')

            # RBF models are converted to pixel maps and fitted with
            # Zernike polynomials
            X, Y = np.mgrid[:self.dimx:400j,:self.dimy:400j]
            dxmap = dxrbf(X, Y)
            dymap = dyrbf(X, Y)

            dxmap_fit, dxmap_res, fit_error = orb.utils.image.fit_map_zernike(
                dxmap, np.ones_like(dxmap), 20)
            dymap_fit, dymap_res, fit_error = orb.utils.image.fit_map_zernike(
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

            logging.info('computing full sized dxdymaps (error maps)')
            X = np.arange(self.dimx)
            Y = np.arange(self.dimy)    
            dxmap = dxspl(X, Y, grid=True)
            dymap = dyspl(X, Y, grid=True)
            
            self.set_dxdymaps(dxmap, dymap)
            
        ## COMPUTE PRECISION
        if compute_precision:
            logging.info('Computing astrometrical precision')

            fit_params = self.fit_stars(
                self.world2pix(sl_cat_deg[:,:2]),
                #local_background=True,
                #multi_fit=False, fix_fwhm=False,
                no_aperture_photometry=True)
            
            # results must be filtered for 'jumping' stars
            # when fitted independantly the most brilliant stars in
            # the fit box gets fitted instead of the star at the
            # center of it. 
            dx = get_filtered_params(
                fit_params, param='dx',
                dist_min=self.get_fwhm_pix()*2.)
            dy = get_filtered_params(
                fit_params, param='dy',
                dist_min=self.get_fwhm_pix()*2.)
            x_err = get_filtered_params(
                fit_params, param='x_err',
                dist_min=self.get_fwhm_pix()*2.)
            y_err = get_filtered_params(
                fit_params, param='y_err',
                dist_min=self.get_fwhm_pix()*2.)

            precision = np.sqrt(dx**2. + dy**2.)
            precision_mean = np.sqrt(np.nanmedian(np.abs(dx))**2.
                                     + np.nanmedian(np.abs(dy))**2.)
            precision_mean_err = np.sqrt(
                (np.nanpercentile(dx, 84) - precision_mean)**2.
                 + (np.nanpercentile(dy, 84) - precision_mean)**2.)

            logging.info(
                "Astrometrical precision [in arcsec]: {:.3f} [+/-{:.3f}] computed over {} stars".format(
                    precision_mean * self.get_scale(),
                    precision_mean_err * self.get_scale(), np.size(dx)))

            fit_params['registration_precision_arcsec'] = precision_mean * self.get_scale()
            fit_params['registration_precision_arcsec_err'] = precision_mean_err * self.get_scale()
            fit_params['registration_precision_starnb'] = np.size(dx)

        logging.info('corrected WCS computed')
        logging.info('internal WCS updated (to update the file on disk use self.writeto function)')
        
        if not return_fit_params:
            if return_error_maps:
                return self.get_wcs(), dxmap, dymap
            elif return_error_spl:
                return self.get_wcs(), dxspl, dyspl
            else:
                return self.get_wcs()
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
          
        :param sip_order: (Optional) SIP order (default 2).

        :param crpix: (Optional) If an initial wcs is not given (init_sip
          set to None) this header value must be given.

        :param crval: (Optional) If an initial wcs is not given (init_sip
          set to None) this header value must be given.

        """
        return orb.utils.astrometry.fit_sip(
            self.get_scale(), star_list1, star_list2,
            params=params, init_sip=init_sip, err=err, sip_order=sip_order,
            crpix=crpix, crval=crval)



    def fit_stars(self, star_list, **kwargs):
        """
        Fit stars in one frame.

        This function is basically a wrapper around
        :meth:`orb.utils.astrometry.fit_stars_in_frame`.

        .. note:: 2 fitting modes are possible::
        
            * Individual fit mode [multi_fit=False]
            * Multi fit mode [multi_fit=True]

            see :meth:`orb.utils.astrometry.fit_stars_in_frame` for more
            information.

        :param star_list: Path to a list of stars
          
        :param kwargs: Same optional arguments as for
          :meth:`orb.utils.astrometry.fit_stars_in_frame`.

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
        star_list = orb.utils.astrometry.load_star_list(star_list)
        protected_kwargs =['profile_name', 'scale', 'fwhm_pix', 'beta', 'fit_tol']
        for ik in protected_kwargs:
            if ik in kwargs:
                raise Exception('{} should not be passed in kwargs'.format(ik))
        
        kwargs['profile_name'] = self.params.profile_name
        kwargs['scale'] = self.get_scale()
        kwargs['fwhm_pix'] = self.get_fwhm_pix()
        kwargs['beta'] = self.default_beta
        kwargs['fit_tol'] = self.fit_tol

        # fit
        fit_results = orb.utils.astrometry.fit_stars_in_frame(
            frame, star_list, self.get_box_size(), **kwargs)
       
        return orb.utils.astrometry.fit2df(fit_results)
    
    def compute_alignment_parameters(self, image2, xy_range, r_range,
                                     correct_distortion=False,
                                     star_list1=None, fwhm_arc=None,
                                     brute_force=True, coeffs=(0,0,0,0,0,1),
                                     skip_check=False):
        """Return the alignment coefficients that match the stars of this
        image to the stars of the image 2.

        :param image2: Image instance.

        :param xy_range: Range of x, y values to brute force around
          the initial parameters passed in coeffs. if two ranges are
          given (e.g. (np.linspace(-5,5,10), np.linspace(-1,1,10))) a
          second brute force pass is done centered on the values of
          the first pass.

        :param r_range: Range of angles to brute force around the
          initial parameters passed in coeffs. if two ranges are given
          (e.g. (np.linspace(-5,5,10), np.linspace(-1,1,10))) a second
          brute force pass is done centered on the values of the first
          pass.

        :param correct_distortion: (Optional) If True, a SIP is computed to
          match stars from frame 2 onto the stars from frame 1. But it
          needs a lot of stars to run correctly (default False).

        :param star_list1: (Optional) Path to a list of stars in the
          cube. If given the fwhm_arc must also be set (default None).

        :param fwhm_arc: (Optional) mean FWHM of the stars in
          arcseconds. Must be given if star_list1 is not None
          (default None).

        :param brute_force: (Optional) If True the first step is a
          brute force guess. This is very useful if the initial
          parameters are not well known (default True).

        :param coeffs: (Optional) initial alignement parameters (dx,
          dy, dr, da, db, zoom). If aligning camera 2 on camera 1,
          use self.get_initial_alignment_parameters() to set coeffs.

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
                slin_t = orb.utils.astrometry.transform_star_position_A_to_B(
                    slin, p, rc, zf,
                    sip_A=sip1, sip_B=sip2)
                result = (slin_t - slout).flatten()
                return result[np.nonzero(~np.isnan(result))]

            try:
                fit = optimize.least_squares(
                    diff, p,
                    args=(slin, slout, rc, zf, sip1, sip2),
                    loss='soft_l1')
            except Exception as e:
                raise Exception('No matching parameters found: {}'.format(e))
            
            if fit['status'] > 0:
                if fit['cost'] > 10:
                    logging.warning(
                        'Star lists not well matched (residual {} > 10)'.format(fit['cost']))
                return fit['x']
            
            else:
                raise Exception('No matching parameters found (fit status {})'.format(fit['status']))
            
        def brute_force_alignment(coeffs, xy_range, r_range):
            (coeffs.dx, coeffs.dy, coeffs.dr, guess_matrix) = (
                orb.utils.astrometry.brute_force_guess(
                    image2.data.astype(np.float64), star_list1,
                    xy_range + coeffs.dx,
                    xy_range + coeffs.dy,
                    r_range + coeffs.dr,
                    coeffs.rc, coeffs.zoom,
                    image2.get_fwhm_pix() * 3))
                    #init_wcs=self.get_wcs(), 
                    #out_wcs=image2.get_wcs()))
            coeffs.da = 0.
            coeffs.db = 0.

            return coeffs
            
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

        # Skip fit checking
        SKIP_CHECK = bool(int(self._get_tuning_parameter('SKIP_CHECK', int(skip_check))))

        COEFF_KEYS = 'dx', 'dy', 'dr', 'da', 'db', 'zoom'

        if isinstance(coeffs, dict):
            _coeffs = list()
            for ikey in COEFF_KEYS:
                if ikey in coeffs:
                    _coeffs.append(coeffs[ikey])
                else:
                    raise TypeError('if coeffs is a dict, it must contain {}'.format(ikey))
            coeffs = _coeffs
        
        if len(coeffs) != 6:
            raise TypeError('coeffs must be a tuple (dx, dy, dr, da, db, zoom)')

        if not isinstance(coeffs, orb.core.Params):
            pcoeffs = orb.core.Params()
            pcoeffs.dx = float(coeffs[0])
            pcoeffs.dy = float(coeffs[1])
            pcoeffs.dr = float(coeffs[2])
            pcoeffs.da = float(coeffs[3])
            pcoeffs.db = float(coeffs[4])
            pcoeffs.zoom = float(coeffs[5])
            pcoeffs.rc = [self.dimx/2., self.dimy/2.]
            coeffs = pcoeffs

        
        # check brute force ranges
        if len(xy_range) < 2:
            raise TypeError('xy_range must be an array of more than 5 values or a tuple of 2 arrays each one containing more than 5 values')
        if len(r_range) < 2:
            raise TypeError('r_range must be an array of more than 5 values or a tuple of 2 arrays each one containing more than 5 values')
        
        if len(xy_range) != 2:
            xy_range = (xy_range, None)
        if np.size(xy_range[0]) < 5:
            raise ValueError('xy_range must have a size of minimum 5 values')
        if xy_range[1] is not None:
            if np.size(xy_range[1]) < 5:
                raise ValueError('xy_range must have a size of minimum 5 values')
        
        
        if len(r_range) != 2:
            r_range = (r_range, None)
        if np.size(r_range[0]) < 5:
            raise ValueError('r_range must have a size of minimum 5 values')
        if r_range[1] is not None:
            if np.size(r_range[1]) < 5:
                raise ValueError('r_range must have a size of minimum 5 values')

        if ((xy_range[1] is None and r_range[1] is not None)
            or (xy_range[1] is not None and r_range[1] is None)):
            raise Exception('both xy_range and r_range must have two ranges')
                
        
        if star_list1 is None:
            star_list1, fwhm_pix = self.detect_stars(
                min_star_number=MIN_STAR_NB)
            fwhm_arc = self.pix2arc(fwhm_pix)
        elif fwhm_arc is None:
            raise Exception('If a list of stars is given (star_list1) the fwhm in arcsec (fwhm_arc) must also be given.')

        star_list1 = orb.utils.astrometry.load_star_list(star_list1)
        star_list1_deg = self.pix2world(star_list1)        
        
        image2.reset_fwhm_arc(fwhm_arc)
        self.reset_fwhm_arc(fwhm_arc)
                
        ##########################################
        ### BRUTE FORCE GUESS (only dx and dy) ###
        ##########################################
        if brute_force:
            logging.info("1st brute force pass")
            coeffs = brute_force_alignment(
                coeffs, xy_range[0], r_range[0])
            logging.info("1st brute force guess:") 
            print_alignment_coeffs()

            if xy_range[1] is not None and r_range[1] is not None:
                logging.info("2nd brute force pass")
                coeffs = brute_force_alignment(
                    coeffs, xy_range[1], r_range[1])
                logging.info("2nd brute force guess:") 
                print_alignment_coeffs()

            
        ##########################
        ## FINE ALIGNMENT STEP ###
        ##########################
        guess = [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db]

        # create sip corrected and transformed list
        star_list2 = orb.utils.astrometry.transform_star_position_A_to_B(
            np.copy(star_list1), guess, coeffs.rc, coeffs.zoom,
            sip_A=self.get_wcs(), sip_B=image2.get_wcs())

        fit_results = image2.fit_stars(
            star_list2, no_aperture_photometry=True,
            multi_fit=False, enable_zoom=False,
            enable_rotation=True, fix_fwhm=True)

        if fit_results.empty:
            raise Exception('fit failed. check initial parameters.')

        [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db] = match_star_lists(
            guess, np.copy(star_list1), orb.utils.astrometry.df2list(fit_results),
            coeffs.rc, coeffs.zoom, sip1=self.get_wcs(), sip2=image2.get_wcs())

        logging.info("Fine alignment parameters:") 
        print_alignment_coeffs()


        #####################################
        ### COMPUTE DISTORTION CORRECTION ###
        #####################################
        star_list2 = orb.utils.astrometry.transform_star_position_A_to_B(
            np.copy(star_list1),
            [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db],
            coeffs.rc, coeffs.zoom,
            sip_A=self.get_wcs(), sip_B=image2.get_wcs())

        if correct_distortion:
            logging.info('Computing distortion correction polynomial (SIP)')

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
            sip = self.fit_sip(
                np.copy(star_list2),
                orb.utils.astrometry.df2list(fit_results),
                init_sip=self.get_wcs(),
                crpix=self.get_wcs().wcs.crpix,
                crval=self.get_wcs().wcs.crval)
            self.set_wcs(sip)
            im2wcs = image2.get_wcs()
            im2wcs.sip = None
            image2.set_wcs(im2wcs)

            star_list2 = orb.utils.astrometry.transform_star_position_A_to_B(
                np.copy(star_list1),
                [coeffs.dx, coeffs.dy, coeffs.dr, coeffs.da, coeffs.db],
                coeffs.rc, coeffs.zoom,
                sip_A=self.get_wcs(), sip_B=image2.get_wcs())

        else:
            # fit stars
            fit_results = image2.fit_stars(star_list2)

            fitted_star_nb = float(np.sum(~np.isnan(
                orb.utils.astrometry.df2list(fit_results)[:,0])))
            
            if (fitted_star_nb < ERROR_RATIO * MIN_STAR_NB):
                raise Exception("Not enough fitted stars in both cubes (%d%%). Alignment parameters might be wrong."%int(fitted_star_nb / MIN_STAR_NB * 100.))
                
            if (fitted_star_nb < WARNING_RATIO * MIN_STAR_NB):
                logging.warning("Poor ratio of fitted stars in both cubes (%d%%). Check alignment parameters."%int(fitted_star_nb / MIN_STAR_NB * 100.))

            
            err = fit_results['x_err'].values
            
        
        fwhm_arc2 = orb.utils.stats.robust_mean(
            orb.utils.stats.sigmacut(fit_results['fwhm_arc'].values))
        
        dx_fit = (star_list2[:,0]
                  - orb.utils.astrometry.df2list(fit_results)[:,0])
        dy_fit = (star_list2[:,1]
                  - orb.utils.astrometry.df2list(fit_results)[:,1])
        dr_fit = np.sqrt(dx_fit**2. + dy_fit**2.)
        final_err = np.median(orb.utils.stats.sigmacut(dr_fit))

        if not SKIP_CHECK:
            if final_err < self.arc2pix(WARNING_DIST):
                logging.info('Mean difference on star positions: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
            elif final_err < self.arc2pix(ERROR_DIST):
                logging.warning('Mean difference on star positions is bad: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
            else:
                raise Exception('Mean difference on star positions is too bad: {} pixels = {} arcsec'.format(final_err, self.pix2arc(final_err)))
        

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
                'sip1': self.get_wcs(),
                'sip2': image2.get_wcs(),
                'star_list1': star_list1,
                'star_list2': star_list2,
                'fwhm_arc1': fwhm_arc,
                'fwhm_arc2': fwhm_arc2}


    def transform(self, params, order=1):
        """Return a transformed image.

        :param params: A dictionary containing the keys 'coeffs'=[dx,
          dy, dr, da, db], 'rc', 'zoom_factor'. This is exactly the
          dictionary returned by self.compute_alignment_coeffs(). may
          optionally contain 'sip1' and 'sip2' where sip1 is the sip
          of the image to transform and sip2 is the sip of the image
          it is aligned to.

        :param order: Interpolation order (default 1)
        """
        keys = 'coeffs', 'rc', 'zoom_factor'
        sip1 = None
        sip2 = None
        if 'sip1' in params: sip1 = params['sip1']
        if 'sip2' in params: sip2 = params['sip2']
        
        if not isinstance(params, dict):
            raise TypeError('params must be a dict')
        for key in keys:
            if key not in params:
                raise TypeError('malformed params dict')

        wcsB = orb.utils.astrometry.transform_wcs(
            self.get_wcs(), params['coeffs'], params['rc'], params['zoom_factor'],
            sip=sip2)
        
        data_t = orb.utils.image.transform_frame(
            np.copy(self.data), 0, self.dimx, 0, self.dimy,
            params['coeffs'], params['rc'], params['zoom_factor'],
            order, sip_A=sip1, sip_B=sip2)

        out = self.copy()
        out.data = data_t
        out.set_wcs(wcsB)
        
        return out


    def compute_panstarrs_photometry(self, star_number=1000, max_xmatch_radius=10,
                                     max_roundness=0.5, eps=None, 
                                     modulated=False, calibrated=False):

        STEP_NB = 200
        filter_file = orb.core.FilterFile(self.params.filter_name)
        
        # compute flambda
        if not calibrated:
            photom = orb.photometry.Photometry(
                self.params.filter_name, self.params.camera,
                airmass=self.params.airmass)
            cm1_axis = orb.utils.spectrum.create_cm1_axis(
                STEP_NB, self.params.step, self.params.order,
                orb.utils.spectrum.theta2corr(self.config['OFF_AXIS_ANGLE_CENTER']))
            flam = photom.compute_flambda(cm1_axis, modulated=modulated, eps=eps)
            flam_mean = flam.mean_in_filter() / STEP_NB
        
        # fit stars in image
        starlist, fwhm = self.detect_stars(
            min_star_number=star_number, saturation_threshold=1e99,
            max_roundness=max_roundness)
        starfit = self.fit_stars(starlist)

        # convert measured flux to AB magnitude
        
        if not calibrated:
            starfit['aperture_erg'] = starfit.aperture_flux  / self.params.EXPTIME / self.params.step_nb * flam_mean
        else:
            starfit['aperture_erg'] = starfit.aperture_flux
            
        starfit['aperture_mag'] = orb.utils.photometry.flambda2ABmag(
            starfit.aperture_erg.values,
            filter_file.get_mean_nm() * 10.)

        if not calibrated:
            starfit['flux_erg'] = starfit.flux  / self.params.EXPTIME / self.params.step_nb * flam_mean
        else:
            starfit['flux_erg'] = starfit.flux
            
        starfit['flux_mag'] = orb.utils.photometry.flambda2ABmag(
            starfit.flux_erg.values,
            filter_file.get_mean_nm() * 10.)
        
        # get catalogue
        cat = self.query_vizier(catalog='pan-starrs', as_pandas=True, max_stars=star_number*10)

        # cross-match
        cat['x'], cat['y'] = self.world2pix([cat['ra'], cat['dec']]).T
        starfit['xmatch_radius'] = np.nan
        starfit['xmatch_index'] = 0
        for i in range(len(starfit)):
            try:
                r = (np.sqrt((cat.x.values - starfit.x.values[i])**2
                             + (cat.y.values - starfit.y.values[i])**2))
                starfit['xmatch_index'].values[i] = np.nanargmin(r)
                starfit['xmatch_radius'].values[i] = r[starfit['xmatch_index'].values[i]]
            except ValueError:
                pass
            
        starfit = starfit[starfit['xmatch_radius'] < max_xmatch_radius]

        for ifilter_name in ['g', 'r', 'i', 'z', 'y']:
            starfit['PS1_{}'.format(ifilter_name)] = cat['{}mag'.format(ifilter_name)].values[starfit['xmatch_index'].values]
            
        return starfit
        
#################################################
#### CLASS StandardImage ########################
#################################################

class StandardImage(Image):

    def __init__(self, data, **kwargs):

        Image.__init__(self, data, **kwargs)

        radec = self.find_object(is_standard=True, return_radec=True)
        self.params.reset('target_ra', radec[0])
        self.params.reset('target_dec', radec[1])
        self.target_ra = self.params.target_ra
        self.target_dec = self.params.target_dec
        self.params.reset('object_name', ''.join(
            self.params.OBJECT.strip().split()).upper())
        self.params.reset('exposure_time', self.params.EXPTIME) # must be forced
        self.params.reset('airmass', self.params.AIRMASS) # must be forced
        
        
    def compute_flux_correction_factor(self):
        """Compute the flux correction factor that can be used by
        orb.photometry.Photometry.get_flambda()"""

        std_xy = self.find_object()
        try:
            orb.utils.validate.index(std_xy[0], 0, self.dimx, clip=False)
            orb.utils.validate.index(std_xy[1], 0, self.dimy, clip=False)
        except orb.utils.err.ValidationError:
            raise Exception('standard star not in the image, check image registration')

        star_list, fwhm = self.detect_stars(min_star_number=30) # used to recompute fwhm properly
        std_fit = self.fit_stars([std_xy], aper_coeff=6)
        std_flux_im = std_fit['aperture_flux'].values[0] / self.params.exposure_time

        std = orb.photometry.Standard(self.params.object_name,
                                      instrument=self.instrument)

        ff = orb.core.FilterFile(self.params.filter_name)
        cm1_axis = orb.core.Axis(orb.utils.spectrum.create_cm1_axis(
            1000, ff.params.step, ff.params.order,
            corr=orb.utils.spectrum.theta2corr(
                self.config.OFF_AXIS_ANGLE_CENTER)))

        std_flux_sim = std.simulate_measured_flux(
            self.params.filter_name, cm1_axis,
            camera_index=self.params.camera,
            modulated=False, airmass=self.params.airmass) 

        eps_mean = std_flux_im / np.nansum(std_flux_sim.data)
        return eps_mean
