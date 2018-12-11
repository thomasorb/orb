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

from scipy import optimize, interpolate, signal

import core

import utils.astrometry
import utils.image
import utils.stats
import utils.vector
import utils.web
import utils.misc
import utils.io

##################################################
#### CLASS StarsParams ###########################
##################################################

class StarsParams(core.Tools):
    """StarsParams manage the parameters of each star in each frame of
    a cube.

    It can be accessed as a simple 3D array of shape (star_number,
    frame_nb, parameter).

    Additional methods provide an easy access to the data.
    """

    def __init__(self, star_nb=1, frame_nb=1, **kwargs):
        """StarsParams init

        :param star_nb: (Optional) Number of stars (default 1)
        
        :param frame_nb: (Optional) Number of frames (default 1)

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        core.Tools.__init__(self, **kwargs)

        self.data = None # the whole data

        if isinstance(star_nb, int):
            if star_nb > 0:
                self.star_nb = star_nb
            else:
                raise StandardError("Star number must be > 0")
        else:
            raise StandardError("Type Error: Star number must be an integer")

        if isinstance(frame_nb, int):
            if frame_nb > 0:
                self.frame_nb = frame_nb
            else:
                raise StandardError("Frame number must be > 0")
        else:
            raise StandardError("Type Error: Frame number must be an integer")
                
        self.keys = list()
        self.reset_data()

    def __setitem__(self, key, item):
        """Implement the evaluation of self[key] = item"""

        # Update keys
        if isinstance(item, dict):
            for ikey in item.keys():
                if ikey not in self.keys:
                    self.keys.append(ikey)
                    
        if np.size(key) > 1:
            if isinstance(key[-1], str):
                if key[-1] not in self.keys:
                    self.keys.append(key[-1])
                    
        if isinstance(item, StarsParams):
            for ikey in item.keys:
                if ikey not in self.keys:
                    self.keys.append(ikey)
        
        if np.size(key) not in [1,2,3]:
            raise StandardError("IndexError: invalid index. Use index of 1, 2 or 3 dimensions to access StarsParams data")
            
        if np.size(key) == 1:
            if self.frame_nb == 1:
                if isinstance(item, StarsParams):
                    self.data[key, 0] = item[:,0]
                elif isinstance(item, dict):
                    self.data[key, 0] = item
                elif item is None:
                    self.data[key, 0] = None
                else:
                    raise StandardError("item to set is neither a StarsParams instance or a dict")
            elif self.star_nb == 1:
                if isinstance(item, StarsParams):
                    self.data[0, key] = item[:,0]
                elif isinstance(item, dict):
                    self.data[0, key] = item
                elif item is None:
                    self.data[0, key] = None
                else:
                    raise StandardError("item to set is neither a StarsParams instance or a dict")
            else:
                raise StandardError("IndexError")
            
        elif np.size(key) == 2  and isinstance(key[-1], int):
            if isinstance(item, StarsParams):
                if self.data[key].shape == item.data.shape:
                    self.data[key] = item.data
                elif np.size(self.data[key]) == np.size(item.data):
                    self.data[key] = np.squeeze(item.data)
                else:
                    raise StandardError("Number of elements is not the same") 
            elif isinstance(item, dict):
                self.data[key] = item
            elif item is None:
                self.data[key] = None
            else:
                raise StandardError("item to set is neither a StarsParams instance or a dict")
                
        elif isinstance(key[-1], str):
            data_key = key[:-1]
            param = key[-1]
            if np.size(self.data[data_key]) == 1:
                self.data[data_key][0][param] = item
            else:
                sliced_data = np.atleast_2d(self.data[data_key])
                for ik in range(sliced_data.shape[0]):
                    for jk in range(sliced_data.shape[1]):
                        if sliced_data[ik,jk] is not None:
                            sliced_data[ik,jk][param] = item
                        else:
                            sliced_data[ik,jk] = dict()
                            sliced_data[ik,jk][param] = item
                self.data[data_key] = sliced_data
        else:
            raise StandardError("KeyError %s"%str(key))

    def __getitem__(self, key):
        """Implement the evaluation of self[key]"""
        if np.size(key) not in [1,2,3]:
            raise StandardError("IndexError: invalid index. Use index of 1, 2 or 3 dimensions to access StarsParams data")
        if np.size(key) == 1:
            return np.atleast_1d(np.squeeze(self.data[key]))[0]
        elif np.size(key) == 2  and isinstance(key[-1], int):
            return self.data[key]
        elif isinstance(key[-1], str):
            data_key = key[:-1]
            param = key[-1]
            if np.size(self.data[data_key]) == 1:
                if np.atleast_1d(np.squeeze(self.data[data_key]))[0] is not None:
                    return np.atleast_1d(np.squeeze(self.data[data_key]))[0][param]
                else:
                    return np.nan
            else:
                sliced_data = np.atleast_2d(self.data[data_key])
                ret_data = np.empty_like(sliced_data)
                for ik in range(sliced_data.shape[0]):
                    for jk in range(sliced_data.shape[1]):
                        if sliced_data[ik,jk] is not None:
                            if param in sliced_data[ik,jk]:
                                ret_data[ik,jk] = sliced_data[ik,jk][param]
                            else:
                                ret_data[ik,jk] = np.nan
                        else:
                            ret_data[ik,jk] = np.nan
                return np.squeeze(np.array(ret_data, dtype=float))
        else:
            raise StandardError("KeyError %s"%str(key))

    def __repr__(self):
        """Return a printable version of StarsParams data"""
        return str(self.data)

    def convert(self, data):
        """Convert StarsParams data to a new StarsParams instance.
        """
        # No conversion to do
        if isinstance(data, StarsParams):
            return data
            
        if len(data.shape) <= 2:
            data = np.atleast_2d(data)
            star_nb = data.shape[0]
            frame_nb = data.shape[1]
            new_params = StarsParams(star_nb, frame_nb,
                                     instrument=self.instrument)
            for ik in range(star_nb):
                for jk in range(frame_nb):
                    if isinstance(data[ik,jk], dict):
                        new_params[ik,jk] = data[ik,jk]
                    elif data[ik,jk] is None:
                        new_params[ik,jk] = None
                    else:
                        raise StandardError("Error occuring in data conversion. Data may be corrupted.")
            return new_params
        
    def reset_data(self):
        """Reset StarsParams data to an empty array"""
        self.data = np.empty((self.star_nb, self.frame_nb),
                             dtype=object)
        self.data.fill(dict())

    def save_stars_params(self, params_file_path, group=None):
        """Save stars parameters in one file

        :param params_file_path: Path to the parameters file

        :param group: (Optional) Group path into the HDF5 file. If
          None the group will be '/' (default None).

        .. warning:: if no group is given, the whole file will be
          deleted if it already exists because new parameters will be
          written in the root group.
        """
        start_t = time.time()
        logging.info("Saving stars parameters in {}".format(
            params_file_path))
        
        # data restruct (much faster to save)
        data_r = np.empty((self.star_nb, self.frame_nb, len(self.keys)))
        
        for istar in range(self.star_nb):
            for index in range(self.frame_nb):
                _d = self[istar, index]
                for ikey in range(len(self.keys)):
                    try:
                        data_r[istar, index, ikey] = _d[self.keys[ikey]]
                    except Exception:
                        data_r[istar, index, ikey] = np.nan

        if group is None and os.path.exists(params_file_path):
            os.remove(params_file_path)

        with utils.io.open_hdf5(params_file_path, 'a') as f:

            if group is not None:
                if group in f:
                    del f[group]
                grp = f.create_group(group)
            else:
                grp = f['/']
            
            grp.attrs['star_nb'] = self.star_nb
            grp.attrs['frame_nb'] = self.frame_nb
            grp.attrs['params'] = self.keys
                    
            for ikey in range(len(self.keys)):
                grp[self.keys[ikey]] = data_r[:,:,ikey]
               
        logging.info("Stars parameters saved in {:.2f} s".format(time.time() - start_t))

    def load_stars_params(self, params_file_path, group=None,
                          silent=False):
        """Load a file containing stars parameters.

        :param fit_results_path: Path to the file parameters

        :param silent: (Optional) If True, no message is printed
          (default False).

        :param group: (Optional) Group path into the HDF5 file. If
          None the group will be '/' (default None).
        """
        if not silent:
            logging.info('Loading stars parameters')

        start_t = time.time()
        with utils.io.open_hdf5(params_file_path, 'r') as f:
        
            star_nb = None
            keys = None
            frame_nb = None

            if group is not None:
                grp = f[group]
            else:
                grp = f['/']
            
            # get header params to construct params array
            if 'star_nb' in grp.attrs:
                star_nb = grp.attrs['star_nb']
            if 'params' in grp.attrs:
                keys = grp.attrs['params']
            if 'frame_nb' in grp.attrs:
                frame_nb = grp.attrs['frame_nb']
            if star_nb is None or keys is None or frame_nb is None:
                raise StandardError('Bad star parameters file')

            if self.star_nb is not None:
                if star_nb != self.star_nb:
                    warnings.warn('The number of stars in the loaded stars parameters and in the star list are not the same ! The star list will be erased')
                    self.star_list = None
            self.star_nb = star_nb

            if frame_nb != self.frame_nb:
                warnings.warn('The number of frames in the loaded stars parameters and in the cube are not the same ! The loaded data will be removed')
                self.data = None
            self.frame_nb = frame_nb

            # reset data
            self.reset_data()

            # load data (much faster if data is 'preloaded')
            dat = dict()
            for ikey in keys:
                dat[ikey] = grp[ikey][:]
                
            # fill data
            for index in range(self.frame_nb):
                for istar in range(self.star_nb):
                    params = dict()
                    for ikey in keys:
                        params[ikey] = dat[ikey][istar, index]
                        self.data[istar, index] = params

            if not silent:
                logging.info('Stars parameters loaded in {:.2f} s'.format(time.time() - start_t))
            

    def get_star_list(self, index=0, all_params=False):
        """Return an array of the positions of the stars in one frame

        :param index: (Optional) The index of the frame to use to
          create the list of stars (default 0)

        :param all_params: (Optional) If True, all params are
          returned. Bad params are returned as NaN values.
        """
        star_list = list()
        for istar in range(self.star_nb):
            if self[istar, index] is not None:
                if 'x' in self[istar, index] and 'y' in self[istar, index]:
                    if (not np.isnan(self[istar,index,'x'])
                        and not np.isnan(self[istar,index,'y'])):
                        star_list.append([self[istar,index,'x'],
                                          self[istar,index,'y']])
                    elif all_params:
                        star_list.append([np.nan, np.nan])
                elif all_params:
                    star_list.append([np.nan, np.nan])
            elif all_params:
                    star_list.append([np.nan, np.nan])
            
        return np.array(star_list)






#################################################
#### CLASS Image ################################
#################################################

class Image(core.Frame2D, core.Tools):

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
        
        core.Frame2D.__init__(self, data, **kwargs)

        self.params['instrument'] = instrument

        # load wcs
        self.wcs = pywcs.WCS(self.params)

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
        return self._data_path_hdr + "star_list"

    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results.hdf5"

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
        self.fit_results = StarsParams(self.star_nb, 1,
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

    def fit_stars(self, save=False, **kwargs):
        """
        Fit stars in one frame.

        This function is basically a wrapper around
        :meth:`utils.astrometry.fit_stars_in_frame`.

        .. note:: 2 fitting modes are possible::
        
            * Individual fit mode [multi_fit=False]
            * Multi fit mode [multi_fit=True]

            see :meth:`utils.astrometry.fit_stars_in_frame` for more
            information.

        :param save: (Optional) If True save the fit results in a file
          (default True).
          
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
        if self.star_list is None: raise StandardError(
            "A star list must be loaded or created first")

        frame = np.copy(self.data)        

        kwargs['profile_name'] = self.profile_name
        kwargs['scale'] = self.scale
        kwargs['fwhm_pix'] = self.fwhm_pix
        kwargs['beta'] = self.default_beta
        kwargs['fit_tol'] = self.fit_tol

        fit_results = StarsParams(star_nb=len(self.star_list), frame_nb=1)

        # fit
        _fit_results = utils.astrometry.fit_stars_in_frame(
            frame, self.star_list, self.box_size, **kwargs)
        

        # convert results to a StarParams instance
        for istar in range(len(self.star_list)):
            fit_results[istar] = _fit_results[istar]

        self.fit_results = fit_results
                
        if save:
            self.fit_results.save_stars_params(self._get_fit_results_path())

        return fit_results


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

        :param saturation_threshold: (Optional) Number of counts above
          which the star can be considered as saturated. Very low by
          default because at the ZPD the intensity of a star can be
          twice the intensity far from it (default 35000).

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
        
        logging.info('Corrected WCS computed')
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
