#!/usr/bin/env python
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: astrometry.py

## Copyright (c) 2010-2016 Thomas Martin <thomas.martin.1@ulaval.ca>
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

import numpy as np
from scipy import optimize, interpolate, signal
from scipy import interpolate
import astropy.wcs as pywcs
import bottleneck as bn

import core
__version__ = core.__version__
from core import Tools, Cube, ProgressBar
import utils.astrometry
import utils.image
import utils.stats
import utils.vector
import utils.web
import cutils
import utils.misc

##################################################
#### CLASS StarsParams ###########################
##################################################

class StarsParams(Tools):
    """StarsParams manage the parameters of each star in each frame of
    a cube.

    It can be accessed as a simple 3D array of shape (star_number,
    frame_nb, parameter).

    Additional methods provide an easy access to the data.
    """
    data = None # the whole data
    star_nb = None # Number of stars
    frame_nb = None # Number of frame
    keys = None # Tuple containing the keys of the parameters
    

    def __init__(self, star_nb=1, frame_nb=1, **kwargs):
        """StarsParams init

        :param star_nb: (Optional) Number of stars (default 1)
        
        :param frame_nb: (Optional) Number of frames (default 1)

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
              
        if isinstance(star_nb, int):
            if star_nb > 0:
                self.star_nb = star_nb
            else:
                self._print_error("Star number must be > 0")
        else:
            self._print_error("Type Error: Star number must be an integer")

        if isinstance(frame_nb, int):
            if frame_nb > 0:
                self.frame_nb = frame_nb
            else:
                self._print_error("Frame number must be > 0")
        else:
            self._print_error("Type Error: Frame number must be an integer")
                
        self.keys = list()
        self.reset_data()
        self._msg_class_hdr = self._get_msg_class_hdr()

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
            self._print_error("IndexError: invalid index. Use index of 1, 2 or 3 dimensions to access StarsParams data")
            
        if np.size(key) == 1:
            if self.frame_nb == 1:
                if isinstance(item, StarsParams):
                    self.data[key, 0] = item[:,0]
                elif isinstance(item, dict):
                    self.data[key, 0] = item
                elif item is None:
                    self.data[key, 0] = None
                else:
                    self._print_error("item to set is neither a StarsParams instance or a dict")
            elif self.star_nb == 1:
                if isinstance(item, StarsParams):
                    self.data[0, key] = item[:,0]
                elif isinstance(item, dict):
                    self.data[0, key] = item
                elif item is None:
                    self.data[0, key] = None
                else:
                    self._print_error("item to set is neither a StarsParams instance or a dict")
            else:
                self._print_error("IndexError")
            
        elif np.size(key) == 2  and isinstance(key[-1], int):
            if isinstance(item, StarsParams):
                if self.data[key].shape == item.data.shape:
                    self.data[key] = item.data
                elif np.size(self.data[key]) == np.size(item.data):
                    self.data[key] = np.squeeze(item.data)
                else:
                    self._print_error("Number of elements is not the same") 
            elif isinstance(item, dict):
                self.data[key] = item
            elif item is None:
                self.data[key] = None
            else:
                self._print_error("item to set is neither a StarsParams instance or a dict")
                
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
            self._print_error("KeyError %s"%str(key))

    def __getitem__(self, key):
        """Implement the evaluation of self[key]"""
        if np.size(key) not in [1,2,3]:
            self._print_error("IndexError: invalid index. Use index of 1, 2 or 3 dimensions to access StarsParams data")
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
            self._print_error("KeyError %s"%str(key))

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
                                     silent=self._silent,
                                     config_file_name=self.config_file_name)
            for ik in range(star_nb):
                for jk in range(frame_nb):
                    if isinstance(data[ik,jk], dict):
                        new_params[ik,jk] = data[ik,jk]
                    elif data[ik,jk] is None:
                        new_params[ik,jk] = None
                    else:
                        self._print_error("Error occuring in data conversion. Data may be corrupted.")
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
        self._print_msg("Saving stars parameters in {}".format(
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

        with self.open_hdf5(params_file_path, 'a') as f:

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
               
        self._print_msg("Stars parameters saved in {:.2f} s".format(time.time() - start_t))

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
            self._print_msg('Loading stars parameters', color=True)

        start_t = time.time()
        with self.open_hdf5(params_file_path, 'r') as f:
        
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
                self._print_error('Bad star parameters file')

            if self.star_nb is not None:
                if star_nb != self.star_nb:
                    self._print_warning('The number of stars in the loaded stars parameters and in the star list are not the same ! The star list will be erased')
                    self.star_list = None
            self.star_nb = star_nb

            if frame_nb != self.frame_nb:
                self._print_warning('The number of frames in the loaded stars parameters and in the cube are not the same ! The loaded data will be removed')
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
                self._print_msg('Stars parameters loaded in {:.2f} s'.format(time.time() - start_t))
            

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
    DETECT_INDEX = 0

    BIG_DATA = None # If True some processes are parallelized
    default_beta = None # Default value of the beta parameter of the
                        # Moffat profile

    data = None # data on which the astrometry is done
    
    master_frame = None # Master frame created from a combination of
                        # the frames of the cube
    
    fwhm_arc = None # Initial guess of the FWHM in arcsec
    fwhm_pix = None # Initial guess of the FWHM in pixels
    
    fov = None # Field of view of the cube in arcminutes (given along
               # x axis)

    scale = None # Scale of the frame in arcsec/pixel
    
    box_size_coeff = None # Coefficient used to define the size of the
                          # box from FWHM
    box_size = None # Size of the fitting box in pixel

    profiles = ['moffat', 'gaussian']
    profile_name = None # Name of the profile
    profile = None # A PSF class (e.g. Moffat or Gaussian)

    star_list = None # List of stars
    star_nb = None # Number of stars in the star list
    frame_nb = None # Number of frames in the data cube
    _silent = False # If True only warnings and error message will be
                    # printed
    deep_frame = None # computed deep frame

    target_x = None # X position of a target
    target_y = None # Y position of a target
    target_ra = None # RA of a target in degrees
    target_dec = None # DEC of a target in degrees
    wcs_rotation = None # Rotation angle of the field relatively to the North.
    wcs = None # When data is registered this pywcs.WCS instance gives
               # the corrected WCS.
    
    detect_stack = None
    """Number of frames stacked together to create one median frame
    used for star detection"""

    fit_results = None
    """Array containing all the resulting parameters (as dictionaries)
    of the fit of each star in each frame"""

    fit_tol = None
    """Tolerance on the fit parameters"""

    reduced_chi_square_limit = None
    """Coefficient on the reduced chi square for bad quality fits
    rejection"""

    _check_mask = None
    """If True use masked frame to reject stars containing masked
    pixels"""

    readout_noise = None
    """Readout noise in ADU/pixel (can be computed from bias frames:
    std(master_bias_frame))"""
    
    dark_current_level = None
    """Dark current level in ADU/pixel (can be computed from dark frames:
    median(master_dark_frame))"""

    

    
    def __init__(self, data, fwhm_arc, fov, profile_name='gaussian',
                 detect_stack=5, fit_tol=1e-2, moffat_beta=2.1,
                 star_list_path=None, box_size_coeff=7.,
                 check_mask=True, reduced_chi_square_limit=1.5,
                 readout_noise=10., dark_current_level=0.,
                 target_radec=None, target_xy=None, wcs_rotation=None,
                 sip=None, **kwargs):

        """
        Init astrometry class.

        :param data: Can be an 2D or 3D Numpy array or an instance of
          core.Cube class. Note that the frames must not be too
          disaligned (a few pixels in both directions).

        :param fwhm_arc: rough FWHM of the stars in arcsec
        
        :param fov: Field of view of the frame in arcminutes (given
          along x axis)

        :param profile_name: (Optional) Name of the PSF profile to use
          for fitting. Can be 'moffat' or 'gaussian' (default
          'gaussian').

        :param detect_stack: (Optional) Number of frames to stack
          before detecting stars (default 5).

        :param fit_tol: (Optional) Tolerance on the paramaters fit
          (the lower the better but the longer too) (default 1e-2).

        :param moffat_beta: (Optional) Default value of the beta
          parameter for the moffat profile (default 3.5).

        :param star_list_path: (Optional) Path to a file containing a
          list of star positions (default None).

        :param box_size_coeff: (Optional) Coefficient giving the size
          of the box created around each star before a fit is
          done. BOX_SIZE = box_size_coeff * STAR_FWHM. (default 10.).
          Note that this coeff is divided by 3 if the moffat profile
          is used (helps to avoid bad fit).
        
        :param check_mask: (Optional) If True and if the data frames
          are masked, masked pixels in a star are considered as bad so
          that no fit is done over it (returned fit parameters for the
          star are None). This is valid only if data is an instance of
          core.Cube() (default True).

        :param reduced_chi_square_limit: (Optional) Coefficient on the
          reduced chi square for bad quality fits rejection (default
          1.5)

        :param readout_noise: (Optional) Readout noise in ADU/pixel
          (can be computed from bias frames: std(master_bias_frame))
          (default 10.)
    
        :param dark_current_level: (Optional) Dark current level in
          ADU/pixel (can be computed from dark frames:
          median(master_dark_frame)) (default 0.)

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
        """
        Tools.__init__(self, **kwargs)
       
        self.BIG_DATA = bool(int(self._get_config_parameter("BIG_DATA")))
        
        # check mask or not
        self._check_mask = check_mask
        
        # load data and init parameters
        if isinstance(data, Cube):
            self.data = data
            self.dimx = self.data.dimx
            self.dimy = self.data.dimy
            self.dimz = self.data.dimz
        elif isinstance(data, np.ndarray):
            self._check_mask = False
            if len(data.shape) == 2 or len(data.shape) == 3:
                self.data = data
                self.dimx = self.data.shape[0]
                self.dimy = self.data.shape[1]
                if len(data.shape) == 3:
                    self.dimz = self.data.shape[2]
                else:
                    self.dimz = 1
            else:
               self._print_error("Data array must have 2 or 3 dimensions") 
        else:
            self._print_error("Cube must be an instance of Cube class or a Numpy array")


        if profile_name == 'moffat':
            self.box_size_coeff = box_size_coeff / 3.
        else:
            self.box_size_coeff = box_size_coeff
            
        self.frame_nb = self.dimz
        
        self.fwhm_arc = float(fwhm_arc)
        self.detect_stack = detect_stack
        self.default_beta = moffat_beta

        # load star list
        if star_list_path is not None:
            self.load_star_list(star_list_path)
        
        # set scale and all dependant parameters
        self.fov = fov
        self.reset_scale(float(self.fov) * 60. / self.dimx)

        self.fit_tol=fit_tol
    
        # define profile
        self.reset_profile_name(profile_name)

        self.reduced_chi_square_limit = reduced_chi_square_limit

        # get noise values
        self.readout_noise = readout_noise
        self.dark_current_level = dark_current_level


        # get RADEC and XY position of a target
        if target_radec is not None:
            self.target_ra = target_radec[0]
            self.target_dec = target_radec[1]
        else:
            self.target_ra = None
            self.target_dec = None
        if target_xy is not None:
            self.target_x = target_xy[0]
            self.target_y = target_xy[1]
        else:
            self.target_x = None
            self.target_y = None
        self.wcs_rotation = wcs_rotation

        self.sip = None
        if sip is not None:
            if isinstance(sip, pywcs.WCS):
                self.sip = sip
            else:
                self._print_error('sip must be an astropy.wcs.WCS instance')
                
    
    def _get_star_list_path(self):
        """Return the default path to the star list file."""
        return self._data_path_hdr + "star_list"

    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results.hdf5"

    def _get_combined_frame(self, use_deep_frame=False, realign=False):
        """Return a combined frame to work on.

        :param use_deep_frame: (Optional) If True returned frame is a
          deep frame instead of a combination of the first frames only
          (default False)

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.
        """
        if self.deep_frame is not None:
            return np.copy(self.deep_frame)

        if self.dimz > 1:
            _cube = self.data[:,:,:]

        # realignment of the frames if necessary
        if realign and self.dimz > 1:
            _cube = utils.astrometry.realign_images(_cube)
                
        # If we have 3D data we work on a combined image of the first
        # frames
        if self.dimz > 1:
            if use_deep_frame:
                if _cube is None:
                    self.deep_frame = self.data.get_median_image().astype(float)
                else:
                    self.deep_frame = np.nanmedian(_cube, axis=2)
                return self.deep_frame
            
            
            stack_nb = self.detect_stack
            if stack_nb + self.DETECT_INDEX > self.frame_nb:
                stack_nb = self.frame_nb - self.DETECT_INDEX

            if _cube is None: dat = _cube
            else: dat = self.data[
                :,:, int(self.DETECT_INDEX):
                int(self.DETECT_INDEX+stack_nb)]
                
            if not self.BIG_DATA:
                im = utils.image.create_master_frame(dat)
            else:
                im = utils.image.pp_create_master_frame(dat)
                
        # else we just return the only frame we have
        else:
            im = np.copy(self.data)
        return im.astype(float)

    def reset_profile_name(self, profile_name):
        """Reset the name of the profile used.

        :param profile_name: Name of the PSF profile to use for
          fitting. Can be 'moffat' or 'gaussian'.
        """
        if profile_name in self.profiles:
            self.profile_name = profile_name
            self.profile = utils.astrometry.get_profile(self.profile_name)
        else:
            self._print_error(
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
                self._print_error('Incorrect star list shape. The star list must be an array of shape (star_nb, 2)')
        else:
            self._print_error('Incorrect star list shape. The star list must be an array of shape (star_nb, 2)')
            
        self.star_list = star_list
        self.star_nb = self.star_list.shape[0]
        # create an empty StarsParams array
        self.fit_results = StarsParams(self.star_nb, self.frame_nb,
                                       silent=self._silent,
                                       config_file_name=self.config_file_name)
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
        self.box_size = int(math.ceil(self.box_size_coeff *  self.fwhm_pix))
        self.box_size += int(~self.box_size%2) # make it odd
    
    def arc2pix(self, x):
        """Convert pixels to arcseconds

        :param x: a value or a vector in pixel
        """
        if self.scale is not None:
            return np.array(x).astype(float) / self.scale
        else:
            self._print_error("Scale not defined")

    def pix2arc(self, x):
        """Convert arcseconds to pixels

        :param x: a value or a vector in arcsec
        """
        if self.scale is not None:
            return np.array(x).astype(float) * self.scale
        else:
            self._print_error("Scale not defined")

    def fwhm(self, x):
        """Return fwhm from width

        :param x: width
        """
        return x * abs(2.*math.sqrt(2. * math.log(2.)))

    def width(self, x):
        """Return width from fwhm

        :param x: fwhm.
        """
        return x / abs(2.*math.sqrt(2. * math.log(2.)))


    def set_deep_frame(self, deep_frame_path):
        deep_frame = self.read_fits(deep_frame_path)
        if deep_frame.shape == (self.dimx, self.dimy):
            self.deep_frame = deep_frame
        else:
            self._print_error('Deep frame must have the same shape')

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
            star_list_path, silent=self._silent)
        
        self.reset_star_list(np.array(star_list, dtype=float))
        
        return self.star_list

    def fit_stars_in_frame(self, index, save=False, **kwargs):
        """
        Fit stars in one frame.

        This function is basically a wrapper around
        :meth:`utils.astrometry.fit_stars_in_frame`.

        .. note:: 2 fitting modes are possible::
        
            * Individual fit mode [multi_fit=False]
            * Multi fit mode [multi_fit=True]

            see :meth:`utils.astrometry.fit_stars_in_frame` for more
            information.

        :param index: Index of the frame to fit. If index is a frame,
          this frame is used instead.

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
            * readout_noise
            * dark_current_level
          
        """
        print
        if self.data is None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list is None: self._print_error(
            "A star list must be loaded or created first")

        if isinstance(index, np.ndarray):
            if index.shape == (self.dimx, self.dimy):
                frame = np.copy(index)
            else:
                self._print_error('Image shape {} must have the same size as the cube size ({},{})'.format(index.shape, self.dimx, self.dimy))

        else:
            if self.dimz > 1:
                frame = self.data[:,:,index]
                if self._check_mask:
                    if self.data._mask_exists:
                        mask = self.data.get_data_frame(index, mask=True)
                        frame = frame.astype(float)
                        frame[np.nonzero(mask)] = np.nan
            else:
                frame = np.copy(self.data)
                

        kwargs['profile_name'] = self.profile_name
        kwargs['scale'] = self.scale
        kwargs['fwhm_pix'] = self.fwhm_pix
        kwargs['beta'] = self.default_beta
        kwargs['fit_tol'] = self.fit_tol
        kwargs['readout_noise'] = self.readout_noise,
        kwargs['dark_current_level'] = self.dark_current_level

        fit_results = StarsParams(star_nb=len(self.star_list), frame_nb=1)

        # fit
        _fit_results = utils.astrometry.fit_stars_in_frame(
            frame, self.star_list, self.box_size, **kwargs)
        

        # convert results to a StarParams instance
        for istar in range(len(self.star_list)):
            fit_results[istar] = _fit_results[istar]

        if not isinstance(index, np.ndarray):
            if self.dimz > 1:
                self.fit_results[:,index] = fit_results
            else:
                self.fit_results = fit_results
        
        if save:
            self.fit_results.save_stars_params(self._get_fit_results_path())

        return fit_results
            


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
        
        self._print_msg("Fitting stars in cube", color=True)

        if self.data is None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list is None: self._print_error(
            "A star list must be loaded or created first")

        if self.dimz < 2: self._print_error(
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
                    self._print_error('Added cube must be a Cube instance. Check add_cube option')
                if np.size(added_cube_scale) != 1:
                    self._print_error('Bad added cube scale. Check add_cube option.')
                
        self.fit_results = StarsParams(self.star_nb, self.frame_nb,
                                       silent=self._silent,
                                       config_file_name=self.config_file_name)

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
        
                # check mask
                if self._check_mask:
                    if self.data._mask_exists:
                        mask = self.data.get_data_frame(ik+ijob, mask=True)
                        frame = frame.astype(float)
                        frame[np.nonzero(mask)] = np.nan
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
                      self.readout_noise, self.dark_current_level,
                      local_background, no_aperture_photometry,
                      precise_guess,
                      aper_coeff, blur, no_fit, estimate_local_noise,
                      multi_fit, enable_zoom, enable_rotation, saturation),
                modules=("import orb.utils.stats",
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
        
        self._print_msg("Mean reduced chi-square: %f"%mean_red_chi_square)
        
        return self.fit_results



    def detect_stars_from_catalogue(self, min_star_number=4, no_save=False,
                                    saturation_threshold=35000, realign=False):
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

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.


        :return: (star_list_path, mean_fwhm_arc) : (a path to a list
          of the dected stars, the mean FWHM of the stars in arcsec)
        """

        LIMIT_RADIUS_RATIO = 1.0 # radius ratio around the center of
                                 # the frame where the stars are kept
  
        self._print_msg("Detecting stars from catalogue", color=True)
        # during registration a star list compted from the catalogue
        # is created.
        self.register()

        deep_frame = self._get_combined_frame(realign=realign)
        fit_params = self.fit_stars_in_frame(deep_frame, multi_fit=False,
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
            if (math.sqrt((istar[0] - rcx)**2. + (istar[1] - rcy)**2.)
                < LIMIT_RADIUS_RATIO * min(rcx, rcy))]

        # keep the brightest stars only
        fitted_star_list.sort(key=lambda star: star[3], reverse=True)
        
        star_list = np.array(fitted_star_list)[:min_star_number,:2]
        snr_list = snr_list[:min_star_number]
        
        # write down detected stars
        mean_fwhm = self.fwhm_arc
        star_list_file = self.open_file(self._get_star_list_path())
        for istar in star_list:
            star_list_file.write(str(istar[0]) + " " + str(istar[1]) + "\n")

        # Print some comments and check number of detected stars    
        self._print_msg("%d stars detected" %(len(star_list)))
        self._print_msg("Detected stars FWHM : %f pixels, %f arc-seconds"%(
            mean_fwhm, self.pix2arc(mean_fwhm)))
        snr_list = np.array(snr_list)
        self._print_msg("SNR Min: %.1e, Max:%.1e, Median:%.1e"%(
            np.min(snr_list), np.max(snr_list), np.median(snr_list)))
        
        if len(star_list) < min_star_number:
            self._print_warning(
                "Not enough detected stars in the image : %d/%d"%(
                    len(star_list), min_star_number))
        if len(star_list) < 4:
            self._print_error(
                "Not enough detected stars: %d < 4"%len(star_list))

        self.reset_star_list(star_list)
        
        return self._get_star_list_path(), self.pix2arc(mean_fwhm)


    def detect_all_sources(self, use_deep_frame=False, realign=False):
        """Detect all point sources in the cube regardless of there FWHM.

        Galaxies, HII regions, filamentary knots and stars might be
        detected.

        :param use_deep_frame: (Optional) If True a deep frame of the
          cube is used instead of combinig only the first frames
          (default False).

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.   
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
        
        self._print_msg("Detecting all point sources in the cube", color=True)

        im = self._get_combined_frame(use_deep_frame=use_deep_frame,
                                      realign=realign)
        
        start_time = time.time()
        self._print_msg("Filtering master image")
        hp_im = utils.image.high_pass_diff_image_filter(im, deg=1)
        self._print_msg("Master image filtered in {} s".format(
            time.time() - start_time))


        # image is binned to help detection
        binning = int(self.fwhm_pix) + 1        
        hp_im = utils.image.nanbin_image(hp_im, binning)

        # detect all pixels above the sky theshold
        detected_pixels = np.nonzero(
            hp_im > 4. * np.nanstd(utils.stats.sigmacut(
                hp_im)))
        
        self._print_msg('{} detected pixels'.format(len(detected_pixels[0])))
        
        # pixels aggregation in sources
        self._print_msg('aggregating detected pixels')
        sources = aggregate(detected_pixels, SOURCE_SIZE)
            
        
        self._print_msg('{} sources detected'.format(len(sources)))
        
        star_list_file = self.open_file(self._get_star_list_path())
        for isource in sources:
            star_list_file.write('{} {}\n'.format(
                isource[0]*binning, isource[1]*binning))
            
        self.reset_star_list(sources)
        
        return self._get_star_list_path(), self.fwhm_arc
       

    def detect_stars(self, min_star_number=4, no_save=False,
                     saturation_threshold=35000, try_catalogue=False,
                     use_deep_frame=False, r_max_coeff=0.6,
                     filter_image=True, realign=False):
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

        :param use_deep_frame: (Optional) If True a deep frame of the
          cube is used instead of combining only the first frames
          (default False).

        :param r_max_coeff: (Optional) Coefficient that sets the limit
          radius of the stars (default 0.6).

        :param filter_image: (Optional) If True, image is filtered
          before detection to remove nebulosities (default True).

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.

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
         

        self._print_msg("Detecting stars", color=True)
        im = self._get_combined_frame(use_deep_frame=use_deep_frame,
                                      realign=realign)

        # high pass filtering of the image to remove nebulosities
        if filter_image:
            start_time = time.time()
            self._print_msg("Filtering master image")
            hp_im = utils.image.high_pass_diff_image_filter(im, deg=1)
            self._print_msg("Master image filtered in {} s".format(
                time.time() - start_time))
        else:
            hp_im = np.copy(im)

        # preselection
        self._print_msg("Stars preselection")
        mean_hp_im = np.nanmean(hp_im)
        std_hp_im = np.nanstd(hp_im)
        max_im = np.nanmax(im)
        # +1 is just here to make sure we enter the loop
        star_number = PRE_DETECT_COEFF * min_star_number + 1 
        
        old_star_list = []
        while(star_number > PRE_DETECT_COEFF * min_star_number):
            pre_star_list = np.array(np.nonzero(
                (hp_im > mean_hp_im + THRESHOLD_COEFF * std_hp_im)
                * (im < saturation_threshold)))
            star_list = list()
            for istar in range(pre_star_list.shape[1]):
                ix = pre_star_list[0, istar]
                iy = pre_star_list[1, istar]
                (box, mins)  = define_box(ix,iy,self.box_size,im)
                ilevel = im[ix, iy]
                if (ilevel == np.max(box)) and (ilevel <= max_im):
                    # filter stars too far from the center
                    cx, cy = self.dimx/2., self.dimy/2.
                    r_max = math.sqrt(cx**2. + cy**2.) * r_max_coeff
                    if math.sqrt((ix - cx)**2. + (iy - cy)**2.) <= r_max:
                        star_list.append([ix, iy])
                    
            star_number = np.array(star_list).shape[0]
            if star_number > PRE_DETECT_COEFF * min_star_number:
                old_star_list = star_list
            THRESHOLD_COEFF += 0.1
        if old_star_list != []:
            star_list = old_star_list
        else: 
            if star_number < min_star_number:
                self._print_warning(
                    "Not enough detected stars in the image : %d"%star_number)
        
        ### FIT POSSIBLE STARS ############
                
        # first fit test to eliminate "bad" stars
        
        self._print_msg("Bad stars rejection based on fitting")
        
        params_list = list()
        
        job_server, ncpus = self._init_pp_server()
        
        progress = ProgressBar(len(star_list), silent=self._silent)
        for istar in range(0, len(star_list), ncpus):
            
            if istar + ncpus >= len(star_list):
                ncpus = len(star_list) - istar

            jobs = [(ijob, job_server.submit(
                test_fit, 
                args=(define_box(star_list[istar+ijob][0],
                                 star_list[istar+ijob][1],
                                 self.box_size, im)[0],
                      define_box(star_list[istar+ijob][0],
                                 star_list[istar+ijob][1],
                                 self.box_size, im)[1],
                      self.profile_name, self.fwhm_pix,
                      self.default_beta, self.fit_tol, MIN_FWHM_COEFF,
                      saturation_threshold, self.profile),
                modules=("import numpy as np",
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
        self._print_msg("Fit check")
        
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
            self._print_error("All detected stars have been rejected !")

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
        star_list_file = self.open_file(self._get_star_list_path())
        for istar in star_list:
            star_list_file.write(str(istar[0]) + " " + str(istar[1]) + "\n")

        # Print some comments and check number of detected stars    
        self._print_msg("%d stars detected" %(len(star_list)))
        self._print_msg("Detected stars FWHM : %f pixels, %f arc-seconds"%(
            mean_fwhm, self.pix2arc(mean_fwhm)))
        snr_list = np.array(snr_list)
        self._print_msg("SNR Min: %.1e, Max:%.1e, Median:%.1e"%(
            np.min(snr_list), np.max(snr_list), np.median(snr_list)))
        
        if len(star_list) < min_star_number:
            self._print_warning(
                "Not enough detected stars in the image : %d/%d"%(
                    len(star_list), min_star_number))
        if len(star_list) < 4:
            self._print_error(
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
        
        if self.data is None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list is None: self._print_error(
            "A star list must be loaded or created first")
    
        if fit_cube:
            self.fit_stars_in_cube(correct_alignment=True,
                                   no_aperture_photometry=True,
                                   hpfilter=HPFILTER, multi_fit=True,
                                   fix_height=False, save=False)
      
        if self.star_nb < 4: 
            self._print_error("Not enough stars to align properly : %d (must be >= 3)"%self.star_nb)
            
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
            self._print_error("Not enough detected stars (%d) in the first frame"%good_nb)

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
        self._print_msg(
            'Alignment vectors median error: %f pixel'%utils.stats.robust_median(alignment_error))
                
        return alignment_vector_x, alignment_vector_y, alignment_error

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
        radius = self.fov / math.sqrt(2)
        if self.target_ra is None or self.target_dec is None:
            self._print_error('No catalogue query can be done. Please make sure to give target_radec and target_xy parameter at class init')
        
        return utils.web.query_vizier(
            radius, self.target_ra, self.target_dec,
            catalog=catalog, max_stars=max_stars)


    def register(self, max_stars_detect=60,
                 full_deep_frame=False,
                 return_fit_params=False, rscale_coeff=1.,
                 compute_precision=True, compute_distortion=False,
                 realign=False, return_error_maps=False,
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
          
        :param full_deep_frame: (Optional) If True all the frames of
          the cube are used to create a deep frame. Use it only when
          the frames in the cube are aligned. In the other case only
          the first frames are combined together (default False).

        :param return_fit_params: (Optional) If True return final fit
          parameters instead of wcs (default False).

        :param rscale_coeff: (Optional) Coefficient on the maximum
          radius of the fitted stars to compute scale. When rscale_coeff
          = 1, rmax is half the longer side of the image (default 1).

        :param compute_distortion: (Optional) If True, optical
          distortion (SIP) are computed. Note that a frame with a lot
          of stars is better for this purpose (default False).

        :param compute_precision: (Optional) If True, astrometrical
          precision is computed (default True).

        :param realign: (Optional) Realign frames with a
          cross-correlation algorithm (default False). Much better if
          used on a small number of frames.

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
                r = math.sqrt((posx - self.dimx/2.)**2.
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
            self._print_error("Not enough parameters to register data. Please set target_xy, target_radec and wcs_rotation parameters at Astrometry init")

        if return_error_maps and return_error_spl: self._print_error('return_error_maps and return_error_spl cannot be both set to True, choose one of them')
        
        self._print_msg('Computing WCS', color=True)

        self._print_msg("Initial scale: {} arcsec/pixel".format(self.scale))
        self._print_msg("Initial rotation: {} degrees".format(self.wcs_rotation))
        self._print_msg("Initial target position in the image (X,Y): {} {}".format(
            self.target_x, self.target_y))

        # get deep frame
        deep_frame = self._get_combined_frame(
            use_deep_frame=full_deep_frame, realign=realign)
        ## utils.io.write_fits('deep_frame.fits', deep_frame, overwrite=True)

        deltax = self.scale / 3600. # arcdeg per pixel
        deltay = float(deltax)

        # get FWHM
        star_list_fit_init_path, fwhm_arc = self.detect_stars(
            min_star_number=max_stars_detect,
            use_deep_frame=full_deep_frame)
        star_list_fit_init = self.load_star_list(star_list_fit_init_path)
        ## star_list_fit_init = self.load_star_list(
        ##     './temp/data.Astrometry.star_list)'
        ## fwhm_arc= 1.
        
        self.box_size_coeff = 5.
        self.reset_fwhm_arc(fwhm_arc)

        # clean deep frame by keeping only the pixels around the
        # detected stars to avoid strong saturated stars.
        deep_frame_corr = np.empty_like(deep_frame)
        deep_frame_corr.fill(np.nan)
        for istar in range(star_list_fit_init.shape[0]):
            x_min, x_max, y_min, y_max = utils.image.get_box_coords(
                star_list_fit_init[istar, 0],
                star_list_fit_init[istar, 1],
                self.fwhm_pix*7,
                0, deep_frame.shape[0],
                0, deep_frame.shape[1])
            deep_frame_corr[x_min:x_max,
                            y_min:y_max] = deep_frame[x_min:x_max,
                                                      y_min:y_max]
        
        # Query to get reference star positions in degrees
        star_list_query = self.query_vizier(max_stars=100 * max_stars_detect)
        ## self.write_fits('star_list_query.fits', star_list_query, overwrite=True)
        ## star_list_query = self.read_fits('star_list_query.fits')
        
        if len(star_list_query) < MIN_STAR_NB:
            self._print_error("Not enough stars found in the field (%d < %d)"%(len(star_list_query), MIN_STAR_NB))
            
        # reference star position list in degrees
        star_list_deg = star_list_query[:max_stars_detect*20]

        ## Define a basic WCS        
        wcs = utils.astrometry.create_wcs(
            self.target_x, self.target_y,
            deltax, deltay, self.target_ra, self.target_dec,
            self.wcs_rotation, sip=self.sip)
        
        # Compute initial star positions from initial transformation
        # parameters
        rmax = max(self.dimx, self.dimy) / math.sqrt(2)
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
            self._print_msg('histogram check: correlation level {}, angle {}, dx {}, dy {}'.format(*max_list[-1]))
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

        self._print_msg(
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
                deep_frame,
                star_list_deg, x_range, y_range, r_range,
                None, izoom, self.fwhm_pix * 3.,
                verbose=False, init_wcs=wcs, raise_border_error=False)

            zoom_guesses.append((izoom, dx, dy, dr, np.nanmax(guess_matrix)))
            self._print_msg('Checking with zoom {}: dx={}, dy={}, dr={}, score={}'.format(*zoom_guesses[-1]))

        # sort brute force guesses to get the best one
        best_guess = sorted(zoom_guesses, key=lambda zoomp: zoomp[4])[-1]

        self.wcs_rotation -= best_guess[3]
        self.target_x -= best_guess[1]
        self.target_y -= best_guess[2]
    
        deltax *= best_guess[0]
        deltay *= best_guess[0]
        
        self._print_msg(
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
            self._print_msg('Computing SIP coefficients')
            if self.sip is None:
                self._print_warning('As no prior SIP has been given, this initial SIP is computed over the field inner circle. To cover the whole field the result of this registration must be passed at the definition of the class')
                r_coeff = 0.5
            else:
                r_coeff = 1./np.sqrt(2)
                
            # compute optical distortion with a greater list of stars
            rmax = max(self.dimx, self.dimy) * r_coeff

            star_list_pix = radius_filter(
                world2pix(wcs, star_list_query), rmax,
                borders=[0, self.dimx, 0, self.dimy])
            self.reset_star_list(star_list_pix)
            
            fit_params = self.fit_stars_in_frame(
                deep_frame, local_background=True,
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
        self._print_msg('Computing distortion maps')
        r_coeff = 1./np.sqrt(2)
        rmax = max(self.dimx, self.dimy) * r_coeff

        star_list_pix = radius_filter(
            world2pix(wcs, star_list_query), rmax, borders=[
                0, self.dimx, 0, self.dimy])
        self.reset_star_list(star_list_pix)

        # fit based on SIP corrected parameters
        fit_params = self.fit_stars_in_frame(
            deep_frame, local_background=False,
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
            self._print_msg('Computing astrometrical precision')

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

            fit_params = self.fit_stars_in_frame(
                deep_frame, local_background=False,
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

            self._print_msg(
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
       
        self._print_msg(
            "Optimization parameters:\n"
            + "> Rotation angle [in degree]: {:.3f}\n".format(self.wcs_rotation)
            + "> Target position [in pixel]: ({:.3f}, {:.3f})\n".format(
                self.target_x, self.target_y)
            + "> Scale X (arcsec/pixel): {:.5f}\n".format(
                deltax * 3600.)
            + "> Scale Y (arcsec/pixel): {:.5f}".format(
                deltay * 3600.))

        self.reset_scale(np.mean((deltax, deltay)) * 3600.)
        
        self._print_msg('Corrected WCS computed')
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
            'RANGE_COEFF', float(self._get_config_parameter(
            'ALIGNER_RANGE_COEFF'))))
        
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
        
        self.astro1 = Astrometry(self.image1, fwhm_arc, fov1,
                                 profile_name='gaussian')
        self.astro2 = Astrometry(self.image2, fwhm_arc, fov2,
                                 profile_name='gaussian')

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
        self._print_msg("\n> dx : " + str(self.dx) + "\n" +
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
            self.write_fits(self._get_guess_matrix_path(),
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
            self._print_error('If the path to a list of stars is given (star_list_path1) the fwhm in arcsec(fwhm_arc) must also be given.')

        self.astro2.reset_fwhm_arc(fwhm_arc)
        self.astro1.reset_fwhm_arc(fwhm_arc)


        ##########################################
        ### BRUTE FORCE GUESS (only dx and dy) ###
        ##########################################
        if brute_force:
            self._print_msg("Brute force guess on large field")
            brute_force_alignment(4*XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS/2, self.range_coeff*10)
            self._print_msg("Brute force guess:") 
            self.print_alignment_coeffs()

            self._print_msg("Finer brute force guess")
            brute_force_alignment(XYSTEP_SIZE, ANGLE_RANGE, ANGLE_STEPS, self.range_coeff)
            self._print_msg("Brute force guess:") 
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

        self._print_msg("Fine alignment parameters:") 
        self.print_alignment_coeffs()
                

        #####################################
        ### COMPUTE DISTORTION CORRECTION ###
        #####################################

        if correct_distortion:
            self._print_msg('Computing distortion correction polynomial (SIP)')
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
                self._print_error("Not enough fitted stars in both cubes (%d%%). Alignment parameters might be wrong."%int(fitted_star_nb / MIN_STAR_NB * 100.))
                
            if (fitted_star_nb < WARNING_RATIO * MIN_STAR_NB):
                self._print_warning("Poor ratio of fitted stars in both cubes (%d%%). Check alignment parameters."%int(fitted_star_nb / MIN_STAR_NB * 100.))

            
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
                self._print_msg('Mean difference on star positions: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
            elif final_err < self.astro1.arc2pix(ERROR_DIST):
                self._print_warning('Mean difference on star positions is bad: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
            else:
                self._print_error('Mean difference on star positions is too bad: {} pixels = {} arcsec'.format(final_err, self.astro1.pix2arc(final_err)))
        

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


