#!/usr/bin/python2.7
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: astrometry.py

"""
The Astrometry module is aimed to manage all astrometry processes:
Fitting star position and photometry for alignement and cubes merging.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'

import core
__version__ = core.__version__
from core import Tools, Cube, ProgressBar
import numpy as np
from scipy import optimize
import math
import urllib2
import StringIO
import pywcs

import utils
import cutils


##################################################
#### CLASS StarsParams ############################
###################################################

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
    

    def __init__(self, star_nb=1, frame_nb=1, logfile_name=None,
                 silent=False):
        """StarsParams init

        :param star_nb: (Optional) Number of stars (default 1)
        
        :param frame_nb: (Optional) Number of frames (default 1)

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).

        :param silent: (Optional) If True no message is displayed
          (default False).
        """
        self._silent = silent
        if logfile_name != None:
            self._logfile_name = logfile_name
            
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
                elif item == None:
                    self.data[key, 0] = None
                else:
                    self._print_error("item to set is neither a StarsParams instance or a dict")
            elif self.star_nb == 1:
                if isinstance(item, StarsParams):
                    self.data[0, key] = item[:,0]
                elif isinstance(item, dict):
                    self.data[0, key] = item
                elif item == None:
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
                        if sliced_data[ik,jk] != None:
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
                if np.atleast_1d(np.squeeze(self.data[data_key]))[0] != None:
                    return np.atleast_1d(np.squeeze(self.data[data_key]))[0][param]
                else:
                    return np.nan
            else:
                sliced_data = np.atleast_2d(self.data[data_key])
                ret_data = np.empty_like(sliced_data)
                for ik in range(sliced_data.shape[0]):
                    for jk in range(sliced_data.shape[1]):
                        if sliced_data[ik,jk] != None:
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
                                     logfile_name=self._logfile_name,
                                     silent=self._silent)
            for ik in range(star_nb):
                for jk in range(frame_nb):
                    if isinstance(data[ik,jk], dict):
                        new_params[ik,jk] = data[ik,jk]
                    elif data[ik,jk] == None:
                        new_params[ik,jk] = None
                    else:
                        self._print_error("Error occuring in data conversion. Data may be corrupted.")
            return new_params
        
    def reset_data(self):
        """Reset StarsParams data to an empty array"""
        self.data = np.empty((self.star_nb, self.frame_nb),
                             dtype=object)
        self.data.fill(dict())
        
    def save_stars_params(self, params_file_path):
        """Save stars parameters in one file

        :param params_file_path: Path to the parameters file
        """
        self._print_msg("Saving stars parameters")
        f = self.open_file(params_file_path, 'w')
       
        f.write(
            """## Stars Parameters
## Generated by ORBS (author: thomas.martin.1@ulaval.ca)
# STAR_NB %d
# FRAME_NB %d
# PARAMS index star %s
"""%(self.star_nb,
     self.frame_nb,
     ' '.join([key for key in self.keys])))
        
        for index in range(self.frame_nb):
            for istar in range(self.star_nb):
                if self[istar, index] != None:
                    params = ''
                    for ikey in self.keys:
                        if ikey in self[istar, index].keys():
                            params += '%s '%str(self[istar, index, ikey])
                        else:
                            params += 'None '
                else:
                    params = 'None'
                f.write("%d %d %s\n"%(index, istar, params))
        f.close()

    def load_stars_params(self, params_file_path, silent=False):
        """Load a file containing stars parameters.

        :param fit_results_path: Path to the file parameters

        :param silent: (Optional) If True, no message is printed
          (default False).
        """
        if not silent:
            self._print_msg('Loading stars parameters', color=True)
        f = self.open_file(params_file_path, 'r')
        lines = f.readlines()
        f.close()
        # get header params to construct params array
        star_nb = None
        keys = None
        frame_nb = None
        
        for line in lines:
            if '##' not in line:
                if 'PARAMS' in line:
                    keys = line.split()[4:]
                if 'STAR_NB' in line:
                    star_nb = int(line.split()[-1])
                if 'FRAME_NB' in line:
                    frame_nb = int(line.split()[-1])
                    
        if star_nb == None or keys == None or frame_nb == None:
            self._print_error('Bad star parameters file')
            
        if self.star_nb != None:
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
        
        # fill data
        for line in lines:         
            if '#' not in line:
                line = line.split()
                if line[-1] != 'None':
                    params = dict()
                    for ik in range(len(keys)):
                        if line[ik+2] != 'None':
                            params[keys[ik]] = float(line[ik+2])
                        else:
                            params[keys[ik]] = None
                    self.data[int(line[1]), int(line[0])] = params

        if not silent:
            self._print_msg('Stars parameters loaded')

    def get_star_list(self, index=0, all_params=False):
        """Return an array of the positions of the stars in one frame

        :param index: (Optional) The index of the frame to use to
          create the list of stars (default 0)

        :param all_params: (Optional) If True, all params are
          returned. Bad params are returned as NaN values.
        """
        star_list = list()
        for istar in range(self.star_nb):
            if self[istar, index] != None:
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
#### CLASS PSF ###################################
##################################################


class PSF(Tools):
    """General class of inheritance for point spread functions (PSFs)
    """
    params = None

    def __repr__(self):
        return(str(self.params))

    def array2d(self, nx, ny):
        """Return a 2D profile given the size of the returned
        array.
        
        :param nx: Length of the returned array along x axis
        :param ny: Length of the returned array along y axis
        """
        return self.varray2d(int(nx), int(ny))
    
    def varray2d(self, nx, ny):
        """Return a vectorized 2D profile given the size of the returned
        array.
        
        :param nx: Length of the returned array along x axis
        :param ny: Length of the returned array along y axis
        """
        return np.fromfunction(
            np.vectorize(self.psf2d), (int(nx), int(ny)))

    

##################################################
#### CLASS Moffat ################################
##################################################


class Moffat(PSF):
    """Class implementing the Moffat profile.

    This profile is useful to fit stars on CCD arrays.

    .. note:: The Moffat profile has been first proposed by Moffat
      (1969) A&A. The exact form of the equation used has been derived
      from Trujillo et al. (2001) MNRAS, 977. The PSF:

      :math:`f(x,y) = H + A \\times [1+(\\frac{r}{\\alpha})^2]^{-\\beta}`

      with,
      :math:`\\alpha = \\frac{\\text{FWHM}}{2\\sqrt{2^{1/\\beta} - 1}}`

      and,
      :math:`r = (x - dx)^2 + (y - dy)^2`
    
      The total flux F under the 2D profile is thus:
      :math:`F = A \\times \\frac{\\pi \\alpha^2}{\\beta - 1}`
    """

    input_params = list(['height', 'amplitude', 'x', 'y', 'fwhm', 'beta'])
    """Keys of the input parameters"""
    
    params = dict()
    """dictionary containing the parameters of the profile"""
    
    alpha = None # Alpha: coefficient defined from beta and FWHM

    def __init__(self, params, logfile_name=None):
        """Init Moffat profile parameters

        :param params: Input parameters of the Moffat profile. Input
          parameters can be given as a dictionary providing {'height',
          'amplitude', 'x', 'y', 'fwhm', 'beta'} or an array of 6
          elements stored in this order: ['height', 'amplitude', 'x',
          'y', 'fwhm', 'beta']

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).
        """
        if logfile_name != None:
            self._logfile_name = logfile_name
            
        MAX_BETA = 30.
        self.params = dict()
        
        if isinstance(params, dict):
            if (set([key for key in params.iterkeys()])
                & set(self.input_params) == set(self.input_params)):
                self.params['height'] = float(params['height'])
                self.params['amplitude'] = float(params['amplitude'])
                self.params['x'] = float(params['x'])
                self.params['y'] = float(params['y'])
                self.params['fwhm'] = abs(float(params['fwhm']))
                self.params['beta'] = float(params['beta'])
            else:
                self._print_error("Input dictionary is not valid. You must provide a dictionary containing all those keys : %s"%str(self.input_params))

        elif (np.size(params) == np.size(self.input_params)):
            self.params['height'] = float(params[0])
            self.params['amplitude'] = float(params[1])
            self.params['x'] = float(params[2])
            self.params['y'] = float(params[3])
            self.params['fwhm'] = abs(float(params[4]))
            self.params['beta'] = float(params[5])

        else:
            self._print_error('Invalid input parameters')


        if self.params['beta'] > MAX_BETA: self.params['beta'] = MAX_BETA

        # Computation of alpha
        # Beta must not be negative or too near 0
        if (self.params['beta'] > .1):
            self.alpha = (self.params['fwhm']
                          / (2. * np.sqrt(2.**(1. / self.params['beta']) - 1.)))
        else:
            self.alpha = np.nan

        self.params['flux'] = self.flux()

        # 1-D PSF function
        self.psf = lambda r: (
            self.params['height'] + self.params['amplitude']
            * (1. + (r/self.alpha)**2.)**(-self.params['beta']))
        
        # 2-D PSF function
        self.psf2d = lambda x, y: (
            self.psf(np.sqrt((x - self.params['x'])**2.
                             +(y - self.params['y'])**2.)))
        

    def flux(self):
        """Return the total flux under the 2D profile.
        """
        if self.params['beta'] != 1.:
            return (self.params['amplitude']
                    * ((math.pi * self.alpha**2.)
                       / (self.params['beta'] - 1.)))
        else:
            return np.nan

    def flux_error(self, amplitude_err, width_err):
        """Return flux error.
        
        :param amplitude_err: estimation of the amplitude error
        
        :param width_err: estimation of the width error

        .. warning:: Not implemented yet!
        """
        return np.nan

    def array2d(self, nx, ny):
        """Return a 2D profile given the size of the returned
        array.
        
        :param nx: Length of the returned array along x axis
        :param ny: Length of the returned array along y axis
        """
        return np.array(cutils.moffat_array2d(
            float(self.params['height']), float(self.params['amplitude']),
            float(self.params['x']), float(self.params['y']),
            float(self.params['fwhm']), self.params['beta'], int(nx), int(ny)))
    
##################################################
#### CLASS Gaussian ##############################
##################################################


class Gaussian(PSF):
    """Class implementing the gaussian profile

    .. note:: The Gaussian profile used here is:
      :math:`f(x,y) = H + A \\times \exp(\\frac{-r^2}{2 W^2})`

      and,
      :math:`r = (x - dx)^2 + (y - dy)^2`
      
      The total flux F under the 2D profile is:
      :math:`F = 2 \\pi A W^2`
    
    """


    input_params = list(['height', 'amplitude', 'x', 'y', 'fwhm'])
    """Keys of the input parameters"""
    
    params = dict()
    """dictionary containing the parameters of the profile"""
    
    width = None # Width = FWHM / abs(2.*sqrt(2. * log(2.)))


    def __init__(self, params, logfile_name=None):
        """Init Gaussian profile parameters

        :param params: Input parameters of the Gaussian profile. Input
          parameters can be given as a dictionary providing {'height',
          'amplitude', 'x', 'y', 'fwhm'} or an array of 5
          elements stored in this order: ['height', 'amplitude', 'x',
          'y', 'fwhm']

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).
        """
        if logfile_name != None:
            self._logfile_name = logfile_name
            
        self.params = dict()
        if isinstance(params, dict):
            if (set([key for key in params.iterkeys()])
                & set(self.input_params) == set(self.input_params)):
                self.params['height'] = float(params['height'])
                self.params['amplitude'] = float(params['amplitude'])
                self.params['x'] = float(params['x'])
                self.params['y'] = float(params['y'])
                self.params['fwhm'] = abs(float(params['fwhm']))
            else:
                self._print_error("Input dictionary is not valid. You must provide a dictionary containing all those keys : %s"%str(self.input_params))

        elif (np.size(params) == np.size(self.input_params)):
            self.params['height'] = float(params[0])
            self.params['amplitude'] = float(params[1])
            self.params['x'] = float(params[2])
            self.params['y'] = float(params[3])
            self.params['fwhm'] =  abs(float(params[4]))

        else:
            self._print_error('Invalid input parameters')


        self.width = self.params['fwhm'] / abs(2.*math.sqrt(2. * math.log(2.)))
        self.params['flux'] = self.flux()


        # 1-D PSF function
        self.psf = lambda r: (
            self.params['height'] + self.params['amplitude']
            * np.exp(-(r)**2./(2.*self.width**2.)))
        
        # 2-D PSF function
        self.psf2d = lambda x, y: (
            self.psf(math.sqrt((x-self.params['x'])**2.
                               +(y-self.params['y'])**2.)))
        

    def flux(self):
        """Return the total flux under the 2D profile.
        
        The total flux F under a 2D profile is :
        :math:`F = 2 \\pi A W^2`
        
        .. note:: Under a 1d profile the flux is :math:`F = \\sqrt{2\\pi}A W`
        """
        return 2. * self.params['amplitude'] * (self.width)**2 * math.pi

    def flux_error(self, amplitude_err, width_err):
        """Return flux error.
        
        :param amplitude_err: estimation of the amplitude error
        
        :param width_err: estimation of the width error
        """
        return self.flux() * math.sqrt(
            (amplitude_err / self.params['amplitude'])**2.
            + 2. * (width_err / self.width)**2.)
    
    def array2d(self, nx, ny):
        """Return a 2D profile given the size of the returned
        array.
        
        :param nx: Length of the returned array along x axis
        :param ny: Length of the returned array along y axis
        """
        return np.array(cutils.gaussian_array2d(float(self.params['height']),
                                                float(self.params['amplitude']),
                                                float(self.params['x']),
                                                float(self.params['y']),
                                                float(self.params['fwhm']),
                                                int(nx), int(ny)))


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

    base_url_ca = "http://vizier.hia.nrc.ca/viz-bin/"
    """Beginning of the URL to the Vizier server in Canada"""
    
    base_url = "http://webviz.u-strasbg.fr/viz-bin/"
    """Beginning of the URL to the Vizier server"""

    BASE_URL = base_url_ca
    """Beginning of the URL to the Vizer server by default"""

    
    def __init__(self, data, fwhm_arc, fov, profile_name='moffat',
                 detect_stack=5, fit_tol=1e-2, moffat_beta=3.,
                 data_prefix="", star_list_path=None, box_size_coeff=7.,
                 check_mask=True, reduced_chi_square_limit=1.5,
                 readout_noise=10., dark_current_level=0.,
                 target_radec=None, target_xy=None, wcs_rotation=None,
                 config_file_name='config.orb', logfile_name=None,
                 tuning_parameters=dict(), silent=False):
        
        """
        Init astrometry class.

        :param data: Can be an 2D or 3D Numpy array or an instance of
          core.Cube class. Note that the frames must not be too
          disaligned (a few pixels in both directions).

        :param fwhm_arc: rough FWHM of the stars in arcsec
        
        :param fov: Field of view of the frame in arcminutes (given
          along x axis)

        :param profile_name: (Optional) Name of the PSF profile to use
          for fitting. Can be 'moffat' or 'gaussian' (default 'moffat').

        :param detect_stack: (Optional) Number of frames to stack
          before detecting stars (default 5).

        :param fit_tol: (Optional) Tolerance on the paramaters fit
          (the lower the better but the longer too) (default 1e-2).

        :param moffat_beta: (Optional) Default value of the beta
          parameter for the moffat profile (default 3.5).

        :param data_prefix: (Optional) Prefix of the data saved on
          disk (default "").

        :param star_list_path: (Optional) Path to a file containing a
          list of star positions (default None).

        :param box_size_coeff: (Optional) Coefficient giving the size
          of the box created around each star before a fit is
          done. BOX_SIZE = box_size_coeff * STAR_FWHM. (default 7.).
          Note that this coeff is divided by 3. if the moffat profile
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

        :param config_file_name: (Optional) name of the config file to
          use. Must be located in orbs/data/ (default 'config.orb').

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).

        :param tuning_parameters: (Optional) Some parameters of the
          methods can be tuned externally using this dictionary. The
          dictionary must contains the full parameter name
          (class.method.parameter_name) and its value. For example :
          {'Astrometry.get_alignment_vectors.HPFILTER': 1}. Note
          that only some parameters can be tuned. This possibility is
          implemented into the method itself with the method
          :py:meth:`Tools._get_tuning_parameter`.

        :param silent: (Optional) If True, a minimum of messages are
          printed (default False).
        """
        self.config_file_name=config_file_name
        self.ncpus = int(self._get_config_parameter("NCPUS"))
        self.BIG_DATA = bool(int(self._get_config_parameter("BIG_DATA")))
        
        self._init_logfile_name(logfile_name)
        self._data_prefix = data_prefix
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()
        self._tuning_parameters = tuning_parameters
        self._silent = silent
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
        if star_list_path != None:
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

      
    def _get_star_list_path(self):
        """Return the default path to the star list file."""
        return self._data_path_hdr + "star_list"

    def _get_fit_results_path(self):
        """Return the default path to the file containing all fit
        results."""
        return self._data_path_hdr + "fit_results"

    def _get_combined_frame(self):
        """Return a combined frame to work on"""
        
        # If we have 3D data we work on a combined image of the first
        # frames to detect stars
        if self.dimz > 1:
            stack_nb = self.detect_stack
            if stack_nb + self.DETECT_INDEX > self.frame_nb:
                stack_nb = self.frame_nb - self.DETECT_INDEX
                
            if not self.BIG_DATA:
                im = utils.create_master_frame(
                    self.data[:,:,
                              int(self.DETECT_INDEX):
                              int(self.DETECT_INDEX+stack_nb)])
            else:
                im = utils.pp_create_master_frame(
                    self.data[:,:,
                              int(self.DETECT_INDEX):
                              int(self.DETECT_INDEX+stack_nb)])
                
        # else we just return the only frame we have
        else:
            im = np.copy(self.data)
        return im

    def reset_profile_name(self, profile_name):
        """Reset the name of the profile used.

        :param profile_name: Name of the PSF profile to use for
          fitting. Can be 'moffat' or 'gaussian'.
        """
        if profile_name in self.profiles:
            self.profile_name = profile_name
            self.profile = get_profile(self.profile_name)
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
                                       logfile_name=self._logfile_name,
                                       silent=self._silent)
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
        """Convert pixels in arcseconds

        :param x: a value or a vector in pixel
        """
        if self.scale != None:
            return np.array(x).astype(float) / self.scale
        else:
            self._print_error("Scale not defined")

    def pix2arc(self, x):
        """Convert arcseconds in pixels

        :param x: a value or a vector in arcsec
        """
        if self.scale != None:
            return np.array(x).astype(float) * self.scale
        else:
            self._print_error("Scale not defined")

    def fwhm(self, x):
        return x * abs(2.*math.sqrt(2. * math.log(2.)))

    def width(self, x):
        return x / abs(2.*math.sqrt(2. * math.log(2.)))

    def load_fit_results(self, fit_results_path=None):
        """Load a file containing the fit results"""
        if fit_results_path == None:
            fit_results_path = self._get_fit_results_path()
        self.fit_results.load_stars_params(fit_results_path)
    
    def load_star_list(self, star_list_path):
        """Load a list of stars coordinates

        :param star_list_path: The path to the star list file.

        .. seealso:: :py:meth:`astrometry.load_star_list`
        """
        star_list = load_star_list(star_list_path, silent=self._silent)
        
        self.reset_star_list(np.array(star_list, dtype=float))
        
        return self.star_list

    def fit_stars_in_frame(self, index, fix_height=True, fix_beta=True,
                           fix_fwhm=False, local_background=True,
                           save=False, fix_aperture_fwhm_pix=None,
                           no_aperture_photometry=False, precise_guess=False,
                           aper_coeff=3., blur=False, no_fit=False,
                           estimate_local_noise=True, multi_fit=False,
                           enable_zoom=False, enable_rotation=False):
        """
        Fit stars in one frame.
        
        :param index: Index of the frame to fit. If index is a frame,
          this frame is used instead.
        
        :param fix_height: (Optional) Fix height parameter to its
          estimation (default False)

        :param fix_beta: (Optional) Fix beta to the given value
          (default True).

        :param fix_fwhm: (Optional) Fix FWHM to to its
          estimation (default False).

        :param local_background: (Optional) If True, height is
          evaluated locally (median of the box around the star). If
          False, the sky background is determined it the whole frame
          and the guess height will be the same for all the stars
          (default True).
          
        :param save: (Optional) If True save the fit results in a file
          (default True).

        :param fix_aperture_fwhm_pix: (Optional) If a positive
          float. FWHM used to scale aperture size is not computed from
          the mean FWHM in the frame but fixed to the given float
          (default None).

        :param no_aperture_photometry: (Optional) If True, aperture
          photometry will not be done after profile fitting (default
          False).

        :param precise_guess: (Optional) If True, the fit guess will
          be more precise but this can lead to errors if the stars
          positions are not already roughly known (default False).

        :param aper_coeff: (Optional) Aperture coefficient. The
          aperture radius is Rap = aper_coeff * FWHM. Better when
          between 1.5 to reduce the variation of the collected photons
          with varying FWHM and 3. to account for the flux in the
          wings (default 3., better for star with a high SNR).

        :param blur: (Optional) If True, blur frame (low pass
          filtering) before fitting stars. It can be used to enhance
          the quality of the fitted flux of undersampled data. Note
          that the error on star position can be greater on blurred
          frame. This option must not be used for alignment purpose
          (default False).

        :param no_fit: (Optional) If True, no fit is done. Only the
          aperture photometry. Star positions in the star list must
          thus be precise (default False).

        :param estimate_local_noise: (Optional) If True, the level of
          noise is computed from the background pixels around the
          stars. readout_noise and dark_current_level are thus not
          used (default True).

        :param multi_fit: (Optional) If True all stars are fitted at
          the same time. More robust for alignment purpose. The
          difference of position between the stars in the star list
          must be precisely known because the overall shift only is
          estimated (default False).

        :param enable_zoom: (Optional) Useful only if multi_fit is
          True.  If True, the stars position pattern can be zoomed to
          better adjust it to the real frame (default False).

        :param enable_rotation: (Optional) Useful only if multi_fit is
          True.  If True, the stars position pattern can be rotated to
          better adjust it to the real frame (default False).
        """
        if self.data == None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list == None: self._print_error(
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
                

        fit_results = fit_stars_in_frame(
            frame, self.star_list, self.box_size,
            profile_name=self.profile_name,
            scale=self.scale,
            fwhm_pix=self.fwhm_pix,
            beta=self.default_beta,
            fix_height=fix_height,
            fix_beta=fix_beta,
            fix_fwhm=fix_fwhm,
            fit_tol=self.fit_tol,
            local_background=local_background,
            fix_aperture_fwhm_pix=fix_aperture_fwhm_pix,
            no_aperture_photometry=no_aperture_photometry,
            readout_noise=self.readout_noise,
            dark_current_level=self.dark_current_level,
            precise_guess=precise_guess,
            aper_coeff=aper_coeff,
            blur=blur,
            no_fit=no_fit,
            estimate_local_noise=estimate_local_noise,
            multi_fit=multi_fit,
            enable_zoom=enable_zoom,
            enable_rotation=enable_rotation)

        if not isinstance(index, np.ndarray):
            if self.dimz > 1:
                self.fit_results[:,index] = fit_results
            else:
                self.fit_results = fit_results
        
        if save:
            self.fit_results.save_stars_params(self._get_fit_results_path())
            
        return fit_results


    def fit_stars_in_cube(self, correct_alignment=False, save=True,
                          fix_height=True, fix_beta=True,
                          fix_fwhm=False,
                          fwhm_min=0.5, local_background=True,
                          no_aperture_photometry=False,
                          fix_aperture_size=False, precise_guess=False,
                          aper_coeff=3., blur=False, hpfilter=False,
                          add_cube=None, no_fit=False,
                          estimate_local_noise=True, multi_fit=False,
                          enable_zoom=False, enable_rotation=False):
        
        """Fit stars in the cube.

        Frames must not be too disaligned. Disalignment can be
        corrected as the cube is fitted by setting correct_alignment
        option to True.

        :param correct_alignment: (Optional) If True, the initial star
          positions from the star list are corrected by their last
          recorded deviation. Useful when the cube is smoothly
          disaligned from one frame to the next.

        :param save: (Optional) If True save the fit results in a file
          (default True).

        :param fix_height: (Optional) Fix height parameter to its
          estimation (default True)

        :param fix_beta: (Optional) Fix beta to its default value
          (default True).

        :param fix_fwhm: (Optional) Fix FWHM to the default value
          (default False)

        :param fwhm_min: (Optional) Minimum valid FWHM of the fitted
          star (default 0.5)

        :param local_background: (Optional) If True, background level
          is estimated locally, i.e. around the stars. If False, a
          global background is estimated for each frame and will be
          used for all the stars in the frame. In this case,
          background level is fixed and will be the same for fitting
          and photometry (default True).

        :param no_aperture_photometry: (Optional) if True no aperture
          photometry will be done after parameters fit. It increases
          the speed of the process (default False).

        :param fix_aperture_size: (Optional) If True, size of the
          aperture is fixed to a radius defined by the default value
          of the FWHM. If False, aperture size is evaluated for each
          frame from the mean FWHM of the stars in the frame (default
          False).

        :param precise_guess: (Optional) If True, the fit guess will
          be more precise but this can lead to errors if the stars
          positions are not already roughly known (default False).

        :param aper_coeff: (Optional) Aperture coefficient. The
          aperture radius is Rap = aper_coeff * FWHM. Better when
          between 1.5 to reduce the variation of the collected photons
          with varying FWHM and 3. to account for the flux in the
          wings (default 3., better for star with a high SNR).

        :param blur: (Optional) If True, blur frame (low pass
          filtering) before fitting stars. It can be used to enhance
          the quality of the fitted flux of undersampled data. Note
          that the error on star position can be greater on blurred
          frame. This option must not be used for alignment purpose
          (default False).

        :param no_fit: (Optional) If True, no fit is done. Only the
          aperture photometry. Star positions in the star list must
          thus be precise (default False).

        :param hpfilter: (Optional) If True, frames are HP filtered
          before fitting stars. Useful for alignment purpose if there
          are too much nebulosities in the frames. This option must
          not be used for photometry (default False).

        :param estimate_local_noise: (Optional) If True, the level of
          noise is computed from the background pixels around the
          stars. readout_noise and dark_current_level are thus not
          used (default True).

        :param multi_fit: (Optional) If True all stars are fitted at
          the same time. More robust for alignment purpose. The
          difference of position between the stars in the star list
          must be precisely known because the overall shift only is
          estimated (default False).

        :param enable_zoom: (Optional) Useful only if multi_fit is
          True.  If True, the stars position pattern can be zoomed to
          better adjust it to the real frame (default False).

        :param enable_rotation: (Optional) Useful only if multi_fit is
          True.  If True, the stars position pattern can be rotated to
          better adjust it to the real frame (default False).
        """
        def get_index_mean_dev(index):
            dx = utils.robust_mean(utils.sigmacut(
                self.fit_results[:,index,'dx']))
            dy = utils.robust_mean(utils.sigmacut(
                self.fit_results[:,index,'dy']))
            return dx, dy


        FOLLOW_NB = 5 # Number of deviation value to get to follow the
                      # stars
        
        self._print_msg("Fitting stars in cube", color=True)

        if self.data == None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list == None: self._print_error(
            "A star list must be loaded or created first")

        if self.dimz < 2: self._print_error(
            "Data must have 3 dimensions. Use fit_stars_in_frame method instead")
        if fix_aperture_size:
            fix_aperture_fwhm_pix = self.fwhm_pix
        else:
            fix_aperture_fwhm_pix = None

        
        if add_cube != None:
            if np.size(add_cube) == 2:
                added_cube = add_cube[0]
                added_cube_scale = add_cube[1]
                if not isinstance(added_cube, Cube):
                    self._print_error('Added cube must be a Cube instance. Check add_cube option')
                if np.size(added_cube_scale) != 1:
                    self._print_error('Bad added cube scale. Check add_cube option.')
                

        self.fit_results = StarsParams(self.star_nb, self.frame_nb,
                                       logfile_name=self._logfile_name,
                                       silent=self._silent)

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
                        x_corr = utils.robust_median(x_mean_dev)
                        y_corr = utils.robust_median(y_mean_dev)
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

            for ijob in range(ncpus):
                frame = np.copy(self.data[:,:,ik+ijob])
                if add_cube != None:
                    frame += added_cube[:,:,ik+ijob] * added_cube_scale
                if self._check_mask:
                    if self.data._mask_exists:
                        mask = self.data.get_data_frame(ik+ijob, mask=True)
                        frame = frame.astype(float)
                        frame[np.nonzero(mask)] = np.nan
                if hpfilter:
                    frame = utils.high_pass_diff_image_filter(frame, deg=1)
                    
                frames[:,:,ijob] = np.copy(frame)
                
            # get stars photometry for each frame
            jobs = [(ijob, job_server.submit(
                fit_stars_in_frame,
                args=(frames[:,:,ijob], star_list, self.box_size,
                      self.profile_name, self.scale, self.fwhm_pix,
                      self.default_beta, self.fit_tol, fwhm_min,
                      fix_height, fix_aperture_fwhm_pix, fix_beta, fix_fwhm,
                      self.readout_noise, self.dark_current_level,
                      local_background, no_aperture_photometry, precise_guess,
                      aper_coeff, blur, no_fit, estimate_local_noise,
                      multi_fit, enable_zoom, enable_rotation),
                modules=("from orb.astrometry import fit_star, StarsParams, sky_background_level, aperture_photometry, get_profile",
                         "import numpy as np",
                         "import math",
                         "import orb.cutils as cutils",
                         "import orb.utils as utils")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                self.fit_results[:, ik+ijob] = job()
                

            progress.update(ik, info="frame : " + str(ik))
            
        self._close_pp_server(job_server)
        
        progress.end()

        if save:
            self.fit_results.save_stars_params(self._get_fit_results_path())

        # print reduced chi square
        mean_red_chi_square = utils.robust_mean(utils.sigmacut(
            self.fit_results[:, 'reduced-chi-square']))
        
        self._print_msg("Mean reduced chi-square: %f"%mean_red_chi_square)
        
        return self.fit_results



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
  
        self._print_msg("Detecting stars from catalogue", color=True)
        # during registration a star list created from a catalogue
        # is created.
        self.register()

        deep_frame = self._get_combined_frame()
        fit_params = self.fit_stars_in_frame(deep_frame, multi_fit=False,
                                             local_background=True)
        
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



    def detect_stars(self, min_star_number=4, no_save=False,
                     saturation_threshold=35000, try_catalogue=True):
        """Detect star positions in data.

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

        :param try_catalogue: If True, try to use a catalogue to
          detect stars if target_ra, target_dec, target_x, target_y
          and wcs_rotation parameters have been given.

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

            params = fit_star(box, profile_name=profile_name,
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
        
        MIN_FWHM_COEFF = 0.8
        """Coefficient used to determine the minimum FWHM given the
        Rough stars FWHM
        """

        # TRY catalogue
        if try_catalogue:
            if (self.target_ra is not None and self.target_dec is not None
                and self.target_x is not None and self.target_y is not None
                and self.wcs_rotation is not None):
                return self.detect_stars_from_catalogue(
                    min_star_number=min_star_number, no_save=no_save,
                    saturation_threshold=saturation_threshold)

        self._print_msg("Detecting stars", color=True)

        im = self._get_combined_frame()
      
        # high pass filtering of the image to remove nebulosities   
        self._print_msg("Filtering master image")
        hp_im = utils.high_pass_diff_image_filter(im, deg=1)

        # preselection
        mean_hp_im = np.mean(hp_im)
        std_hp_im = np.std(hp_im)
        max_im = np.max(im)
        # +1 is just here to make sure we enter the loop
        star_number = PRE_DETECT_COEFF * min_star_number + 1 
        
        old_star_list = []
        while(star_number > PRE_DETECT_COEFF * min_star_number):
            pre_star_list = np.array(np.nonzero(hp_im > 
                                                mean_hp_im 
                                                + THRESHOLD_COEFF 
                                                * std_hp_im))
            star_list = list()
            for istar in range(pre_star_list.shape[1]):
                ix = pre_star_list[0, istar]
                iy = pre_star_list[1, istar]
                (box, mins)  = define_box(ix,iy,self.box_size,im)
                ilevel = im[ix, iy]
                if (ilevel == np.max(box)) and (ilevel <= max_im):
                    # filter stars near the border of the frame
                    if ((ix < 0.95 * self.dimx) and 
                        (iy < 0.95 * self.dimy) and
                        (ix > 0.05 * self.dimx) and 
                        (iy > 0.05 * self.dimy)):
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
        
        self._print_msg("Fit stars")
        
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
                         "import orb.utils",
                         "from orb.astrometry import fit_star")))
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
        median_fwhm = utils.robust_mean(utils.sigmacut(fwhm_list, sigma=3.))
        std_fwhm = utils.robust_std(utils.sigmacut(fwhm_list, sigma=3.))
      
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
        
        if self.data == None: self._print_error(
            "Some data must be loaded first")
        
        if self.star_list == None: self._print_error(
            "A star list must be loaded or created first")
    
        if fit_cube:
            self.fit_stars_in_cube(correct_alignment=True,
                                   no_aperture_photometry=True,
                                   hpfilter=HPFILTER, multi_fit=True,
                                   fix_height=False)

        if self.star_nb < 4: 
            self._print_error("Not enough stars to align properly : %d (must be >= 3)"%self.star_nb)
            
        fit_x = self.fit_results[:,:,'x']
        fit_y = self.fit_results[:,:,'y']
        rcs = self.fit_results[:,:,'reduced-chi-square']
        fit_x_err = self.fit_results[:,:,'x_err']
        fit_y_err = self.fit_results[:,:,'y_err']
    
        start_x = np.copy(fit_x[:, 0])
        start_y = np.copy(fit_y[:, 0])

        # Check if enough stars have been fitted in the first frame
        good_nb = len(np.nonzero(~np.isnan(start_x))[0])
        
        if good_nb < 4 or good_nb < min_coeff * self.star_nb:
            self._print_error("Not enough detected stars (%d) in the first frame"%good_nb)

        ## Create alignment vectors from fitted positions
        alignment_vector_x = np.empty(self.frame_nb, dtype=float)
        alignment_vector_y = np.empty(self.frame_nb, dtype=float)

        alignment_error_list = list()
        for ik in range(self.frame_nb):
            for istar in range(self.star_nb):
                if start_x[istar] != np.nan and start_y[istar] != np.nan:
                    if fit_x[istar, ik] != np.nan:
                        fit_x[istar, ik] -= start_x[istar]
                    else:
                        fit_x[istar, ik] = np.nan
                    
                    if fit_y[istar, ik] != np.nan:
                        fit_y[istar, ik] -= start_y[istar]
                    else:
                        fit_y[istar, ik] = np.nan

            # reduced chi square is used to reject bad fits
            mean_reduced_chi_square = utils.robust_mean(utils.sigmacut(
                rcs[:,ik]))
            good_fits = np.nonzero(
                rcs[:,ik] < (self.reduced_chi_square_limit
                             * mean_reduced_chi_square))
            
            fit_x_list = fit_x[good_fits, ik]
            fit_x_err_list = fit_x_err[good_fits, ik]
            fit_y_list = fit_y[good_fits, ik]
            fit_y_err_list = fit_y_err[good_fits, ik]
            
            # Check if enough stars are detected to get a good
            # deviation value
            if np.size(fit_x_list) > min_coeff * self.star_nb:
                # weights are taken from the error on the position
                weights_x = (1. / fit_x_err_list)
                weights_x /= np.sum(weights_x)
                weights_x *= np.size(fit_x_err_list)
                weights_y = (1. / fit_y_err_list)
                weights_y /= np.sum(weights_y)
                weights_y *= np.size(fit_y_err_list)
                
                # averaging
                alignment_vector_x[ik] = utils.robust_mean(fit_x_list,
                                                     weights=weights_x)
                
                alignment_vector_y[ik] = utils.robust_mean(fit_y_list,
                                                     weights=weights_y)

                # compute alignment error on frame
                alignment_error = np.sqrt(np.mean(fit_x_err_list
                                                  + fit_y_err_list)**2)
                alignment_error /= np.sqrt(np.size(fit_x_err_list))
                alignment_error_list.append(alignment_error)

            else:
                alignment_vector_x[ik] = np.nan
                alignment_vector_y[ik] = np.nan
                alignment_error_list.append(np.nan)

        # correct alignment vectors for NaN values
        alignment_vector_x = utils.correct_vector(
            alignment_vector_x, polyfit=True, deg=3)
        alignment_vector_y = utils.correct_vector(
            alignment_vector_y, polyfit=True, deg=3)

        # print some info
        self._print_msg(
            'Alignment vectors median error: %f pixel'%utils.robust_median(
                alignment_error_list))
                
        return alignment_vector_x, alignment_vector_y, alignment_error_list


    def query_vizier(self, radius=7, catalog='USNO-B1.0',
                      max_stars=100):
        """Return a list of star coordinates around an object in a
        given radius based on a query to VizieR Services
        (http://vizier.u-strasbg.fr/viz-bin/VizieR)

        Note that the idea of this method has been picked from an IDL
        function: QUERYVIZIER
        (http://idlastro.gsfc.nasa.gov/ftp/pro/sockets/queryvizier.pro)

        :param radius: (Optional) Radius around the target in
          arc-minutes (default 7).

        :param catalog: (Optional) Catalog to ask on the VizieR
          database (see notes) (default 'USNO-B1')

        :param max_stars: (Optional) Maximum number of row to retrieve
          (default 100)

        .. note:: Some catalogs that can be used::
          'V/139' - Sloan SDSS photometric catalog Release 9 (2012)
          '2MASS-PSC' - 2MASS point source catalog (2003)
          'GSC2.3' - Version 2.3.2 of the HST Guide Star Catalog (2006)
          'USNO-B1' - Verson B1 of the US Naval Observatory catalog (2003)
          'UCAC4'  - 4th U.S. Naval Observatory CCD Astrograph Catalog (2012)
          'B/DENIS/DENIS' - 2nd Deep Near Infrared Survey of southern Sky (2005)
          'I/259/TYC2' - Tycho-2 main catalog (2000)
          'I/311/HIP2' - Hipparcos main catalog, new reduction (2007)
        """
        if self.target_ra is None or self.target_dec is None:
            self._print_error('No catalogue query can be done. Please make sure to give target_radec and target_xy parameter at class init')
            
        self._print_msg("Sending query to VizieR server")
        self._print_msg("Looking for stars at RA: %f DEC: %f"%(
            self.target_ra, self.target_dec))
        
        URL = (self.BASE_URL + "asu-tsv/?-source=" + catalog
               + "&-c.ra=%f"%self.target_ra + '&-c.dec=%f'%self.target_dec
               + "&-c.rm=%d"%int(radius)
               + '&-out.max=unlimited&-out.meta=-huD'
               + '&-out=_RAJ2000,_DEJ2000,R1mag&-sort=R1mag'
               + '&-out.max=%d'%max_stars)
        
        query_result = urllib2.urlopen(URL).read()
        output = StringIO.StringIO(query_result)
        star_list = list()
        for iline in output:
            if iline[0] != '#' and iline[0] != '-' and len(iline) > 3:
                iline = iline.split()
                if len(iline) == 3:
                    star_list.append((float(iline[0]),
                                      float(iline[1]),
                                      float(iline[2])))

        # sorting list to get only the brightest stars first
        star_list = sorted(star_list, key=lambda istar: istar[2])
        
        self._print_msg(
            "%d stars recorded in the given field"%len(star_list))
        
        return star_list



    def register(self, max_stars=50):
        """Register data and return a corrected pywcs.WCS object.
        
        Precise RA/DEC positions of the stars in the field are
        recorded from a catalog of the VIZIER server.

        Using the real position of the same stars in the frame, WCS
        transformation parameters are optimized.
        
        :param max_stars: (Optional) Maximum number of stars used to
          fit (default 50)
        """
        def get_transformation_error(guess, deg_list, fit_list,
                                     target_ra, target_dec):
            _wcs = pywcs.WCS(naxis=2)
            _wcs.wcs.crpix = [guess[1], guess[2]]
            _wcs.wcs.cdelt = np.array([-guess[3], guess[3]])
            _wcs.wcs.crval = [target_ra, target_dec]
            _wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
            _wcs.wcs.crota = [guess[0], guess[0]]
            
            trans_list = list()
            for istar in deg_list:
                pos = _wcs.wcs_sky2pix(istar[0], istar[1], 0)
                trans_list.append((pos[0][0], pos[1][0]))

            error_list = np.sqrt((np.array(trans_list)
                                  - np.array(fit_list))**2)
            return np.mean(error_list)
        
      
        MIN_STAR_NB = 5 # Minimum number of stars to get a correct WCS


        if not (self.target_ra is not None and self.target_dec is not None
                and self.target_x is not None and self.target_y is not None
                and self.wcs_rotation is not None):
            self._print_error("Not enough parameters to register the data. Please give target_xy, target_radec and wcs_rotation options at the init")
        
        self._print_msg('Computing correct WCS', color=True)


        # get deep frame
        deep_frame = self._get_combined_frame()
        
        delta = self.scale / 3600. # arcdeg per pixel
        
        # Query to get reference star positions in degrees
        star_list = self.query_vizier(max_stars=10 * max_stars)

        if len(star_list) < MIN_STAR_NB:
            self._print_error("Not enough stars found in the field (%d < %d)"%(len(star_list), MIN_STAR_NB))
            
        # reference star position list in degrees
        star_list_deg = star_list[:max_stars]

        ## Define a basic WCS
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [self.target_x, self.target_y]
        wcs.wcs.cdelt = np.array([-delta, delta])
        wcs.wcs.crval = [self.target_ra, self.target_dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crota = [self.wcs_rotation, self.wcs_rotation]
        
        # Compute initial star positions from initial transformation
        # parameters
        star_list_pix = list()
        for istar in star_list_deg:
            pos = wcs.wcs_sky2pix(istar[0], istar[1], 0)
            star_list_pix.append((pos[0][0], pos[1][0]))


        ## # Save stars coords
        ## f = self.open_file(self._get_stars_coords_path(), 'w')
        ## for istar in star_list_deg:
        ##     f.write("%f %f\n"%(istar[0],
        ##                        istar[1]))

        star_list_pix = np.array(star_list_pix)
        ## import pylab as pl
        ## pl.imshow(deep_frame.T)
        ## pl.scatter(star_list_pix[:,0], star_list_pix[:,1], color='r')
        ## pl.show()
        
        # Fit stars from computed position in the image
        self.reset_star_list(star_list_pix)
        fit_params = self.fit_stars_in_frame(deep_frame, local_background=False,
                                             multi_fit=True, enable_zoom=True,
                                             enable_rotation=True)
        star_list_fit = fit_params.get_star_list(all_params=True)
        self.reset_star_list(star_list_pix)
        fwhm_pix = utils.robust_median(fit_params[:,'fwhm_pix'])
        self.reset_fwhm_arc(self.pix2arc(fwhm_pix))
        if self.dimz > 1:
            self.fit_results[:,0] = fit_params
        else:
            self.fit_results = fit_params
        
        # get median difference on the position to remove bad fitted
        # stars from all the lists
        delta_pos = (star_list_pix - star_list_fit)**2.
        delta_pos = np.sqrt(delta_pos[:,0] + delta_pos[:,1])
        
        delta_med = utils.robust_mean(delta_pos)
        
        star_list_deg_temp = list()
        star_list_pix_temp = list()
        star_list_fit_temp = list()
        for istar in range(len(delta_pos)):
            if np.mean(np.array(delta_pos[istar])) < delta_med:
                star_list_deg_temp.append(star_list_deg[istar])
                star_list_pix_temp.append(star_list_pix[istar])
                star_list_fit_temp.append(star_list_fit[istar])

        star_list_deg = star_list_deg_temp
        star_list_pix = star_list_pix_temp
        star_list_fit = star_list_fit_temp

        self._print_msg("Best fitted stars: %d (threshold: %f)"%(
            len(star_list_deg), delta_med))
        
        # Optimization of the transformation parameters
        progress = ProgressBar(0)

        guess = np.array([self.wcs_rotation, self.target_x,
                          self.target_y, delta])
        
        [self.wcs_rotation, self.target_x, self.target_y, delta] = (
            optimize.fmin_powell(get_transformation_error, guess, 
                                 args=(star_list_deg,
                                       star_list_fit,
                                       self.target_ra,
                                       self.target_dec),
                                 ftol=1e-3, xtol=1e-3, disp=False))
        progress.end()
        
        final_error = get_transformation_error(
            [self.wcs_rotation, self.target_x, self.target_y, delta],
            star_list_deg, star_list_fit,
            self.target_ra, self.target_dec)

        self._print_msg("Final error after optimization: %f"%final_error)
        self._print_msg("Optimization parameters:\n"
                        + "> Rotation angle: %f\n"%self.wcs_rotation
                        + "> Target position: (%f, %f)\n"%(self.target_x,
                                                           self.target_y)
                        + "> Scale (arcseconds/pixel): %f"%(delta*3600.))

        self.reset_scale(delta*3600.)
        
        # update WCS
        wcs = pywcs.WCS(naxis=2)
        wcs.wcs.crpix = [self.target_x, self.target_y]
        wcs.wcs.cdelt = np.array([-delta, delta])
        wcs.wcs.crval = [self.target_ra, self.target_dec]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crota = [self.wcs_rotation, self.wcs_rotation]

        self._print_msg('Corrected WCS computed')
        
        return wcs

########################################
######## ASTROMETRY UTILS ##############
########################################

def mag(flux):
    """Return the instrumental magnitude of a given flux (magnitude 0
      is set to 1 e-)

    :param flux: Flux in e-
    """
    return np.where(flux > 0., -2.5 * np.log10(flux), np.nan)

def guess(star_box, pos=None, height=None, precise_pos=False):
    """Return an estimation of the star parameters.

    :param star_box: Sub-part of an image surrounding a star. The
      center of the star must be placed near the center of the
      box. The dimensions of the box must be greater than 3 times
      the FWHM of the star.

    :param pos: (Optional) Position guess as a tuple (x,y). Used to
      estimate amplitude (default None).

    :param height: (Optional) Guess of the background level. Used to
      estimate amplitude (default None).

    :param precise_pos: (Optional) If True, position is estimated from
      the marginal distribution of the PSF. Return a far better
      estimation if and only if the star is well centered in the box,
      i.e. if and only if the position of the star is already
      known. This can lead to errors when trying to find the star in
      the box, in this case precise_pos must be set to False (default
      False).

    :return: [height,amplitude,x,y,width]
    """
    if pos != None:
        x_guess = pos[0]
        y_guess = pos[1]
    else:
        # Estimation of the position from the marginal distribution of the psf
        # [e.g. Howell 2006]
        if precise_pos:
            box_dimx = star_box.shape[0]
            box_dimy = star_box.shape[1]
            Ii = np.sum(star_box, axis=1)
            Jj = np.sum(star_box, axis=0)
            I = Ii - np.mean(Ii)
            nzi = np.nonzero(I > 0)
            J = Jj - np.mean(Jj)
            nzj = np.nonzero(J > 0)
            x_guess = np.sum((I[nzi])*np.arange(box_dimx)[nzi])/np.sum(I[nzi])
            y_guess = np.sum((J[nzj])*np.arange(box_dimy)[nzj])/np.sum(J[nzj])
            if np.isnan(x_guess):
                x_guess = np.argmax(np.sum(star_box, axis=1))
            if np.isnan(y_guess):
                y_guess = np.argmax(np.sum(star_box, axis=0))
        else:
            x_guess = np.argmax(np.sum(star_box, axis=1))
            y_guess = np.argmax(np.sum(star_box, axis=0))
            
    if height != None:
        h_guess = height
    else:
        h_guess = sky_background_level(star_box)
        
    a_guess = star_box[int(x_guess), int(y_guess)] - h_guess
    fwhm_guess = float(min(star_box.shape)) * 0.2
    
    return [h_guess,a_guess,x_guess,y_guess,fwhm_guess]


def fit_star(star_box, profile_name='gaussian', fwhm_pix=None,
             amp=None, beta=3.5, height=None, pos=None,
             fix_height=False, fix_amp=False, fix_beta=True,
             fix_fwhm=False, fix_pos=False,
             fit_tol=1e-3, check=True, fwhm_min=0.5,
             check_reject=False, ron=10., dcl=0.,
             estimate_local_noise=True, precise_guess=False):

    """Fit a single star

    :param star_box: The box where the star has to be fitted.

    :param profile_name: (Optional) Name of the PSF profile to use to
      fit stars. May be 'gaussian' or 'moffat' (default 'gaussian').

    :param amp: (Optional) Amplitude guess, replace the value of the
      automatic estimation (default None).

    :param fwhm_pix: (Optional) Estimate of the FWHM in pixels. If
      None given FWHM is estimated to half the box size (default
      None).
      
    :param beta: (Optional) Beta parameter of the moffat psf. Used
      only if the fitted profile is a Moffat psf (default 3.5).

    :param height: (Optional) Height guess, replace the value of the
      automatic estimation (default None).

    :param pos: (Optional) Position guess as a tuple (x,y), replace
      the value of the automatic estimation (default None).

    :param fix_amp: (Optional) Fix amplitude parameter to its
      estimation (default False)

    :param fix_height: (Optional) Fix height parameter to its
      estimation (default False)

    :param fix_beta: (Optional) Fix beta to the given value (default
      True).

    :param fix_fwhm: (Optional) Fix FWHM to its estimation (default
      False).
      
    :param fix_pos: (Optional) Fix position parameters (x,y) at their
      estimated value (default False).

    :param fit_tol: (Optional) Tolerance on the paramaters fit (the
      lower the better but the longer too) (default 1e-2).

    :param check: (Optional) If True, check fit results for oddities
      (default True).

    :param fwhm_min: (Optional) Minimum valid FWHM [in pixel] of the
      fitted star (default 0.5)

    :param check_reject: (Optional) [Debug] If True, print the reason
      why a fit is rejected (default False).

    :param ron: (Optional) Readout noise in ADU/pixel (default
      10.). estimate_local_noise must be set to False for this noise
      to be taken into account.

    :param dcl: (Optional) Dark current level in ADU/pixel (default
      0.). estimate_local_noise must be set to False for this noise to
      be taken into account.

    :param estimate_local_noise: (Optional) If True, the level of
      noise is computed from the background pixels around the
      stars. ron and dcl are thus not used (default True).

    :param precise_guess: (Optional) If True, the fit guess will be
      more precise but this can lead to errors if the stars positions
      are not already roughly known (default False).
    """
    star_box = star_box.astype(float)
    if np.any(np.isnan(star_box)):
        return []
    if np.all(star_box == 0):
        return []

    # correct star_box level if the background level is < 0
    if np.median(star_box) < 0:
        level_correction = np.min(star_box)
        star_box -= level_correction
    else:
        level_correction = 0.

    def get_background_noise(data, fwhm, posx, posy):
        FWHM_SKY_COEFF = 1.5
        SUB_DIV = 10
        # background noise is computed from the std of 'sky'
        # pixels around the object. The poisson noise is removed
        # because it is added when the real sigma is calculated.
        if fwhm != None:
            # if FWHM is known sky pixels are considered to be at more
            # than 3 sigma of the guessed star position
            S_sky = cutils.surface_value(
                data.shape[0], data.shape[1],
                posx, posy, FWHM_SKY_COEFF * fwhm,
                max(data.shape[0], data.shape[1]), SUB_DIV)
            sky_pixels = data * S_sky
            sky_pixels = sky_pixels[np.nonzero(sky_pixels)]
        else:
            # else a sigma cut is made over the pixels to remove too
            # high values
            sky_pixels = utils.sigmacut(data, sigma=4.)

        mean_sky = utils.robust_mean(utils.sigmacut(sky_pixels, sigma=4.))
        if mean_sky < 0: mean_sky = 0
        background_noise = (
            utils.robust_std(sky_pixels)
            - np.sqrt(mean_sky))
   
        return background_noise
        
    def sigma(data, ron, dcl):
        # guessing sigma as sqrt(photon noise + readout noise^2 + dark
        # current level)
        return np.sqrt(abs(data) + (ron)**2. + dcl)
        
    def diff(free_p, free_i, fixed_p, profile, data, sig):
        data_dimx = data.shape[0]
        data_dimy = data.shape[1]
        params = fixed_p
        params[free_i] = free_p
        prof = profile(params)
        result = (data - prof.array2d(data_dimx, data_dimy)) / sig
        result = result[np.nonzero(~np.isnan(result))]
        result = result[np.nonzero(~np.isinf(result))]
        return result.flatten()

    # Get parameters guess
    guess_params = guess(star_box, pos=pos, height=height,
                         precise_pos=precise_guess)

    # Guessed params are replaced by given values (if not None)
    if height != None:
        guess_params[0] = height
    if amp != None:
        guess_params[1] = amp
    if fwhm_pix != None:
        guess_params[4] = fwhm_pix
    if pos != None:
        if np.size(pos) == 2:
            guess_params[2] = pos[0]
            guess_params[3] = pos[1]
        else:
            Tools()._print_error('Bad position guess : must be a tuple (x,y)')

    # local estimate of the noise
    if estimate_local_noise:
        ron = get_background_noise(star_box, fwhm_pix,
                                   guess_params[2], guess_params[3])
        dcl = 0.
        

    profile = get_profile(profile_name)

    if profile_name == 'moffat':
        guess_params = np.concatenate((guess_params, [beta]))
        
    guess_params = np.array(guess_params, dtype=float)

    fixed_params = np.copy(guess_params)
    masked_params = np.ones_like(guess_params, dtype=np.bool)
    if fix_height:
        masked_params[0] = False
    if fix_amp:
        masked_params[1] = False
    if fix_pos:
        masked_params[2] = False
        masked_params[3] = False
    if fix_fwhm:
        masked_params[4] = False
    if fix_beta and profile_name == 'moffat':
        masked_params[5] = False
    

    free_params = guess_params[np.nonzero(masked_params)]
    free_index = np.arange(guess_params.shape[0])[np.nonzero(masked_params)]
    fixed_params[np.nonzero(masked_params)] = np.nan

    fit_params = optimize.leastsq(diff, free_params,
                                  args=(free_index, fixed_params,
                                        profile, star_box, sigma(star_box,
                                                                 ron, dcl)),
                                  maxfev=100, full_output=True,
                                  xtol=fit_tol)
    if fit_params[-1] <= 4:
        fixed_params[free_index] = fit_params[0]
        cov_x = fit_params[1]
        fit_params = profile(fixed_params).params
        fit_params['fwhm_pix'] = fit_params['fwhm']
        
        # Check fit params for oddities
        if check:
            box_size = min(star_box.shape)
            if not fix_fwhm:
                if fit_params['fwhm_pix'] < fwhm_min:
                    if check_reject: print 'FWHM < fwhm_min'
                    return []
                if fit_params['fwhm_pix'] > box_size:
                    if check_reject: print 'FWHM > box_size'
                    return []
            if not fix_pos:
                if abs(fit_params['x'] - guess_params[2]) > box_size/2.:
                    if check_reject: print 'DX > box_size / 2'
                    return []
                if abs(fit_params['y'] - guess_params[3]) > box_size/2.:
                    if check_reject: print 'DY > box_size / 2'
                    return []
            if not fix_amp:
                if fit_params['amplitude'] < 0.:
                    if check_reject: print 'AMP < 0'
                    return []
            if not fix_amp and not fix_height:
                if ((fit_params['amplitude'] + fit_params['height']
                     - np.min(star_box))
                    < ((np.max(star_box) - np.min(star_box)) * 0.5)):
                    if check_reject: print 'AMP + HEI < 0.5 * max'
                    return []
                if ((fit_params['height'] + fit_params['amplitude']
                     - np.min(star_box))
                    < (np.median(star_box) - np.min(star_box))):
                    if check_reject: print 'AMP + HEI < median'
                    return []
        
        # reduced chi-square
        fit_params['chi-square'] = np.sum(
            diff([],[], fixed_params, profile, star_box,
                 sigma(star_box, ron, dcl))**2.)
            
        fit_params['reduced-chi-square'] = (
            fit_params['chi-square']
            / (np.size(star_box - np.size(free_params))))

        # restore level correction:
        fit_params['height'] += level_correction
        fit_params['amplitude'] += level_correction

        # SNR estimation (from Mighell, MNRAS 361,3 (2005))
        S = fit_params['fwhm_pix']/(2. * math.sqrt(math.log(4)))
        Beta = 4. * math.pi * (S**2.)
        N = np.size(star_box)
        flux = fit_params['flux']
        background_noise = (math.sqrt(abs(fit_params['height'])
                                      + (ron)**2. + dcl))
        SNR = (flux / np.sqrt(
            flux + Beta * (1. + math.sqrt(Beta/N))**2.
            * (background_noise**2.)))
        fit_params['snr'] = SNR
        
        # error estimation
        fit_params['height_err'] = 0.
        fit_params['amplitude_err'] = 0.
        fit_params['x_err'] = 0.
        fit_params['y_err'] = 0.
        fit_params['fwhm_err'] = 0.
        if profile_name == 'moffat':
            fit_params['beta_err'] = 0.
        fit_params['flux_err'] = 0.

        if cov_x != None:
            cov_x *= fit_params['reduced-chi-square']
            for ip in range(len(free_index)):
                err = math.sqrt(abs(cov_x[ip,ip]))
                if free_index[ip] == 0:
                    fit_params['height_err'] = err
                elif free_index[ip] == 1:
                    fit_params['amplitude_err'] = err
                elif free_index[ip] == 2:
                    fit_params['x_err'] = err
                elif free_index[ip] == 3:
                    fit_params['y_err'] = err
                elif free_index[ip] == 4:
                    fit_params['fwhm_err'] = err
                elif free_index[ip] == 5 and profile_name == 'moffat':
                    fit_params['beta_err'] = err
            fit_params['flux_err'] = profile(fixed_params).flux_error(
                fit_params['amplitude_err'],
                fit_params['fwhm_err']/(2. * math.sqrt(math.log(4))))
        else:
            return []
        
        return fit_params
        
    else:
        return []

def aperture_photometry(star_box, fwhm_guess, background_guess=None,
                        background_guess_err=0.,
                        aper_coeff=3., warn=True, x_guess=None,
                        y_guess=None):
    """Return the aperture photometry of a star centered in a star box.

    :param star_box: Star box

    :param fwhm_guess: Guessed FWHM. Used to get the aperture radius.

    :param background_guess: (Optional) If not None, this guess is
      used instead of the background determination in an annulus
      around the star (default None).

    :param background_guess_err: (Optional) Error on the background
      guess. Used to compute the aperture photometry error (default 0.).

    :param aper_coeff: (Optional) Aperture coefficient. The aperture
      radius is Rap = aper_coeff * FWHM. Better when between 1.5 to
      reduce the variation of the collected photons with varying FWHM
      and 3. to account for the flux in the wings (default 3., better
      for star with a high SNR).

    :param warn: If True, print a warning when the background cannot
      be well estimated (default True).

    :param x_guess: (Optional) position of the star along x axis. If
      None, star is assumed to lie at the very center of the frame
      (default None).

    :param y_guess: (Optional) position of the star along y axis. If
      None, star is assumed to lie at the very center of the frame
      (default None).

    :return: A Tuple (flux, aperture surface, bad). If bad estimation:
      bad set to 1, else bad set to 0.
    
    .. note:: Best aperture for maximum S/N: 1. FWHM (Howell 1989,
      Howell 1992). But that works only when the PSF is well sampled
      which is not always the case so a higher aperture coefficient
      may be better. More over, to get exact photometry the result
      must be corrected by aperture growth curve for the 'missing
      light'. A coefficient of 1.27 FWHM corresponds to 3 sigma and
      collects more than 99% of the light. A coefficient of 1.5 reduce
      the variations of the proportion of collected photons with the
      FWHM.

    .. note:: Best radius for sky background annulus is determined
      from this rule of thumb: The number of pixels to estimate the
      background must be al least 3 times the number of pixel in the
      aperture (Merline & Howell 1995). Choosing the aperture radius
      coefficient(Cap) as Rap = Cap * FWHM and the inner radius
      coefficient (Cin) as Rin = Cin * FWHM, gives the outer radius
      coefficient (Cout): Cout = sqrt(3*Cap^2 + Cin^2)

    .. warning:: The star MUST be at the center (+/- a pixel) of the
      star box.
    """
    MIN_APER_SIZE = 0.5 # Minimum warning flux coefficient in the
                        # aperture
    
    C_AP = aper_coeff # Aperture coefficient
    
    C_IN = C_AP + 1. # Inner radius coefficient of the bckg annulus
    
    MIN_BACK_COEFF = 5. # Minimum percentage of the pixels in the
                        # annulus to estimate the background
                        
    SUR_VAL_COEFF = 10 # Number of pixel division to estimate the
                       # surface value

    # Outer radius coefficient of the annulus
    C_OUT = math.sqrt((MIN_BACK_COEFF*C_AP**2.) + C_IN**2.)

    bad = 0
            
    box_dimx = star_box.shape[0]
    box_dimy = star_box.shape[1]
    if x_guess == None:
        x_guess = box_dimx / 2. - 0.5
    if y_guess == None:
        y_guess = box_dimy / 2. - 0.5
                                     
    # Aperture radius
    aper_rmax = C_AP * fwhm_guess

    # Get approximate pixels surface value of the pixels for the aperture
    aperture_surface = cutils.surface_value(box_dimx, box_dimy,
                                            x_guess, y_guess,
                                            0., aper_rmax, SUR_VAL_COEFF)
    total_aperture = np.sum(star_box * aperture_surface)

    if np.sum(aperture_surface) < MIN_APER_SIZE:
        if warn:
            Tools()._print_warning(
                'Not enough pixels in the aperture')
        return np.nan, np.nan, np.nan
    
    # Estimation of the background
    if background_guess == None:
        ann_rmin = math.floor(C_IN * fwhm_guess) + 0.5

        # C_OUT definition just does not work well at small radii, so the
        # outer radius has to be enlarged until we have a good ratio of
        # background counts
        ann_rmax = math.ceil(C_OUT * fwhm_guess)
        not_enough = True
        while not_enough:
            annulus = cutils.surface_value(box_dimx, box_dimy,
                                           x_guess, y_guess,
                                           ann_rmin, ann_rmax,
                                           SUR_VAL_COEFF)
            
            annulus[np.nonzero(annulus < 1.)] = 0. # no partial pixels are used
       
            if (np.sum(annulus) >
                float(MIN_BACK_COEFF) *  np.sum(aperture_surface)):
                not_enough = False
            elif ann_rmax >= min(box_dimx, box_dimy) / 2.:
                not_enough = False
            else:
                ann_rmax += 0.5

        # background in counts / pixel
        if (np.sum(annulus) >
            float(MIN_BACK_COEFF) *  np.sum(aperture_surface)):
            background_pixels = star_box[np.nonzero(annulus)]
            # background level is computed from the mode of the sky
            # pixels distribution
            background, background_err = sky_background_level(
                background_pixels, return_error=True)
            
        else:
            background_pixels = utils.sigmacut(star_box)
            background = utils.robust_mean(background_pixels)
            background_err = (utils.robust_std(background_pixels)
                              / math.sqrt(np.size(background_pixels)))
            
            if warn:
                Tools()._print_warning(
                'Estimation of the background might be bad')
                bad = 1
    else:
        background = background_guess
        background_err = background_guess_err

    aperture_flux = total_aperture - (background *  np.sum(aperture_surface))
    aperture_flux_error = background_err * np.sum(aperture_surface)
    
    return aperture_flux, aperture_flux_error, np.sum(aperture_surface), bad


def fit_stars_in_frame(frame, star_list, box_size,
                       profile_name='gaussian', scale=None,
                       fwhm_pix=None, beta=3.5, fit_tol=1e-2,
                       fwhm_min=0.5, fix_height=True,
                       fix_aperture_fwhm_pix=None, fix_beta=True,
                       fix_fwhm=False, readout_noise=10.,
                       dark_current_level=0., local_background=True,
                       no_aperture_photometry=False,
                       precise_guess=False, aper_coeff=3., blur=False,
                       no_fit=False, estimate_local_noise=True, multi_fit=False,
                       enable_zoom=False, enable_rotation=False,
                       nozero=False, silent=True):
    
    ## WARNING : DO NOT CHANGE THE ORDER OF THE ARGUMENTS OR TAKE CARE
    ## OF THE CALL IN astrometry.Astrometry.fit_stars_in_cube()
  
    """Fit stars in a frame.

    :param frame: The frame from which stars must be fitted.

    :param star_list: A list of star positions as an array of shape
      (star_nb, 2)

    :param box_size: The size of the box created around a star to fit
      its parameter.

    :param profile_name: (Optional) Name of the PSF profile to use to
      fit stars. May be 'gaussian' or 'moffat' (default 'gaussian').

    :param fwhm_pix: (Optional) Estimate of the FWHM in pixels. If
      None given FWHM is estimated to half the box size (default
      None).

    :param scale: (Optional) Scale of the frame in arcsec/pixel. If
      given the fwhm in arcseconds is also computed (keyword:
      'fwhm_arc') with the fit parameters (default None).

    :param beta: (Optional) Beta parameter of the moffat psf. Used
      only if the fitted profile is a Moffat psf (default 3.5).

    :param fix_height: (Optional) Fix height parameter to its
      estimation (default True)

    :param fix_beta: (Optional) Fix beta to the given value (default
      True).

    :param fix_fwhm: (Optional) Fix FWHM to the given value or the
      estimated value (default False).

    :param fit_tol: (Optional) Tolerance on the paramaters fit (the
      lower the better but the longer too) (default 1e-2).

    :param nozero: (Optional) If True do not fit any star which box
      (the pixels around it) contains a zero (default False).

    :param fwhm_min: (Optional) Minimum valid FWHM of the fitted star
      (default 0.5)
      
    :param silent: (Optional) If True no messages are printed (default
      True).

    :param local_background: (Optional) If True, height is estimated
      localy, i.e. around the star. If False, the sky background is
      determined it the whole frame and the height will be the same
      for all the stars, so fix_height option is automatically set to
      True (default True).

    :param fix_aperture_fwhm_pix: (Optional) If a positive float. FWHM
      used to scale aperture size is not computed from the mean FWHM
      in the frame but fixed to the given float (default None).

    :param no_aperture_photometry: (Optional) If True, aperture
      photometry will not be done after profile fitting (default
      False).

    :param precise_guess: (Optional) If True, the fit guess will be
      more precise but this can lead to errors if the stars positions
      are not already roughly known (default False).
          
    :param readout_noise: (Optional) Readout noise in ADU/pixel (can
      be computed from bias frames: std(master_bias_frame)) (default
      10.)
    
    :param dark_current_level: (Optional) Dark current level in
      ADU/pixel (can be computed from dark frames:
      median(master_dark_frame)) (default 0.)

    :param aper_coeff: (Optional) Aperture coefficient. The aperture
      radius is Rap = aper_coeff * FWHM. Better when between 1.5 to
      reduce the variation of the collected photons with varying FWHM
      and 3. to account for the flux in the wings (default 3., better
      for star with a high SNR).

    :param blur: (Optional) If True, blur frame (low pass filtering)
      before fitting stars. It can be used to enhance the quality of
      the fitted flux of undersampled data. Note that the error on
      star position can be greater on blurred frame. This option must
      not be used for alignment purpose (default False).

    :param no_fit: (Optional) If True, no fit is done. Only the
      aperture photometry. Star positions in the star list must thus
      be precise (default False).

    :param multi_fit: (Optional) If True all stars are fitted at the
      same time. More robust for alignment purpose. The difference of
      position between the stars in the star list must be precisely
      known because the overall shift only is estimated (default
      False).

    :param enable_zoom: (Optional) Useful only if multi_fit is True.
      If True, the stars position pattern can be zoomed to better
      adjust it to the real frame (default False).

    :param enable_rotation: (Optional) Useful only if multi_fit is
      True.  If True, the stars position pattern can be rotated to
      better adjust it to the real frame (default False).

    :param estimate_local_noise: (Optional) If True, the level of
      noise is computed from the background pixels around the
      stars. readout_noise and dark_current_level are thus not used
      (default True).
    
    :return: Parameters of a 2D fit of the stars positions.

    .. seealso:: :py:meth:`process.Astrometry.load_star_list` to load
      a predefined list of stars or
      :py:meth:`process.Astrometry.detect_stars` to automatically
      create it.

    .. seealso:: :meth:`process.Astrometry.fit_star`
    
    """        
    BOX_COEFF = 7. # Coefficient to redefine the box size if the
                   # fitted FWHM is too large
    
    BIG_BOX_COEFF = 4. # Coefficient to apply to create a bigger box
                       # than the normal star box. This box is used for
                       # background determination and aperture
                       # photometry

    BLUR_FWHM = 3.5   # FWHM of the gaussian kernel used to blur frames
    BLUR_DEG = int(math.ceil(
        BLUR_FWHM * 2. / (2. * math.sqrt(2. * math.log(2.)))))
    
    dimx = frame.shape[0]
    dimy = frame.shape[1]

    fitted_stars_params = list()
    fit_count = 0

    fit_results = StarsParams(star_list.shape[0], 1,
                              silent=silent)

    star_list = np.array(star_list, dtype=float)
    
    ## Frame background determination if wanted
    if not local_background:
        if precise_guess:
            background = sky_background_level(frame)
        else:
            background = np.median(frame)
        fix_height = True
    else:
        background = None

    ## Blur frame to avoid undersampled data
    if blur:
        fit_frame = np.copy(utils.low_pass_image_filter(frame, deg=BLUR_DEG))
        fwhm_pix = BLUR_FWHM
    else:
        fit_frame = np.copy(frame)

    ## Profile fitting
    if not no_fit:
        if multi_fit:
            fit_params = cutils.multi_fit_stars(
                np.array(frame, dtype=float), np.array(star_list), box_size,
                height_guess=np.array(background, dtype=np.float),
                fwhm_guess=np.array(fwhm_pix, dtype=np.float),
                cov_height=False,
                cov_pos=True,
                cov_fwhm=True,
                fix_height=fix_height,
                fix_pos=False,
                fix_fwhm=fix_fwhm,
                fit_tol=fit_tol,
                ron=np.array(readout_noise, dtype=np.float),
                dcl=np.array(dark_current_level, dtype=np.float),
                enable_zoom=enable_zoom,
                enable_rotation=enable_rotation,
                estimate_local_noise=estimate_local_noise)

            # save results as a StarsParams instance
            for istar in range(star_list.shape[0]):
                if fit_params != []:
                    star_params = dict()
                    p = fit_params['stars-params'][istar, :]
                    e = fit_params['stars-params-err'][istar, :]
                    star_params['height'] = p[0]
                    star_params['height_err'] = e[0]
                    star_params['amplitude'] = p[1]
                    star_params['amplitude_err'] = e[1]
                    star_params['snr'] = (star_params['amplitude']
                                          / star_params['amplitude_err'])
                    star_params['x'] = p[2]
                    star_params['x_err'] = e[2]
                    star_params['y'] = p[3]
                    star_params['y_err'] = e[3]
                    star_params['fwhm'] = p[4]
                    star_params['fwhm_pix'] = star_params['fwhm']
                    star_params['fwhm_err'] = e[4]
                    star_params['chi-square'] = fit_params['chi-square']
                    star_params['reduced-chi-square'] = fit_params[
                        'reduced-chi-square']
                    
                    star_params['flux'] = (get_profile(profile_name))(
                        star_params).flux()
                    star_params['dx'] = star_params['x'] - star_list[istar,0]
                    star_params['dy'] = star_params['y'] - star_list[istar,1]
                    if scale != None:
                        star_params['fwhm_arc'] = (
                            float(star_params['fwhm_pix']) * scale)
                    
                    fit_results[istar] = dict(star_params)
                else:
                    fit_results[istar] = None
                
            
        else:
            for istar in range(star_list.shape[0]):        
                ## Create fit box
                guess = star_list[istar,:]
                if guess.shape[0] == 2:
                    [x_test, y_test] = guess
                elif guess.shape[0] >= 4:
                    [x_test, y_test] = guess[0:2]
                else:
                    Tools()._print_error("The star list must give 2 OR at least 4 parameters for each star [x, y, fwhm_x, fwhm_y]")

                if (x_test > 0 and x_test < dimx
                    and y_test > 0  and y_test < dimy):
                    (x_min, x_max,
                     y_min, y_max) = utils.get_box_coords(
                        x_test, y_test, box_size,
                        0, dimx, 0, dimy)

                    star_box = fit_frame[x_min:x_max, y_min:y_max]
                else:
                    (x_min, x_max, y_min, y_max) = (
                        np.nan, np.nan, np.nan, np.nan)
                    star_box = np.empty((1,1))

                ## Profile Fitting
                if (min(star_box.shape) > float(box_size/2.)
                    and (x_max < dimx) and (x_min >= 0)
                    and (y_max < dimy) and (y_min >= 0)):

                    ## Local background determination for fitting
                    if local_background:
                        background_box_size = BIG_BOX_COEFF * box_size
                        (x_min_back, x_max_back,
                         y_min_back, y_max_back) = utils.get_box_coords(
                            x_test, y_test, background_box_size,
                            0, dimx, 0, dimy)
                        background_box = fit_frame[x_min_back:x_max_back,
                                                   y_min_back:y_max_back]
                        if precise_guess:
                            background = sky_background_level(background_box)
                        else:
                            background = np.median(background_box)


                    if nozero and len(np.nonzero(star_box == 0)[0]) > 0:
                        fit_params = []

                    else:
                        fit_params = fit_star(
                            star_box, profile_name=profile_name,
                            fwhm_pix=fwhm_pix,
                            beta=beta, fix_height=fix_height,
                            fix_beta=fix_beta,
                            fix_fwhm=fix_fwhm,
                            fit_tol=fit_tol,
                            fwhm_min=fwhm_min,
                            height=background,
                            ron=readout_noise,
                            dcl=dark_current_level,
                            precise_guess=precise_guess,
                            estimate_local_noise=estimate_local_noise)
                else:
                    fit_params = []

                if (fit_params != []):
                    fit_count += 1
                    # compute real position in the frame
                    fit_params['x'] += float(x_min)
                    fit_params['y'] += float(y_min)

                    # compute deviation from the position given in the
                    # star list (the center of the star box)
                    fit_params['dx'] = fit_params['x'] - x_test
                    fit_params['dy'] = fit_params['y'] - y_test

                    # compute FWHM in arcsec
                    if scale != None:
                        fit_params['fwhm_arc'] = (float(fit_params['fwhm_pix'])
                                                  * scale)

                    # save results as a StarsParams instance
                    fit_results[istar] = dict(fit_params)
                else:
                    fit_results[istar] = None
    
    ## Compute aperture photometry
    if not no_aperture_photometry:
        
        if not no_fit:
            # get mean FWHM in the frame
            if fix_aperture_fwhm_pix != None:
                if fix_aperture_fwhm_pix > 0.:
                    mean_fwhm = fix_aperture_fwhm_pix
                else:
                    Tools()._print_error(
                        'Fixed FWHM for aperture photometry must be > 0.')
            elif star_list.shape[0] > 1:
                mean_fwhm = utils.robust_mean(utils.sigmacut(
                    fit_results[:,'fwhm_pix']))
            else:
                mean_fwhm = fit_results[0,'fwhm_pix']
        else:
            mean_fwhm = fwhm_pix

        ## Local background determination for aperture
        if local_background:
            background = None
        
        # get aperture given the mean FWHM
        for istar in range(star_list.shape[0]):
            if (fit_results[istar] != None) or (no_fit):
                new_box_size = BOX_COEFF * mean_fwhm
                aperture_box_size = BIG_BOX_COEFF * max(box_size, new_box_size)
                if not no_fit:
                    ix = fit_results[istar, 'x']
                    iy = fit_results[istar, 'y']
                else:
                    ix = star_list[istar,0]
                    iy = star_list[istar,1]
                if ix > 0 and ix < dimx and iy > 0  and iy < dimy:
                    (x_min, x_max,
                     y_min, y_max) = utils.get_box_coords(ix, iy,
                                                          aperture_box_size,
                                                          0, dimx, 0, dimy)
                    star_box = frame[x_min:x_max, y_min:y_max]
                    photom_result = aperture_photometry(
                        star_box, mean_fwhm, background_guess=background,
                        aper_coeff=aper_coeff)

                    if no_fit:
                        fit_params = {'aperture_flux':photom_result[0],
                                      'aperture_flux_err':photom_result[1],
                                      'aperture_surface':photom_result[2],
                                      'aperture_flux_bad':photom_result[3]}
                        fit_results[istar] = dict(fit_params)

                    else:
                        fit_results[istar, 'aperture_flux'] = (
                            photom_result[0])
                        fit_results[istar, 'aperture_flux_err'] = (
                            photom_result[1])
                        fit_results[istar, 'aperture_surface'] = (
                            photom_result[2])
                        fit_results[istar, 'aperture_flux_bad'] = (
                            photom_result[3])
            
    ## Print number of fitted stars
    if not silent:
        Tools()._print_msg("%d/%d stars fitted" %(
            len(fitted_stars_params), star_list.shape[0]))

    return fit_results


def load_star_list(star_list_path, silent=False):
    """Load a list of stars coordinates

    :param star_list_path: The path to the star list file.

    :param silent: (Optional) If True no message is printed (default
      False).

    .. note:: A list of stars is a list of star coordinates (x and
       y). Each set of coordinates is separated by a line
       break. There must not be any blank line or comments.

       For example::

           221.994164678 62.8374036151
           135.052291354 274.848787038
           186.478298303 11.8162949818
           362.642981933 323.083868198
           193.546595814 321.017948051

    The star list can be created using DS9
    (http://hea-www.harvard.edu/RD/ds9/site/Home.html) on the
    first image of the sequence :

          1. Select more than 3 stars with the circular tool (the
             more you select, the better will be the alignment)
          2. Save the regions you have created with the options:

             * Format = 'XY'
             * Coordinate system = 'Image'
    """
    star_list = []
    star_list_file = Tools().open_file(star_list_path, "r")
    for star_coords in star_list_file:
        coords = star_coords.split()
        star_list.append((coords[0], coords[1]))

    star_list = np.array(star_list, dtype=float)
    if not silent:
        Tools()._print_msg("Star list of " + str(star_list.shape[0])
                                +  " stars loaded")
    return star_list


def get_profile(profile_name):
    """Return the PSF profile class corresponding to the given profile name.

    :param profile name: The name of the PSF profile. Must be 'moffat'
      or 'gaussian'.
    
    """
    if profile_name == 'gaussian':
        return Gaussian
    elif profile_name == 'moffat':
        return Moffat
    else:
        Tools()._print_error("Bad profile name (%s) ! Profile name must be 'gaussian' or 'moffat'"%str(profile_name))


def radial_profile(a, xc, yc, rmax):
    """Return the averaged radial profile on a region of a 2D array.

    :param a: A 2D array
    
    :param xc: Center of the profile along x axis
    
    :param yc: Center of the profile along y axis
    
    :param rmax: Radius of the profile

    :return: (R axis, V axis). A tuple of 2 vectors giving the radius
      axis and the corresponding values axis.
    """
    
    xmin = int(math.floor(xc-rmax))
    if xmin < 0: xmin = 0
    xmax = int(round(xc+rmax+1))
    if xmax > a.shape[0]: xmax = a.shape[0]
    ymin = int(math.floor(yc-rmax))
    if ymin < 0: ymin = 0
    ymax = int(round(yc+rmax+1))
    if ymax > a.shape[0]: ymax = a.shape[1]

    # collecting pixels values and their respective radius
    r_list = dict()
    for ii in range(xmin, xmax):
        for ij in range(ymin, ymax):
            r = np.sqrt(float(ii-xc)**2 + float(ij-yc)**2)
            v = a[ii,ij]
            if r in r_list:
                r_list[r] += list([v])
            else:
                r_list[r] = list([v])
            
    # reducing the list by averaging the values for each different radius
    reduced_r_list = list()
    for ir in r_list:
        reduced_r_list.append((ir, np.mean(r_list[ir]), len(r_list[ir])))

    reduced_r_list = np.array(reduced_r_list,
                              dtype=[('r', float), ('v', float), ('n', float)])
    reduced_r_list = np.sort(reduced_r_list, order='r')
    r_axis = np.array([reduced_r_list[ik][0]
                       for ik in range(reduced_r_list.shape[0])])
    v_axis = np.array([reduced_r_list[ik][1]
                       for ik in range(reduced_r_list.shape[0])])

    return (r_axis, v_axis)


def sky_background_level(im, smooth_coeff=0.1, return_mode=False, bins=25,
                         return_error=False):
    """Return the level of the sky background based on the maximum of
    the histogram of the pixels distribution in the image.

    :param im: Image.

    :param smooth_coeff: (Optional) the smoothing degree, i.e. the
      number of smoothing points is defined by smooth_coeff *
      size(histogram) (default 0.05). If smooth_coeff <= 0. no
      smoothing is applied.

    :param return_mode: (Optional) If True the returned value is the
      mode (an entire value for a distribution of integers). If False,
      return the mean of a sigmacut realized around the mode (a
      fractional value, generally more precise).

    :param bins: (Optional) Number of bins for the histogram (default
      20).

    :param return_error: (Optional) If True, the error on the
      estimation is returned (default False).
    """
    sig_im = utils.sigmacut(im, sigma=2.5)
    hist, bin_edges = np.histogram(sig_im, bins=bins)
    
    if np.size(hist) == 0.:
        Tools()._print_warning(
            'Bad sky histogram: returning median of the distribution')
        return np.median(im)
    if smooth_coeff > 0.:
        hist = utils.smooth(hist, deg=int(smooth_coeff * bins) + 1,
                            kind='gaussian_conv')
    index_max = np.argmax(hist)
    mode = (bin_edges[index_max] + bin_edges[index_max+1]) / 2.
    im_cut = utils.sigmacut(im, sigma=2.5, central_value=mode)
    if not return_error:
        return utils.robust_mean(im_cut)
    else:
        return utils.robust_mean(im_cut), (utils.robust_std(im_cut)
                                           / math.sqrt(np.size(im_cut)))
