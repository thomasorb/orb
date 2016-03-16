#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

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
The Core module contains all the core classes of ORB.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'
import version
__version__ = version.__version__

## BASIC IMPORTS
import os
import sys
import warnings
import time
import math
import traceback
import inspect
import re
import datetime

## MODULES IMPORTS
import numpy as np
import bottleneck as bn
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from scipy import interpolate
import h5py

## ORB IMPORTS
import utils.spectrum, utils.parallel, utils.io, utils.filters
import utils.photometry
import constants

#################################################
#### CLASS TextColor ############################
#################################################

class TextColor:
    """
    Define ANSI Escape sequences to display text with colors.
    
    .. warning:: Note that colored text doesn't work on Windows, use
       disable() function to disable coloured text.
    """

    BLUE = '\033[1;94m'
    GREEN = '\033[1;92m'
    PURPLE = '\033[1;95m'
    WARNING = '\033[4;33m'
    ERROR = '\033[4;91m'
    END = '\033[0m'

    def disable(self):
        """ Disable ANSI Escape sequences for windows portability
        """
        self.BLUE = ''
        self.GREEN = ''
        self.PURPLE = ''
        self.WARNING = ''
        self.ERROR = ''
        self.END = ''


#################################################
#### CLASS Tools ################################
#################################################

class Tools(object):
    """
    Parent class of all classes of orb.

    Manage configuration file, implement basic methods and manage the
    server for parallel processing.
    """
    config_file_name = 'config.orb' # Name of the config file
    ncpus = 0 # number of CPUs to use for parallel processing
    
    _MASK_FRAME_TAIL = '_mask.fits' # Tail of a mask frame
    
    _msg_class_hdr = "" # header of any message printed by this class
                        # or its inheritance (see _get_msg_class_hdr()
                        # function)
                        
    _data_path_hdr = "" # first part of the path of all the data files
                        # created during the reduction process.
                        
    _data_prefix = "" # prefix used in the creation of _data_path_hdr
                      # (see _get_data_path_hdr() function)

    _no_log = False # If True no logfile is created

    _tuning_parameters = dict() # Dictionay containing the full names of the
                                # parameter and their new value.

    _silent = False # If True only error messages will be diplayed on screen

    def __init__(self, data_prefix="./temp/data.", no_log=False,
                 tuning_parameters=dict(), 
                 config_file_name='config.orb', silent=False):
        """Initialize Tools class.

        :param data_prefix: (Optional) Prefix used to determine the
          header of the name of each created file (default
          'temp_data')

        :param no_log: (Optional) If True (and if sys.stdout has been
          redirected to Logger) no log file is created (default
          False).

        :param tuning_parameters: (Optional) Some parameters of the
          methods can be tuned externally using this dictionary. The
          dictionary must contains the full parameter name
          (class.method.parameter_name) and its value. For example :
          {'InterferogramMerger.find_alignment.BOX_SIZE': 7}. Note
          that only some parameters can be tuned. This possibility is
          implemented into the method itself with the method
          :py:meth:`core.Tools._get_tuning_parameter`.

        :param config_file_name: (Optional) name of the config file to
          use. Must be located in orb/data/.

        :param silent: If True only error messages will be diplayed on
          screen (default False).
        """
        self.config_file_name = config_file_name
        if (os.name == 'nt'):
            TextColor.disable()
        self._data_prefix = data_prefix
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()
        self._no_log = no_log
        self._tuning_parameters = tuning_parameters
        self.ncpus = int(self._get_config_parameter("NCPUS"))
        self._silent = silent
        warnings.showwarning = self._custom_warn
        if self._no_log:
            if isinstance(sys.stdout, Logger):
                Logger.nolog = True

    def _get_msg_class_hdr(self):
        """Return the header of the displayed messages."""
        return "# " + self.__class__.__name__ + "."
    
    def _get_data_path_hdr(self):
        """Return the header of the created files."""
        return self._data_prefix + self.__class__.__name__ + "."

    def _get_orb_data_file_path(self, file_name):
        """Return the path to a file in ORB data folder: orb/data/file_name

        :param file_name: Name of the file in ORB data folder.
        """
        return os.path.join(os.path.split(__file__)[0], "data", file_name)
        
    def _get_date_str(self):
        """Return local date and hour as a short string 
        for messages"""
        return time.strftime("%y-%m-%d|%H:%M:%S ", time.localtime())

    def _get_config_file_path(self):
        """Return the full path to the configuration file given its name. 

        The configuration file must exist and it must be located in
        orb/data/.

        :param config_file_name: Name of the configuration file.
        """
        config_file_path = self._get_orb_data_file_path(
            self.config_file_name)
        if not os.path.exists(config_file_path):
             self._print_error(
                 "Configuration file %s does not exist !"%config_file_path)
        return config_file_path

    def _get_filter_file_path(self, filter_name):
        """Return the full path to the filter file given the name of
        the filter.

        The filter file name must be filter_FILTER_NAME and it must be
        located in orb/data/.

        :param filter_name: Name of the filter.
        """
        filter_file_path =  self._get_orb_data_file_path(
            "filter_" + filter_name + ".orb")
        if not os.path.exists(filter_file_path):
             self._print_warning(
                 "Filter file %s does not exist !"%filter_file_path)
             return None
         
        return filter_file_path

    
    def _get_phase_file_path(self, filter_name):
        """Return the full path to the phase file given the name of
        the filter.

        The filter name must be filter_FILTER_NAME and the phase file
        must be located in orb/data/.

        :param filter_name: Name of the filter.
        """
        phase_file_path =  self._get_orb_data_file_path(
            "phase_" + filter_name + ".orb")
        
        if not os.path.exists(phase_file_path):
             self._print_warning(
                 "Phase file %s does not exist !"%phase_file_path)
             return None
         
        return phase_file_path

    def _get_optics_file_path(self, filter_name):
        """Return the full path to the optics transmission file given
        the name of the filter.

        The filter file name must be filter_FILTER_NAME and it must be
        located in orb/data/.

        :param filter_name: Name of the filter.
        """
        optics_file_path =  self._get_orb_data_file_path(
            "optics_" + filter_name + ".orb")
        if not os.path.exists(optics_file_path):
             self._print_warning(
                 "Optics file %s does not exist !"%optics_file_path)
             return None
         
        return optics_file_path

    def _get_standard_table_path(self, standard_table_name):
        """Return the full path to the standard table giving name,
        location and type of the recorded standard spectra.

        :param standard_table_name: Name of the standard table file
        """
        standard_table_path = self._get_orb_data_file_path(
            standard_table_name)
        if not os.path.exists(standard_table_path):
             self._print_error(
                 "Standard table %s does not exist !"%standard_table_path)
        return standard_table_path


    def _get_standard_list(self, standard_table_name='std_table.orb',
                           group=None):
        """Return the list of standards recorded in the standard table

        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).
        """
        groups = ['MASSEY', 'MISC', 'CALSPEC', None]
        if group not in groups:
            self._print_error('Group must be in %s'%str(groups))
        std_table = self.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')
        std_list = list()
        for iline in std_table:
            iline = iline.split()
            if len(iline) == 3:
                if group is None:
                    std_list.append(iline[0])
                elif iline[1] == group:
                    std_list.append(iline[0])
                    
        std_list.sort()
        return std_list
    
    def _get_standard_file_path(self, standard_name,
                                standard_table_name='std_table.orb'):
        """
        Return a standard spectrum file path.
        
        :param standard_name: Name of the standard star. Must be
          recorded in the standard table.
        
        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).

        :return: A tuple [standard file path, standard type]. Standard type
          can be 'MASSEY' of 'CALSPEC'.
        """
        std_table = self.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')

        for iline in std_table:
            iline = iline.split()
            if len(iline) >= 3:
                if iline[0] == standard_name:
                    file_path = self._get_orb_data_file_path(iline[2])
                    if os.path.exists(file_path):
                        return file_path, iline[1]

        self._print_error('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')

    def _get_standard_radec(self, standard_name,
                            standard_table_name='std_table.orb'):
        """
        Return a standard spectrum file path.
        
        :param standard_name: Name of the standard star. Must be
          recorded in the standard table.
        
        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).

        :return: A tuple [standard file path, standard type]. Standard type
          can be 'MASSEY' of 'CALSPEC'.
        """
        std_table = self.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')

        for iline in std_table:
            iline = iline.strip().split()
            if len(iline) >= 3:
                if iline[0] == standard_name:
                    if len(iline) > 3:
                        ra = float(iline[3])
                        dec = float(iline[4])
                        return ra, dec
                    else:
                        self._print_error('No RA DEC recorded for standard: {}'.format(
                            standard_name))
                    

        self._print_error('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')


    def _get_atmospheric_extinction_file_path(self):
        """Return the path to the atmospheric extinction file"""
        file_name = self._get_config_parameter('ATM_EXTINCTION_FILE')
        return self._get_orb_data_file_path(file_name)

    def _get_mirror_transmission_file_path(self):
        """Return the path to the telescope mirror transmission file"""
        file_name = self._get_config_parameter('MIR_TRANSMISSION_FILE')
        return self._get_orb_data_file_path(file_name)

    def _get_quantum_efficiency_file_path(self, camera_number):
        """Return the path to the quantum efficiency file

        :param camera_number: Number of the camera, can be 1 or 2.
        """
        file_name = self._get_config_parameter(
            'CAM{}_QE_FILE'.format(camera_number))
        return self._get_orb_data_file_path(file_name)
    
  
    def _get_config_parameter(self, param_key, optional=False):
        """Return a parameter written in a config file located in
          orb/data/

        :param param_key: Key of the parameter to be read

        :param optional: (Optional) If True, a parameter key which is
          not found only raise a warning and the method returns
          None. Else, an error is raised (Default False).

        .. Note:: A parameter key is a string in upper case
          (e.g. PIX_SIZE_CAM1) which starts a line and which must be
          followed in the configuration file by an empty space and the
          parameter (in one word - no empty space). The following
          words on the same line are not read. A line not starting by
          a parameter key is considered as a comment. Please refer to
          the configuration file in orb/data/ folder::
          
             ## ORB configuration file 
             # Author: Thomas Martin <thomas.martin.1@ulaval.ca>
             
             ## Instrumental parameters
             PIX_SIZE_CAM1 20 # Size of one pixel of the camera 1 in um
             PIX_SIZE_CAM2 15 # Size of one pixel of the camera 2 in um  
        """ 
        f = self.open_file(
            self._get_config_file_path(), 'r')
        for line in f:
            if len(line) > 2:
                if line.split()[0] == param_key:
                    return line.split()[1]
        if not optional:
            self._print_error("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
        else:
            self._print_warning("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
            return None


    def _print_error(self, message):
        """Print an error message and raise an exception which will
        stop the execution of the program.

        Error messages are written in the log file.
        
        :param message: The message to be displayed.
        """
        error_msg = self._get_date_str() + self._msg_class_hdr + sys._getframe(1).f_code.co_name + " > Error: " + message
        if isinstance(sys.stdout, Logger):
            sys.stdout.write('\r' + error_msg + '\n', color=TextColor.ERROR)
        else:
            print "\r" + TextColor.ERROR + error_msg + TextColor.END
            
        raise StandardError(message)


    def _print_caller_traceback(self):
        """Print the traceback of the calling function."""
        traceback = inspect.stack()
        traceback_msg = ''
        for i in range(len(traceback))[::-1]:
            traceback_msg += ('  File %s'%traceback[i][1]
                              + ', line %d'%traceback[i][2]
                              + ', in %s\n'%traceback[i][3] +
                              traceback[i][4][0])
            
        print '\r' + traceback_msg

    def _print_warning(self, message, traceback=False):
        """Print a warning message. No exception is raised.
        
        Warning  messages are written in the log file.
        
        :param message: The message to be displayed.

        :param traceback: (Optional) If True, print traceback (default
          False)
        """
        if self._silent: return
        if traceback:
            self._print_caller_traceback()
            
        warning_msg = self._get_date_str() + self._msg_class_hdr + sys._getframe(1).f_code.co_name + " > Warning: " + message
        
        if isinstance(sys.stdout, Logger):
            sys.stdout.write('\r' + warning_msg + '\n', color=TextColor.WARNING)
        else:
            print '\r' + TextColor.WARNING + warning_msg + TextColor.END
        
    def _custom_warn(self, message, category, filename, lineno,
                     file=None, line=None):
        """Redirect warnings thrown by external functions
        (e.g. :py:mod:`orb.utils`) to :py:meth:`orb._print_warning`.

        The parameters are the same parameters as the method
        warnings.showwarning (see the python module warnings).
        """
        self._print_warning(warnings.formatwarning(
            message, category, filename, lineno))

    def _print_msg(self, message, color=False, no_hdr=False):
        """Print a simple message.
        
        Simple messages are written in the log file.

        :param message: The message to be displayed.
        
        :param color: (Optional) If True, the message is diplayed in
          color. If 'alt', an alternative color is displayed.

        :param no_hdr: (Optional) If True, The message is displayed
          as it is, without any header.   
        """
        if self._silent: return
        if not no_hdr:
            message = (self._get_date_str() + self._msg_class_hdr + 
                       sys._getframe(1).f_code.co_name + " > " + message)
            
        text_col = TextColor.BLUE
        if color == 'alt':
            color = True
            text_col = TextColor.PURPLE
            
        if color:
            if isinstance(sys.stdout, Logger):
                sys.stdout.write('\r' + message + '\n', color=text_col)
            else:
                print '\r' + text_col + message + TextColor.END
        else:
            print message
        
    def _print_traceback(self, no_hdr=False):
        """Print a traceback

        :param no_hdr: (Optional) If True, The message is displayed
          as it is, without any header.

        .. note:: This method returns a traceback only if an error has
          been raised.
        """
        message = traceback.format_exc()
        if not no_hdr:
            message = (self._get_date_str() + self._msg_class_hdr + 
                       sys._getframe(1).f_code.co_name + " > " + message)
            
        print "\r" + message
            
    def _update_fits_key(self, fits_name, key, value, comment):
        """Update one key of a FITS file. If the key doesn't exist
        create one.
        
        :param fits_name: Path to the file, can be either
          relative or absolut.
        
        :param key: Exact name of the key to be updated.
        
        :param value: New value of the key

        :param comment: Comment on this key

        .. note:: Please refer to
          http://www.stsci.edu/institute/software_hardware/pyfits/ for
          more information on PyFITS module. And
          http://fits.gsfc.nasa.gov/ for more information on FITS
          files.
        """
        fits_name = (fits_name.splitlines())[0]
        try:
            hdulist = pyfits.open(fits_name, mode='update')
        except:
            self._print_error(
                "The file '%s' could not be opened"%fits_name)
            return None

        prihdr = hdulist[0].header
        prihdr.set(key, value, comment)
        hdulist.flush()
        hdulist.close()
        return True

    def _get_basic_header(self, file_type, config_file_name='config.orb'):
        """
        Return a basic header as a list of tuples (key, value,
        comment) that can be added to any FITS file created by
        :meth:`core.Tools.write_fits`.
        
        :param file_type: Description of the type of file created.
        
        :param config_file_name: (Optional) Name of the configuration
          file (config.orb by default).
        """
        hdr = list()
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','General',''))
        hdr.append(('COMMENT','-------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('FILETYPE', file_type, 'Type of file'))
        hdr.append(('OBSERVAT', self._get_config_parameter(
                    "OBSERVATORY_NAME"), 
                    'Observatory name'))
        hdr.append(('TELESCOP', self._get_config_parameter(
                    "TELESCOPE_NAME"),
                    'Telescope name'))  
        hdr.append(('INSTRUME', self._get_config_parameter(
                    "INSTRUMENT_NAME"),
                    'Instrument name'))
        return hdr
        
    def _get_basic_frame_header(self, dimx, dimy,
                                config_file_name='config.orb'):
        """
        Return a specific part of a header that can be used for image
        files or frames. It creates a false WCS useful to get a rough
        idea of the angular separation between two objects in the
        image.

        The header is returned as a list of tuples (key, value,
        comment) that can be added to any FITS file created by
        :meth:`core.Tools.write_fits`. You can add this part to the
        basic header returned by :meth:`core.Tools._get_basic_header`

        :param dimx: dimension of the first axis of the frame
        :param dimy: dimension of the second axis of the frame
        :param config_file_name: (Optional) Name of the configuration
          file (config.orb by default).
        """
        hdr = list()
        FIELD_OF_VIEW = float(self._get_config_parameter(
            "FIELD_OF_VIEW_1"))
        x_delta = FIELD_OF_VIEW / dimx * (1/60.)
        y_delta = FIELD_OF_VIEW / dimy * (1/60.)
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Frame WCS',''))
        hdr.append(('COMMENT','---------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('CTYPE1', 'DEC--TAN', 'Gnomonic projection'))
        hdr.append(('CRVAL1', 0.000000, 'DEC at reference point'))
        hdr.append(('CUNIT1', 'deg', ''))
        hdr.append(('CRPIX1', 0, 'Pixel coordinate of reference point'))
        hdr.append(('CDELT1', x_delta, 'Degrees per pixel'))
        hdr.append(('CROTA1', 0.000000, ''))
        hdr.append(('CTYPE2', 'RA---TAN', 'Gnomonic projection'))
        hdr.append(('CRVAL2', 0.000000, 'RA at reference point'))
        hdr.append(('CUNIT2', 'deg', ''))
        hdr.append(('CRPIX2', 0, 'Pixel coordinate of reference point'))
        hdr.append(('CDELT2', y_delta, 'Degrees per pixel'))
        hdr.append(('CROTA2', 0.000000, ''))
        return hdr

    def _get_fft_params_header(self, apodization_function):
        """Return a specific part of the header containing the
        parameters of the FFT.

        :param apodization_function: Name of the apodization function.
        """
        hdr = list()
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','FFT Parameters',''))
        hdr.append(('COMMENT','--------------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('APODIZ', '%s'%str(apodization_function),
                    'Apodization function'))
        return hdr

    def _get_basic_spectrum_frame_header(self, frame_index, axis,
                                         wavenumber=False):
        """
        Return a specific part of a header that can be used for
        independant frames of a spectral cube. It gives the wavelength
        position of the frame.

        The header is returned as a list of tuples (key, value,
        comment) that can be added to any FITS file created by
        :meth:`core.Tools.write_fits`. You can add this part to the
        basic header returned by :meth:`core.Tools._get_basic_header`
        and :meth:`core.Tools._get_frame_header`
        
        :param frame_index: Index of the frame.
        
        :param axis: Spectrum axis. The axis must have the same length
          as the number of frames in the cube. Must be an axis in
          wavenumber (cm1) or in wavelength (nm).

        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber (cm1). If False it is considered to be
          in wavelength (nm) (default False).
        """
        hdr = list()
        if not wavenumber:
            wave_type = 'wavelength'
            wave_unit = 'nm'
        else:
            wave_type = 'wavenumber'
            wave_unit = 'cm-1'

        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Frame {}'.format(wave_type),''))
        hdr.append(('COMMENT','----------------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('FRAMENB', frame_index, 'Frame number'))
        
        hdr.append(('WAVEMIN', axis[frame_index], 
                    'Minimum {} in the frame in {}'.format(wave_type,
                                                           wave_unit)))
        hdr.append(('WAVEMAX', axis[frame_index] + axis[1] - axis[0], 
                    'Maximum {} in the frame in {}'.format(wave_type,
                                                           wave_unit)))
        return hdr

    def _get_basic_spectrum_cube_header(self, axis, wavenumber=False):
        """
        Return a specific part of a header that can be used for
        a spectral cube. It creates the wavelength axis of the cube.

        The header is returned as a list of tuples (key, value,
        comment) that can be added to any FITS file created by
        :meth:`core.Tools.write_fits`. You can add this part to the
        basic header returned by :meth:`core.Tools._get_basic_header`
        and :meth:`core.Tools._get_frame_header`

        :param axis: Spectrum axis. The axis must have the same length
          as the number of frames in the cube. Must be an axis in
          wavenumber (cm1) or in wavelength (nm)
          
        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber (cm1). If False it is considered to be
          in wavelength (nm) (default False).
        """
        hdr = list()
        if not wavenumber:
            wave_type = 'wavelength'
            wave_unit = 'nm'
        else:
            wave_type = 'wavenumber'
            wave_unit = 'cm-1'
        hdr.append(('COMMENT','',''))
        hdr.append(('COMMENT','Spectrum axis',''))
        hdr.append(('COMMENT','-------------',''))
        hdr.append(('COMMENT','',''))
        hdr.append(('WAVTYPE', '{}'.format(wave_type.upper()),
                    'Spectral axis type: wavenumber or wavelength'))
        hdr.append(('CTYPE3', 'WAVE', '{} in {}'.format(wave_type,
                                                        wave_unit)))
        hdr.append(('CRVAL3', axis[0], 'Minimum {} in {}'.format(wave_type,
                                                                    wave_unit)))
        hdr.append(('CUNIT3', '{}'.format(wave_unit), ''))
        hdr.append(('CRPIX3', 1.000000, 'Pixel coordinate of reference point'))
        hdr.append(('CDELT3', 
                    ((axis[-1] - axis[0]) / (len(axis) - 1)), 
                    '{} per pixel'.format(wave_unit)))
        hdr.append(('CROTA3', 0.000000, ''))
        return hdr

    def _get_basic_spectrum_header(self, axis, wavenumber=False):
        """
        Return a specific part of a header that can be used for
        a 1D spectrum. It creates the wavelength axis.

        The header is returned as a list of tuples (key, value,
        comment) that can be added to any FITS file created by
        :meth:`core.Tools.write_fits`. You can add this part to the
        basic header returned by :meth:`core.Tools._get_basic_header`
        and :meth:`core.Tools._get_frame_header`

        :param axis: Spectrum axis. The axis must have the same length
          as the spectrum length. Must be an axis in wavenumber (cm1)
          or in wavelength (nm)
          
        :param wavenumber: (Optional) If True the axis is considered
          to be in wavenumber (cm1). If False it is considered to be
          in wavelength (nm) (default False).
        """
        hdr = list()
        if not wavenumber:
            wave_type = 'wavelength'
            wave_unit = 'nm'
        else:
            wave_type = 'wavenumber'
            wave_unit = 'cm-1'
        hdr.append(('CTYPE1', 'WAVE', '{} in {}'.format(
            wave_type, wave_unit)))
        hdr.append(('CRVAL1', axis[0], 'Minimum {} in {}'.format(
            wave_type, wave_unit)))
        hdr.append(('CUNIT1', '{}'.format(wave_unit),
                    'Spectrum coordinate unit'))
        hdr.append(('CRPIX1', 1.000000, 'Pixel coordinate of reference point'))
        hdr.append(('CD1_1', 
                    ((axis[-1] - axis[0]) / (len(axis) - 1)), 
                    '{} per pixel'.format(wave_unit)))
        return hdr

    def _init_pp_server(self, silent=False):
        """Initialize a server for parallel processing.

        :param silent: (Optional) If silent no message is printed
          (Default False).

        .. note:: Please refer to http://www.parallelpython.com/ for
          sources and information on Parallel Python software
        """
        return utils.parallel.init_pp_server(ncpus=self.ncpus,
                                             silent=silent)

    def _close_pp_server(self, js):
        """
        Destroy the parallel python job server to avoid too much
        opened files.

        :param js: job server.
        
        .. note:: Please refer to http://www.parallelpython.com/ for
            sources and information on Parallel Python software.
        """
        return utils.parallel.close_pp_server(js)
        
    def _get_mask_path(self, path):
        """Return the path to the mask given the path to the original
        FITS file.

        :param path: Path of the origial FITS file.
        """
        return os.path.splitext(path)[0] + self._MASK_FRAME_TAIL

    def _get_tuning_parameter(self, parameter_name, default_value):
        """Return the value of the tuning parameter if it exists. In
        the other case return the default value.

        This method is used to help in setting some tuning parameters
        of some method externally.

        .. warning:: The value returned if not the default value will
          be a string.
        
        :param parameter_name: Name of the parameter

        :param default_value: Default value.
        """
        caller_name = (self.__class__.__name__ + '.'
                       + sys._getframe(1).f_code.co_name)
        full_parameter_name = caller_name + '.' + parameter_name
        self._print_msg('looking for tuning parameter: {}'.format(
            full_parameter_name))
        if full_parameter_name in self._tuning_parameters:
            self._print_warning(
                'Tuning parameter {} changed to {} (default {})'.format(
                    full_parameter_name,
                    self._tuning_parameters[full_parameter_name],
                    default_value))
            return self._tuning_parameters[full_parameter_name]
        else:
            return default_value

    def _create_list_from_dir(self, dir_path, list_file_path,
                              image_mode=None, chip_index=None,
                              prebinning=None, check=True):
        """Create a file containing the list of all FITS files at
        a specified directory and returns the path to the list 
        file.

        :param dir_path: Directory containing the FITS files
        
        :param list_file_path: Path to the list file to be created. If None a
          list of the lines is returned.

        :param image_mode: (Optional) Image mode. If not None the given string
          will be written at the first line of the file along with the chip
          index.

        :param chip_index: (Optional) Index of the chip, must be an
          integer. Used in conjonction with image mode to write the first
          directive line of the file which indicates the image mode.

        :param prebinning: (Optional) If not None, must be an integer. Add
          another directive line for data prebinning (default None).


        :param check: (Optional) If True, files dimensions are checked. Else no
          check is done. this is faster but less safe (default True).
        
        :returns: Path to the created list file
        """
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        dir_path = os.path.dirname(str(dir_path))
        if os.path.exists(dir_path):
            list_file_str = list()

            # print image mode directive line
            if image_mode is not None:
                list_file_str.append('# {} {}'.format(image_mode, chip_index))

            # print prebinning directive line
            if prebinning is not None:
                list_file_str.append(
                    '# prebinning {}'.format(int(prebinning)))
    
            file_list = os.listdir(dir_path)
            file_list = [os.path.join(dir_path,_path) for _path in file_list]
            
            # image list sort
            file_list = self.sort_image_list(file_list, image_mode)
            if file_list is None:
                 self._print_error('There is no *.fits file in {}'.format(
                     dir_path))
            
            first_file = True
            file_nb = 0
            if check:
                self._print_msg('Reading and checking {}'.format(dir_path))
            else:
                self._print_msg('Reading {}'.format(dir_path))
            for filename in file_list:
                if (os.path.splitext(filename)[1] == ".fits"
                    and '_bias.fits' not in filename):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.exists(file_path):
                        
                        if first_file:
                            fits_hdul = self.read_fits(
                                file_path, return_hdu_only=True)
                            hdu_data_index = self._get_hdu_data_index(
                                fits_hdul)
                                                     
                            fits_hdu = fits_hdul[hdu_data_index]
                            dimx = fits_hdu.header['NAXIS1']
                            dimy = fits_hdu.header['NAXIS2']
                            first_file = False

                        elif check:
                            fits_hdu = self.read_fits(
                                file_path, return_hdu_only=True)[hdu_data_index]
                            
                            dims = fits_hdu.header['NAXIS']
                            if not (
                                dims == 2
                                or fits_hdu.header['NAXIS1'] == dimx
                                or fits_hdu.header['NAXIS2'] == dimy):
                                self._print_error("All FITS files in the directory %s do not have the same shape. Please remove bad files."%str(dir_path))
                            
                        list_file_str.append(str(file_path))
                        file_nb += 1
                            
                    else:
                        self._print_error(str(file_path) + " does not exists !")
            if file_nb > 0:
                self._print_msg('{} Images found'.format(file_nb))
                if list_file_path is not None:
                    with self.open_file(list_file_path) as list_file:
                        for iline in list_file_str:
                            list_file.write(iline + "\n")
                    return list_file_path
                else:
                    return list_file_str
            else:
                self._print_error('No FITS file in the folder: %s'%dir_path)
        else:
            self._print_error(str(dir_path) + " does not exists !")
        

    def _get_hdu_data_index(self, hdul):
        """Return the index of the first header data unit (HDU) containing data.

        :param hdul: A pyfits.HDU instance
        """
        return utils.io.get_hdu_data_index(hdul)
        
    def write_fits(self, fits_path, fits_data, fits_header=None,
                   silent=False, overwrite=False, mask=None,
                   replace=False, record_stats=False):
        
        """Write data in FITS format. If the file doesn't exist create
        it with its directories.

        If the file already exists add a number to its name before the
        extension (unless 'overwrite' option is set to True).

        :param fits_path: Path to the file, can be either
          relative or absolut.
     
        :param fits_data: Data to be written in the file.

        :param fits_header: (Optional) Optional keywords to update or
          create. It can be a pyfits.Header() instance or a list of
          tuples [(KEYWORD_1, VALUE_1, COMMENT_1), (KEYWORD_2,
          VALUE_2, COMMENT_2), ...]. Standard keywords like SIMPLE,
          BITPIX, NAXIS, EXTEND does not have to be passed.
     
        :param silent: (Optional) If True turn this function won't
          display any message (default False)

        :param overwrite: (Optional) If True overwrite the output file
          if it exists (default False).

        :param mask: (Optional) It not None must be an array with the
          same size as the given data but filled with ones and
          zeros. Bad values (NaN or Inf) are converted to 1 and the
          array is converted to 8 bit unsigned integers (uint8). This
          array will be written to the disk with the same path
          terminated by '_mask'. The header of the mask FITS file will
          be the same as the original data (default None).

        :param replace: (Optional) If True and if the file already
          exist, new data replace old data in the existing file. NaN
          values do not replace old values. Other values replace old
          values. New array MUST have the same size as the existing
          array. Note that if replace is True, overwrite is
          automatically set to True.

        :param record_stats: (Optional) If True, record mean and
          median of data. Useful if they often have to be computed
          (default False).
        
        .. note:: float64 data is converted to float32 data to avoid
          too big files with unnecessary precision

        .. note:: Please refer to
          http://www.stsci.edu/institute/software_hardware/pyfits/ for
          more information on PyFITS module and
          http://fits.gsfc.nasa.gov/ for more information on FITS
          files.
        """
        return utils.io.write_fits(
            fits_path, fits_data, fits_header=fits_header,
            silent=silent, overwrite=overwrite, mask=mask,
            replace=replace, record_stats=record_stats,
            mask_path=self._get_mask_path(fits_path))
        
                
            

    def read_fits(self, fits_path, no_error=False, nan_filter=False, 
                  return_header=False, return_hdu_only=False,
                  return_mask=False, silent=False, delete_after=False,
                  data_index=0, image_mode='classic', chip_index=None,
                  binning=None, fix_header=True, memmap=False, dtype=float):
        """Read a FITS data file and returns its data.
    
        :param fits_path: Path to the file, can be either
          relative or absolut.
        
        :param no_error: (Optional) If True this function will only
          display a warning message if the file does not exist (so it
          does not raise an exception) (default False)
        
        :param nan_filter: (Optional) If True replace NaN by zeros
          (default False)

        :param return_header: (Optional) If True return a tuple (data,
           header) (default False).

        :param return_hdu_only: (Optional) If True return FITS header
          data unit only. No data will be returned (default False).

        :param return_mask: (Optional) If True return only the mask
          corresponding to the data file (default False).

        :param silent: (Optional) If True no message is displayed
          except if an error is raised (default False).

        :param delete_after: (Optional) If True delete file after
          reading (default False).

        :param data_index: (Optional) Index of data in the header data
          unit (Default 0).

        :param image_mode: (Optional) Can be 'sitelle', 'spiomm' or
          'classic'. In 'sitelle' mode, the parameter
          chip_index must also be set to 0 or 1. In this mode only
          one of both SITELLE quadrants is returned. In 'classic' mode
          the whole frame is returned (default 'classic').

        :param chip_index: (Optional) Index of the chip of the
          SITELLE image. Used only if image_mode is set to 'sitelle'
          In this case, must be 1 or 2. Else must be None (default
          None).

        :param binning: (Optional) If not None, returned data is
          binned by this amount (must be an integer >= 1)

        :param fix_header: (Optional) If True, fits header is
          fixed to avoid errors due to header inconsistencies
          (e.g. WCS errors) (default True).

        :param memmap: (Optional) If True, use the memory mapping
          option of pyfits. This is useful to avoid loading a full cube
          in memory when opening a large data cube (default False).

        :param dtype: (Optional) Data is converted to
          the given dtype (e.g. np.float32, default float).
        
        .. note:: Please refer to
          http://www.stsci.edu/institute/software_hardware/pyfits/ for
          more information on PyFITS module. And
          http://fits.gsfc.nasa.gov/ for more information on FITS
          files.
        """
        return utils.io.read_fits(
            fits_path, no_error=no_error, nan_filter=nan_filter, 
            return_header=return_header, return_hdu_only=return_hdu_only,
            return_mask=return_mask, silent=silent, delete_after=delete_after,
            data_index=data_index, image_mode=image_mode, chip_index=chip_index,
            binning=binning, fix_header=fix_header, memmap=memmap, dtype=dtype,
            mask_path=self._get_mask_path(fits_path))

    def _bin_image(self, a, binning):
        """Return mean binned image. 

        :param image: 2d array to bin.

        :param binning: binning (must be an integer >= 1).

        ..note:: Only the complete sets of rows or columns are binned
          so that depending on the bin size and the image size the
          last columns or rows can be ignored. This ensures that the
          binning surface is the same for every pixel in the binned
          array.
        """
        return utils.io.bin_image(a, binning)


    def _get_sitelle_slice(self, slice_str):
        """
        Strip a string containing SITELLE like slice coordinates.

        :param slice_str: Slice string.
        """
        return utils.io.get_sitelle_slice(slice_str)


    def _read_sitelle_chip(self, hdu, chip_index, substract_bias=True):
        """Return chip data of a SITELLE FITS image.

        :param hdu: pyfits.HDU Instance of the SITELLE image
        
        :param chip_index: Index of the chip to read. Must be 1 or 2.

        :param substract_bias: If True bias is automatically
          substracted by using the overscan area (default True).
        """
        return utils.io.read_sitelle_chip(hdu, chip_index,
                                          substract_bias=substract_bias)


    def _read_spiomm_data(self, hdu, image_path, substract_bias=True):
        """Return data of an SpIOMM FITS image.

        :param hdu: pyfits.HDU Instance of the SpIOMM image

        :param image_path: Image path
        
        :param substract_bias: If True bias is automatically
          substracted by using the associated bias frame as an
          overscan frame. Mean bias level is thus computed along the y
          axis of the bias frame (default True).
        """
        return utils.io.read_spiomm_data(hdu, image_path,
                                         substract_bias=substract_bias)
        

    def open_file(self, file_name, mode='w'):
        """Open a file in write mode (by default) and return a file
        object.
        
        Create the file if it doesn't exist (only in write mode).

        :param fits_name: Path to the file, can be either
          relative or absolute.

        :param mode: (Optional) Can be 'w' for write mode, 'r' for
          read mode and 'a' for append mode.
        """
        if mode not in ['w','r','a','rU']:
            self._print_error("mode option must be 'w', 'r', 'rU' or 'a'")
            
        if mode in ['w','a']:
            # create folder if it does not exist
            dirname = os.path.dirname(file_name)
            if dirname != '':
                if not os.path.exists(dirname): 
                    os.makedirs(dirname)
        if mode == 'r': mode = 'rU' # read in universal mode by
                                    # default to handle Windows files.

        return open(file_name, mode)
    

    def sort_image_list(self, file_list, image_mode, cube=True):
        """Sort a list of fits files.

        :param file_list: A list of file names

        :param image_mode: Image mode, can be 'sitelle' or 'spiomm'.

        :param cube: If True, image list is considered as a cube
          list. Headers are used to get the right order based on step
          number instead of file path (default True).
        """
    
        file_list = [path for path in file_list if
                     (('.fits' in path) or ('.hdf5' in path))]
        
        if len(file_list) == 0: return None

        if image_mode == 'spiomm':
            file_list = [path for path in file_list
                         if not '_bias.fits' in path]

        # get all numbers
        file_seq = [re.findall("[0-9]+", path)
                        for path in file_list if
                    (('.fits' in path) or ('.hdf5' in path))]
        
        try:
            file_keys = np.array(file_seq, dtype=int)
        except Exception, e:
            self._print_error('Malformed sequence of files: {}:\n{}'.format(
                e, file_seq))
                             
            
        # get changing column
        test = np.sum(file_keys == file_keys[0,:], axis=0)
        
        if np.min(test) > 1:
            self._print_warning('Images list cannot be safely sorted. Two images at least have the same index')
            column_index = np.nan
        else:
            column_index = np.argmin(test)

        # get changing step (if possible)
        steplist = list()
        if cube:
            for path in file_list:
                if '.fits' in path:
                    try:
                        hdr = self.read_fits(
                            path, return_hdu_only=True)[0].header
                        if 'SITSTEP' in hdr:
                            steplist.append(int(hdr['SITSTEP']))
                    except Exception: pass
                
        
        if len(steplist) == len(file_list):
            _list = list()
            for i in range(len(file_list)):
                _list.append({'path':file_list[i], 'step':steplist[i]})
            _list.sort(key=lambda x: x['step'])
            file_list = [_path['path'] for _path in _list]
        elif not np.isnan(column_index):
            file_list.sort(key=lambda x: float(re.findall("[0-9]+", x)[
                column_index]))
        else:
            self._print_error('Image list cannot be sorted.')
            
        return file_list

    def _clean_sip(self, hdr):
        """Clean a hdr from all but the SIP keywords.

        :param hdr: The header to clean
        """
        ## sip_keys = ('WCSAXES', 'CRPIX1', 'CRPIX2', 'CTYPE1', 'CTYPE2',
        ##             'BP_ORDER', 'A_1_1', 'BP_2_0', 'B_1_1', 'B_ORDER',
        ##             'B_2_0', 'BP_0_2', 'AP_2_0', 'A_0_2', 'BP_1_1',
        ##             'BP_1_0', 'A_2_0', 'A_ORDER', 'AP_1_0', 'AP_1_1',
        ##             'AP_ORDER', 'BP_0_1', 'AP_0_1', 'B_0_2', 'AP_0_2')

        if 'CTYPE1' in hdr:
            if hdr['CTYPE1'] != 'RA---TAN-SIP':
                self._print_error('This file does not contain any SIP transformation parameters')
                
        clean_hdr = pywcs.WCS(hdr).to_fits(relax=True)[0].header
     
        return clean_hdr

    def save_sip(self, fits_path, hdr, overwrite=True):
        """Save SIP parameters from a header to a blanck FITS file.

        :param fits_path: Path to the FITS file
        :param hdr: header from which SIP parameters must be read
        :param overwrite: (Optional) Overwrite the FITS file.
        """    
        clean_hdr = self._clean_sip(hdr)
        data = np.empty((1,1))
        data.fill(np.nan)
        self.write_fits(
            fits_path, data, fits_header=clean_hdr, overwrite=overwrite)

    def load_sip(self, fits_path):
        """Return a astropy.wcs.WCS object from a FITS file containing
        SIP parameters.
    
        :param fits_path: Path to the FITS file    
        """
        hdr = self.read_fits(fits_path, return_hdu_only=True)[0].header
        clean_hdr = self._clean_sip(hdr)
        return pywcs.WCS(clean_hdr)


    def open_hdf5(self, file_path, mode):
        """Return a :py:class:`h5py.File` instance with some
        informations.

        :param file_path: Path to the hdf5 file.
        
        :param mode: Opening mode. Can be 'r', 'r+', 'w', 'w-', 'x',
          'a'.

        .. note:: Please refer to http://www.h5py.org/.
        """
        if mode in ['w', 'a', 'w-', 'x']:
            # create folder if it does not exist
            dirname = os.path.dirname(file_path)
            if dirname != '':
                if not os.path.exists(dirname): 
                    os.makedirs(dirname)
                    
        f = h5py.File(file_path, mode)
        
        if mode in ['w', 'a', 'w-', 'x', 'r+']:
            f.attrs['program'] = 'Generated by ORB version {}'.format(
                __version__)
            f.attrs['class'] = str(self.__class__.__name__)
            f.attrs['author'] = 'Thomas Martin (thomas.martin.1@ulaval.ca)'
            f.attrs['date'] = str(datetime.datetime.now())
            
        return f

    def write_hdf5(self, file_path, data, header=None,
                   silent=False, overwrite=False, max_hdu_check=True,
                   compress=False):

        """    
        Write data in HDF5 format.

        A header can be added to the data. This method is useful to
        handle an HDF5 data file like a FITS file. It implements most
        of the functionality of the method
        :py:meth:`core.Tools.write_fits`.

        .. note:: The output HDF5 file can contain mutiple data header
          units (HDU). Each HDU is in a spcific group named 'hdu*', *
          being the index of the HDU. The first HDU is named
          HDU0. Each HDU contains one data group (HDU*/data) which
          contains a numpy.ndarray and one header group
          (HDU*/header). Each subgroup of a header group is a keyword
          and its associated value, comment and type.

        :param file_path: Path to the HDF5 file to create

        :param data: A numpy array (numpy.ndarray instance) of numeric
          values. If a list of arrays is given, each array will be
          placed in a specific HDU. The header keyword must also be
          set to a list of headers of the same length.

        :param header: (Optional) Optional keywords to update or
          create. It can be a pyfits.Header() instance or a list of
          tuples [(KEYWORD_1, VALUE_1, COMMENT_1), (KEYWORD_2,
          VALUE_2, COMMENT_2), ...]. Standard keywords like SIMPLE,
          BITPIX, NAXIS, EXTEND does not have to be passed (default
          None). It can also be a list of headers if a list of arrays
          has been passed to the option 'data'.    

        :param max_hdu_check: (Optional): When True, ff the input data
          is a list (interpreted as a list of data unit), check if
          it's length is not too long to make sure that the input list
          is not a single data array that has not been converted to a
          numpy.ndarray format. If the number of HDU to create is
          indeed very long this can be set to False (default True).
        
        :param silent: (Optional) If True turn this function won't
          display any message (default False)

        :param overwrite: (Optional) If True overwrite the output file
          if it exists (default False).

        :param compress: (Optional) If True data is compressed using
          the SZIP library (see
          https://www.hdfgroup.org/doc_resource/SZIP/). SZIP library
          must be installed (default False).
    

        .. note:: Please refer to http://www.h5py.org/.
        """
        MAX_HDUS = 3

        start_time = time.time()
        
        # change extension if nescessary
        if os.path.splitext(file_path)[1] != '.hdf5':
            file_path = os.path.splitext(file_path)[0] + '.hdf5'
    
        # Check if data is a list of arrays.
        if not isinstance(data, list):
            data = [data]

        if max_hdu_check and len(data) > MAX_HDUS:
            self._print_error('Data list length is > {}. As a list is interpreted has a list of data unit make sure to pass a numpy.ndarray instance instead of a list. '.format(MAX_HDUS))

        # Check header format
        if header is not None:
        
            if isinstance(header, pyfits.Header):
                header = [header]
                
            elif isinstance(header, list):
                
                if (isinstance(header[0], list)
                    or isinstance(header[0], tuple)):
                    
                    header_seems_ok = False
                    if (isinstance(header[0][0], list)
                        or isinstance(header[0][0], tuple)):
                        # we have a list of headers
                        if len(header) == len(data):
                            header_seems_ok = True
                            
                    elif isinstance(header[0][0], str):
                        # we only have one header
                        if len(header[0]) > 2:
                            header = [header]
                            header_seems_ok = True
                            
                    if not header_seems_ok:
                        self._print_error('Badly formated header')
                            
                elif not isinstance(header[0], pyfits.Header):
                    
                    self._print_error('Header must be a pyfits.Header instance or a list')

            else:
                self._print_error('Header must be a pyfits.Header instance or a list')
            

            if len(header) != len(data):
                self._print_error('The number of headers must be the same as the number of data units.')
            

        # change path if file exists and must not be overwritten
        new_file_path = str(file_path)
        if not overwrite and os.path.exists(new_file_path):
            index = 0
            while os.path.exists(new_file_path):
                new_file_path = (os.path.splitext(file_path)[0] + 
                                 "_" + str(index) + 
                                 os.path.splitext(file_path)[1])
                index += 1
                

        # open file
        with self.open_hdf5(new_file_path, 'w') as f:
        
            ## add data + header
            for i in range(len(data)):

                idata = data[i]
                
                # Check if data has a valid format.
                if not isinstance(idata, np.ndarray):
                    try:
                        idata = np.array(idata, dtype=float)
                    except Exception, e:
                        self._print_error('Data to write must be convertible to a numpy array of numeric values: {}'.format(e))


                # convert data to float32
                if idata.dtype == np.float64:
                    idata = idata.astype(np.float32)

                # hdu name
                hdu_group_name = 'hdu{}'.format(i)
                if compress:
                    data_group = f.create_dataset(
                        hdu_group_name + '/data', data=idata,
                        compression='lzf', compression_opts=None)
                        #compression='szip', compression_opts=('nn', 32))
                        #compression='gzip', compression_opts=9)
                else:
                    data_group = f.create_dataset(
                        hdu_group_name + '/data', data=idata)
                
                # add header
                if header is not None:
                    iheader = header[i]
                    if not isinstance(iheader, pyfits.Header):
                        iheader = pyfits.Header(iheader)

                    f[hdu_group_name + '/header'] = self._header_fits2hdf5(
                        iheader)

        self._print_msg('Data written as {} in {:.2f} s'.format(
            new_file_path, time.time() - start_time))
        
        return new_file_path

    def _header_fits2hdf5(self, fits_header):
        """convert a pyfits.Header() instance to a header for an hdf5 file

        :param fits_header: Header of the FITS file
        """
        hdf5_header = list()
        
        for ikey in range(len(fits_header)):
            _tstr = str(type(fits_header[ikey]))
            ival = np.array(
                (fits_header.keys()[ikey], str(fits_header[ikey]),
                 fits_header.comments[ikey], _tstr))
            
            hdf5_header.append(ival)
        return np.array(hdf5_header)

    def _header_hdf52fits(self, hdf5_header):
        """convert an hdf5 header to a pyfits.Header() instance.

        :param hdf5_header: Header of the HDF5 file
        """
        def cast(a, t_str):
            for _t in [int, float, bool, str, np.int64,
                       np.float64, long, np.float128]:
                if t_str == repr(_t):
                    return _t(a)
            raise Exception('Bad type string {}'.format(t_str))
                    
        fits_header = pyfits.Header()
        for i in range(hdf5_header.shape[0]):
            ival = hdf5_header[i,:]
            if ival[3] != 'comment':
                fits_header[ival[0]] = (cast(ival[1], ival[3]), str(ival[2]))
            else:
                fits_header['comment'] = ival[1]
        return fits_header
        
    def read_hdf5(self, file_path, return_header=False, dtype=float):
        
        """Read an HDF5 data file created with
        :py:meth:`core.Tools.write_hdf5`.
        
        :param file_path: Path to the file, can be either
          relative or absolute.        

        :param return_header: (Optional) If True return a tuple (data,
           header) (default False).
    
        :param dtype: (Optional) Data is converted to the given type
          (e.g. np.float32, default float).
        
        .. note:: Please refer to http://www.h5py.org/."""
        
        
        with self.open_hdf5(file_path, 'r') as f:
            data = list()
            header = list()
            for hdu_name in f:
                data.append(f[hdu_name + '/data'][:].astype(dtype))
                if return_header:
                    if hdu_name + '/header' in f:
                        # extract header
                        header.append(
                            self._header_hdf52fits(f[hdu_name + '/header'][:]))
                    else: header.append(None)

        if len(data) == 1:
            if return_header:
                return data[0], header[0]
            else:
                return data[0]
        else:
            if return_header:
                return data, header
            else:
                return data
            
    def _get_hdf5_data_path(self, frame_index, mask=False):
        """Return path to the data of a given frame in an HDF5 cube.

        :param frame_index: Index of the frame
        
        :param mask: (Optional) If True, path to the masked frame is
          returned (default False).
        """
        if mask: return self._get_hdf5_frame_path(frame_index) + '/mask'
        else: return self._get_hdf5_frame_path(frame_index) + '/data'

    def _get_hdf5_quad_data_path(self, quad_index):
        """Return path to the data of a given quad in an HDF5 cube.

        :param quad_index: Index of the quadrant
        """
        return self._get_hdf5_quad_path(quad_index) + '/data'
        

    def _get_hdf5_header_path(self, frame_index):
        """Return path to the header of a given frame in an HDF5 cube.

        :param frame_index: Index of the frame
        """
        return self._get_hdf5_frame_path(frame_index) + '/header'

    def _get_hdf5_quad_header_path(self, quad_index):
        """Return path to the header of a given quadrant in an HDF5 cube.

        :param frame_index: Index of the quadrant
        """
        return self._get_hdf5_quad_path(quad_index) + '/header'

    def _get_hdf5_frame_path(self, frame_index):
        """Return path to a given frame in an HDF5 cube.

        :param frame_index: Index of the frame.
        """
        return 'frame{:05d}'.format(frame_index)

    
    def _get_hdf5_quad_path(self, quad_index):
        """Return path to a given quadrant in an HDF5 cube.

        :param quad_index: Index of the quad.
        """
        return 'quad{:03d}'.format(quad_index)
        
            
    def _get_quadrant_dims(self, quad_number, dimx, dimy, div_nb):
        """Return the indices of a quadrant along x and y axes.

        :param quad_number: Quadrant number

        :param dimx: X axis dimension.
          
        :param dimy: Y axis dimension.
        
        :param div_nb: Number of divisions along x and y axes. (e.g. if
          div_nb = 3, the number of quadrant is 9 ; if div_nb = 4, the
          number of quadrant is 16)

        """
        quad_nb = div_nb**2

        if (quad_number < 0) or (quad_number > quad_nb - 1L):
            self._print_error("quad_number out of bounds [0," + str(quad_nb- 1L) + "]")
            return None

        index_x = quad_number % div_nb
        index_y = (quad_number - index_x) / div_nb

        x_min = long(index_x * math.ceil(dimx / div_nb))
        if (index_x != div_nb - 1L):            
            x_max = long((index_x  + 1L) * math.ceil(dimx / div_nb))
        else:
            x_max = dimx

        y_min = long(index_y * math.ceil(dimy / div_nb))
        if (index_y != div_nb - 1L):            
            y_max = long((index_y  + 1L) * math.ceil(dimy / div_nb))
        else:
            y_max = dimy

        return x_min, x_max, y_min, y_max

            
##################################################
#### CLASS Cube ##################################
##################################################
class Cube(Tools):
    """
    Generate and manage a **virtual frame-divided cube**.

    .. note:: A **frame-divided cube** is a set of frames grouped
      together by a list.  It avoids storing a data cube in one large
      data file and loading an entire cube to process it.

    This class has been designed to handle large data cubes. Its data
    can be accessed virtually as if it was loaded in memory.

    .. code-block:: python
      :linenos:

      cube = Cube('liste') # A simple list is enough to initialize a Cube instance
      quadrant = Cube[25:50, 25:50, :] # Here you just load a small quadrant
      spectrum = Cube[84,58,:] # load spectrum at pixel [84,58]
    """
    # Processing
    ncpus = None # number of CPUs to use for parallel processing
    DIV_NB = None # number of division of the frames in quadrants
    QUAD_NB = None # number of quadrant = DIV_NB**2
    BIG_DATA = None # If True some processes are parallelized

    image_list_path = None
    image_list = None
    dimx = None
    dimy = None
    dimz = None
    
    star_list = None
    z_median = None
    z_mean = None
    z_std = None
    mean_image = None

    overwrite = None

    _hdf5 = None
    _data_prefix = None
    _project_header = None
    _calibration_laser_header = None
    _wcs_header = None

    _silent_load = False

    _mask_exists = None
    
    _return_mask = False # When True, __get_item__ return mask data
                         # instead of 'normal' data

    # read directives
    _image_mode = 'classic' # opening mode. can be sitelle, spiomm or classic
    _chip_index = None # in sitelle mode, this gives the index of
                       # the chip to read
    _prebinning = None # prebinning directive

    
    _parallel_access_to_data = None # authorize parallel access to
                                    # data (False for HDF5)

    is_complex = None # tell if cube's data is complex.
    dtype = None # data type
    
    def __init__(self, image_list_path, image_mode='classic',
                 chip_index=1, binning=1, data_prefix="./temp/data.",
                 config_file_name="config.orb", project_header=list(),
                 wcs_header=list(), calibration_laser_header=list(),
                 overwrite=False, silent_init=False, no_log=False,
                 tuning_parameters=dict(), indexer=None, no_sort=False):
        
        """
        Initialize Cube class.
       
        :param image_list_path: Path to the list of images which form
          the virtual cube. If image_list_path is set to '' then
          this class will not try to load any data.  Can be useful
          when the user don't want to use or process any data.

        :param image_mode: (Optional) Image mode. Can be 'spiomm',
          'sitelle' or 'classic'. In 'sitelle' mode bias, is
          automatically substracted and the overscan regions are not
          present in the data cube. The chip index option can also be
          used in this mode to read only one of the two chips
          (i.e. one of the 2 cameras). In 'spiomm' mode, if
          :file:`*_bias.fits` frames are present along with the image
          frames, bias is substracted from the image frames, this
          option is used to precisely correct the bias of the camera
          2. In 'classic' mode, the whole array is extracted in its
          raw form (default 'classic').

        :param chip_index: (Optional) Useful only in 'sitelle' mode
          (see image_mode option). Gives the number of the ship to
          read. Must be an 1 or 2 (default 1).

        :param binning: (Optional) Data is pre-binned numerically by
          this amount. i.e. 1000x1000xN raw frames with a prebinning
          of 2 will give a cube of 500x500xN (default 1).
    
        :param config_file_name: (Optional) name of the config file to
          use. Must be located in orb/data/ (default 'config.orb').

        :param data_prefix: (Optional) Prefix used to determine the
          header of the name of each created file (default
          'temp_data').

        :param project_header: (Optional) header section describing
          the observation parameters that can be added to each output
          files (an empty list() by default).

        :param wcs_header: (Optional) header section describing WCS
          that can be added to each created image files (an empty
          list() by default).

        :param calibration_laser_header: (Optional) header section
          describing the calibration laser parameters that can be
          added to the concerned output files e.g. calibration laser map,
          spectral cube (an empty list() by default).

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param silent_init: (Optional) If True no message is displayed
          at initialization.

        :param no_log: (Optional) If True no log file is created
          (default False).

        :param tuning_parameters: (Optional) Some parameters of the
          methods can be tuned externally using this dictionary. The
          dictionary must contains the full parameter name
          (class.method.parameter_name) and its value. For example :
          {'InterferogramMerger.find_alignment.BOX_SIZE': 7}. Note
          that only some parameters can be tuned. This possibility is
          implemented into the method itself with the method
          :py:meth:`core.Tools._get_tuning_parameter`.

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param no_sort: (Optional) If True, no sort of the file list
          is done. Files list is taken as is (default False).
        """
        self.is_complex = False
        self.dtype = float
        self.overwrite = overwrite
        self._no_log = no_log
        if self._no_log:
            if isinstance(sys.stdout, Logger):
                Logger.nolog = True
                
        self.indexer = indexer
        
        # read config file to get parameters
        self.config_file_name=config_file_name
        self.DIV_NB = int(self._get_config_parameter(
            "DIV_NB"))
        self.BIG_DATA = bool(int(self._get_config_parameter(
            "BIG_DATA")))
        self.ncpus = int(self._get_config_parameter(
            "NCPUS"))
 
        self._data_prefix = data_prefix
        self._project_header = project_header
        self._wcs_header = wcs_header
        self._calibration_laser_header = calibration_laser_header
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()
        self._tuning_parameters = tuning_parameters
        
        self.QUAD_NB = self.DIV_NB**2L
        self.image_list_path = image_list_path

        self._image_mode = image_mode
        self._chip_index = chip_index
        self._prebinning = binning

        self._parallel_access_to_data = True

        if (self.image_list_path != ""):
            # read image list and get cube dimensions  
            image_list_file = self.open_file(self.image_list_path, "r")
            image_name_list = image_list_file.readlines()
            if len(image_name_list) == 0:
                self._print_error('No image path in the given image list')
            is_first_image = True
            
            for image_name in image_name_list:
                image_name = (image_name.splitlines())[0]    
                
                if self._image_mode == 'spiomm' and '_bias' in image_name:
                    spiomm_bias_frame = True
                else: spiomm_bias_frame = False
                
                if is_first_image:
                    # check list parameter
                    if '#' in image_name:
                        if 'sitelle' in image_name:
                            self._image_mode = 'sitelle'
                            self._chip_index = int(image_name.split()[-1])
                        elif 'spiomm' in image_name:
                            self._image_mode = 'spiomm'
                            self._chip_index = None
                        elif 'prebinning' in image_name:
                            self._prebinning = int(image_name.split()[-1])
                            
                    elif not spiomm_bias_frame:
                        self.image_list = [image_name]

                        # detect if hdf5 format or not
                        if os.path.splitext(image_name)[1] == '.hdf5':
                            self._hdf5 = True
                        elif os.path.splitext(image_name)[1] == '.fits':
                            self._hdf5 = False
                        else:
                            self._print_error("Unrecognized extension of file {}. File extension must be '*.fits' or '*.hdf5' depending on its format.".format(image_name))

                        if self._hdf5 :
                            if self._image_mode != 'classic': self._print_warning("Image mode changed to 'classic' because 'spiomm' and 'sitelle' modes are not supported in hdf5 format.")
                            if self._prebinning != 1: self._print_warning("Prebinning is not supported for images in hdf5 format")
                            self._image_mode = 'classic'
                            self._prebinning = 1
        
                        if not self._hdf5:
                            image_data = self.read_fits(
                                image_name,
                                image_mode=self._image_mode,
                                chip_index=self._chip_index,
                                binning=self._prebinning)
                            self.dimx = image_data.shape[0]
                            self.dimy = image_data.shape[1]
                            
                            hdul = self.read_fits(
                                image_name, return_hdu_only=True)
                            
                        else:
                            with self.open_hdf5(image_name, 'r') as f:
                                if 'hdu0/data' in f:
                                    shape = f['hdu0/data'].shape
                                    
                                    if len(shape) == 2:
                                        self.dimx, self.dimy = shape
                                    else: self._print_error('Image shape must have 2 dimensions: {}'.format(shape))
                                else: self._print_error('Bad formatted hdf5 file. Use Tools.write_hdf5 to get a correct hdf5 file for ORB.')
                            
                        is_first_image = False
                        # check if masked frame exists
                        if os.path.exists(self._get_mask_path(image_name)):
                            self._mask_exists = True
                        else:
                            self._mask_exists = False
                            
                elif (self._MASK_FRAME_TAIL not in image_name
                      and not spiomm_bias_frame):
                    self.image_list.append(image_name)

            image_list_file.close()

            # image list is sorted
            if not no_sort:
                self.image_list = self.sort_image_list(self.image_list,
                                                       self._image_mode)
            
            self.image_list = np.array(self.image_list)
            self.dimz = self.image_list.shape[0]
            
            
            if (self.dimx) and (self.dimy) and (self.dimz):
                if not silent_init:
                    self._print_msg("Data shape : (" + str(self.dimx) 
                                    + ", " + str(self.dimy) + ", " 
                                    + str(self.dimz) + ")")
            else:
                self._print_error("Incorrect data shape : (" 
                                  + str(self.dimx) + ", " + str(self.dimy) 
                                  + ", " +str(self.dimz) + ")")


    def __getitem__(self, key):
        """Implement the evaluation of self[key].
        
        .. note:: To make this function silent just set
          Cube()._silent_load to True.
        """
        # check return mask possibility
        if self._return_mask and not self._mask_exists:
            self._print_error("No mask found with data, cannot return mask")
        
        # produce default values for slices
        x_slice = self._get_default_slice(key[0], self.dimx)
        y_slice = self._get_default_slice(key[1], self.dimy)
        z_slice = self._get_default_slice(key[2], self.dimz)
        
        # get first frame
        data = self._get_frame_section(x_slice, y_slice, z_slice.start)
        
        # return this frame if only one frame is wanted
        if z_slice.stop == z_slice.start + 1L:
            return data

        if self._parallel_access_to_data:
            # load other frames
            job_server, ncpus = self._init_pp_server(silent=self._silent_load) 

            if not self._silent_load:
                progress = ProgressBar(z_slice.stop - z_slice.start - 1L)
            
            for ik in range(z_slice.start + 1L, z_slice.stop, ncpus):
                # No more jobs than frames to compute
                if (ik + ncpus >= z_slice.stop): 
                    ncpus = z_slice.stop - ik

                added_data = np.empty((x_slice.stop - x_slice.start,
                                       y_slice.stop - y_slice.start, ncpus),
                                      dtype=float)

                jobs = [(ijob, job_server.submit(
                    self._get_frame_section,
                    args=(x_slice, y_slice, ik+ijob),
                    modules=("numpy as np",)))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    added_data[:,:,ijob] = job()

                data = np.dstack((data, added_data))
                if not self._silent_load:
                    progress.update(ik - z_slice.start, info="Loading data")
            if not self._silent_load:
                progress.end()
            self._close_pp_server(job_server)
        else:
            if not self._silent_load:
                progress = ProgressBar(z_slice.stop - z_slice.start - 1L)
            
            for ik in range(z_slice.start + 1L, z_slice.stop):

                added_data = self._get_frame_section(x_slice, y_slice, ik)

                data = np.dstack((data, added_data))
                if not self._silent_load:
                    progress.update(ik - z_slice.start, info="Loading data")
            if not self._silent_load:
                progress.end()
            
        return np.squeeze(data)

    def _get_default_slice(self, _slice, _max):
        """Utility function used by __getitem__. Return a valid slice
        object given an integer or slice.

        :param _slice: a slice object or an integer
        :param _max: size of the considered axis of the slice.
        """
        if isinstance(_slice, slice):
            if _slice.start is not None:
                if (isinstance(_slice.start, int)
                    or isinstance(_slice.start, long)):
                    if (_slice.start >= 0) and (_slice.start <= _max):
                        slice_min = int(_slice.start)
                    else:
                        self._print_error(
                            "Index error: list index out of range")
                else:
                    self._print_error("Type error: list indices of slice must be integers")
            else: slice_min = 0

            if _slice.stop is not None:
                if (isinstance(_slice.stop, int)
                    or isinstance(_slice.stop, long)):
                    if _slice.stop < 0: # transform negative index to real index
                        slice_stop = _max + _slice.stop
                    else:  slice_stop = _slice.stop
                    if ((slice_stop <= _max)
                        and slice_stop > slice_min):
                        slice_max = int(slice_stop)
                    else:
                        self._print_error(
                            "Index error: list index out of range")

                else:
                    self._print_error("Type error: list indices of slice must be integers")
            else: slice_max = _max

        elif isinstance(_slice, int) or isinstance(_slice, long):
            slice_min = _slice
            slice_max = slice_min + 1
        else:
            self._print_error("Type error: list indices must be integers or slices")
        return slice(slice_min, slice_max, 1)

    def _get_frame_section(self, x_slice, y_slice, frame_index):
        """Utility function used by __getitem__.

        Return a section of one frame in the cube.

        :param x_slice: slice object along x axis.
        :param y_slice: slice object along y axis.
        :param frame_index: Index of the frame.

        .. warning:: This function must only be used by
           __getitem__. To get a frame section please use the method
           :py:meth:`orb.core.get_data_frame` or
           :py:meth:`orb.core.get_data`.
        """
        if not self._hdf5:
            hdu = self.read_fits(self.image_list[frame_index],
                                 return_hdu_only=True,
                                 return_mask=self._return_mask)
        else:
            hdu = self.open_hdf5(self.image_list[frame_index], 'r')
        image = None
        stored_file_path = None
        if self._prebinning > 1:
            # already binned data is stored in a specific folder
            # to avoid loading more than one time the same image.
            # check if already binned data exists

            if not self._hdf5:
                stored_file_path = os.path.join(
                    os.path.split(self._get_data_path_hdr())[0],
                    'STORED',
                    (os.path.splitext(
                        os.path.split(self.image_list[frame_index])[1])[0]
                     + '.{}.bin{}.fits'.format(self._image_mode, self._prebinning)))
            else:
                raise Exception(
                    'prebinned data is not handled for hdf5 cubes')

            if os.path.exists(stored_file_path):
                image = self.read_fits(stored_file_path)


        if self._image_mode == 'sitelle': # FITS only
            if image is None:
                image = self._read_sitelle_chip(hdu, self._chip_index)
                image = self._bin_image(image, self._prebinning)
            section = image[x_slice, y_slice]

        elif self._image_mode == 'spiomm': # FITS only
            if image is None:
                image, header = self._read_spiomm_data(
                    hdu, self.image_list[frame_index])
                image = self._bin_image(image, self._prebinning)
            section = image[x_slice, y_slice]

        else:
            if self._prebinning > 1: # FITS only
                if image is None:
                    image = np.copy(
                        hdu[0].data.transpose())
                    image = self._bin_image(image, self._prebinning)
                section = image[x_slice, y_slice]
            else:
                if image is None: # HDF5 and FITS
                    if not self._hdf5:
                        section = np.copy(
                            hdu[0].section[y_slice, x_slice].transpose())
                    else:
                        section = hdu['hdu0/data'][x_slice, y_slice]

                else: # FITS only
                    section = image[y_slice, x_slice].transpose()
        del hdu

         # FITS only
        if stored_file_path is not None and image is not None:
            self.write_fits(stored_file_path, image, overwrite=True,
                            silent=True)

        self._return_mask = False # always reset self._return_mask to False
        return section

    def is_same_2D_size(self, cube_test):
        """Check if another cube has the same dimensions along x and y
        axes.

        :param cube_test: Cube to check
        """
        
        if ((cube_test.dimx == self.dimx) and (cube_test.dimy == self.dimy)):
            return True
        else:
            return False

    def is_same_3D_size(self, cube_test):
        """Check if another cube has the same dimensions.

        :param cube_test: Cube to check
        """
        
        if ((cube_test.dimx == self.dimx) and (cube_test.dimy == self.dimy) 
            and (cube_test.dimz == self.dimz)):
            return True
        else:
            return False    

    def get_data_frame(self, index, silent=False, mask=False):
        """Return one frame of the cube.

        :param index: Index of the frame to be returned

        :param silent: (Optional) if False display a progress bar
          during data loading (default False)

        :param mask: (Optional) if True return mask (default False).
        """
        if silent:
            self._silent_load = True
        if mask:
            self._return_mask = True
        data = self[:,:,index]
        self._silent_load = False
        self._return_mask = False
        return data

    def get_data(self, x_min, x_max, y_min, y_max,
                 z_min, z_max, silent=False, mask=False):
        """Return a part of the data cube.

        :param x_min: minimum index along x axis
        
        :param x_max: maximum index along x axis
        
        :param y_min: minimum index along y axis
        
        :param y_max: maximum index along y axis
        
        :param z_min: minimum index along z axis
        
        :param z_max: maximum index along z axis
        
        :param silent: (Optional) if False display a progress bar
          during data loading (default False)

        :param mask: (Optional) if True return mask (default False).
        """
        if silent:
            self._silent_load = True
        if mask:
            self._return_mask = True
        data = self[x_min:x_max, y_min:y_max, z_min:z_max]
        self._silent_load = False
        self._return_mask = False
        return data
        
    
    def get_all_data(self, mask=False):
        """Return the whole data cube

        :param mask: (Optional) if True return mask (default False).
        """
        if mask:
            self._return_mask = True
        data = self[:,:,:]
        self._return_mask = False
        return data
    
    def get_column_data(self, icol, silent=True, mask=False):
        """Return data as a slice along x axis

        :param icol: Column index
        
        :param silent: (Optional) if False display a progress bar
          during data loading (default True)
          
        :param mask: (Optional) if True return mask (default False).
        """
        if silent:
            self._silent_load = True
        if mask:
            self._return_mask = True
        data = self.get_data(icol, icol+1, 0, self.dimy, 0, self.dimz,
                             silent=silent)
        self._silent_load = False
        self._return_mask = False
        return data

    def get_frame_header(self, index):
        """Return the header of a frame given its index in the list.

        The header is returned as an instance of pyfits.Header().

        :param index: Index of the frame

        .. note:: Please refer to
          http://www.stsci.edu/institute/software_hardware/pyfits/ for
          more information on PyFITS module and
          http://fits.gsfc.nasa.gov/ for more information on FITS
          files.
        """
        if not self._hdf5:
            hdu = self.read_fits(self.image_list[index],
                                 return_hdu_only=True)
            return hdu[0].header
        else:
            hdu = self.open_hdf5(self.image_list[index], 'r')
            if 'hdu0/header' in hdu:
                return hdu['hdu0/header']
            else: return pyfits.Header()

    
    def get_cube_header(self):
        """
        Return the header of a cube from the header of the first frame
        by keeping only the general keywords.
        """
        def del_key(key, header):
            if '*' in key:
                key = key[:key.index('*')]
                for k in header.keys():
                    if key in k:
                        header.remove(k)
            else:
                while key in header:
                    header.remove(key)
            return header
                
        cube_header = self.get_frame_header(0)
        cube_header = del_key('COMMENT', cube_header)
        cube_header = del_key('EXPNUM', cube_header)
        cube_header = del_key('BSEC*', cube_header)
        cube_header = del_key('DSEC*', cube_header)
        cube_header = del_key('FILENAME', cube_header)
        cube_header = del_key('PATHNAME', cube_header)
        cube_header = del_key('OBSID', cube_header)
        cube_header = del_key('IMAGEID', cube_header)
        cube_header = del_key('CHIPID', cube_header)
        cube_header = del_key('DETSIZE', cube_header)
        cube_header = del_key('RASTER', cube_header)
        cube_header = del_key('AMPLIST', cube_header)
        cube_header = del_key('CCDSIZE', cube_header)
        cube_header = del_key('DATASEC', cube_header)
        cube_header = del_key('BIASSEC', cube_header)
        cube_header = del_key('CSEC1', cube_header)
        cube_header = del_key('CSEC2', cube_header)
        cube_header = del_key('TIME-OBS', cube_header)
        cube_header = del_key('DATEEND', cube_header)
        cube_header = del_key('TIMEEND', cube_header)
        cube_header = del_key('SITNEXL', cube_header)
        cube_header = del_key('SITPZ*', cube_header)
        cube_header = del_key('SITSTEP', cube_header)
        cube_header = del_key('SITFRING', cube_header)
        
        return cube_header


    def get_size_on_disk(self):
        """Return the expected size of the cube if saved on disk in Mo.
        """
        return self.dimx * self.dimy * self.dimz * 4 / 1e6 # 4 octets in float32

    def get_resized_frame(self, index, size_x, size_y, degree=3):
        """Return a resized frame using spline interpolation.

        :param index: Index of the frame to resize
        :param size_x: New size of the cube along x axis
        :param size_y: New size of the cube along y axis
        :param degree: (Optional) Interpolation degree (Default 3)
        
        .. warning:: To use this function on images containing
           star-like objects a linear interpolation must be done
           (set degree to 1).
        """
        resized_frame = np.empty((size_x, size_y), dtype=self.dtype)
        x = np.arange(self.dimx)
        y = np.arange(self.dimy)
        x_new = np.linspace(0, self.dimx, num=size_x)
        y_new = np.linspace(0, self.dimy, num=size_y)
        z = self.get_data_frame(index)
        interp = interpolate.RectBivariateSpline(x, y, z, kx=degree, ky=degree)
        resized_frame = interp(x_new, y_new)
        data = np.array(resized_frame)
        dimx = data.shape[0]
        dimy = data.shape[1]
        self._print_msg("Data resized to shape : (" + str(dimx) +  ", " + str(dimy) + ")")
        return resized_frame

    def get_resized_data(self, size_x, size_y):
        """Resize the data cube and return it using spline
          interpolation.
          
        This function is used only to resize a cube of flat or dark
        frames. Note that resizing dark or flat frames must be
        avoided.

        :param size_x: New size of the cube along x axis
        :param size_y: New size of the cube along y axis
        
        .. warning:: This function must not be used to resize images
          containing star-like objects (a linear interpopolation must
          be done in this case).
        """
        resized_cube = np.empty((size_x, size_y, self.dimz), dtype=self.dtype)
        x = np.arange(self.dimx)
        y = np.arange(self.dimy)
        x_new = np.linspace(0, self.dimx, num=size_x)
        y_new = np.linspace(0, self.dimy, num=size_y)
        progress = ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            z = self.get_data_frame(_ik)
            interp = interpolate.RectBivariateSpline(x, y, z)
            resized_cube[:,:,_ik] = interp(x_new, y_new)
            progress.update(_ik, info="resizing cube")
        progress.end()
        data = np.array(resized_cube)
        dimx = data.shape[0]
        dimy = data.shape[1]
        dimz = data.shape[2]
        self._print_msg("Data resized to shape : (" + str(dimx) +  ", " + str(dimy) + ", " + str(dimz) + ")")
        return data

    def get_interf_energy_map(self):
        """Return the energy map of an interferogram cube."""
        mean_map = self.get_mean_image()
        energy_map = np.zeros((self.dimx, self.dimy), dtype=self.dtype)
        progress = ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            energy_map += np.abs(self.get_data_frame(_ik) - mean_map)**2.
            progress.update(_ik, info="Creating interf energy map")
        progress.end()
        return np.sqrt(energy_map / self.dimz)

    def get_spectrum_energy_map(self):
        """Return the energy map of a spectrum cube.
    
        .. note:: In this process NaNs are considered as zeros.
        """
        energy_map = np.zeros((self.dimx, self.dimy), dtype=self.dtype)
        progress = ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            frame = self.get_data_frame(_ik)
            non_nans = np.nonzero(~np.isnan(frame))
            energy_map[non_nans] += (frame[non_nans])**2.
            progress.update(_ik, info="Creating spectrum energy map")
        progress.end()
        return np.sqrt(energy_map) / self.dimz

    def get_mean_image(self, recompute=False):
        """Return the mean image of a cube (corresponding to a deep
        frame for an interferogram cube or a specral cube).

        :param recompute: (Optional) Force to recompute mean image
          even if it is already present in the cube (default False).
        
        .. note:: In this process NaNs are considered as zeros.
        """
        if self.mean_image is None or recompute:
            mean_im = np.zeros((self.dimx, self.dimy), dtype=self.dtype)
            progress = ProgressBar(self.dimz)
            for _ik in range(self.dimz):
                frame = self.get_data_frame(_ik)
                non_nans = np.nonzero(~np.isnan(frame))
                mean_im[non_nans] += frame[non_nans]
                progress.update(_ik, info="Creating mean image")
            progress.end()
            self.mean_image = mean_im / self.dimz
        return self.mean_image

    def get_median_image(self):
        """Return the median image of a cube

        .. note:: This is not a real median image. Frames are combined
          10 by 10 with a median.
        """
        median_im = np.zeros((self.dimx, self.dimy), dtype=self.dtype)
        progress = ProgressBar(self.dimz)
        BLOCK_SIZE = 10
        counts = 0
        for _ik in range(0, self.dimz, BLOCK_SIZE):
            progress.update(_ik, info="Creating median image")
            _ik_end = _ik+BLOCK_SIZE
            if _ik_end > self.dimz - 1:
                _ik_end = self.dimz - 1
            frames = self.get_data(0,self.dimx,
                                   0, self.dimy,
                                   _ik,_ik_end, silent=True)
            median_im += np.nanmedian(frames, axis=2)
            counts += 1
        progress.end()
        return median_im / counts

    def get_zstd(self, nozero=False, center=False):
        """Return a vector containing frames std.

        :param nozero: If True zeros are removed from the std
          computation. If there's only zeros, std frame value will
          be a NaN (default False).

        :param center: If True only the center of the frame is used to
          compute std.
        """
        return self.get_zstat(nozero=nozero, center=center, stat='std')
    
    def get_zmean(self, nozero=False, center=False):
        """Return a vector containing frames median.

        :param nozero: If True zeros are removed from the median
          computation. If there's only zeros, median frame value will
          be a NaN (default False).

        :param center: If True only the center of the frame is used to
          compute median.
        """
        return self.get_zstat(nozero=nozero, center=center, stat='mean')

    def get_zmedian(self, nozero=False, center=False):
        """Return a vector containing frames mean.

        :param nozero: If True zeros are removed from the mean
          computation. If there's only zeros, mean frame value will
          be a NaN (default False).

        :param center: If True only the center of the frame is used to
          compute mean.
        """
        return self.get_zstat(nozero=nozero, center=center, stat='median')


    def _get_frame_stat(self, ik, nozero, stat_key, center,
                        xmin, xmax, ymin, ymax):
        """Utilitary function for :py:meth:`orb.orb.Cube.get_zstat`
        which returns the stats of a frame in a box.

        Check if the frame stats are not already in the header of the
        frame.    
        """
        if not nozero and not center:
            frame_hdr = self.get_frame_header(ik)
            if stat_key in frame_hdr:
                if not np.isnan(float(frame_hdr[stat_key])):
                    return float(frame_hdr[stat_key])

        frame = np.copy(self.get_data(
            xmin, xmax, ymin, ymax, ik, ik+1)).astype(float)

        if nozero: # zeros filtering
            frame[np.nonzero(frame == 0)] = np.nan

        if bn.allnan(frame, axis=None):
            return np.nan
        if stat_key == 'MEDIAN':
            return bn.nanmedian(frame, axis=None)
        elif stat_key == 'MEAN':
            return bn.nanmean(frame, axis=None)
        elif stat_key == 'STD':
            return bn.nanstd(frame, axis=None)
        else: self._print_error('stat_key must be set to MEDIAN, MEAN or STD')
        

    def get_zstat(self, nozero=False, stat='mean', center=False):
        """Return a vector containing frames stat (mean, median or
        std).

        :param nozero: If True zeros are removed from the stat
          computation. If there's only zeros, stat frame value will
          be a NaN (default False).

        :param stat: Type of stat to return. Can be 'mean', 'median'
          or 'std'

        :param center: If True only the center of the frame is used to
          compute stat.
        """
        BORDER_COEFF = 0.15
        
        if center:
            xmin = int(self.dimx * BORDER_COEFF)
            xmax = self.dimx - xmin + 1
            ymin = int(self.dimy * BORDER_COEFF)
            ymax = self.dimy - ymin + 1
        else:
            xmin = 0
            xmax = self.dimx
            ymin = 0
            ymax = self.dimy

        if stat == 'mean':
            if self.z_mean is None: stat_key = 'MEAN'
            else: return self.z_mean
            
        elif stat == 'median':
            if self.z_median is None: stat_key = 'MEDIAN'
            else: return self.z_median
            
        elif stat == 'std':
            if self.z_std is None: stat_key = 'STD'
            else: return self.z_std
            
        else: self._print_error(
            "Bad stat option. Must be 'mean', 'median' or 'std'")
        
        stat_vector = np.empty(self.dimz, dtype=self.dtype)

        job_server, ncpus = self._init_pp_server()
        progress = ProgressBar(self.dimz)
        for ik in range(0, self.dimz, ncpus):
            if ik + ncpus >= self.dimz:
                ncpus = self.dimz - ik

            jobs = [(ijob, job_server.submit(
                self._get_frame_stat, 
                args=(ik + ijob, nozero, stat_key, center,
                      xmin, xmax, ymin, ymax),
                modules=("import numpy as np",
                         "import bottleneck as bn")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                stat_vector[ik+ijob] = job()   

            progress.update(ik, info="Creating stat vector")
        progress.end()
        self._close_pp_server(job_server)
        
        if stat == 'mean':
            self.z_mean = stat_vector
            return self.z_mean
        if stat == 'median':
            self.z_median = stat_vector
            return self.z_median
        if stat == 'std':
            self.z_std = stat_vector
            return self.z_std
            
    def get_quadrant_dims(self, quad_number, div_nb=None,
                          dimx=None, dimy=None):
        """Return the indices of a quadrant along x and y axes.

        :param quad_number: Quadrant number

        :param div_nb: (Optional) Use another number of divisions
          along x and y axes. (e.g. if div_nb = 3, the number of
          quadrant is 9 ; if div_nb = 4, the number of quadrant is 16)

        :param dimx: (Optional) Use another x axis dimension. Other
          than the axis dimension of the managed data cube
          
        :param dimy: (Optional) Use another y axis dimension. Other
          than the axis dimension of the managed data cube
        """
        if div_nb is None: 
            div_nb = self.DIV_NB
        quad_nb = div_nb**2
        if dimx is None: dimx = self.dimx
        if dimy is None: dimy = self.dimy

        return Tools._get_quadrant_dims(
            self, quad_number, dimx, dimy, div_nb)
       
    def export(self, export_path, x_range=None, y_range=None,
               z_range=None, header=None, overwrite=False,
               force_hdf5=False, force_fits=False):
        
        """Export cube as one FITS/HDF5 file.

        :param export_path: Path of the exported FITS file

        :param x_range: (Optional) Tuple (x_min, x_max) (default
          None).
        
        :param y_range: (Optional) Tuple (y_min, y_max) (default
          None).
        
        :param z_range: (Optional) Tuple (z_min, z_max) (default
          None).

        :param header: (Optional) Header of the output file (default
          None).

        :param overwrite: (Optional) Overwrite output file (default
          False).

        :param force_hdf5: (Optional) If True, output is in HDF5
          format even if the input files are FITS files. If False it
          will be in the format of the input files (default False).

        :param force_fits: (Optional) If True, output is in FITS
          format. If False it will be in the format of the input files
          (default False).
        """
        if force_fits and force_hdf5:
            self._print_error('force_fits and force_hdf5 cannot be both set to True')
        
        if x_range is None:
            xmin = 0
            xmax = self.dimx
        else:
            xmin = np.min(x_range)
            xmax = np.max(x_range)
            
        if y_range is None:
            ymin = 0
            ymax = self.dimy
        else:
            ymin = np.min(y_range)
            ymax = np.max(y_range)

        if z_range is None:
            zmin = 0
            zmax = self.dimz
        else:
            zmin = np.min(z_range)
            zmax = np.max(z_range)
           
            
        if (self._hdf5 or force_hdf5) and not force_fits: # HDF5 export
            self._print_msg('Exporting cube to an HDF5 cube: {}'.format(
                export_path))
            if not self.is_quad_cube:
                outcube = OutHDFCube(
                    export_path,
                    (xmax - xmin, ymax - ymin, zmax - zmin),
                    overwrite=overwrite)
            else:
                outcube = OutHDFQuadCube(
                    export_path,
                    (xmax - xmin, ymax - ymin, zmax - zmin),
                    self.QUAD_NB,
                    overwrite=overwrite)
            
            outcube.append_image_list(self.image_list)
            if header is not None:
                outcube.append_header(header)

            if not self.is_quad_cube: # frames export
                job_server, ncpus = self._init_pp_server()
                progress = ProgressBar(zmax-zmin)

                data_frames = np.empty((xmax - xmin, ymax - ymin, ncpus),
                                       dtype=self.dtype)


                for iframe in range(0, zmax-zmin, ncpus):
                    progress.update(
                        iframe, info='exporting data frame {}'.format(
                            iframe))
                    if iframe + ncpus >= zmax - zmin:
                        ncpus = zmax - zmin - iframe

                    # get data
                    jobs = [(ijob, job_server.submit(
                        self.get_data, 
                        args=(xmin, xmax, ymin, ymax, zmin + iframe +ijob,
                              zmin + iframe +ijob + 1)))
                            for ijob in range(ncpus)]
                    for ijob, job in jobs:
                        data_frames[:,:,ijob] = job()

                    # get header
                    jobs = [(ijob, job_server.submit(
                        self.get_frame_header, 
                        args=(zmin + iframe +ijob,)))
                            for ijob in range(ncpus)]

                    # write data + header
                    for ijob, job in jobs:
                        outcube.write_frame(zmin + iframe + ijob,
                                            data=data_frames[:,:,ijob],
                                            header=job(),
                                            force_float32=True)
                        
                    progress.end()
                self._close_pp_server(job_server)
            else: # quad export
                
                progress = ProgressBar(self.QUAD_NB)
                for iquad in range(self.QUAD_NB):
                    progress.update(
                        iquad, info='exporting quad {}'.format(
                            iquad))
                    
                    x_min, x_max, y_min, y_max = self._get_quadrant_dims(
                        iquad, self.dimx, self.dimy, int(math.sqrt(float(self.quad_nb))))
                    
                    data_quad = self.get_data(x_min, x_max, y_min, y_max, 0, self.dimz,
                                              silent=True)

                    # write data
                    outcube.write_quad(iquad,
                                       data=data_quad,
                                       force_float32=True)

                progress.end()
                
            outcube.close()
            del outcube
                
        else: # FITS export
            self._print_msg('Exporting cube to a FITS cube: {}'.format(
                export_path))
            data = np.empty((xmax-xmin, ymax-ymin, zmax-zmin), dtype=float)
        
            job_server, ncpus = self._init_pp_server()
            progress = ProgressBar(zmax-zmin)
            for iframe in range(0, zmax-zmin, ncpus):
                progress.update(iframe,
                                info='exporting data frame {}'.format(
                    iframe))
                if iframe + ncpus >= zmax - zmin:
                    ncpus = zmax - zmin - iframe

                jobs = [(ijob, job_server.submit(
                    self.get_data, 
                    args=(xmin, xmax, ymin, ymax, zmin + iframe +ijob,
                          zmin + iframe +ijob + 1)))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    data[:,:,iframe+ijob] = job()
                
            progress.end()        
            self._close_pp_server(job_server)
        
            self.write_fits(export_path, data, overwrite=overwrite,
                            fits_header=header)


##################################################
#### CLASS ProgressBar ###########################
##################################################


class ProgressBar:
    """Display a simple progress bar in the terminal

    :param max_index: Index representing a 100% completed task.
    """

    REFRESH_COUNT = 3L # number of steps used to calculate a remaining time
    MAX_CARAC = 78 # Maximum number of characters in a line

    _start_time = None
    _max_index = None
    _bar_length = 10.
    _max_index = None
    _count = 0
    _time_table = []
    _index_table = []
    _silent = False

    def __init__(self, max_index, silent=False):
        """Initialize ProgressBar class

        :param max_index: The index considered as 100%. If 0 print a
          'please wait' message.

        :param silent: (Optional) If True progress bar is not printed
          (default False).
        """
        self._start_time = time.time()
        self._max_index = float(max_index)
        self._time_table = np.zeros((self.REFRESH_COUNT), np.float)
        self._index_table = np.zeros((self.REFRESH_COUNT), np.float)
        self._silent = silent
        self.count = 0
        
    def _erase_line(self):
        """Erase the progress bar"""
        if not self._silent:
            if isinstance(sys.stdout, Logger):
                sys.stdout.write("\r" + " " * self.MAX_CARAC, nolog=True)
            else:
                sys.stdout.write("\r" + " " * self.MAX_CARAC)
            sys.stdout.flush()

    def _time_str_convert(self, sec):
        """Convert a number of seconds in a human readable string
        
        :param sec: Number of seconds to convert
        """
        if (sec < 1):
            return '{:.3f} s'.format(sec)
        elif (sec < 5):
            return '{:.2f} s'.format(sec)
        elif (sec < 60.):
            return '{:.1f} s'.format(sec)
        elif (sec < 3600.):
            minutes = int(math.floor(sec/60.))
            seconds = int(sec - (minutes * 60.))
            return str(minutes) + "m" + str(seconds) + "s"
        else:
            hours = int(math.floor(sec/3600.))
            minutes = int(math.floor((sec - (hours*3600.))/60.))
            seconds = int(sec - (hours * 3600.) - (minutes * 60.))
            return str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s"


    def update(self, index, info="", remains=True, nolog=True):
        """Update the progress bar.

        :param index: Index representing the progress of the
          process. Must be less than index_max.
          
        :param info: (Optional) Information to be displayed as
          comments (default '').
          
        :param remains: (Optional) If True, remaining time is
          displayed (default True).

        :param nolog: (Optional) No logging of the printed text is
          made (default True).
        """
        if (self._max_index > 0):
            color = TextColor.BLUE
            self._count += 1
            for _icount in range(self.REFRESH_COUNT - 1L):
                self._time_table[_icount] = self._time_table[_icount + 1L]
                self._index_table[_icount] = self._index_table[_icount + 1L]
            self._time_table[-1] = time.time()
            self._index_table[-1] = index
            if (self._count > self.REFRESH_COUNT):
                index_by_step = ((self._index_table[-1] - self._index_table[0])
                                 /float(self.REFRESH_COUNT - 1))
                time_to_end = (((self._time_table[-1] - self._time_table[0])
                                /float(self.REFRESH_COUNT - 1))
                               * (self._max_index - index) / index_by_step)
            else:
                time_to_end = 0.
            pos = (float(index) / self._max_index) * self._bar_length
            line = ("\r [" + "="*int(math.floor(pos)) + 
                    " "*int(self._bar_length - math.floor(pos)) + 
                    "] [%d%%] [" %(pos*100./self._bar_length) + 
                    str(info) +"]")
            if remains:
                line += (" [remains: " + 
                         self._time_str_convert(time_to_end) + "]")
            
        else:
            color = TextColor.GREEN
            line = ("\r [please wait] [" +
                    str(info) +"]")
            
        self._erase_line()
        if (len(line) > self.MAX_CARAC):
            rem_len = len(line) - self.MAX_CARAC + 1
            line = line[:-rem_len]
        if not self._silent:
            if isinstance(sys.stdout, Logger):
                sys.stdout.write(line, color=color, nolog=nolog)
            else:
                sys.stdout.write(line)
            sys.stdout.flush()

    def end(self, silent=False):
        """End the progress bar and display the total time needed to
        complete the process.

        :param silent: If True remove the progress bar from the
          screen. Further diplayed text will be displayed above the
          progress bar.
        """
        
        if not silent:
            self._erase_line()
            self.update(self._max_index, info="completed in " +
                        self._time_str_convert(
                            time.time() - self._start_time),
                        remains=False, nolog=False)
            if not self._silent:
                if isinstance(sys.stdout, Logger):
                    sys.stdout.write('\n', nolog=False)
                else:
                    sys.stdout.write("\n")
        else:
            self._erase_line()
            self.update(self._max_index, info="completed in " +
                        self._time_str_convert(time.time() - self._start_time),
                        remains=False)
            if not self._silent:
                sys.stdout.flush()
                




##################################################
#### CLASS Indexer ###############################
##################################################

class Indexer(Tools):
    """Manage locations of created files.

    All files locations are stored in a text-like file: the index
    file. This file is the 'real' counterpart of the index (which is
    'virtual' until :py:meth:`core.Indexer.update_index` is
    called). This method is called each time
    :py:meth:`core.Indexer.__setitem__` is called.

    This class can be accessed like a dictionary.
    """

    file_groups = ['cam1', 'cam2', 'merged']
    file_group_indexes = [0, 1, 2]
    index = dict()
    file_group = None

    def __getitem__(self, file_key):
        """Implement the evaluation of self[file_key]

        :param file_key: Key name of the file to be located
        """
        if file_key in self.index:
            return self.index[file_key]
        else:
            self._print_warning("File key '%s' does not exist"%file_key)
            return None
            
    def __setitem__(self, file_key, file_path):
        """Implement the evaluation of self[file_key] = file_path

        :param file_key: Key name of the file

        :param file_path: Path to the file
        """
        
        if self.file_group is not None:
            file_key = self.file_group + '.' + file_key
        self.index[file_key] = file_path
        self.update_index()

    def __str__(self):
        """Implement the evaluation of str(self)"""
        return str(self.index)

    def _get_index_path(self):
        """Return path of the index"""
        return self._data_path_hdr + 'file_index'

    def _index2group(self, index):
        """Convert an integer (0, 1 or 2) to a group of files
        ('merged', 'cam1' or 'cam2').
        """
        if index == 0:
            return 'merged'
        elif index == 1:
            return 'cam1'
        elif index == 2:
            return 'cam2'
        else:
            self._print_error(
                'Group index must be in %s'%(str(self.file_group_indexes)))

    def get_path(self, file_key, file_group=None, err=False):
        """Return the path of a file recorded in the index.

        Equivalent to self[file_key] if the option file_group is not used.

        :param file_key: Key name of the file to be located

        :param file_group: (Optional) Add group prefix to the key
          name. File group must be 'cam1', 'cam2', 'merged' or their
          integer equivalent 1, 2, 0. File group can also be set to
          None (default None).

        :param err: (Optional) Print an error instead of a warning if
          the file is not indexed.
        """
        if (file_group in self.file_groups):
            file_key = file_group + '.' + file_key
        elif (file_group in self.file_group_indexes):
            file_key = self._index2group(file_group) + '.' + file_key
        elif file_group is not None:
            self._print_error('Bad file group. File group can be in %s, in %s or None'%(str(self.file_groups), str(self.file_group_indexes)))

        if file_key in self.index:
            return self[file_key]
        else:
            if err:
                self._print_error("File key '%s' does not exist"%file_key)
            else:
                self._print_warning("File key '%s' does not exist"%file_key)

    def set_file_group(self, file_group):
        """Set the group of the next files to be recorded. All given
        file keys will be prefixed by the file group.

        :param file_group: File group must be 'cam1', 'cam2', 'merged'
          or their integer equivalent 1, 2, 0. File group can also be
          set to None.
        """
        if (file_group in self.file_group_indexes):
            file_group = self._index2group(file_group)
            
        if (file_group in self.file_groups) or (file_group is None):
            self.file_group = file_group
            
        else: self._print_error(
            'Bad file group name. Must be in %s'%str(self.file_groups))

    def load_index(self):
        """Load index file and rebuild index of already located files"""
        self.index = dict()
        if os.path.exists(self._get_index_path()):
            f = self.open_file(self._get_index_path(), 'r')
            for iline in f:
                if len(iline) > 2:
                    iline = iline.split()
                    self.index[iline[0]] = iline[1]
            f.close()

    def update_index(self):
        """Update index files with data in the virtual index"""
        f = self.open_file(self._get_index_path(), 'w')
        for ikey in self.index:
            f.write('%s %s\n'%(ikey, self.index[ikey]))
        f.close()
        
        

#################################################
#### CLASS Lines ################################
#################################################
class Lines(Tools):
    """
    This class manages emission lines names and wavelengths.
    
    Spectral lines rest wavelength::
    
      ============ ======== =======
        Em. Line    Vaccum    Air
      ============ ======== =======
      [OII]3726    372.709  372.603
      [OII]3729    372.988  372.882
      Hepsilon     397.119  397.007
      Hdelta       410.292  410.176
      Hgamma       434.169  434.047
      [OIII]4363   436.444  436.321
      Hbeta        486.269  486.133
      [OIII]4959   496.030  495.892
      [OIII]5007   500.824  500.684
      [NII]6548    654.984  654.803
      Halpha       656.461  656.280
      [NII]6583    658.523  658.341
      [SII]6716    671.832  671.647
      [SII]6731    673.271  673.085
    """
    sky_lines_file_name = 'sky_lines.orb'
    """Name of the sky lines data file."""

    air_sky_lines_nm = None
    """Air sky lines wavelength"""


    vac_lines_nm = {'[OII]3726':372.709,
                    '[OII]3729':372.988,
                    'Hepsilon':397.119,
                    'Hdelta':410.292,
                    'Hgamma':434.169,
                    '[OIII]4363':436.444,
                    'Hbeta':486.269,
                    '[OIII]4959':496.030,
                    '[OIII]5007':500.824,
                    '[NII]6548':654.984,
                    'Halpha':656.461,
                    '[NII]6583':658.523,
                    '[SII]6716':671.832,
                    '[SII]6731':673.271}
    """Vacuum emission lines wavelength"""
    
    air_lines_nm = {'[OII]3726':372.603,
                    '[OII]3729':372.882,
                    '[NeIII]3869':386.875,
                    'Hepsilon':397.007,
                    'Hdelta':410.176,
                    'Hgamma':434.047,
                    '[OIII]4363':436.321,
                    'Hbeta':486.133,
                    '[OIII]4959':495.891,
                    '[OIII]5007':500.684,
                    'HeI5876':587.567,
                    '[OI]6300':630.030,
                    '[SIII]6312':631.21,
                    '[NII]6548':654.803,
                    'Halpha':656.280,
                    '[NII]6583':658.341,
                    'HeI6678':667.815,
                    '[SII]6716':671.647,
                    '[SII]6731':673.085,
                    'HeI7065':706.528,
                    '[ArIII]7136':713.578,
                    '[OII]7120':731.965,
                    '[OII]7130':733.016,
                    '[ArIII]7751':775.112}
    """Air emission lines wavelength"""

    vac_lines_name = None
    air_lines_name = None
    
    def __init__(self, **kwargs):
        """Lines class constructor.

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)

        # create corresponding inverted dicts
        self.air_lines_name = dict()
        for ikey in self.air_lines_nm.iterkeys():
            self.air_lines_name[str(self.air_lines_nm[ikey])] = ikey

        self.vac_lines_name = dict()
        for ikey in self.vac_lines_nm.iterkeys():
            self.vac_lines_name[str(self.vac_lines_nm[ikey])] = ikey
            
        self._read_sky_file()
        

    def _read_sky_file(self):
        """Return sky file (sky_lines.orb) as a dict.
        """
        sky_lines_file_path = self._get_orb_data_file_path(
            self.sky_lines_file_name)
        f = self.open_file(sky_lines_file_path, 'r')
        self.air_sky_lines_nm = dict()
        try:
            for line in f:
                if '#' not in line and len(line) > 2:
                    line = line.split()
                    self.air_sky_lines_nm[line[1]] = (float(line[0]) / 10., float(line[2]))
        except:
            self._print_error('Error during parsing of '%sky_lines_file_path)
        finally:
            f.close()

    def get_sky_lines(self, nm_min, nm_max, delta_nm, line_nb=0,
                      get_names=False):
        """Return sky lines in a range of optical wavelength.

        :param nm_min: min Wavelength of the lines in nm
        
        :param nm_max: max Wavelength of the lines in nm

        :param delta_nm: Wavelength resolution in nm as the minimum
          wavelength interval of the spectrum. Lines comprises in half
          of this interval are merged.
        
        :param line_nb: (Optional) Number of the most intense lines to
          retrieve. If 0 all lines are given (default 0).

        :param get_name: (Optional) If True return lines name also.
        """
        def merge(merged_lines):
            
            merged_lines_nm = np.array([line[1] for line in merged_lines])
           
            
            merged_lines_nm = (np.sum(merged_lines_nm[:,0]
                                      * merged_lines_nm[:,1])
                               /np.sum(merged_lines_nm[:,1]),
                               np.sum(merged_lines_nm[:,1]))

            merged_lines_name = [line[0] for line in merged_lines]
            temp_list = list()
            
            for name in merged_lines_name:
                if 'MEAN' in name:
                    name = name[5:-1]
                temp_list.append(name)
            merged_lines_name = 'MEAN[' + ','.join(temp_list) + ']'
            return (merged_lines_name, merged_lines_nm)

        lines = [(line_name, self.air_sky_lines_nm[line_name])
                 for line_name in self.air_sky_lines_nm
                 if (self.air_sky_lines_nm[line_name][0] >= nm_min
                     and self.air_sky_lines_nm[line_name][0] <= nm_max)]
        
        lines.sort(key=lambda l: l[1][0])
    
        merged_lines = list()
        final_lines = list()
        
        for iline in range(len(lines)):
            if iline + 1 < len(lines):
                if (abs(lines[iline][1][0] - lines[iline+1][1][0])
                    < delta_nm / 2.):
                    merged_lines.append(lines[iline])
                else:
                    if len(merged_lines) > 0:
                        merged_lines.append(lines[iline])
                        final_lines.append(merge(merged_lines))
                        merged_lines = list()
                    else:
                        final_lines.append(lines[iline])
                        
        # correct a border effect if the last lines of the list are to
        # be merged
        if len(merged_lines) > 0:
            merged_lines.append(lines[iline])
            final_lines.append(merge(merged_lines))

        lines = final_lines
        
        # get only the most intense lines
        if line_nb > 0:
            lines.sort(key=lambda l: l[1][1], reverse=True)
            lines = lines[:line_nb]

        lines_nm = list(np.array([line[1] for line in lines])[:,0])
        lines_name = [line[0] for line in lines]
        
        # add balmer lines
        balmer_lines = ['Halpha', 'Hbeta', 'Hgamma', 'Hdelta', 'Hepsilon']
        for iline in balmer_lines:
            if (self.air_lines_nm[iline] >= nm_min
                and self.air_lines_nm[iline] <= nm_max):
                lines_nm.append(self.air_lines_nm[iline])
                lines_name.append(iline)

        if not get_names:
            lines_nm.sort()
            return lines_nm
        else:
            lines = [(lines_name[iline], lines_nm[iline])
                     for iline in range(len(lines_nm))]
            lines.sort(key=lambda l: l[1])
            lines_nm = [line[1] for line in lines]
            lines_name = [line[0] for line in lines]
            return lines_nm, lines_name
        

    def get_line_nm(self, lines_name, air=True, round_ang=False):
        """Return the wavelength of a line or a list of lines

        :param lines_name: List of line names

        :param air: (Optional) If True, air rest wavelength are
          returned. If False, vacuum rest wavelength are
          returned (default True).

        :param round_ang: (Optional) If True return the rounded
          wavelength of the line in angstrom (default False)
        """
        if isinstance(lines_name, str):
            lines_name = [lines_name]
        if air:
            lines_nm = [self.air_lines_nm[line_name]
                        for line_name in lines_name]
        else:
            lines_nm = [self.vac_lines_nm[line_name]
                        for line_name in lines_name]

        if len(lines_nm) == 1:
            lines_nm = lines_nm[0]
            
        if round_ang:
            return self.round_nm2ang(lines_nm)
        else:
            return lines_nm

    def get_line_name(self, lines, air=True):
        """Return the name of a line or a list of lines given their
        wavelength.

        :param lines: List of lines wavelength

        :param air: (Optional) If True, rest wavelength is considered
          to be in air. If False it is considered to be in
          vacuum (default True).
        """
        if isinstance(lines, (float, int, np.float128)):
            lines = [lines]

        names = list()
        if air:
            for iline in lines:
                if str(iline) in self.air_lines_name:
                    names.append(self.air_lines_name[str(iline)])
                else:
                    names.append('None')

        else:
            for iline in lines:
                if str(iline) in self.vac_lines_name:
                    names.append(self.vac_lines_name[str(iline)])
                else:
                    names.append('None')

        if len(names) == 1: return names[0]
        else: return names
    
    def round_nm2ang(self, nm):
        """Convert a wavelength in nm into a rounded value in angstrom

        :param nm: Line wavelength in nm
        """
        return np.squeeze(np.rint(np.array(nm) * 10.).astype(int))
    
#################################################
#### CLASS OptionFile ###########################
#################################################
        
class OptionFile(Tools):
    """Manage an option file.

    An option file is a file containing keywords and values and
    optionally some comments indicated by '#', e.g.::
    
      ## ORBS configuration file 
      # Author: Thomas Martin <thomas.martin.1@ulaval.ca>
      # File : config.orb
      ## Observatory
      OBSERVATORY_NAME OMM # Observatory name
      TELESCOPE_NAME OMM # Telescope name
      INSTRUMENT_NAME SpIOMM # Instrument name
      OBS_LAT 45.455000 # Observatory latitude
      OBS_LON -71.153000 # Observatory longitude

    .. note:: The special keyword **INCLUDE** can be used to give the
      path to another option file that must be included. Note that
      existing keywords will be overriden so the INCLUDE keyword is
      best placed at the very beginning of the option file.

    .. note:: Repeated keywords override themselves. Only the
      protected keywords ('REG', 'TUNE') are all kept.

    .. note:: The first line starting with '##' is considered as a header
      of the file and can be used to give a description of the file.
    """

    input_file_path = None
    option_file = None # the option file object
    options = None # the core list containing the parsed file
    lines = None # a core.Lines instance

    # special keywords that can be used mulitple times without being
    # overriden.
    protected_keys = ['REG', 'TUNE'] 

    header_line = None
    
    def __init__(self, option_file_path, protected_keys=[], **kwargs):
        """Initialize class

        :param option_file_path: Path to the option file

        :param protected_keys: (Optional) Add other protected keys to
          the basic ones (default []).

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        
        # append new protected keys
        for key in protected_keys:
            self.protected_keys.append(key) 
        
        self.option_file = self.open_file(option_file_path, 'r')
        self.input_file_path = str(option_file_path)
        self.options = dict()
        self.lines = Lines(config_file_name=self.config_file_name)
        for line in self.option_file:
            if len(line) > 2:
                if line[0:2] == '##':
                    if self.header_line is None:
                        self.header_line = line[2:-1]
                    
                if line[0] != '#': # check if line is commented
                    if '#' in line:
                        line = line[:line.find('#')]
                    line = line.split()
                    if not np.size(line) == 0:
                        key = line[0]

                        # manage INCLUDE keyword
                        if key == 'INCLUDE':
                            included_optfile = self.__class__(line[1])
                            self.options = dict(
                                self.options.items()
                                + included_optfile.options.items())

                        # manage protected keys
                        final_key = str(key)
                        if final_key in self.protected_keys:
                            index = 2
                            while final_key in self.options:
                                final_key = key + str(index)
                                index += 1

                        if len(line) > 2:
                            self.options[final_key] = line[1:]
                        elif len(line) > 1:
                            self.options[final_key] = line[1]
                        
        self.option_file.close()
        
                        
    def __getitem__(self, key):
        """Implement the evaluation of self[key]."""
        if key in self.options:
            return self.options[key]
        else: return None
        
    def iteritems(self):
        """Implement iteritems function often used with iterable objects"""
        return self.options.iteritems()

    def get(self, key, cast=str):
        """Return the value associated to a keyword.
        
        :param key: keyword
        
        :param cast: (Optional) Cast function for the returned value
          (default str).
        """
        param = self[key]
        if param is not None:
            if cast is not bool:
                return cast(param)
            else:
                return bool(int(param))
        else:
            return None

    def get_regions_parameters(self):
        """Get regions parameters.

        Defined for the special keyword 'REG'.
        """
        return {k:v for k,v in self.iteritems() if k.startswith('REG')}

    def get_lines(self, nm_min=None, nm_max=None, delta_nm=None):
        """Get lines parameters.

        Defined for the special keyword 'LINES'.

        All optional keywords are only used if the keyword SKY is
        used.

        :param nm_min: (Optional) min wavelength of the lines in nm
          (default None).
        
        :param nm_max: (Optional) max wavelength of the lines in nm
          (default None).

        :param delta_nm: (Optional) wavelength resolution in nm as the minimum
          wavelength interval of the spectrum. Lines comprises in half
          of this interval are merged (default None).
    
        """
        lines_names = self['LINES']
        lines_nm = list()
        if len(lines_names) > 2:
            lines_names = lines_names.split(',')
            for iline in lines_names:
                try:
                    lines_nm.append(float(iline))
                except:
                    if iline == 'SKY':
                        if (nm_min is not None and nm_max is not None
                            and delta_nm is not None):
                            lines_nm += self.lines.get_sky_lines(
                                nm_min, nm_max, delta_nm)
                        else: self._print_error('Keyword SKY used but nm_min, nm_max or delta_nm parameter not set')
                            
                    else:
                        lines_nm.append(self.lines.get_line_nm(iline))
        else:
            return None
        
        return lines_nm
            
    def get_filter_edges(self):
        """Get filter eges parameters.

        Defined for the special keyword 'FILTER_EDGES'.
        """
        filter_edges = self['FILTER_EDGES']
        if filter_edges is not None:
            filter_edges = filter_edges.split(',')
            if len(filter_edges) == 2:
                return np.array(filter_edges).astype(float)
            else:
                self._print_error(
                    'Bad filter edges definition: check option file')
        else:
            return None

    def get_fringes(self):
        """Get fringes

        Defined for the special keyword 'FRINGES'.
        """
        fringes = self['FRINGES']
        if fringes is not None:
            fringes = fringes.split(':')
            try:
                return np.array(
                    [ifringe.split(',') for ifringe in fringes],
                    dtype=float)
            except ValueError:
                self._print_error("Fringes badly defined. Use no whitespace, each fringe must be separated by a ':'. Fringes parameters must be given in the order [frequency, amplitude] separated by a ',' (e.g. 150.0,0.04:167.45,0.095 gives 2 fringes of parameters [150.0, 0.04] and [167.45, 0.095]).")
        return None

    def get_bad_frames(self):
        """Get bad frames.

        Defined for the special keyword 'BAD_FRAMES'.
        """
        if self['BAD_FRAMES'] is not None:
            bad_frames = self['BAD_FRAMES'].split(",")
        else: return None
        
        bad_frames_list = list()
        try:
            for ibad in bad_frames:
                if (ibad.find(":") > 0):
                    min_bad = int(ibad.split(":")[0])
                    max_bad = int(ibad.split(":")[1])+1
                    for i in range(min_bad, max_bad):
                        bad_frames_list.append(i)
                else:
                    bad_frames_list.append(int(ibad))   
            return np.array(bad_frames_list)
        except ValueError:
            self._print_error("Bad frames badly defined. Use no whitespace, bad frames must be comma separated. the sign ':' can be used to define a range of bad frames, commas and ':' can be mixed. Frame index must be integer.")

    def get_tuning_parameters(self):
        """Return the list of tuning parameters.

        Defined for the special keyword 'TUNE'.
        """
        return {v[0]:v[1] for k,v in self.iteritems() if k.startswith('TUNE')}
        
                   
#################################################
#### CLASS ParamsFile ###########################
#################################################

class ParamsFile(Tools):
    """Manage correspondance between multiple dict containing the
    same parameters and a file on disk.

    Its behaviour is similar to :py:class:`astrometry.StarsParams`.
    """

    _params_list = None
    _keys = None

    f = None
    
    def __init__(self, file_path, reset=True, **kwargs):
        """Init ParamsFile class.

        :param file_path: Path of the output file where all
          parameters are stored (Note that this file will
          automatically be overwritten if reset is set to True).

        :param reset: (Optional) If True the output file is
          overwritten. If False and if the output file already exists
          data in the file are read and new data is appended (default
          True).

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        
        self._params_list = list()
        if not reset and os.path.exists(file_path):
            self.f = self.open_file(file_path, 'r')
            for iline in self.f:
                if '##' not in iline and len(iline) > 3:
                    if '# KEYS' in iline:
                        self._keys = iline.split()[2:]
                    elif self._keys is not None:
                        iline = iline.split()
                        line_dict = dict()
                        for ikey in range(len(self._keys)):
                            line_dict[self._keys[ikey]] = iline[ikey]
                        self._params_list.append(line_dict)
                    else:
                        self._print_error(
                            'Wrong file format: {:s}'.format(file_path))
            self.f.close()
            self.f = self.open_file(file_path, 'a')

        else:
            self.f = self.open_file(file_path, 'w')
            self.f.write("## PARAMS FILE\n## created by {:s}\n".format(
                self.__class__.__name__))
            self.f.flush()

    def __del__(self):
        """ParamsFile destructor"""
        
        if self.f is not None:
            self.f.close()

    def __getitem__(self, key):
        """implement Instance[key]"""
        return self._params_list[key]
            
    def append(self, params):
        """Append a dict to the file.

        :param params: A dict of parameters
        """
        if len(self._params_list) == 0:
            self._params_list.append(params)
            self._keys = params.keys()
            self._keys.sort()
            self.f.write('# KEYS')
            for ikey in self._keys:
                self.f.write(' {:s}'.format(ikey))
            self.f.write('\n')
        else:
            keys = params.keys()
            keys.sort()
            if keys == self._keys:
                self._params_list.append(params)
            else:
                self._print_error('parameters of the new entry are not the same as the old entries')
        
        for ikey in self._keys:
            self.f.write(' {}'.format(self._params_list[-1][ikey]))
        self.f.write('\n')
        self.f.flush()

    def get_data(self):
        return self._params_list


#################################################
#### CLASS HDFCube ##############################
#################################################


class HDFCube(Cube):
    """ This class implements the use of an HDF5 cube.

    An HDF5 cube is similar to the *frame-divided cube* implemented by
    the class :py:class:`orb.core.Cube` but it makes use of the really
    fast data access provided by HDF5 files. The "frame-divided"
    concept is keeped but all the frames are grouped into one hdf5
    file.

    An HDF5 cube must have a certain architecture:
    
    * Each frame has its own group called 'frameIIIII', IIIII being a
      integer giving the position of the frame on 5 characters filled
      with zeros. e.g. the first frame group is called frame00000

    * Each frame group is divided into at least 2 datasets: **data**
      and *header* (e.g. the data of the first frame will be in the
      dataset *frame00000/data*)

    * A **mask** dataset can be added to each frame.
    """

    _hdf5f = None # Instance of h5py.File
    _silent_load = False
    is_complex = None
    _prebinning = None
    quad_nb = None # number of quads (set to None if HDFCube is not a
                   # cube split in quads but a cube split in frames)
    is_quad_cube = None # set to True if cube is split in quad. set to
                        # False if split in frames.
    
    """Set to True if data is complex"""
    
    def __init__(self, cube_path, project_header=list(),
                 wcs_header=list(), calibration_laser_header=list(),
                 overwrite=False, indexer=None, silent_init=False,
                 binning=None, **kwargs):
        
        """
        Initialize HDFCube class.
        
        :param cube_path: Path to the HDF5 cube.

        :param project_header: (Optional) header section describing
          the observation parameters that can be added to each output
          files (an empty list() by default).

        :param wcs_header: (Optional) header section describing WCS
          that can be added to each created image files (an empty
          list() by default).

        :param calibration_laser_header: (Optional) header section
          describing the calibration laser parameters that can be
          added to the concerned output files e.g. calibration laser map,
          spectral cube (an empty list() by default).

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param binning: (Optional) Cube binning. If > 1 data will be
          transparently binned so that the cube will behave as as if
          it was already binned (default None).

        :param silent_init: (Optional) If True Init is silent (default False).

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
 
        Tools.__init__(self, **kwargs)
        self.overwrite = overwrite
        self.indexer = indexer
        self._project_header = project_header
        self._wcs_header = wcs_header
        self._calibration_laser_header = calibration_laser_header

        self._prebinning = None
        if binning is not None:
            if int(binning) > 1:
                self._prebinning = int(binning)
            
        self.DIV_NB = int(self._get_config_parameter(
            "DIV_NB"))
        self.QUAD_NB = self.DIV_NB**2L
        self.BIG_DATA = bool(int(self._get_config_parameter(
            "BIG_DATA")))
        
        self._parallel_access_to_data = False

        if cube_path is None or cube_path == '': return
        
        with self.open_hdf5(cube_path, 'r') as f:
            self.cube_path = cube_path
            self.dimz = self._get_attribute('dimz')
            self.dimx = self._get_attribute('dimx')
            self.dimy = self._get_attribute('dimy')
            if 'image_list' in f:
                self.image_list = f['image_list'][:]
            
            

            # check if cube is quad or frames based
            self.quad_nb = self._get_attribute('quad_nb', optional=True)
            if self.quad_nb is not None:
                self.is_quad_cube = True
            else:
                self.is_quad_cube = False
        
            # sanity check
            if self.is_quad_cube:
                quad_nb = len(
                    [igrp for igrp in f
                     if 'quad' == igrp[:4]])
                if quad_nb != self.quad_nb:
                    self._print_error("Corrupted HDF5 cube: 'quad_nb' attribute ([]) does not correspond to the real number of quads ({})".format(self.quad_nb, quad_nb))

                if self._get_hdf5_quad_path(0) in f:
                    # test whether data is complex
                    if np.iscomplexobj(f[self._get_hdf5_quad_data_path(0)]):
                        self.is_complex = True
                        self.dtype = complex
                    else:
                        self.is_complex = False
                        self.dtype = float
                
                else:
                    self._print_error('{} is missing. A valid HDF5 cube must contain at least one quadrant'.format(
                        self._get_hdf5_quad_path(0)))
                    

            else:
                frame_nb = len(
                    [igrp for igrp in f
                     if 'frame' == igrp[:5]])

                if frame_nb != self.dimz:
                    self._print_error("Corrupted HDF5 cube: 'dimz' attribute ({}) does not correspond to the real number of frames ({})".format(self.dimz, frame_nb))
                
            
                if self._get_hdf5_frame_path(0) in f:                
                    if ((self.dimx, self.dimy)
                        != f[self._get_hdf5_data_path(0)].shape):
                        self._print_error('Corrupted HDF5 cube: frame shape {} does not correspond to the attributes of the file {}x{}'.format(f[self._get_hdf5_data_path(0)].shape, self.dimx, self.dimy))

                    if self._get_hdf5_data_path(0, mask=True) in f:
                        self._mask_exists = True
                    else:
                        self._mask_exists = False

                    # test whether data is complex
                    if np.iscomplexobj(f[self._get_hdf5_data_path(0)]):
                        self.is_complex = True
                        self.dtype = complex
                    else:
                        self.is_complex = False
                        self.dtype = float
                else:
                    self._print_error('{} is missing. A valid HDF5 cube must contain at least one frame'.format(
                        self._get_hdf5_frame_path(0)))
                

        # binning
        if self._prebinning is not None:
            self.dimx = self.dimx / self._prebinning
            self.dimy = self.dimy / self._prebinning

        if (self.dimx) and (self.dimy) and (self.dimz):
            if not silent_init:
                self._print_msg("Data shape : (" + str(self.dimx) 
                                + ", " + str(self.dimy) + ", " 
                                + str(self.dimz) + ")")
        else:
            self._print_error("Incorrect data shape : (" 
                            + str(self.dimx) + ", " + str(self.dimy) 
                              + ", " +str(self.dimz) + ")")

        self.shape = (self.dimx, self.dimy, self.dimz)

        
    def __getitem__(self, key):
        """Implement the evaluation of self[key].
        
        .. note:: To make this function silent just set
          Cube()._silent_load to True.
        """
        def slice_in_quad(ax_slice, ax_min, ax_max):
            ax_range = range(ax_min, ax_max)
            for ii in range(ax_slice.start, ax_slice.stop):
                if ii in ax_range:
                    return True
            return False
        # check return mask possibility
        if self._return_mask and not self._mask_exists:
            self._print_error("No mask found with data, cannot return mask")
        
        # produce default values for slices
        x_slice = self._get_default_slice(key[0], self.dimx)
        y_slice = self._get_default_slice(key[1], self.dimy)
        z_slice = self._get_default_slice(key[2], self.dimz)

        data = np.empty((x_slice.stop - x_slice.start,
                         y_slice.stop - y_slice.start,
                         z_slice.stop - z_slice.start), dtype=self.dtype)

        if self._prebinning is not None:
            x_slice = slice(x_slice.start * self._prebinning,
                            x_slice.stop * self._prebinning, 1)
            y_slice = slice(y_slice.start * self._prebinning,
                            y_slice.stop * self._prebinning, 1)

        # frame based cube
        if not self.is_quad_cube:

            if z_slice.stop - z_slice.start == 1:
                only_one_frame = True
            else:
                only_one_frame = False

            with self.open_hdf5(self.cube_path, 'r') as f:
                if not self._silent_load and not only_one_frame:
                    progress = ProgressBar(z_slice.stop - z_slice.start - 1L)

                for ik in range(z_slice.start, z_slice.stop):
                    unbin_data = f[
                        self._get_hdf5_data_path(
                            ik, mask=self._return_mask)][x_slice, y_slice]

                    if self._prebinning is not None:
                        data[0:x_slice.stop - x_slice.start,
                             0:y_slice.stop - y_slice.start,
                             ik - z_slice.start] = utils.image.nanbin_image(
                            unbin_data, self._prebinning)
                    else:
                        data[0:x_slice.stop - x_slice.start,
                             0:y_slice.stop - y_slice.start,
                             ik - z_slice.start] = unbin_data

                    if not self._silent_load and not only_one_frame:
                        progress.update(ik - z_slice.start, info="Loading data")

                if not self._silent_load and not only_one_frame:
                    progress.end()

        # quad based cube
        else:
            with self.open_hdf5(self.cube_path, 'r') as f:
                if not self._silent_load:
                    progress = ProgressBar(self.quad_nb)

                for iquad in range(self.quad_nb):
                    if not self._silent_load:
                        progress.update(iquad, info='Loading data')
                    x_min, x_max, y_min, y_max = self._get_quadrant_dims(
                        iquad, self.dimx, self.dimy, int(math.sqrt(float(self.quad_nb))))
                    if slice_in_quad(x_slice, x_min, x_max) and slice_in_quad(y_slice, y_min, y_max):
                        data[max(x_min, x_slice.start) - x_slice.start:
                             min(x_max, x_slice.stop) - x_slice.start,
                             max(y_min, y_slice.start) - y_slice.start:
                             min(y_max, y_slice.stop) - y_slice.start,
                             0:z_slice.stop-z_slice.start] = f[self._get_hdf5_quad_data_path(iquad)][
                            max(x_min, x_slice.start) - x_min:min(x_max, x_slice.stop) - x_min,
                            max(y_min, y_slice.start) - y_min:min(y_max, y_slice.stop) - y_min,
                            z_slice.start:z_slice.stop]
                if not self._silent_load:
                    progress.end()
                        

        return np.squeeze(data)

                

    def _get_attribute(self, attr, optional=False):
        """Return the value of an attribute of the HDF5 cube

        :param attr: Attribute to return
        
        :param optional: If True and if the attribute does not exist
          only a warning is raised. If False the HDF5 cube is
          considered as invalid and an exception is raised.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            if attr in f.attrs:
                return f.attrs[attr]
            else:
                if not optional:
                    self._print_error('Attribute {} is missing. The HDF5 cube seems badly formatted. Try to create it again with the last version of ORB.'.format(attr))
                else:
                    return None

    def get_frame_attributes(self, index):
        """Return an attribute attached to a frame.

        If the attribute does not exist returns None.

        :param index: Index of the frame.

        :param attr: Attribute name.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            attrs = list()
            for attr in f[self._get_hdf5_frame_path(index)].attrs:
                attrs.append(
                    (attr, f[self._get_hdf5_frame_path(index)].attrs[attr]))
            return attrs

    def get_frame_attribute(self, index, attr):
        """Return an attribute attached to a frame.

        If the attribute does not exist returns None.

        :param index: Index of the frame.

        :param attr: Attribute name.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            if attr in f[self._get_hdf5_frame_path(index)].attrs:
                return f[self._get_hdf5_frame_path(index)].attrs[attr]
            else: return None

    def get_frame_header(self, index):
        """Return the header of a frame given its index in the list.

        The header is returned as an instance of pyfits.Header().

        :param index: Index of the frame
        """
        
        with self.open_hdf5(self.cube_path, 'r') as f:
            return self._header_hdf52fits(
                f[self._get_hdf5_header_path(index)][:])

    def get_cube_header(self):
        """Return the header of a the cube.

        The header is returned as an instance of pyfits.Header().
        """
        
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'header' in f:
                return self._header_hdf52fits(f['header'][:])
            else:
                return pyfits.Header()
           
    def get_mean_image(self, recompute=False):
        """Return the deep frame of the cube.

        :param recompute: (Optional) Force to recompute mean image
          even if it is already present in the cube (default False).
        
        If a deep frame has already been computed ('deep_frame'
        dataset) return it directly.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'deep_frame' in f and not recompute:
                return f['deep_frame'][:]
            else:
                return Cube.get_mean_image(self, recompute=recompute)

    def get_interf_energy_map(self, recompute=False):
        """Return the energy map of an interferogram cube.
    
        :param recompute: (Optional) Force to recompute energy map
          even if it is already present in the cube (default False).
          
        If an energy map has already been computed ('energy_map'
        dataset) return it directly.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'energy_map' in f and not recompute:
                return f['energy_map'][:]
            else:
                return Cube.get_interf_energy_map(self)
    
    def get_spectrum_energy_map(self, recompute=False):
        """Return the energy map of a spectral cube.
        
        :param recompute: (Optional) Force to recompute energy map
          even if it is already present in the cube (default False).
          
        If an energy map has already been computed ('energy_map'
        dataset) return it directly.
        """
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'energy_map' in f and not recompute:
                return f['energy_map'][:]
            else:
                return Cube.get_spectrum_energy_map(self)

        
##################################################
#### CLASS OutHDFCube ############################
##################################################           

class OutHDFCube(Tools):
    """Output HDF5 Cube class.

    This class must be used to output a valid HDF5 cube.
    
    .. warning:: The underlying dataset is not readonly and might be
      overwritten.

    .. note:: This class has been created because
      :py:class:`orb.core.HDFCube` must not be able to change its
      underlying dataset (the HDF5 cube is always read-only).
    """
    
    def __init__(self, export_path, shape, overwrite=False,
                 reset=False, **kwargs):
        """Init OutHDFCube class.

        :param export_path: Path ot the output HDF5 cube to create.

        :param shape: Data shape. Must be a 3-Tuple (dimx, dimy, dimz)

        :param overwrite: (Optional) If True data will be overwritten
          but existing data will not be removed (default False).

        :param reset: (Optional) If True and if the file already
          exists, it is deleted (default False).
        
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        
        # change path if file exists and must not be overwritten
        self.export_path = str(export_path)
        if reset and os.path.exists(self.export_path):
            os.remove(self.export_path)
        
        if not overwrite and os.path.exists(self.export_path):
            index = 0
            while os.path.exists(self.export_path):
                self.export_path = (os.path.splitext(export_path)[0] + 
                                   "_" + str(index) + 
                                   os.path.splitext(export_path)[1])
                index += 1
        if self.export_path != export_path:
            self._print_warning('Cube path changed to {} to avoid overwritting an already existing file'.format(self.export_path))

        if len(shape) == 3: self.shape = shape
        else: self._print_error('An HDF5 cube shape must be a tuple (dimx, dimy, dimz)')

        try:
            self.f = self.open_hdf5(self.export_path, 'a')
        except IOError, e:
            if overwrite:
                os.remove(self.export_path)
                self.f = self.open_hdf5(self.export_path, 'a')
            else:
                self._print_error(
                    'IOError while opening HDF5 cube: {}'.format(e))
                
        self._print_msg('Opening OutHDFCube {} ({},{},{})'.format(
            self.export_path, *self.shape))
        
        # add attributes
        self.f.attrs['dimx'] = self.shape[0]
        self.f.attrs['dimy'] = self.shape[1]
        self.f.attrs['dimz'] = self.shape[2]

        self.imshape = (self.shape[0], self.shape[1])

    def write_frame_attribute(self, index, attr, value):
        """Write a frame attribute

        :param index: Index of the frame

        :param attr: Attribute name

        :param value: Value of the attribute to write
        """
        self.f[self._get_hdf5_frame_path(index)].attrs[attr] = value

    def write_frame(self, index, data=None, header=None, mask=None,
                    record_stats=False, force_float32=True, section=None,
                    force_complex64=False, compress=False):
        """Write a frame

        :param index: Index of the frame
        
        :param data: (Optional) Frame data (default None).
        
        :param header: (Optional) Frame header (default None).
        
        :param mask: (Optional) Frame mask (default None).
        
        :param record_stats: (Optional) If True Mean and Median of the
          frame are appended as attributes (data must be set) (defaut
          False).

        :param force_float32: (Optional) If True, data type is forced
          to numpy.float32 type (default True).

        :param section: (Optional) If not None, must be a 4-tuple
          [xmin, xmax, ymin, ymax] giving the section to write instead
          of the whole frame. Useful to modify only a part of the
          frame (deafult None).

        :param force_complex64: (Optional) If True, data type is
          forced to numpy.complex64 type (default False).

        :param compress: (Optional) If True, data is lossely
          compressed using a gzip algorithm (default False).
        """
        def _replace(name, dat):
            if name == 'data':
                if force_complex64: dat = dat.astype(np.complex64)
                elif force_float32: dat = dat.astype(np.float32)
                dat_path = self._get_hdf5_data_path(index)
            if name == 'mask':
                dat_path = self._get_hdf5_data_path(index, mask=True)
            if name == 'header':
                dat_path = self._get_hdf5_header_path(index)
            if name == 'data' or name == 'mask':
                if section is not None:
                    old_dat = None
                    if  dat_path in self.f:
                        if (self.f[dat_path].shape == self.imshape):
                            old_dat = self.f[dat_path][:]
                    if old_dat is None:
                        frame = np.empty(self.imshape, dtype=dat.dtype)
                        frame.fill(np.nan)
                    else:
                        frame = np.copy(old_dat).astype(dat.dtype)
                    frame[section[0]:section[1],section[2]:section[3]] = dat
                else:
                    if dat.shape == self.imshape:
                        frame = dat
                    else:
                        self._print_error(
                            "Bad data shape {}. Must be {}".format(
                                dat.shape, self.imshape))
                dat = frame
                    
            if dat_path in self.f: del self.f[dat_path]
            if compress:
                ## szip_types = (np.float32, np.float64, np.int16, np.int32, np.int64,
                ##               np.uint8, np.uint16)
            
                ## if dat.dtype in szip_types:
                ##     compression = 'szip'
                ##     compression_opts = ('nn', 32)
                ## else:
                ##     compression = 'gzip'
                ##     compression_opts = 4
                compression = 'lzf'
                compression_opts = None
            else:
                compression = None
                compression_opts = None
            self.f.create_dataset(
                dat_path, data=dat,
                compression=compression, compression_opts=compression_opts)
            return dat

        if force_float32 and force_complex64:
            self._print_error('force_float32 and force_complex64 cannot be both set to True')

            
        if data is None and header is None and mask is None:
            self._print_warning('Nothing to write in the frame {}').format(
                index)
            return
        
        if data is not None:
            data = _replace('data', data)
            
            if record_stats:
                mean = bn.nanmean(data.real)
                median = bn.nanmedian(data.real)
                self.f[self._get_hdf5_frame_path(index)].attrs['mean'] = (
                    mean)
                self.f[self._get_hdf5_frame_path(index)].attrs['median'] = (
                    mean)
            else:
                mean = None
                median = None
                
        if mask is not None:
            mask = mask.astype(np.bool_)
            _replace('mask', mask)


        # Creating pyfits.ImageHDU instance to format header
        if header is not None:
            if not isinstance(header, pyfits.Header):
                header = pyfits.Header(header)

        if data.dtype != np.bool:
            if np.iscomplexobj(data) or force_complex64:
                hdu = pyfits.ImageHDU(data=data.real, header=header)
            else:
                hdu = pyfits.ImageHDU(data=data, header=header)
        else:
            hdu = pyfits.ImageHDU(data=data.astype(np.uint8), header=header)
            
        
        if hdu is not None:
            if record_stats:
                if mean is not None:
                    if not np.isnan(mean):
                        hdu.header.set('MEAN', mean,
                                       'Mean of data (NaNs filtered)',
                                       after=5)
                if median is not None:
                    if not np.isnan(median):
                        hdu.header.set('MEDIAN', median,
                                       'Median of data (NaNs filtered)',
                                       after=5)
            
            hdu.verify(option=u'silentfix')
                
            
            _replace('header', self._header_fits2hdf5(hdu.header))
            

    def append_image_list(self, image_list):
        """Append an image list to the HDF5 cube.

        :param image_list: Image list to append.
        """
        if 'image_list' in self.f:
            del self.f['image_list']

        if image_list is not None:
            self.f['image_list'] = np.array(image_list)
        else:
            self._print_warning('empty image list')
        

    def append_deep_frame(self, deep_frame):
        """Append a deep frame to the HDF5 cube.

        :param deep_frame: Deep frame to append.
        """
        if 'deep_frame' in self.f:
            del self.f['deep_frame']
            
        self.f['deep_frame'] = deep_frame

    def append_energy_map(self, energy_map):
        """Append an energy map to the HDF5 cube.

        :param energy_map: Energy map to append.
        """
        if 'energy_map' in self.f:
            del self.f['energy_map']
            
        self.f['energy_map'] = energy_map

    def append_header(self, header):
        """Append a header to the HDF5 cube.

        :param header: header to append.
        """
        if 'header' in self.f:
            del self.f['header']
            
        self.f['header'] = self._header_fits2hdf5(header)
        
    def close(self):
        """Close the HDF5 cube. Class cannot work properly once this
        method is called so delete it. e.g.::
        
          outhdfcube.close()
          del outhdfcube
        """
        try:
            self.f.close()
        except Exception:
            pass




##################################################
#### CLASS OutHDFQuadCube ########################
##################################################           

class OutHDFQuadCube(OutHDFCube):
    """Output HDF5 Cube class saved in quadrants.

    This class can be used to output a valid HDF5 cube.
    """

    def __init__(self, export_path, shape, quad_nb, overwrite=False,
                 reset=False, **kwargs):
        """Init OutHDFQuadCube class.

        :param export_path: Path ot the output HDF5 cube to create.

        :param shape: Data shape. Must be a 3-Tuple (dimx, dimy, dimz)

        :param quad_nb: Number of quadrants in the cube.

        :param overwrite: (Optional) If True data will be overwritten
          but existing data will not be removed (default False).

        :param reset: (Optional) If True and if the file already
          exists, it is deleted (default False).
        
        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        OutHDFCube.__init__(self, export_path, shape, overwrite=overwrite,
                            reset=reset, **kwargs)

        self.f.attrs['quad_nb'] = quad_nb

    def write_quad(self, index, data=None, header=None, force_float32=True,
                   force_complex64=False,
                   compress=False):
        """"Write a quadrant

        :param index: Index of the quadrant
        
        :param data: (Optional) Frame data (default None).
        
        :param header: (Optional) Frame header (default None).
        
        :param mask: (Optional) Frame mask (default None).
        
        :param record_stats: (Optional) If True Mean and Median of the
          frame are appended as attributes (data must be set) (defaut
          False).

        :param force_float32: (Optional) If True, data type is forced
          to numpy.float32 type (default True).

        :param section: (Optional) If not None, must be a 4-tuple
          [xmin, xmax, ymin, ymax] giving the section to write instead
          of the whole frame. Useful to modify only a part of the
          frame (deafult None).

        :param force_complex64: (Optional) If True, data type is
          forced to numpy.complex64 type (default False).

        :param compress: (Optional) If True, data is lossely
          compressed using a gzip algorithm (default False).
        """
        
        if force_float32 and force_complex64:
            self._print_error('force_float32 and force_complex64 cannot be both set to True')

            
        if data is None and header is None:
            self._print_warning('Nothing to write in the frame {}').format(
                index)
            return
        
        if data is not None:
            if force_complex64: data = data.astype(np.complex64)
            elif force_float32: data = data.astype(np.float32)
            dat_path = self._get_hdf5_quad_data_path(index)

            if dat_path in self.f:
                del self.f[dat_path]

            if compress:
                
                #szip_types = (np.float32, np.float64, np.int16, np.int32, np.int64,
                #              np.uint8, np.uint16)
                ## if data.dtype in szip_types:
                ##     compression = 'szip'
                ##     compression_opts = ('nn', 32)
                ## else:
                ##     compression = 'gzip'
                ##     compression_opts = 4
                compression = 'lzf'
                compression_opts = None
            else:
                compression = None
                compression_opts = None
                
            self.f.create_dataset(
                dat_path, data=data,
                compression=compression,
                compression_opts=compression_opts)

            return data
        
        


##################################################
#### CLASS Logger ################################
##################################################           

class Logger(object):
    """Logger class."""

    noprint = None
    nolog = None
    stdout = None
    stderr = None
    file = None
    
    def __init__(self, name, mode='a', noprint=False, nolog=False):
        """Init Logger class.

        :param name: Name of the logfile.

        :param mode: (Optional) File mode (same as file mode for the classic
          open() function) (default 'a').

        :param noprint: (Optional) If True nothing is printed on the terminal (or
          more generally on the regular stdout) (default False).

        :param nolog: (Optional) If True nothing is logged in the
          logfile (default False).
        """
        self.file = open(name, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        self.noprint = noprint
        self.nolog = nolog
        
        
    def __del__(self):
        """Class destructor."""
        sys.stdout = self.stdout
        self.file.close()
        
    def write(self, msg, color=None, nolog=False):
        """Write in the logfile and on the terminal (stdout)

        :param msg: Message to write.

        :param color: (Optional) Output will be colored. Must be a
          :py:class:`~orb.core.TextColor` attribute (default None).

        :param nolog: (Optional) If True nothing is printed in the log
          (default False).
        """
        msg_color = str(msg)
        if color is not None:
            msg_color = color + msg + TextColor.END

        if not nolog and not self.nolog:
            self.file.write(msg)
        if not self.noprint:
            self.stdout.write(msg_color)
        self.flush()

    def flush(self):
        """Flush stdout and logfile."""
        self.stdout.flush()
        self.file.flush()
        
##################################################
#### CLASS Waves #################################
##################################################           

class Waves(object):
    """Wave class that keep the best conversions possible from nm to cm1."""

    def __init__(self, nm, velocity=0.):
        """
        :param nm: Rest frame wavelength in nm. Must be a float, a
          string or an array of 1 dimension of floats and
          strings. Strings must be line names stored in Lines class.
        
        :param velocity: (Optional) Velocity in km/s (default 0.)

        .. note:: all parameters can be arrays of the same shape. If
        velocity is a float and nm is an array with a certain
        shape, the same velocity will be attributed to all
        wavelengths.
        """
        if len(np.array(nm).shape) > 1:
            raise Exception('nm must be an array of dimension 1')

        if not isinstance(nm, list):
            if np.size(nm) == 1:
                nm = list([nm])

        nm_list = list()
        for inm in nm:
            if isinstance(inm, str):
                nm_list.append(Lines().get_line_nm(inm))
            else:
                nm_list.append(float(inm))

        self.nm = np.squeeze(np.array(nm_list).astype(np.longdouble))
        self.set_velocity(velocity)
        
    def set_velocity(self, velocity):
        """Set waves velocity.

        :param velocity: velocity in km/s
        """
        if np.size(velocity) == 1:
            self.velocity = float(velocity)
        elif np.array(velocity).shape != self.nm.shape:
            raise Exception('Velocity array shape must be the same as nm shape')
        else:
            self.velocity = np.array(velocity).astype(np.longdouble)

    def get_nm(self):
        """Return wavelength of waves in nm (taking velocity into account)"""
        return self.nm + utils.spectrum.line_shift(
            self.velocity, self.nm, wavenumber=False)

    def get_cm1(self):
        """Return wavenumber of waves in cm-1 (taking velocity into account)"""
        cm1 = self.get_cm1_rest()
        return cm1 + utils.spectrum.line_shift(
            self.velocity, cm1, wavenumber=True)

    def get_nm_rest(self):
        """"Return restframe wavelength of waves in nm"""
        return np.copy(self.nm)

    def get_cm1_rest(self):
        """Return restframe wavelength of waves in cm-1"""
        return utils.spectrum.nm2cm1(self.nm)




#################################################
#### CLASS Standard #############################
#################################################
class Standard(Tools):
    """Manage standard files and photometrical calibration"""

    ang = None # Angstrom axis of the standard file
    flux = None # Flux of the standard in erg/cm2/s/Ang

    def __init__(self, std_name, **kwargs):
        """Initialize Standard class.

        :param std_name: Name of the standard.

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
             
        std_file_path, std_type = self._get_standard_file_path(std_name)

        if std_type == 'MASSEY' or std_type == 'MISC':
            self.ang, self.flux = self._read_massey_dat(std_file_path)
        elif std_type == 'CALSPEC':
            self.ang, self.flux = self._read_calspec_fits(std_file_path)
        else:
            self._print_error(
                "Bad type of standard file. Must be 'MASSEY', 'CALSPEC' or 'MISC'")
       

    def _get_data_prefix(self):
        return (os.curdir + os.sep + 'STANDARD' + os.sep
                + 'STD' + '.')

    def get_spectrum(self, step, order, n, wavenumber=False):
        """Return part of the standard spectrum corresponding to the
        observation parameters.

        Returned spectrum is calibrated in erg/cm^2/s/A

        :param order: Folding order
        :param step: Step size in um
        :param n: Number of steps    
        :param wavenumber: If True spectrum is returned along a
          wavenumber axis.
        """
        if wavenumber:
            axis = utils.spectrum.create_cm1_axis(n, step, order)
            old_axis = utils.spectrum.nm2cm1(self.ang / 10.)
        else:
            axis = utils.spectrum.create_nm_axis(n, step, order)
            old_axis = self.ang / 10.
        
        return axis, utils.vector.interpolate_axis(
            self.flux, axis, 3, old_axis=old_axis)

    def _read_massey_dat(self, file_path):
        """Read a data file from Massey et al., Spectrophotometric
        Standards (1988) and return a tuple of arrays (wavelength,
        flux).
        
        Returned wavelength axis is in A. Returned flux is converted
        in erg/cm^2/s/A.

        :param file_path: Path to the Massey dat file (generally
          'spXX.dat').
        """
        std_file = self.open_file(file_path, 'r')
        
        spec_ang = list()
        spec_mag = list()
        for line in std_file:
             line = line.split()
             spec_ang.append(line[0])
             spec_mag.append(line[1])

        spec_ang = np.array(spec_ang, dtype=float)
        spec_mag = np.array(spec_mag, dtype=float)
        
        # convert mag to flux in erg/cm^2/s/A
        spec_flux = utils.photometry.ABmag2flambda(spec_mag, spec_ang)

        return spec_ang, spec_flux

    def _read_calspec_fits(self, file_path):
        """Read a CALSPEC fits file containing a standard spectrum and
          return a tuple of arrays (wavelength, flux).

        Returned wavelength axis is in A. Returned flux is in
        erg/cm^2/s/A.
        
        :param file_path: Path to the Massey dat file (generally
          'spXX.dat').
        """
        hdu = self.read_fits(file_path, return_hdu_only=True)
        hdr = hdu[1].header
        data = hdu[1].data

        self._print_msg('Calspec file flux unit: %s'%hdr['TUNIT2'])
        
        # wavelength is in A
        spec_ang = np.array([data[ik][0] for ik in range(len(data))])

        # flux is in erg/cm2/s/A
        spec_flux = np.array([data[ik][1] for ik in range(len(data))])

        return spec_ang, spec_flux


    def compute_star_flux_in_frame(self, step, order, filter_file_path,
                                   optics_file_path,
                                   camera_number, airmass=1.):
        """Return flux in ADU/s in an image.

        :param step: Step size in nm
        :param order: Folding order
        :param filter_file_path: Path to the filter file
        :param optics_file_path: Path to the optics file
        :param camera_number: Number of the camera
        :param airmass: (Optional) Airmass (default 1)
        """
        
        STEP_NB = 1000
        camera_number = int(camera_number)
        if camera_number not in (1,2):
            self._print_error('Camera number must be 1 or 2')
        
        (filter_trans,
         filter_min, filter_max) = utils.filters.get_filter_function(
            filter_file_path, step, order, STEP_NB)
        
        nm_axis, std_spectrum = self.get_spectrum(step, order, STEP_NB)

        atm_trans = utils.photometry.get_atmospheric_transmission(
            self._get_atmospheric_extinction_file_path(),
            step, order, STEP_NB, airmass=airmass)

        qe_cam = utils.photometry.get_quantum_efficiency(
            self._get_quantum_efficiency_file_path(camera_number),
            step, order, STEP_NB)

        mirror_trans = utils.photometry.get_mirror_transmission(
            self._get_mirror_transmission_file_path(),
            step, order, STEP_NB)
        
        optics_trans = utils.photometry.get_optics_transmission(
            optics_file_path,
            step, order, STEP_NB)

        star_flux = utils.photometry.compute_star_flux_in_frame(
            nm_axis, std_spectrum,
            filter_trans, optics_trans, atm_trans,
            mirror_trans, qe_cam,
            float(self._get_config_parameter('MIR_SURFACE')),
            float(self._get_config_parameter('CAM{}_GAIN'.format(camera_number))))
        return star_flux
        

    def compute_optimal_texp(self, step, order, seeing, filter_name,
                             camera_number,
                             saturation=30000, airmass=1.):

        """Compute the optimal exposition time given the total flux of
        the star in ADU/s.

        :param step: Step size in nm
        
        :param order: Folding order
        
        :param seeing: Star's FWHM in arcsec
        
        :param filter_name: Name of the filter
        
        :param camera_number: Number of the camera
        
        :param saturation: (Optional) Saturation value of the detector
          (default 30000).

        :param airmass: (Optional) Airmass (default 1)    
        """
        star_flux = self.compute_star_flux_in_frame(
            step, order, self._get_filter_file_path(filter_name),
            self._get_optics_file_path(filter_name),
            camera_number, airmass=airmass)
        
        dimx = float(self._get_config_parameter(
            'CAM{}_DETECTOR_SIZE_X'.format(camera_number)))
        dimy = float(self._get_config_parameter(
            'CAM{}_DETECTOR_SIZE_Y'.format(camera_number)))
        fov = float(self._get_config_parameter(
            'FIELD_OF_VIEW_{}'.format(camera_number)))

        plate_scale = fov / max(dimx, dimy) * 60 # arcsec
        return utils.photometry.compute_optimal_texp(
            star_flux, seeing,
            plate_scale,
            saturation=saturation)
