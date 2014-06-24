#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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

## MODULES IMPORTS

import numpy as np
import bottleneck as bn
import pyfits
from scipy import interpolate
import pp

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
#### CLASS MemFile ##############################
#################################################
        
class MemFile:
    """Manage memory file.

    A memory file is a file containing options that must be shared by
    multiple classes. Options in this file are temporary.

    This class behaves as a dictionary reflecting the content of the
    memory file on the disk.
    """

    mem = dict()

    def __init__(self, memfile_path):
        """Init Memfile Class

        :param memfile_path: Path to the memory file.
        """
        self.memfile_path = memfile_path
        if os.path.exists(self.memfile_path):
            memfile = open(memfile_path, 'r')
            for iline in memfile:
                if len(iline) > 3:
                    iline = iline.split()
                    if len(iline) > 1:
                        self.mem[iline[0]] = iline[1]
            memfile.close()

    def __getitem__(self, key):
        """Implement the evaluation of self[key]"""
        return self.mem[key]

    def __setitem__(self, key, value):
        """Implement the evaluation of self[key] = value"""
        self.mem[key] = value
        self.update()

    def __iter__(self):
        """Return an iterator object"""
        return self.mem.__iter__()

    def update(self):
        """Update memory file with the data contained in self.mem"""
        if len(self.mem) > 0:
            memfile = open(self.memfile_path, 'w')
            for ikey in self.mem:
                memfile.write('%s %s\n'%(ikey, self.mem[ikey]))
            memfile.close()


#################################################
#### CLASS Tools ################################
#################################################

class Tools(object):
    """
    Parent class of all classes of orb.

    Implement basic methods to read and write FITS data, display
    messages, feed a logfile and manage the server for parallel
    processing.
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

    _logfile_name = "./logfile.orb" # Name of the log file

    _memfile_name = "./.memfile.orb"

    _no_log = False # If True no logfile is created

    _tuning_parameters = dict() # Dictionay containing the full names of the
                                # parameter and their new value.

    _silent = False # If True only error messages will be diplayed on screen

    def __init__(self, data_prefix="temp_data", no_log=False,
                 tuning_parameters=dict(), logfile_name=None,
                 config_file_name='config.orb', silent=False):
        """Initialize Tools class.

        :param data_prefix: (Optional) Prefix used to determine the
          header of the name of each created file (default
          'temp_data')

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

        :param config_file_name: (Optional) name of the config file to
          use. Must be located in orb/data/.

        :param silent: If True only error messages will be diplayed on
          screen (default False).
        """
        self.config_file_name = config_file_name
        self._init_logfile_name(logfile_name)
        if (os.name == 'nt'):
            TextColor.disable()
        self._data_prefix = data_prefix
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()
        self._no_log = no_log
        self._tuning_parameters = tuning_parameters
        self.ncpus = int(self._get_config_parameter("NCPUS"))
        self._silent = silent

    def _init_logfile_name(self, logfile_name):
        """Initialize logfile name."""
        memfile = MemFile(self._memfile_name)
        if logfile_name != None:
            self._logfile_name = logfile_name
            memfile['LOGFILENAME'] = self._logfile_name

        if logfile_name == None:
            if 'LOGFILENAME' in memfile:
                self._logfile_name = memfile['LOGFILENAME']

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
                if group == None:
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
            if len(iline) == 3:
                if iline[0] == standard_name:
                    file_path = self._get_orb_data_file_path(iline[2])
                    if os.path.exists(file_path):
                        return file_path, iline[1]

        self._print_error('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')

        
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
        print "\r" + TextColor.ERROR + error_msg + TextColor.END
        if not self._no_log:
            self.open_file(self._logfile_name, "a").write(error_msg + "\n")
        raise StandardError(message)


    def _print_caller_traceback(self):
        """Print the traceback of the calling function and log it into
        the log file.
        """
        traceback = inspect.stack()
        traceback_msg = ''
        for i in range(len(traceback))[::-1]:
            traceback_msg += ('  File %s'%traceback[i][1]
                              + ', line %d'%traceback[i][2]
                              + ', in %s\n'%traceback[i][3] +
                              traceback[i][4][0])
            
        print '\r' + traceback_msg
        if not self._no_log:
            self.open_file(self._logfile_name, "a").write(traceback_msg + "\n")

    def _print_warning(self, message, traceback=False):
        """Print a warning message. No exception is raised.
        
        Warning  messages are written in the log file.
        
        :param message: The message to be displayed.

        :param traceback: (Optional) If True, print traceback (default
          False)
        """
        if traceback:
            self._print_caller_traceback()
            
        warning_msg = self._get_date_str() + self._msg_class_hdr + sys._getframe(1).f_code.co_name + " > Warning: " + message
        if not self._silent:
            print "\r" + TextColor.WARNING + warning_msg + TextColor.END
        if not self._no_log:
            self.open_file(self._logfile_name, "a").write(warning_msg + "\n")

    def _print_msg(self, message, color=False, no_hdr=False):
        """Print a simple message.
        
        Simple messages are written in the log file.

        :param message: The message to be displayed.
        
        :param color: (Optional) If True, the message is diplayed in
          color. If 'alt', an alternative color is displayed.

        :param no_hdr: (Optional) If True, The message is displayed
          as it is, without any header.

        
        """
        if not no_hdr:
            message = (self._get_date_str() + self._msg_class_hdr + 
                       sys._getframe(1).f_code.co_name + " > " + message)

        text_col = TextColor.BLUE
        if color == 'alt':
            color = True
            text_col = TextColor.PURPLE
            
        if color and not self._silent:
            print "\r" + text_col + message + TextColor.END
        elif not self._silent:
            print "\r" + message
        if not self._no_log:
            self.open_file(self._logfile_name, "a").write(message + "\n")

    def _print_traceback(self, no_hdr=False):
        """Print a traceback and log it into the logfile.

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
        if not self._no_log:
            self.open_file(self._logfile_name, "a").write(message + "\n")
            
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
            "FIELD_OF_VIEW"))
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

    def _get_basic_spectrum_cube_header(self, nm_axis, wavenumber=False):
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
        hdr.append(('CRVAL3', nm_axis[0], 'Minimum {} in {}'.format(wave_type,
                                                                    wave_unit)))
        hdr.append(('CUNIT3', '{}'.format(wave_unit), ''))
        hdr.append(('CRPIX3', 1.000000, 'Pixel coordinate of reference point'))
        hdr.append(('CDELT3', 
                    ((nm_axis[-1] - nm_axis[0]) / (len(nm_axis) - 1)), 
                    '{} per pixel'.format(wave_unit)))
        hdr.append(('CROTA3', 0.000000, ''))
        return hdr

    def _init_pp_server(self, silent=False):
        """Initialize a server for parallel processing.

        :param silent: (Optional) If silent no message is printed
          (Default False).

        .. note:: Please refer to http://www.parallelpython.com/ for
          sources and information on Parallel Python software
        """
        ppservers = ()
        
        if self.ncpus == 0:
            job_server = pp.Server(ppservers=ppservers)
        else:
            job_server = pp.Server(self.ncpus, ppservers=ppservers)
            
        ncpus = job_server.get_ncpus()
        if not silent:
            self._print_msg(
                "Init of the parallel processing server with %d threads"%ncpus)
        return job_server, ncpus

    def _close_pp_server(self, js):
        """
        Destroy the parallel python job server to avoid too much
        opened files.
        
        .. note:: Please refer to http://www.parallelpython.com/ for
            sources and information on Parallel Python software.
        """
        # First shut down the normal way
        js.destroy()
        # access job server methods for shutting down cleanly
        js._Server__exiting = True
        js._Server__queue_lock.acquire()
        js._Server__queue = []
        js._Server__queue_lock.release()
        for worker in js._Server__workers:
            worker.t.exiting = True
            try:
                # add worker close()
                worker.t.close()
                os.kill(worker.pid, 0)
                os.waitpid(worker.pid, os.WNOHANG)
            except OSError:
                # PID does not exist
                pass
        
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
        if full_parameter_name in self._tuning_parameters:
            return self._tuning_parameters[full_parameter_name]
        else:
            return default_value

    def _create_list_from_dir(self, dir_path, list_file_path):
        """Create a file containing the list of all FITS files at
        a specified directory and returns the path to the list 
        file.

        :param dir_path: Directory containing the FITS files
        :param list_file_path: Path to the list file to be created
        :returns: Path to the created list file
        """
        if dir_path[-1] != os.sep:
            dir_path += os.sep
        dir_path = os.path.dirname(str(dir_path))
        if os.path.exists(dir_path):
            list_file = self.open_file(list_file_path)
            file_list = os.listdir(dir_path)
            file_list.sort()
            first_file = True
            file_nb = 0
            self._print_msg('Cheking {}'.format(dir_path))
            for filename in file_list:
                if (os.path.splitext(filename)[1] == ".fits"
                    and '_bias.fits' not in filename):
                    file_path = os.path.join(dir_path, filename)
                    if os.path.exists(file_path):
                        fits_hdu = self.read_fits(file_path,
                                                  return_hdu_only=True)
                        dims = fits_hdu[0].header['NAXIS']
                        if first_file:
                            dimx = fits_hdu[0].header['NAXIS1']
                            dimy = fits_hdu[0].header['NAXIS2']
                            first_file = False
                        if (dims == 2
                            and fits_hdu[0].header['NAXIS1'] == dimx
                            and fits_hdu[0].header['NAXIS2'] == dimy):
                            list_file.write(str(file_path)  + "\n")
                            file_nb += 1
                        else:
                            self._print_error("All FITS files in the directory %s do not have the same shape. Please remove bad files."%str(dir_path))
                    else:
                        self._print_error(str(file_path) + " does not exists !")
            if file_nb > 0:
                return list_file_path
            else:
                self._print_error('No FITS file in the folder: %s'%dir_path)
        else:
            self._print_error(str(dir_path) + " does not exists !")
        
    
    def write_fits(self, fits_name, fits_data, fits_header=None,
                    silent=False, overwrite=False, mask=None,
                    replace=False, record_stats=False):
        
        """Write data in FITS format. If the file doesn't exist create
        it with its directories.

        If the file already exists add a number to its name before the
        extension (unless 'overwrite' option is set to True).

        :param fits_name: Path to the file, can be either
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
        if mask != None:
            if np.shape(mask) != np.shape(fits_data):
                self._print_error('Mask must have the same shape as data')
                
        if replace: overwrite=True
        
        if overwrite:
            warnings.filterwarnings('ignore',
                                    message='Overwriting existing file.*',
                                    module='pyfits\.hdu.*')

        if replace and os.path.exists(fits_name):
            old_data = self.read_fits(fits_name)
            if old_data.shape == fits_data.shape:
                fits_data[np.isnan(fits_data)] = old_data[np.isnan(fits_data)]
            else:
                self._print_error("New data shape %s and old data shape %s are not the same. Do not set the option 'replace' to True in this case"%(str(fits_data.shape), str(old_data.shape)))
        
        # float64 data conversion to float32 to avoid too big files
        # with unnecessary precision
        if fits_data.dtype == np.float64:
            fits_data = fits_data.astype(np.float32)

        base_fits_name = fits_name

        dirname = os.path.dirname(fits_name)
        if (dirname != []) and (dirname != ''):
            if not os.path.exists(dirname): 
                os.makedirs(dirname)
        
        index=0
        file_written = False
        while not file_written:
            if ((not (os.path.exists(fits_name))) or overwrite):
                hdu = pyfits.PrimaryHDU(fits_data.transpose())
                if mask != None:
                    # mask conversion to only zeros or ones
                    mask = mask.astype(float)
                    mask[np.nonzero(np.isnan(mask))] = 1.
                    mask[np.nonzero(np.isinf(mask))] = 1.
                    mask[np.nonzero(mask)] = 1.
                    mask = mask.astype(np.uint8) # UINT8 is the
                                                 # tiniest allowed
                                                 # type
                    hdu_mask = pyfits.PrimaryHDU(mask.transpose())
                # add header optional keywords
                if fits_header != None:
                    hdu.header.extend(fits_header, strip=True,
                                      update=True, end=True)
                    # Remove 3rd axis related keywords if there is no
                    # 3rd axis
                    if hdu.header['NAXIS'] == 2:
                        for ikey in range(len(hdu.header)):
                            if isinstance(hdu.header[ikey], str):
                                if ('Wavelength axis' in hdu.header[ikey]):
                                    del hdu.header[ikey]
                                    del hdu.header[ikey]
                                    break
                            
                        del hdu.header['CTYPE3']
                        del hdu.header['CRVAL3']
                        del hdu.header['CRPIX3']
                        del hdu.header['CDELT3']
                        del hdu.header['CROTA3']
                        del hdu.header['CUNIT3']

                # add median and mean of the image in the header
                # data is nan filtered before
                if record_stats:
                    fdata = fits_data[np.nonzero(~np.isnan(fits_data))]
                    if np.size(fdata) > 0:
                        data_mean = bn.nanmean(fdata)
                        data_median = bn.nanmedian(fdata)
                    else:
                        data_mean = np.nan
                        data_median = np.nan
                    hdu.header.set('MEAN', str(data_mean),
                                   'Mean of data (NaNs filtered)',
                                   after=6)
                    hdu.header.set('MEDIAN', str(data_median),
                                   'Median of data (NaNs filtered)',
                                   after=6)
                
                # add some basic keywords in the header
                date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
                hdu.header.set('DATE', date, 'Creation date', after=6)
                hdu.header.set('PROGRAM', "ORB v%s"%__version__, 
                               'Thomas Martin: thomas.martin.1@ulaval.ca',
                               after=6)
                hdu.header.set('MASK', 'False', '', after=6)

                
                # write FITS file
                hdu.writeto(fits_name, clobber=overwrite)
                
                if mask != None:
                    hdu_mask.header = hdu.header
                    hdu_mask.header.set('MASK', 'True', '', after=6)
                    mask_name = self._get_mask_path(fits_name)
                    hdu_mask.writeto(mask_name, clobber=overwrite)
                
                if not (silent):
                    self._print_msg("data written as : " + fits_name)
                return fits_name
            else :
                fits_name = (os.path.splitext(base_fits_name)[0] + 
                             "_" + str(index) + 
                             os.path.splitext(base_fits_name)[1])
                index += 1
        
    def read_fits(self, fits_name, no_error=False, nan_filter=False, 
                  return_header=False, return_hdu_only=False,
                  return_mask=False, silent=False, delete_after=False,
                  data_index=0):
        """Read a FITS data file and returns its data.
        
        :param fits_name: Path to the file, can be either
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
        
        .. note:: Please refer to
          http://www.stsci.edu/institute/software_hardware/pyfits/ for
          more information on PyFITS module. And
          http://fits.gsfc.nasa.gov/ for more information on FITS
          files.
        """
        fits_name = (fits_name.splitlines())[0]
        if return_mask:
            fits_name = self._get_mask_path(fits_name)
            
        try:
            hdulist = pyfits.open(fits_name)
            fits_header = hdulist[data_index].header
        except:
            if not no_error:
                self._print_error(
                    "The file '%s' could not be opened"%fits_name)
                return None
            else:
                if not silent:
                    self._print_warning(
                        "The file '%s' could not be opened"%fits_name)
                return None

        if return_hdu_only:
            return hdulist
        else:
            fits_data = np.array(
                hdulist[data_index].data.transpose()).astype(float)
        
        if (nan_filter):
            fits_data = np.nan_to_num(fits_data)
        hdulist.close
        
        if delete_after:
            try:
                os.remove(fits_name)
            except:
                 self._print_warning(
                     "The file '%s' could not be deleted"%fits_name)
                 
        if return_header:
            return fits_data, fits_header
        else:
            return fits_data

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
            
##################################################
#### CLASS Cube ##################################
##################################################

class Cube(Tools):
    """
    Generate and manage a **virtual frame-divided cube**.
    
    :param image_list_path: Path to the list of images which form the
        virtual cube. If ``image_list_path`` is set to '' then this
        class will not try to load any data.  Can be useful when the
        user don't want to use or process any data.
   
    :param config_file_name: (Optional) name of the config file to
        use. Must be located in orb/data/

    :param data_prefix: (Optional) Prefix used to determine the header
        of the name of each created file (default 'temp_data')

    .. note:: A **frame-divided cube** is a set of frames grouped
      together by a list.  It avoids storing a data cube in one large
      data file and loading an entire cube to process it.
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
    
    _data_prefix = None
    _project_header = None
    _calibration_laser_header = None
    _wcs_header = None

    _silent_load = False

    _mask_exists = None
    
    _return_mask = False # When True, __get_item__ return mask data
                         # instead of 'normal' data

    def __init__(self, image_list_path, data_prefix="temp_data_",
                 config_file_name="config.orb", project_header=list(),
                 wcs_header=list(), calibration_laser_header=list(),
                 overwrite=False, silent_init=False, no_log=False,
                 tuning_parameters=dict(), indexer=None, logfile_name=None):
        """
        Initialize Cube class.
        
        :param image_list_path: Path to the list of images which form
          the virtual cube. If ``image_list_path`` is set to '' then
          this class will not try to load any data.  Can be useful
          when the user don't want to use or process any data.

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

        :param logfile_name: (Optional) Give a specific name to the
          logfile (default None).
        """
        self._init_logfile_name(logfile_name)
        self.overwrite = overwrite
        self._no_log = no_log
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

        if (self.image_list_path != ""):
        # read image list and get cube dimensions  
            image_list_file = self.open_file(self.image_list_path, "r")
            image_name_list = image_list_file.readlines()
            if len(image_name_list) == 0:
                self._print_error('No image path in the given image list')
            is_first_image = True
            for image_name in image_name_list:
                image_name = (image_name.splitlines())[0]
                if is_first_image:          
                    self.image_list = [image_name]
                    image_data = self.read_fits(image_name)
                    self.dimx = image_data.shape[0]
                    self.dimy = image_data.shape[1]
                    is_first_image = False
                    # check if masked frame exists
                    if os.path.exists(self._get_mask_path(image_name)):
                        self._mask_exists = True
                    else:
                        self._mask_exists = False
                elif self._MASK_FRAME_TAIL not in image_name:
                    self.image_list.append(image_name)

            image_list_file.close()
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

        .. note:: There is no interpretation of the negative indexes.
        
        .. note:: To make this function silent just set
          Cube()._silent_load to True.
        """
        def _get_default_slice(_slice, _max):
            if isinstance(_slice, slice):
                if _slice.start != None:
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

                if _slice.stop != None:
                    if (isinstance(_slice.stop, int)
                        or isinstance(_slice.stop, long)):
                        if ((_slice.stop >= 0) and (_slice.stop <= _max)
                            and _slice.stop > slice_min):
                            slice_max = int(_slice.stop)
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
        
        def _get_frame_section(cube, x_slice, y_slice, frame_index):
            hdu = cube.read_fits(cube.image_list[frame_index],
                                 return_hdu_only=True,
                                 return_mask=cube._return_mask)
            section = np.copy(hdu[0].section[y_slice, x_slice].transpose())
            del hdu
            cube._return_mask = False # always reset self._return_mask to False
            return section


        # check return mask possibility
        if self._return_mask and not self._mask_exists:
            self._print_error("No mask found with data, cannot return mask")
        
        # produce default values for slices
        x_slice = _get_default_slice(key[0], self.dimx)
        y_slice = _get_default_slice(key[1], self.dimy)
        z_slice = _get_default_slice(key[2], self.dimz)
        
        # get first frame
        data = _get_frame_section(self, x_slice, y_slice, z_slice.start)
        # return this frame if only one frame is wanted
        if z_slice.stop == z_slice.start + 1L:
            return data

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
                _get_frame_section,
                args=(self, x_slice, y_slice, ik+ijob),
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
            
        return data

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
        hdu = self.read_fits(self.image_list[index],
                             return_hdu_only=True)
        return hdu[0].header

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
        resized_frame = np.empty((size_x, size_y), dtype=float)
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
        resized_cube = np.empty((size_x, size_y, self.dimz), dtype=float)
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
        """Return the energy map of an interferogram cube"""
        mean_map = self.get_mean_image()
        energy_map = np.zeros((self.dimx, self.dimy), dtype=float)
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
        energy_map = np.zeros((self.dimx, self.dimy), dtype=float)
        progress = ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            frame = self.get_data_frame(_ik)
            non_nans = np.nonzero(~np.isnan(frame))
            energy_map[non_nans] += (frame[non_nans])**2.
            progress.update(_ik, info="Creating spectrum energy map")
        progress.end()
        return np.sqrt(energy_map) / self.dimz

    def get_mean_image(self):
        """Return the mean image of a cube (corresponding to a deep
        frame for an interferogram cube or a specral cube).

        .. note:: In this process NaNs are considered as zeros.
        """
        if self.mean_image == None:
            mean_im = np.zeros((self.dimx, self.dimy), dtype=float)
            progress = ProgressBar(self.dimz)
            for _ik in range(self.dimz):
                frame = self.get_data_frame(_ik)
                non_nans = np.nonzero(~np.isnan(frame))
                mean_im[non_nans] += frame[non_nans]
                progress.update(_ik, info="Creating mean image")
            progress.end()
            self.mean_image = mean_im / self.dimz
        return self.mean_image

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
        
        def get_frame_stat(cube, ik, nozero, stat_key, center,
                           xmin, xmax, ymin, ymax):
            if not nozero and not center:
                frame_hdr = cube.get_frame_header(ik)
                if stat_key in frame_hdr:
                    if not np.isnan(float(frame_hdr[stat_key])):
                        return float(frame_hdr[stat_key])
                        
            frame = np.copy(cube.get_data(
                xmin, xmax, ymin, ymax, ik, ik+1)).astype(float)
            
            if nozero: # zeros filtering
                frame[np.nonzero(frame == 0)] = np.nan
                
            if np.all(np.isnan(frame)):
                return np.nan
            if stat_key == 'MEDIAN':
                statf, a = bn.func.nanmedian_selector(frame, None)
            elif stat_key == 'MEAN':
                statf, a = bn.func.nanmean_selector(frame, None)
            elif stat_key == 'STD':
                statf, a = bn.func.nanstd_selector(frame, None)
            return statf(a)


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
            if self.z_mean == None: stat_key = 'MEAN'
            else: return self.z_mean
            
        elif stat == 'median':
            if self.z_median == None: stat_key = 'MEDIAN'
            else: return self.z_median
            
        elif stat == 'std':
            if self.z_std == None: stat_key = 'STD'
            else: return self.z_std
            
        else: self._print_error(
            "Bad stat option. Must be 'mean', 'median' or 'std'")
        
        stat_vector = np.empty(self.dimz, dtype=float)

        job_server, ncpus = self._init_pp_server()
        progress = ProgressBar(self.dimz)
        for ik in range(0, self.dimz, ncpus):
            if ik + ncpus >= self.dimz:
                ncpus = self.dimz - ik

            jobs = [(ijob, job_server.submit(
                get_frame_stat, 
                args=(self, ik + ijob, nozero, stat_key, center,
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
        if div_nb == None: 
            div_nb = self.DIV_NB
        quad_nb = div_nb**2
        if dimx == None: dimx = self.dimx
        if dimy == None: dimy = self.dimy

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
       
    def export(self, export_path, x_range=None, y_range=None, z_range=None,
               fits_header=None, overwrite=False):
        """Export cube as one FITS file.

        :param export_path: Path of the exported FITS file

        :param x_range: Tuple (x_min, x_max)
        
        :param y_range: Tuple (y_min, y_max)
        
        :param z_range: Tuple (z_min, z_max)
        """
        if x_range == None:
            xmin = 0
            xmax = self.dimx
        else:
            xmin = np.min(x_range)
            xmax = np.max(x_range)
            
        if y_range == None:
            ymin = 0
            ymax = self.dimy
        else:
            ymin = np.min(y_range)
            ymax = np.max(y_range)

        if z_range == None:
            zmin = 0
            zmax = self.dimz
        else:
            zmin = np.min(z_range)
            zmax = np.max(z_range)

        data = np.empty((xmax-xmin, ymax-ymin, zmax-zmin), dtype=float)
        
        job_server, ncpus = self._init_pp_server()
        progress = ProgressBar(zmax-zmin)
        for iframe in range(0, zmax-zmin, ncpus):
            progress.update(iframe, info='exporting data')
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
                        fits_header=fits_header)

##################################################
#### CLASS ProgressBar ###########################
##################################################


class ProgressBar:
    """Display a simple progress bar in the terminal

    :param max_index: The index considered as 100%
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
            sys.stdout.write("\r" + " " * self.MAX_CARAC)
            sys.stdout.flush()

    def _time_str_convert(self, sec):
        """Convert a number of seconds in a human readable string
        
        :param sec: Number of seconds to convert
        """
        
        if (sec < 60.):
            return str(int(sec)) + "s"
        elif (sec < 3600.):
            minutes = int(math.floor(sec/60.))
            seconds = int(sec - (minutes * 60.))
            return str(minutes) + "m" + str(seconds) + "s"
        else:
            hours = int(math.floor(sec/3600.))
            minutes = int(math.floor((sec - (hours*3600.))/60.))
            seconds = int(sec - (hours * 3600.) - (minutes * 60.))
            return str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s"


    def update(self, index, info=""):
        """Update the progress bar.

        :param index: Index representing the progress of the
          process. Must be less than index_max.
          
        :param info: (Optional) Information to be displayed as
          comments
        """
        if (self._max_index > 0):
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
            line = (TextColor.BLUE + "\r [" + "="*int(math.floor(pos)) + 
                    " "*int(self._bar_length - math.floor(pos)) + 
                    "] [%d%%] [" %(pos*100./self._bar_length) + 
                    str(info) +"] [remains: " + 
                    self._time_str_convert(time_to_end) + "]" + TextColor.END)
        else:
            line = (TextColor.GREEN + "\r [please wait] [" +
                    str(info) +"]" + TextColor.END)
        self._erase_line()
        if (len(line) > self.MAX_CARAC):
            rem_len = len(line) - self.MAX_CARAC + 1
            line = line[:-len(TextColor.END)]
            line = line[:-(rem_len+len(TextColor.END))]
            line += TextColor.END
        if not self._silent:
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
                        self._time_str_convert(time.time() -
                                               self._start_time))
            if not self._silent:
                sys.stdout.write("\n")
        else:
            self._erase_line()
            self.update(self._max_index, info="completed in " +
                        self._time_str_convert(time.time() - self._start_time))
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
        
        if self.file_group != None:
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
        elif file_group != None:
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
            
        if (file_group in self.file_groups) or (file_group == None):
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
                    'Hepsilon':397.007,
                    'Hdelta':410.176,
                    'Hgamma':434.047,
                    '[OIII]4363':436.321,
                    'Hbeta':486.133,
                    '[OIII]4959':495.892,
                    '[OIII]5007':500.684,
                    '[NII]6548':654.803,
                    'Halpha':656.280,
                    '[NII]6583':658.341,
                    '[SII]6716':671.647,
                    '[SII]6731':673.085}
    """Air emission lines wavelength"""

    vac_lines_name = None
    air_lines_name = None
    
    def __init__(self):
        """Lines class constructor"""
        Tools.__init__(self)

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

    def get_sky_lines(self, nm_min, nm_max, delta_nm, line_nb=0):
        """Return sky lines in a range of optical wavelength.

        :param nm_min: min wavelength of the lines in nm
        
        :param nm_max: max wavelength of the lines in nm

        :param delta_nm: wavelength resolution in nm as the minimum
          wavelength interval of the spectrum. Lines comprises in half
          of this interval are merged.
        
        :param line_nb: (Optional) number of the most intense lines to
          retrive. If 0 all lines are given (default 0).
        """
        def merge(merged_lines):
            merged_lines = np.array(merged_lines)
            return (np.mean(merged_lines[:,0]),
                    np.sum(merged_lines[:,1]))
            

        lines = [line_nm for line_nm in self.air_sky_lines_nm.itervalues()
                 if (line_nm[0] >= nm_min and line_nm[0] <= nm_max)]

        

        lines.sort(key=lambda l: l[0])
        merged_lines = list()
        final_lines = list()
        for iline in range(len(lines)):
            if iline + 1 < len(lines):
                if abs(lines[iline][0] - lines[iline+1][0]) < delta_nm / 2.:
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
            merged_line = merge(merged_lines)
            final_lines.append(merge(merged_lines))

        lines = final_lines
        
        # get only the most intense lines
        if line_nb > 0:
            lines.sort(key=lambda l: l[1], reverse=True)
            lines = lines[:line_nb]

        lines = list(np.array(lines)[:,0])

        # add balmer lines
        balmer_lines = ['Halpha', 'Hbeta', 'Hgamma', 'Hdelta', 'Hepsilon']
        for iline in balmer_lines:
            if (self.air_lines_nm[iline] >= nm_min
                and self.air_lines_nm[iline] <= nm_max):
                lines.append(self.air_lines_nm[iline])
                
        lines.sort()
        return lines
        

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
        if isinstance(lines, (float, int)):
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
    """

    option_file = None # the option file object
    options = None # the core list containing the parsed file
    lines = None # a core.Lines instance

    protected_keys = ['REG', 'TUNE'] # special keywords that can be
                                     # used mulitple times without
                                     # being overriden.
    
    def __init__(self, option_file_path):
        """Initialize class

        :param option_file_path: Path to the option file
        """
        self.option_file = self.open_file(option_file_path, 'r')
        self.options = dict()
        self.lines = Lines()
        for line in self.option_file:
            if len(line) > 2:
                if line[0] != '#':
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

    def get_lines(self):
        """Get lines parameters.

        Defined for the special keyword 'LINES'.
        """
        lines_names = self['LINES']
        lines_nm = list()
        if len(lines_names) > 2:
            lines_names = lines_names.split(',')
            for iline in lines_names:
                try:
                    lines_nm.append(float(iline))
                except:
                    lines_nm.append(self.lines.get_line_nm(iline))
        else:
            return None
        
        return lines_nm
            
    def get_filter_edges(self):
        """Get filter eges parameters.

        Defined for the special keyword 'FILTER_EDGES'.
        """
        filter_edges = self['FILTER_EDGES']
        if filter_edges != None:
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
                print ibad
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
    """Manage the correspondance between multiple dict containing the
    same parameters and a file on the disk.
    """

    _params_list = None
    _keys = None

    f = None
    
    def __init__(self, file_path, reset=True):
        """Init ParamsFile class.

        :param file_path: Path of the output file where all
          parameters are stored (Note that this file will
          automatically be overwritten if reset is set to True).

        :param reset: (Optional) If True the output file is
          overwritten. If False and if the output file already exists
          data in the file are read and new data is appended (default
          True).
        """
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
                
        

        

