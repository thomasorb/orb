#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

## Copyright (c) 2010-2018 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import orb.version
__version__ = orb.version.__version__

## BASIC IMPORTS
import os
import sys
import time
import math
import traceback
import inspect
import re
import datetime
import logging
import warnings

import threading
import socketserver
import logging.handlers
import struct
import pickle
import select
import socket

import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.io.fits.verify import VerifyWarning, VerifyError, AstropyUserWarning
import pylab as pl

import gvar
import h5py

import scipy.fftpack
import scipy.interpolate
import pandas

try: import pygit2
except ImportError: pass

## MODULES IMPORTS
import orb.cutils
import orb.utils.spectrum, orb.utils.parallel, orb.utils.io, orb.utils.filters
import orb.utils.photometry

#################################################
#### CLASS TextColor ############################
#################################################

class TextColor:
    """Define ANSI Escape sequences to display text with colors."""
    DEFAULT = '\x1b[0m'
    RED     = '\x1b[31m'
    GREEN   = '\x1b[32m'
    YELLOW  = '\x1b[33m'
    CYAN    = '\x1b[36m'
    OKGREEN = '\033[92m'
    KORED = '\033[91m'
    END = DEFAULT


#################################################
#### CLASS ColorStreamHandler ###################
#################################################

class ColorStreamHandler(logging.StreamHandler):
    """Manage colored logging

    copied from https://gist.github.com/mooware/a1ed40987b6cc9ab9c65
    """
    
    CRITICAL = TextColor.RED
    ERROR    = TextColor.RED
    WARNING  = TextColor.YELLOW
    INFO     = TextColor.DEFAULT
    DEBUG    = TextColor.CYAN
    DEFAULT  = TextColor.DEFAULT

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:  return cls.CRITICAL
        elif level >= logging.ERROR:   return cls.ERROR
        elif level >= logging.WARNING: return cls.WARNING
        elif level >= logging.INFO:    return cls.INFO
        elif level >= logging.DEBUG:   return cls.DEBUG
        else:                          return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return color + text + self.DEFAULT

#################################################
#### CLASS LoggingFilter ########################
#################################################
class NoLoggingFilter(logging.Filter):
    def filter(self, record):
        return True

class ExtInfoLoggingFilter(logging.Filter):
    bad_names = ['pp']
    def filter(self, record):
        if record.levelname in ['INFO']:
            if record.module in self.bad_names: return False
        return True

class ExtDebugLoggingFilter(logging.Filter):
    bad_names = ['pp']
    def filter(self, record):
        if record.levelname in ['INFO', 'DEBUG']:
            if record.module in self.bad_names: return False
        return True

class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for a streaming logging request.

    This basically logs the record using whatever logging policy is
    configured locally.
    """
    def handle(self):
        """
        Handle multiple requests - each expected to be a 4-byte length,
        followed by the LogRecord in pickle format. Logs the record
        according to whatever policy is configured locally.
        """
        #while True:
        chunk = self.connection.recv(4)
        if len(chunk) < 4: return
        slen = struct.unpack('>L', chunk)[0]
        chunk = self.connection.recv(slen)
        while len(chunk) < slen:
            chunk = chunk + self.connection.recv(slen - len(chunk))
        obj = self.unPickle(chunk)
        record = logging.makeLogRecord(obj)
        self.handleLogRecord(record)

    def unPickle(self, data):
        return pickle.loads(data)

    def handleLogRecord(self, record):
        # if a name is specified, we use the named logger rather than the one
        # implied by the record.
        logger = logging.getLogger()
        # N.B. EVERY record gets logged. This is because Logger.handle
        # is normally called AFTER logger-level filtering. If you want
        # to do filtering, do it at the client end to save wasting
        # cycles and network bandwidth!
        logger.handle(record)
    
class LogRecordSocketReceiver(socketserver.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """
    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        socketserver.ThreadingTCPServer.__init__(self, (host, port), handler)
        self.abort = False
        self.timeout = True

    def server_bind(self):
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)

    def serve_until_stopped(self):
        abort = False
        try:
            while not abort:
                rd, wr, ex = select.select([self.socket.fileno()],
                                           [], [], self.timeout)
                if rd:
                    self.handle_request()
                abort = self.abort
                time.sleep(0.001)
                
        except Exception:
            pass


#################################################
#### CLASS Logger ###############################
#################################################
class Logger(object):
    
    logfilters = {
        'default': ExtDebugLoggingFilter(),
        'extinfo': ExtInfoLoggingFilter(),
        'extdebug': ExtDebugLoggingFilter(),
        'none': NoLoggingFilter()}

    
    def __init__(self, debug=False, logfilter='default'):
        """Init

        :param logfilter: If set to None, no logfilter will be applied
          (default 'default')

        """
        self.logfilter = self.get_logfilter(logfilter)
        self.debug = bool(debug)
        
        if self.debug:
            self.level = logging.DEBUG
        else:
            self.level = logging.INFO

        self.branch_name = orb.version.__version__ + '|'
        
        self.start_logging()
        
        # start tcp listener
        if self.debug:
            try:
                self.listen()
            except Exception as e:
                warnings.warn('Exception occured during logging server init (maybe it is already initialized): {}'.format(e))


    def _reset_logging_state(self, logfilter=None):
        """Force a logging reset

        :param logfilter: If set to None, default logfilter set at
          init will be applied (default None)
        """
        def excepthook_with_log(exctype, value, tb):
            try:
                logging.error(value, exc_info=(exctype, value, tb))
            except Exception: pass

        # clear old logging state
        self.root = self.getLogger()
        [self.root.removeHandler(ihand) for ihand in self.root.handlers[:]]
        [self.root.removeFilter(ihand) for ifilt in self.root.filters[:]]

        # init logging
        self.root.setLevel(self.level)
            
        ch = ColorStreamHandler()
        ch.setLevel(self.level)
            
        if self.debug:
            formatter = logging.Formatter(
                self.get_logformat(),
                self.get_logdateformat())
        else:
            formatter = logging.Formatter(
                self.get_simplelogformat(),
                self.get_logdateformat())
        ch.setFormatter(formatter)
        ch.addFilter(self.get_logfilter(logfilter))
        self.root.addHandler(ch)

        logging.captureWarnings(True)

        sys.excepthook = excepthook_with_log

    def getLogger(self):
        return logging.getLogger()

    def get_logfilter(self, logfilter):
        if logfilter is None:
            if hasattr(self, 'logfilter'):
                return self.logfilter
            else: return self.logfilters['none']
            
        elif logfilter not in self.logfilters:
            raise ValueError('logfilter must be in {}'.format(list(self.logfilters.keys())))
        return self.logfilters[logfilter]
    
    def start_logging(self, logfilter=None):
        """Reset logging only if logging is not set

        :param logfilter: If set to None, default logfilter set at
          init will be applied (default None)
        """
        if not self.get_logging_state():
            self._reset_logging_state(logfilter=logfilter)

    def start_file_logging(self, logfile_path=None,
                           logfilter=None):
        """Start file logging

        :param logfile_path: Path to the logfile. If none is provided
          a default logfile path is used."""
        self.logfile_path = logfile_path
        self.start_logging()

        if not self.get_file_logging_state():
            self.root = self.getLogger()
            self.root.setLevel(self.level)

            ch = logging.StreamHandler(
                open(self._get_logfile_path(), 'a'))
            ch.setLevel(self.level)
            formatter = logging.Formatter(
                self.get_logformat(),
                self.get_logdateformat())
            ch.setFormatter(formatter)
            ch.addFilter(self.get_logfilter(logfilter))
            self.root.addHandler(ch)

    def get_logging_state(self):
        """Return True if the logging is set"""
        _len = len(self.getLogger().handlers)
        if _len == 0: return False
        elif _len < 3: return True
        else:
            raise Exception('Logging in strange state: {}'.format(self.getLogger().handlers))

    def get_file_logging_state(self):
        """Return True if the file logging appears set"""
        _len = len(self.getLogger().handlers)
        if _len < 2: return False
        elif _len == 2: return True
        else:
            raise Exception('File Logging in strange state: {}'.format(self.getLogger().handlers))
        
        
    def get_logformat(self):
        """Return a string describing the logging format"""        
        return '%(asctime)s|%(module)s:%(lineno)s:%(funcName)s|{}%(levelname)s> %(message)s'.format(self.branch_name)

    def get_simplelogformat(self):
        """Return a string describing the simple logging format"""
        return '{}%(levelname)s| %(message)s'.format(self.branch_name)


    def get_logdateformat(self):
        """Return a string describing the logging date format"""
        return '%y%m%d-%H:%M:%S'

    def _get_logfile_path(self):
        """Return logfile name"""
        if self.logfile_path is None:
            today = datetime.datetime.today()
            self.logfile_path = 'orb.{:04d}{:02d}{:02d}.log'.format(
                today.year, today.month, today.day)
        return self.logfile_path
    
    def listen(self):
        """Listen and handle logging sent on TCP socket"""
        # start socket listener
        logging.debug('logging listener started')
        listener = LogRecordSocketReceiver()

        thread = threading.Thread(
            target=listener.serve_until_stopped, args=())
        thread.daemon = True # Daemonize thread
        try:
            thread.start() # Start the execution
        except Exception as e:
            warnings.warn('Error during listener execution')



################################################
#### CLASS Params ############################
################################################
class Params(dict):
    """Special dictionary which elements can be accessed like
    attributes.
    """
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__
    

    def __getstate__(self):
        """Used to pickle object"""
        state = self.copy()
        return state

    def __setstate__(self, state):
        """Used to unpickle object"""
        self.update(state)

    def convert(self):
        """Convert to a nice pickable object"""
        conv = dict()
        conv.update(self)
        return conv

    def save(self, path):
        """Try to save data in an HDF5 format.

        .. warning:: All elements might not be saved
        """
        data = self.convert()
        with orb.utils.io.open_hdf5(path, 'w') as f:
            for ikey in data:
                try:
                    f.create_dataset(ikey, data=data[ikey])
                except TypeError:
                    logging.debug('error saving {} of type {}'.format(ikey, type(data[ikey])))
                    
    def load(self, path):
        """Load data from an HDF5 file saved with save method.
        """
        with orb.utils.io.open_hdf5(path, 'r') as f:
            for ikey in f:
                val = f[ikey]
                if val.shape == ():
                    self[ikey] = val[()]
                else:
                    self[ikey] = val[:]
        return self
    
################################################
#### CLASS ROParams ############################
################################################
class ROParams(Params):
    """Special dictionary which elements can be accessed like
    attributes.

    Attributes are read-only and may be defined only once.
    """
    def __setattr__(self, key, value):
        """Special set attribute function. Always raise a read-only
        error.

        :param key: Attribute name.

        :param value: Attribute value.
        """
        raise Exception('Parameter is read-only')

    def __setitem__(self, key, value):
        """Special set item function. Raises a warning when parameter
        already exists.
    
        :param key: Item key.

        :param value: Item value.
        """
        if key in self:
            try:
                if not np.all(np.isclose(self[key] - value, 0.)):
                    logging.debug('Parameter {} already defined'.format(key))
                    logging.debug('Old value={} / new_value={}'.format(self[key], value))
            except TypeError:
                pass
        dict.__setitem__(self, key, value)
    
    def reset(self, key, value):
        """Force parameter reset"""
        dict.__setitem__(self, key, value)

################################################
#### CLASS NoInstrumentConfigParams ############
################################################
class NoInstrumentConfigParams(ROParams):
    """Special dictionary which elements can be accessed like
    attributes.

    Attributes are read-only and may be defined only once.
    """
    def __getitem__(self, key):
        if key not in self:
            raise AttributeError("Instrument configuration not loaded. Set the option 'instrument' to a valid instrument name")
        return ROParams.__getitem__(self, key)

    __getattr__ = __getitem__

     
#################################################
#### CLASS Tools ################################
#################################################
class Tools(object):
    """Base parent class for processing classes.

    Load instrument config file and give access to orb data files.
    """
    instruments = ['sitelle', 'spiomm']
    filters = ['SN1', 'SN2', 'SN3', 'C1', 'C2', 'C3', 'C4', 'FULL', 'PS1_r', 'PS1_i', 'PS1_g', 'PS1_y', 'PS1_z', 'F656N']
                
    def __init__(self, instrument=None, config=None,
                 data_prefix="./temp/data."):
        """Initialize Tools class.

        :param instrument: (Optional) Instrument configuration to
          load. If None a minimal configuration file is loaded. Some
          functions are not available in this case (default None).

        :param data_prefix: (Optional) Prefix used to determine the
          header of the name of each created file (default
          'temp_data')

        :param config: (Optional) Configuration dictionary to update
          default loaded config values (default None).
        """
        if instrument is not None:
            if instrument in self.instruments:
                self.config = ROParams()
                self.config_file_name = 'config.{}.orb'.format(instrument)
            else:
                raise ValueError(
                    "instrument must be in {}".format(self.instruments))
        else:
            self.config_file_name = 'config.none.orb'
            self.config = NoInstrumentConfigParams()
        
        # loading minimal config
        self.instrument = instrument
        self.set_config('DIV_NB', int)
        self.config['QUAD_NB'] = self.config.DIV_NB**2
        self.set_config('BIG_DATA', bool)
        self.set_config('DETECT_STAR_NB', int)
        self.set_config('INIT_FWHM', float)
        self.set_config('PSF_PROFILE', str)
        self.set_config('MOFFAT_BETA', float)
        self.set_config('DETECT_STACK', int)
        self.set_config('ALIGNER_RANGE_COEFF', float)
            
        if self.instrument is not None:
            # load instrument configuration
            self.set_config('OBSERVATORY_NAME', str)
            self.set_config('TELESCOPE_NAME', str)
            self.set_config('INSTRUMENT_NAME', str)

            self.set_config('OBS_LAT', float)
            self.set_config('OBS_LON', float)
            self.set_config('OBS_ALT', float)
            
            self.set_config('ATM_EXTINCTION_FILE', str)
            self.set_config('MIR_TRANSMISSION_FILE', str)
            self.set_config('MIR_SURFACE', float)
            
            self.set_config('FIELD_OF_VIEW_1', float)
            self.set_config('FIELD_OF_VIEW_2', float)
            
            self.set_config('PIX_SIZE_CAM1', float)
            self.set_config('PIX_SIZE_CAM2', float)

            self.set_config('BALANCED_CAM', int)

            self.set_config('CAM1_DETECTOR_SIZE_X', int)
            self.set_config('CAM1_DETECTOR_SIZE_Y', int)
            self.set_config('CAM2_DETECTOR_SIZE_X', int)
            self.set_config('CAM2_DETECTOR_SIZE_Y', int)

            self.set_config('CAM1_GAIN', float)
            self.set_config('CAM2_GAIN', float)
            self.set_config('CAM1_QE_FILE', str)
            self.set_config('CAM2_QE_FILE', str)
        
            self.set_config('OFF_AXIS_ANGLE_MIN', float)
            self.set_config('OFF_AXIS_ANGLE_MAX', float)
            self.set_config('OFF_AXIS_ANGLE_CENTER', float)

            self.set_config('INIT_ANGLE', float)
            self.set_config('INIT_DX', float)
            self.set_config('INIT_DY', float)
            self.set_config('CALIB_NM_LASER', float)
            self.set_config('CALIB_ORDER', int)
            self.set_config('CALIB_STEP_SIZE', float)
            self.set_config('PHASE_FIT_DEG', int)
            self.set_config('PHASE_BINNING', int)

            self.set_config('NCPUS', int)
            self.set_config('DIV_NB', int)
            self.set_config('BIG_DATA', bool)
            self.set_config('OPTIM_DARK_CAM1', bool) 
            self.set_config('OPTIM_DARK_CAM2', bool)
            self.set_config('EXT_ILLUMINATION', bool)

            self.set_config('SATURATION_THRESHOLD', float)
            self.set_config('WCS_ROTATION', float)
            
            self.set_config('OPD_JITTER', float)
            self.set_config('WF_ERROR', float)
            self.set_config('4RT_FILE', str)

            
            # optional parameters
            self.set_config('BIAS_CALIB_PARAM_A', float)
            self.set_config('BIAS_CALIB_PARAM_B', float)
            self.set_config('DARK_ACTIVATION_ENERGY', float)
            

        self._data_prefix = data_prefix
        self._data_path_hdr = self._get_data_path_hdr()

        if config is not None:
            self.update_config(config)


    def set_config(self, key, cast, optional=False):
        """Set configuration parameter (from the configuration file)
        """
        if optional:
            if self._get_config_parameter(key, optional=optional) is None:
                return None
            
        if cast is not bool:
            self.config[key] = cast(self._get_config_parameter(key))
        else:
            self.config[key] = bool(int(self._get_config_parameter(key)))


    def update_config(self, config):
        """Update config values from a dict

        :param config: Configuration dictionary
        """
        if not isinstance(config, dict):
            raise TypeError('config must be a dict')
        for ikey in config:
            if ikey in self.config:
                self.config[ikey] = type(self.config[ikey])(config[ikey])
            else:
                logging.debug('Unknown config key: {}'.format(ikey))
                self.config[ikey] = config[ikey]
            
            
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
        if self.config_file_name is None:
            raise Exception('No instrument configuration given')
        config_file_path = self._get_orb_data_file_path(
            self.config_file_name)
        if not os.path.exists(config_file_path):
             raise Exception(
                 "Configuration file %s does not exist !"%config_file_path)
        return config_file_path

    def _parse_filter_name(self, filter_name):
        """Parse a filter name which can sometimes be a full filter file path
        """
        for ifilter in self.filters:
            if ifilter in filter_name: return ifilter
        raise Exception('this name or path does not point to any known filter: {}'.format(
            filter_name))

    def _get_filter_file_path(self, filter_name):
        """Return the full path to the filter file given the name of
        the filter.

        The filter file name must be filter_FILTER_NAME and it must be
        located in orb/data/.

        :param filter_name: Name of the filter.
        """
        filter_name = self._parse_filter_name(filter_name)
        filter_file_path =  self._get_orb_data_file_path(
            "filter_" + filter_name + ".hdf5")
        if not os.path.exists(filter_file_path):
             warnings.warn(
                 "Filter file %s does not exist !"%filter_file_path)
             return None
         
        return filter_file_path

    
    def _get_phase_file_path(self, filter_name):
        """Return the full path to the phase file given the name of
        the filter.

        The file name must be 'phase_FILTER_NAME.orb' and it must be
        located in orb/data/.

        :param filter_name: Name of the filter.
        """
        filter_name = self._parse_filter_name(filter_name)
        phase_file_path =  self._get_orb_data_file_path(
            "phase_" + filter_name + ".hdf5")
        
        if not os.path.exists(phase_file_path):
             warnings.warn(
                 "Phase file %s does not exist !"%phase_file_path)
             return None
         
        return phase_file_path

    def _get_sip_file_path(self, camera_number):
        """Return the full path to the FITS file containing the SIP
        header given the camera number.
    

        The file name must be 'sip.*.fits' and it must be
        located in orb/data/.

        :param camera_number: Camera number (can be 1,2 or 0 for
          merged data)
        """
        if camera_number == 0: cam_name = 'merged'
        elif camera_number == 1: cam_name = 'cam1'
        elif camera_number == 2: cam_name = 'cam2'
        else: raise Exception('Bad camera number, must be 0, 1 or 2')
        
        sip_file_path =  self._get_orb_data_file_path(
            "sip." + cam_name + ".fits")
        
        if not os.path.exists(sip_file_path):
             warnings.warn(
                 "SIP file %s does not exist !"%sip_file_path)
             return None
         
        return sip_file_path


    def _get_optics_file_path(self, filter_name):
        """Return the full path to the optics transmission file given
        the name of the filter.

        The filter file name must be filter_FILTER_NAME and it must be
        located in orb/data/.

        :param filter_name: Name of the filter.
        """
        filter_name = self._parse_filter_name(filter_name)
        optics_file_path =  self._get_orb_data_file_path(
            "optics_" + filter_name + ".hdf5")
        if not os.path.exists(optics_file_path):
             warnings.warn(
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
             raise Exception(
                 "Standard table %s does not exist !"%standard_table_path)
        return standard_table_path


    def _get_standard_list(self, standard_table_name='std_table.orb',
                           group=None):
        """Return the list of standards recorded in the standard table

        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).
        """
        groups = ['MASSEY', 'MISC', 'CALSPEC', 'OKE', None]
        if group not in groups:
            raise Exception('Group must be in %s'%str(groups))
        std_table = orb.utils.io.open_file(self._get_standard_table_path(
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
          can be 'MASSEY', 'CALSPEC', 'MISC' or 'OKE'.
        """
        std_table = orb.utils.io.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')

        for iline in std_table:
            iline = iline.split()
            if len(iline) >= 3:
                if iline[0] in standard_name:
                    file_path = self._get_orb_data_file_path(iline[2])
                    if os.path.exists(file_path):
                        return file_path, iline[1]

        raise Exception('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')

    def _get_standard_radec(self, standard_name,
                            standard_table_name='std_table.orb',
                            return_pm=False):
        """
        Return standard RA and DEC and optionally PM
        
        :param standard_name: Name of the standard star. Must be
          recorded in the standard table.
        
        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).

        :param return_pm: (Optional) Returns also proper motion if
          recorded (in mas/yr), else returns 0.
        """
        std_table = orb.utils.io.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')

        for iline in std_table:
            iline = iline.strip().split()
            if len(iline) >= 3:
                if iline[0] in standard_name:
                    if len(iline) > 3:
                        ra = float(iline[3])
                        dec = float(iline[4])
                        if len(iline) > 5:
                            pm_ra = float(iline[5])
                            pm_dec = float(iline[6])
                        else:
                            pm_ra = 0.
                            pm_dec = 0.
                        if return_pm:
                            return ra, dec, pm_ra, pm_dec
                        else:
                            return ra, dec
                    else:
                        raise Exception('No RA DEC recorded for standard: {}'.format(
                            standard_name))
                    

        raise Exception('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')


    def _get_atmospheric_extinction_file_path(self):
        """Return the path to the atmospheric extinction file"""
        file_name = self.config.ATM_EXTINCTION_FILE
        return self._get_orb_data_file_path(file_name)

    def _get_mirror_transmission_file_path(self):
        """Return the path to the telescope mirror transmission file"""
        file_name = self.config.MIR_TRANSMISSION_FILE
        return self._get_orb_data_file_path(file_name)

    def _get_quantum_efficiency_file_path(self, camera_number):
        """Return the path to the quantum efficiency file

        :param camera_number: Number of the camera, can be 1 or 2.
        """
        file_name = self.config['CAM{}_QE_FILE'.format(camera_number)]
        return self._get_orb_data_file_path(file_name)

    def _get_4rt_file_path(self):
        """Return the path to 4RT transmission file"""
        file_name = self.config['4RT_FILE']
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
        with orb.utils.io.open_file(self._get_config_file_path(), 'r') as f:
            for line in f:
                if len(line) > 2:
                    if line.split()[0] == param_key:
                        return line.split()[1]
        if not optional:
            raise Exception("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
        else:
            warnings.warn("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
            return None

    def _get_ncpus(self):
        """Return the number of CPUS available
        """
        return orb.utils.parallel.get_ncpus(int(self.config.NCPUS))
        
    def _init_pp_server(self, silent=False, timeout=100):
        """Initialize a server for parallel processing.

        :param silent: (Optional) If silent no message is printed
          (Default False).

        :param timeout: (Optional) Job timeout in s.

        .. note:: Please refer to http://www.parallelpython.com/ for
          sources and information on Parallel Python software
        """
        return orb.utils.parallel.init_pp_server(
            ncpus=int(self.config.NCPUS),
            silent=silent, timeout=timeout)

    def _close_pp_server(self, js):
        """
        Destroy the parallel python job server to avoid too much
        opened files.

        :param js: job server.
        
        .. note:: Please refer to http://www.parallelpython.com/ for
            sources and information on Parallel Python software.
        """
        return orb.utils.parallel.close_pp_server(js)
        

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
        logging.info('looking for tuning parameter: {}'.format(
            full_parameter_name))
        if full_parameter_name in self.config:
            warnings.warn(
                'Tuning parameter {} changed to {} (default {})'.format(
                    full_parameter_name,
                    self.config[full_parameter_name],
                    default_value))
            return self.config[full_parameter_name]
        else:
            return default_value
                
    def save_sip(self, fits_path, hdr, overwrite=True):
        """Save SIP parameters from a header to a blanck FITS file.

        :param fits_path: Path to the FITS file
        :param hdr: header from which SIP parameters must be read
        :param overwrite: (Optional) Overwrite the FITS file.
        """    
        clean_hdr = self._clean_sip(hdr)
        data = np.empty((1,1))
        data.fill(np.nan)
        orb.utils.io.write_fits(
            fits_path, data, fits_header=clean_hdr, overwrite=overwrite)

    def load_sip(self, fits_path):
        """Return a astropy.wcs.WCS object from a FITS file containing
        SIP parameters.
    
        :param fits_path: Path to the FITS file    
        """
        hdr = orb.utils.io.read_fits(fits_path, return_hdu_only=True).header
        return pywcs.WCS(hdr)
                    
    def _get_quadrant_dims(self, quad_number, dimx, dimy, div_nb):
        """Return the indices of a quadrant along x and y axes.

        :param quad_number: Quadrant number

        :param dimx: X axis dimension.
          
        :param dimy: Y axis dimension.
        
        :param div_nb: Number of divisions along x and y axes. (e.g. if
          div_nb = 3, the number of quadrant is 9 ; if div_nb = 4, the
          number of quadrant is 16)
        """
        return orb.utils.image.get_quadrant_dims(quad_number, dimx, dimy, div_nb)
    
##################################################
#### CLASS ProgressBar ###########################
##################################################
class ProgressBar(object):
    """Display a simple progress bar in the terminal

    :param max_index: Index representing a 100% completed task.
    """

    REFRESH_COUNT = 3 # number of steps used to calculate a remaining time
    MAX_CARAC = 78 # Maximum number of characters in a line
    BAR_LENGTH = 10. # Length of the bar

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
        self._count = 0
        
    def _erase_line(self):
        """Erase the progress bar"""
        if not self._silent:
            sys.stdout.write("\r" + " " * self.MAX_CARAC)
            sys.stdout.flush()

    def _time_str_convert(self, sec):
        """Convert a number of seconds in a human readable string
        
        :param sec: Number of seconds to convert
        """
        if sec is None: return 'unknown'
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
            color = TextColor.CYAN
            self._count += 1
            for _icount in range(self.REFRESH_COUNT - 1):
                self._time_table[_icount] = self._time_table[_icount + 1]
                self._index_table[_icount] = self._index_table[_icount + 1]
            self._time_table[-1] = time.time()
            self._index_table[-1] = index
            if (self._count > self.REFRESH_COUNT):
                index_by_step = ((self._index_table[-1] - self._index_table[0])
                                 /float(self.REFRESH_COUNT - 1))
                if index_by_step > 0:
                    time_to_end = (((self._time_table[-1] - self._time_table[0])
                                    /float(self.REFRESH_COUNT - 1))
                                   * (self._max_index - index) / index_by_step)
                else: time_to_end = None
            else:
                time_to_end = None
            pos = (float(index) / self._max_index) * self.BAR_LENGTH
            line = ("\r [" + "="*int(math.floor(pos)) + 
                    " "*int(self.BAR_LENGTH - math.floor(pos)) + 
                    "] [%d%%] [" %(pos*100./self.BAR_LENGTH) + 
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
            warnings.warn("File key '%s' does not exist"%file_key)
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
            raise Exception(
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
            raise Exception('Bad file group. File group can be in %s, in %s or None'%(str(self.file_groups), str(self.file_group_indexes)))

        if file_key in self.index:
            return self[file_key]
        else:
            if err:
                raise Exception("File key '%s' does not exist"%file_key)
            else:
                warnings.warn("File key '%s' does not exist"%file_key)

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
            
        else: raise Exception(
            'Bad file group name. Must be in %s'%str(self.file_groups))

    def load_index(self):
        """Load index file and rebuild index of already located files"""
        self.index = dict()
        if os.path.exists(self._get_index_path()):
            f = orb.utils.io.open_file(self._get_index_path(), 'r')
            for iline in f:
                if len(iline) > 2:
                    iline = iline.split()
                    self.index[iline[0]] = iline[1]
            f.close()

    def update_index(self):
        """Update index files with data in the virtual index"""
        f = orb.utils.io.open_file(self._get_index_path(), 'w')
        for ikey in self.index:
            f.write('%s %s\n'%(ikey, self.index[ikey]))
        f.close()
        
        

#################################################
#### CLASS Lines ################################
#################################################
class Lines(Tools):
    """This class manages emission lines names and wavelengths.
    
    Spectral lines rest wavelength (excerpt, all recorded lines are in
    self.air_lines_nm)::
    
      ============ =======
        Em. Line     Air
      ============ =======
      [OII]3726    372.603
      [OII]3729    372.882
      Hepsilon     397.007
      Hdelta       410.176
      Hgamma       434.047
      [OIII]4363   436.321
      Hbeta        486.133
      [OIII]4959   495.892
      [OIII]5007   500.684
      [NII]6548    654.803
      Halpha       656.279
      [NII]6583    658.341
      [SII]6716    671.647
      [SII]6731    673.085

    .. note: Values were taken from NIST: https://www.nist.gov/PhysRefData/ASD/lines_form.html and https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H%20I&limits_type=0&unit=1&submit=Retrieve%20Data&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&show_av=2&tsb_value=0&A_out=0&intens_out=on&allowed_out=1&forbid_out=1&conf_out=on&term_out=on&enrg_out=on&J_out=on&level_id=001001.001.000059
      Ritz wavelength was used when more precise than observed wavelength

    """
    sky_lines_file_name = 'sky_lines.orb'
    """Name of the sky lines data file."""

    air_sky_lines_nm = None
    """Air sky lines wavelength"""

    
    air_lines_nm = {
        'H15': 371.19774,
        'H14': 372.19449,
        'H13': 373.43746,
        'H12': 375.01584,
        'H11': 377.06368,
        'H10': 379.79044,
        'H9': 383.53909,
        'H8': 388.90557,
        'Hepsilon':397.00788,
        'Hdelta':410.17415,
        'Hgamma':434.0471,
        'Hbeta':486.1333,
        'Halpha':656.2819,
        '[OII]3726':372.6032, 
        '[OII]3729':372.8815, 
        '[NeIII]3869':386.876, 
        '[OIII]4363':436.3209,
        '[OIII]4959':495.8911,
        '[OIII]5007':500.6843,
        'HeI5876':587.567, 
        '[OI]6300':630.0304, 
        '[SIII]6312':631.206, 
        '[NII]6548':654.805,
        '[NII]6583':658.345, 
        'HeI6678':667.815170,
        '[SII]6716':671.6440,
        '[SII]6731':673.0816, 
        'HeI7065':706.521530,
        '[ArIII]7136':713.579,
        '[OII]7320':731.992, 
        '[OII]7330':733.019, 
        '[ArIII]7751':775.111
    }

    other_names = {
        'Halpha': ['H3'],
        'Hbeta': ['H4'],
        'Hgamma': ['H5'],
        'Hdelta': ['H6'],
        'Hepsilon': ['H7'],
        '[OII]3726': ['[OII]3727'],
        '[NII]6583': ['[NII]6584'],
        '[SII]6716': ['[SII]6717'],
    }
    """Air emission lines wavelength"""

    air_lines_name = None
    
    def __init__(self, **kwargs):
        """Lines class constructor.

        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)

        # create corresponding inverted dicts
        self.air_lines_name = dict()
        for ikey in self.air_lines_nm.keys():
            self.air_lines_name[str(self.air_lines_nm[ikey])] = ikey

        for ikey in self.other_names:
            if ikey in self.air_lines_nm:
                for iname in self.other_names[ikey]:
                    self.air_lines_nm[iname] = float(self.air_lines_nm[ikey])
            else: raise ValueError('Bad key in self.other_names: {}'.format(ikey))
            
        self._read_sky_file()
        

    def _read_sky_file(self):
        """Return sky file (sky_lines.orb) as a dict.
        """
        sky_lines_file_path = self._get_orb_data_file_path(
            self.sky_lines_file_name)
        f = orb.utils.io.open_file(sky_lines_file_path, 'r')
        self.air_sky_lines_nm = dict()
        try:
            for line in f:
                if '#' not in line and len(line) > 2:
                    line = line.split()
                    self.air_sky_lines_nm[line[1]] = (float(line[1]) / 10., float(line[3]))
        except Exception as e:
            raise Exception('Error during parsing of {}: {}'.format(sky_lines_file_path, e))
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
        

    def get_line_nm(self, lines_name, round_ang=False):
        """Return the wavelength of a line or a list of lines

        :param lines_name: List of line names

        :param round_ang: (Optional) If True return the rounded
          wavelength of the line in angstrom (default False)
        """
        if isinstance(lines_name, str):
            lines_name = [lines_name]

        lines_nm = [self.air_lines_nm[line_name]
                    for line_name in lines_name]

        if len(lines_nm) == 1:
            lines_nm = lines_nm[0]
            
        if round_ang:
            return self.round_nm2ang(lines_nm)
        else:
            return lines_nm

    def get_line_cm1(self, lines_name, round_ang=False):
        """Return the wavenumber of a line or a list of lines

        :param lines_name: List of line names
        """
        return orb.utils.spectrum.nm2cm1(
            self.get_line_nm(lines_name))

    def get_line_name(self, lines):
        """Return the name of a line or a list of lines given their
        wavelength.

        :param lines: List of lines wavelength
        """
        if isinstance(lines, (float, int, np.float128)):
            lines = [lines]

        names = list()
        for iline in lines:
            if str(iline) in self.air_lines_name:
                names.append(self.air_lines_name[str(iline)])
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
#### CLASS ParamsFile ###########################
#################################################

class ParamsFile(Tools):
    """Manage correspondance between multiple dict containing the
    same parameters and a file on disk.

    Its behaviour is similar to :py:class:`astrometry.StarsParams`.
    """

    _params_list = None
    _keys = None
    _file_path = None

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

        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
        """
        Tools.__init__(self, **kwargs)
        
        self._params_list = list()
        if not reset and os.path.exists(file_path):
            self.f = orb.utils.io.open_file(file_path, 'r')
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
                        raise Exception(
                            'Wrong file format: {:s}'.format(file_path))
            self.f.close()
            self.f = orb.utils.io.open_file(file_path, 'a')

        else:
            self.f = orb.utils.io.open_file(file_path, 'w')
            self.f.write("## PARAMS FILE\n## created by {:s}\n".format(
                self.__class__.__name__))
            self.f.flush()
        self._file_path = file_path

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
            self._keys = list(params.keys())
            self._keys.sort()
            self.f.write('# KEYS')
            for ikey in self._keys:
                self.f.write(' {:s}'.format(ikey))
            self.f.write('\n')
        else:
            keys = list(params.keys())
            keys.sort()
            if keys == self._keys:
                self._params_list.append(params)
            else:
                raise Exception('parameters of the new entry are not the same as the old entries')
        
        for ikey in self._keys:
            self.f.write(' {}'.format(self._params_list[-1][ikey]))
        self.f.write('\n')
        self.f.flush()

    def get_data(self):
        return self._params_list
        
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
        return self.nm + orb.utils.spectrum.line_shift(
            self.velocity, self.nm, wavenumber=False)

    def get_cm1(self):
        """Return wavenumber of waves in cm-1 (taking velocity into account)"""
        cm1 = self.get_cm1_rest()
        return cm1 + orb.utils.spectrum.line_shift(
            self.velocity, cm1, wavenumber=True)

    def get_nm_rest(self):
        """"Return restframe wavelength of waves in nm"""
        return np.copy(self.nm)

    def get_cm1_rest(self):
        """Return restframe wavelength of waves in cm-1"""
        return orb.utils.spectrum.nm2cm1(self.nm)

#################################################
#### CLASS Data #################################
#################################################
class Data(object):
    """base class for data objects.

    data = array 
    + params (equiv to header, wcs params are contained in the params) 
    + axis (for 1d data and 3d data)
    + mask (1d for 1d, 2d for 2d and 3d)
    """
    needed_params = ()
    convert_params = {}

    def __init__(self, data, err=None, axis=None, params=None, mask=None, **kwargs):
        """Init method.

        :param data: A numpy.ndarray or a path to an
          hdf5 file. If an hdf5 file is loaded the values of the its
          axis, params, and mask can be changed by setting their
          respective keywords to something else than None. Note that
          file parameters are updated from the dictionary supplied
          with the params keyword and the kwargs.

        :param err: (Optional) Error on data. A 1d numpy.ndarray axis
          with the same size as vector (default None) .

        :param axis: (Optional) Data axis. A 1d numpy.ndarray axis
          with the same size as vector (default None).

        :param params: (Optional) A dict containing additional
          parameters.

        :param mask: (Optional) A numpy.ndarray with the same shape as
          the input data (for 3D data, a 2D array can be provided with
          shape (dimx, dimy))

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.

        """
        LIMIT_SIZE = 100
                
        # load from file
        if isinstance(data, str):    
            self.axis = None
            self.params = dict()
            self.mask = None
            self.err = None

            if params is not None:
                # params are transformed into kwargs
                kwargs.update(params)
        
            if 'fit' in os.path.splitext(data)[1] or '.fz' == os.path.splitext(data)[1]:
                self.data, _header = orb.utils.io.read_fits(data, return_header=True)
                self.params = dict(_header)
                if 'COMMENT' in self.params:
                    self.params.pop('COMMENT')

            elif 'hdf' in os.path.splitext(data)[1]:
                with orb.utils.io.open_hdf5(data, 'r') as hdffile:
                    _data_path = 'data'

                    # backward compatibility issue
                    if 'vector' in hdffile and not 'data' in hdffile:
                        _data_path = 'vector'

                    # load data
                    if (hdffile[_data_path].ndim > 2
                        and np.any(hdffile[_data_path].shape > LIMIT_SIZE)):
                        raise Exception('file too big to be opened this way')
                    else:
                        self.data = hdffile[_data_path][:]

                    # load params
                    for iparam in hdffile.attrs:
                        try:
                            ivalue = hdffile.attrs[iparam]
                            if isinstance(ivalue, bytes):
                                ivalue = ivalue.decode()
                            self.params[iparam] = ivalue
                        except TypeError as e:
                            logging.debug('error reading param from attributes {}: {}'.format(
                                iparam, e))

                    # load axis
                    if '/axis' in hdffile:
                        if axis is None:
                            self.axis = Axis(hdffile['/axis'][:])

                    # load err
                    if '/err' in hdffile:
                        if err is None:
                            self.err = hdffile['/err'][:]

                    # load mask
                    if '/mask' in hdffile:
                        if mask is None:
                            self.mask = hdffile['/mask'][:]
            else:
                raise ValueError('extension not recognized, must be fits or hdf5')

        # load from another instance
        #elif isinstance(data, self.__class__) or isinstance(self, data.__class__):
        elif all([hasattr(data, attr) for attr in ['data', 'err', 'params', 'axis', 'mask']]):
            if data.data.ndim < 3:
                _data = data.copy()
            else: _data = data

            self.data = _data.data

            # load error
            self.err = _data.err

            # load params
            self.params = _data.params

            # load axis
            self.axis = _data.axis

            # load mask
            self.mask = _data.mask

        # load from a bundle (created with to_bundle())
        elif isinstance(data, dict):
            if 'DataBundle' not in data or 'data' not in data or 'class' not in data:
                raise TypeError('if a dict is passed it must be a bundle created with to_bundle()')
            if self.__class__.__name__ != data['class']:
                warnings.warn('bundle was loaded with {} but its original class is {}'.format(
                    self.__class__.__name__, data['class']))
            self.data = data['data']
            self.params = data['params']
            self.err = data['err']
            self.mask = data['mask']
            self.axis = Axis(data['axis'])
        
        # load from np.ndarray
        else:
            self.axis = None
            self.params = dict()
            self.mask = None
            self.err = None

            if not isinstance(data, np.ndarray):
                raise TypeError('input data is a {} but must be a numpy.ndarray'.format(type(data)))
            data = np.squeeze(np.copy(data))
            if data.ndim > 3: raise TypeError('data dimension > 3 is not supported')
            self.data = data
            
        self.dimx = self.data.shape[0]
        if self.data.ndim > 1:
            self.dimy = self.data.shape[1]
            if self.data.ndim > 2:
                self.dimz = self.data.shape[2]
        self.shape = self.data.shape


        # reduce complex data to real data if imaginary part is null
        if np.iscomplexobj(self.data):
            if not np.any(np.iscomplex(self.data)):
                try:
                    self.data = self.data.real
                except AttributeError:
                    pass

        # load params
        if params is not None:
            self.params.update(params)
        self.params.update(kwargs)

        for iparam in self.convert_params:
            self.copy_param(iparam, self.convert_params[iparam])
            
        # check params
        if self.has_param('filter_file_path'):
            if not self.has_param('filter_name'):
                self.set_param('filter_name', self.params['filter_file_path'])
        
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise Exception('param {} must be set'.format(iparam))

        # load err
        if err is not None:
            if not isinstance(err, np.ndarray):
                raise TypeError('input err is a {} but must be a numpy.ndarray'.format(type(err)))

            if err.shape != self.data.shape:
                raise TypeError('err must have the same shape as data')
            self.err = err

        # load axis
        if axis is not None:
            axis = Axis(axis)
            if axis.dimx != self.dimx:
                raise TypeError('axis must have the same length as data first axis')
            self.axis = axis
            
        # load mask
        if mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError('input mask is a {} but must be a numpy.ndarray'.format(type(mask)))

            if mask.dtype != np.bool: raise TypeError('input mask must be of boolean type')
            
            if self.data.ndim < 3:
                if mask.shape != self.data.shape:
                    raise TypeError('mask has shape {} but must have the same shape as data: {}'.format(mask.shape, self.data.shape))

            else:
                if mask.shape != (self.data.shape[0:2]):
                    raise TypeError('mask has shape {} but must have shape {}'.format(mask.shape, self.data.shape[0:2]))
            self.mask = mask
               
        self.params = ROParams(self.params) # self.params must always be an ROParams instance
        
        
    def __getitem__(self, key):
        _data = self.data.__getitem__(key)
        if self.has_mask():
            if _data.size > 1:
                _data[np.nonzero(self.mask.__getitem__(key) == 0)] = np.nan
            else:
                if not self.mask.__getitem__(key): _data = np.nan
        return _data

    def copy_param(self, oldkey, newkey):
        """Copy a parameter with a given key to a parameter with another
           key. Do it only if the new key is not already set
        """
        if self.has_param(oldkey):
            if not self.has_param(newkey):
                self.params[newkey] = self.params[oldkey]
                   
    def has_params(self):
        """Check the presence of observation parameters"""
        if self.params is None:
            raise TypeError('params should not be None')
        elif len(self.params) == 0:
            return False
        else: return True

    def assert_params(self):
        """Assert the presence of parameters"""
        if not self.has_params():
            raise Exception(
                'Parameters not supplied, please give: {} at init'.format(
                    self.needed_params))
    
    def get_param(self, key):
        """Get class parameter

        :param key: parameter key
        """
        return self.params[key]

    def set_param(self, key, value):
        """Set class parameter

        :param key: parameter key
        """
        self.params[key] = value

    def has_param(self, key):
        """Test the presence of a parameter
        """
        return (key in self.params)
        
    def update_params(self, params):
        """Update params with a dictionary or an astropy.io.fits.Header

        :param params: A dict or an astropy.io.fits.Header instance.
        """
        if isinstance(params, pyfits.Header):
            params = dict(params)
            if self.data.ndim >= 3:
                params['CTYPE3'] = 'WAVE-SIP' # avoid a warning for
                                              # inconsistency

        elif not isinstance(params, dict):
            raise TypeError('params must be a dict or an astropy.io.fits.Header instance')
            
        for ipar in params:
            self.set_param(ipar, params[ipar])
        self.assert_params()

    def get_header(self):
        """Return params as an astropy.io.fits.Header instance
        """
        if not self.has_params(): return None
        
        warnings.simplefilter('ignore', category=VerifyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)

        header = orb.utils.io.dict2header(dict(self.params))
        if self.data.ndim >= 1:
            header['NAXIS1'] = self.dimx

        if self.data.ndim >= 2:
            header['NAXIS2'] = self.dimy
        
        if self.data.ndim >= 3:
            header['NAXIS3'] = self.dimz
        
        if 'CTYPE1' in header:
            header['CTYPE1'] = 'RA---TAN-SIP'

        if 'CTYPE2' in header:
            header['CTYPE2'] = 'DEC--TAN-SIP'

        if self.data.ndim >= 3:
            header['CTYPE3'] = 'WAVE-SIP' # avoid a warning for
                                          # inconsistency
        return header

    def get_wcs(self):
        naxis = None
        if self.data.ndim == 3: naxis = 2
        
        warnings.simplefilter('ignore', category=VerifyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        return copy.copy(pywcs.WCS(self.get_header(), relax=True, naxis=naxis))

    def get_wcs_header(self):
        hdr = self.get_wcs().to_header(relax=True)
        for i in range(hdr['WCSAXES']):
            hdr['NAXIS{}'.format(i+1)] = self.shape[i]
        return hdr

    def set_header(self, header):
        """update params from an astropy.io.fits.Header instance.

        :param header: An astropy.io.fits.Header instance.
        """
        self.update_params(header)

    def has_err(self):
        if self.err is None: return False
        return True
        
    def assert_err(self):
        """Assert the presence of an error"""
        if not self.has_err(): raise Exception('No error supplied')

    def get_err(self):
        """Return a copy of self.err"""
        return np.copy(self.err)
        
    def has_axis(self):
        if self.axis is None: return False
        return True
        
    def assert_axis(self):
        """Assert the presence of an axis"""
        if not self.has_axis(): raise Exception('No axis supplied')

    def get_axis(self):
        """Return a copy of self.axis"""
        return np.copy(self.axis.data)

    def has_mask(self):
        if self.mask is None: return False
        return True
        
    def assert_mask(self):
        """Assert the presence of a mask"""
        if not self.has_mask(): raise Exception('No mask supplied')

    def set_mask(self, data):
        """Set mask. 

        A mask must have the shape of the data but for 3d data which
        has a 2d mask (self.dimx, self.dimy). A Zero indicates a pixel
        which should be masked (Nans are returned for this pixel).

        :param data: mask. Must be a boolean array
        """
        if self.data.ndim >= 2:
            orb.utils.validate.is_2darray(data, object_name='data')
            if data.shape != (self.dimx, self.dimy):
                raise TypeError('data must have shape ({}, {}) but has shape {}'.format(
                    self.dimx, self.dimy, data.shape))

        else:
            orb.utils.validata.is_1darray(data, object_name='data')
            if data.shape != self.dimx:
                raise TypeError('data must have shape ({}) but has shape'.format(
                    self.dimx, data.shape))
                          
        if data.dtype != np.bool:
            raise TypeError('data should be boolean')
        
        _data = np.copy(data).astype(float)
        _data[np.nonzero(_data == 0)] = np.nan
        self.mask = _data
        
    def get_mask(self):
        """Return a copy of self.mask"""
        _mask = np.copy(self.mask)
        _mask[np.isnan(_mask)] = 0
        return _mask.astype(bool)

    def get_gvar(self):
        """Return data and err as a gvar.GVar instance"""
        if self.has_err():
            return gvar.gvar(self.data, self.err)
        else:
            return gvar.gvar(self.data)
                    
    def copy(self, data=None, **kwargs):
        """Return a copy of the instance

        :param data: (Optional) can be used to change data

        :param kwargs: Addition kwargs (useful to copy child classes
          with more kwargs at init)
        """
        if self.has_params():
            _params = self.params.convert()
        else:
            _params = None

        if self.has_axis():
            _axis = np.copy(self.axis.data)
        else:
            _axis = None

        if self.has_err():
            _err = np.copy(self.err)
        else:
            _err = None

        if self.has_mask():
            _mask = np.copy(self.mask)
        else:
            _mask = None

        if data is None:
            data = np.copy(self.data)

        return self.__class__(
            data,
            err=_err,
            axis=_axis,
            params=_params,
            mask=_mask,
            **kwargs)
        
    def writeto(self, path):
        """Write data to an hdf file

        :param path: hdf file path.
        """
        if np.iscomplexobj(self.data):
            _data = self.data.astype(complex)
        else:
            _data = self.data.astype(float)

        if os.path.exists(path):
            os.remove(path)
        with orb.utils.io.open_hdf5(path, 'w') as hdffile:
            if self.has_params():
                for iparam in self.params:
                    try:
                        hdffile.attrs[iparam] = self.params[iparam]
                    except TypeError:
                        warnings.warn('{} ({}) could not be written'.format(
                            iparam, type(self.params[iparam])))

            hdffile.create_dataset(
                '/data',
                data=_data)

            if self.has_axis():
                hdffile.create_dataset(
                    '/axis',
                    data=self.axis.data.astype(float))

            if self.has_mask():
                hdffile.create_dataset(
                    '/mask',
                    data=self.mask)

            if self.has_err():
                hdffile.create_dataset(
                    '/err',
                    data=self.err)


    def to_fits(self, path):
        """write data to a FITS file. 

        Note that most of the information will be lost in the
        process. The only output guaranteed format is hdf5 (use
        writeto() method)

        :param path: Path to the FITS file

        """
        orb.utils.io.write_fits(path, self.data, fits_header=self.get_header())

    def to_bundle(self):
        """Return a bundle of picleable objects that can be passed to a
        parallelized process and recreate the Data object.
        """
        bundle = dict()
        bundle['DataBundle'] = True
        bundle['class'] = self.__class__.__name__
        bundle['data'] = np.copy(self.data)
        if self.has_params():
            bundle['params'] = self.params.convert()
        else:
            bundle['params'] = None
        if self.has_err():
            bundle['err'] = self.get_err()
        else:
            bundle['err'] = None
        if self.has_axis():
            bundle['axis'] = self.get_axis()
        else:
            bundle['axis'] = None
        if self.has_mask():
            bundle['mask'] = self.get_mask()
        else:
            bundle['mask'] = None
        
        return bundle

#################################################
#### CLASS Vector1d #############################
#################################################
class Vector1d(Data):
    """Basic 1d vector management class.

    Vector can have a projection axis.
    """
    
    def __init__(self, *args, **kwargs):

        Data.__init__(self, *args, **kwargs)

        # checking
        if self.data.ndim != 1:
            raise TypeError('input vector has {} dims but must have only one dimension'.format(self.data.ndim))

    def reverse(self):
        """Reverse data. Do not reverse the axis.
        """
        self.data = self.data[::-1]
        if self.has_err():
            self.err = self.err[::-1]

    def project(self, new_axis, returned_class=None, quality=10, timing=False):
        """Project vector on a new axis

        :param new_axis: Axis. Must be an orb.core.Axis instance.

        :param returned_class: (Optional) If not None, set the
          returned class. Must be a subclass of Vector1d.

        :param quality: an integer from 2 to infinity which gives the
          zero padding factor before interpolation. The more zero
          padding, the better will be the interpolation, but the
          slower too.

        :return: A new Spectrum instance

        .. warning:: Though much (much!) faster than pure resampling, this can
          be a little less precise for complex data. For non complex
          data, its nothing more than a linear interpolation.
        """
        if timing:
            import time
            times = list()
            times.append(time.time()) ###

        self.assert_axis()

        if returned_class is None:
            returned_class = self.__class__
        else:
            if not issubclass(returned_class, Vector1d):
                raise TypeError('Returned class must be a subclass of Vector1d')
            
        if not isinstance(new_axis, Axis):
            try:
                new_axis = Axis(new_axis)
            except Exception:
                raise TypeError('axis must be compatible with an orb.core.Axis instance: {}'.format(e))
            
        if len(self.axis.data) == len(new_axis.data):
            if np.all(np.isclose(self.axis.data - new_axis.data, 0)):
                if timing:
                    return self.copy(), None
                else:
                    return self.copy()
                
        if timing: times.append(time.time()) ####
        if np.any(np.iscomplex(self.data)):
            quality = int(quality)
            if quality < 2: raise ValueError('quality must be an integer > 2')
            interf_complex = scipy.fftpack.ifft(self.data)
            best_n = orb.utils.fft.next_power_of_two(self.dimx * quality)
            zp_interf = np.zeros(best_n, dtype=complex)
            center = interf_complex.shape[0] // 2
            zp_interf[:center] = interf_complex[:center]
            zp_interf[
                -center-int(interf_complex.shape[0]&1):] = interf_complex[
                -center-int(interf_complex.shape[0]&1):]

            if timing: times.append(time.time()) ####
            zp_spec = scipy.fftpack.fft(zp_interf)
            ax_ratio = float(self.axis.data.size) / float(zp_spec.size)
            zp_axis = (np.arange(zp_spec.size)
                       * (self.axis.data[1] - self.axis.data[0]) * ax_ratio
                       + self.axis.data[0])
            if timing: times.append(time.time()) ####
            f = scipy.interpolate.interp1d(zp_axis,
                                           zp_spec,
                                           bounds_error=False)
            
        else:
            logging.debug('data is not complex and is interpolated the bad way')
            if timing: times.append(time.time()) ####
            f = scipy.interpolate.interp1d(self.axis.data.astype(np.float128),
                                           self.data.real.astype(np.float128),
                                           bounds_error=False)
            # (added to get the same number of timings as if data is
            # complex)
            if timing: times.append(time.time()) 
            
        if timing: times.append(time.time()) ####
        data  = f(new_axis.data)
        if timing: times.append(time.time()) ####
            
        if self.has_err():
            new_err = scipy.interpolate.interp1d(self.axis.data.astype(np.float128),
                                                 self.err.astype(np.float128),
                                                 bounds_error=False)(new_axis.data)
        else:
            new_err = None

        ret = returned_class(
            data, err=new_err, axis=new_axis.data, params=self.params)

        if not timing:
            return ret
        else:
            return ret, list([times[-1] - times[0]]) + list(np.diff(times))        

    def crop(self, xmin, xmax, returned_class=None):
        """Crop data. 
        :param xmin: xmin
        :param xmax: xmax
        """
        return self.project(Axis(self.axis[xmin:xmax]),
                            returned_class=returned_class)

        
    def math(self, opname, arg=None):
        """Do math operations with another vector instance.

        :param opname: math operation, must be a numpy.ufuncs.

        :param arg: If None, no argument is supplied. Else, can be a
          float or a Vector1d instance.
        """
        # check if complex
        iscomplex = False
        if np.any(np.iscomplex(self.data)):
            iscomplex = True
        if isinstance(arg, Vector1d):
            if np.any(np.iscomplex(arg.data)) or iscomplex:
                iscomplex = True
                arg_real = arg.copy()
                arg_real.data = arg.data.real.astype(float)
                arg_imag = arg.copy()
                arg_imag.data = arg.data.imag.astype(float)
        elif np.size(arg) <= 1:
            if np.iscomplex(arg) or iscomplex:
                iscomplex = True
                arg_real = float(arg.real)
                arg_imag = float(arg.imag)
        else:
            raise TypeError('arg must be a float or a Vector1d instance')

        if iscomplex:
            data_real = self.copy()
            data_imag = self.copy()
            data_real.data = data_real.data.real.astype(float)
            data_imag.data = data_imag.data.imag.astype(float)

            data_real = data_real.math(opname, arg=arg_real)
            data_imag = data_imag.math(opname, arg=arg_imag)
            data_real.data = data_real.data.astype(complex)
            data_real.data.imag = data_imag.data
            return data_real

        self.assert_axis()

        out = self.copy()
        
        if not hasattr(np, opname):
            raise AttributeError('unknown operation: {}'.format(e))

        # project arg on self axis
        if arg is not None:
            if isinstance(arg, Vector1d):
                project = True
                if len(out.axis.data) == len(arg.axis.data):
                    if np.all(np.isclose(out.axis.data - arg.axis.data, 0)):
                        project = False
                if project:
                    arg = arg.project(out.axis)

            elif np.size(arg) != 1:
                raise TypeError('arg must be a float or a Vector1d instance')

        # transform self in gvar
        if out.has_err():
            _out = gvar.gvar(out.data.real, out.err.real)
        else:
            _out = out.data.real

        # transform arg in gvar
        if arg is not None:
            if isinstance(arg, Vector1d):
                if arg.has_err():
                    _arg = gvar.gvar(arg.data.real, arg.err.real)
                else:
                    _arg = np.copy(arg.data.real)
            else: _arg = arg

        # do the math
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if arg is None:
                _out = getattr(np, opname)(_out)
            else:            
                _out = getattr(np, opname)(_out, _arg)
                del _arg

        # transform gvar to data and err
        if isinstance(_out[0], gvar._gvarcore.GVar):
            out.data = gvar.mean(_out)
            out.err = gvar.sdev(_out)
            
        else:
            out.data = _out
            out.err = None

        del _out
        return out

    def add(self, vector):
        """Add another vector. Note that, if the axis differs, only the
        common part is kept.

        :param vector: Must be a Vector1d instance.
        """
        return self.math('add', vector)

    def subtract(self, vector):
        """Subtract another vector. Note that, if the axis differs, only the
        common part is kept.

        :param vector: Must be a Cm1Vector1d instance.
        """
        return self.math('subtract', vector)

    def multiply(self, vector):
        """Multiply by another vector. Note that, if the axis differs, only the
        common part is kept.

        :param vector: Must be a Cm1Vector1d instance.
        """
        return self.math('multiply', vector)

    def sum(self):
        """Sum of the data"""
        return np.sum(self.data)


    def plot(self, plot_real=True, **kwargs):
        """Plot vector

        :param plot_real: If True, plot only the real part, if False
          plot only the imaginary part, if both, plot the real and
          imaginary part. Plot only the real part by default.

        :param kwargs: All keyword arguments accepted by
          matplotlib.plot()
        """
        if not np.any(np.iscomplex(self.data)):
            pl.plot(self.axis.data, self.data, **kwargs)
        else:
            if plot_real == True or plot_real == 'both':
                if 'label' not in kwargs:
                    kwargs['label'] = 'real part'
                pl.plot(self.axis.data, self.data.real, **kwargs)
                
            if plot_real == False or plot_real == 'both':
                if 'label' not in kwargs:
                    kwargs['label'] = 'imaginary part'
                pl.plot(self.axis.data, self.data.imag, **kwargs)

#################################################
#### CLASS Axis #################################
#################################################
class Axis(Vector1d):
    """Axis class"""

    def __init__(self, data, axis=None, params=None, mask=None, **kwargs):
        """Init class with an axis vector

        :param data: 1d np.ndarray.
        """
        if axis is not None: raise ValueError('axis must be set to None')
        if mask is not None: raise ValueError('mask must be set to None')
        
        Vector1d.__init__(self, data, **kwargs)

        # check that axis is regularly sampled
        diff = np.diff(self.data)
        if np.any(~np.isclose(diff - diff[0], 0.)):
            raise Exception('axis must be regularly sampled')
        if self.data[0] > self.data[-1]:
            raise Exception('axis must be naturally ordered')

        self.axis_step = diff[0]

    def __call__(self, pos):
        """return the position in channels from an input in axis unit

        :param pos: Postion in the axis in the axis unit

        :return: Position in index
        """
        pos_index = (pos - self.data[0]) / float(self.axis_step)
        if np.any(pos_index < 0) or np.any(pos_index >= self.dimx):
            warnings.warn('requested position is off axis')
        return pos_index

    def convert(self, pos):
        """convert the position in channel to a a value in axis unit

        :param pos: Position in channel

        :return: Value in axis unit
        """
        return self.data[0] + float(self.axis_step) * pos
            

    
#################################################
#### CLASS Cm1Vector1d ##########################
#################################################
class Cm1Vector1d(Vector1d):
    """1d vector class for data projected on a cm-1 axis (e.g. complex
    spectrum, phase)

    """
    needed_params = ('filter_name', )
    obs_params = ('step', 'order', 'calib_coeff')
   
    def __init__(self, spectrum, axis=None, params=None, **kwargs):
        """Init method.

        :param spectrum: A 1d numpy.ndarray vector or a path to an
          hdf5 cm1 vector file (note that axis must be set to None in
          this case).

        :param axis: (Optional) A 1d numpy.ndarray axis. Must be given
          if observation paramters are not provided and if spectrum is
          a pure np.ndarray. If a file is loaded, i.e. spectrum is a
          path to an hdf5 file, it must be set to None (default None).
        
        :param params: (Optional) A dict containing observation
          parameters (default None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.

        """
        Vector1d.__init__(self, spectrum, axis=axis, params=params, **kwargs)

        if self.has_params():
            if len(set(self.obs_params).intersection(self.params)) == len(self.obs_params):
                check_axis = orb.utils.spectrum.create_cm1_axis(
                    self.dimx, self.params.step, self.params.order, corr=self.params.calib_coeff)
                if self.axis is None:
                    self.axis = Axis(check_axis)
                else:
                    if np.any(check_axis != self.axis.data):
                        logging.debug('provided axis is inconsistent with the given parameters')
            #else:
                #raise StandardError('{} must all be provided'.format(self.obs_params))
            
        if self.axis is None: raise Exception('an axis must be provided or the observation parameters ({}) must be provided'.format(self.obs_params))
            
    def get_filter_bandpass_cm1(self):
        """Return filter bandpass in cm-1"""
        if 'filter_cm1_min' not in self.params or 'filter_cm1_max' not in self.params:
            
            cm1_min, cm1_max = FilterFile(self.params.filter_name).get_filter_bandpass_cm1()
            logging.debug('Uneffective call to get filter bandpass. Please provide filter_cm1_min and filter_cm1_max in the parameters.')
            self.set_param('filter_cm1_min', cm1_min)
            self.set_param('filter_cm1_max', cm1_max)
            
        return self.params.filter_cm1_min, self.params.filter_cm1_max

    def get_filter_bandpass_pix(self, border_ratio=0.):
        """Return filter bandpass in channels

        :param border_ratio: (Optional) Relative portion of filter
          border removed (can be a negative float to get a bandpass
          larger than the filter, default 0.)
        
        :return: (min, max)

        """
        if not -0.2 <= border_ratio <= 0.2:
            raise ValueError('border ratio must be between -0.2 and 0.2')

        zmin = int(self.axis(self.get_filter_bandpass_cm1()[0]))
        zmax = int(self.axis(self.get_filter_bandpass_cm1()[1]))
        if border_ratio != 0:
            border = int((zmax - zmin) * border_ratio)
            zmin += border
            zmax -= border
        
        return zmin, zmax

    def mean_in_filter(self):
        ff = FilterFile(self.params.filter_name)
        ftrans = ff.get_transmission(self.dimx)
        return np.nansum(self.multiply(ftrans).data) / ftrans.sum()

    def velocity_shift(self, velocity):
        """Return a vector with its axis shifted by a given velocity.

        :param velocity: Velocity in km/s
        """
        new_axis = self.axis.copy()
        new_axis.data -= orb.utils.spectrum.line_shift(
            velocity, new_axis.data, wavenumber=True)
        spec = self.project(new_axis)
        spec.axis = self.axis
        return spec
        
    
#################################################
#### CLASS FilterFile ###########################
#################################################
class FilterFile(Vector1d):
    """Manage filter files"""

    needed_params = ('step', 'order', 'phase_fit_order', 'modulation_efficiency',
                     'bandpass_min_nm', 'bandpass_max_nm', 'instrument')
    
    def __init__(self, filter_name, axis=None, params=None, **kwargs):
        """Initialize FilterFile class.

        :param filter_name: Name of the filter or path to the filter file.
        """
        self.tools = Tools()

        if filter_name in [None, 'None']: filter_name = 'FULL'
        
        if os.path.exists(filter_name):
            self.basic_path = filter_name
            self.filter_name = filter_name
        else:
            self.filter_name = filter_name
            self.basic_path = self.tools._get_filter_file_path(filter_name)

        if not os.path.exists(self.basic_path):
            raise ValueError('filter_name is not a valid filter name and is not a valid filter file path')
        Vector1d.__init__(self, self.basic_path, axis=None, params=params, **kwargs)

        # reload self.tools with new params
        self.tools = Tools(instrument=self.params.instrument) 


    def read_filter_file(self, return_spline=False):
        """Return transmission, axis and bandpass

        :param return_spline: If True a cubic spline
          (scipy.interpolate.UnivariateSpline instance) is returned
          instead of a tuple (filter_nm, filter_trans, filter_min, filter_max)
        """
        if not return_spline:
            return (self.axis.data, self.data,
                    self.params.bandpass_min_pix,
                    self.params.bandpass_max_pix)
        else:
            return scipy.interpolate.UnivariateSpline(
                self.axis.data, self.data, k=3, s=0, ext=0)


    def project(self, new_axis):
        """Project vector on a new axis

        :param new_axis: Axis. Must be an orb.core.Axis instance.
        """
        return Vector1d.project(self, new_axis, returned_class=Vector1d)

    def get_transmission(self, step_nb, corr=None):
        """Return transmission in the filter bandpass
        :param step_nb: number of steps

        :param corr: calibration coeff (at center if None)
        """
        if corr is None:
            corr = orb.utils.spectrum.theta2corr(
                self.tools.config['OFF_AXIS_ANGLE_CENTER'])
        cm1_axis = Axis(orb.utils.spectrum.create_cm1_axis(
            step_nb, self.params.step, self.params.order,
            corr=corr))
    
        return self.project(cm1_axis)
            
    def get_modulation_efficiency(self):
        """Return modulation efficiency."""
        return self.params.modulation_efficiency

    def get_observation_params(self):
        """Return observation params as tuple (step, order)."""
        return self.params.step, self.params.order

    def get_phase_fit_order(self):
        """Return phase fit order."""
        return self.params.phase_fit_order

    def get_filter_bandpass(self):
        """Return filter bandpass in nm"""
        return self.params.bandpass_min_nm, self.params.bandpass_max_nm

    def get_filter_bandpass_cm1(self):
        """Return filter bandpass in cm-1"""
        return orb.utils.spectrum.nm2cm1(self.get_filter_bandpass())[::-1]

    def get_mean_cm1(self):
        """Return mean wavenumber """
        filter_trans = self.get_transmission(1000)
        return np.sum(filter_trans.axis.data * filter_trans.data) / np.sum(filter_trans.data)

    def get_mean_nm(self):
        """Return mean wavelength"""
        return orb.utils.spectrum.cm12nm(self.get_mean_cm1())
        
    def get_sky_lines(self, step_nb):
        """Return the sky lines in a given filter
        """
        corr = orb.utils.spectrum.theta2corr(
            self.tools.config['OFF_AXIS_ANGLE_CENTER'])
        axis = orb.utils.spectrum.create_cm1_axis(
            step_nb, self.params.step, self.params.order,
            corr=corr)

        _delta_nm = orb.utils.spectrum.fwhm_cm12nm(
            axis[1] - axis[0],
            (np.min(axis) + np.max(axis)) / 2.)

        _nm_min, _nm_max = self.get_filter_bandpass()
        
        # we add 5% to the computed size of the filter
        _nm_range = _nm_max - _nm_min
        _nm_min -= _nm_range * 0.05
        _nm_max += _nm_range * 0.05

        _lines_nm = Lines().get_sky_lines(
            _nm_min, _nm_max, _delta_nm)

        return orb.utils.spectrum.nm2cm1(_lines_nm)



    
#################################################
#### CLASS WCSData ##############################
#################################################
class WCSData(Data, Tools):
    """Add WCS functionalities to a Data instance.
    """
    default_params = {'camera':1,
                      'target_ra':0.,
                      'target_dec':0.,
                      'target_x':0,
                      'target_y':0}

    convert_params = {'AIRMASS':'airmass',
                      'EXPTIME':'exposure_time',
                      'FILTER':'filter_name',
                      'INSTRUME':'instrument',
                      'camera_number':'camera',
                      'CAMERA': 'camera',
                      'BINNING': 'binning'}

    def __init__(self, data, instrument=None, config=None,
                 data_prefix="./", sip=None, **kwargs):

        if isinstance(data, str):
            data_path = str(data)
        else:
            data_path = None
            
        # try to read instrument parameter from file
        if instrument is None:
            if data_path is not None:
                if os.path.exists(data_path):
                    instrument = orb.utils.misc.read_instrument_value_from_file(data_path)
        
        if instrument is None: # important even if it seems duplicated ;)
            if 'params' in kwargs:
                if 'instrument' in kwargs['params']:
                    instrument = kwargs['params']['instrument']

        Tools.__init__(self, instrument=instrument,
                       data_prefix=data_prefix,
                       config=config)
        
        Data.__init__(self, data, **kwargs) # note that this init may change the value of data

        # try to load wcs from fits keywords if a FITS file
        if data_path is not None:
            if 'fit' in os.path.splitext(data_path)[1]:
                self.set_wcs(data_path)
                # reingest kwargs in this case, because wcs params were forced
                self.params.update(kwargs)
                
                    
        # load dxdymaps
        self.dxdymaps = None
        if data_path is not None:
            if 'hdf' in data_path:
                with orb.utils.io.open_hdf5(data_path, 'r') as hdffile:
                    if '/dxmap' in hdffile and '/dymap' in hdffile:
                        self.dxdymaps= (hdffile['/dxmap'][:], hdffile['/dymap'][:])

        # checking
        if self.data.ndim < 2:
            raise TypeError('A dataset must have at least 2 dimensions to support WCS')

        for iparam in ['camera']:
            if iparam not in self.params:
                self.params.reset(iparam, self.default_params[iparam])
                logging.debug('{} set to default value: {}'.format(
                    iparam, self.default_params[iparam]))
                
        # check params
        self.params.reset('instrument', self.instrument)
        if self.instrument is None:
            warnings.warn('Instrument set to None: some parameters will be automatically defined')
            self.config['CAM1_DETECTOR_SIZE_X'] = self.dimx
            self.config['CAM1_DETECTOR_SIZE_Y'] = self.dimy
            if not self.has_param('delta_x'): # wcs not loaded properly
                raise Exception('instrument set to None, wcs not loaded properly. Please set a correct WCS to your file or choose one of the available instrument configurations')
            self.config['FIELD_OF_VIEW_1'] = (self.params.delta_x * 60.
                                              * max(self.config.CAM1_DETECTOR_SIZE_X,
                                                    self.config.CAM1_DETECTOR_SIZE_Y))
        
        # convert camera param
        self.params.reset(
            'camera', orb.utils.misc.convert_camera_parameter(
                self.params.camera))

        # compute binning
        if self.is_cam1(): cam = 'CAM1'
        else: cam = 'CAM2'

        if 'cropped_bbox' in self.params: cropped = True
        else: cropped = False
        
        if self.is_cam1():
            if self.has_param('bin_cam_1'):
                self.params.reset('binning', self.params.bin_cam_1)
        if self.is_cam2():
            if self.has_param('bin_cam_2'):
                self.params.reset('binning', self.params.bin_cam_2)
            
        if not self.has_param('binning'):
            if cropped:
                warnings.warn('data is cropped. computed binning might be inconsistent')
            detector_shape = [self.config[cam + '_DETECTOR_SIZE_X'],
                              self.config[cam + '_DETECTOR_SIZE_Y']]

            binning = orb.utils.image.compute_binning(
                (self.dimx, self.dimy), detector_shape)

            if binning[0] != binning[1]:
                raise Exception('Images with different binning along X and Y axis are not handled by ORBS')
            self.set_param('binning', binning[0])

            logging.debug('Computed binning of camera {}: {}x{}'.format(
                self.params.camera, self.params.binning, self.params.binning))

            
        if self.dimx != self.config[cam + '_DETECTOR_SIZE_X'] // self.params.binning:
            warnings.warn('image might be cropped, target_x, target_y and other parameters might be wrong')

        # load wcs parameters
        if 'target_x' not in self.params:
            target_x = float(self.dimx / 2.)
        else: target_x = self.params.target_x
        if 'target_y' not in self.params:
            target_y = float(self.dimy / 2.)
        else: target_y = self.params.target_y

        if self.is_cam2():
            coeffs = self.get_initial_alignment_parameters()
            
            logging.debug('target_x, target_y initially at {}, {}'.format(target_x, target_y))
            target_x, target_y = orb.utils.astrometry.transform_star_position_A_to_B(
                [[self.params.target_x, self.params.target_y]],
                [coeffs.dx, coeffs.dy, coeffs.dr, 0., 0.],
                [coeffs.rc[0], coeffs.rc[1]],
                [coeffs.zoom, coeffs.zoom])
            logging.debug('target_x, target_y recomputed to {}, {}'.format(target_x, target_y))

        self.params.reset('target_x', target_x)
        self.params.reset('target_y', target_y)

        if 'target_ra' in self.params:
            if self.params.target_ra == self.default_params['target_ra']:
                del self.params['target_ra']

        if 'target_dec' in self.params:
            if self.params.target_dec == self.default_params['target_dec']:
                del self.params['target_dec']
        
        
        if 'target_ra' not in self.params:
            if 'TARGETR' in self.params:
                self.params['target_ra'] = orb.utils.astrometry.ra2deg(
                    self.params['TARGETR'].split(':'))
            else:
                self.params['target_ra'] = self.default_params['target_ra']

        if 'target_dec' not in self.params:
            if 'TARGETD' in self.params:
                self.params['target_dec'] = orb.utils.astrometry.dec2deg(
                    self.params['TARGETD'].split(':'))
            else:
                self.params['target_dec'] = self.default_params['target_dec']

        if 'target_ra' in self.params:
            if not isinstance(self.params.target_ra, float):
                raise TypeError('target_ra must be a float')
        if 'target_dec' in self.params:
            if not isinstance(self.params.target_dec, float):
                raise TypeError('target_dec must be a float')            
          
        if 'data_prefix' not in kwargs:
            kwargs['data_prefix'] = self._data_prefix
            
        if not self.has_param('wcs_rotation'):
            if self.is_cam1():
                self.params['wcs_rotation'] = float(self.config.WCS_ROTATION)
            else:
                self.params['wcs_rotation'] = (float(self.config.WCS_ROTATION)
                                               - float(self.config.INIT_ANGLE))
            logging.debug('wcs_rotation computed from config parameters: {}'.format(
                self.params.wcs_rotation))

        # define platescale
        if 'delta_x' not in self.params:
            self.params['delta_x'] = self.get_scale() / 3600.
            
        if 'delta_y' not in self.params:
            self.params['delta_y'] = self.get_scale() / 3600.
                
        # check if all needed parameters are present
        for iparam in self.default_params:
            if iparam not in self.params:
                raise Exception('param {} must be set'.format(iparam))

        # load sip
        if sip is not None:
            if not isinstance(sip, pywcs.WCS):
                raise Exception('sip must be an astropy.wcs.WCS instance')
        # else:
        #     if self.get_wcs().sip is None:
        #         sip = self.load_sip(self._get_sip_file_path(self.params.camera))
        #         logging.debug('SIP Loaded from{}\n{}'.format(
        #             self._get_sip_file_path(self.params.camera), sip))
        #     else:
        #         sip = self.get_wcs()
        #         logging.debug('SIP already defined\n{}'.format(sip))

        # reset wcs
        wcs = orb.utils.astrometry.create_wcs(
            self.params.target_x, self.params.target_y,
            self.params.delta_x, self.params.delta_y,
            self.params.target_ra, self.params.target_dec,
            self.params.wcs_rotation, sip=sip)

        self.set_wcs(wcs)

        logging.debug(self.get_wcs())
        self.validate_wcs()
        
    def has_dxdymaps(self):
        """Return True is self.dxmap and self.dymap exist"""
        if not hasattr(self, 'dxdymaps'): return False
        if self.dxdymaps is None: return False
        return True

    def assert_dxdymaps(self):
        """Raise an exception if dxdymaps are not loaded"""
        if not self.has_dxdymaps():
            raise Exception('no dxdymaps loaded')

    def set_dxdymaps(self, dxmap, dymap):
        """Set dxmap and dymap. Must have the same shape as the image shape.

        :param dxmap: Path to a dxmap or a numpy.ndarray
        :param dymap: Path to a dymap or a numpy.ndarray
        """
        if isinstance(dxmap, str):
            dxmap = orb.utils.io.read_fits(dxmap)
        if isinstance(dymap, str):
            dymap = orb.utils.io.read_fits(dymap)
        if dxmap.shape != self.shape:
            raise TypeError('dxmap must have same shape as image')
        if dymap.shape != self.shape:
            raise TypeError('dymap must have same shape as image')
        self.dxdymaps = (dxmap, dymap)

        
    def is_cam1(self):
        """Return true is image comes from camera 1 or is a merged frame
        """
        if self.params.camera not in [0, 1, 2]: raise ValueError('camera must be 0, 1 or 2')
        if self.params.camera == 1 or self.params.camera == 0:
            return True
        return False

    def is_cam2(self):
        """Return true is image comes from camera 2
        """
        if self.params.camera not in [0, 1, 2]: raise ValueError('camera must be 0, 1 or 2')
        if self.params.camera == 2:
            return True
        return False
    
    def set_wcs(self, wcs):
        """Set WCS from a WCS instance or a FITS image

        :param wcs: Must be an astropy.wcs.WCS instance or a path to a FITS image
        """
        if isinstance(wcs, str):
            warnings.simplefilter('ignore', category=VerifyWarning)
            warnings.simplefilter('ignore', category=AstropyUserWarning)
            wcs = pywcs.WCS(
                orb.utils.io.read_fits(wcs, return_hdu_only=True).header,
                naxis=2, relax=True)
        try:        
            _params = orb.utils.astrometry.get_wcs_parameters(wcs)
        except Exception as e:
            logging.debug('wcs could not be loaded:', e)
            return
            
        # remove old sip params if they exist
        for ipar in list(self.params.keys()):
            if ('A_' == ipar[:2]
                or 'B_' == ipar[:2]
                or 'AP_' == ipar[:3]
                or 'BP_' == ipar[:3]):
                del self.params[ipar]

        self.update_params(wcs.to_header(relax=True))
        
        # convert wcs to parameters so that FITS keywords and
        # comprehensive parameters are coherent.
        self.params['target_x'] = _params[0]
        self.params['target_y'] = _params[1]
        self.params['delta_x'] = _params[2]
        self.params['delta_y'] = _params[3]
        self.params['target_ra'] = _params[4]
        self.params['target_dec'] = _params[5]
        self.params['wcs_rotation'] = _params[6]
        
        self.validate_wcs()

    def get_wcs(self, validate=True):
        """Return the WCS of the cube as an astropy.wcs.WCS instance """
        if validate: self.validate_wcs()
        warnings.simplefilter('ignore', category=VerifyWarning)
        warnings.simplefilter('ignore', category=AstropyUserWarning)
        return pywcs.WCS(self.get_header(), naxis=2, relax=True)

    def validate_wcs(self):
        """Verify the internal coherence between comprehensive wcs parameters
        and FITS keywords.
        """
        try:
            _fits_params = np.array(orb.utils.astrometry.get_wcs_parameters(
                self.get_wcs(validate=False)))
        except Exception:
            warnings.warn('bad WCS in header')
            return
        
        _wcs_params = np.array([self.params['target_x'],
                                self.params['target_y'],
                                self.params['delta_x'],
                                self.params['delta_y'],
                                self.params['target_ra'],
                                self.params['target_dec'],
                                self.params['wcs_rotation']])

        if not np.all(np.isclose(_wcs_params - _fits_params, 0)):
            logging.debug('WCS FITS keywords and parameters are different:\n{}\n{}'.format(
                _wcs_params, _fits_params))
        
        
    def get_wcs_header(self):
        """Return the WCS of the cube as a astropy.io.fits.Header instance """
        return self.get_wcs().to_header(relax=True)

    def pix2world(self, xy, deg=True):
        """Convert pixel coordinates to celestial coordinates

        :param xy: A tuple (x,y) of pixel coordinates or a list of
          tuples ((x0,y0), (x1,y1), ...)

        :param deg: (Optional) If true, celestial coordinates are
          returned in sexagesimal format (default False).

        .. note:: it is much more efficient to pass a list of
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
            raise Exception('xy must be a tuple (x,y) of coordinates or a list of tuples ((x0,y0), (x1,y1), ...)')

        if not self.has_dxdymaps():
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
                self.dxdymaps[0], self.dxdymaps[1])
        if deg:
            return coords
        else: return np.array(
            [orb.utils.astrometry.deg2ra(coords[:,0]),
             orb.utils.astrometry.deg2dec(coords[:,1])])


    def world2pix(self, radec):
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
            raise Exception('radec must be a tuple (ra,dec) of coordinates or a list of tuples ((ra0,dec0), (ra1,dec1), ...)')

        if not self.has_dxdymaps():
            coords = np.array(
                self.get_wcs().all_world2pix(
                    ra, dec, 0,
                    detect_divergence=False,
                    quiet=True)).T
        else:
            radecarr = np.atleast_2d([ra, dec]).T
            coords = orb.utils.astrometry.world2pix(
                self.get_wcs_header(), self.dimx, self.dimy, radecarr,
                self.dxdymaps[0], self.dxdymaps[1])

        return coords


    def get_scale(self):
        """Return mean platescale in arcsec/pixel"""
        if self.is_cam1():
            basic_scale = (self.config.FIELD_OF_VIEW_1
                           / max(self.config.CAM1_DETECTOR_SIZE_X,
                                 self.config.CAM1_DETECTOR_SIZE_Y) * 60.)
        else:
            basic_scale = (self.config.FIELD_OF_VIEW_2
                           / max(self.config.CAM2_DETECTOR_SIZE_X,
                                 self.config.CAM1_DETECTOR_SIZE_Y) * 60.)

        if not self.has_param('delta_x') or not self.has_param('delta_y'):
            logging.debug('scale computed from config parameters')
            return basic_scale

        # check platescale
        platescale_ok = True
        if np.abs(((basic_scale / 3600.) / self.params.delta_x) - 1) > 0.05:
            platescale_ok = False
            warnings.warn('wcs platescale along X ({}) seems incoherent with known platescale ({})'.format(self.params.delta_x, basic_scale/3600.))
        if np.abs(((basic_scale / 3600.) / self.params.delta_y) - 1) > 0.05:
            platescale_ok = False
            warnings.warn('wcs platescale along Y ({}) seems incoherent with known platescale ({})'.format(self.params.delta_y, basic_scale/3600.))

        if platescale_ok:
            return np.mean([self.params.delta_x, self.params.delta_y]) * 3600.
        else:
            logging.debug('scale computed from config parameters')
            return basic_scale

    
    def arc2pix(self, x):
        """Convert pixels to arcseconds

        :param x: a value or a vector in pixel
        """
        return np.array(x).astype(float) / self.get_scale()

    def pix2arc(self, x):
        """Convert arcseconds to pixels

        :param x: a value or a vector in arcsec
        """
        return np.array(x).astype(float) * self.get_scale()

    def query_vizier(self, catalog='gaia', max_stars=100, as_pandas=False):
        """Return a list of star coordinates around an object in a
        given radius based on a query to VizieR Services
        (http://vizier.u-strasbg.fr/viz-bin/VizieR)    

        :param catalog: (Optional) Catalog to ask on the VizieR
          database (see notes) (default 'gaia')

        :param max_stars: (Optional) Maximum number of row to retrieve
          (default 100)

        :param as_pandas: (Optional) If True, results are returned as a
          pandas.DataFrame instance. Else a numpy.ndarray instance is
          returned (default False).

        .. seealso:: :py:meth:`orb.orb.utils.web.query_vizier`
        """
        radius = self.config['FIELD_OF_VIEW_1'] / np.sqrt(2)
        center_radec = self.pix2world([self.dimx/2., self.dimy/2])
        return orb.utils.web.query_vizier(
            radius, center_radec[0][0], center_radec[0][1],
            catalog=catalog, max_stars=max_stars, as_pandas=as_pandas)

    def get_initial_alignment_parameters(self):
        """Return initial alignemnt coefficients for camera 2 as a core.Params instance"""
        if self.instrument == 'spiomm':
            raise NotImplementedError()
        else:
            bin_cam_1 = self.params.binning
            bin_cam_2 = self.params.binning

        init_dx = self.config["INIT_DX"] / bin_cam_2
        init_dy = self.config["INIT_DY"] / bin_cam_2
        pix_size_1 = self.config["PIX_SIZE_CAM1"]
        pix_size_2 = self.config["PIX_SIZE_CAM2"]
        zoom = (pix_size_2 * bin_cam_2) / (pix_size_1 * bin_cam_1)
        xrc = self.dimx / 2.
        yrc = self.dimy / 2.
        
        coeffs = Params()
        coeffs.dx = float(init_dx)
        coeffs.dy = float(init_dy)
        coeffs.dr = float(self.config['INIT_ANGLE'])
        coeffs.da = 0.
        coeffs.db = 0.
        coeffs.rc = float(xrc), float(yrc)
        coeffs.zoom = float(zoom)

        return coeffs


    def writeto(self, path):
        """Write data to an hdf file

        :param path: hdf file path.
        """
        Data.writeto(self, path)
        with orb.utils.io.open_hdf5(path, 'a') as hdffile:
            if self.has_dxdymaps():
                hdffile.create_dataset(
                    '/dxmap',
                    data=self.dxdymaps[0].astype(float))
                hdffile.create_dataset(
                    '/dymap',
                    data=self.dxdymaps[1].astype(float))
                


