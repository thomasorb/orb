#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: core.py

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
import time
import math
import traceback
import inspect
import re
import datetime
import logging
import warnings

import threading
import SocketServer
import logging.handlers
import struct
import pickle
import select
import socket

import numpy as np
import bottleneck as bn
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from scipy import interpolate

try: import pygit2
except ImportError: pass

## MODULES IMPORTS
import cutils
import utils.spectrum, utils.parallel, utils.io, utils.filters
import utils.photometry

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

class LogRecordStreamHandler(SocketServer.StreamRequestHandler):
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
    
class LogRecordSocketReceiver(SocketServer.ThreadingTCPServer):
    """
    Simple TCP socket-based logging receiver suitable for testing.
    """
    def __init__(self, host='localhost',
                 port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
                 handler=LogRecordStreamHandler):
        SocketServer.ThreadingTCPServer.__init__(self, (host, port), handler)
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

        # get git branch (if possible)
        repo_path = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
        try:
            pygit2
            try:
                self.branch_name = pygit2.Repository(repo_path).head.shorthand + '|'
            except pygit2.GitError:
                self.branch_name = ''
        except NameError:
            self.branch_name = ''

        
        self.start_logging()
        
        # start tcp listener
        if self.debug:
            try:
                self.listen()
            except Exception, e:
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
            raise ValueError('logfilter must be in {}'.format(self.logfilters.keys()))
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
            raise StandardError('Logging in strange state: {}'.format(self.getLogger().handlers))

    def get_file_logging_state(self):
        """Return True if the file logging appears set"""
        _len = len(self.getLogger().handlers)
        if _len < 2: return False
        elif _len == 2: return True
        else:
            raise StandardError('File Logging in strange state: {}'.format(self.getLogger().handlers))
        
        
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
        except Exception, e:
            warnings.warn('Error during listener execution')



    
################################################
#### CLASS ROParams ############################
################################################
class ROParams(dict):
    """Special dictionary which elements can be accessed like
    attributes.

    Attributes are read-only and may be defined only once.
    """
    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    
    def __setattr__(self, key, value):
        """Special set attribute function. Always raise a read-only
        error.

        :param key: Attribute name.

        :param value: Attribute value.
        """
        raise StandardError('Parameter is read-only')

    def __setitem__(self, key, value):
        """Special set item function. Raises a warning when parameter
        already exists.
    
        :param key: Item key.

        :param value: Item value.
        """
        if key in self:
            if self[key] != value:
                warnings.warn('Parameter {} already defined'.format(key))
                warnings.warn('Old value={} / new_value={}'.format(self[key], value))
        dict.__setitem__(self, key, value)

    def __getstate__(self):
        """Used to pickle object"""
        state = self.copy()
        return state

    def __setstate__(self, state):
        """Used to unpickle object"""
        self.update(state)

    def reset(self, key, value):
        """Force config parameter reset"""
        dict.__setitem__(self, key, value)

    def convert(self):
        """Convert to a nice pickable object"""
        conv = dict()
        conv.update(self)
        return conv
    
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
            raise AttributeError("Instrumental configuration not loaded. Set the option 'instrument' to a valid instrument name")
        return ROParams.__getitem__(self, key)

    __getattr__ = __getitem__

     
#################################################
#### CLASS Tools ################################
#################################################

class Tools(object):
    """
    Parent class of all classes of orb.

    Manage configuration file, implement basic methods and manage the
    server for parallel processing.
    """
    instruments = ['sitelle', 'spiomm']
    
    _MASK_FRAME_TAIL = '_mask.fits' # Tail of a mask frame
    
    def __init__(self, instrument=None, data_prefix="./temp/data.",
                 config=None,
                 tuning_parameters=dict(), ncpus=None, silent=False):
        """Initialize Tools class.

        :param instrument: (Optional) Instrument configuration to
          load. If None a minimal configuration file is loaded. Some
          functions are not available in this case (default None).

        :param data_prefix: (Optional) Prefix used to determine the
          header of the name of each created file (default
          'temp_data')

        :param config: (Optional) Configuration dictionary to update
          default loaded config values (default None).

        :param ncpus: (Optional) Number of CPUs to use for parallel
          processing. set to None gives the maximum number available
          (default None).

        :param tuning_parameters: (Optional) Some parameters of the
          methods can be tuned externally using this dictionary. The
          dictionary must contains the full parameter name
          (class.method.parameter_name) and its value. For example :
          {'InterferogramMerger.find_alignment.BOX_SIZE': 7}. Note
          that only some parameters can be tuned. This possibility is
          implemented into the method itself with the method
          :py:meth:`core.Tools._get_tuning_parameter`.

        :param silent: If True only error messages will be diplayed on
          screen (default False).
        """
        self.params = ROParams()

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
        self.config['QUAD_NB'] = self.config.DIV_NB**2L
        self.set_config('BIG_DATA', bool)
        self.set_config('NCPUS', int)
        if ncpus is None:
            self.ncpus = int(self.config.NCPUS)
        else:
            self.ncpus = int(ncpus)

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

            self.set_config('OPTIM_DARK_CAM1', bool) 
            self.set_config('OPTIM_DARK_CAM2', bool)
            self.set_config('EXT_ILLUMINATION', bool)

            self.set_config('DETECT_STAR_NB', int)
            self.set_config('INIT_FWHM', float)
            self.set_config('PSF_PROFILE', str)
            self.set_config('MOFFAT_BETA', float)
            self.set_config('DETECT_STACK', int)
            self.set_config('ALIGNER_RANGE_COEFF', float)
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
        self._msg_class_hdr = self._get_msg_class_hdr()
        self._data_path_hdr = self._get_data_path_hdr()
        self._tuning_parameters = tuning_parameters
        self._silent = silent
        if self.instrument is not None:
            self.set_param('instrument', self.instrument)

        if config is not None:
            self.update_config(config)

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
            if ikey not in self.config:
                raise ValueError('Unknown config key: {}'.format(ikey))
            self.config[ikey] = config[ikey]
        
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
        if self.config_file_name is None:
            raise StandardError('No instrument configuration given')
        config_file_path = self._get_orb_data_file_path(
            self.config_file_name)
        if not os.path.exists(config_file_path):
             raise StandardError(
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
        else: raise StandardError('Bad camera number, must be 0, 1 or 2')
        
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
             raise StandardError(
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
            raise StandardError('Group must be in %s'%str(groups))
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
          can be 'MASSEY', 'CALSPEC', 'MISC' or 'OKE'.
        """
        std_table = self.open_file(self._get_standard_table_path(
            standard_table_name=standard_table_name), 'r')

        for iline in std_table:
            iline = iline.split()
            if len(iline) >= 3:
                if iline[0] in standard_name:
                    file_path = self._get_orb_data_file_path(iline[2])
                    if os.path.exists(file_path):
                        return file_path, iline[1]

        raise StandardError('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')

    def _get_standard_radec(self, standard_name,
                            standard_table_name='std_table.orb',
                            return_pm=False):
        """
        Return a standard spectrum file path.
        
        :param standard_name: Name of the standard star. Must be
          recorded in the standard table.
        
        :param standard_table_name: (Optional) Name of the standard
          table file (default std_table.orb).

        :param return_pm: (Optional) Returns also proper motion if
          recorded (in mas/yr), else returns 0.

        :return: A tuple [standard file path, standard type]. Standard type
          can be 'MASSEY', 'MISC', 'CALSPEC' or 'OKE'.
        """
        std_table = self.open_file(self._get_standard_table_path(
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
                        raise StandardError('No RA DEC recorded for standard: {}'.format(
                            standard_name))
                    

        raise StandardError('Standard name unknown. Please see data/std_table.orb for the list of recorded standard spectra')


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
        f = self.open_file(
            self._get_config_file_path(), 'r')
        for line in f:
            if len(line) > 2:
                if line.split()[0] == param_key:
                    return line.split()[1]
        if not optional:
            raise StandardError("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
        else:
            warnings.warn("Parameter key %s not found in file %s"%(
                param_key, self.config_file_name))
            return None

    def _print_caller_traceback(self):
        """Print the traceback of the calling function."""
        traceback = inspect.stack()
        traceback_msg = ''
        for i in range(len(traceback))[::-1]:
            traceback_msg += ('  File %s'%traceback[i][1]
                              + ', line %d'%traceback[i][2]
                              + ', in %s\n'%traceback[i][3] +
                              traceback[i][4][0])
            
        logging.debug('\r' + traceback_msg)

        
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
            
        logging.debug("\r" + message)
            

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
        hdr.append(('OBSERVAT', self.config.OBSERVATORY_NAME, 
                    'Observatory name'))
        hdr.append(('TELESCOP', self.config.TELESCOPE_NAME,
                    'Telescope name'))  
        hdr.append(('INSTRUME', self.config.INSTRUMENT_NAME,
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
        FIELD_OF_VIEW = self.config.FIELD_OF_VIEW_1
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
        logging.info('looking for tuning parameter: {}'.format(
            full_parameter_name))
        if full_parameter_name in self._tuning_parameters:
            warnings.warn(
                'Tuning parameter {} changed to {} (default {})'.format(
                    full_parameter_name,
                    self._tuning_parameters[full_parameter_name],
                    default_value))
            return self._tuning_parameters[full_parameter_name]
        else:
            return default_value
                

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
            raise StandardError("mode option must be 'w', 'r', 'rU' or 'a'")
            
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
            raise StandardError('Malformed sequence of files: {}:\n{}'.format(
                e, file_seq))
                             
            
        # get changing column
        test = np.sum(file_keys == file_keys[0,:], axis=0)
        
        if np.min(test) > 1:
            warnings.warn('Images list cannot be safely sorted. Two images at least have the same index')
            column_index = np.nan
        else:
            column_index = np.argmin(test)

        # get changing step (if possible)
        steplist = list()
        if cube:
            for path in file_list:
                if '.fits' in path:
                    try:
                        hdr = utils.io.read_fits(
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
            raise StandardError('Image list cannot be sorted.')
            
        return file_list

    def save_sip(self, fits_path, hdr, overwrite=True):
        """Save SIP parameters from a header to a blanck FITS file.

        :param fits_path: Path to the FITS file
        :param hdr: header from which SIP parameters must be read
        :param overwrite: (Optional) Overwrite the FITS file.
        """    
        clean_hdr = self._clean_sip(hdr)
        data = np.empty((1,1))
        data.fill(np.nan)
        utils.io.write_fits(
            fits_path, data, fits_header=clean_hdr, overwrite=overwrite)

    def load_sip(self, fits_path):
        """Return a astropy.wcs.WCS object from a FITS file containing
        SIP parameters.
    
        :param fits_path: Path to the FITS file    
        """
        hdr = utils.io.read_fits(fits_path, return_hdu_only=True)[0].header
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
        quad_nb = div_nb**2
        
        if (quad_number < 0) or (quad_number > quad_nb - 1L):
            raise StandardError("quad_number out of bounds [0," + str(quad_nb- 1L) + "]")
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


            

        
#################################################
#### CLASS OCube ################################
#################################################
class OCube(Cube):
    """Provide additional cube methods when observation parameters are known.
    """

    needed_params = ('step', 'order', 'filter_name', 'exposure_time',
                     'step_nb', 'calibration_laser_map_path', 'zpd_index')

    optional_params = ('target_ra', 'target_dec', 'target_x', 'target_y',
                       'dark_time', 'flat_time', 'camera_index', 'wcs_rotation')

    computed_params = ('filter_nm_min', 'filter_nm_max', 'filter_file_path',
                       'filter_cm1_min', 'filter_cm1_max', 'binning')
    
    def __init__(self, data, params, **kwargs):
        """
        Initialize Cube class.

        :param data: Can be a path to a FITS file containing a data
          cube or a 3d numpy.ndarray. Can be None if data init is
          handled differently (e.g. if this class is inherited)

        :param params: Path to an option file or dict instance.

        :param kwargs: Cube kwargs + other parameters not supplied in
          params (else overwrite parameters set in params)
        """
        kwargs_params = dict()
        for ikey in kwargs.keys():
            if ikey in (self.needed_params + self.optional_params):
                kwargs_params[ikey] = kwargs.pop(ikey)
        Cube.__init__(self, data, **kwargs)
        self.needed_params = tuple(self.params.keys() + list(self.needed_params))
        self.load_params(params, **kwargs_params)
        if data is not None:
            # else it should be computed in the init of the child
            # class when data parameters are known (e.g. self.dimx,
            # self.dimy etc.)
            self.compute_data_parameters()
            
    def load_params(self, params, **kwargs):
        """Load observation parameters

        :params: Path to an option file or a dict

        :param kwargs: other parameters not supplied in params (else
          overwrite parameters set in params)
        """
        def set_param(pkey, okey, cast):
            if okey in ofile.options:
                self.set_param(pkey, ofile.get(okey, cast=cast))
            else:
                raise Exception('Malformed option file. {} not set.'.format(okey))

        # parse optionfile
        if isinstance(params, str):
            if os.path.exists(params):
                ofile = OptionFile(params)
                for iopt in ofile.options:
                    logging.debug('{}: {}'.format(iopt, ofile[iopt]))
                set_param('step', 'SPESTEP', float)
                set_param('order', 'SPEORDR', int)
                set_param('filter_name', 'FILTER', str)                
                set_param('exposure_time', 'SPEEXPT', float)
                set_param('step_nb', 'SPESTNB', int)
                set_param('calibration_laser_map_path', 'CALIBMAP', str)
                
            else:
                raise IOError('file {} does not exist'.format(params))

        # parse dict
        elif isinstance(params, dict):
            for iparam in (self.needed_params + self.optional_params):
                if iparam in params:
                    self.set_param(iparam, params[iparam])
                
        else: raise TypeError('params type ({}) not handled'.format(type(params)))

        # parse additional parameters
        for iparam in kwargs:
            if iparam in (self.needed_params + self.optional_params):
                self.set_param(iparam, kwargs[iparam])
            else: raise ValueError('parameter {} supplied as a keyword argument but not used. Please remove it'.format(iparam))
            

        # validate needed params
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise ValueError('parameter {} must be defined in params'.format(iparam))

        for iparam in self.params:
            if iparam not in (self.needed_params + self.optional_params):
                warnings.warn('parameter {} defined but not used'.format(iparam))

        
        # compute additional parameters
        self.filterfile = FilterFile(self.params.filter_name)
        self.set_param('filter_file_path', self.filterfile.basic_path)
        self.set_param('filter_nm_min', self.filterfile.get_filter_bandpass()[0])
        self.set_param('filter_nm_max', self.filterfile.get_filter_bandpass()[1])
        self.set_param('filter_cm1_min', self.filterfile.get_filter_bandpass_cm1()[0])
        self.set_param('filter_cm1_max', self.filterfile.get_filter_bandpass_cm1()[1])

        
        self.params_defined = True

    def validate(self):
        """Check if this class is valid"""
        if self.instrument not in ['sitelle', 'spiomm']: raise StandardError("class not valid: set instrument to 'sitelle' or 'spiomm' at init")
        try: self.params_defined
        except AttributeError: raise StandardError("class not valid: set params at init")
        if not self.params_defined: raise StandardError("class not valid: set params at init")

        # validate needed params
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise ValueError('parameter {} must be defined in params'.format(iparam))

        # validate optional parameters
        for iparam in self.params:
            if iparam not in (self.needed_params + self.optional_params + self.computed_params):
                warnings.warn('parameter {} defined but not used'.format(iparam))


    def compute_data_parameters(self):
        """Compute some more parameters when data paramters are known (like self.dimx, self.dimy etc.)"""
        if 'camera_index' in self.params:
            detector_shape = [self.config['CAM{}_DETECTOR_SIZE_X'.format(self.params.camera_index)],
                              self.config['CAM{}_DETECTOR_SIZE_Y'.format(self.params.camera_index)]]
            binning = utils.image.compute_binning(
                (self.dimx, self.dimy), detector_shape)
                            
            if binning[0] != binning[1]:
                raise StandardError('Images with different binning along X and Y axis are not handled by ORBS')
            self.set_param('binning', binning[0])
            
            logging.debug('Computed binning of camera {}: {}x{}'.format(
                self.params.camera_index, self.params.binning, self.params.binning))


    def get_uncalibrated_filter_bandpass(self):
        """Return filter bandpass as two 2d matrices (min, max) in pixels"""
        self.validate()
        filterfile = FilterFile(self.get_param('filter_file_path'))
        filter_min_cm1, filter_max_cm1 = utils.spectrum.nm2cm1(
            filterfile.get_filter_bandpass())[::-1]
        
        cm1_axis_step_map = cutils.get_cm1_axis_step(
            self.dimz, self.params.step) * self.get_calibration_coeff_map()

        cm1_axis_min_map = (self.params.order / (2 * self.params.step)
                            * self.get_calibration_coeff_map() * 1e7)
        if int(self.params.order) & 1:
            cm1_axis_min_map += cm1_axis_step_map
        filter_min_pix_map = (filter_min_cm1 - cm1_axis_min_map) / cm1_axis_step_map
        filter_max_pix_map = (filter_max_cm1 - cm1_axis_min_map) / cm1_axis_step_map
        
        return filter_min_pix_map, filter_max_pix_map

    
    def get_calibration_laser_map(self):
        """Return calibration laser map"""
        self.validate()
        try:
            return np.copy(self.calibration_laser_map)
        except AttributeError:
            self.calibration_laser_map = utils.io.read_fits(self.params.calibration_laser_map_path)
        if (self.calibration_laser_map.shape[0] != self.dimx):
            self.calibration_laser_map = utils.image.interpolate_map(
                self.calibration_laser_map, self.dimx, self.dimy)
        if not self.calibration_laser_map.shape == (self.dimx, self.dimy):
            raise StandardError('Calibration laser map shape is {} and must be ({} {})'.format(self.calibration_laser_map.shape[0], self.dimx, self.dimy))
        return self.calibration_laser_map
        
    def get_calibration_coeff_map(self):
        """Return calibration laser map"""
        self.validate()
        try:
            return np.copy(self.calibration_coeff_map)
        except AttributeError:
            self.calibration_coeff_map = self.get_calibration_laser_map() / self.config.CALIB_NM_LASER 
        return self.calibration_coeff_map

    def get_theta_map(self):
        """Return the incident angle map from the calibration laser map"""
        self.validate()
        try:
            return np.copy(self.theta_map)
        except AttributeError:
            self.theta_map = utils.spectrum.corr2theta(self.get_calibration_coeff_map())

    def get_base_axis(self):
        """Return the spectral axis (in cm-1) at the center of the cube"""
        self.validate()
        try: return Axis(np.copy(self.base_axis))
        except AttributeError:
            calib_map = self.get_calibration_coeff_map()
            self.base_axis = utils.spectrum.create_cm1_axis(
                self.dimz, self.params.step, self.params.order,
                corr=calib_map[calib_map.shape[0]/2, calib_map.shape[1]/2])
        return Axis(np.copy(self.base_axis))

    def get_axis(self, x, y):
        """Return the spectral axis at x, y
        """
        self.validate()
        utils.validate.index(x, 0, self.dimx, clip=False)
        utils.validate.index(y, 0, self.dimy, clip=False)
        
        axis = utils.spectrum.create_cm1_axis(
            self.dimz, self.params.step, self.params.order,
            corr=self.get_calibration_coeff_map()[x, y])
        return Axis(np.copy(axis))

    def add_params_to_hdf_file(self, hdffile):
        """Write parameters as attributes to an hdf5 file"""
        self.validate()
        for iparam in self.params:
            hdffile.attrs[iparam] = self.params[iparam]

    def get_astrometry(self, data=None, profile_name=None, **kwargs):
        """Return an astrometry.Astrometry instance.

        :param data: (Optional) data to pass. If None, the cube itself
          is passed.

        :param profile_name: (Optional) PSF profile. Can be gaussian or
          moffat. default is read in config file (PSF_PROFILE).

        :param kwargs: orb.astrometry.Astrometry kwargs.
        """
        self.validate()
        from astrometry import Astrometry # cannot be imported at the
                                          # beginning of the file

        if profile_name is None:
            profile_name = self.config.PSF_PROFILE

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

        if data is None:
            data = self
        else:
            utils.validate.is_2darray(data, object_name='data')
            if data.shape != (self.dimx, self.dimy): raise TypeError('data must be a 2d array of shape {} {}'.format(self.dimx, self.dimy))

        if 'data_prefix' not in kwargs:
            kwargs['data_prefix'] = self._data_prefix

        if 'wcs_rotation' in self.params:
            wcs_rotation = self.params.wcs_rotation
        else:
            wcs_rotation=self.config.INIT_ANGLE

        self.validate()
        
        return Astrometry(
            data, profile_name=profile_name,
            moffat_beta=self.config.MOFFAT_BETA,
            tuning_parameters=self._tuning_parameters,
            instrument=self.params.instrument,
            ncpus=self.ncpus,
            target_radec=target_radec,
            target_xy=target_xy,
            wcs_rotation=wcs_rotation,
            **kwargs)
        
##################################################
#### CLASS ProgressBar ###########################
##################################################


class ProgressBar(object):
    """Display a simple progress bar in the terminal

    :param max_index: Index representing a 100% completed task.
    """

    REFRESH_COUNT = 3L # number of steps used to calculate a remaining time
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
            for _icount in range(self.REFRESH_COUNT - 1L):
                self._time_table[_icount] = self._time_table[_icount + 1L]
                self._index_table[_icount] = self._index_table[_icount + 1L]
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
            raise StandardError(
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
            raise StandardError('Bad file group. File group can be in %s, in %s or None'%(str(self.file_groups), str(self.file_group_indexes)))

        if file_key in self.index:
            return self[file_key]
        else:
            if err:
                raise StandardError("File key '%s' does not exist"%file_key)
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
            
        else: raise StandardError(
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
        for ikey in self.air_lines_nm.iterkeys():
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
        f = self.open_file(sky_lines_file_path, 'r')
        self.air_sky_lines_nm = dict()
        try:
            for line in f:
                if '#' not in line and len(line) > 2:
                    line = line.split()
                    self.air_sky_lines_nm[line[1]] = (float(line[1]) / 10., float(line[3]))
        except Exception, e:
            raise StandardError('Error during parsing of {}: {}'.format(sky_lines_file_path, e))
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
        return utils.spectrum.nm2cm1(
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
#### CLASS OptionFile ###########################
#################################################
        
class OptionFile(object):
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

    # special keywords that can be used mulitple times without being
    # overriden.
    protected_keys = ['REG', 'TUNE'] 

    def __init__(self, option_file_path, protected_keys=[]):
        """Initialize class

        :param option_file_path: Path to the option file

        :param protected_keys: (Optional) Add other protected keys to
          the basic ones (default []).
        """        
        # append new protected keys
        for key in protected_keys:
            self.protected_keys.append(key) 
        
        self.option_file = open(option_file_path, 'r')
        self.input_file_path = str(option_file_path)
        self.options = dict()
        self.lines = Lines()
        self.header_line = None
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
                        else: raise StandardError('Keyword SKY used but nm_min, nm_max or delta_nm parameter not set')
                            
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
                raise StandardError(
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
                raise StandardError("Fringes badly defined. Use no whitespace, each fringe must be separated by a ':'. Fringes parameters must be given in the order [frequency, amplitude] separated by a ',' (e.g. 150.0,0.04:167.45,0.095 gives 2 fringes of parameters [150.0, 0.04] and [167.45, 0.095]).")
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
            raise StandardError("Bad frames badly defined. Use no whitespace, bad frames must be comma separated. the sign ':' can be used to define a range of bad frames, commas and ':' can be mixed. Frame index must be integer.")

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
                        raise StandardError(
                            'Wrong file format: {:s}'.format(file_path))
            self.f.close()
            self.f = self.open_file(file_path, 'a')

        else:
            self.f = self.open_file(file_path, 'w')
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
                raise StandardError('parameters of the new entry are not the same as the old entries')
        
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
            raise StandardError('nm must be an array of dimension 1')

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
            raise StandardError('Velocity array shape must be the same as nm shape')
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
#### CLASS Header ###############################
#################################################
class Header(pyfits.Header):
    """Extension of :py:class:`astropy.io.fits.Header`"""

    def __init__(self, *args, **kwargs):
        """Initialize Header class.

        :param args: args of :py:class:`astropy.io.fits.Header`

        :param kwargs: Kwargs of :py:class:`astropy.io.fits.Header`
        """
        pyfits.Header.__init__(self, *args, **kwargs)
        try:
            self.wcs = pywcs.WCS(self, relax=True)
        except pywcs.WcsError, e:
            warnings.warn('Exception occured during wcs interpretation: {}'.format(e))
            self.wcs = None

    def bin_wcs(self, binning):
        """Bin WCS

        :param binning: Binning
        """
        if self.wcs is None:
            raise StandardError('No WCS is set')
        self.wcs.wcs.crpix /= binning
        self.wcs.wcs.cdelt *= binning
        self.extend(self.wcs.to_header(), update=True)

    def tostr(self):
        """Return a nice string to print"""
        return self.tostring(sep='\n')

#################################################
#### CLASS Vector1d #############################
#################################################
class Vector1d(object):
    """Basic 1d vector management class.

    The vector can have a projection axis.

    Useful for checking purpose.
    """
    needed_params = ()
    optional_params = ()
    
    def __init__(self, vector, axis=None, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray vector or a path to an hdf5
          vector file.

        :param axis: (optional) A 1d numpy.ndarray axis (default None)
          with the same size as vector.

        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are stored in self.needed_params (default None). If vector
          is a path it must be set to None.

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.

        """
        self.axis = None

        if isinstance(vector, str):
            if params is not None:
                # params are transformed into kwargs
                kwargs.update(params)
                
            with utils.io.open_hdf5(vector, 'r') as hdffile:
                params = dict()
                for iparam in hdffile.attrs:
                    params[iparam] = hdffile.attrs[iparam]
                vector = hdffile['/vector'][:]
                if '/axis' in hdffile:
                    if axis is not None:
                        raise StandardError('An axis is already provided in the input file')
                    axis = hdffile['/axis'][:]


        elif isinstance(vector, self.__class__):
            if vector.axis is not None:
                if axis is not None:
                    raise StandardError('An axis is already provided in the input class')
                axis = np.copy(vector.axis.data)
            vector = np.copy(vector.data)
            
        else:
            vector = np.copy(vector)

        # checking
        if not isinstance(vector, np.ndarray):
            raise TypeError('input vector is a {} but must be a numpy.ndarray'.format(type(vector)))
        if vector.ndim != 1:
            vector = np.squeeze(vector)
            if vector.ndim != 1:
                raise TypeError('input vector has {} dims but must have only one dimension'.format(vector.ndim))
        if len(np.nonzero(vector)[0]) == 0:
            self.allzero = True
        else:
            self.allzero = False

        if np.all(np.isnan(vector)):
            self.allnan = True
        else:
            self.allnan = False

        if np.any(np.isnan(vector)):
            self.anynan = True
        else:
            self.anynan = False


        self.data = np.copy(vector)
        self.data_orig = np.copy(vector)
        self.step_nb = self.data.shape[0]

        if axis is not None:
            axis = Axis(axis)
            if axis.step_nb != self.step_nb:
                raise TypeError('axis must have the same length as vector')
            self.axis = axis
            
        # load parameters
        self.params = None        

        def load_params(params, plist, needed=False, pop=True):
            for iparam in plist:
                try:
                    self.params[iparam] = kwargs.pop(iparam)
                except KeyError: 
                    try:
                        if pop:
                            self.params[iparam] = params.pop(iparam)
                        else:
                            self.params[iparam] = params[iparam]
                    except KeyError:
                        if needed:
                            raise KeyError('param {} needed in params dict'.format(iparam))
            return params
            
        
        if params is not None:
            if not isinstance(params, dict):
                raise TypeError('params must be a dict ! not a {}'.format(type(params)))

            params = dict(params)
            
            self.params = ROParams()
                
            params = load_params(params, self.needed_params, needed=True)
            params = load_params(params, self.optional_params, needed=False)

            # for iparam in params:
            #     logging.debug('parameter {} not added'.format(iparam))

        # check keyword arguments validity
        for iparam in kwargs:
            if iparam not in self.needed_params and iparam not in self.optional_params:
                raise KeyError('supplied keyword argument {} not understood'.format(iparam))


    def __getitem__(self, key):
        """Getitem special method"""
        return self.__class__(self.data.__getitem__(key), params=self.params)

    def has_params(self):
        """Check the presence of observation parameters"""
        if self.params is None:
            return False
        elif len(self.params) == 0:
            return False
        else: return True

    def assert_params(self):
        """Assert the presence of observation parameters"""
        if not self.has_params():
            raise StandardError(
                'Parameters not supplied, please give: {} at init'.format(
                    self.needed_params))

    def assert_axis(self):
        """Assert the presence of an axis"""
        if self.axis is None: raise StandardError('No axis supplied')

    def copy(self):
        """Return a copy of the instance"""
        if self.has_params():
            _params = self.params.convert()
        else:
            _params = None

        if self.axis is not None:
            _axis = np.copy(self.axis.data)
        else:
            _axis = None

        return self.__class__(np.copy(self.data), axis=_axis, params=_params)

    def reverse(self):
        """Reverse data. Do not reverse the axis.
        """
        self.data = self.data[::-1]

    def project(self, new_axis, returned_class=None):
        """Project vector on a new axis

        :param new_axis: Axis. Must be an orb.core.Axis instance.

        :param returned_class: (Optional) If not None, set the
          returned class. Must be a subclass of Vector1d.
        """
        self.assert_axis()

        if returned_class is None:
            returned_class = self.__class__
        else:
            if not issubclass(returned_class, Vector1d):
                raise TypeError('Returned class must be a subclass of Vector1d')
            
        if not isinstance(new_axis, Axis):
            raise TypeError('axis must be an orb.core.Axis instance')
        f = interpolate.interp1d(self.axis.data.astype(np.float128),
                                 self.data.astype(np.float128),
                                 bounds_error=False)
        return returned_class(
            f(new_axis.data), axis=new_axis.data, params=self.params)

    def math(self, opname, arg=None):
        """Do math operations with another vector instance.

        :param opname: math operation, must be a numpy.ufuncs.

        :param arg: If None, no argument is supplied. Else, can be a
          float or a Vector1d instance.
        """
        self.assert_axis()

        out = self.copy()
        try:
            if arg is None:
                getattr(np, opname)(out.data, out=out.data)
            else:
                if isinstance(arg, Vector1d):
                    arg = arg.project(out.axis).data
                elif np.size(arg) != 1:
                    raise TypeError('arg must be a float or a Vector1d instance')

                getattr(np, opname)(out.data, arg, out=out.data)

        except AttributeError:
            raise AttributeError('unknown operation')

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

    def writeto(self, path):
        """Write cm1 vector and params to an hdf file

        :param path: hdf file path.
        """
        if np.iscomplexobj(self.data):
            _data = self.data.astype(complex)
        else:
            _data = self.data.astype(float)
            
        with utils.io.open_hdf5(path, 'w') as hdffile:
            if self.has_params():
                for iparam in self.params:
                    hdffile.attrs[iparam] = self.params[iparam]

            hdffile.create_dataset(
                '/vector',
                data=_data)

            if self.axis is not None:
                hdffile.create_dataset(
                    '/axis',
                    data=self.axis.data.astype(float))


#################################################
#### CLASS Axis #################################
#################################################
class Axis(Vector1d):
    """Axis class"""

    def __init__(self, axis, params=None, **kwargs):
        """Init class with an axis vector

        :param axis: Regularly sampled and naturally ordered axis.

        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are stored in self.needed_params (default None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        Vector1d.__init__(self, axis, axis=None, params=params, **kwargs)

        # check that axis is regularly sampled
        diff = np.diff(self.data)
        if np.any(~np.isclose(diff - diff[0], 0.)):
            raise StandardError('axis must be regularly sampled')
        if self.data[0] > self.data[-1]:
            raise StandardError('axis must be naturally ordered')

        self.axis_step = diff[0]

    def __call__(self, pos):
        """return the position in channels from an input in axis unit

        :param pos: Postion in the axis in the axis unit

        :return: Position in index
        """
        pos_index = (pos - self.data[0]) / float(self.axis_step)
        if pos_index < 0 or pos_index >= self.step_nb:
            warnings.warn('requested position is off axis')
        return pos_index

    
#################################################
#### CLASS Cm1Vector1d ##########################
#################################################
class Cm1Vector1d(Vector1d):
    """1d vector class for data projected on a cm-1 axis (e.g. complex
    spectrum, phase)

    """
    needed_params = ('filter_file_path', )
    optional_params = ('filter_cm1_min', 'filter_cm1_max', 'step', 'order',
                       'zpd_index', 'calib_coeff', 'filter_file_path', 'nm_laser')

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
                check_axis = utils.spectrum.create_cm1_axis(
                    self.step_nb, self.params.step, self.params.order, corr=self.params.calib_coeff)
                if self.axis is None:
                    self.axis = Axis(check_axis)
                else:
                    if np.any(check_axis != self.axis.data):
                        warnings.warn('provided axis is inconsistent with the given parameters')
            
        if self.axis is None: raise StandardError('an axis must be provided or the observation parameters ({}) must be provided'.format(self.obs_params))
            
    def get_filter_bandpass_cm1(self):
        """Return filter bandpass in cm-1"""
        if 'filter_cm1_min' not in self.params or 'filter_cm1_max' not in self.params:
            
            cm1_min, cm1_max = FilterFile(self.params.filter_file_path).get_filter_bandpass_cm1()
            warnings.warn('Uneffective call to get filter bandpass. Please provide filter_cm1_min and filter_cm1_max in the parameters.')
            self.params['filter_cm1_min'] = cm1_min
            self.params['filter_cm1_max'] = cm1_max
            
        return self.params.filter_cm1_min, self.params.filter_cm1_max

    def get_filter_bandpass_pix(self, border_ratio=0.):
        """Return filter bandpass in channels

        :param border_ratio: (Optional) Relative portion of the phase
          in the filter range removed (can be a negative float,
          default 0.)
        
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
        ff = FilterFile(self.params.filter_file_path)
        ftrans = ff.get_transmission(self.step_nb)
        return np.sum(self.multiply(ftrans).data) / ftrans.sum() 
    
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
            return interpolate.UnivariateSpline(
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
            corr = utils.spectrum.theta2corr(
                self.tools.config['OFF_AXIS_ANGLE_CENTER'])
        cm1_axis = Axis(utils.spectrum.create_cm1_axis(
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
        return utils.spectrum.nm2cm1(self.get_filter_bandpass())[::-1]
