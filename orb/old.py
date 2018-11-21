import core
import utils.io

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
from core import ProgressBar

##################################################
#### CLASS Cube ##################################
##################################################
class Cube(core.Tools):
    """3d numpy data cube handling. Base class for all Cube classes"""
    def __init__(self, data,
                 project_header=list(),
                 wcs_header=list(), calibration_laser_header=list(),
                 overwrite=True, 
                 indexer=None, **kwargs):
        """
        Initialize Cube class.

        :param data: Can be a path to a FITS file containing a data
          cube or a 3d numpy.ndarray. Can be None if data init is
          handled differently (e.g. if this class is inherited)
        
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
          be overwritten (default True).
          
        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param kwargs: (Optional) :py:class:`~orb.core.Tools` kwargs.
        """
        core.Tools.__init__(self, **kwargs)

        self.star_list = None
        self.z_median = None
        self.z_mean = None
        self.z_std = None
        self.mean_image = None
        self._silent_load = False

        self._return_mask = False # When True, __get_item__ return mask data
                                  # instead of 'normal' data

        # read directives
        self.is_hdf5_frames = None # frames are hdf5 frames (i.e. the cube is
                                   # not an HDF5 cube but is made from
                                   # hdf5 frames)
        self.is_hdf5_cube = None # tell if the cube is an hdf5 cube
                                 # (i.e. created via OutHDFCube or
                                 # OutHDFQuadCube)
        self.is_quad_cube = False # Basic cube is not quad cube (see
                                  # class HDFCube and OutHDFQuadCube)


        self.is_complex = False
        self.dtype = float
        
        if overwrite in [True, False]:
            self.overwrite = bool(overwrite)
        else:
            raise ValueError('overwrite must be True or False')
        
                
        self.indexer = indexer
        self._project_header = project_header
        self._wcs_header = wcs_header
        self._calibration_laser_header = calibration_laser_header

        if data is None: return
        
        # check data
        if isinstance(data, str):
            data = utils.io.read_fits(data)

        utils.validate.is_3darray(data)
        utils.validate.has_dtype(data, float)
        
        self._data = np.copy(data)
        self.dimx = self._data.shape[0]
        self.dimy = self._data.shape[1]
        self.dimz = self._data.shape[2]
        self.shape = (self.dimx, self.dimy, self.dimz)

    def __getitem__(self, key):
        """Getitem special method"""
        return self._data.__getitem__(key)

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
                        raise StandardError(
                            "Index error: list index out of range")
                else:
                    raise StandardError("Type error: list indices of slice must be integers")
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
                        raise StandardError(
                            "Index error: list index out of range")

                else:
                    raise StandardError("Type error: list indices of slice must be integers")
            else: slice_max = _max

        elif isinstance(_slice, int) or isinstance(_slice, long):
            slice_min = _slice
            slice_max = slice_min + 1
        else:
            raise StandardError("Type error: list indices must be integers or slices")
        return slice(slice_min, slice_max, 1)


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
            


#################################################
#### CLASS FDCube ###############################
#################################################
class FDCube(Cube):
    """
    Generate and manage a **virtual frame-divided cube**.

    .. note:: A **frame-divided cube** is a set of frames grouped
      together by a list.  Avoids storing a data cube in one large
      data file and loading an entire cube to process it.

    This class has been designed to handle large data cubes. Its data
    can be accessed virtually as if it was loaded in memory.

    .. code-block:: python
      :linenos:

      cube = Cube('liste') # A simple list is enough to initialize a Cube instance
      quadrant = Cube[25:50, 25:50, :] # Here you just load a small quadrant
      spectrum = Cube[84,58,:] # load spectrum at pixel [84,58]
    """

    def __init__(self, image_list_path, image_mode='classic',
                 chip_index=1, binning=1, params=None, no_sort=False, silent_init=False,
                 **kwargs):
        """Init frame-divided cube class

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

        :param params: Path to an option file or dictionary
          containting observation parameters.

        :param no_sort: (Optional) If True, no sort of the file list
          is done. Files list is taken as is (default False).

        :param silent_init: (Optional) If True no message is displayed
          at initialization.


        :param kwargs: (Optional) :py:class:`~orb.core.Cube` kwargs.
        """
        core.Tools.__init__(self, None, **kwargs)

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
                raise StandardError('No image path in the given image list')
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
                            self.is_hdf5_frames = True
                        elif os.path.splitext(image_name)[1] == '.fits':
                            self.is_hdf5_frames = False
                        else:
                            raise StandardError("Unrecognized extension of file {}. File extension must be '*.fits' or '*.hdf5' depending on its format.".format(image_name))

                        if self.is_hdf5_frames :
                            if self._image_mode != 'classic': warnings.warn("Image mode changed to 'classic' because 'spiomm' and 'sitelle' modes are not supported in hdf5 format.")
                            if self._prebinning != 1: warnings.warn("Prebinning is not supported for images in hdf5 format")
                            self._image_mode = 'classic'
                            self._prebinning = 1
        
                        if not self.is_hdf5_frames:
                            image_data = utils.io.read_fits(
                                image_name,
                                image_mode=self._image_mode,
                                chip_index=self._chip_index,
                                binning=self._prebinning)
                            self.dimx = image_data.shape[0]
                            self.dimy = image_data.shape[1]
                                                        
                        else:
                            with utils.io.open_hdf5(image_name, 'r') as f:
                                if 'hdu0/data' in f:
                                    shape = f['hdu0/data'].shape
                                    
                                    if len(shape) == 2:
                                        self.dimx, self.dimy = shape
                                    else: raise StandardError('Image shape must have 2 dimensions: {}'.format(shape))
                                else: raise StandardError('Bad formatted hdf5 file. Use core.Tools.write_hdf5 to get a correct hdf5 file for ORB.')
                            
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
                    logging.info("Data shape : (" + str(self.dimx) 
                                    + ", " + str(self.dimy) + ", " 
                                    + str(self.dimz) + ")")
            else:
                raise StandardError("Incorrect data shape : (" 
                                  + str(self.dimx) + ", " + str(self.dimy) 
                                  + ", " +str(self.dimz) + ")")
            
            if params is not None:
                self.compute_data_parameters()

    def __getitem__(self, key):
        """Implement the evaluation of self[key].
        
        .. note:: To make this function silent just set
          Cube()._silent_load to True.
        """
        # check return mask possibility
        if self._return_mask and not self._mask_exists:
            raise StandardError("No mask found with data, cannot return mask")
        
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
                    modules=("import logging",
                             "numpy as np",)))
                        for ijob in range(ncpus)]

                for ijob, job in jobs:
                    added_data[:,:,ijob] = job()

                data = np.dstack((data, added_data))
                if not self._silent_load and not (ik - z_slice.start)%500:
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
                if not self._silent_load and not (ik - z_slice.start)%500:
                    progress.update(ik - z_slice.start, info="Loading data")
            if not self._silent_load:
                progress.end()
            
        return np.squeeze(data)

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
        if not self.is_hdf5_frames:
            hdu = utils.io.read_fits(self.image_list[frame_index],
                                 return_hdu_only=True,
                                 return_mask=self._return_mask)
        else:
            hdu = utils.io.open_hdf5(self.image_list[frame_index], 'r')
        image = None
        stored_file_path = None
        if self._prebinning > 1:
            # already binned data is stored in a specific folder
            # to avoid loading more than one time the same image.
            # check if already binned data exists

            if not self.is_hdf5_frames:
                stored_file_path = os.path.join(
                    os.path.split(self._get_data_path_hdr())[0],
                    'STORED',
                    (os.path.splitext(
                        os.path.split(self.image_list[frame_index])[1])[0]
                     + '.{}.bin{}.fits'.format(self._image_mode, self._prebinning)))
            else:
                raise StandardError(
                    'prebinned data is not handled for hdf5 cubes')

            if os.path.exists(stored_file_path):
                image = utils.io.read_fits(stored_file_path)


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
                    if not self.is_hdf5_frames:
                        section = np.copy(
                            hdu[0].section[y_slice, x_slice].transpose())
                    else:
                        section = hdu['hdu0/data'][x_slice, y_slice]

                else: # FITS only
                    section = image[y_slice, x_slice].transpose()
        del hdu

         # FITS only
        if stored_file_path is not None and image is not None:
            utils.io.write_fits(stored_file_path, image, overwrite=True,
                            silent=True)

        self._return_mask = False # always reset self._return_mask to False
        return section

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
        if not self.is_hdf5_frames:
            hdu = utils.io.read_fits(self.image_list[index],
                                 return_hdu_only=True)
            hdu.verify('silentfix')
            return hdu[0].header
        else:
            hdu = utils.io.open_hdf5(self.image_list[index], 'r')
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
    def __init__(self, cube_path, params=None,
                 silent_init=False,
                 binning=None, **kwargs):
        
        """
        Initialize HDFCube class.
        
        :param cube_path: Path to the HDF5 cube.

        :param params: Path to an option file or dictionary containtin
          observation parameters.

        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default True).

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param binning: (Optional) Cube binning. If > 1 data will be
          transparently binned so that the cube will behave as as if
          it was already binned (default None).

        :param silent_init: (Optional) If True Init is silent (default False).

        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
        """
        Cube.__init__(self, None, **kwargs)
            
        self._hdf5f = None # Instance of h5py.File
        self.quad_nb = None # number of quads (set to None if HDFCube
                            # is not a cube split in quads but a cube
                            # split in frames)
        self.is_quad_cube = None # set to True if cube is split in quad. set to
                                 # False if split in frames.

        
        self.is_hdf5_cube = True
        self.is_hdf5_frames = False
        self.image_list = None
        self._prebinning = None
        if binning is not None:
            if int(binning) > 1:
                self._prebinning = int(binning)
            
        self._parallel_access_to_data = False

        if cube_path is None or cube_path == '': return
        
        with utils.io.open_hdf5(cube_path, 'r') as f:
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
                    raise StandardError("Corrupted HDF5 cube: 'quad_nb' attribute ([]) does not correspond to the real number of quads ({})".format(self.quad_nb, quad_nb))

                if self._get_hdf5_quad_path(0) in f:
                    # test whether data is complex
                    if np.iscomplexobj(f[self._get_hdf5_quad_data_path(0)]):
                        self.is_complex = True
                        self.dtype = complex
                    else:
                        self.is_complex = False
                        self.dtype = float
                
                else:
                    raise StandardError('{} is missing. A valid HDF5 cube must contain at least one quadrant'.format(
                        self._get_hdf5_quad_path(0)))
                    

            else:
                frame_nb = len(
                    [igrp for igrp in f
                     if 'frame' == igrp[:5]])

                if frame_nb != self.dimz:
                    raise StandardError("Corrupted HDF5 cube: 'dimz' attribute ({}) does not correspond to the real number of frames ({})".format(self.dimz, frame_nb))
                
            
                if self._get_hdf5_frame_path(0) in f:                
                    if ((self.dimx, self.dimy)
                        != f[self._get_hdf5_data_path(0)].shape):
                        raise StandardError('Corrupted HDF5 cube: frame shape {} does not correspond to the attributes of the file {}x{}'.format(f[self._get_hdf5_data_path(0)].shape, self.dimx, self.dimy))

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
                    raise StandardError('{} is missing. A valid HDF5 cube must contain at least one frame'.format(
                        self._get_hdf5_frame_path(0)))
                

        # binning
        if self._prebinning is not None:
            self.dimx = self.dimx / self._prebinning
            self.dimy = self.dimy / self._prebinning

        if (self.dimx) and (self.dimy) and (self.dimz):
            if not silent_init:
                logging.info("Data shape : (" + str(self.dimx) 
                                + ", " + str(self.dimy) + ", " 
                                + str(self.dimz) + ")")
        else:
            raise StandardError("Incorrect data shape : (" 
                            + str(self.dimx) + ", " + str(self.dimy) 
                              + ", " +str(self.dimz) + ")")

        self.shape = (self.dimx, self.dimy, self.dimz)

        if params is not None:
            self.compute_data_parameters()

        
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
            raise StandardError("No mask found with data, cannot return mask")
        
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

            with utils.io.open_hdf5(self.cube_path, 'r') as f:
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
            with utils.io.open_hdf5(self.cube_path, 'r') as f:
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
        with utils.io.open_hdf5(self.cube_path, 'r') as f:
            if attr in f.attrs:
                return f.attrs[attr]
            else:
                if not optional:
                    raise StandardError('Attribute {} is missing. The HDF5 cube seems badly formatted. Try to create it again with the last version of ORB.'.format(attr))
                else:
                    return None
                   

