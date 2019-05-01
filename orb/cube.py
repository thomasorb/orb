#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: cube.py

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
Cube handling module
"""

import numpy as np
import warnings
import os
import logging

import core
import old
import utils.io
import utils.misc
import utils.astrometry
import fft
import image

import scipy.interpolate
import gvar
import pyregion

import photometry

#################################################
#### CLASS HDFCube ##############################
#################################################
class HDFCube(core.WCSData):
    """ This class implements the use of an HDF5 cube."""        

    protected_datasets = 'data', 'mask', 'header', 'deep_frame', 'params', 'axis', 'calib_map', 'phase_map', 'phase_map_err', 'phase_maps_coeff_map', 'phase_maps_cm1_axis', 'standard_image'

    optional_params = ('target_ra', 'target_dec', 'target_x', 'target_y',
                       'dark_time', 'flat_time', 'camera', 'wcs_rotation',
                       'calibration_laser_map_path')

    def __init__(self, path, indexer=None,
                 instrument=None, config=None, data_prefix='./',
                 **kwargs):

        """Init HDFCube

        :param path: Path to an HDF5 cube

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param instrument: (Optional) instrument name ('sitelle' or
          'spiomm'). If it cannot be read from the file itself (in
          attributes) it must be set.

        :param kwargs: (Optional) :py:class:`~orb.core.Data` kwargs.
        """
        self.cube_path = str(path)
        self.writeable = False

        # create file if it does not exists
        if not os.path.exists(self.cube_path):
            raise IOError('File does not exist')
                                
        # check if cube has an old format in which case it must be
        # loaded before and passed as an instance to Data.
        self.is_old = False
        if isinstance(self.cube_path, str):
            with utils.io.open_hdf5(self.cube_path, 'r') as f:
                if 'level2' not in f.attrs:
                    warnings.warn('old cube architecture. IO performances could be reduced.')
                    self.is_old = True
                    self.oldcube = old.HDFCube(self.cube_path, silent_init=True)
                    self.data = old.FakeData(self.oldcube.shape,
                                             self.oldcube.dtype, 3)
                    self.axis = None
                    self.mask = None
                    self.params = core.ROParams()
                    for param in f.attrs:
                        self.params[param] = f.attrs[param]
                    if 'instrument' in self.params and instrument is None:
                        instrument = self.params['instrument']

        # init Tools and Data
        if not self.is_old:
            instrument = utils.misc.read_instrument_value_from_file(self.cube_path)

        # load header if present and update params
        header = self._read_old_header()
        if 'params' not in kwargs:
            kwargs['params'] = header
        else:
            if kwargs['params'] is None:
                kwargs['params'] = header
            else:
                for ikey in header:
                    if ikey not in kwargs['params']:
                        kwargs['params'][ikey] = header[ikey]

        # check if instrument is in params
        if instrument is None:
            if 'params' in kwargs:
                if 'instrument' in kwargs['params']:
                    instrument = kwargs['params']['instrument']
        
        if self.is_old:
            core.WCSData.__init__(self, self, instrument=instrument,
                                  data_prefix=data_prefix,
                                  config=config, **kwargs)
        else:
            core.WCSData.__init__(self, self.cube_path, instrument=instrument,
                                  data_prefix=data_prefix,
                                  config=config, **kwargs)

        # checking dims
        if self.data.ndim != 3:
            raise TypeError('input cube has {} dims but must have exactly 3 dimensions'.format(self.data.ndim))
        
        if indexer is not None:
            if not isinstance(indexer, core.Indexer):
                raise TypeError('indexer must be an orb.core.Indexer instance')
        self.indexer = indexer
    
    def __getitem__(self, key):
        """Implement getitem special method"""
        if self.is_old:
            if self.has_dataset('mask'):
                warnings.warn('mask is not handled for old cubes format')
            return self.oldcube.__getitem__(key)
        
        f = self.open_hdf5()
        _data = np.copy(f['data'].__getitem__(key))
        if self.has_dataset('mask'):
            _data *= self.get_dataset('mask').__getitem__((key[0], key[1]))

        # increase representation in case of complex or floats
        if _data.dtype == np.float32:
            _data = _data.astype(np.float64)
        elif _data.dtype == np.complex64:
            _data = _data.astype(np.complex128)

        return np.squeeze(_data)


    def _read_old_header(self):
        """Backward compatibility method. Read old 'header' dataset and return a dict()
        """
        obs_date_f = lambda x: np.array(x.strip().split('-'), dtype=int)
        hour_ut_f = lambda x: np.array(x.strip().split(':'), dtype=float)
        instrument_f = lambda x: x.strip().lower()
        
        def apodiz_f(x):
            if x != 'None':
                return float(x)
            else: return 1.

        translate = {'STEP': ('step', float),
                     'ORDER': ('order', int),
                     'AXISCORR': ('axis_corr', float),
                     'CALIBNM': ('nm_laser', float),
                     'OBJECT': ('object_name', str),
                     'FILTER': ('filter_name', str),
                     'APODIZ': ('apodization', apodiz_f),
                     'EXPTIME': ('exposure_time', float),
                     'FLAMBDA': ('flambda', float),
                     'STEPNB': ('step_nb', int),
                     'ZPDINDEX': ('zpd_index', int),
                     'WAVTYPE': ('wavetype', str),
                     'WAVCALIB': ('wavenumber_calibration', bool),
                     'DATE-OBS': ('obs_date', obs_date_f),
                     'HOUT_UT': ('hour_ut', hour_ut_f),
                     'INSTRUME': ('instrument', instrument_f)}
        
        if not self.has_dataset('header'): return dict()
        header = utils.io.header_hdf52fits(self.get_dataset('header', protect=False))
        params = dict()
        for ikey in header:
            if ikey in translate:
                params[translate[ikey][0]] = translate[ikey][1](header[ikey])
            elif ikey != 'COMMENT':
                params[ikey] = header[ikey]
                
        return params


    
    def copy(self):
        raise NotImplementedError('HDFCube instance cannot be copied')
    
    def open_hdf5(self, mode=None):
        """Return a handle on the hdf5 file.

        :param mode: opening mode. can be 'r' or 'a'. If None, opening
          mode is set to the same as the writeability of the class
          (use is_writeable() to check it). Note that an unwriteable
          cube cannot be forced to a mode different from 'r' unless
          self.writeable is manually set to True.
        """
        if mode is None:
            mode = 'r'
        else:
            if mode not in ['r', 'a']:
                raise ValueError('mode must be r or a')
            if not self.is_writeable() and mode != 'r':
                raise IOError('HDF5 file is not writeable.')

        # if a handle already exists try first to close the file
        # before opening it again
        try:
            self.hdffile.close()
        except Exception: pass

        self.hdffile = utils.io.open_hdf5(self.cube_path, mode)
        
        try:
            self.data = self.hdffile['data']
        except Exception: pass
        return self.hdffile
    
    def get_data(self, x_min, x_max, y_min, y_max, z_min, z_max, silent=False):
        """Return a part of the data cube.

        :param x_min: minimum index along x axis
        
        :param x_max: maximum index along x axis
        
        :param y_min: minimum index along y axis
        
        :param y_max: maximum index along y axis
        
        :param z_min: minimum index along z axis
        
        :param z_max: maximum index along z axis

        :param silent: (Optional) deprecated, only here for old cube
          architecture (default False).

        """
        if self.is_old:
            return self.oldcube.get_data(
                x_min, x_max, y_min, y_max, z_min, z_max,
                silent=silent)
        
        return self[x_min:x_max, y_min:y_max, z_min:z_max]

    def get_region(self, region):
        """Return a list of valid pixels from a ds9 region file or a ds9-style
        region definition

        e.g. for a circle defined in celestial coordinates:
          "fk5;circle(290.96388,14.019167,843.31194")"
        """
        return utils.misc.get_mask_from_ds9_region_file(
            region, [0, self.dimx], [0, self.dimy], integrate=True,
            header=self.get_header())
    

    def get_data_from_region(self, region):
        """Return a list of vectors extracted along the 3rd axis at the pixel
        positions defined by a list of pixels.

        Return also a list of corresponding axes if self.get_axis()
        returns something else than None.

        .. note:: pixels do not have to be contiguous but, as the
          quadrant containing all the pixels is extracted primarily to
          speed up the process, they should be contained in a field
          small enough to avoid filling the RAM (a 400x400 pixels box
          is generally a good limit).

        :param region: A ds9-like region file or a list of pixels
          having the same format as the list returned by np.nonzero(),
          i.e. (x_positions_1d_array, y_positions_1d_array).

        """
        SIZE_LIMIT = 400*400
        if isinstance(region, str):
            region = self.get_region(region)

        if len(region) != 2: raise TypeError('badly formatted region.')
        if not utils.validate.is_iterable(region[0], raise_exception=False):
            raise TypeError('badly formatted region.')
        if not utils.validate.is_iterable(region[1], raise_exception=False):
            raise TypeError('badly formatted region.')
        if not len(region[0]) == len(region[1]):
            raise TypeError('badly formatted region.')

        if len(region[0]) == 1:
            return np.atleast_2d(self[region[0][0], region[1][0], :])

        xmin = self.validate_x_index(np.nanmin(region[0]), clip=False)
        xmax = self.validate_y_index(np.nanmax(region[0]), clip=False) + 1
        ymin = self.validate_x_index(np.nanmin(region[1]), clip=False)
        ymax = self.validate_y_index(np.nanmax(region[1]), clip=False) + 1

        if (xmax - xmin)  * (ymax - ymin) > SIZE_LIMIT:
            raise StandardError('size limit exceeded, try a smaller region')

        quadrant = self.get_data(xmin, xmax, ymin, ymax, 0, self.dimz)
        xpix = np.copy(region[0])
        ypix = np.copy(region[1])
        xpix -= xmin
        ypix -= ymin

        out = list()
        for i in range(len(xpix)):
            ix = xpix[i]
            iy = ypix[i]
            out.append(quadrant[ix, iy, :])

        return np.array(out)
        
    def has_dataset(self, path):
        """Check if a dataset is present"""
        f = self.open_hdf5()
        if path not in f: return False
        else: return True
    
    def get_dataset(self, path, protect=True):
        """Return a dataset (but not 'data', instead use get_data).

        :param path: dataset path

        :param protect: (Optional) check if dataset is protected
          (default True).
        """
        if protect:
            for iprot_path in self.protected_datasets:
                if path in iprot_path:
                    raise IOError('dataset {} is protected. please use the corresponding higher level method (something like set_{} should do)'.format(path, path))

        if path == 'data':
            raise ValueError('to get data please use your cube as a classic 3d numpy array. e.g. arr = cube[:,:,:].')
        f = self.open_hdf5()
        if path not in f:
            raise AttributeError('{} dataset not in the hdf5 file'.format(path))
        return f[path][:]


    def get_dataset_attrs(self, path):
        """Return the attributes attached to a dataset
        """
        f = self.open_hdf5()
        if path not in f:
            raise AttributeError('{} dataset not in the hdf5 file'.format(path))
        return f[path].attrs


    def get_datasets(self):
        """Return all datasets contained in the cube
        """
        ds = list()
        f = self.open_hdf5()
        for path in f:
            ds.append(path)
        return ds
        
    def has_same_2D_size(self, cube_test):
        """Check if another cube has the same dimensions along x and y
        axes.

        :param cube_test: Cube to check
        """
        
        if ((cube_test.dimx == self.dimx) and (cube_test.dimy == self.dimy)):
            return True
        else:
            return False

    def get_data_frame(self, index):
        """Return one frame of the cube.

        :param index: Index of the frame to be returned
        """
        return self[:,:,index]

    def get_all_data(self):
        """Return the whole data cube"""
        return self[:,:,:]

    def get_binned_cube(self, binning):
        """Return the binned version of the cube
        """
        binning = int(binning)
        if binning < 2:
            raise ValueError('Bad binning value')
        logging.info('Binning interferogram cube')
        image0_bin = utils.image.nanbin_image(
            self.get_data_frame(0), binning)

        cube_bin = np.empty((image0_bin.shape[0],
                             image0_bin.shape[1],
                             self.dimz), dtype=float)
        cube_bin.fill(np.nan)
        cube_bin[:,:,0] = image0_bin
        progress = core.ProgressBar(self.dimz-1)
        for ik in range(1, self.dimz):
            progress.update(ik, info='Binning cube')
            cube_bin[:,:,ik] = utils.image.nanbin_image(
                self.get_data_frame(ik), binning)
        progress.end()
        return cube_bin

    def get_resized_data(self, size_x, size_y):
        """Resize the data cube and return it using spline
          interpolation.
          
        This function is used only to resize a cube of flat or dark
        frames. Note that resizing dark or flat frames must be
        avoided.

        :param size_x: New size of the cube along x axis
        :param size_y: New size of the cube along y axis
        
        .. warning:: This function must not be used to resize images
          containing star-like objects (a linear interpolation must
          be done in this case).
        """
        resized_cube = np.empty((size_x, size_y, self.dimz), dtype=self.dtype)
        x = np.arange(self.dimx)
        y = np.arange(self.dimy)
        x_new = np.linspace(0, self.dimx, num=size_x)
        y_new = np.linspace(0, self.dimy, num=size_y)
        progress = core.ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            z = self.get_data_frame(_ik)
            interp = scipy.interpolate.RectBivariateSpline(x, y, z)
            resized_cube[:,:,_ik] = interp(x_new, y_new)
            progress.update(_ik, info="resizing cube")
        progress.end()
        data = np.array(resized_cube)
        dimx = data.shape[0]
        dimy = data.shape[1]
        dimz = data.shape[2]
        logging.info("Data resized to shape : ({}, {}, {})".format(dimx, dimy,dimz))
        return data

    def get_gain(self):
        """Return camera gain value
        """
        if self.params.camera == 0:
            gain = (self.config['CAM1_GAIN'] + self.config['CAM1_GAIN']) / 2
        elif self.params.camera == 1:
            gain = self.config['CAM1_GAIN']
        elif self.params.camera == 2:
            gain = self.config['CAM2_GAIN']
        else:
            raise ValueError('camera parameter must be 0, 1 or 2')
        return gain

    
    def get_deep_frame(self, recompute=False):
        """Return the deep frame of a cube.

        :param recompute: (Optional) Force to recompute deep frame
          even if it is already present in the cube (default False).
        
        .. note:: In this process NaNs are handled correctly
        """
        try:
            return image.Image(self.deep_frame, params=self.params)
        except AttributeError: pass

        gain = self.get_gain()
        
        df = None
        if not recompute:
            if self.has_dataset('deep_frame'):
                df = self.get_dataset('deep_frame', protect=False)
        
            elif self.is_old:
                df = self.oldcube.get_mean_image(recompute=recompute) * self.dimz / gain

        if df is None:
            df = self.compute_sum_image() / gain

        self.deep_frame = df
        return image.Image(self.deep_frame, params=self.params)
    
    def get_phase_maps(self):
        """Return a PhaseMaps instance if phase maps are set
        """
        try:
            return fft.PhaseMaps(self.cube_path)
        except StandardError:
            return None

    def get_high_order_phase(self):
        """Return high order phase.

        This is just the config high order phase
        """
        return fft.Phase(self._get_phase_file_path(self.params.filter_name))
        

    def compute_sum_image(self):
        """compute the sum along z axis
        """
        SIZE = 30
        sum_im = None
        progress = core.ProgressBar(self.dimz)
        for ik in range(0, self.dimz, SIZE):
            frames = self[:,:,ik:ik+SIZE]
            if sum_im is None: # avoid creating a zeros frame with a
                               # possibly uncompatible dtype
                sum_im = np.nansum(frames, axis=2)
            else:
                sum_im += np.nansum(frames, axis=2)
            progress.update(ik, info="Creating sum image")
        progress.end()
        return sum_im


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
            div_nb = self.config.DIV_NB

        if dimx is None: dimx = self.dimx
        if dimy is None: dimy = self.dimy

        return core.Tools._get_quadrant_dims(
            self, quad_number, dimx, dimy, div_nb)

    def writeto(self, path):
        """Write data to an hdf file

        :param path: hdf file path.
        """
        old_keys = 'quad', 'frame'
        def is_exportable(key):
            for iold in old_keys:
                if iold in key:
                    return False
            return True

        with utils.io.open_hdf5(path, 'w') as fout:
            fout.attrs['level2'] = True
            
            f = self.open_hdf5()
            for iattr in f.attrs:
                fout.attrs[iattr] = f.attrs[iattr]

            # write datasets
            for ikey in f:
                if is_exportable(ikey):
                    logging.info('adding {}'.format(ikey))
                    fout.create_dataset(ikey, data=f[ikey], chunks=True)
                    
            # write data
            dtype = self[0,0,0].dtype
            fout.create_dataset('data', shape=self.shape, dtype=dtype, chunks=True)

            DIV_NB = 6
            QUAD_NB = DIV_NB**2
            
            for iquad in range(QUAD_NB):
                xmin, xmax, ymin, ymax = self.get_quadrant_dims(iquad, div_nb=DIV_NB)
                logging.info('loading quad {}/{}'.format(
                    iquad, QUAD_NB))

                data = self.get_data(
                    xmin, xmax, ymin, ymax, 0, self.dimz, silent=False)
                
                logging.info('writing quad {}/{}'.format(
                    iquad, QUAD_NB))
                fout['data'][xmin:xmax, ymin:ymax, :] = data
        

    def crop(self, path, cx, cy, size):
        """Extract a part of the file and write it to a new hdf file

        WCS and most datasets will be croped also to try to keep a valid cube

        :param cx: X center position

        :param xy: Y center position

        :param size: Size of the cropped rectangle. A tuple (sz,
          sy). Can be single int in which case the cropped data is a
          box.

        .. warning:: size of the returned box is not guaranteed if cx
          and cy are on the border of the image.
        """
        # a fake image is used to compute the cropped wcs
        df = image.Image(np.empty((self.dimx, self.dimy), dtype=float),
                         params=self.params)
        df = df.crop(cx, cy, size)
        xmin, xmax, ymin, ymax = df.params.cropped_bbox
        outcube = RWHDFCube(path, shape=(df.dimx, df.dimy, self.dimz),
                            instrument=self.instrument, reset=True, params=df.params)
        logging.info('writing cube')
        data = self[xmin:xmax, ymin:ymax, :]
        outcube[:,:,:] = data
        logging.info('cropped cube written at {}'.format(path))
                
    def to_fits(self, path):
        """write data to a FITS file. 

        Note that most of the information will be lost in the
        process. The only output guaranteed format is hdf5 (usr
        writeto() method instead)

        :param path: Path to the FITS file
        """
        raise NotImplementedError()
                
    def get_frame_header(self, index):
        """Return the header of a frame given its index in the list.

        The header is returned as an instance of pyfits.Header().

        :param index: Index of the frame
        """
        return utils.io.header_hdf52fits(
            self.get_dataset('frame_header_{}'.format(index)))
    
    def get_calibration_laser_map(self):
        """Return calibration laser map"""
        try:
            return np.copy(self.calibration_laser_map)
        except AttributeError:
            if self.has_dataset('calib_map'):
                self.calibration_laser_map = self.get_dataset('calib_map', protect=False)
            else:
                if 'calibration_laser_map_path' not in self.params:
                    raise StandardError("no calibration laser map in the hdf file. 'calibration_laser_map_path' must be set in params")
                calib_path = self.params.calibration_laser_map_path
                if not os.path.exists(calib_path):
                    if not os.path.isabs(calib_path):
                        calib_path = os.path.join(os.path.dirname(self.hdffile.filename), calib_path)
                
                self.calibration_laser_map = utils.io.read_fits(calib_path)
                if 'cropped_bbox' in self.params:
                    xmin, xmax, ymin, ymax = self.params.cropped_bbox
                    self.calibration_laser_map = self.calibration_laser_map[xmin:xmax, ymin:ymax]
                    
        if (self.calibration_laser_map.shape[0] != self.dimx):
            self.calibration_laser_map = utils.image.interpolate_map(
                self.calibration_laser_map, self.dimx, self.dimy)
        if not self.calibration_laser_map.shape == (self.dimx, self.dimy):
            raise StandardError('Calibration laser map shape is {} and must be ({} {})'.format(self.calibration_laser_map.shape[0], self.dimx, self.dimy))
        return self.calibration_laser_map
        
    def get_calibration_coeff_map(self):
        """Return calibration laser map"""
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

    def validate_x_index(self, x, clip=True):
        """validate an x index, return an integer inside the boundaries or
        raise an exception if it is off boundaries

        :param x: x index

        :param clip: (Optional) If True return an index inside the
          boundaries, else: raise an exception (default True).
        """
        return utils.validate.index(x, 0, self.dimx, clip=clip)
    
    def validate_y_index(self, y, clip=True):
        """validate an y index, return an integer inside the boundaries or
        raise an exception if it is off boundaries

        :param y: y index (can be an array or a list of indexes)

        :param clip: (Optional) If True return an index inside the
          boundaries, else: raise an exception (default True).
        """
        return utils.validate.index(y, 0, self.dimy, clip=clip)


    def get_master_frame(self, combine=None, reject=None):

        """Combine frames along z to create a master frame.
        
        :param reject: (Optional) Rejection operation for master
          frames creation. Can be 'sigclip', 'minmax', 'avsigclip' or
          None (default 'avsigclip'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        
        :param combine: (Optional) Combining operation for master
          frames creation. Can be 'average' or 'median' (default
          'average'). See
          :py:meth:`orb.utils.image.create_master_frame`.
        """

        if self.dimz > 25:
            raise StandardError('master combination is useful for a small set of frames')
        
        if combine is None:
            combine='average'
        if reject is None:
            reject='avsigclip'

        if not self.config.BIG_DATA:
            master = utils.image.create_master_frame(
                self[:,:,:], combine=combine, reject=reject)
        else:
            master = utils.image.pp_create_master_frame(
                self[:,:,:], combine=combine, reject=reject,
                ncpus=self.config.NCPUS)

        return master


    def get_axis(self, x, y):
        """Return the axis at x, y"""
        return None

    def get_zvector(self, x, y, r=0, return_region=False):
        """Return an orb.fft.Vector1d instance taken at a given position in x, y.
        
        :param x: x position 
        
        :param y: y position 

        :param r: (Optional) If r > 0, vector is integrated over a
          circular aperture of radius r. In this case the number of
          pixels is returned as a parameter: pixels

        :param return_region: (Optional) If True, region is returned
          also (default False)

        """
        x = self.validate_x_index(x, clip=False)
        y = self.validate_y_index(y, clip=False)

        region = self.get_region('circle({},{},{})'.format(x+1, y+1, r))
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[region])
        interfs = self.get_data_from_region(region)
        interf = np.nansum(interfs, axis=0)
        params = dict(self.params)
        params['pixels'] = len(interfs)

        vec = core.Vector1d(interf, params=params,
                            zpd_index=self.params.zpd_index,
                            calib_coeff=calib_coeff,
                            axis=np.arange(self.dimz))
        if not return_region:
            return vec
        else:
            return vec, region


    def get_airmass(self):
        """Return the airmass"""
        if self.has_param('airmass'):
            if np.size(self.params.airmass) == self.dimz:
                return self.params.airmass
            elif not self.has_dataset('frame_header_0'):
                logging.debug('airmass size is {} but cube dimz is {}. no frame header present so the airmass is returned as is'.format(np.size(self.params.airmass), self.dimz))
                return self.params.airmass
        elif not self.has_dataset('frame_header_0'):
            raise StandardError('airmass not set and no frame headers')

        airmass = list()
        for i in range(self.dimz):
            try:
                airmass.append(float(self.get_frame_header(i)['AIRMASS']))
            except KeyError:
                raise StandardError('frame_header_{} not present'.format(i))
        self.params['airmass'] = np.array(airmass)
        return self.params.airmass

    def detect_stars(self, **kwargs):
        """Detect valid stars in the image

        :param kwargs: image.Image.detect_stars kwargs.
        """
        if self.has_dataset('deep_frame'):
            logging.debug('detecting stars using the deep frame')
            df = self.get_deep_frame().data
        else:
            logging.debug('detecting stars using a stack of the first {}'.format(
                self.config.DETECT_STACK))
            _stack = self[:,:,:self.config.DETECT_STACK]
            df = np.nanmedian(_stack, axis=2)
        df = image.Image(df, params=self.params)
        return df.detect_stars(**kwargs)

    def fit_stars_in_frame(self, star_list, index, **kwargs):
        """Fit stars in frame

        :param star_list: Path to a list of stars

        :param index: frame index

        :param kwargs: image.Image.fit_stars kwargs.
        """
        im = image.Image(self[:,:,index], params=self.params)
        return im.fit_stars(star_list, **kwargs)

    def fit_stars_in_cube(self, star_list,
                          correct_alignment=False,
                          alignment_vectors=None,
                          add_cube=None, **kwargs):
        
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

        :param alignment_vectors: (Optional) If not None, must be a
          tuple of 2 vectors (dx, dy), each one having the same length
          as the numebr of frames in the cube. It is used as a guess
          for the alignement of the stars.

        :param add_cube: (Optional) A tuple [Cube instance,
          coeff]. This cube is added to the data before the fit so
          that the fitted data is self.data[] + coeff * Cube[].

        :param kwargs: (Optional) utils.astrometry.fit_stars_in_frame
          kwargs.

        """
        def _fit_stars_in_frame(frame, star_list, fwhm_pix, params, kwargs):

            import warnings
            warnings.simplefilter('ignore')
            
            im = orb.image.Image(frame, params=params)
            if fwhm_pix is not None:
                im.reset_fwhm_pix(fwhm_pix)
            return im.fit_stars(star_list, **kwargs)

        FOLLOW_NB = 5 # Number of deviation value to get to follow the
                      # stars

        if alignment_vectors is not None:
            if len(alignment_vectors) != 2:
                raise TypeError('alignment_vectors must be a tuple of 2 vectors')
            if len(alignment_vectors[0]) != self.dimz or len(alignment_vectors[1]) != self.dimz:
                raise TypeError('each vector of alignment_vectors must have the same len as the number of frames in the cube ({} or {} != {})'.format(len(alignment_vectors[0]), len(alignment_vectors[1]), self.dimz))
            
            correct_alignment = False
                
        star_list = utils.astrometry.load_star_list(star_list)
                
        logging.info("Fitting stars in cube")

        fit_results = list([None] * self.dimz)
        dx_mean = list([np.nan] * self.dimz)
        dy_mean = list([np.nan] * self.dimz)
        fwhm_mean = list([np.nan] * self.dimz)
        
        if self.dimz < 2: raise StandardError(
            "Data must have 3 dimensions. Use fit_stars_in_frame method instead")
        
        if add_cube is not None:
            if np.size(add_cube) >= 2:
                added_cube = add_cube[0]
                added_cube_scale = add_cube[1]
                if not isinstance(added_cube, Cube):
                    raise StandardError('Added cube must be a Cube instance. Check add_cube option')
                if np.size(added_cube_scale) != 1:
                    raise StandardError('Bad added cube scale. Check add_cube option.')

        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()
        
        progress = core.ProgressBar(int(self.dimz))
        x_corr = 0.
        y_corr = 0.
        for ik in range(0, self.dimz, ncpus):
            # no more jobs than frames to compute
            if (ik + ncpus >= self.dimz):
                ncpus = self.dimz - ik
    
            if correct_alignment:
                if ik > 0:
                    old_x_corr = float(x_corr)
                    old_y_corr = float(y_corr)

                    if ik > FOLLOW_NB - 1:
                        # try to get the mean deviation over the
                        # last fitted frames
                        x_corr = np.nanmedian(dx_mean[ik-FOLLOW_NB:ik])
                        y_corr = np.nanmedian(dy_mean[ik-FOLLOW_NB:ik])
                    else:
                        x_corr = np.nan
                        y_corr = np.nan
                        if dx_mean[-1] is not None:
                            x_corr = dx_mean[ik-1]
                        if dy_mean[-1] is not None:
                            y_corr = dy_mean[ik-1]
                                       
                    if np.isnan(x_corr):
                        x_corr = float(old_x_corr)
                    if np.isnan(y_corr):
                        y_corr = float(old_y_corr)
                    
                star_list[:,0] += x_corr
                star_list[:,1] += y_corr

            if alignment_vectors is None:
                star_lists = list([np.array(star_list)]) * ncpus
            else:
                star_lists = list()
                for ijob in range(ncpus):
                    istar_list = np.copy(star_list)
                    istar_list[:,0] += alignment_vectors[0][ik+ijob]
                    istar_list[:,1] += alignment_vectors[1][ik+ijob]
                    star_lists.append(istar_list)                    
    
                
            # follow FWHM variations
            fwhm_pix = None
            if ik > FOLLOW_NB - 1:
                fwhm_pix = np.nanmean(utils.stats.sigmacut(fwhm_mean[ik-FOLLOW_NB:ik]))
                if np.isnan(fwhm_pix): fwhm_pix = None
          
            # load data
            progress.update(ik, info="loading: " + str(ik))
            frames = self[:,:,ik:ik+ncpus]
            if add_cube is not None:
                frames += added_cube[:,:,ik:ik+ncpus] * added_cube_scale
                
            # get stars photometry for each frame
            params = self.params.convert()
            progress.update(ik, info="computing photometry: " + str(ik))
            jobs = [(ijob, job_server.submit(
                _fit_stars_in_frame,
                args=(frames[:,:,ijob], star_lists[ijob], fwhm_pix, params, kwargs),
                modules=("import logging",
                         "import orb.utils.stats",
                         "import orb.utils.image",
                         'import orb.image',
                         "import numpy as np",
                         "import math",
                         "import orb.cutils",
                         "import bottleneck as bn",
                         "import warnings",
                         "from orb.utils.astrometry import *")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                res = job()
                fit_results[ik+ijob] = res
                if res is not None:
                    if not res.empty:
                        if 'dx' in res and 'dy' in res:
                            dx_mean[ik+ijob] = np.nanmean(utils.stats.sigmacut(res['dx'].values))
                            dy_mean[ik+ijob] = np.nanmean(utils.stats.sigmacut(res['dy'].values))
                        if 'fwhm_pix' in res:
                            fwhm_mean[ik+ijob] = np.nanmean(utils.stats.sigmacut(res['fwhm_pix'].values))
            
            
        self._close_pp_server(job_server)
        
        progress.end()
  
        return fit_results

    def get_raw_alignement_vectors(self, star_number=60, searched_size=80):
        """Return raw alignment vectors based on brute force.

        Slow but useful when alignment between frames is very bad.

        :return: dx, dy vectors
        """
        star_list_init, fwhm = image.Image(
            self[:,:,0], instrument=self.instrument).detect_stars(
                min_star_number=star_number)
        star_list_init = utils.astrometry.df2list(star_list_init)
        dxs = list()
        dys = list()
        progress = core.ProgressBar(self.dimz-1)
        dxs.append(0)
        dys.append(0)
        for iframe in range(1, self.dimz):
            x_range = np.linspace(-searched_size,searched_size,searched_size)
            y_range = np.linspace(-searched_size,searched_size,searched_size)
            r_range = [0]
            dx, dy, _, _ = utils.astrometry.brute_force_guess(
                self[:,:,iframe], star_list_init, x_range, y_range,r_range,
                [self.dimx/2., self.dimy/2.], 1, 15, verbose=False)
            
            progress.update(iframe-1, info='{:.2f}, {:.2f}'.format(dx, dy))
            dxs.append(dx)
            dys.append(dy)
        progress.end()
        return np.array(dxs), np.array(dys)
            
    
    def get_alignment_vectors(self, star_list, min_coeff=0.2,
                              alignment_vectors=None):
        """Return alignement vectors

        :param star_list: list of stars

        :param min_coeff: The minimum proportion of stars correctly
            fitted to assume a good enough calculated disalignment
            (default 0.2).

        :return: alignment_vector_x, alignment_vector_y, alignment_error
        """
        star_list = utils.astrometry.load_star_list(star_list)
        fit_results = self.fit_stars_in_cube(star_list, correct_alignment=True,
                                             alignment_vectors=alignment_vectors,
                                             no_aperture_photometry=True,
                                             multi_fit=False, # sure ???
                                             fix_height=False)
        return utils.astrometry.compute_alignment_vectors(fit_results)
    

#################################################
#### CLASS RWHDFCube ############################
#################################################
class RWHDFCube(HDFCube):

    def __init__(self, path, shape=None, instrument=None, reset=False, **kwargs):
        """:param path: Path to an HDF5 cube

        :param shape: (Optional) Must be set to something else than
          None to create an empty file. It the file already exists,
          shape must be set to None.

        :param reset: (Optional) If True and if a file already exists,
          it is deleted before being created again.

        :param kwargs: (Optional) :py:class:`~orb.core.HDFCube` kwargs.

        """
        # reset
        if reset:
            if os.path.exists(path):
                os.remove(path)

        self.empty = False
        # create file if it does not exists
        if not os.path.exists(path):
            if shape is None:
                raise ValueError('cube does not exist. If you want to create one, shape must be set.')
            with utils.io.open_hdf5(path, 'w') as f:
                utils.validate.has_len(shape, 3, object_name='shape')
                f.create_dataset('data', shape=shape, chunks=True)
                self.empty = True
                f.attrs['level2'] = True
                f.attrs['instrument'] = instrument

        elif shape is not None:
            raise ValueError('shape must be set only when creating a new HDFCube')

        HDFCube.__init__(self, path, instrument=instrument, **kwargs)
        if self.is_old: raise StandardError('Old cubes are not writable. Please export the old cube to a new cube with writeto()')

        # reopening in rw mode            
        if self.hdffile is not None:
            _dataname = self.data.name
            del self.hdffile
            del self.data
            self.set_writeable(True)
            self.hdffile = self.open_hdf5('a')
            self.data = self.hdffile[_dataname]

        if self.has_params:
            self.set_params(self.params)

    def __setitem__(self, key, value):
        """Implement setitem special method"""        
        # decrease representation in case of complex or floats to
        # minimize data size
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.dtype == np.float64:
            value = value.astype(np.float32)
        elif value.dtype == np.complex128:
            value = value.astype(np.complex64)

        f = self.open_hdf5('a')
        if self.is_empty():
            del f['data']
            f.create_dataset('data', shape=self.shape, chunks=True, dtype=value.dtype)
            self.empty = False

        f['data'].__setitem__(key, value)

    def open_hdf5(self, mode=None):
        """Return a handle on the hdf5 file."""
        self.set_writeable(True)
        return HDFCube.open_hdf5(self, mode=mode)


    def is_empty(self):
        """return True if data is empty
        """
        return bool(self.empty)
        
    def set_param(self, key, value):
        """Set class parameter

        :param key: parameter key

        :param value: parameter value
        """
        self.params[key] = value
        f = self.open_hdf5('a')
        _update = True
        value = utils.io.cast2hdf5(value)
        if key in f.attrs:
            if np.all(f.attrs[key] == value):
                _update = False
        if _update:
            f.attrs[key] = value

    def set_params(self, params):
        """Set class parameters

        :param params: a dict of parameters
        """
        for ipar in params:
            try:
                self.set_param(ipar, params[ipar])
            except TypeError:
                warnings.warn('error setting param {}'.format(ipar))

    def set_mask(self, data):
        """Set mask

        A mask must have the shape of the data but for 3d data which
        has a 2d mask (self.dimx, self.dimy). A Zero indicates a pixel
        which should be masked (Nans are returned for this pixel).

        :param data: mask. Must be a boolean array
        """
        HDFCube.set_mask(self, data)
        self.set_dataset('mask', data, protect=False)
            
    def set_dataset(self, path, data, protect=True):
        """Write a dataset to the hdf5 file

        :param path: dataset path

        :param data: data to write.

        :param protect: (Optional) check if dataset is protected
          (default True).
        """
        if protect:
            if path in self.protected_datasets:
                raise IOError('dataset {} is protected. please use the corresponding higher level method (something like set_{} should do)'.format(path, path))

        if path == 'data':
            raise ValueError('to set data please use your cube as a classic 3d numpy array. e.g. cube[:,:,:] = value.')
        f = self.open_hdf5('a')
        if path in f:
            del f[path]
            warnings.warn('{} dataset changed'.format(path))

        if isinstance(data, dict):
            data = utils.io.dict2array(data)
        f.create_dataset(path, data=data, chunks=True)

    def set_deep_frame(self, deep_frame):
        """Append a deep frame to the HDF5 cube.

        :param deep_frame: Deep frame to append.
        """
        if deep_frame.shape != (self.dimx, self.dimy):
            raise TypeError('deep frame must have shape ({}, {})'.format(self.dimx, self.dimy))
        
            
        self.set_dataset('deep_frame', deep_frame, protect=False)


    def set_frame_header(self, index, header):
        """Set the header of a frame.

        The header must be an instance of pyfits.Header().

        :param index: Index of the frame

        :param header: Header as a pyfits.Header instance.
        """
        self.set_dataset('frame_header_{}'.format(index),
                         utils.io.header_fits2hdf5(header))

    def set_calibration_laser_map(self, calib_map):
        """Set calibration map.

        :param calib_map: Calibration map
        """
        if calib_map.shape != (self.dimx, self.dimy):
            raise TypeError('calib_map must have shape ({}, {})'.format(self.dimx, self.dimy))

        self.set_dataset('calib_map', calib_map, protect=False)

    def set_phase_maps(self, phase_maps):
        """Set phase maps from a PhaseMaps instance
        """
        check_params = ['order', 'step', 'filter_name']

        if not isinstance(phase_maps, fft.PhaseMaps):
            raise TypeError('phase_maps must be a PhaseMaps instance')

        for ipar in check_params:
            if phase_maps.params[ipar] != self.params[ipar]:
                raise ValueError('parameter {} in phase_maps is {}, but is {} for this cube'.format(
                    ipar, phase_maps.params[ipar], self.params[ipar]))
        
        for i in range(len(phase_maps.phase_maps)):
            self.set_dataset('phase_map_{}'.format(int(i)),
                             phase_maps.phase_maps[i], protect=False)
            self.set_dataset('phase_map_err_{}'.format(int(i)),
                             phase_maps.phase_maps_err[i], protect=False)
            
        self.set_dataset('phase_maps_coeff_map',
                         phase_maps.calibration_coeff_map, protect=False)
        self.set_dataset('phase_maps_cm1_axis',
                         phase_maps.axis, protect=False)
        
        # #self.set_calibration_laser_map()
        # try:
        #     clm = self.get_calibration_laser_map()
        # except StandardError:
        #     clm = None
        # if clm is None:
        #     calibration_laser_map = (phase_maps.unbinned_calibration_coeff_map
        #                              * self.config.CALIB_NM_LASER)
        #     self.set_calibration_laser_map(calibration_laser_map)
        # else:
        #     self.set_calibration_laser_map(self.calibration_laser_map)
        #     warnings.warn('calibration laser map unchanged since it already exists')

    def set_standard_image(self, std_im):
        """Set standard image

        :param std_im: An image.Image instance or a path to an
          hdf5 image.

        """
        try:
            std_im = image.Image(std_im, instrument=self.instrument)
        except ValueError:
            raise TypeError('std_im must be an orb.image.Image instance or a path to valid hdf5 image')
        self.set_dataset('standard_image', std_im.data, protect=False)

        f = self.open_hdf5('a')
        for iparam in std_im.params:
            f['standard_image'].attrs[iparam] = std_im.params[iparam]
            
            
    def write_frame(self, index, data=None, header=None, section=None,
                    record_stats=False):
        """Write a frame. 

        This function is here for backward compatibility but a simple
        self[:,:,index] may be used instead.

        :param index: Index of the frame
        
        :param data: (Optional) Frame data (default None).
        
        :param header: (Optional) Frame header (default None).
                
        :param section: (Optional) If not None, must be a 4-tuple
          [xmin, xmax, ymin, ymax] giving the section to write instead
          of the whole frame. Useful to modify only a part of the
          frame (default None).

        :param record_stats: (Optional) If True, frame stats are
          recorder in its header (default False).
        """
        if section is not None:
            xmin, xmax, ymin, ymax = section
        else:
            xmin, xmax, ymin, ymax = 0, self.dimx, 0, self.dimy

        if data is not None:
            self[xmin:xmax, ymin:ymax, index] = data

        if record_stats:
            if header is None:
                header = dict()
            header['MEAN'] = np.nanmean(self[xmin:xmax, ymin:ymax, index].real)
            header['MEDIAN'] = np.nanmedian(self[xmin:xmax, ymin:ymax, index].real)
                        
        if header is not None:
            self.set_frame_header(index, header)


#################################################
#### CLASS Cube ################################
#################################################
class Cube(HDFCube):
    """Provide additional cube methods when observation parameters are known.
    """

    needed_params = ('step', 'order', 'filter_name', 'exposure_time',
                     'step_nb', 'zpd_index')

    
    def __init__(self, path, params=None, instrument=None, **kwargs):
        """Initialize Cube class.

        :param data: Path to an HDF5 Cube

        :param params: (Optional) Path to a dict.

        :param kwargs: Cube kwargs + other parameters not supplied in
          params (else overwrite parameters set in params)

        .. note:: params are first read from the HDF5 file
          itself. They can be modified through the params dictionary
          and the params dictionary can itself be modified with
          keyword arguments.
        """
        HDFCube.__init__(self, path, instrument=instrument, params=params, **kwargs)
        
        # compute additional parameters
        self.filterfile = core.FilterFile(self.params.filter_name)
        self.set_param('filter_name', self.filterfile.basic_path)
        self.set_param('filter_nm_min', self.filterfile.get_filter_bandpass()[0])
        self.set_param('filter_nm_max', self.filterfile.get_filter_bandpass()[1])
        self.set_param('filter_cm1_min', self.filterfile.get_filter_bandpass_cm1()[0])
        self.set_param('filter_cm1_max', self.filterfile.get_filter_bandpass_cm1()[1])
        
        self.params_defined = True

        self.validate()
            
        
    def validate(self):
        """Check if this class is valid"""
        if self.instrument not in ['sitelle', 'spiomm']:
            raise StandardError("class not valid: set instrument to 'sitelle' or 'spiomm' at init")
        else: self.set_param('instrument', self.instrument)
        try: self.params_defined
        except AttributeError: raise StandardError("class not valid: set params at init")
        if not self.params_defined: raise StandardError("class not valid: set params at init")

        # validate needed params
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise ValueError('parameter {} must be defined in params'.format(iparam))

                
    def get_axis_corr(self):
        """Return the reference wavenumber correction"""
        if self.has_param('axis_corr'):
            return float(self.params.axis_corr)
        else:
            calib_map = self.get_calibration_coeff_map()
            return calib_map[calib_map.shape[0]//2, calib_map.shape[1]//2]
                    
    def get_base_axis(self):
        """Return the spectral axis (in cm-1) at the center of the cube"""
        if not self.has_param('base_axis'):
            base_axis = utils.spectrum.create_cm1_axis(
                self.dimz, self.params.step, self.params.order,
                corr=self.get_axis_corr())
        else:
            base_axis = self.params.base_axis
            
        return core.Axis(base_axis)

    def get_uncalibrated_filter_bandpass(self):
        """Return filter bandpass as two 2d matrices (min, max) in pixels"""
        filterfile = FilterFile(self.get_param('filter_name'))
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
    

        
        
#################################################
#### CLASS InteferogramCube #####################
#################################################
class InterferogramCube(Cube):
    """Provide additional methods for an interferogram cube when
    observation parameters are known.
    """

    def get_interferogram(self, x, y, r=0):
        """Return an orb.fft.Interferogram instance.

        See Cube.get_zvector for the parameters.        
        """
        vector, region = Cube.get_zvector(self, x, y, r=r, return_region=True)
        vector.params['source_counts'] = np.nansum(self.get_deep_frame().data[region])

        vector.axis = None

        err = np.ones(self.dimz, dtype=float) * np.sqrt(
            vector.params.source_counts / self.dimz) * self.get_gain()
        
        return fft.RealInterferogram(vector, err=err)

    def get_phase(self, x, y):
        """If phase maps are set, return the phase at x, y
        """
        pm = self.get_phase_maps()
        if pm is None: return None 
        return pm.get_phase(x, y, unbin=True)

    def get_spectrum(self, x, y, r=0):
        """Return an orb.fft.Spectrum instance.

        phase_maps must be set to get a reliable spectrum

        See Cube.get_zvector for the parameters.        
        """
        interf = self.get_interferogram(x, y, r=r)
        spectrum = interf.get_spectrum()
        spectrum.correct_phase(self.get_phase(x, y))
        return spectrum

    def get_mean_interferogram(self, xmin, xmax, ymin, ymax):
        """Return mean interferogram in a box [xmin:xmax, ymin:ymax, :]
        along z axis
        
        :param xmin: min boundary along x axis
        :param xmax: max boundary along x axis
        :param ymin: min boundary along y axis
        :param ymax: max boundary along y axis
        """
        self.validate()
        xmin, xmax = np.sort(self.validate_x_index([xmin, xmax], clip=False))
        ymin, ymax = np.sort(self.validate_y_index([ymin, ymax], clip=False))

        if xmin == xmax or ymin == ymax:
            raise ValueError('Boundaries badly defined, please check xmin, xmax, ymin, ymax')
        
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[xmin:xmax, ymin:ymax])
        interf = np.nanmean(np.nanmean(self[xmin:xmax, ymin:ymax, :], axis=0), axis=0)
        return fft.RealInterferogram(interf, params=self.params,
                                     zpd_index=self.params.zpd_index,
                                     calib_coeff=calib_coeff)

    
#################################################
#### CLASS FDCube ###############################
#################################################
class FDCube(core.Tools):
    """Basic handling class for a set of frames grouped into one virtual
    cube.

    This is a basic class which is mainly used to export data into an
    hdf5 cube.
    """

    def __init__(self, image_list_path, image_mode='classic',
                 chip_index=1, no_sort=False, silent_init=False, 
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

        :param params: Path to an option file or dictionary
          containting observation parameters.

        :param no_sort: (Optional) If True, no sort of the file list
          is done. Files list is taken as is (default False).

        :param silent_init: (Optional) If True no message is displayed
          at initialization.

        :param kwargs: (Optional) :py:class:`~orb.core.Cube` kwargs.
        """
        core.Tools.__init__(self, **kwargs)

        self.image_list_path = image_list_path

        self._image_mode = image_mode
        self._chip_index = chip_index

        if (self.image_list_path != ""):
            # read image list and get cube dimensions  
            image_list_file = utils.io.open_file(self.image_list_path, "r")
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
                            
                    elif not spiomm_bias_frame:
                        self.image_list = [image_name]

                        image_data = utils.io.read_fits(
                            image_name,
                            image_mode=self._image_mode,
                            chip_index=self._chip_index)
                        self.dimx = image_data.shape[0]
                        self.dimy = image_data.shape[1]
                        
                        is_first_image = False
                            
                elif not spiomm_bias_frame:
                    self.image_list.append(image_name)

            image_list_file.close()

            # image list is sorted
            if not no_sort:
                self.image_list = utils.misc.sort_image_list(self.image_list,
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
            

    def __getitem__(self, key):
        """Implement the evaluation of self[key].
        
        .. note:: To make this function silent just set
          Cube()._silent_load to True.
        """        
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
                progress = core.ProgressBar(z_slice.stop - z_slice.start - 1L)
            
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
                progress = core.ProgressBar(z_slice.stop - z_slice.start - 1L)
            
            for ik in range(z_slice.start + 1L, z_slice.stop):

                added_data = self._get_frame_section(x_slice, y_slice, ik)

                data = np.dstack((data, added_data))
                if not self._silent_load and not (ik - z_slice.start)%500:
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
        hdu = utils.io.read_fits(self.image_list[frame_index],
                                 return_hdu_only=True)
        image = None
        stored_file_path = None

        if self._image_mode == 'sitelle': 
            if image is None:
                image = utils.io.read_sitelle_chip(hdu, self._chip_index)
            section = image[x_slice, y_slice]

        elif self._image_mode == 'spiomm': 
            if image is None:
                image, header = utils.io.read_spiomm_data(
                    hdu, self.image_list[frame_index])
            section = image[x_slice, y_slice]

        else:
            if image is None:
                section = np.copy(
                    hdu[0].section[y_slice, x_slice].transpose())
            else: 
                section = image[y_slice, x_slice].transpose()
        del hdu

         # FITS only
        if stored_file_path is not None and image is not None:
            utils.io.write_fits(stored_file_path, image, overwrite=True,
                            silent=True)

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
        hdu = utils.io.read_fits(self.image_list[index],
                             return_hdu_only=True)
        hdu.verify('silentfix')
        return hdu[0].header

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


    def export(self, export_path, mask=None, params=None):
        """Export FDCube as an hdf5 cube

        :param export_path: Export path

        :param mask: (Optional) A boolean array of shape (self.dimx,
          self.dimy) which zeros indicates bad pixels to be replaced
          with Nans when reading data (default None).

        :param params: (Optional) A dict of parameters that will be
          added to the exported cube.
        """
        cube = RWHDFCube(
            export_path, shape=(self.dimx, self.dimy, self.dimz),
            instrument=self.instrument)

        if mask is not None:
            cube.set_mask(mask)

        if params is not None:
            cube.update_params(params)

        cube.set_header(self.get_cube_header())

        progress = core.ProgressBar(self.dimz)
        for iframe in range(self.dimz):
            progress.update(iframe, info='writing frame {}/{}'.format(
                iframe + 1, self.dimz))
            cube[:,:,iframe] = self[:,:,iframe]
            cube.set_frame_header(iframe, self.get_frame_header(iframe))
        progress.end()
            
#################################################
#### CLASS SpectralCube #########################
#################################################
class SpectralCube(Cube):
    """Provide additional methods for a spectral cube when observation
    parameters are known.

    """
    def __init__(self, *args, **kwargs):
        """Init class"""
        Cube.__init__(self, *args, **kwargs)

        self.reset_params()
        js, ncpus = self._init_pp_server(silent=True)
        self._close_pp_server(js)
        self.set_param('ncpus', int(ncpus))
        self.validate()

        logging.info('shape: {}'.format(self.shape))
        logging.info('wavenumber calibration: {}'.format(self.has_wavenumber_calibration()))
        logging.info('flux calibration: {}'.format(self.has_flux_calibration()))
        logging.info('wcs calibration: {}'.format(self.has_wcs_calibration()))

    def get_filter_range(self):
        """Return the range of the filter in the unit of the spectral
        cube as a tuple (min, max)"""
        if 'filter_range' not in self.params:
            self.params.reset(
                'filter_range', self.filterfile.get_filter_bandpass_cm1())

        return self.params.filter_range


    def get_filter_range_pix(self, xy=None, border_ratio=0.):
        """Return the range of the filter in channel index as a tuple
        (min, max)"""
        if xy is None:
            if not self.has_wavenumber_calibration():
                raise StandardError('if the spectral cube is not calibrated, xy must be provided')
            xy = [self.dimx//2, self.dimy//2]

        return utils.spectrum.cm12pix(
            self.get_axis(xy[0], xy[1]), self.get_filter_range())

    def get_axis(self, x, y):
        """Return the spectral axis at x, y
        """
        self.validate()
        utils.validate.index(x, 0, self.dimx, clip=False)
        utils.validate.index(y, 0, self.dimy, clip=False)
        
        axis = utils.spectrum.create_cm1_axis(
            self.dimz, self.params.step, self.params.order,
            corr=self.get_calibration_coeff_map()[x, y])
        return core.Axis(np.copy(axis))

    def has_wavenumber_calibration(self):
        """Return True if the cube is calibrated in wavenumber"""
        return bool(self.params.wavenumber_calibration)

    def has_flux_calibration(self):
        """Return True if the cube is calibrated in flux"""
        return bool(self.params.flux_calibration)

    def has_wcs_calibration(self):
        """Return True if the cube has valid wcs"""
        return bool(self.params.wcs_calibration)
    
    def reset_params(self):
        """Reset parameters"""

        self.filterfile = core.FilterFile(self.params.filter_name)

        self.set_param('flux_calibration', True)
        
        if not self.has_param('flambda'):
            logging.debug('flambda keyword not in cube header.')
            self.set_param('flux_calibration', False)
            self.set_param('flambda', 1.)
            
        if not self.has_param('apodization'):
            logging.debug('apodization unknown. automatically set to 1.')
            self.set_param('apodization', 1.)

        if not self.has_param('nm_laser'):
            logging.debug('nm_laser set to config value')
            self.set_param('nm_laser', self.config.CALIB_NM_LASER)

        if not self.has_param('wavetype'):
            logging.debug('wavetype unknown. set to wavenumber.')
            self.set_param('wavetype', 'WAVENUMBER')

        if not self.has_param('axis_corr'):
            if self.has_param('wavenumber_calibration'):
                if self.params.wavenumber_calibration:
                    raise StandardError('wavenumber_calibration is True but axis_corr is not set')
            logging.debug('axis_corr not set: cube is considered uncalibrated in wavenumber')
            self.set_param('wavenumber_calibration', False)
            self.set_param('axis_corr', self.get_axis_corr())
        else:
            self.set_param('wavenumber_calibration', True)

        step_nb = self.params.step_nb
        if step_nb != self.dimz:
            logging.debug('Malformed spectral cube. The number of steps in the header ({}) does not correspond to the real size of the data cube ({})'.format(step_nb, self.dimz))
            step_nb = int(self.dimz)
        self.set_param('step_nb', step_nb)

        if not self.has_param('zpd_index'):
            raise KeyError('ZPDINDEX not in cube header. Please run again the last step of ORBS reduction process.')

        # resolution
        resolution = utils.spectrum.compute_resolution(
            self.dimz - self.params.zpd_index,
            self.params.step, self.params.order,
            self.params.axis_corr)
        self.set_param('resolution', resolution)

        # incident angle of reference (in degrees)
        self.set_param(
            'theta_proj',
            utils.spectrum.corr2theta(self.params.axis_corr))

        # wavenumber
        if self.params.wavetype == 'WAVELENGTH':
            raise Exception('ORCS cannot handle wavelength cubes')
        
        self.params['wavenumber'] = True
        logging.debug('Cube is in WAVENUMBER (cm-1)')
        self.unit = 'cm-1'

        # check wcs calibration
        self.set_param('wcs_calibration', True)
        if self.params.wcs_rotation == self.config.WCS_ROTATION:
            self.set_param('wcs_calibration', False)

        # load hour_ut
        if not self.has_param('hour_ut'):
            self.params['hour_ut'] = np.array([0, 0, 0], dtype=float)

        # create base axis of the data
        self.set_param('base_axis', self.get_base_axis().data)

        self.set_param('axis_min', np.min(self.params.base_axis))
        self.set_param('axis_max', np.max(self.params.base_axis))
        self.set_param('axis_step', np.min(self.params.base_axis[1] - self.params.base_axis[0]))
        self.set_param('line_fwhm', utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index, self.params.step, self.params.order,
            apod_coeff=self.params.apodization,
            corr=self.params.axis_corr,
            wavenumber=self.params.wavenumber))
        self.set_param('filter_range', self.get_filter_range())

    def get_sky_velocity_map(self):
        if hasattr(self, 'sky_velocity_map'):
            return self.sky_velocity_map
        else: return None

    def get_calibration_laser_map_orig(self):
        """Return the original calibration laser map (not the version
        computed by :py:meth:`~HDFCube.get_calibration_laser_map`)"""
        return Cube.get_calibration_laser_map(self)

    def get_calibration_coeff_map_orig(self):
        """Return the original calibration coeff map (not the version
        computed by :py:meth:`~HDFCube.get_calibration_coeff_map`)"""
        return self.get_calibration_laser_map_orig() / self.params.nm_laser
        
    def get_calibration_laser_map(self):
        """Return the calibration laser map of the cube"""
        if hasattr(self, 'calibration_laser_map'):
            return self.calibration_laser_map

        calib_map = self.get_calibration_laser_map_orig()
        if calib_map is None:
            raise StandardError('No calibration laser map given. Please redo the last step of the data reduction')

        if self.has_wavenumber_calibration():
            calib_map = (np.ones((self.dimx, self.dimy), dtype=float)
                    * self.params.nm_laser * self.params.axis_corr)

        elif (calib_map.shape[0] != self.dimx):
            calib_map = utils.image.interpolate_map(
                calib_map, self.dimx, self.dimy)

        # calibration correction
        if self.get_sky_velocity_map() is not None:
            ratio = 1 + (self.get_sky_velocity_map() / constants.LIGHT_VEL_KMS)
            calib_map /= ratio

        self.calibration_laser_map = calib_map
        self.reset_calibration_coeff_map()
        return self.calibration_laser_map

    def get_calibration_coeff_map(self):
        """Return the calibration coeff map based on the calibration
        laser map and the laser wavelength.
        """
        if hasattr(self, 'calibration_coeff_map'):
            return self.calibration_coeff_map
        else:
            self.calibration_coeff_map = self.get_calibration_laser_map() / self.params.nm_laser
        return self.calibration_coeff_map


    def get_fwhm_map(self):
        """Return the theoretical FWHM map in cm-1 based only on the angle
        and the theoretical attained resolution."""
        return utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index, self.params.step, self.params.order,
            self.params.apodization, self.get_calibration_coeff_map_orig(),
            wavenumber=self.params.wavenumber)

    def get_theta_map(self):
        """Return the incident angle map (in degree)"""
        return utils.spectrum.corr2theta(self.get_calibration_coeff_map_orig())

    def reset_calibration_laser_map(self):
        """Reset the compute calibration laser map (and also the
        calibration coeff map). Must be called when the wavenumber
        calibration has changed

        ..seealso :: :py:meth:`~HDFCube.correct_wavelength`
        """
        if hasattr(self, 'calibration_laser_map'):
            del self.calibration_laser_map
        self.reset_calibration_coeff_map()

    def reset_calibration_coeff_map(self):
        """Reset the computed calibration coeff map alone"""
        if hasattr(self, 'calibration_coeff_map'):
            del self.calibration_coeff_map

    def get_sky_lines(self):
        """Return the wavenumber of the sky lines in the
        filter range"""
        return self.filterfile.get_sky_lines(self.dimz)

    def get_spectrum_from_region(self, region):
        """Return a list of spectra taken in a given region.

        :param region: A ds9-like region file or a list of pixels
          having the same format as the list returned by np.nonzero(),
          i.e. (x_positions_1d_array, y_positions_1d_array).

        .. note:: the region must not have a size greater than 400x400
          pixels. If you really need a larger region, you can split
          you region into smaller ones and combines the resulting
          spectra.
        """
        if not self.has_wavenumber_calibration():
            warnings.warn('spectral cube is not calibrated in wavenumber, a large region may result in a deformation of the ILS.')
            
        if isinstance(region, str):
            region = self.get_region(region)
        
        spectra = self.get_data_from_region(region)
        spectrum = np.nansum(spectra, axis=0)

        # calculate number of integrated pixels
        params = dict(self.params)
        params['pixels'] = len(spectra)

        # compute axis
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[region])
        axis = utils.spectrum.create_cm1_axis(
            self.dimz, self.params.step, self.params.order,
            corr=calib_coeff)
        
        # compute counts and err
        counts = np.nansum(self.get_deep_frame().data[region])
        err = np.ones(self.dimz, dtype=float) * np.sqrt(counts) * self.get_gain()
        err = core.Cm1Vector1d(err, axis, params=params)
        flambda = core.Cm1Vector1d(
            self.params.flambda, self.get_base_axis(), params=params)
        err = err.multiply(flambda)
        
        params['source_counts'] = counts

        return fft.RealSpectrum(spectrum, err=err.data, axis=axis, params=params)
                

    def get_spectrum(self, x, y, r=0):
        """Return a. orb.fft.RealSpectrum extracted at x, y and integrated
        over a circular aperture or radius r.

        :param x: x position 
        
        :param y: y position 

        :param r: (Optional) If r > 0, vector is integrated over a
          circular aperture of radius r. In this case the number of
          pixels is returned as a parameter: pixels
        """
        x = self.validate_x_index(x, clip=False)
        y = self.validate_y_index(y, clip=False)
        region = self.get_region('circle({},{},{})'.format(x+1, y+1, r))
        return self.get_spectrum_from_region(region)
        
    # def _extract_spectrum_from_region(self, region,
    #                                   subtract_spectrum=None,
    #                                   median=False,
    #                                   mean_flux=False,
    #                                   silent=False,
    #                                   return_spec_nb=False,
    #                                   return_mean_theta=False,
    #                                   return_gvar=False,
    #                                   output_axis=None):
    #     """
    #     Extract the integrated spectrum from a region of the cube.

    #     All extraction of spectral data must use this core function
    #     because it makes sure that all the updated calibrations are
    #     taken into account.

    #     :param region: A list of the indices of the pixels integrated
    #       in the returned spectrum.

    #     :param subtract_spectrum: (Optional) Remove the given spectrum
    #       from the extracted spectrum before fitting
    #       parameters. Useful to remove sky spectrum. Both spectra must
    #       have the same size.

    #     :param median: (Optional) If True the integrated spectrum is computed
    #       from the median of the spectra multiplied by the number of
    #       pixels integrated. Else the integrated spectrum is the pure
    #       sum of the spectra. In both cases the flux of the spectrum
    #       is the total integrated flux (Default False).

    #     :param mean_flux: (Optional) If True the flux of the spectrum
    #       is the mean flux of the extracted region (default False).

    #     :param return_spec_nb: (Optional) If True the number of
    #       spectra integrated is returned (default False).

    #     :param silent: (Optional) If True, nothing is printed (default
    #       False).

    #     :param return_mean_theta: (Optional) If True, the mean of the
    #       theta values covered by the region is returned (default False).

    #     :param return_gvar: (Optional) If True, returned spectrum will be a
    #       gvar. i.e. a data vector with it's uncetainty (default False).

    #     :param output_axis: (Optional) If not None, the spectrum is
    #       projected on the output axis. Else a scipy.UnivariateSpline
    #       object is returned (default None).

    #     :return: A scipy.UnivariateSpline object or a spectrum
    #       projected on the ouput_axis if it is not None.
    #     """
    #     raise NotImplementedError('check if HDFCube.extract_from_region can be used instead')
        
    #     def _interpolate_spectrum(spec, corr, wavenumber, step, order, base_axis):
    #         import utils.spectrum
    #         import utils.vector
    #         if wavenumber:
    #             corr_axis = utils.spectrum.create_cm1_axis(
    #                 spec.shape[0], step, order, corr=corr)
    #             return utils.vector.interpolate_axis(
    #                 spec, base_axis, 5, old_axis=corr_axis)
    #         else: raise NotImplementedError()


    #     def _extract_spectrum_in_column(data_col, calib_coeff_col, mask_col,
    #                                     median,
    #                                     wavenumber, base_axis, step, order,
    #                                     base_axis_corr):

    #         for icol in range(data_col.shape[0]):
    #             if mask_col[icol]:
    #                 corr = calib_coeff_col[icol]
    #                 if corr != base_axis_corr:
    #                     data_col[icol, :] = _interpolate_spectrum(
    #                         data_col[icol, :], corr, wavenumber, step, order, base_axis)
    #             else:
    #                 data_col[icol, :].fill(np.nan)

    #         if median:
    #             with np.warnings.catch_warnings():
    #                 np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    #                 return (np.nanmedian(data_col, axis=0) * np.nansum(mask_col),
    #                         np.nansum(mask_col))
    #         else:
    #             return (np.nansum(data_col, axis=0),
    #                     np.nansum(mask_col))

    #     if median:
    #         warnings.warn('Median integration')


    #     calibration_coeff_map = self.get_calibration_coeff_map()

    #     calibration_coeff_center = calibration_coeff_map[
    #         calibration_coeff_map.shape[0]/2,
    #         calibration_coeff_map.shape[1]/2]

    #     mask = np.zeros((self.dimx, self.dimy), dtype=np.uint8)
    #     mask[region] = 1
    #     if not silent:
    #         logging.info('Number of integrated pixels: {}'.format(np.sum(mask)))

    #     if np.sum(mask) == 0: raise StandardError('A region must contain at least one valid pixel')

    #     elif np.sum(mask) == 1:
    #         ii = region[0][0] ; ij = region[1][0]
    #         spectrum = _interpolate_spectrum(
    #             self.get_data(ii, ii+1, ij, ij+1, 0, self.dimz, silent=silent),
    #             calibration_coeff_map[ii, ij],
    #             self.params.wavenumber, self.params.step, self.params.order,
    #             self.params.base_axis)
    #         counts = 1

    #     else:
    #         spectrum = np.zeros(self.dimz, dtype=float)
    #         counts = 0

    #         # get range to check if a quadrants extraction is necessary
    #         mask_x_proj = np.nanmax(mask, axis=1).astype(float)
    #         mask_x_proj[np.nonzero(mask_x_proj == 0)] = np.nan
    #         mask_x_proj *= np.arange(self.dimx)
    #         x_min = int(np.nanmin(mask_x_proj))
    #         x_max = int(np.nanmax(mask_x_proj)) + 1

    #         mask_y_proj = np.nanmax(mask, axis=0).astype(float)
    #         mask_y_proj[np.nonzero(mask_y_proj == 0)] = np.nan
    #         mask_y_proj *= np.arange(self.dimy)
    #         y_min = int(np.nanmin(mask_y_proj))
    #         y_max = int(np.nanmax(mask_y_proj)) + 1

    #         if (x_max - x_min < self.dimx / float(self.config.DIV_NB)
    #             and y_max - y_min < self.dimy / float(self.config.DIV_NB)):
    #             quadrant_extraction = False
    #             QUAD_NB = 1
    #             DIV_NB = 1
    #         else:
    #             quadrant_extraction = True
    #             QUAD_NB = self.config.QUAD_NB
    #             DIV_NB = self.config.DIV_NB

    #         # check if parallel extraction is necessary
    #         parallel_extraction = True
    #         # It takes roughly ncpus/4 s to initiate the parallel server
    #         # The non-parallel algo runs at ~400 pixel/s
    #         ncpus = self.params['ncpus']
    #         if ncpus/4. > np.sum(mask)/400.:
    #             parallel_extraction = False
    #         for iquad in range(0, QUAD_NB):

    #             if quadrant_extraction:
    #                 # x_min, x_max, y_min, y_max are now used for quadrants boundaries
    #                 x_min, x_max, y_min, y_max = self.get_quadrant_dims(iquad)

    #             iquad_data = self.get_data(x_min, x_max, y_min, y_max,
    #                                        0, self.dimz, silent=silent)
    #             if parallel_extraction:
    #                 logging.debug('Parallel extraction')
    #                 # multi-processing server init
    #                 job_server, ncpus = self._init_pp_server(silent=silent)
    #                 if not silent: progress = core.ProgressBar(x_max - x_min)
    #                 for ii in range(0, x_max - x_min, ncpus):
    #                     # no more jobs than columns
    #                     if (ii + ncpus >= x_max - x_min):
    #                         ncpus = x_max - x_min - ii

    #                     # jobs creation
    #                     jobs = [(ijob, job_server.submit(
    #                         _extract_spectrum_in_column,
    #                         args=(iquad_data[ii+ijob,:,:],
    #                               calibration_coeff_map[x_min + ii + ijob,
    #                                                     y_min:y_max],
    #                               mask[x_min + ii + ijob, y_min:y_max],
    #                               median, self.params.wavenumber,
    #                               self.params.base_axis, self.params.step,
    #                               self.params.order, self.params.axis_corr),
    #                         modules=("import logging",
    #                                  'import numpy as np',
    #                                  'import orb.utils as utils'),
    #                         depfuncs=(_interpolate_spectrum,)))
    #                             for ijob in range(ncpus)]

    #                     for ijob, job in jobs:
    #                         spec_to_add, spec_nb = job()
    #                         if not np.all(np.isnan(spec_to_add)):
    #                             spectrum += spec_to_add
    #                             counts += spec_nb

    #                     if not silent:
    #                         progress.update(ii, info="ext column : {}/{}".format(
    #                             ii, int(self.dimx/float(DIV_NB))))
    #                 self._close_pp_server(job_server)
    #                 if not silent: progress.end()

    #             else:
    #                 logging.debug('Non Parallel extraction')
    #                 local_mask = mask[x_min:x_max, y_min:y_max]
    #                 local_calibration_coeff_map = calibration_coeff_map[x_min:x_max, y_min:y_max]
    #                 if not silent:
    #                     progress = core.ProgressBar(local_mask.size)
    #                     k = 0
    #                 for i,j in np.ndindex(iquad_data.shape[:-1]):
    #                     if local_mask[i,j]:
    #                         corr = local_calibration_coeff_map[i,j]
    #                         if corr != self.params.axis_corr:
    #                             iquad_data[i,j] = _interpolate_spectrum(
    #                                 iquad_data[i,j], corr, self.params.wavenumber,
    #                                 self.params.step, self.params.order, self.params.base_axis)
    #                     else:
    #                         iquad_data[i,j].fill(np.nan)
    #                     if not silent:
    #                         k+=1
    #                         if not k%500: progress.update(k)
    #                 if not silent: progress.end()
    #                 if median:
    #                     with np.warnings.catch_warnings():
    #                         np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
    #                         spec_to_add = np.nanmedian(iquad_data, axis=(0,1)) * np.nansum(local_mask)
    #                         spec_nb = np.nansum(local_mask)
    #                 else:
    #                     spec_to_add = np.nansum(iquad_data, axis=(0,1))
    #                     spec_nb = np.nansum(local_mask)
    #                 if not np.all(np.isnan(spec_to_add)):
    #                     spectrum += spec_to_add
    #                     counts += spec_nb


    #     # add uncertainty on the spectrum
    #     if return_gvar:
    #         flux_uncertainty = self.get_flux_uncertainty()

    #         if flux_uncertainty is not None:
    #             uncertainty = np.nansum(flux_uncertainty[np.nonzero(mask)])
    #             logging.debug('computed mean flux uncertainty: {}'.format(uncertainty))
    #             spectrum = gvar.gvar(spectrum, np.ones_like(spectrum) * uncertainty)


    #     if subtract_spectrum is not None:
    #         spectrum -= subtract_spectrum * counts

    #     if mean_flux:
    #         spectrum /= counts

    #     returns = list()
    #     if output_axis is not None and np.all(output_axis == self.params.base_axis):
    #         spectrum[np.isnan(gvar.mean(spectrum))] = 0. # remove nans
    #         returns.append(spectrum)

    #     else:
    #         nonans = ~np.isnan(gvar.mean(spectrum))
    #         spectrum_function = scipy.interpolate.UnivariateSpline(
    #             self.params.base_axis[nonans], gvar.mean(spectrum)[nonans],
    #             s=0, k=1, ext=1)
    #         if return_gvar:
    #             spectrum_function_sdev = scipy.interpolate.UnivariateSpline(
    #                 self.params.base_axis[nonans], gvar.sdev(spectrum)[nonans],
    #                 s=0, k=1, ext=1)
    #             raise Exception('now a tuple is returned with both functions for mean and sdev, this will raise an error somewhere and must be checked before')
    #             spectrum_function = (spectrum_function, spectrum_function_sdev)

    #         if output_axis is None:
    #             returns.append(spectrum_function(gvar.mean(output_axis)))
    #         else:
    #             returns.append(spectrum_function)

    #     if return_spec_nb:
    #         returns.append(counts)
    #     if return_mean_theta:
    #         theta_map = self.get_theta_map()
    #         mean_theta = np.nanmean(theta_map[np.nonzero(mask)])
    #         logging.debug('computed mean theta: {}'.format(mean_theta))
    #         returns.append(mean_theta)

    #     return returns

    def extract_spectrum_bin(self, x, y, b, **kwargs):
        """Extract a spectrum integrated over a binned region.

        :param x: X position of the bottom-left pixel

        :param y: Y position of the bottom-left pixel

        :param b: Binning. If 1, only the central pixel is extracted

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if b < 1: raise StandardError('Binning must be at least 1')

        mask = np.zeros((self.dimx, self.dimy), dtype=bool)
        mask[int(x):int(x+b), int(y):int(y+b)] = True
        region = np.nonzero(mask)

        return self.extract_integrated_spectrum(region, **kwargs)

    def extract_spectrum(self, x, y, r, **kwargs):
        """Extract a spectrum integrated over a circular region of a
        given radius.

        :param x: X position of the center

        :param y: Y position of the center

        :param r: Radius. If 0, only the central pixel is extracted.

        :param kwargs: Keyword arguments of the function
          :py:meth:`~HDFCube._extract_spectrum_from_region`.

        :returns: (axis, spectrum)
        """
        if r < 0: r = 0.001
        X, Y = np.mgrid[0:self.dimx, 0:self.dimy]
        R = np.sqrt(((X-x)**2 + (Y-y)**2))
        region = np.nonzero(R <= r)

        return self.extract_integrated_spectrum(region, **kwargs)


    def extract_integrated_spectrum(self, region, **kwargs):
        """Extract a spectrum integrated over a given region (can be a
        list of pixels as returned by the function
        :py:meth:`numpy.nonzero` or a ds9 region file).

        :param region: Region to integrate (can be a list of pixel
          coordinates as returned by the function
          :py:meth:`numpy.nonzero` or the path to a ds9 region
          file). If it is a ds9 region file, multiple regions can be
          defined and all will be integrated into one spectrum.
        """
        region = self.get_mask_from_ds9_region_file(region)

        returns = list()
        returns.append(self.params.base_axis.astype(float))
        returns += list(self._extract_spectrum_from_region(
            region, output_axis=self.params.base_axis.astype(float), **kwargs))
        return returns


    def get_mask_from_ds9_region_file(self, region, integrate=True):
        """Return a mask from a ds9 region file.

        :param region: Path to a ds9 region file.

        :param integrate: (Optional) If True, all pixels are integrated
          into one mask, else a list of region masks is returned (default
          True)
        """
        if isinstance(region, str):
            return utils.misc.get_mask_from_ds9_region_file(
                region,
                [0, self.dimx],
                [0, self.dimy],
                header=self.header,
                integrate=integrate)
        else: return region

    def get_standard_image(self):
        """Return standard image
        """
        if not self.has_param('standard_image_path'):
            if self.has_param('standard_image_path_1'):
                self.params['standard_image_path'] = self.params['standard_image_path_1']
        
        if not self.has_dataset('standard_image'):
            if self.has_param('standard_image_path'):
                std_cube = HDFCube(self.params.standard_image_path)
                std_im = std_cube.compute_sum_image() / std_cube.dimz 
            else: raise StandardError('if no standard image can be found in the archive, standard_image_path must be set')
            return photometry.StandardImage(
                std_im, params=std_cube.params)
        else:
            return photometry.StandardImage(
                self.get_dataset('standard_image', protect=False),
                params=self.get_dataset_attrs('standard_image'),
                instrument=self.params.instrument)
        
        
    def compute_flambda(self, deg=2, std_im=None):
        """Return flamba calibration function
        """
        # compute flambda from configuration curves
        photom = photometry.Photometry(self.params.filter_name, self.params.camera,
                                       instrument=self.params.instrument)
        flam_config = photom.compute_flambda()
        logging.info('mean flambda config: {}'.format(np.nanmean(flam_config.data)))

        # compute the correction from a standard star spectrum
        if self.has_param('standard_path'):
            data, hdr = utils.io.read_fits(self.params.standard_path, return_header=True)
            std_flux_sp, wav = data.T
            wav = np.linspace(wav[0], wav[-1], wav.size) # avoids rounding errors due to float32 conversion
            std_flux_sp = photometry.StandardSpectrum(std_flux_sp, axis=wav, params=hdr)
            eps_vector = std_flux_sp.compute_flux_correction_vector(deg=deg)

            logging.info('relative correction vector: max {:.2f}, min {:.2f}'.format(
                np.nanmax(eps_vector.data), np.nanmin(eps_vector.data)))
        else:
            logging.debug('standard_path not set: no relative vector correction computed')
            eps_vector = core.Cm1Vector1d(np.ones(self.dimz, dtype=float), params=self.params,
                                          axis=self.get_base_axis())
        

        # compute the correction from a standard star calibration image
        if std_im is None:
            try:
                std_im = self.get_standard_image()
                logging.info('absolute correction factor: {:.2f}'.format(eps_mean))

            except StandardError:
                logging.debug('standard_image_path not set: no absolute vector correction computed')
                std_im = None
        
        if std_im is not None:
            eps_mean = std_im.compute_flux_correction_factor()
            eps_vector = eps_vector.multiply(eps_mean)
                
        return photom.compute_flambda(eps=eps_vector)

        
        
            
        
        
        
        
