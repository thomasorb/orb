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
import scipy.interpolate
import fft

#################################################
#### CLASS HDFCube ##############################
#################################################

class HDFCube(core.Tools):
    """ This class implements the use of an HDF5 cube."""        

    protected_datasets = 'data', 'mask', 'header', 'deep_frame', 'params'
    
    def __init__(self, path, indexer=None, instrument=None, shape=None, **kwargs):

        """:param path: Path to an HDF5 cube

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param instrument: (Optional) instrument name ('sitelle' or
          'spiomm'). If it cannot be read from the file itself (in
          attributes) it must be set.

        :param kwargs: (Optional) :py:class:`~orb.core.Tools` kwargs.

        """
        self.cube_path = str(path)

        # create file if it does not exists
        if not os.path.exists(path):
            if shape is None:
                raise ValueError('cube does not exist. If you want to create one, shape must be set.')
            if instrument is None:
                raise ValueError('to create a new cube, instrument must be set')
            with utils.io.open_hdf5(path, 'w') as f:
                utils.validate.has_len(shape, 3, object_name='shape')
                f.create_dataset('data', shape=shape, chunks=True)
                f.attrs['level2'] = True
                

        elif shape is not None:
            raise ValueError('shape must be set only when creating a new HDFCube')
            
        
        # read instrument parameter from file
        with utils.io.open_hdf5(path, 'r') as f:
            if instrument is None:
                if 'instrument' not in f.attrs:
                    raise ValueError("instrument could be read from the file attributes. Please set it to 'sitelle' or 'spiomm'")                
                instrument = f.attrs['instrument']
                    
                
        core.Tools.__init__(self, instrument=instrument, **kwargs)        

        self.star_list = None
        self.z_median = None
        self.z_mean = None
        self.z_std = None
        self.is_old = False
        
        with utils.io.open_hdf5(self.cube_path, 'r') as f:
            if 'level2' not in f.attrs:
                warnings.warn('old cube architecture. IO performances could be reduced.')
                self.is_old = True

            if not self.is_old:
                if 'data' not in f:
                    raise TypeError('cube does not contain any data')
                self.shape = f['data'].shape
                if len(self.shape) != 3:
                    raise TypeError('malformed data cube, data shape is {}'.format(self.shape))
                self.dtype = f['data'].dtype

        if self.is_old:
            self.oldcube = old.HDFCube(self.cube_path, silent_init=True)
            self.shape = self.oldcube.shape
            self.dtype = self.oldcube.dtype
            
        self.dimx, self.dimy, self.dimz = self.shape

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
        
        with self.open_hdf5('r') as f:
            _data = np.copy(f['data'].__getitem__(key))
            if self.has_dataset('mask'):
                _data *= self.get_dataset('mask').__getitem__((key[0], key[1]))
                
            return np.squeeze(_data)

    def __setitem__(self, key, value):
        """Implement setitem special method"""
        if self.is_old:
            raise StandardError('Old cubes cannot be modified. Please export the actual cube first with cube.export().')
        
        with self.open_hdf5('a') as f:
            return f['data'].__setitem__(key, value)
    
    def open_hdf5(self, mode='r'):
        """Return the hdf5 file.

        :param mode: Opening mode (can be 'r', 'w', 'a')
        """
        return utils.io.open_hdf5(self.cube_path, mode)
    
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
        
        return self[x_min:x_max, y_min:y_max, zmin_zmax]

        
    def has_dataset(self, path):
        """Check if a dataset is present"""
        with self.open_hdf5('r') as f:
            if path not in f: return False
            else: return True
    
    def get_dataset(self, path):
        """Return a dataset (but not 'data', instead use get_data).

        :param path: dataset path
        """
        if path == 'data':
            raise ValueError('to set data please use your cube as a classic 3d numpy array. e.g. arr = cube[:,:,:].')
        with self.open_hdf5('r') as f:
            if path not in f:
                raise AttributeError('{} dataset not in the hdf5 file'.format(path))
            return f[path][:]

    def set_dataset(self, path, data, protect=True):
        """Write a dataset to the hdf5 file

        :param path: dataset path

        :param data: data to write.

        :param protect: (Optional) check if dataset is protected
          (default True).
        """
        if protect:
            if path in self.protected_datasets:
                raise IOError('dataset {} is protected. please use the corresponding higher level method (something like get_{} might do)'.format(path, path))

        if path == 'data':
            raise ValueError('to set data please use your cube as a classic 3d numpy array. e.g. cube[:,:,:] = value.')
        with self.open_hdf5('a') as f:
            if path in f:
                del f[path]
                warnings.warn('{} dataset changed'.format(path))

            if isinstance(data, dict):
                data = utils.io.dict2array(data)
            f.create_dataset(path, data=data, chunks=True)


    def set_mask(self, data):
        """Set mask. A mask must be a 2d frame of shape (self.dimx,
        self.dimy). A Zero indicates a pixel which should be masked
        (Nans are returned for this pixel).

        :param data: Must be a 2d frame of shape (self.dimx,
        self.dimy).
        """
        utils.validate.is_2darray(data, object_name='data')
        if data.shape != (self.dimx, self.dimy):
            raise TypeError('data must have shape ({}, {})'.format(
                self.dimx, self.dimy))
        if data.dtype != np.bool:
            raise TypeError('data should be of type boolean')
        _data = np.copy(data).astype(float)
        _data[np.nonzero(_data == 0)] = np.nan
        self.set_dataset('mask', _data, protect=False)

    def get_mask(self):
        """Return mask."""
        _mask = np.copy(self.get_dataset('mask'))
        _mask[np.isnan(_mask)] = 0
        return _mask.astype(bool)
        
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

    def get_interf_energy_map(self, recompute=False):
        """Return the energy map of an interferogram cube.
    
        :param recompute: (Optional) Force to recompute energy map
          even if it is already present in the cube (default False).
          
        If an energy map has already been computed ('energy_map'
        dataset) return it directly.
        """
        if not recompute:
            if self.has_dataset('energy_map'):
                return self.get_dataset('energy_map')
            
        mean_map = self.get_mean_image()
        energy_map = np.zeros((self.dimx, self.dimy), dtype=self.dtype)
        progress = ProgressBar(self.dimz)
        for _ik in range(self.dimz):
            energy_map += np.abs(self.get_data_frame(_ik) - mean_map)**2.
            progress.update(_ik, info="Creating interf energy map")
        progress.end()
        
        f['energy_map'] = np.sqrt(energy_map / self.dimz)
        return f['energy_map'][:]

    def get_mean_image(self, recompute=False):
        """Return the mean image of a cube (corresponding to a deep
        frame for an interferogram cube or a specral cube).

        :param recompute: (Optional) Force to recompute mean image
          even if it is already present in the cube (default False).
        
        .. note:: In this process NaNs are handled correctly
        """
        if self.is_old:
            f['deep_frame'] = self.oldcube.get_mean_image(recompute=recompute)
            return f['deep_frame'][:]

        if not recompute:
            if self.has_dataset('deep_frame'):
                return self.get_dataset('deep_frame')
        
            else:
                raise NotImplementedError()

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

    def export(self, export_path, x_range=None, y_range=None, z_range=None):
        
        """Export cube as one HDF5 file.

        :param export_path: Path of the exported FITS file

        :param x_range: (Optional) Tuple (x_min, x_max) (default
          None).
        
        :param y_range: (Optional) Tuple (y_min, y_max) (default
          None).
        
        :param z_range: (Optional) Tuple (z_min, z_max) (default
          None).

        :param mask: (Optional)
        """
        old_keys = 'quad', 'frame'
        def is_exportable(key):
            for iold in old_keys:
                if iold in key:
                    return False
            return True

        with utils.io.open_hdf5(export_path, 'w') as fout:
            fout.attrs['level2'] = True
            
            with self.open_hdf5('r') as f:
                for iattr in f.attrs:
                    fout.attrs[iattr] = f.attrs[iattr]
                    
                for ikey in f:
                    if is_exportable(ikey):
                        logging.info('adding {}'.format(ikey))
                        fout.create_dataset(ikey, data=f[ikey], chunks=True)
                fout.create_dataset('data', shape=self.shape, chunks=True)
            for iquad in range(self.config.QUAD_NB):
                logging.info('writing quad {}/{}'.format(
                    iquad, self.config.QUAD_NB))
                xmin, xmax, ymin, ymax = self.get_quadrant_dims(iquad)
                fout['data'][xmin:xmax, ymin:ymax, :] = self[xmin:xmax, ymin:ymax, :]

    def get_frame_header(self, index):
        """Return the header of a frame given its index in the list.

        The header is returned as an instance of pyfits.Header().

        :param index: Index of the frame
        """
        return utils.io.header_hdf52fits(
            self.get_dataset('frame_header_{}'.format(index)))

            
    def set_frame_header(self, index, header):
        """Set the header of a frame.

        The header must be an instance of pyfits.Header().

        :param index: Index of the frame

        :param header: Header as a pyfits.Header instance.
        """
        self.set_dataset('frame_header_{}'.format(index),
                         utils.io.header_fits2hdf5(header))
            

    def get_cube_header(self):
        """Return the header of a the cube.

        The header is returned as an instance of pyfits.Header().
        """
        if self.has_dataset('header'):
            return utils.io.header_hdf52fits(self.get_dataset('header'))
        else:
            return pyfits.Header()

    def set_cube_header(self, header):
        """Set cube header.

        :param header: Header as a pyfits.Header instance.
        """
        self.set_dataset(
            'header', utils.io.header_fits2hdf5(header),
            protect=False)


    def get_calibration_laser_map(self):
        """Return stored calibration laser map"""
        if self.has_dataset('calib_map'):
            return self.get_dataset('calib_map')
        else:
            if isinstance(self, OCube):
                return OCube.get_calibration_laser_map(self)
            else:
                warnings.warn('No calibration laser map stored')
                return None

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
        
#################################################
#### CLASS Cube ################################
#################################################
class Cube(HDFCube):
    """Provide additional cube methods when observation parameters are known.
    """

    needed_params = ('step', 'order', 'filter_name', 'exposure_time',
                     'step_nb', 'calibration_laser_map_path', 'zpd_index')

    optional_params = ('target_ra', 'target_dec', 'target_x', 'target_y',
                       'dark_time', 'flat_time', 'camera_index', 'wcs_rotation')

    computed_params = ('filter_nm_min', 'filter_nm_max', 'filter_file_path',
                       'filter_cm1_min', 'filter_cm1_max', 'binning')
    
    def __init__(self, path, params=None, instrument=None, **kwargs):
        """Initialize Cube class.

        :param data: Path to an HDF5 Cube

        :param params: (Optional) Path to an option file or dict
        instance.

        :param kwargs: Cube kwargs + other parameters not supplied in
          params (else overwrite parameters set in params)

        .. note:: params are first read from the HDF5 file
          itself. They can be modified through the params dictionary
          and the params dictionary can itself be modified with
          keyword arguments.
        """
        kwargs_params = dict()
        for ikey in kwargs.keys():
            if ikey in (self.needed_params + self.optional_params):
                kwargs_params[ikey] = kwargs.pop(ikey)

        HDFCube.__init__(self, path, instrument=instrument, **kwargs)
                
        self.needed_params = tuple(self.params.keys() + list(self.needed_params))
        self.load_params(params, **kwargs_params)
        self.compute_data_parameters()
        self.validate()
            
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
                ofile = core.OptionFile(params)
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
                
        elif params is not None:
            raise TypeError('params type ({}) not handled'.format(type(params)))


        # parse additional parameters
        for iparam in kwargs:
            if iparam in (self.needed_params + self.optional_params):
                self.set_param(iparam, kwargs[iparam])
            else: raise ValueError('parameter {} supplied as a keyword argument but not used. Please remove it'.format(iparam))
            

        # load params from file
        if self.has_dataset('params'):
            fileparams = utils.io.array2dict(self.get_dataset('params'))
            
            for iparam in fileparams:
                if iparam not in self.params:
                    self.set_param(iparam, fileparams[iparam])

            
        # validate needed params
        for iparam in self.needed_params:
            if iparam not in self.params:
                raise ValueError('parameter {} must be defined in params'.format(iparam))

        for iparam in self.params:
            if iparam not in (self.needed_params + self.optional_params + self.computed_params):
                warnings.warn('parameter {} defined but not used'.format(iparam))

        
        # compute additional parameters
        self.filterfile = core.FilterFile(self.params.filter_name)
        self.set_param('filter_file_path', self.filterfile.basic_path)
        self.set_param('filter_nm_min', self.filterfile.get_filter_bandpass()[0])
        self.set_param('filter_nm_max', self.filterfile.get_filter_bandpass()[1])
        self.set_param('filter_cm1_min', self.filterfile.get_filter_bandpass_cm1()[0])
        self.set_param('filter_cm1_max', self.filterfile.get_filter_bandpass_cm1()[1])

        
        self.params_defined = True

        
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

        # validate optional parameters
        for iparam in self.params:
            if iparam not in (self.needed_params + self.optional_params + self.computed_params):
                warnings.warn('parameter {} defined but not used'.format(iparam))

    def write_params(self):
        """Write loaded params to the file."""
        self.set_dataset('params', self.params, protect=False)
        
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


#################################################
#### CLASS InteferogramCube ####################
#################################################
class InterferogramCube(Cube):
    """Provide additional methods for an interferogram cube when
    observation parameters are known.
    """

    def get_interferogram(self, x, y):
        """Return an orb.fft.Interferogram instance
        
        :param x: x position
        :param y: y position
        """
        self.validate()
        x = self.validate_x_index(x, clip=False)
        y = self.validate_x_index(y, clip=False)

        calib_coeff = self.get_calibration_coeff_map()[x, y]
        return fft.Interferogram(self[int(x), int(y), :], params=self.params,
                                 zpd_index=self.params.zpd_index, calib_coeff=calib_coeff)

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
        return fft.Interferogram(interf, params=self.params,
                                 zpd_index=self.params.zpd_index,
                                 calib_coeff=calib_coeff)

    
#################################################
#### CLASS FDCube ###############################
#################################################
class FDCube(core.Tools):
    """Basic handling class for a set of frames grouped into one virtual
    cube.

    This is a basic class which is mainly used to export data into an
    hdf5 format.

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


    def export(self, export_path, mask):
        """Export FDCube as an hdf5 cube
        """
        cube = HDFCube(
            export_path, shape=(self.dimx, self.dimy, self.dimz),
            instrument=self.instrument)

        if mask is not None:
            cube.set_mask(mask)
        cube.set_cube_header(self.get_cube_header())
            
        progress = core.ProgressBar(self.dimz)
        for iframe in range(self.dimz):
            progress.update(iframe, info='writing frame {}/{}'.format(
                iframe + 1, self.dimz))
            cube[:,:,iframe] = self[:,:,iframe]
            cube.set_frame_header(iframe, self.get_frame_header(iframe))
        progress.end()
            
            
        
# ##################################################
# #### CLASS OutHDFCube ############################
# ##################################################
# class OutHDFCube(core.Tools):
#     """Output HDF5 Cube class.

#     This class must be used to output a valid HDF5 cube.
    
#     .. warning:: The underlying dataset is not readonly and might be
#       overwritten.

#     .. note:: This class has been created because
#       :py:class:`orb.core.HDFCube` must not be able to change its
#       underlying dataset (the HDF5 cube is always read-only).
#     """    
#     def __init__(self, export_path, shape, overwrite=False,
#                  reset=False, **kwargs):
#         """Init OutHDFCube class.

#         :param export_path: Path ot the output HDF5 cube to create.

#         :param shape: Data shape. Must be a 3-Tuple (dimx, dimy, dimz)

#         :param overwrite: (Optional) If True data will be overwritten
#           but existing data will not be removed (default True).

#         :param reset: (Optional) If True and if the file already
#           exists, it is deleted (default False).
        
#         :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
#         """
#         raise NotImplementedError()
#         core.Tools.__init__(self, **kwargs)
        
#         # change path if file exists and must not be overwritten
#         self.export_path = str(export_path)
#         if reset and os.path.exists(self.export_path):
#             os.remove(self.export_path)
        
#         if not overwrite and os.path.exists(self.export_path):
#             index = 0
#             while os.path.exists(self.export_path):
#                 self.export_path = (os.path.splitext(export_path)[0] + 
#                                    "_" + str(index) + 
#                                    os.path.splitext(export_path)[1])
#                 index += 1
#         if self.export_path != export_path:
#             warnings.warn('Cube path changed to {} to avoid overwritting an already existing file'.format(self.export_path))

#         if len(shape) == 3: self.shape = shape
#         else: raise StandardError('An HDF5 cube shape must be a tuple (dimx, dimy, dimz)')

#         try:
#             self.f = self.open_hdf5(self.export_path, 'a')
#         except IOError, e:
#             if overwrite:
#                 os.remove(self.export_path)
#                 self.f = self.open_hdf5(self.export_path, 'a')
#             else:
#                 raise StandardError(
#                     'IOError while opening HDF5 cube: {}'.format(e))
                
#         logging.info('Opening OutHDFCube {} ({},{},{})'.format(
#             self.export_path, *self.shape))
        
#         # add attributes
#         self.f.attrs['dimx'] = self.shape[0]
#         self.f.attrs['dimy'] = self.shape[1]
#         self.f.attrs['dimz'] = self.shape[2]

#         self.imshape = (self.shape[0], self.shape[1])

#     def write_frame_attribute(self, index, attr, value):
#         """Write a frame attribute

#         :param index: Index of the frame

#         :param attr: Attribute name

#         :param value: Value of the attribute to write
#         """
#         self.f[self._get_hdf5_frame_path(index)].attrs[attr] = value

#     def write_frame(self, index, data=None, header=None, mask=None,
#                     record_stats=False, force_float32=True, section=None,
#                     force_complex64=False, compress=False):
#         """Write a frame

#         :param index: Index of the frame
        
#         :param data: (Optional) Frame data (default None).
        
#         :param header: (Optional) Frame header (default None).
        
#         :param mask: (Optional) Frame mask (default None).
        
#         :param record_stats: (Optional) If True Mean and Median of the
#           frame are appended as attributes (data must be set) (defaut
#           False).

#         :param force_float32: (Optional) If True, data type is forced
#           to numpy.float32 type (default True).

#         :param section: (Optional) If not None, must be a 4-tuple
#           [xmin, xmax, ymin, ymax] giving the section to write instead
#           of the whole frame. Useful to modify only a part of the
#           frame (deafult None).

#         :param force_complex64: (Optional) If True, data type is
#           forced to numpy.complex64 type (default False).

#         :param compress: (Optional) If True, data is lossely
#           compressed using a gzip algorithm (default False).
#         """
#         def _replace(name, dat):
#             if name == 'data':
#                 if force_complex64: dat = dat.astype(np.complex64)
#                 elif force_float32: dat = dat.astype(np.float32)
#                 dat_path = self._get_hdf5_data_path(index)
#             if name == 'mask':
#                 dat_path = self._get_hdf5_data_path(index, mask=True)
#             if name == 'header':
#                 dat_path = self._get_hdf5_header_path(index)
#             if name == 'data' or name == 'mask':
#                 if section is not None:
#                     old_dat = None
#                     if  dat_path in self.f:
#                         if (self.f[dat_path].shape == self.imshape):
#                             old_dat = self.f[dat_path][:]
#                     if old_dat is None:
#                         frame = np.empty(self.imshape, dtype=dat.dtype)
#                         frame.fill(np.nan)
#                     else:
#                         frame = np.copy(old_dat).astype(dat.dtype)
#                     frame[section[0]:section[1],section[2]:section[3]] = dat
#                 else:
#                     if dat.shape == self.imshape:
#                         frame = dat
#                     else:
#                         raise StandardError(
#                             "Bad data shape {}. Must be {}".format(
#                                 dat.shape, self.imshape))
#                 dat = frame
                    
#             if dat_path in self.f: del self.f[dat_path]
#             if compress:
#                 ## szip_types = (np.float32, np.float64, np.int16, np.int32, np.int64,
#                 ##               np.uint8, np.uint16)
            
#                 ## if dat.dtype in szip_types:
#                 ##     compression = 'szip'
#                 ##     compression_opts = ('nn', 32)
#                 ## else:
#                 ##     compression = 'gzip'
#                 ##     compression_opts = 4
#                 compression = 'lzf'
#                 compression_opts = None
#             else:
#                 compression = None
#                 compression_opts = None
#             self.f.create_dataset(
#                 dat_path, data=dat,
#                 compression=compression, compression_opts=compression_opts)
#             return dat

#         if force_float32 and force_complex64:
#             raise StandardError('force_float32 and force_complex64 cannot be both set to True')

            
#         if data is None and header is None and mask is None:
#             warnings.warn('Nothing to write in the frame {}').format(
#                 index)
#             return
        
#         if data is not None:
#             data = _replace('data', data)
            
#             if record_stats:
#                 mean = bn.nanmean(data.real)
#                 median = bn.nanmedian(data.real)
#                 self.f[self._get_hdf5_frame_path(index)].attrs['mean'] = (
#                     mean)
#                 self.f[self._get_hdf5_frame_path(index)].attrs['median'] = (
#                     mean)
#             else:
#                 mean = None
#                 median = None
                
#         if mask is not None:
#             mask = mask.astype(np.bool_)
#             _replace('mask', mask)


#         # Creating pyfits.ImageHDU instance to format header
#         if header is not None:
#             if not isinstance(header, pyfits.Header):
#                 header = pyfits.Header(header)

#         if data.dtype != np.bool:
#             if np.iscomplexobj(data) or force_complex64:
#                 hdu = pyfits.ImageHDU(data=data.real, header=header)
#             else:
#                 hdu = pyfits.ImageHDU(data=data, header=header)
#         else:
#             hdu = pyfits.ImageHDU(data=data.astype(np.uint8), header=header)
            
        
#         if hdu is not None:
#             if record_stats:
#                 if mean is not None:
#                     if not np.isnan(mean):
#                         hdu.header.set('MEAN', mean,
#                                        'Mean of data (NaNs filtered)',
#                                        after=5)
#                 if median is not None:
#                     if not np.isnan(median):
#                         hdu.header.set('MEDIAN', median,
#                                        'Median of data (NaNs filtered)',
#                                        after=5)
            
#             hdu.verify(option=u'silentfix')
                
            
#             _replace('header', self._header_fits2hdf5(hdu.header))
            

#     def append_image_list(self, image_list):
#         """Append an image list to the HDF5 cube.

#         :param image_list: Image list to append.
#         """
#         if 'image_list' in self.f:
#             del self.f['image_list']

#         if image_list is not None:
#             self.f['image_list'] = np.array(image_list)
#         else:
#             warnings.warn('empty image list')
        

#     def append_deep_frame(self, deep_frame):
#         """Append a deep frame to the HDF5 cube.

#         :param deep_frame: Deep frame to append.
#         """
#         if 'deep_frame' in self.f:
#             del self.f['deep_frame']
            
#         self.f['deep_frame'] = deep_frame

#     def append_energy_map(self, energy_map):
#         """Append an energy map to the HDF5 cube.

#         :param energy_map: Energy map to append.
#         """
#         if 'energy_map' in self.f:
#             del self.f['energy_map']
            
#         self.f['energy_map'] = energy_map

    
#     def append_calibration_laser_map(self, calib_map, header=None):
#         """Append a calibration laser map to the HDF5 cube.

#         :param calib_map: Calibration laser map to append.

#         :param header: (Optional) Header to append (default None)
#         """
#         if 'calib_map' in self.f:
#             del self.f['calib_map']
            
#         self.f['calib_map'] = calib_map
#         if header is not None:
#             self.f['calib_map_hdr'] = self._header_fits2hdf5(header)

#     def append_header(self, header):
#         """Append a header to the HDF5 cube.

#         :param header: header to append.
#         """
#         if 'header' in self.f:
#             del self.f['header']
            
#         self.f['header'] = self._header_fits2hdf5(header)
        
#     def close(self):
#         """Close the HDF5 cube. Class cannot work properly once this
#         method is called so delete it. e.g.::
        
#           outhdfcube.close()
#           del outhdfcube
#         """
#         try:
#             self.f.close()
#         except Exception:
#             pass


# ##################################################
# #### CLASS OutHDFQuadCube ########################
# ##################################################           

# class OutHDFQuadCube(OutHDFCube):
#     """Output HDF5 Cube class saved in quadrants.

#     This class can be used to output a valid HDF5 cube.
#     """

#     def __init__(self, export_path, shape, quad_nb, overwrite=False,
#                  reset=False, **kwargs):
#         """Init OutHDFQuadCube class.

#         :param export_path: Path ot the output HDF5 cube to create.

#         :param shape: Data shape. Must be a 3-Tuple (dimx, dimy, dimz)

#         :param quad_nb: Number of quadrants in the cube.

#         :param overwrite: (Optional) If True data will be overwritten
#           but existing data will not be removed (default False).

#         :param reset: (Optional) If True and if the file already
#           exists, it is deleted (default False).
        
#         :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
#         """
#         raise NotImplementedError()
#         OutHDFCube.__init__(self, export_path, shape, overwrite=overwrite,
#                             reset=reset, **kwargs)

#         self.f.attrs['quad_nb'] = quad_nb

#     def write_quad(self, index, data=None, header=None, force_float32=True,
#                    force_complex64=False,
#                    compress=False):
#         """"Write a quadrant

#         :param index: Index of the quadrant
        
#         :param data: (Optional) Frame data (default None).
        
#         :param header: (Optional) Frame header (default None).
        
#         :param mask: (Optional) Frame mask (default None).
        
#         :param record_stats: (Optional) If True Mean and Median of the
#           frame are appended as attributes (data must be set) (defaut
#           False).

#         :param force_float32: (Optional) If True, data type is forced
#           to numpy.float32 type (default True).

#         :param section: (Optional) If not None, must be a 4-tuple
#           [xmin, xmax, ymin, ymax] giving the section to write instead
#           of the whole frame. Useful to modify only a part of the
#           frame (deafult None).

#         :param force_complex64: (Optional) If True, data type is
#           forced to numpy.complex64 type (default False).

#         :param compress: (Optional) If True, data is lossely
#           compressed using a gzip algorithm (default False).
#         """
        
#         if force_float32 and force_complex64:
#             raise StandardError('force_float32 and force_complex64 cannot be both set to True')

            
#         if data is None and header is None:
#             warnings.warn('Nothing to write in the frame {}').format(
#                 index)
#             return
        
#         if data is not None:
#             if force_complex64: data = data.astype(np.complex64)
#             elif force_float32: data = data.astype(np.float32)
#             dat_path = self._get_hdf5_quad_data_path(index)

#             if dat_path in self.f:
#                 del self.f[dat_path]

#             if compress:
                
#                 #szip_types = (np.float32, np.float64, np.int16, np.int32, np.int64,
#                 #              np.uint8, np.uint16)
#                 ## if data.dtype in szip_types:
#                 ##     compression = 'szip'
#                 ##     compression_opts = ('nn', 32)
#                 ## else:
#                 ##     compression = 'gzip'
#                 ##     compression_opts = 4
#                 compression = 'lzf'
#                 compression_opts = None
#             else:
#                 compression = None
#                 compression_opts = None
                
#             self.f.create_dataset(
#                 dat_path, data=data,
#                 compression=compression,
#                 compression_opts=compression_opts)

#             return data
        
        
