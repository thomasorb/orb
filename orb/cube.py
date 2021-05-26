#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: cube.py

## Copyright (c) 2010-2020 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import datetime

import orb.core
import orb.old
import orb.utils.io
import orb.utils.misc
import orb.utils.astrometry
import orb.utils.err
import orb.fft
import orb.image
import orb.cutils
import orb.version
import astropy.io.fits

import scipy.interpolate
import gvar
import pyregion

import orb.photometry
import gc
import h5py




class MockArray(object):

    def __init__(self, path=None):

        if path is None: return
        
        if not isinstance(path, str):
            raise TypeError('path must be a string')

        if not os.path.exists(path):
            raise IOError('File does not exist')

        with orb.utils.io.open_hdf5(path, 'r') as f:
            if 'data' not in f:
                raise Exception('badly formatted hdf5 file')

            self.shape = f['data'].shape
            self.ndim = f['data'].ndim
            self.dtype = f['data'].dtype
                
    def __getitem__(self, key):
        raise Exception('you are using a MockArray :)')

#################################################
#### CLASS HDFCube ##############################
#################################################
class HDFCube(orb.core.WCSData):
    """ This class implements the use of an HDF5 cube."""        

    protected_datasets = 'data', 'mask', 'header', 'deep_frame', 'params', 'axis', 'calib_map', 'phase_map', 'phase_map_err', 'phase_maps_coeff_map', 'phase_maps_cm1_axis', 'standard_image', 'standard_spectrum'

    optional_params = ('target_ra', 'target_dec', 'target_x', 'target_y',
                       'dark_time', 'flat_time', 'camera', 'wcs_rotation',
                       'calibration_laser_map_path')
    
    def __init__(self, path, indexer=None,
                 instrument=None, config=None, data_prefix='./',
                 **kwargs):

        """Init HDFCube

        :param path: Path to an HDF5 cube

        :param indexer: (Optional) Must be a :py:class:`orb.core.Indexer`
          instance. If not None created files can be indexed by this
          instance.

        :param instrument: (Optional) instrument name ('sitelle' or
          'spiomm'). If it cannot be read from the file itself (in
          attributes) it must be set.

        :param kwargs: (Optional) :py:class:`~orb.orb.core.Data` kwargs.
        """
        self.cube_path = str(path)

        if not os.path.exists(self.cube_path):
            raise IOError('File {} does not exist'.format(self.cube_path))
                                
        # check if cube has an old format in which case it must be
        # loaded before and passed as an instance to Data.
        
        logging.info('Cube is level {}'.format(self.get_level()))
        
        if isinstance(self.cube_path, str):
            with orb.utils.io.open_hdf5(self.cube_path, 'r') as f:
                if self.is_level1():
                    self.oldcube = orb.old.HDFCube(self.cube_path, silent_init=True)
                    self.data = MockArray()
                    self.data.shape = self.oldcube.shape
                    self.data.dtype = self.oldcube.dtype
                    self.data.ndim = 3                    
                else:
                    self.data = MockArray(self.cube_path)
                    
                self.axis = None
                self.mask = None
                self.err = None
                self.params = orb.core.ROParams()
                for param in f.attrs:
                    try:
                        param_value = f.attrs[param]
                        if isinstance(param_value, bytes):
                            param_value = param_value.decode()
                        self.params[param] = param_value
                    except TypeError as e:
                        logging.debug('error reading param from attributes {}: {}'.format(
                            param, e))

                if 'instrument' in self.params and instrument is None:
                    instrument = self.params['instrument']

                
        # init Tools and Data
        if not self.is_level1() and instrument is None:
            instrument = orb.utils.misc.read_instrument_value_from_file(self.cube_path)
            
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

        super().__init__(self, instrument=instrument,
                         data_prefix=data_prefix,
                         config=config, **kwargs)

        # checking dims
        if self.data.ndim != 3:
            raise TypeError('input cube has {} dims but must have exactly 3 dimensions'.format(self.data.ndim))
        
        if indexer is not None:
            if not isinstance(indexer, orb.core.Indexer):
                raise TypeError('indexer must be an orb.orb.core.Indexer instance')
        self.indexer = indexer


    def __getitem__(self, key):
        """Implement getitem special method"""
        if self.is_level1():
            if self.has_dataset('mask'):
                logging.warning('mask is not handled for old cubes format')
            return self.oldcube.__getitem__(key)
        
        with self.open_hdf5() as f:
            _data = np.copy(f['data'].__getitem__(key))
            
            if 'mask' in f:
                _data *= f['mask'].__getitem__((key[0], key[1]))

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
        header = orb.utils.io.header_hdf52fits(self.get_dataset('header', protect=False))
        params = dict()
        for ikey in header:
            if ikey in translate:
                params[translate[ikey][0]] = translate[ikey][1](header[ikey])
            elif ikey != 'COMMENT':
                params[ikey] = header[ikey]
                
        return params
        
    def is_level1(self):
        """Return True if cube is level 1"""
        return self.get_level() == 1

    def is_level2(self):
        """Return True if cube is level 2"""
        return self.get_level() == 2

    def is_level3(self):
        """Return True if cube is level 3"""
        return self.get_level() == 3
    
    def get_level(self):
        """Return reduction level of the cube.

        * level 1: old hdf5 architecture, real output, unit in
          erg/cm2/s/A, deep frame is the mean of the interferogram cube

        * level 2: new hdf5 architecture, real output, unit in
          erg/cm2/s/A, deep frame is the sum of the interferogram cube

        * level 3: new hdf5 architecture, complex output, unit in
          counts, data can be calibrated via flambda parameter :
          spectrum *= cube.params.flambda / cube.dimz /
          cube.exposure_time, deep frame is the sum of the
          interferogram cube

        """
        try:
            return int(self.level)
        except AttributeError: pass
        
        with self.open_hdf5('r') as f:
            self.level = 1
            if 'level2' in f.attrs:
                self.level = 2
            if 'level3' in f.attrs:
                self.level = 3
                if 'level2' in f.attrs:
                    logging.warning('both level2 and level3 in attrs')
            
            if self.level == 1:
                logging.warning('old cube architecture (level 1). IO performances could be reduced.')
                
        return self.level
                    
    
    def copy(self):
        raise NotImplementedError('HDFCube instance cannot be copied')
    
    def open_hdf5(self, mode='r'):
        """Return a handle on the hdf5 file.

        :param mode: opening mode. can be 'r' or 'a'. 
        """
        if mode not in ['r', 'a', 'r+']:
            raise ValueError('mode is {} and must be r, r+ or a'.format(mode))

        return orb.utils.io.open_hdf5(self.cube_path, mode)
        
    
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
        if self.is_level1():
            return self.oldcube.get_data(
                x_min, x_max, y_min, y_max, z_min, z_max,
                silent=silent)
        
        return self[x_min:x_max, y_min:y_max, z_min:z_max]

    def get_region(self, region, integrate=True):
        """Return a list of valid pixels from a ds9 region file or a ds9-style
        region definition

        e.g. for a circle defined in celestial coordinates:
          "fk5;circle(290.96388,14.019167,843.31194")"

        :param region: a ds9 region file path or a ds9-style
          region definition as a string.

        :param integrate: Used when multiple regions are defined. If
          True, all regions are integrated. If False, a list of
          individual regions is returned.
        """
        return orb.utils.misc.get_mask_from_ds9_region_file(
            region, [0, self.dimx], [0, self.dimy], integrate=integrate,
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
        if not orb.utils.validate.is_iterable(region[0], raise_exception=False):
            raise TypeError('badly formatted region.')
        if not orb.utils.validate.is_iterable(region[1], raise_exception=False):
            raise TypeError('badly formatted region.')
        if not len(region[0]) == len(region[1]):
            raise TypeError('badly formatted region.')

        if len(region[0]) == 1:
            return np.atleast_2d(self[int(region[0][0]), int(region[1][0]), :])

        xmin = self.validate_x_index(np.nanmin(region[0]), clip=False)
        xmax = self.validate_y_index(np.nanmax(region[0]), clip=False) + 1
        ymin = self.validate_x_index(np.nanmin(region[1]), clip=False)
        ymax = self.validate_y_index(np.nanmax(region[1]), clip=False) + 1

        if (xmax - xmin)  * (ymax - ymin) > SIZE_LIMIT:
            raise Exception('size limit exceeded, try a smaller region')

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
        with self.open_hdf5() as f: 
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
        
        with self.open_hdf5() as f:
            if path not in f:
                raise AttributeError('{} dataset not in the hdf5 file'.format(path))
            return f[path][:]


    def get_dataset_attrs(self, path):
        """Return the attributes attached to a dataset
        """
        with self.open_hdf5('r') as f:
            if path not in f:
                raise AttributeError('{} dataset not in the hdf5 file'.format(path))
            attrs = f[path].attrs
            out = dict()
            for i in list(attrs.keys()):
                out[i] = attrs[i]
            return out


    def get_datasets(self):
        """Return all datasets contained in the cube
        """
        ds = list()
        with self.open_hdf5('r') as f:
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
        image0_bin = orb.utils.image.nanbin_image(
            self.get_data_frame(0), binning)

        cube_bin = np.empty((image0_bin.shape[0],
                             image0_bin.shape[1],
                             self.dimz), dtype=float)
        cube_bin.fill(np.nan)
        cube_bin[:,:,0] = image0_bin
        progress = orb.core.ProgressBar(self.dimz-1)
        for ik in range(1, self.dimz):
            progress.update(ik, info='Binning cube')
            cube_bin[:,:,ik] = orb.utils.image.nanbin_image(
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
        progress = orb.core.ProgressBar(self.dimz)
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

    
    def get_deep_frame(self, recompute=False, compute=True):
        """Return the deep frame of a cube (in counts, i.e. e- x gain).


        :param compute: (Optional) If True, deep frame can be computed
          if not already present. If False, raise an exception when
          deep frame is not already present (default True).

        :param recompute: (Optional) Force deep frame computation
          even if it is already present in the cube (default False).

        ... warning:: deep frame computation should only be done on an
          interferogram cube. Deep frame computed on a spectral cube
          is *much* more noisy by definition.
        
        .. note:: In this process NaNs are handled correctly

        """
        if not recompute:
            try:
                im = orb.image.Image(self.deep_frame, params=self.params)
                if self.has_dataset('dxmap') and self.has_dataset('dymap'):
                    im.set_dxdymaps(self.get_dxdymaps()[0], self.get_dxdymaps()[1])
                return im
            except AttributeError: pass

        df = None
        if not recompute:
            if self.has_dataset('deep_frame'):
                df = self.get_dataset('deep_frame', protect=False)
                if self.is_level1():
                    df *= self.dimz
        
            elif self.is_level1():
                if compute:
                    df = self.oldcube.get_mean_image(recompute=recompute) * self.dimz
                else: raise orb.utils.err.DeepFrameError('deep frame not already computed.')
                    
        if df is None:
            df = self.compute_sum_image()

        self.deep_frame = df
        df = orb.image.Image(self.deep_frame, params=self.params)
        if self.has_dataset('dxmap') and self.has_dataset('dymap'):
            df.set_dxdymaps(self.get_dxdymaps()[0], self.get_dxdymaps()[1])
        return df
            
    
    def get_phase_maps(self):
        """Return a PhaseMaps instance if phase maps are set
        """
        try:
            return orb.fft.PhaseMaps(self.cube_path)
        except Exception:
            return None

    def get_high_order_phase(self):
        """Return high order phase.

        This is just the config high order phase
        """
        try:
            path = self._get_phase_file_path(self.params.filter_name)
        except Exception as e:
            logging.warning('No high order phase loaded for filter {}!: {}'.format(
                self.params.filter_name, e))
            path = None
        
        if path is not None:
            return orb.fft.HighOrderPhaseCube(path)
        else:
            logging.warning('No high order phase loaded for filter {}!'.format(
                self.params.filter_name))
            return None

    def get_dxdymaps(self):
        """Return dxdymaps.
        """
        return (self.get_dataset('dxmap'),
                self.get_dataset('dymap'))
    
    def compute_sum_image(self, step_size=100):
        """compute the sum along z axis
        """
        sum_im = None
        progress = orb.core.ProgressBar(self.dimz)
        for ik in range(0, self.dimz, step_size):
            progress.update(ik, info="Creating sum image")
            frames = self[:,:,ik:ik+step_size]
            if sum_im is None: # avoid creating a zeros frame with a
                               # possibly uncompatible dtype
                sum_im = np.nansum(frames, axis=2)
            else:
                sum_im += np.nansum(np.reshape(
                    frames, (frames.shape[0], frames.shape[1],
                             int(frames.size//(frames.shape[0] * frames.shape[1])))),
                                    axis=2)
            
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

        return orb.core.Tools._get_quadrant_dims(
            self, quad_number, dimx, dimy, div_nb)

    def writeto(self, path, div_nb=4):
        """Write data to an hdf file

        :param path: hdf file path.

        :param div_nb: cube is extracted by quadrant to avoid feeling
          the RAM. In case of memory error just use a higher number of
          divisions.
        """
        quad_nb = int(div_nb)**2
            
        old_keys = 'quad', 'frame', 'data'
        def is_exportable(key):
            for iold in old_keys:
                if iold in key:
                    return False
            return True

        with orb.utils.io.open_hdf5(path, 'w') as fout:
            fout.attrs['level3'] = True
            
            with self.open_hdf5() as f:
                for iattr in f.attrs:
                    fout.attrs[iattr] = f.attrs[iattr]

                # write datasets
                for ikey in f:
                    if is_exportable(ikey):
                        logging.info('adding {}'.format(ikey))
                        ikeyval = f[ikey]
                        if isinstance(ikeyval, np.ndarray):
                            ikeyval = orb.utils.io.cast_storing_dtype(ikeyval)
                        fout.create_dataset(ikey, data=ikeyval, chunks=True)

                # write data
                dtype = orb.utils.io.get_storing_dtype(self[0,0,0])
                fout.create_dataset('data', shape=self.shape, dtype=dtype, chunks=True)

                for iquad in range(quad_nb):
                    xmin, xmax, ymin, ymax = self.get_quadrant_dims(iquad, div_nb=div_nb)
                    logging.info('loading quad {}/{}'.format(
                        iquad+1, quad_nb))

                    data = self.get_data(
                        xmin, xmax, ymin, ymax, 0, self.dimz, silent=False)
                    data = orb.utils.io.cast_storing_dtype(data)
                    logging.info('writing quad {}/{}'.format(
                        iquad+1, quad_nb))
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
        df = orb.image.Image(np.empty((self.dimx, self.dimy), dtype=float),
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
        # https://docs.astropy.org/en/stable/generated/examples/io/skip_create-large-fits.html

        wcshdr = self.get_wcs().to_header()

        hdr = astropy.io.fits.PrimaryHDU().header
        hdr['NAXIS'] = 3
        hdr['NAXIS1'] = self.dimx
        hdr['NAXIS2'] = self.dimy
        hdr['NAXIS3'] = self.dimz
        hdr['BITPIX'] = (-32, 'np.float32')
        hdr.update(wcshdr)
        
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        
        shdu = astropy.io.fits.StreamingHDU(path, hdr)
        progress = orb.core.ProgressBar(self.dimz)
        for iz in range(self.dimz):
            progress.update(iz, info='Exporting frame {}'.format(iz))
            shdu.write(self[:,:,iz].real.astype(np.float32).T)
        progress.end()
        shdu.close()
            
    def get_frame_header(self, index):
        """Return the header of a frame given its index in the list.

        The header is returned as an instance of pyfits.Header().

        :param index: Index of the frame
        """
        return orb.utils.io.header_hdf52fits(
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
                    raise Exception("no calibration laser map in the hdf file. 'calibration_laser_map_path' must be set in params")
                calib_path = self.params.calibration_laser_map_path
                if not os.path.exists(calib_path):
                    if not os.path.isabs(calib_path):
                        calib_path = os.path.join(os.path.dirname(self.cube_path), calib_path)
                
                self.calibration_laser_map = orb.utils.io.read_fits(calib_path)
                if 'cropped_bbox' in self.params:
                    xmin, xmax, ymin, ymax = self.params.cropped_bbox
                    self.calibration_laser_map = self.calibration_laser_map[xmin:xmax, ymin:ymax]
                    
        if (self.calibration_laser_map.shape[0] != self.dimx):
            self.calibration_laser_map = orb.utils.image.interpolate_map(
                self.calibration_laser_map, self.dimx, self.dimy)
        if not self.calibration_laser_map.shape == (self.dimx, self.dimy):
            raise Exception('Calibration laser map shape is {} and must be ({} {})'.format(self.calibration_laser_map.shape[0], self.dimx, self.dimy))
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
            self.theta_map = orb.utils.spectrum.corr2theta(self.get_calibration_coeff_map())

    def validate_x_index(self, x, clip=True):
        """validate an x index, return an integer inside the boundaries or
        raise an exception if it is off boundaries

        :param x: x index

        :param clip: (Optional) If True return an index inside the
          boundaries, else: raise an exception (default True).
        """
        return orb.utils.validate.index(x, 0, int(self.dimx), clip=clip)
    
    def validate_y_index(self, y, clip=True):
        """validate an y index, return an integer inside the boundaries or
        raise an exception if it is off boundaries

        :param y: y index (can be an array or a list of indexes)

        :param clip: (Optional) If True return an index inside the
          boundaries, else: raise an exception (default True).
        """
        return orb.utils.validate.index(y, 0, int(self.dimy), clip=clip)


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
            raise Exception('master combination is useful for a small set of frames')
        
        if combine is None:
            combine='average'
        if reject is None:
            reject='avsigclip'

        if not self.config.BIG_DATA:
            master = orb.utils.image.create_master_frame(
                self[:,:,:], combine=combine, reject=reject)
        else:
            master = orb.utils.image.pp_create_master_frame(
                self[:,:,:], combine=combine, reject=reject,
                ncpus=self.config.NCPUS)

        return master


    def get_axis(self, x, y):
        """Return the axis at x, y"""
        return None

    def get_zvector_from_region(self, region, median=False):
        """Return an orb.fft.Vector1d instance integrated over a given region

        :param region: A ds9-like region file or a list of pixels
          having the same format as the list returned by np.nonzero(),
          i.e. (x_positions_1d_array, y_positions_1d_array).


        :param median: If True, a median is used instead of a mean to
          combine vectors. As the resulting vector is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        """
        if isinstance(region, str):
            region = self.get_region(region)
        
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[region])
        
        vectors = self.get_data_from_region(region)
        if not median:
            vector = np.nansum(vectors, axis=0)
        else:
            # if vectors are complex, the median of the imaginary part
            # and the median of the real part must be computed
            # independanlty due to a bug in
            # numpy. https://github.com/numpy/numpy/issues/12943
            if np.iscomplexobj(vectors):
                vector = np.nanmedian(vectors.real, axis=0).astype(vectors.dtype)
                vector.imag = np.nanmedian(vectors.imag, axis=0)
            else:    
                vector = np.nanmedian(vectors, axis=0)
            vector *= len(vectors)

        params = dict(self.params)
        params['pixels'] = len(vectors)
        return orb.core.Vector1d(vector, params=params,
                                 zpd_index=self.params.zpd_index,
                                 calib_coeff=calib_coeff,
                                 axis=np.arange(self.dimz))
        
    def get_zvector(self, x, y, r=0, return_region=False, median=False):
        """Return an orb.fft.Vector1d instance taken at a given position in x, y.
        
        :param x: x position 
        
        :param y: y position 

        :param r: (Optional) If r > 0, vector is integrated over a
          circular aperture of radius r. In this case the number of
          pixels is returned as a parameter: pixels

        :param median: If True, a median is used instead of a mean to
          combine vectors. As the resulting vector is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        :param return_region: (Optional) If True, region is returned
          also (default False)

        """
        x = self.validate_x_index(int(x), clip=False)
        y = self.validate_y_index(int(y), clip=False)
        
        region = self.get_region('circle({},{},{})'.format(x+1, y+1, r))
        vec = self.get_zvector_from_region(region, median=median)
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
            raise Exception('airmass not set and no frame headers')

        airmass = list()
        for i in range(self.dimz):
            try:
                airmass.append(float(self.get_frame_header(i)['AIRMASS']))
            except KeyError:
                raise Exception('frame_header_{} not present'.format(i))
        self.params['airmass'] = np.array(airmass)
        return self.params.airmass

    def get_detection_frame(self):
        if self.has_dataset('deep_frame') or hasattr(self, 'deep_frame'):
            logging.info('detecting stars using the deep frame')
            df = self.get_deep_frame().data
        else:
            logging.info('detecting stars using a stack of the first {} frames'.format(
                self.config.DETECT_STACK))
            _stack = self[:,:,:self.config.DETECT_STACK]
            df = np.nanmedian(_stack, axis=2)
        return orb.image.Image(df, params=self.params)
        
        
    def detect_stars(self, **kwargs):
        """Detect valid stars in the image

        :param kwargs: orb.image.Image.detect_stars kwargs.
        """
        return self.get_detection_frame().detect_stars(**kwargs)

    def detect_fwhm(self, star_list):
        return self.get_detection_frame().detect_fwhm(star_list)


    def fit_stars_in_frame(self, star_list, index, **kwargs):
        """Fit stars in frame

        :param star_list: Path to a list of stars

        :param index: frame index

        :param kwargs: orb.image.Image.fit_stars kwargs.
        """
        im = orb.image.Image(self[:,:,index], params=self.params)
        return im.fit_stars(star_list, **kwargs)

    def fit_stars_in_cube(self, star_list,
                          path=None,
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
    
        :param path: (Optional) path where the results are saved to.

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

        :param kwargs: (Optional) orb.utils.astrometry.fit_stars_in_frame
          kwargs.

        """
        def _fit_stars_in_frame(frame, star_list, fwhm_pix, params, kwargs):

            import warnings
            warnings.simplefilter('ignore')
            
            im = orb.image.Image(frame, params=params)
            if fwhm_pix is not None:
                im.reset_fwhm_arc(im.pix2arc(fwhm_pix))
            return im.fit_stars(star_list, **kwargs)

        FOLLOW_NB = 5 # Number of deviation value to get to follow the
                      # stars

        if alignment_vectors is not None:
            if len(alignment_vectors) != 2:
                raise TypeError('alignment_vectors must be a tuple of 2 vectors')
            if len(alignment_vectors[0]) != self.dimz or len(alignment_vectors[1]) != self.dimz:
                raise TypeError('each vector of alignment_vectors must have the same len as the number of frames in the cube ({} or {} != {})'.format(len(alignment_vectors[0]), len(alignment_vectors[1]), self.dimz))
            
            correct_alignment = False
                
        star_list = orb.utils.astrometry.load_star_list(star_list)
                
        logging.info("Fitting stars in cube")

        fit_results = list([None] * self.dimz)
        dx_mean = list([np.nan] * self.dimz)
        dy_mean = list([np.nan] * self.dimz)
        fwhm_mean = list([np.nan] * self.dimz)
        
        if self.dimz < 2: raise Exception(
            "Data must have 3 dimensions. Use fit_stars_in_frame method instead")
        
        if add_cube is not None:
            if np.size(add_cube) >= 2:
                added_cube = add_cube[0]
                added_cube_scale = add_cube[1]
                if not isinstance(added_cube, Cube):
                    raise Exception('Added cube must be a Cube instance. Check add_cube option')
                if np.size(added_cube_scale) != 1:
                    raise Exception('Bad added cube scale. Check add_cube option.')

        # Init of the multiprocessing server
        job_server, ncpus = self._init_pp_server()
        
        progress = orb.core.ProgressBar(int(self.dimz))
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
                fwhm_pix = np.nanmean(orb.utils.stats.sigmacut(fwhm_mean[ik-FOLLOW_NB:ik]))
                if np.isnan(fwhm_pix): fwhm_pix = None
          
            # load data
            progress.update(ik, info="loading: " + str(ik))
            frames = np.copy(self[:,:,ik:ik+ncpus])
            if add_cube is not None:
                frames += added_cube[:,:,ik:ik+ncpus] * added_cube_scale
            frames = np.atleast_3d(frames)
                
            # get stars photometry for each frame
            params = self.params.convert()
            progress.update(ik, info="computing photometry: " + str(ik))
            jobs = [(ijob, job_server.submit(
                _fit_stars_in_frame,
                args=(frames[:,:,ijob], star_lists[ijob],
                      fwhm_pix,
                      params,
                      dict(kwargs)),
                modules=("import logging",
                         "import orb.utils.stats",
                         "import orb.utils.image",
                         'import orb.image',
                         "import numpy as np",
                         "import math",
                         "import orb.cutils",
                         "import warnings",
                         "from orb.utils.astrometry import *")))
                    for ijob in range(ncpus)]

            for ijob, job in jobs:
                res = job()
                fit_results[ik+ijob] = res
                if res is not None:
                    if not res.empty:
                        if 'dx' in res and 'dy' in res:
                            dx_mean[ik+ijob] = np.nanmean(orb.utils.stats.sigmacut(res['dx'].values))
                            dy_mean[ik+ijob] = np.nanmean(orb.utils.stats.sigmacut(res['dy'].values))
                        if 'fwhm_pix' in res:
                            fwhm_mean[ik+ijob] = np.nanmean(orb.utils.stats.sigmacut(res['fwhm_pix'].values))
            

        progress.end()
        self._close_pp_server(job_server)
        
        
        if path is not None:
            orb.utils.io.save_dflist(fit_results, path)
            logging.info('fit results saved as {}'.format(path))
        
        return fit_results

    def get_raw_alignement_vectors(self, star_number=60, searched_size=80):
        """Return raw alignment vectors based on brute force.

        Slow but useful when alignment between frames is very bad.

        :return: dx, dy vectors
        """
        star_list_init, fwhm = orb.image.Image(
            self[:,:,0], instrument=self.instrument).detect_stars(
                min_star_number=star_number)
        star_list_init = orb.utils.astrometry.df2list(star_list_init)
        dxs = list()
        dys = list()
        progress = orb.core.ProgressBar(self.dimz-1)
        dxs.append(0)
        dys.append(0)
        for iframe in range(1, self.dimz):
            x_range = np.linspace(-searched_size,searched_size,searched_size)
            y_range = np.linspace(-searched_size,searched_size,searched_size)
            r_range = [0]
            dx, dy, _, _ = orb.utils.astrometry.brute_force_guess(
                self[:,:,iframe], star_list_init, x_range, y_range,r_range,
                [self.dimx/2., self.dimy/2.], 1, 15, verbose=False)
            
            progress.update(iframe-1, info='{:.2f}, {:.2f}'.format(dx, dy))
            dxs.append(dx)
            dys.append(dy)
        progress.end()
        return np.array(dxs), np.array(dys)
            
    
    def get_alignment_vectors(self, star_list, min_coeff=0.2,
                              alignment_vectors=None, path=None):
        """Return alignement vectors

        :param star_list: list of stars

        :param min_coeff: The minimum proportion of stars correctly
            fitted to assume a good enough calculated disalignment
            (default 0.2).

        :param path: If not None, fit results are written to this path

        :return: alignment_vector_x, alignment_vector_y, alignment_error
        """
        star_list = orb.utils.astrometry.load_star_list(star_list)
        # warning: multi_fit must stay at False (if multi_fit is True,
        # a problem on one star affects evrything else)
        fit_results = self.fit_stars_in_cube(star_list,
                                             path=path,
                                             correct_alignment=True,
                                             alignment_vectors=alignment_vectors,
                                             no_aperture_photometry=True,
                                             multi_fit=False,
                                             # must star af FALSE
                                             fix_height=False,
                                             filter_background=True)
            
        return orb.utils.astrometry.compute_alignment_vectors(fit_results)
    

#################################################
#### CLASS RWHDFCube ############################
#################################################
class RWHDFCube(HDFCube):
    
    def __init__(self, path, shape=None, instrument=None, reset=False, dtype=np.float32, **kwargs):
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
                logging.info('deleting {} before writing a new cube'.format(path))
                os.remove(path)

        # create file if it does not exists
        if not os.path.exists(path):
            logging.info('Creating cube {}'.format(path))
            if shape is None:
                raise ValueError('cube does not exist. If you want to create one, shape must be set.')
            
            with orb.utils.io.open_hdf5(path, 'w') as f:
                orb.utils.validate.has_len(shape, 3, object_name='shape')
                f.create_dataset('data', shape=shape, chunks=True, dtype=dtype)
                f.attrs['level3'] = True
                f.attrs['instrument'] = instrument
                f.attrs['program'] = 'ORB version {}'.format(orb.version.__version__)
                f.attrs['author'] = 'thomas.martin.1@ulaval.ca'
                f.attrs['date'] = str(datetime.datetime.now())


        elif shape is not None:
            raise ValueError('shape or dtype must be set only when creating a new HDFCube')

        super().__init__(path, instrument=instrument, **kwargs)
        
        if self.is_level1(): raise Exception('Old cubes are not writable. Please export the old cube to a new cube with writeto()')

        if self.has_params:
            self.params['program'] = 'ORB version {}'.format(orb.version.__version__)
            self.params['author'] = 'thomas.martin.1@ulaval.ca'
            self.params['date'] = str(datetime.datetime.now())
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

        with self.open_hdf5('a') as f:
            if f['data'].dtype != value.dtype:
                logging.warning('wrong types: cube is {} and new data is {}'.format(
                    f['data'].dtype, value.dtype))
                # warning !! never do the following since all data is
                # reset, if only a part of the data must be set this
                # is just insane
                #del f['data']
                #f.create_dataset('data', shape=self.data.shape, dtype=value.dtype, chunks=True)
                value = value.astype(f['data'].dtype)
            f['data'].__setitem__(key, value)

    def set_param(self, key, value):
        """Set class parameter

        :param key: parameter key

        :param value: parameter value
        """
        self.params[key] = value
        with self.open_hdf5('a') as f:
            _update = True
            value = orb.utils.io.cast2hdf5(value)
            try:
                if key in f.attrs:
                    if np.all(f.attrs[key] == value):
                        _update = False
            except ValueError: pass
                
            if _update:
                try:
                    f.attrs[key] = value
                except TypeError:
                    logging.debug('error setting param {} ()'.format(key, type(value)))
                    
    def set_params(self, params):
        """Set class parameters

        :param params: a dict of parameters
        """
        for ipar in params:
            try:
                self.set_param(ipar, params[ipar])
            except TypeError:
                logging.warning('error setting param {}'.format(ipar))

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
        with self.open_hdf5('a') as f:
            if path in f:
                del f[path]
                logging.warning('{} dataset changed'.format(path))

            if isinstance(data, dict):
                data = orb.utils.io.dict2array(data)

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
                         orb.utils.io.header_fits2hdf5(header))

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

        if not isinstance(phase_maps, orb.fft.PhaseMaps):
            raise TypeError('phase_maps must be a PhaseMaps instance')

        for ipar in check_params:
            paramok = True
            if phase_maps.params[ipar] != self.params[ipar]:
                paramok = False
                if ipar == 'filter_name':
                    if ((phase_maps.params[ipar] in self.params[ipar])
                        or (self.params[ipar] in phase_maps.params[ipar])):
                        paramok = True
            if not paramok:        
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
        #     logging.warning('calibration laser map unchanged since it already exists')

    def set_standard_image(self, std_im):
        """Set standard image

        :param std_im: An image.Image instance or a path to an
          hdf5 image.

        """
        try:
            std_im = orb.image.Image(std_im, instrument=self.instrument)
        except ValueError:
            raise TypeError('std_im must be an orb.image.Image instance or a path to valid hdf5 image')
        self.set_dataset('standard_image', std_im.data, protect=False)

        with self.open_hdf5('a') as f:
            for iparam in std_im.params:
                f['standard_image'].attrs[iparam] = std_im.params[iparam]

    def set_standard_spectrum(self, std_sp):
        """Set standard spectrum

        :param std_sp: a orb.photometry.StandardSpectrum instance or a
          path to an hdf5 spectrum.
        """
        std_sp = orb.fft.StandardSpectrum(std_sp, instrument=self.instrument)
        try:
            std_sp = orb.fft.StandardSpectrum(std_sp, instrument=self.instrument)
        except Exception as e:
            raise TypeError('std_sp must be a StandardSpectrum instance or a path to a valid hdf5 spectrum: {}'.format(e))

        self.set_dataset('standard_spectrum', std_sp.data, protect=False)
        
        with self.open_hdf5('a') as f:
            for iparam in std_sp.params:
                if iparam == 'COMMENT':
                    continue
                f['standard_spectrum'].attrs[iparam] = std_sp.params[iparam]
            f['standard_spectrum'].attrs['axis'] = std_sp.axis.data
            
    def set_dxdymaps(self, dxmap, dymap):
        """Set dxdymaps

        :param dxmap: dxmap
        :param dymap: dymap
        """
        orb.utils.validate.is_2darray(dxmap, object_name='dxmap')
        orb.utils.validate.is_2darray(dymap, object_name='dymap')
        if dxmap.shape != (self.dimx, self.dimy) or dymap.shape != (self.dimx, self.dimy):
            raise TypeError('dxmap and dymap must have shape ({},{})'.format(
                self.dimx, self.dimy))
        self.set_dataset('dxmap', dxmap, protect=False)
        self.set_dataset('dymap', dymap, protect=False)
                
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
        super().__init__(path, instrument=instrument, params=params, **kwargs)
        
        # compute additional parameters
        self.filterfile = orb.core.FilterFile(self.params.filter_name)
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
            raise Exception("class not valid: set instrument to 'sitelle' or 'spiomm' at init")
        else: self.set_param('instrument', self.instrument)
        try: self.params_defined
        except AttributeError: raise Exception("class not valid: set params at init")
        if not self.params_defined: raise Exception("class not valid: set params at init")

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
            base_axis = orb.utils.spectrum.create_cm1_axis(
                self.dimz, self.params.step, self.params.order,
                corr=self.get_axis_corr())
        else:
            base_axis = self.params.base_axis
            
        return orb.core.Axis(base_axis)

    def get_uncalibrated_filter_bandpass(self):
        """Return filter bandpass as two 2d matrices (min, max) in pixels"""
        filterfile = FilterFile(self.get_param('filter_name'))
        filter_min_cm1, filter_max_cm1 = orb.utils.spectrum.nm2cm1(
            filterfile.get_filter_bandpass())[::-1]
        
        cm1_axis_step_map = orb.cutils.get_cm1_axis_step(
            self.dimz, self.params.step) * self.get_calibration_coeff_map()

        cm1_axis_min_map = (self.params.order / (2 * self.params.step)
                            * self.get_calibration_coeff_map() * 1e7)
        if int(self.params.order) & 1:
            cm1_axis_min_map += cm1_axis_step_map
        filter_min_pix_map = (filter_min_cm1 - cm1_axis_min_map) / cm1_axis_step_map
        filter_max_pix_map = (filter_max_cm1 - cm1_axis_min_map) / cm1_axis_step_map
        
        return filter_min_pix_map, filter_max_pix_map
    
    def integrate(self, filter_function, xmin=None, xmax=None, ymin=None, ymax=None, split=100,
                  mean=True, square_filter=False):
        """Integrate a cube under a filter function and generate an image

        :param filter_function: Must be an orb.core.Cm1Vector1d
          instance or the name of a filter registered in orb/data/

        :param xmin: (Optional) lower boundary of the ROI along x axis (default
          None, i.e. min)

        :param xmax: (Optional) lower boundary of the ROI along y axis (default
          None, i.e. min)

        :param ymin: (Optional) upper boundary of the ROI along x axis (default
          None, i.e. max)

        :param ymax: (Optional) upper boundary of the ROI along y axis (default None, i.e. max)
        
        """
        if isinstance(filter_function, str):
            filter_function = orb.core.Vector1d(self._get_filter_file_path(filter_function))
            
        if not isinstance(filter_function, orb.core.Vector1d):
            raise TypeError('filter function must be an orb.core.Vector1d instance or the name of a registered filter: {}'.format(self.filters))


        start = np.argmax(filter_function.data > 0.05)
        end = np.argmin(filter_function.data[start:] > 0.05) + start
        
        if (filter_function.axis[start] <= self.params.base_axis[0]
            or filter_function.axis[end] >= self.params.base_axis[-1]):
            raise ValueError(
                'filter passband (>5%) between {} - {} out of cube band {} - {}'.format(
                    filter_function.axis[start],
                    filter_function.axis[end],
                    self.params.base_axis[0],
                    self.params.base_axis[-1]))
        
        if xmin is None: xmin = 0
        if ymin is None: ymin = 0
        if xmax is None: xmax = self.dimx
        if ymax is None: ymax = self.dimy

        xmin = int(np.clip(xmin, 0, self.dimx))
        xmax = int(np.clip(xmax, 0, self.dimx))
        ymin = int(np.clip(ymin, 0, self.dimy))
        ymax = int(np.clip(ymax, 0, self.dimy))

        start_pix, end_pix = orb.utils.spectrum.cm12pix(
            self.params.base_axis, [filter_function.axis[start], filter_function.axis[end]]).astype(int)
        
        sframe = np.zeros((self.dimx, self.dimy), dtype=np.complex128)
        zsize = end_pix - start_pix + 1
        # This splits the range in zsize//10 +1 chunks (not necessarily of same
        # size). The endpix is correctly handled in the extraction
        if split > 0:
            izranges = np.array_split(range(start_pix, end_pix+1), zsize//split+1)
        else:
            izranges = (np.arange(start_pix, end_pix+1),)
            
        progress = orb.core.ProgressBar(len(izranges))
        _index = 0
        for izrange in izranges:
            progress.update(_index)
            
            axis_onrange = orb.core.Axis(self.params.base_axis[izrange].astype(float))
            filter_onrange = filter_function.project(axis_onrange)
            filter_onrange = filter_onrange.data.astype(np.complex128)
            filter_onrange.imag = filter_onrange.real
            progress.update(_index, info='loading data')
            
            idata = self.get_data(
                xmin, xmax, ymin, ymax,
                izrange.min(), izrange.max()+1, silent=True).astype(np.complex128)
            progress.update(_index, info='data loaded')
            
            if not square_filter:
                sframe[xmin:xmax, ymin:ymax] += np.sum(
                    idata
                    * filter_onrange, axis=2).astype(np.complex128)
            else:
                sframe[xmin:xmax, ymin:ymax] += np.sum(
                    idata, axis=2).astype(np.complex128)

            _index += 1
        progress.end()
        if mean:
            sframe /= np.sum(filter_function.project(orb.core.Axis(self.params.base_axis.astype(float))).data)
        return sframe
        
#################################################
#### CLASS InteferogramCube #####################
#################################################
class InterferogramCube(Cube):
    """Provide additional methods for an interferogram cube when
    observation parameters are known.
    """

    def get_interferogram(self, x, y, r=0, median=False):
        """Return an orb.fft.Interferogram instance.

        See Cube.get_zvector for the parameters. 
        """
        vector, region = Cube.get_zvector(self, x, y, r=r, return_region=True)
        vector.params['source_counts'] = np.nansum(self.get_deep_frame().data[region])

        vector.axis = None

        err = np.ones(self.dimz, dtype=float) * np.sqrt(
            vector.params.source_counts / self.dimz * self.get_gain())
        
        return orb.fft.RealInterferogram(vector, err=err)

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
        return orb.fft.RealInterferogram(interf, params=self.params,
                                     zpd_index=self.params.zpd_index,
                                     calib_coeff=calib_coeff)

    
#################################################
#### CLASS FDCube ###############################
#################################################
class FDCube(orb.core.Tools):
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
        super().__init__(**kwargs)

        self.image_list_path = image_list_path

        self._image_mode = image_mode
        self._chip_index = chip_index

        if (self.image_list_path != ""):
            # read image list and get cube dimensions  
            image_list_file = orb.utils.io.open_file(self.image_list_path, "r")
            image_name_list = image_list_file.readlines()
            if len(image_name_list) == 0:
                raise Exception('No image path in the given image list')
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

                        image_data = orb.utils.io.read_fits(
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
                logging.info('sorting images')
                self.image_list = orb.utils.misc.sort_image_list(self.image_list,
                                                                 self._image_mode)
            
            self.image_list = np.array(self.image_list)
            self.dimz = self.image_list.shape[0]
            
            
            if (self.dimx) and (self.dimy) and (self.dimz):
                if not silent_init:
                    logging.info("Data shape : (" + str(self.dimx) 
                                    + ", " + str(self.dimy) + ", " 
                                    + str(self.dimz) + ")")
            else:
                raise Exception("Incorrect data shape : (" 
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
        if z_slice.stop == z_slice.start + 1:
            return data

        if self._parallel_access_to_data:
            # load other frames
            job_server, ncpus = self._init_pp_server(silent=self._silent_load) 

            if not self._silent_load:
                progress = orb.core.ProgressBar(z_slice.stop - z_slice.start - 1)
            
            for ik in range(z_slice.start + 1, z_slice.stop, ncpus):
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
                progress = orb.core.ProgressBar(z_slice.stop - z_slice.start - 1)
            
            for ik in range(z_slice.start + 1, z_slice.stop):

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
                    or isinstance(_slice.start, int)):
                    if (_slice.start >= 0) and (_slice.start <= _max):
                        slice_min = int(_slice.start)
                    else:
                        raise Exception(
                            "Index error: list index out of range")
                else:
                    raise Exception("Type error: list indices of slice must be integers")
            else: slice_min = 0

            if _slice.stop is not None:
                if (isinstance(_slice.stop, int)
                    or isinstance(_slice.stop, int)):
                    if _slice.stop < 0: # transform negative index to real index
                        slice_stop = _max + _slice.stop
                    else:  slice_stop = _slice.stop
                    if ((slice_stop <= _max)
                        and slice_stop > slice_min):
                        slice_max = int(slice_stop)
                    else:
                        raise Exception(
                            "Index error: list index out of range")

                else:
                    raise Exception("Type error: list indices of slice must be integers")
            else: slice_max = _max

        elif isinstance(_slice, int) or isinstance(_slice, int):
            slice_min = _slice
            slice_max = slice_min + 1
        else:
            raise Exception("Type error: list indices must be integers or slices")
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
        hdu = orb.utils.io.read_fits(self.image_list[frame_index],
                                 return_hdu_only=True)
        image = None
        stored_file_path = None

        if self._image_mode == 'sitelle': 
            if image is None:
                image = orb.utils.io.read_sitelle_chip(hdu, self._chip_index)
            section = image[x_slice, y_slice]

        elif self._image_mode == 'spiomm': 
            if image is None:
                image, header = orb.utils.io.read_spiomm_data(
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
            orb.utils.io.write_fits(stored_file_path, image, overwrite=True,
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
        hdu = orb.utils.io.read_fits(self.image_list[index],
                                 return_hdu_only=True)
        hdu.verify('silentfix')
        return hdu.header

    def get_cube_header(self):
        """
        Return the header of a cube from the header of the first frame
        by keeping only the general keywords.
        """
        def del_key(key, header):
            if '*' in key:
                key = key[:key.index('*')]
                for k in list(header.keys()):
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
            instrument=self.instrument, reset=True)

        if mask is not None:
            cube.set_mask(mask)

        if params is not None:
            cube.update_params(params)

        cube.set_header(self.get_cube_header())
        progress = orb.core.ProgressBar(self.dimz)
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
        super().__init__(*args, **kwargs)

        self.reset_params()
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
                raise Exception('if the spectral cube is not calibrated, xy must be provided')
            xy = [self.dimx//2, self.dimy//2]
        return orb.utils.spectrum.cm12pix(
            self.get_axis(int(xy[0]), int(xy[1])).data, self.get_filter_range())

    def get_deep_frame(self, recompute=False, compute=False):
        try:
            return Cube.get_deep_frame(self, recompute=recompute, compute=compute)
        except orb.utils.err.DeepFrameError:
            logging.warning("Deep frame not present in the cube. Replaced with a map of zeros. Please attach it with cube.deep_frame = orb.utils.io.read_fits('deep_frame.fits')")
            self.deep_frame = np.zeros((self.dimx, self.dimy), dtype=float)
            return self.get_deep_frame()
    
    def get_axis(self, x, y):
        """Return the spectral axis at x, y
        """
        orb.utils.validate.index(x, 0, int(self.dimx), clip=False)
        orb.utils.validate.index(y, 0, int(self.dimy), clip=False)
        
        axis = orb.utils.spectrum.create_cm1_axis(
            self.dimz, self.params.step, self.params.order,
            corr=self.get_calibration_coeff_map()[x, y])
        return orb.core.Axis(np.copy(axis))

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

        self.filterfile = orb.core.FilterFile(self.params.filter_name)

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
                    raise Exception('wavenumber_calibration is True but axis_corr is not set')
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
        resolution = orb.utils.spectrum.compute_resolution(
            self.dimz - self.params.zpd_index,
            self.params.step, self.params.order,
            self.params.axis_corr)
        self.set_param('resolution', resolution)

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
        self.set_param('line_fwhm', orb.utils.spectrum.compute_line_fwhm(
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
            raise Exception('No calibration laser map given. Please redo the last step of the data reduction')

        if self.has_wavenumber_calibration():
            calib_map = (np.ones((self.dimx, self.dimy), dtype=float)
                    * self.params.nm_laser * self.params.axis_corr)

        elif (calib_map.shape[0] != self.dimx):
            calib_map = orb.utils.image.interpolate_map(
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
        return orb.utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index, self.params.step, self.params.order,
            self.params.apodization, self.get_calibration_coeff_map_orig(),
            wavenumber=self.params.wavenumber)

    def get_theta_map(self):
        """Return the incident angle map (in degree)"""
        return orb.utils.spectrum.corr2theta(self.get_calibration_coeff_map_orig())

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

    def get_spectrum_from_region(self, region, median=False, mean_flux=False):
        """Return the integrated spectrum of a given region.

        :param region: A ds9-like region file or a list of pixels
          having the same format as the list returned by np.nonzero(),
          i.e. (x_positions_1d_array, y_positions_1d_array).

        :param median: If True, a median is used instead of a mean to
          combine spectra. As the resulting spectrum is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        :param mean_flux: If True, the mean spectrum (ie per pixel
          flux) is returned.

        .. note:: the region must not have a size greater than 400x400
          pixels. If you really need a larger region, you can split
          you region into smaller ones and combine the resulting
          spectra.
        """
        if not self.has_wavenumber_calibration():
            logging.warning('spectral cube is not calibrated in wavenumber, a large region may result in a deformation of the ILS.')
            
        if isinstance(region, str):
            region = self.get_region(region)
        
        spectra = self.get_data_from_region(region)
        if not median:
            spectrum = np.nansum(spectra, axis=0)
        else:
            # if spectra are complex, the median of the imaginary part
            # and the median of the real part must be computed
            # independanlty due to a bug in
            # numpy. https://github.com/numpy/numpy/issues/12943
            if np.iscomplexobj(spectra):
                spectrum = np.nanmedian(spectra.real, axis=0).astype(spectra.dtype)
                spectrum.imag = np.nanmedian(spectra.imag, axis=0)
            else:    
                spectrum = np.nanmedian(spectra, axis=0)
            spectrum *= len(spectra)

        # calculate number of integrated pixels
        params = dict(self.params)
        params['pixels'] = len(spectra)

        # compute axis
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[tuple(region)])
        calib_coeff_orig = np.nanmean(self.get_calibration_coeff_map_orig()[tuple(region)])
        axis = orb.utils.spectrum.create_cm1_axis(
            self.dimz, self.params.step, self.params.order,
            corr=calib_coeff)
        
        # compute counts and err
        counts = np.nansum(self.get_deep_frame().data[tuple(region)])
        err = np.ones(self.dimz, dtype=float) * np.sqrt(counts  * self.get_gain())
        err = orb.core.Cm1Vector1d(err, axis, params=params)
        if isinstance(self.params.flambda, float):
            flambda = np.ones(self.dimz, dtype=float) * self.params.flambda
        else:
            flambda = np.copy(self.params.flambda)
        flambda = orb.core.Cm1Vector1d(
            flambda, self.get_base_axis(), params=params)
        err = err.multiply(flambda)
        
        params['source_counts'] = counts
        params['calib_coeff'] = calib_coeff
        params['calib_coeff_orig'] = calib_coeff_orig

        if mean_flux:
            spectrum /= params['pixels']
            err.data /= params['pixels']
            params['pixels'] = 1
            
        return orb.fft.RealSpectrum(spectrum, err=err.data, axis=axis, params=params)
                

    def get_spectrum(self, x, y, r=0, median=False, mean_flux=False):
        """Return a orb.fft.RealSpectrum extracted at x, y and integrated
        over a circular aperture or radius r.

        :param x: x position 
        
        :param y: y position 

        :param r: (Optional) If r > 0, vector is integrated over a
          circular aperture of radius r. In this case the number of
          pixels is returned as a parameter: pixels

        :param median: If True, a median is used instead of a mean to
          combine spectra. As the resulting spectrum is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        :param mean_flux: If True, the mean spectrum (ie per pixel
          flux) is returned.
        """
        x = self.validate_x_index(x, clip=False)
        y = self.validate_y_index(y, clip=False)
        region = self.get_region('circle({},{},{})'.format(x+1, y+1, r))
        return self.get_spectrum_from_region(region, median=median, mean_flux=mean_flux)

    def get_spectrum_in_annulus(self, x, y, rmin, rmax, median=False, mean_flux=False):
        """Return a. orb.fft.RealSpectrum extracted at x, y and integrated
        over a circular annulus of min radius rmin and max radius rmax.

        :param x: x position 
        
        :param y: y position 

        :param rmin: rmin of the annulus

        :param rmax: rmax of the annulus

        :param median: If True, a median is used instead of a mean to
          combine spectra. As the resulting spectrum is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        :param mean_flux: If True, the mean spectrum (ie per pixel
          flux) is returned.
        """
        x = self.validate_x_index(x, clip=False)
        y = self.validate_y_index(y, clip=False)
        if rmin <= 0: raise ValueError('rmin must be > 0, use get_spectrum to extract spectrum in a circular aperture')
        if rmax <= rmin: raise ValueError('rmax must be greater than rmin')
        region = self.get_region('annulus({},{},{},{})'.format(x+1, y+1, float(rmin), float(rmax)))
        return self.get_spectrum_from_region(region, median=median, mean_flux=mean_flux)

    def get_spectrum_bin(self, x, y, b, median=False, mean_flux=False):
        """Return a spectrum integrated over a binned region.

        :param x: X position of the bottom-left pixel

        :param y: Y position of the bottom-left pixel

        :param b: Binning. If 1, only the central pixel is extracted

        :param median: If True, a median is used instead of a mean to
          combine spectra. As the resulting spectrum is integrated,
          the median value of the combined spectra is then scaled to
          the number of integrated pixels.

        :param mean_flux: If True, the mean spectrum (ie per pixel
          flux) is returned.

        :returns: (axis, spectrum)
        """
        if not isinstance(b, int):
            raise TypeError('b must be an int')
        if b < 1: raise Exception('Binning must be at least 1')
        
        x = self.validate_x_index(x, clip=False)
        y = self.validate_y_index(y, clip=False)
        region = self.get_region('box({},{},{},{},0)'.format(
            x+float(b)/2.+0.5, y+float(b)/2.+0.5, b, b))
        return self.get_spectrum_from_region(region, median=median, mean_flux=mean_flux)

    def get_mask_from_ds9_region_file(self, region, integrate=True):
        """Return a mask from a ds9 region file.

        :param region: Path to a ds9 region file.

        :param integrate: (Optional) If True, all pixels are integrated
          into one mask, else a list of region masks is returned (default
          True)
        """
        if isinstance(region, str):
            return orb.utils.misc.get_mask_from_ds9_region_file(
                region,
                [0, self.dimx],
                [0, self.dimy],
                header=self.get_header(),
                integrate=integrate)
        else: return region

    def get_standard_spectrum(self):
        """Return standard spectrum
        """
        if self.has_dataset('standard_spectrum'):
           std_sp = self.get_dataset('standard_spectrum', protect=False)
           hdr = self.get_dataset_attrs('standard_spectrum')
           return orb.fft.StandardSpectrum(std_sp, axis=hdr['axis'],
                                              params=hdr, instrument=self.instrument)
        
        if self.has_param('standard_path'):
            std_flux_sp = orb.fft.StandardSpectrum(self.params.standard_path)
            return std_flux_sp
        
        else: raise Exception('standard spectrum dataset or standard_path parameter are not defined')
        
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
            else: raise Exception('if no standard image can be found in the archive, standard_image_path must be set')
            return orb.image.StandardImage(
                std_im, params=std_cube.params)
        else:
            return orb.image.StandardImage(
                self.get_dataset('standard_image', protect=False),
                params=self.get_dataset_attrs('standard_image'),
                instrument=self.params.instrument)
        
        
    def compute_flambda(self, deg=1, std_im=None, std_sp=None):
        """Return flamba calibration function

        :param deg: Degree of the polynomial used to fit the flux
          correction vector (this is only used if std_sp is set to a
          standard spectrum.)

        :param std_im: Standard image used to correct the absolute calibration.

        :param std_sp: Standard spectrum used to compute the
          wavelength dependant flux calibration.
        """
        # compute flambda from configuration curves
        photom = orb.photometry.Photometry(self.params.filter_name, self.params.camera,
                                           instrument=self.params.instrument)
        flam_config = photom.compute_flambda(self.get_base_axis())
        logging.info('mean flambda config: {}'.format(np.nanmean(flam_config.data)))

        # compute the correction from a standard star spectrum
        if std_sp is None:
            try:
                std_sp = self.get_standard_spectrum()
            except Exception:
                logging.info('standard_spectrum not set: no relative vector correction computed')
                std_sp = None

        if std_sp is not None:
            eps_vector = std_sp.compute_flux_correction_vector(deg=deg)

            logging.info('relative correction vector: max {:.2f}, min {:.2f}'.format(
                np.nanmax(eps_vector.data), np.nanmin(eps_vector.data)))
        else:
            logging.info('standard_path not set: no relative vector correction computed')
            eps_vector = orb.core.Cm1Vector1d(np.ones(self.dimz, dtype=float), params=self.params,
                                              axis=self.get_base_axis())
        
        # compute the correction from a standard star calibration image
        if std_im is None:
            try:
                std_im = self.get_standard_image()
            except Exception:
                logging.info('standard_image not set: no absolute vector correction computed')
                std_im = None
        
        if std_im is not None:
            try:
                eps_mean = std_im.compute_flux_correction_factor()
            except Exception as e:
                logging.warning('error during compute_flux_correction_factor: {}'.format(e))
            else:
                logging.info('absolute correction factor: {:.2f}'.format(eps_mean))
                eps_vector = eps_vector.multiply(eps_mean)
                
        return photom.compute_flambda(self.get_base_axis(), eps=eps_vector)

    def set_flambda(self, flambda):
        """Set flux calibration.

        :param flambda: must be core.Cm1Vector1d instance.
        """
        self.set_param('flambda', flambda.project(self.get_base_axis()).data)
        if self.get_level() < 3:
            logging.warning('internal data should be already calibrated in erg/cm2/s/A. Setting a new flambda will not change the data.')
        
    def compute_modulation_ratio(self):
        deep_spectral = self.compute_sum_image()
        deep_interf = self.get_deep_frame().data
        mod = np.abs(deep_spectral).real / deep_interf
        mod[mod == 0] = np.nan
        return mod
        
        
            
            
        
        
        
        
