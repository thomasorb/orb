##################################################
#### CLASS Cube ##################################
##################################################
class Cube(Tools):
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
        Tools.__init__(self, **kwargs)

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
            data = self.read_fits(data)

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

    def get_size_on_disk(self):
        """Return the expected size of the cube if saved on disk in Mo.
        """
        return self.dimx * self.dimy * self.dimz * 4 / 1e6 # 4 octets in float32

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
        progress = ProgressBar(self.dimz-1)
        for ik in range(1, self.dimz):
            progress.update(ik, info='Binning cube')
            cube_bin[:,:,ik] = utils.image.nanbin_image(
                self.get_data_frame(ik), binning)
        progress.end()
        return cube_bin


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
        logging.info("Data resized to shape : (" + str(dimx) +  ", " + str(dimy) + ")")
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
          containing star-like objects (a linear interpolation must
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
        logging.info("Data resized to shape : (" + str(dimx) +  ", " + str(dimy) + ", " + str(dimz) + ")")
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
            if _ik_end >= self.dimz:
                _ik_end = self.dimz - 1
            if _ik_end > _ik:
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
        else: raise StandardError('stat_key must be set to MEDIAN, MEAN or STD')
        

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
            
        else: raise StandardError(
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
                modules=("import logging",
                         "import numpy as np",
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
            div_nb = self.config.DIV_NB

        if dimx is None: dimx = self.dimx
        if dimy is None: dimy = self.dimy

        return Tools._get_quadrant_dims(
            self, quad_number, dimx, dimy, div_nb)

    def get_calibration_laser_map(self):
        """Not implemented in Cube class but implemented in child
        classes."""
        return None
       
    def export(self, export_path, x_range=None, y_range=None,
               z_range=None, header=None, overwrite=False,
               force_hdf5=False, force_fits=False,
               calibration_laser_map_path=None, mask=None,
               deep_frame_path=None):
        
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

        :param calibration_laser_map_path: (Optional) Path to a
          calibration laser map to append (default None).

        :param deep_frame_path: (Optional) Path to a deep frame to
          append (default None)

        :param mask: (Optional) If a mask is given. Exported data is
          masked. A NaN in the mask flags for a masked pixel. Other
          values are not considered (default None).
        """
        if force_fits and force_hdf5:
            raise StandardError('force_fits and force_hdf5 cannot be both set to True')
        
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
            
        if ((self.is_hdf5_frames or force_hdf5 or self.is_hdf5_cube)
            and not force_fits): # HDF5 export
            logging.info('Exporting cube to an HDF5 cube: {}'.format(
                export_path))
            if not self.is_quad_cube:
                outcube = OutHDFCube(
                    export_path,
                    (xmax - xmin, ymax - ymin, zmax - zmin),
                    overwrite=overwrite,
                    reset=True)
            else:
                outcube = OutHDFQuadCube(
                    export_path,
                    (xmax - xmin, ymax - ymin, zmax - zmin),
                    self.config.QUAD_NB,
                    overwrite=overwrite,
                    reset=True)

            outcube.append_image_list(self.image_list)
            if header is None:
                header = self.get_cube_header()
            outcube.append_header(header)

            if calibration_laser_map_path is None:
                try:
                    calibration_laser_map = self.get_calibration_laser_map()
                    calib_map_hdr = None
                except StandardError:
                    calibration_laser_map = None
            else:
                calibration_laser_map, calib_map_hdr = self.read_fits(
                    calibration_laser_map_path, return_header=True)
                if (calibration_laser_map.shape[0] != self.dimx):
                    calibration_laser_map = utils.image.interpolate_map(
                        calibration_laser_map, self.dimx, self.dimy)

            if calibration_laser_map is not None:
                logging.info('Append calibration laser map')
                outcube.append_calibration_laser_map(calibration_laser_map,
                                                     header=calib_map_hdr)

            if deep_frame_path is not None:
                deep_frame = self.read_fits(deep_frame_path)
                logging.info('Append deep frame')
                outcube.append_deep_frame(deep_frame)
                
            if not self.is_quad_cube: # frames export
                progress = ProgressBar(zmax-zmin)
                for iframe in range(zmin, zmax):
                    progress.update(iframe-zmin, info='exporting frame {}'.format(iframe))
                    idata = self.get_data(
                        xmin, xmax, ymin, ymax,
                        iframe, iframe + 1,
                        silent=True)
                    
                    if mask is not None:
                        idata[np.nonzero(np.isnan(mask))] = np.nan
                    
                    outcube.write_frame(
                        iframe,
                        data=idata,
                        header=self.get_frame_header(iframe),
                        force_float32=True)
                progress.end()
                
            else: # quad export
                
                progress = ProgressBar(self.config.QUAD_NB)
                for iquad in range(self.config.QUAD_NB):
                    progress.update(
                        iquad, info='exporting quad {}'.format(
                            iquad))
                    
                    x_min, x_max, y_min, y_max = self._get_quadrant_dims(
                        iquad, self.dimx, self.dimy,
                        int(math.sqrt(float(self.quad_nb))))
                    
                    data_quad = self.get_data(x_min, x_max,
                                              y_min, y_max,
                                              0, self.dimz,
                                              silent=True)

                    if mask is not None:
                        data_quad[np.nonzero(np.isnan(mask)),:] = np.nan
                    
                    # write data
                    outcube.write_quad(iquad,
                                       data=data_quad,
                                       force_float32=True)
                    
                progress.end()
                
            outcube.close()
            del outcube
                
        else: # FITS export
            logging.info('Exporting cube to a FITS cube: {}'.format(
                export_path))

            if not self.is_hdf5_cube:
                
                data = np.empty((xmax-xmin, ymax-ymin, zmax-zmin),
                                dtype=float)
                job_server, ncpus = self._init_pp_server()
                progress = ProgressBar(zmax-zmin)
                for iframe in range(0, zmax-zmin, ncpus):
                    progress.update(
                        iframe,
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
            else:
                data = self.get_data(xmin, xmax, ymin, ymax, zmin, zmax)

            if mask is not None:
                data[np.nonzero(np.isnan(mask)),:] = np.nan
                
            self.write_fits(export_path, data, overwrite=overwrite,
                            fits_header=header)

    
            
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
#### CLASS FDCube ###############################
#################################################

class FDCube(OCube):
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
        if params is None:
            Cube.__init__(self, None, **kwargs)
        else:
            OCube.__init__(self, None, params, **kwargs)

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
                            image_data = self.read_fits(
                                image_name,
                                image_mode=self._image_mode,
                                chip_index=self._chip_index,
                                binning=self._prebinning)
                            self.dimx = image_data.shape[0]
                            self.dimy = image_data.shape[1]
                                                        
                        else:
                            with self.open_hdf5(image_name, 'r') as f:
                                if 'hdu0/data' in f:
                                    shape = f['hdu0/data'].shape
                                    
                                    if len(shape) == 2:
                                        self.dimx, self.dimy = shape
                                    else: raise StandardError('Image shape must have 2 dimensions: {}'.format(shape))
                                else: raise StandardError('Bad formatted hdf5 file. Use Tools.write_hdf5 to get a correct hdf5 file for ORB.')
                            
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
            self.write_fits(stored_file_path, image, overwrite=True,
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
            hdu = self.read_fits(self.image_list[index],
                                 return_hdu_only=True)
            hdu.verify('silentfix')
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



#################################################
#### CLASS HDFCube ##############################
#################################################


class HDFCube(OCube):
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
        if params is None:
            Cube.__init__(self, None, **kwargs)
        else:
            OCube.__init__(self, None, params, **kwargs)
            
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
                    raise StandardError('Attribute {} is missing. The HDF5 cube seems badly formatted. Try to create it again with the last version of ORB.'.format(attr))
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

    def get_calibration_laser_map(self):
        """Return stored calibration laser map"""
        with self.open_hdf5(self.cube_path, 'r') as f:
            if 'calib_map' in f:
                return f['calib_map'][:]
            else:
                if isinstance(self, OCube):
                    return OCube.get_calibration_laser_map(self)
                else:
                    warnings.warn('No calibration laser map stored')
                    return None

           
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
          but existing data will not be removed (default True).

        :param reset: (Optional) If True and if the file already
          exists, it is deleted (default False).
        
        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
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
            warnings.warn('Cube path changed to {} to avoid overwritting an already existing file'.format(self.export_path))

        if len(shape) == 3: self.shape = shape
        else: raise StandardError('An HDF5 cube shape must be a tuple (dimx, dimy, dimz)')

        try:
            self.f = self.open_hdf5(self.export_path, 'a')
        except IOError, e:
            if overwrite:
                os.remove(self.export_path)
                self.f = self.open_hdf5(self.export_path, 'a')
            else:
                raise StandardError(
                    'IOError while opening HDF5 cube: {}'.format(e))
                
        logging.info('Opening OutHDFCube {} ({},{},{})'.format(
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
                        raise StandardError(
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
            raise StandardError('force_float32 and force_complex64 cannot be both set to True')

            
        if data is None and header is None and mask is None:
            warnings.warn('Nothing to write in the frame {}').format(
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
            warnings.warn('empty image list')
        

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

    
    def append_calibration_laser_map(self, calib_map, header=None):
        """Append a calibration laser map to the HDF5 cube.

        :param calib_map: Calibration laser map to append.

        :param header: (Optional) Header to append (default None)
        """
        if 'calib_map' in self.f:
            del self.f['calib_map']
            
        self.f['calib_map'] = calib_map
        if header is not None:
            self.f['calib_map_hdr'] = self._header_fits2hdf5(header)

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
        
        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
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
            raise StandardError('force_float32 and force_complex64 cannot be both set to True')

            
        if data is None and header is None:
            warnings.warn('Nothing to write in the frame {}').format(
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
        
        
