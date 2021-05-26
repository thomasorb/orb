#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: io.py

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

import logging
import os
import numpy as np
import time
import warnings
import astropy.io.fits as pyfits
from astropy.io.fits.verify import VerifyWarning, VerifyError, AstropyUserWarning
from astropy.wcs import FITSFixedWarning
import astropy.io.votable
import pandas as pd

import orb.cutils
import h5py
import datetime
import orb.utils.validate


def open_file(file_name, mode='r'):
    """Open a file in write mode (by default) and return a file
    object.

    Create the file if it doesn't exist (only in write mode).

    :param file_name: Path to the file, can be either
      relative or absolute.

    :param mode: (Optional) Can be 'w' for write mode, 'r' for
      read mode and 'a' for append mode.
    """
    if mode not in ['w','r','a']:
        raise Exception("mode option must be 'w', 'r' or 'a'")

    if mode in ['w','a']:
        # create folder if it does not exist
        dirname = os.path.dirname(file_name)
        if dirname != '':
            if not os.path.exists(dirname): 
                os.makedirs(dirname)

    return open(file_name, mode)


def write_fits(fits_path, fits_data, fits_header=None,
               silent=False, overwrite=True, mask=None,
               replace=False, record_stats=False, mask_path=None):

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
      if it exists (default True).

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

    :param mask_path: (Optional) Path to the corresponding mask image.

    .. note:: float64 data is converted to float32 data to avoid
      too big files with unnecessary precision

    .. note:: Please refer to
      http://www.stsci.edu/institute/software_hardware/pyfits/ for
      more information on PyFITS module and
      http://fits.gsfc.nasa.gov/ for more information on FITS
      files.
    """
    SECURED_KEYS = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1',
                    'NAXIS2', 'NAXIS3', 'EXTEND', 'INHERIT',
                    'BZERO', 'BSCALE']

    if not isinstance(fits_data, np.ndarray):
        raise TypeError('Data type must be numpy.ndarray')

    start_time = time.time()
    # change extension if nescessary
    if os.path.splitext(fits_path)[1] != '.fits':
        fits_path = os.path.splitext(fits_path)[0] + '.fits'

    if mask is not None:
        if np.shape(mask) != np.shape(fits_data):
            raise ValueError('Mask must have the same shape as data')

    if replace: overwrite=True

    if overwrite:
        warnings.filterwarnings(
            'ignore', message='Overwriting existing file.*',
            module='astropy.io.*')

    if replace and os.path.exists(fits_path):
        old_data = read_fits(fits_path)
        if old_data.shape == fits_data.shape:
            fits_data[np.isnan(fits_data)] = old_data[np.isnan(fits_data)]
        else:
            raise Exception("New data shape %s and old data shape %s are not the same. Do not set the option 'replace' to True in this case"%(str(fits_data.shape), str(old_data.shape)))

    # float64/128 data conversion to float32 to avoid too big files
    # with unnecessary precision
    if fits_data.dtype == np.float64 or fits_data.dtype == np.float128:
        fits_data = fits_data.astype(np.float32)

    # complex data cannot be written in fits
    if np.iscomplexobj(fits_data):
        fits_data = fits_data.real.astype(np.float32)
        logging.warning('Complex data cast to float32 (FITS format do not support complex data)')

    base_fits_path = fits_path

    dirname = os.path.dirname(fits_path)
    if (dirname != []) and (dirname != ''):
        if not os.path.exists(dirname): 
            os.makedirs(dirname)

    index=0
    file_written = False
    while not file_written:
        if ((not (os.path.exists(fits_path))) or overwrite):

            if len(fits_data.shape) > 1:
                hdu = pyfits.PrimaryHDU(fits_data.transpose())
            elif len(fits_data.shape) == 1:
                hdu = pyfits.PrimaryHDU(fits_data[np.newaxis, :])
            else: # 1 number only
                hdu = pyfits.PrimaryHDU(np.array([fits_data]))

            if mask is not None:
                # mask conversion to only zeros or ones
                mask = mask.astype(float)
                mask[np.nonzero(np.isnan(mask))] = 1.
                mask[np.nonzero(np.isinf(mask))] = 1.
                mask[np.nonzero(mask)] = 1.
                mask = mask.astype(np.uint8) # UINT8 is the
                                             # smallest allowed
                                             # type
                hdu_mask = pyfits.PrimaryHDU(mask.transpose())
            # add header optional keywords
            if fits_header is not None:
                ## remove keys of the passed header which corresponds
                ## to the description of the data set
                for ikey in SECURED_KEYS:
                    if ikey in fits_header: fits_header.pop(ikey)
                hdu.header.extend(fits_header, strip=False,
                                  update=True, end=True)
                
                # Remove 3rd axis related keywords if there is no
                # 3rd axis
                if len(fits_data.shape) <= 2:
                    for ikey in range(len(hdu.header)):
                        if isinstance(hdu.header[ikey], str):
                            if ('Wavelength axis' in hdu.header[ikey]):
                                del hdu.header[ikey]
                                del hdu.header[ikey]
                                break
                    if 'CTYPE3' in hdu.header:
                        del hdu.header['CTYPE3']
                    if 'CRVAL3' in hdu.header:
                        del hdu.header['CRVAL3']
                    if 'CRPIX3' in hdu.header:
                        del hdu.header['CRPIX3']
                    if 'CDELT3' in hdu.header:
                        del hdu.header['CDELT3']
                    if 'CROTA3' in hdu.header:
                        del hdu.header['CROTA3']
                    if 'CUNIT3' in hdu.header:
                        del hdu.header['CUNIT3']

            # add median and mean of the image in the header
            # data is nan filtered before
            if record_stats:
                fdata = fits_data[np.nonzero(~np.isnan(fits_data))]
                if np.size(fdata) > 0:
                    data_mean = np.nanmean(fdata)
                    data_median = np.nanmedian(fdata)
                else:
                    data_mean = np.nan
                    data_median = np.nan
                hdu.header.set('MEAN', str(data_mean),
                               'Mean of data (NaNs filtered)',
                               after=5)
                hdu.header.set('MEDIAN', str(data_median),
                               'Median of data (NaNs filtered)',
                               after=5)

            # add some basic keywords in the header
            date = time.strftime("%Y-%m-%d", time.localtime(time.time()))
            hdu.header.set('MASK', 'False', '', after=5)
            hdu.header.set('DATE', date, 'Creation date', after=5)
            hdu.header.set('PROGRAM', "ORB", 
                           'Thomas Martin: thomas.martin.1@ulaval.ca',
                           after=5)

            # write FITS file
            hdu.writeto(fits_path, overwrite=overwrite)

            if mask is not None:
                hdu_mask.header = hdu.header
                hdu_mask.header.set('MASK', 'True', '', after=6)
                if mask_path is None:
                    mask_path = os.path.splitext(fits_path)[0] + '_mask.fits'
                    
                hdu_mask.writeto(mask_path, overwrite=overwrite)

            if not (silent):
                logging.info("Data written as {} in {:.2f} s ".format(
                    fits_path, time.time() - start_time))

            return fits_path
        else :
            fits_path = (os.path.splitext(base_fits_path)[0] + 
                         "_" + str(index) + 
                         os.path.splitext(base_fits_path)[1])
            index += 1


def read_fits(fits_path, no_error=False, nan_filter=False, 
              return_header=False, return_hdu_only=False,
              return_mask=False, silent=False, delete_after=False,
              data_index=None, image_mode='classic', chip_index=None,
              binning=None, fix_header=True, dtype=float,
              mask_path=None):
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
      unit (Default None).

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

    :param dtype: (Optional) Data is converted to
      the given dtype (e.g. np.float32, default float).

    :param mask_path: (Optional) Path to the corresponding mask image.
    
    .. note:: Please refer to
      http://www.stsci.edu/institute/software_hardware/pyfits/ for
      more information on PyFITS module. And
      http://fits.gsfc.nasa.gov/ for more information on FITS
      files.
    """
    # avoid bugs fits with no data in the first hdu
    fits_path = ((fits_path.splitlines())[0]).strip()
    if return_mask:
        if mask_path is None:
            mask_path = os.path.splitext(fits_path)[0] + '_mask.fits'
        fits_path = mask_path

    try:
        warnings.filterwarnings('ignore', module='astropy')
        warnings.filterwarnings('ignore', category=ResourceWarning)
    
        hdulist = pyfits.open(fits_path)
        if data_index is None:
            data_index = get_hdu_data_index(hdulist)

        fits_header = hdulist[data_index].header
    except Exception as e:
        if not no_error:
            raise IOError(
                "File '%s' could not be opened: {}, {}".format(fits_path, e))

        else:
            if not silent:
                logging.warning(
                    "File '%s' could not be opened {}, {}".format(fits_path, e))
            return None

    # Correct header
    if fix_header:
        if fits_header['NAXIS'] == 2:
            if 'CTYPE3' in fits_header: del fits_header['CTYPE3']
            if 'CRVAL3' in fits_header: del fits_header['CRVAL3']
            if 'CUNIT3' in fits_header: del fits_header['CUNIT3']
            if 'CRPIX3' in fits_header: del fits_header['CRPIX3']
            if 'CROTA3' in fits_header: del fits_header['CROTA3']

    if return_hdu_only:
        return hdulist[data_index]
    else:
        if image_mode == 'classic':
            fits_data = np.array(
                hdulist[data_index].data.transpose()).astype(dtype)
        elif image_mode == 'sitelle':
            fits_data = read_sitelle_chip(hdulist[data_index], chip_index)
        elif image_mode == 'spiomm':
            fits_data, fits_header = read_spiomm_data(
                hdulist, fits_path)
        else:
            raise ValueError("Image_mode must be set to 'sitelle', 'spiomm' or 'classic'")

    hdulist.close

    if binning is not None:
        fits_data = utils.image.bin_image(fits_data, binning)

    if (nan_filter):
        fits_data = np.nan_to_num(fits_data)


    if delete_after:
        try:
            os.remove(fits_path)
        except:
             logging.warning("The file '%s' could not be deleted"%fits_path)

    if return_header:
        return np.squeeze(fits_data), fits_header
    else:
        return np.squeeze(fits_data)



def get_hdu_data_index(hdul):
    """Return the index of the first header data unit (HDU) containing data.

    :param hdul: A pyfits.HDU instance
    """
    hdu_data_index = 0
    while (hdul[hdu_data_index].data is None):
        hdu_data_index += 1
        if hdu_data_index >= len(hdul):
            raise Exception('No data recorded in FITS file')
    return hdu_data_index


def read_sitelle_chip(hdu, chip_index, substract_bias=True):
    """Return chip data of a SITELLE FITS image.

    :param hdu: pyfits.HDU Instance of the SITELLE image

    :param chip_index: Index of the chip to read. Must be 1 or 2.

    :param substract_bias: If True bias is automatically
      substracted by using the overscan area (default True).
    """    
    def get_slice(key, index):
        key = '{}{}'.format(key, index)
        if key not in hdu.header: raise Exception(
            'Bad SITELLE image header')
        chip_section = hdu.header[key]
        return get_sitelle_slice(chip_section)

    def get_data(key, index, frame):
        xslice, yslice = get_slice(key, index)
        return np.copy(frame[yslice, xslice]).transpose()

    if int(chip_index) not in (1,2): raise Exception(
        'Chip index must be 1 or 2')

    frame = hdu.data.astype(np.float)

    # get data without bias substraction
    if not substract_bias:
        return get_data('DSEC', chip_index, frame)

    if chip_index == 1:
        amps = ['A', 'B', 'C', 'D']
    elif chip_index == 2:
        amps = ['E', 'F', 'G', 'H']

    xchip, ychip = get_slice('DSEC', chip_index)
    data = np.empty((xchip.stop - xchip.start, ychip.stop - ychip.start),
                    dtype=float)

    # removing bias
    for iamp in amps:
        xamp, yamp = get_slice('DSEC', iamp)
        amp_data = get_data('DSEC', iamp, frame)
        bias_data = get_data('BSEC', iamp, frame)
        overscan_size = int(bias_data.shape[0]/2) 
        if iamp in ['A', 'C', 'E', 'G']:
            bias_data = bias_data[-overscan_size:,:]
        else:
            bias_data = bias_data[:overscan_size,:]
        
        bias_data = np.mean(bias_data, axis=0)
        amp_data = amp_data - bias_data

        data[xamp.start - xchip.start: xamp.stop - xchip.start,
             yamp.start - ychip.start: yamp.stop - ychip.start] = amp_data

    return data


def get_sitelle_slice(slice_str):
    """
    Strip a string containing SITELLE like slice coordinates.

    :param slice_str: Slice string.
    """
    if "'" in slice_str:
        slice_str = slice_str[1:-1]

    section = slice_str[1:-1].split(',')
    x_min = int(section[0].split(':')[0]) - 1
    x_max = int(section[0].split(':')[1])
    y_min = int(section[1].split(':')[0]) - 1
    y_max = int(section[1].split(':')[1])
    return slice(x_min,x_max,1), slice(y_min,y_max,1)



def read_spiomm_data(hdu, image_path, substract_bias=True):
    """Return data of an SpIOMM FITS image.

    :param hdu: pyfits.HDU Instance of the SpIOMM image

    :param image_path: Image path

    :param substract_bias: If True bias is automatically
      substracted by using the associated bias frame as an
      overscan frame. Mean bias level is thus computed along the y
      axis of the bias frame (default True).
    """
    CENTER_SIZE_COEFF = 0.1

    data_index = get_hdu_data_index(hdu)
    frame = np.array(hdu[data_index].data.transpose()).astype(np.float)
    hdr = hdu[data_index].header
    # check presence of a bias
    bias_path = os.path.splitext(image_path)[0] + '_bias.fits'

    if os.path.exists(bias_path):
        bias_frame = read_fits(bias_path)

        if substract_bias:
            ## create overscan line
            overscan = orb.cutils.meansigcut2d(bias_frame, axis=1)
            frame = (frame.T - overscan.T).T

        x_min = int(bias_frame.shape[0]/2.
                    - CENTER_SIZE_COEFF * bias_frame.shape[0])
        x_max = int(bias_frame.shape[0]/2.
                    + CENTER_SIZE_COEFF * bias_frame.shape[0] + 1)
        y_min = int(bias_frame.shape[1]/2.
                    - CENTER_SIZE_COEFF * bias_frame.shape[1])
        y_max = int(bias_frame.shape[1]/2.
                    + CENTER_SIZE_COEFF * bias_frame.shape[1] + 1)

        bias_level = np.nanmedian(bias_frame[x_min:x_max, y_min:y_max])

        if bias_level is not np.nan:
            hdr['BIAS-LVL'] = (
                bias_level,
                'Bias level (moment, at the center of the frame)')

    return frame, hdr


def open_hdf5(file_path, mode):
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
        f.attrs['program'] = 'Created/modified with ORB'
        f.attrs['date'] = str(datetime.datetime.now())

    return f

def write_hdf5(file_path, data, header=None,
               silent=False, overwrite=True, max_hdu_check=True,
               compress=False):

    """    
    Write data in HDF5 format.

    A header can be added to the data. This method is useful to
    handle an HDF5 data file like a FITS file. It implements most
    of the functionality of the method
    :py:meth:`core.Tools.write_fits`.

    .. note:: The output HDF5 file can contain mutiple data header
      units (HDU). Each HDU is in a specific group named 'hdu*', *
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

    :param max_hdu_check: (Optional): When True, if the input data
      is a list (interpreted as a list of data unit), check if
      it's length is not too long to make sure that the input list
      is not a single data array that has not been converted to a
      numpy.ndarray format. If the number of HDU to create is
      indeed very long this can be set to False (default True).

    :param silent: (Optional) If True turn this function won't
      display any message (default False)

    :param overwrite: (Optional) If True overwrite the output file
      if it exists (default True).

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
        raise Exception('Data list length is > {}. As a list is interpreted has a list of data unit make sure to pass a numpy.ndarray instance instead of a list. '.format(MAX_HDUS))

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
                    raise Exception('Badly formated header')

            elif not isinstance(header[0], pyfits.Header):

                raise Exception('Header must be a pyfits.Header instance or a list')

        else:
            raise Exception('Header must be a pyfits.Header instance or a list')


        if len(header) != len(data):
            raise Exception('The number of headers must be the same as the number of data units.')


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
    with open_hdf5(new_file_path, 'w') as f:

        ## add data + header
        for i in range(len(data)):

            idata = data[i]

            # Check if data has a valid format.
            if not isinstance(idata, np.ndarray):
                try:
                    idata = np.array(idata, dtype=float)
                except Exception as e:
                    raise Exception('Data to write must be convertible to a numpy array of numeric values: {}'.format(e))


            # convert data to float32
            if idata.dtype == np.float64:
                idata = idata.astype(np.float32)

            # hdu name
            hdu_group_name = 'hdu{}'.format(i)
            if compress:
                f.create_dataset(
                    hdu_group_name + '/data', data=idata,
                    compression='lzf', compression_opts=None)
                    #compression='szip', compression_opts=('nn', 32))
                    #compression='gzip', compression_opts=9)
            else:
                f.create_dataset(
                    hdu_group_name + '/data', data=idata)

            # add header
            if header is not None:
                iheader = header[i]
                if not isinstance(iheader, pyfits.Header):
                    iheader = pyfits.Header(iheader)

                f[hdu_group_name + '/header'] = header_fits2hdf5(
                    iheader)

    logging.info('Data written as {} in {:.2f} s'.format(
        new_file_path, time.time() - start_time))

    return new_file_path


castables = [int, float, bool, str, 
             np.int64, np.float64, int, np.float128, np.bool_]
    
def cast(a, t_str):
    if isinstance(t_str, bytes):
        t_str = t_str.decode()
    if 'type' in t_str: t_str = t_str.replace('type', 'class')
    if 'long' in t_str: t_str = t_str.replace('long', 'int')
    for _t in castables:
        if t_str == repr(_t):
            return _t(a)
    raise Exception('Bad type string {} should be in {}'.format(t_str, [repr(_t) for _t in castables]))

def dict2array(data):
    """Convert a dictionary to an array that can be written in an hdf5 file

    :param data: Must be a dict instance
    """
    if not isinstance(data, dict): raise TypeError('data must be a dict')
    arr = list()
    for key in data:
        if type(data[key]) in castables: 
            _tstr = str(type(data[key]))
            arr.append(np.array(
                (key, data[key], _tstr)))
        else:
            logging.debug('{} of type {} not passed to array'.format(key, type(data[key])))
    return np.array(arr)

def array2dict(data):
    """Convert an array read from an hdf5 file to a dict.
    :param data: array of params returned by dict2array
    """
    _dict = dict()
    for i in range(len(data)):
        _dict[data[i][0]] = cast(data[i][1], data[i][2])
    return _dict


def dict2header(params):
    """convert a dict to a pyfits.Header() instance

    .. warning:: this is a destructive process, illegal values are
    removed from the header.

    :param params: a dict instance
    """
    # filter illegal header values
    cards = list()

    for iparam in params:
        val = params[iparam]
        val_ok = False
        for itype in castables:
            if isinstance(val, itype):
                val_ok = True
        
        if val_ok:
            if isinstance(val, bool):
                val = int(val)
            card = pyfits.Card(
                keyword=iparam,
                value=val,
                comment=None)
            try:
                card.verify(option='exception')
                cards.append(card)
            except (VerifyError, ValueError, TypeError):
                pass

    warnings.simplefilter('ignore', category=VerifyWarning)
    warnings.simplefilter('ignore', category=AstropyUserWarning)
    warnings.simplefilter('ignore', category=FITSFixedWarning)
    header = pyfits.Header(cards)
    return header



def header_fits2hdf5(fits_header):
    """convert a pyfits.Header() instance to a header for an hdf5 file

    :param fits_header: Header of the FITS file
    """
    hdf5_header = list()

    for ikey in range(len(fits_header)):
        _tstr = str(type(fits_header[ikey]))
        ival = np.array(
            (list(fits_header.keys())[ikey], str(fits_header[ikey]),
             fits_header.comments[ikey], _tstr))

        hdf5_header.append(ival)
    return np.array(hdf5_header, dtype='S300')


def header_hdf52fits(hdf5_header):
    """convert an hdf5 header to a pyfits.Header() instance.

    :param hdf5_header: Header of the HDF5 file
    """
    fits_header = pyfits.Header()
    for i in range(hdf5_header.shape[0]):
        ival = hdf5_header[i,:]
        ival = [iival.decode() for iival in ival]
        if ival[3] != 'comment':
            fits_header[ival[0]] = cast(ival[1], ival[3]), str(ival[2])
        else:
            fits_header['comment'] = ival[1]
    return fits_header

def read_hdf5(file_path, return_header=False, dtype=float):

    """Read an HDF5 data file created with
    :py:meth:`core.Tools.write_hdf5`.

    :param file_path: Path to the file, can be either
      relative or absolute.        

    :param return_header: (Optional) If True return a tuple (data,
       header) (default False).

    :param dtype: (Optional) Data is converted to the given type
      (e.g. np.float32, default float).

    .. note:: Please refer to http://www.h5py.org/."""


    with open_hdf5(file_path, 'r') as f:
        data = list()
        header = list()
        for hdu_name in f:
            data.append(f[hdu_name + '/data'][:].astype(dtype))
            if return_header:
                if hdu_name + '/header' in f:
                    # extract header
                    header.append(
                        header_hdf52fits(f[hdu_name + '/header'][:]))
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

def cast2hdf5(val):
    if val is None:
        return 'None'
    elif isinstance(val, np.float128):
        return val.astype(np.float64)
    #elif isinstance(val, int):
    #    return str(val)
    elif isinstance(val, np.ndarray):
        if val.dtype == np.float128:
            return val.astype(np.float64)
        
    return val

def get_storing_dtype(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError('arr must be a numpy.ndarray instance')
    if arr.dtype == np.float64:
        return np.float32
    if arr.dtype == np.complex128:
        return np.complex64
    else: return arr.dtype

def cast_storing_dtype(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError('arr must be a numpy.ndarray instance')
    return arr.astype(get_storing_dtype(arr))


def save_dflist(dflist, path):
    """Save a list of dataframes

    :param dflist: list of pandas dataframes

    :param path: path to the output file
    """
    if os.path.exists(path):
        os.remove(path)

    with open_hdf5(path, 'w') as f:
        f.attrs['len'] = len(dflist)
        
    for idf in range(len(dflist)):
        if dflist[idf] is not None:
            dflist[idf].to_hdf(path, 'df{:06d}'.format(idf),
                               format='table', mode='a')
        
def load_dflist(path):
    """Save a list of dataframes

    :param path: path to the output file
    """
    with open_hdf5(path, 'r') as f:
        _len = f.attrs['len']

    dflist = list()

    for i in range(_len):
        try:
            idf = pd.read_hdf(path, key='df{:06d}'.format(i))
            dflist.append(idf)
        except KeyError:
            dflist.append(None)
    return dflist

def read_votable(votable_file):
    """read a votable and transfer it as as pandas dataframe.

    taken from https://gist.github.com/icshih/52ca49eb218a2d5b660ee4a653301b2b
    """
    votable = astropy.io.votable.parse(votable_file)
    table = votable.get_first_table().to_table(use_names_over_ids=True)
    return table.to_pandas()

def save_starlist(path, starlist):
    """Save a star list as a two columnfile X, Y readable by ds9
    """
    orb.utils.validate.is_2darray(starlist, object_name='starlist')
    if starlist.shape[1] != 2:
        raise TypeError('starlist must be of shape (n,2)')

    with open_file(path, 'w') as f:
        for i in range(starlist.shape[0]):
            f.write('{} {}\n'.format(starlist[i,0], starlist[i,1]))
        f.flush()
