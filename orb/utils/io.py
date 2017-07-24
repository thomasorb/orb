#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: io.py

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

import os
import numpy as np
import time
import warnings
import astropy.io.fits as pyfits
import bottleneck as bn
import orb.cutils

import orb.version
__version__ = orb.version.__version__

def write_fits(fits_path, fits_data, fits_header=None,
               silent=False, overwrite=False, mask=None,
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
        raise ValueError('Data type must be numpy.ndarray')

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
        warnings.warn('Complex data cast to float32 (FITS format do not support complex data)')

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
                    data_mean = bn.nanmean(fdata)
                    data_median = bn.nanmedian(fdata)
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
            hdu.header.set('PROGRAM', "ORB v%s"%__version__, 
                           'Thomas Martin: thomas.martin.1@ulaval.ca',
                           after=5)

            # write FITS file
            hdu.writeto(fits_path, overwrite=overwrite)

            if mask is not None:
                hdu_mask.header = hdu.header
                hdu_mask.header.set('MASK', 'True', '', after=6)
                if mask_path is None:
                    mask_path = os.path.splitext(fits_path)[0] + '_mask.fits'
                    
                hdu_mask.writeto(mask_path, clobber=overwrite)

            if not (silent):
                print "Data written as {} in {:.2f} s ".format(
                    fits_path, time.time() - start_time)

            return fits_path
        else :
            fits_path = (os.path.splitext(base_fits_path)[0] + 
                         "_" + str(index) + 
                         os.path.splitext(base_fits_path)[1])
            index += 1


def read_fits(fits_path, no_error=False, nan_filter=False, 
              return_header=False, return_hdu_only=False,
              return_mask=False, silent=False, delete_after=False,
              data_index=0, image_mode='classic', chip_index=None,
              binning=None, fix_header=True, memmap=False, dtype=float,
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

    :param mask_path: (Optional) Path to the corresponding mask image.
    
    .. note:: Please refer to
      http://www.stsci.edu/institute/software_hardware/pyfits/ for
      more information on PyFITS module. And
      http://fits.gsfc.nasa.gov/ for more information on FITS
      files.
    """
    fits_path = ((fits_path.splitlines())[0]).strip()
    if return_mask:
        if mask_path is None:
            mask_path = os.path.splitext(fits_path)[0] + '_mask.fits'
        fits_path = mask_path

    try:
        hdulist = pyfits.open(fits_path, memmap=memmap)
        fits_header = hdulist[data_index].header
    except Exception, e:
        if not no_error:
            raise Exception(
                "File '%s' could not be opened: {}, {}".format(fits_path, e))

        else:
            if not silent:
                warnings.warn(
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
        return hdulist
    else:
        if image_mode == 'classic':
            # avoid bugs fits with no data in the first hdu
            data_index = get_hdu_data_index(hdulist)

            fits_data = np.array(
                hdulist[data_index].data.transpose()).astype(dtype)
        elif image_mode == 'sitelle':
            fits_data = read_sitelle_chip(hdulist, chip_index)
        elif image_mode == 'spiomm':
            fits_data, fits_header = read_spiomm_data(
                hdulist, fits_path)
        else:
            raise ValueError("Image_mode must be set to 'sitelle', 'spiomm' or 'classic'")

    hdulist.close

    if binning is not None:
        fits_data = bin_image(fits_data, binning)

    if (nan_filter):
        fits_data = np.nan_to_num(fits_data)


    if delete_after:
        try:
            os.remove(fits_path)
        except:
             warnings.warn("The file '%s' could not be deleted"%fits_path)

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
        if key not in hdu[0].header: raise Exception(
            'Bad SITELLE image header')
        chip_section = hdu[0].header[key]
        return get_sitelle_slice(chip_section)

    def get_data(key, index, frame):
        xslice, yslice = get_slice(key, index)
        return np.copy(frame[yslice, xslice]).transpose()

    if int(chip_index) not in (1,2): raise Exception(
        'Chip index must be 1 or 2')

    frame = hdu[0].data.astype(np.float)

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

        if iamp in ['A', 'C', 'E', 'G']:
            bias_data = bias_data[int(bias_data.shape[0]/2):,:]
        else:
            bias_data = bias_data[:int(bias_data.shape[0]/2),:]

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

        bias_level = bn.nanmedian(bias_frame[x_min:x_max, y_min:y_max])

        if bias_level is not np.nan:
            hdr['BIAS-LVL'] = (
                bias_level,
                'Bias level (moment, at the center of the frame)')

    return frame, hdr



def bin_image(a, binning):
    """Return mean binned image. 

    :param image: 2d array to bin.

    :param binning: binning (must be an integer >= 1).

    .. note:: Only the complete sets of rows or columns are binned
      so that depending on the bin size and the image size the
      last columns or rows can be ignored. This ensures that the
      binning surface is the same for every pixel in the binned
      array.
    """
    binning = int(binning)

    if binning < 1: raise Exception('binning must be an integer >= 1')
    if binning == 1: return a

    if a.dtype is not np.float:
        a = a.astype(np.float)

    # x_bin
    xslices = np.arange(0, a.shape[0]+1, binning).astype(np.int)
    a = np.add.reduceat(a[0:xslices[-1],:], xslices[:-1], axis=0)

    # y_bin
    yslices = np.arange(0, a.shape[1]+1, binning).astype(np.int)
    a = np.add.reduceat(a[:,0:yslices[-1]], yslices[:-1], axis=1)

    return a / (binning**2.)
