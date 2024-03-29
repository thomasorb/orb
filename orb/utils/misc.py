#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: misc.py

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

import os
import re
import logging
import numpy as np
import math
import warnings
import sys

import orb.cutils                       
import pyregion
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import orb.utils.io

import fnmatch

def find(pattern, path):
    """Find a file from a pattern at a given path
    https://stackoverflow.com/questions/1724693/find-a-file-in-python
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

find('*.txt', '/path/to/dir')

def aggregate_pixels(pixel_list, radius=1.42):
    """Aggregate neighbouring pixels into a set of sources. Two
    neighbours are found if there distance is smaller than a given
    radius (in pixels).

    :param pixel_list: A list of pixel position as returned by a
      function like numpy.nonzero.

    :param radius: (Optional) Max separation between two pixels of the
      same source (default 1.42).
    
    :returns: A list of pixel list. Each item of the list corresponds
      to a source and each source is itself a list of pixel positions
      (x,y).
    """

    def get_neighbours(ix, iy, pixel_list, agg_list_ok, radius):
        
        _x = pixel_list[0]
        _y = pixel_list[1]
        _r = np.sqrt((_x - ix)**2. + (_y - iy)**2.)
        _r[agg_list_ok == 1] = 2. * radius
        nei_x = _x[_r < radius]
        nei_y = _y[_r < radius]
        agg_list_ok[_r < radius] = True
        neighbours = list()
        for inei in range(len(nei_x)):
            neighbours.append((nei_x[inei], nei_y[inei]))
    
        return neighbours, agg_list_ok
        
    sources = list()
    agg_list_ok = np.zeros(len(pixel_list[0]), dtype=bool)
    x = pixel_list[0]
    y = pixel_list[1]

    for isource in range(len(x)):
        ii = x[isource]
        ij = y[isource]
        sys.stdout.write(' '*10)
        sys.stdout.write('\r {}/{}'.format(isource, len(x)))
        if not agg_list_ok[isource]:
            agg_list_ok[isource] = True
            new_source = list()
            new_source.append((ii,ij))
            more = True
            while more:
                for pix in new_source:
                    more = False
                    neighbours, agg_list_ok = get_neighbours(
                        pix[0], pix[1], pixel_list, agg_list_ok, radius)
                    if len(neighbours) > 0:
                        more = True
                        for inei in neighbours:
                            new_source.append((inei))
            if len(new_source) > 1:
                sources.append(new_source)
        sys.stdout.flush()
    print(' '*20)
    logging.info('{} sources detected'.format(len(sources)))
    return sources


def get_axis_from_hdr(hdr, axis_index=1):
    """Return axis from a classic FITS header

    :param hdr: FITS header

    :param axis_index: (Optional) Index of the axis to retrieve
      (default 1)
    """
    naxis = int(hdr['NAXIS{}'.format(axis_index)])
    crpix = float(hdr['CRPIX{}'.format(axis_index)])
    crval = float(hdr['CRVAL{}'.format(axis_index)])
    cdelt = float(hdr['CDELT{}'.format(axis_index)])
    return (np.arange(naxis, dtype=float) + 1. - crpix) * cdelt + crval
                        
def get_mask_from_ds9_region_file(reg_path, x_range, y_range,
                                  integrate=True, header=None):
    """Return a mask from a ds9 region file or from a ds9-like region
    string.

    :param reg_path: Path to a ds9 region file or ds9-like region string

    :param x_range: Range of x image coordinates
        considered as valid. Pixels outside this range are
        rejected..

    :param y_range: Range of y image coordinates
        considered as valid. Pixels outside this range are
        rejected.

    :param integrate: (Optional) If True, all pixels are integrated
      into one mask, else a list of region masks is returned (default
      True)

    :param header: (Optional) Header containing the WCS transformation
      if the region file is in celestial coordinates (default None).
    
    .. note:: The returned array can be used like a list of
        indices returned by e.g. numpy.nonzero().

    .. note:: Coordinates can be celestial or image coordinates
      (x,y). if coordinates are celestial a header must be passed to
      the function.

    """
    ### Warning: pyregion works in 'transposed' coordinates
    ### We will work here in python (y,x) convention

    if os.path.exists(reg_path):
        _regions = pyregion.open(reg_path)
    else:
        _regions = pyregion.parse(reg_path)
        
    if not _regions.check_imagecoord():
        if header is None: raise Exception('DS9 region file is not in image coordinates. Please change it to image coordinates or pass a astropy.io.fits.Header instance to the function to transform the actual coordinates to image coordinates.')
        else:
            wcs = pywcs.WCS(header, naxis=2, relax=True)
            #_regions = _regions.as_imagecoord(wcs.to_header())
            # WCS does not export NAXIS1, NAXIS2 anymore...
            h = wcs.to_header(relax=True)
            h.set('NAXIS1', header['NAXIS1'])
            h.set('NAXIS2', header['NAXIS2'])
            _regions = _regions.as_imagecoord(h)
    shape = (np.max(y_range), np.max(x_range))
    mask = np.zeros(shape, dtype=float)
    hdu = pyfits.PrimaryHDU(mask)
    mask_list = list()
    for _region in _regions:
        if len(_regions) > 1:
            sys.stdout.write('\r loading region: {}'.format(_region))
        imask2d = pyregion.get_mask([_region], hdu)
        imask2d[:np.min(x_range), :] = 0
        imask2d[:, :np.min(y_range)] = 0
        imask = np.nonzero(imask2d)
        mask[imask] = 1
        
        if integrate:
            mask[imask] = True
        else:
            mask_list.append([imask[1], imask[0]]) # transposed to
                                                   # return
        if len(_regions) > 1:
            sys.stdout.flush()
            
    if len(_regions) > 1:
        print('\n')
    if integrate:
        return np.nonzero(mask.T) # transposed to return
    else:
        return mask_list


def compute_obs_params(nm_min_filter, nm_max_filter,
                       theta_min=5.01, theta_max=11.28):
    """Compute observation parameters (order, step size) given the
    filter bandpass.

    :param nm_min_filter: Min wavelength of the filter in nm.

    :param nm_max_filter: Max wavelength of the filter in nm.
    
    :param theta_min: (Optional) Min angle of the detector (default
      5.01).

    :param theta_max: (Optional) Max angle of the detector (default
      11.28).

    :return: A tuple (order, step size, max wavelength)
    """
    def get_step(nm_min, n, cos_min):
        return int(nm_min * ((n+1.)/(2.*cos_min)))

    def get_nm_max(step, n, cos_max):
        return 2. * step * cos_max / float(n)
    
    cos_min = math.cos(math.radians(theta_min))
    cos_max = math.cos(math.radians(theta_max))

    n = 0
    order_found = False
    while n < 200 and not order_found:
        n += 1
        step = get_step(nm_min_filter, n, cos_min)
        nm_max = get_nm_max(step, n, cos_max)
        if nm_max <= nm_max_filter:
            order_found = True
            order = n - 1
       
    step = get_step(nm_min_filter, order, cos_min)
    nm_max = get_nm_max(step, order, cos_max)
    
    return order, step, nm_max


def correct_bad_frames_vector(bad_frames_vector, dimz):
    """Remove bad indexes of the bad frame vector.

    :param bad_frames_vector: The vector of indexes to correct
    :param dimz: Dimension of the cube along the 3rd axis.
    """
    if (bad_frames_vector is None
        or np.size(bad_frames_vector) == 0):
        return bad_frames_vector
    
    bad_frames_vector= np.array(np.copy(bad_frames_vector))
    bad_frames_vector = [bad_frames_vector[badindex]
                         for badindex in range(bad_frames_vector.shape[0])
                         if (bad_frames_vector[badindex] >= 0
                             and bad_frames_vector[badindex] < dimz)]
    return bad_frames_vector

def restore_error_settings(old_settings):
    """Restore old floating point error settings of numpy.
    """
    np.seterr(divide = old_settings["divide"])
    np.seterr(over = old_settings["over"])
    np.seterr(under = old_settings["under"])
    np.seterr(invalid = old_settings["invalid"])


def get_cfht_odometer(path):
    """Return the odometer of a cfht FITS file from its path."""
    return int(os.path.splitext(os.path.split(path.strip())[-1])[0][:-1])


def sort_image_list(file_list, image_mode, cube=True):
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
    except Exception as e:
        raise Exception('Malformed sequence of files: {}:\n{}'.format(
            e, file_seq))


    # get changing column
    test = np.sum(file_keys == file_keys[0,:], axis=0)

    if np.min(test) > 1:
        logging.warning('Images list cannot be safely sorted. Two images at least have the same index')
        column_index = np.nan
    else:
        column_index = np.argmin(test)

    # get changing step (if possible)
    steplist = list()
    if cube:
        for path in file_list:
            sys.stdout.write('\rreading {}'.format(path))
            if '.fits' in path:
                try:
                    hdr = orb.utils.io.read_fits(
                        path, return_hdu_only=True).header
                    if 'SITSTEP' in hdr:
                        steplist.append(int(hdr['SITSTEP']))
                except Exception: pass
        sys.stdout.write('\n')
            

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
        raise Exception('Image list cannot be sorted.')

    return file_list


def read_instrument_value_from_file(path):
    """Read the instrument value form an hdf5/fits file

    :param path: path to an hdf5/fits file.
    """
    instrument = None
    if 'hdf' in path:
        with orb.utils.io.open_hdf5(path, 'r') as f:
            if 'instrument' in f.attrs:
                instrument = f.attrs['instrument']
            elif 'INSTRUME' in f.attrs:
                instrument = f.attrs['INSTRUME'].lower()

    elif 'fit' in path:
        hdu = orb.utils.io.read_fits(path, return_hdu_only=True)
        _hdr = hdu.header
        if 'INSTRUME' in _hdr:
            instrument = _hdr['INSTRUME'].lower()
    return instrument


def convert_camera_parameter(param):
    """Convert camera parameter to an integer value
    """
    if param == 'MERGED_DATA': return 0
    elif '1' in str(param): return 1
    elif '2' in str(param): return 2
    elif '0' in str(param): return 0
    else:
        raise ValueError('camera parameter {} not understood'.format(param))

    

