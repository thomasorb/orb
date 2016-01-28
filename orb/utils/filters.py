#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: filters.py

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

import numpy as np
import warnings
from scipy import interpolate

import orb.utils.spectrum

def read_filter_file(filter_file_path):
    """
    Read a file containing a the filter transmission function.

    :param filter_file_path: Path to the filter file.

    :returns: (list of filter wavelength, list of corresponding
      transmission coefficients, minimum edge of the filter, maximum
      edge of the filter) (Both min and max edges can be None if they
      were not recorded in the file)
      
    .. note:: The filter file used must have two colums separated by a
      space character. The first column contains the wavelength axis
      in nm. The second column contains the transmission
      coefficients. Comments are preceded with a #.  Filter edges can
      be specified using the keywords : FILTER_MIN and FILTER_MAX::

        ## ORBS filter file 
        # Author: Thomas Martin <thomas.martin.1@ulaval.ca>
        # Filter name : SpIOMM_R
        # Wavelength in nm | Transmission percentage
        # FILTER_MIN 648
        # FILTER_MAX 678
        1000 0.001201585284
        999.7999878 0.009733387269
        999.5999756 -0.0004460749624
        999.4000244 0.01378122438
        999.2000122 0.002538740868

    """
    filter_file = open(filter_file_path, 'r')
    filter_trans_list = list()
    filter_nm_list = list()
    filter_min = None
    filter_max = None
    for line in filter_file:
        if len(line) > 2:
            line = line.split()
            if '#' not in line[0]: # avoid comment lines
                filter_nm_list.append(float(line[0]))
                filter_trans_list.append(float(line[1]))
            else:
                if "FILTER_MIN" in line:
                    filter_min = float(line[line.index("FILTER_MIN") + 1])
                if "FILTER_MAX" in line:
                    filter_max = float(line[line.index("FILTER_MAX") + 1])
    filter_nm = np.array(filter_nm_list)
    filter_trans = np.array(filter_trans_list)
    # sort coefficients the correct way
    if filter_nm[0] > filter_nm[1]:
        filter_nm = filter_nm[::-1]
        filter_trans = filter_trans[::-1]
        
    return filter_nm, filter_trans, filter_min, filter_max


def get_filter_edges_pix(filter_file_path, correction_factor, step, order,
                         n, filter_min=None, filter_max=None):
    """Return the position in pixels of the edges of a filter
    corrected for the off-axis effect.

    Note that the axis is assumed to be in wavenumber. Spectra are
    generally given in wavelength but phase vectors are not. So this
    function is best used with phase vectors.
    
    :param filter_file_path: Path to the filter file. If None,
      filter_min and filter_max must be specified.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order.

    :param correction_factor: Correction factor
      (i.e. calibration_map_value / laser_wavelength)
      
    :param n: Number of points of the interpolation axis.

    :param filter_min: (Optional) Edge min of the filter in nm
      (default None).

    :param filter_max: (Optional) Edge max of the filter in nm
      (default None).

    .. seealso:: :py:meth:`utils.read_filter_file`
    """
    if filter_file_path is not None:
        (filter_nm, filter_trans,
         filter_min, filter_max) = read_filter_file(filter_file_path)
    elif (filter_min is None or filter_max is None):
        raise Exception("filter_min and filter_max must be specified if filter_file_path is None")
    
    nm_axis_ireg = orb.utils.spectrum.create_nm_axis_ireg(
        n, step, order, corr=correction_factor)

    # note that nm_axis_ireg is an reversed axis
    fpix_axis = interpolate.UnivariateSpline(nm_axis_ireg[::-1],
                                             np.arange(n))
    
    try:
        filter_min_pix = int(fpix_axis(filter_min)) # axis is reversed
        filter_max_pix = int(fpix_axis(filter_max)) # axis is reversed
    except Exception:
        filter_min_pix = np.nan
        filter_max_pix = np.nan

    if (filter_min_pix <= 0. or filter_min_pix > n
        or filter_max_pix <= 0. or filter_max_pix > n
        or filter_min_pix >= filter_max_pix):
        filter_min_pix == np.nan
        filter_max_pix == np.nan 

    return filter_min_pix, filter_max_pix

def get_filter_function(filter_file_path, step, order, n,
                        wavenumber=False, silent=False):
    """Read a filter file and return its function interpolated over
    the desired number of points. Return also the edges position over
    its axis in pixels.
    
    :param filter_file_path: Path to the filter file.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order.

    :param n: Number of points of the interpolation axis.

    :param wavenumber: (Optional) If True the function is interpolated
      and returned along a wavenumber axis. If False it is returned
      along a wavelength axis (default False).

    :param silent: (Optional) If True, no message is displayed
      (default False).

    :returns: (interpolated filter function, min edge, max edge). Min
      and max edges are given in pixels over the interpolation axis.

    .. seealso:: :py:meth:`utils.read_filter_file`
    """

    THRESHOLD_COEFF = 0.7

    # Preparing filter function
    (filter_nm, filter_trans,
     filter_min, filter_max) = read_filter_file(filter_file_path)

    # Spectrum wavelength axis creation.
    if not wavenumber:
        spectrum_axis = orb.utils.spectrum.create_nm_axis(n, step, order)
    else:
        spectrum_axis = orb.utils.spectrum.create_nm_axis_ireg(n, step, order)

    f_axis = interpolate.UnivariateSpline(np.arange(n),
                                          spectrum_axis)
    fpix_axis = interpolate.UnivariateSpline(spectrum_axis,
                                             np.arange(n))

    # Interpolation of the filter function
    interpol_f = interpolate.UnivariateSpline(filter_nm, filter_trans, 
                                              k=5, s=0)
    filter_function = interpol_f(spectrum_axis.astype(float))

    # Filter function is expressed in percentage. We want it to be
    # between 0 and 1.
    filter_function /= 100.

    # If filter edges were not specified in the filter file look
    # for parts with a transmission coefficient higher than
    # 'threshold_coeff' of the maximum transmission
    if (filter_min is None) or (filter_max is None):
        filter_threshold = ((np.max(filter_function) 
                             - np.min(filter_function)) 
                            * THRESHOLD_COEFF + np.min(filter_function))
        ok_values = np.nonzero(filter_function > filter_threshold)
        filter_min = np.min(ok_values)
        filter_max = np.max(ok_values)
        warnings.warn("Filter edges (%f -- %f nm) determined automatically using a threshold of %f %% transmission coefficient"%(f_axis(filter_min), f_axis(filter_max), filter_threshold*100.))
    else:
        if not silent:
            print "Filter edges read from filter file: %f -- %f"%(filter_min, filter_max)
        # filter edges converted to index of the filter vector
        filter_min = int(fpix_axis(filter_min))
        filter_max = int(fpix_axis(filter_max))

    if not wavenumber:
        return filter_function, filter_min, filter_max
    else:
        return filter_function, filter_max, filter_min
