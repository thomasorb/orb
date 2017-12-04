#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: filters.py

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

import logging
import numpy as np
import warnings
from scipy import interpolate

import orb.utils.spectrum
import orb.cutils

def compute_weights(calib, nm_laser, step_nb, step, order,
                    range_border_coeff, filter_min_cm1, filter_max_cm1):
    """Compute weights for a fit based on a spectrum with a given
    filter bandpass

    :param calib: Calibration laser observed wavelength
    
    :param nm_laser: Calibration laser theoretical wavelength
    
    :param step_nb: Vector length
    
    :param step: Step size (in nm)
    
    :param order: Folding order
    
    :param range_border_coeff: Percentage of the vector size
      considered as bad borders near the filter edges (must be between
      0.2 and 0.).
    
    :param filter_min_cm1: Minimum wavenumber of the filter in cm-1

    :param filter_max_cm1: Maximum wavenumber of the filter in cm-1
    """

    filter_range = get_filter_edges_pix(
        None, calib / nm_laser, step, order, step_nb,
        orb.utils.spectrum.cm12nm(filter_max_cm1),
        orb.utils.spectrum.cm12nm(filter_min_cm1))
    range_size = abs(np.diff(filter_range)[0])
    
    weights = np.ones(step_nb, dtype=float)
    weights[:int(np.min(filter_range))
            + int(range_border_coeff * range_size)] = 1e-35
    weights[int(np.max(filter_range))
            - int(range_border_coeff * range_size):] = 1e-35
    return weights, filter_range

def get_key(filter_file_path, key, cast):
    """
    Return key value if it exists, None instead.
    
    :param filter_file_path: Path to the filter file.
    """
    with open(filter_file_path, 'r') as f:
        for line in f:
            if len(line) > 2:
                if '# {}'.format(key) in line:
                    return cast(line.strip().split()[2])
    warnings.warn('{} keyword not in filter file: {}'.format(key, filter_file_path))
    return None


def get_phase_fit_order(filter_file_path):
    """
    Return phase fit order if it exists and None instead.
    
    :param filter_file_path: Path to the filter file.
    """
    return get_key(filter_file_path, 'PHASE_FIT_ORDER', int)

def get_observation_params(filter_file_path):
    """
    Return observation params as tuple (step, order).
    
    :param filter_file_path: Path to the filter file.
    """
    _order = get_key(filter_file_path, 'ORDER', int)
    _step = get_key(filter_file_path, 'STEP', float)
    if _order is None or _step is None:
        raise Exception('Observations params (ORDER, STEP) not in filter file')
    else: return (_step, _order)


def get_modulation_efficiency(filter_file_path):
    """
    Return modulation efficiency if it exists and 1. instead
    
    :param filter_file_path: Path to the filter file.
    """
    _me = get_key(filter_file_path, 'MODULATION_EFFICIENCY', float)
    if _me is None: return 1.
    else: return _me


def read_filter_file(filter_file_path, return_spline=False):
    """
    Read a file containing the filter transmission function.

    :param filter_file_path: Path to the filter file.

    :param return_spline: If True a cubic spline
      (scipy.interpolate.UnivariateSpline instance) is returned
      instead of a tuple.

    :returns: (list of filter wavelength, list of corresponding
      transmission coefficients, minimum edge of the filter, maximum
      edge of the filter) (Both min and max edges can be None if they
      were not recorded in the file) or a spline if return_spline set
      to True.
      
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

    filter_file.close()

    if not return_spline:
        return filter_nm, filter_trans, filter_min, filter_max
    else:
        return interpolate.UnivariateSpline(
            filter_nm, filter_trans, k=3, s=0, ext=0)


def get_filter_bandpass(filter_file_path):
    """Return the filter bandpass in nm as a tuple (nm_min, nm_max)
    
    :param get_filter_bandpass: Path to the filter file.

    .. seealso:: :py:meth:`utils.read_filter_file`
    """
    if filter_file_path is not None:
        (filter_nm, filter_trans,
         filter_min, filter_max) = read_filter_file(filter_file_path)
    else:
        warnings.warn('filter_file_path is None')
        return None

    return filter_min, filter_max

def get_filter_edges_pix(filter_file_path, correction_factor, step, order,
                         n, filter_min=None, filter_max=None):
    """Return the position in pixels of the edges of a filter
    corrected for the off-axis effect.

    Note that the axis is assumed to be in wavenumber.
    
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
        (filter_min, filter_max) = get_filter_bandpass(filter_file_path)
    elif (filter_min is None or filter_max is None):
        raise Exception("filter_min and filter_max must be specified if filter_file_path is None")

    filter_min_cm1 = orb.utils.spectrum.nm2cm1(filter_max)
    filter_max_cm1 = orb.utils.spectrum.nm2cm1(filter_min)

    cm1_axis_step = orb.cutils.get_cm1_axis_step(
        n, step, corr=correction_factor)
    cm1_axis_min = orb.cutils.get_cm1_axis_min(
        n, step, order, corr=correction_factor)
    filter_range = orb.cutils.fast_w2pix(
        np.array([filter_min_cm1, filter_max_cm1]),
        cm1_axis_min, cm1_axis_step)

    #if int(order) & 1:
    #    filter_range = n - filter_range
    #    filter_range = filter_range[::-1]
    
    filter_range[filter_range < 0] = 0
    filter_range[filter_range > n] = (n - 1)

    if min(filter_range) == max(filter_range):
        raise Exception('Filter range out of axis. Check step, order and correction factor')
    return filter_range


def get_filter_function(filter_file_path, step, order, n,
                        wavenumber=False, silent=False, corr=1.):
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
    
    :param corr: (Optional) Correction coefficient related to the
      incident angle (default 1).

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
        spectrum_axis = orb.utils.spectrum.create_nm_axis(
            n, step, order, corr=corr)
    else:
        spectrum_axis = orb.utils.spectrum.create_nm_axis_ireg(
            n, step, order, corr=corr)

    f_axis = interpolate.UnivariateSpline(np.arange(n),
                                          spectrum_axis)
    fpix_axis = interpolate.UnivariateSpline(spectrum_axis[::-1],
                                             np.arange(n)[::-1])

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
            logging.info("Filter edges read from filter file: %f -- %f"%(filter_min, filter_max))

        # filter edges converted to index of the filter vector
        filter_min = int(fpix_axis(filter_min))
        filter_max = int(fpix_axis(filter_max))
        
    if not wavenumber:
        return filter_function, filter_min, filter_max
    else:
        return filter_function, filter_max, filter_min



