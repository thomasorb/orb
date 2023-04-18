#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: stats.py

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
import numpy as np
import warnings
import orb.cutils
import sys

def robust_modulo(_dat, mod):
    """Return an array modulo mod. 

    Returned values have the smallest possible absolute
    value. e.g. -1.1 % 1 = -0.1, -0.6 % 1 = 0.4. This is perfect to
    compute the residual of a phase fit (which is known modulo pi)

    This is robust to NaN and fast.
    """
    warnings.simplefilter('ignore', RuntimeWarning)
    _dat = np.copy(_dat)
    nonan = ~np.isnan(_dat)
    sup = (_dat > mod/2.) * nonan
    while np.any(sup):
        sys.stdout.write('\r{}'.format(np.sum(sup)))
        sys.stdout.flush()
        _dat[sup] -= mod
        sup = (_dat > mod/2.) * nonan
    inf = (_dat < -(mod/2.)) * nonan
    while np.any(inf):
        sys.stdout.write('\r{}'.format(np.sum(inf)))
        sys.stdout.flush()
        _dat[inf] += mod
        inf = (_dat < -(mod/2.)) * nonan
    sys.stdout.write('completed\n')
    return _dat

def unbiased_std(a):
    """Return the std based on the interquartile range.

    Robust to outliers, but the underlying distribution must be Normal-ish

    https://en.wikipedia.org/wiki/Robust_measures_of_scale
    """
    return np.diff(np.nanpercentile(a, [25,75]))[0]/1.349

def unbiased_mean(a, perc=16):
    """Return an unbiased measure of the mean.

    Robust to outliers, but the underlying distribution must be Normal-ish between the choosen percentile limit.
    """
    a_sorted = np.array(a).flatten()
    a_sorted = np.sort(a_sorted[~np.isnan(a_sorted)])
    return np.nanmean(a_sorted[int(len(a_sorted)*perc/100):int(np.ceil(len(a_sorted)* (1-perc/100)))])

def robust_mean(a, weights=None, warn=True):
    """Compute the mean of a distribution even with NaN values

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param weights: Weights of each value of a (Must have the same
      length as a). If None, weights are all considered equal to 1
      (default None).
      
    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    
    if weights is None:
        result = orb.cutils.robust_mean(a)
    else:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
            
        result = orb.cutils.robust_average(a, weights)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        logging.warning('Only NaN values found in the given array')
        
    return result

def robust_std(a, warn=True):
    """Compute the std of a distribution even with NaN values
    
    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = orb.cutils.robust_std(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        logging.warning('Only NaN values found in the given array')
        
    return result

def robust_sum(a, warn=True):
    """Compute the sum of a distribution (skip NaN values)

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = orb.cutils.robust_sum(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        logging.warning('Only NaN values found in the given array')
        
    return result

def robust_median(a, warn=True):
    """Compute the median of a distribution (skip NaN values).

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck

    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = orb.cutils.robust_median(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        logging.warning('Only NaN values found in the given array')
        
    return result

def sigmacut(x, sigma=3., min_values=3, central_value=None, warn=False,
             return_index_list=False):
    """Return a distribution after a sigma cut rejection
    of the too deviant values.

    :param x: The distribution to cut
    
    :param sigma: (Optional) Number of sigma above which values are
      considered as deviant (default 3.)

    :param min_values: (Optional) Minimum number of values to return
      (default 3)

    :param central_value: (Optional) If not none, this value is used as
      the central value of the cut. Else the median of the
      distribution is used as the central value (default None)

    :param warn: (Optional) If False no warning message is printed
      (default False).

    :param return_index_list: (Optional) If True the list of the non
      rejected values is returned also (default False).
    """
    if central_value is None:
        central_value = 0.
        use_central_value = False
    else:
        use_central_value = True

    if np.size(x) <= min_values:
        if warn:
            logging.warning("No sigma-cut done because the number of values (%d) is too low"%np.size(x))
        return x
        
    return orb.cutils.sigmacut(
        np.array(x).astype(float).flatten(), central_value,
        use_central_value, sigma, min_values,
        return_index_list=return_index_list)
