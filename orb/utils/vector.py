#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: vector.py

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
import math
from scipy import interpolate, signal

import orb.utils.spectrum
import orb.utils.stats
import orb.cutils
import orb.utils.validate

def smooth(a, deg=2, kind='gaussian', keep_sides=True):
    """Smooth a given vector.

    :param a: Vector to smooth
    
    :param deg: (Optional) Smoothing degree (or kernel
      radius) Must be an integer (default 2).
    
    :param kind: Kind of smoothing function. 'median' or 'mean' are
      self-explanatory. 'gaussian' uses a gaussian function for a
      weighted average. 'gaussian_conv' and 'cos_conv' make use of
      convolution with a gaussian kernel or a cosine
      kernel. Convolution is much faster but less rigorous on the
      edges of the vector (default 'gaussian').

    :params keep_sides: If True, the vector is seen as keeping its
      side values above its real boudaries (If False, the values
      outside the vector are 0. and this creates an undesirable border
      effect when convolving).
    """
    def cos_kernel(deg):
        x = (np.arange((deg*2)+1, dtype=float) - float(deg))/deg * math.pi
        return (np.cos(x) + 1.)/2.

    deg = int(deg)
    
    if keep_sides:
        a_large = np.empty(a.shape[0]+2*deg, dtype=float)
        a_large[deg:a.shape[0]+deg] = a
        a_large[:deg+1] = a[0]
        a_large[-deg:] = a[-1]
        a = np.copy(a_large)

    if (kind=="gaussian") or (kind=="gaussian_conv"):
        kernel = np.array(orb.utils.spectrum.gaussian1d(
            np.arange((deg*2)+1),0.,1.,deg,
            deg/4.*abs(2*math.sqrt(2. * math.log(2.)))))

    if (kind== 'cos_conv'):
        kernel = cos_kernel(deg)

    if (kind=="gaussian_conv") or (kind=='cos_conv') or (kind=="gaussian"):
        kernel /= np.nansum(kernel)
        smoothed_a = signal.convolve(a, kernel, mode='same')
        if keep_sides:
            return smoothed_a[deg:-deg]
        else:
            return smoothed_a
    
    else:
        smoothed_a = np.copy(a)
        
        for ii in range(a.shape[0]):
            if (kind=="gaussian"): weights = np.copy(kernel)
            x_min = ii-deg
            if x_min < 0:
                if (kind=="gaussian"):
                    weights = np.copy(kernel[abs(x_min):])
                x_min = 0
            x_max = ii+deg+1
            if x_max > a.shape[0]:
                if (kind=="gaussian"):
                    weights = np.copy(kernel[:-(x_max-a.shape[0])])
                x_max = a.shape[0]
            box = a[x_min:x_max]

            if (kind=="median"):
                smoothed_a[ii] = orb.utils.stats.robust_median(box)
            elif (kind=="mean"):
                smoothed_a[ii] = orb.utils.stats.robust_mean(box) 
            elif (kind=="gaussian"):
                smoothed_a[ii] = orb.utils.stats.robust_mean(
                    box, weights=weights)
            else: raise Exception("Kind parameter must be 'median', 'mean', 'gaussian', 'gaussian_conv' or 'cos_conv'")
    if keep_sides:
        return smoothed_a[deg:-deg]
    else:
        return smoothed_a


def correct_vector(vector, bad_value=np.nan, deg=3,
                   polyfit=False, smoothing=True):
    """Correct a given vector for non valid values by interpolation or
    polynomial fit.

    :param vector: The vector to be corrected.
    
    :param bad_value: (Optional) Bad value to correct (default np.nan)

    :param deg: (Optional) Spline degree or polyfit degree (default 3)

    :param polyfit: (Optional) If True non valid values are guessed
      using a polynomial fit to the data instead of an spline
      interpolation (default False)
    """
  
    n = np.size(vector)
    vector = vector.astype(float)
    if not np.isnan(bad_value):
        vector[np.nonzero(vector == bad_value)] = np.nan

    if np.all(np.isnan(vector)):
        raise Exception("The given vector has only bad values")

    # vector used to fit is smoothed
    if smoothing:
        smooth_vector = smooth(vector, deg=int(n*0.1), kind='median')
    else: smooth_vector = np.copy(vector)


    # create vectors containing only valid values for interpolation
    x = np.arange(n)[np.nonzero(~np.isnan(smooth_vector*vector))]
    new_vector = smooth_vector[np.nonzero(~np.isnan(smooth_vector*vector))]
 
    if not polyfit:
        finterp = interpolate.UnivariateSpline(x, new_vector, k=deg, s=0)
        result = finterp(np.arange(n))
    else:
        coeffs = np.polynomial.polynomial.polyfit(
            x, new_vector, deg, w=None, full=True)
        fit = np.polynomial.polynomial.polyval(
            np.arange(n), coeffs[0])
        result = np.copy(vector)
        result[np.nonzero(np.isnan(vector))] = fit[np.nonzero(np.isnan(vector))]

    # borders around the corrected points are smoothed
    if smoothing:
        zeros_vector = np.zeros_like(result)
        zeros_vector[np.nonzero(np.isnan(vector))] = 1.
        zeros_vector = smooth(zeros_vector, deg=int(n*0.1),
                              kind='cos_conv')
        zeros_vector[np.nonzero(np.isnan(vector))] = 1.
        
        vector[np.nonzero(np.isnan(vector))] = 0.
        result = result*zeros_vector + vector*(1.-zeros_vector)
        
    return result

def fft_filter(a, cutoff_coeff, width_coeff=0.2, filter_type='high_pass'):
    """
    Simple lowpass or highpass FFT filter (high pass or low pass)

    Filter shape is a gaussian.
    
    :param a: Vector to filter

    :param cutoff_coeff: Coefficient defining the position of the cut
      frequency (Cut frequency = cut_coeff * vector length)

    :param width_coeff: (Optional) Coefficient defining the width of
      the smoothed part of the filter (width = width_coeff * vector
      length) (default 0.2)

    :param filter_type: (Optional) Type of filter to use. Can be
      'high_pass' or 'low_pass'.
    """
    if cutoff_coeff < 0. or cutoff_coeff > 1.:
        raise Exception('cutoff_coeff must be between 0. and 1.')
    if width_coeff < 0. or width_coeff > 1.:
        raise Exception('width_coeff must be between 0. and 1.')
        
    if filter_type == 'low_pass':
        lowpass = True
    elif filter_type == 'high_pass':
        lowpass = False
    else:
        raise Exception(
            "Bad filter type. Must be 'low_pass' or 'high_pass'")
    
    return orb.cutils.fft_filter(a, cutoff_coeff, width_coeff, lowpass)

def polyfit1d(a, deg, w=None, return_coeffs=False):
    """Fit a polynomial to a 1D vector.
    
    :param a: Vector to fit
    
    :param deg: Fit degree
    
    :param return_coeffs: (Optional) If True return fit coefficients
      as returned by numpy.polynomial.polynomial.polyfit() (default
      False).

    :param w: (Optional) If not None, weights to apply to the
      fit. Must have the same shape as the vector to fit (default
      None)
    """
    n = a.shape[0]
    nonans = ~np.isnan(a) * ~np.isinf(a)
    nonans_ind = np.nonzero(nonans)
    
    if w is not None:
        w = np.array(w, dtype=float)
        nonans *= (~np.isnan(w) * ~np.isinf(w))
        nonans_ind = np.nonzero(nonans)
        w = w[nonans_ind]
        
    x = np.arange(n)[nonans_ind]
    
    a = a[nonans_ind]
    
    coeffs = np.polynomial.polynomial.polyfit(
        x, a, deg, w=w, full=True)
        
    fit = np.polynomial.polynomial.polyval(
        np.arange(n), coeffs[0])
    if not return_coeffs:
        return fit
    else:
        return fit, coeffs

def interpolate_size(a, size, deg):
    """Change size of a vector by interpolation

    :param a: vector to interpolate
    
    :param size: New size of the vector
    
    :param deg: Interpolation degree
    """
    if a.dtype != np.complex:
        f = interpolate.UnivariateSpline(np.arange(a.shape[0]), a, k=deg, s=0)
        return f(np.arange(size)/float(size - 1) * float(a.shape[0] - 1))
    else:
        result = np.empty(size, dtype=complex)
        f_real = interpolate.UnivariateSpline(
            np.arange(a.shape[0]), a.real, k=deg, s=0)
        result.real = f_real(
            np.arange(size)/float(size - 1) * float(a.shape[0] - 1))
        f_imag = interpolate.UnivariateSpline(
            np.arange(a.shape[0]), a.imag, k=deg, s=0)
        result.imag = f_imag(
            np.arange(size)/float(size - 1) * float(a.shape[0] - 1))
        return result
        
def interpolate_axis(a, new_axis, deg, old_axis=None, fill_value=np.nan):
    """Interpolate a vector along a new axis.

    :param a: vector to interpolate
    
    :param new_axis: Interpolation axis
    
    :param deg: Interpolation degree
    
    :param old_axis: (Optional) Original vector axis. If None,
      a regular range axis is assumed (default None).

    :param fill_value: (Optional) extrapolated points are filled with
      this value (default np.nan)
    """
    returned_vector=False
    if old_axis is None:
        old_axis = np.arange(a.shape[0])
    elif old_axis[0] > old_axis[-1]:
        old_axis = old_axis[::-1]
        a = a[::-1]
        new_axis = new_axis[::-1]
        returned_vector=True

    if np.sum(np.isnan(a)) > 0:
            nonnans = np.nonzero(~np.isnan(a))
            old_axis = old_axis[nonnans]
            a = a[nonnans]

    if a.dtype != np.complex:
        f = interpolate.UnivariateSpline(old_axis, a, k=deg, s=0)
        result = f(new_axis.astype(np.float64))
    else:
        result = np.empty(new_axis.shape[0], dtype=complex)
        f_real = interpolate.UnivariateSpline(old_axis, np.real(a), k=deg, s=0)
        result.real = f_real(new_axis.astype(np.float64))
        f_imag = interpolate.UnivariateSpline(old_axis, np.imag(a), k=deg, s=0)
        result.imag = f_imag(new_axis.astype(np.float64))
            
    if returned_vector:
        result = result[::-1]
        old_axis = old_axis[::-1]
        new_axis = new_axis[::-1]

    # extrapolated parts are set to fil__value
    result[np.nonzero(new_axis > np.max(old_axis))] = fill_value
    result[np.nonzero(new_axis < np.min(old_axis))] = fill_value
    return result


def robust_unwrap(vec, dis):
    """Unwrap a vector with a given discontinuity. Robust to nans.

    Note that the returned vector will start somewhere around 0 since
    all modulo bias is removed.
    
    :param vec: 1d Vector to unwrap.

    :param dis: discontinuity (eg. np.pi)
    """
    orb.utils.validate.is_1darray(vec)
    
    nvec = np.copy(vec)

    nvec = np.fmod(nvec, dis)
    last = None
    for i in range(nvec.size):
        if not np.isnan(nvec[i]):
            if last is not None:
                nvec[i] = last + np.fmod(nvec[i] - last, dis)
                if np.abs(nvec[i] - last - dis) < np.abs(nvec[i] - last):
                    nvec[i] -= dis
                elif np.abs(nvec[i] - last + dis) < np.abs(nvec[i] - last):
                    nvec[i] += dis
                
                last = float(nvec[i])
            else:
                last = np.fmod(nvec[i], dis)
                if np.abs(last - dis) < np.abs(last):
                    last -= dis
                
                nvec[i] = last
            # validation
            check = (nvec[i] - vec[i]) / dis # should be an integer
            if not np.isclose(check, round(check)): raise RuntimeError('unwrapped value {} is not close to the original value {} modulo {}'.format(nvec[i], vec[i], dis))
    return nvec


def complex2float(a):
    return np.concatenate([a[0], a[1]])

def float2complex(a):
    return (a[:a.size//2], a[a.size//2:])
    
