#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fft.py

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
## or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

import logging
import time
import sys
import numpy as np
import math
import warnings
import scipy
import scipy.special as ss
from scipy import signal, interpolate, optimize
import gvar

import orb.utils.vector
import orb.utils.spectrum
import orb.utils.stats
import orb.utils.filters
import orb.cutils
import orb.constants


def next_power_of_two(n):
    """Return the next power of two greater than n.
    
    :param n: The number from which the next power of two has to be
      computed. Can be an array of numbers.
    """
    return np.array(2.**np.ceil(np.log2(n))).astype(int)

def raw_fft(x, apod=None, inverse=False, return_complex=False,
            return_phase=False):
    """
    Compute the raw FFT of a vector.

    Return the absolute value of the complex vector by default.
    
    :param x: Interferogram.
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.norton_beer_window` (default None)

    :param inverse: (Optional) If True compute the inverse FFT
      (default False).

    :param return_complex: (Optional) If True, the complex vector is
      returned (default False).

    :param return_phase: (Optional) If True, the phase is
      returned.(default False)
    
    """
    x = np.copy(x)
    windows = ['1.1', '1.2', '1.3', '1.4', '1.5',
               '1.6', '1.7', '1.8', '1.9', '2.0']
    N = x.shape[0]
    
    # mean substraction
    x -= np.mean(x)
    
    # apodization
    if apod in windows:
        x *= gaussian_window(apod, N)
    elif apod is not None:
        raise Exception("Unknown apodization function try %s"%
                        str(windows))
        
    # zero padding
    zv = np.zeros(N*2, dtype=x.dtype)
    zv[int(N/2):int(N/2)+N] = x

    # zero the centerburst
    zv = np.roll(zv, zv.shape[0]/2)
    
    # FFT
    if not inverse:
        x_fft = (np.fft.fft(zv))[:N]
    else:
        x_fft = (np.fft.ifft(zv))[:N]
        
    if return_complex:
        return x_fft
    elif return_phase:
        return np.unwrap(np.angle(x_fft))
    else:
        return np.abs(x_fft)
     
def cube_raw_fft(x, apod=None):
    """Compute the raw FFT of a cube (the last axis
    beeing the interferogram axis)

    :param x: Interferogram cube
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.gaussian_window` (default None)
    """
    x = np.copy(x)
    N = x.shape[-1]
    # mean substraction
    x = (x.T - np.mean(x, axis=-1)).T
    # apodization
    if apod is not None:
        x *= gaussian_window(apod, N)

    # zero padding
    zv_shape = np.array(x.shape)
    zv_shape[-1] = N*2
    zv = np.zeros(zv_shape)
    zv[:,int(N/2):int(N/2)+N] = x
    # FFT
    return np.abs((np.fft.fft(zv))[::,:N])

def norton_beer_window(fwhm='1.6', n=1000):
    """
    Return an extended Norton-Beer window function (see [NAY2007]_).

    Returned window is symmetrical.
    
    :param fwhm: FWHM relative to the sinc function. Must be: 1.1,
       1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 or 2.0. (default '1.6')
       
    :param n: Number of points (default 1000)

    .. note:: Coefficients of the extended Norton-Beer functions
       apodizing functions [NAY2007]_ :
    
       ==== ======== ========= ======== ======== ======== ======== 
       FWHM    C0       C1        C2       C4       C6       C8
       ---- -------- --------- -------- -------- -------- -------- 
       1.1  0.701551 -0.639244 0.937693 0.000000 0.000000 0.000000
       1.2  0.396430 -0.150902 0.754472 0.000000 0.000000 0.000000
       1.3  0.237413 -0.065285 0.827872 0.000000 0.000000 0.000000
       1.4  0.153945 -0.141765 0.987820 0.000000 0.000000 0.000000
       1.5  0.077112 0.000000  0.703371 0.219517 0.000000 0.000000
       1.6  0.039234 0.000000  0.630268 0.234934 0.095563 0.000000
       1.7  0.020078 0.000000  0.480667 0.386409 0.112845 0.000000
       1.8  0.010172 0.000000  0.344429 0.451817 0.193580 0.000000
       1.9  0.004773 0.000000  0.232473 0.464562 0.298191 0.000000
       2.0  0.002267 0.000000  0.140412 0.487172 0.256200 0.113948
       ==== ======== ========= ======== ======== ======== ========

    .. [NAY2007] Naylor, D. A., & Tahic, M. K. (2007). Apodizing
       functions for Fourier transform spectroscopy. Journal of the
       Optical Society of America A.
    """
    
    norton_beer_coeffs = [
        [1.1, 0.701551, -0.639244, 0.937693, 0., 0., 0., 0., 0., 0.],
        [1.2, 0.396430, -0.150902, 0.754472, 0., 0., 0., 0., 0., 0.],
        [1.3, 0.237413, -0.065285, 0.827872, 0., 0., 0., 0., 0., 0.],
        [1.4, 0.153945, -0.141765, 0.987820, 0., 0., 0., 0., 0., 0.],
        [1.5, 0.077112, 0., 0.703371, 0., 0.219517, 0., 0., 0., 0.],
        [1.6, 0.039234, 0., 0.630268, 0., 0.234934, 0., 0.095563, 0., 0.],
        [1.7, 0.020078, 0., 0.480667, 0., 0.386409, 0., 0.112845, 0., 0.],
        [1.8, 0.010172, 0., 0.344429, 0., 0.451817, 0., 0.193580, 0., 0.],
        [1.9, 0.004773, 0., 0.232473, 0., 0.464562, 0., 0.298191, 0., 0.],
        [2.0, 0.002267, 0., 0.140412, 0., 0.487172, 0., 0.256200, 0., 0.113948]]

    fwhm_list = ['1.1', '1.2', '1.3', '1.4', '1.5',
                 '1.6', '1.7', '1.8', '1.9', '2.0']
    if fwhm in fwhm_list:
        fwhm_index = fwhm_list.index(fwhm)
    else:
        raise Exception("Bad extended Norton-Beer window FWHM. Must be in : " + str(fwhm_list))

    x = np.linspace(-1., 1., n)

    nb = np.zeros_like(x)
    for index in range(9):
        nb += norton_beer_coeffs[fwhm_index][index+1]*(1. - x**2)**index
    return nb

def apod2width(apod):
    """Return the width of the gaussian window for a given apodization level.

    :param apod: Apodization level (must be >= 1.)

    The apodization level is the broadening factor of the line (an
    apodization level of 2 mean that the line fwhm will be 2 times
    wider).
    """
    if apod < 1.: raise Exception(
        'Apodization level (broadening factor) must be > 1')

    return apod - 1. + (gvar.erf(math.pi / 2. * gvar.sqrt(apod - 1.))
                        * orb.constants.FWHM_SINC_COEFF)

def width2apod(width):
    """This is the inverse of apod2width.  

    As the inverse is at least complicated to compute. This is done via
    minimization.
    """
    def diff(apod, width):
        return apod2width(apod) - width

    if width < 0: raise ValueError('width must be a positive float')

    fit = optimize.least_squares(diff, 1, args=(width, ))
    if fit.success:
        return fit.x[0]
    else:
        raise Exception('error when inverting apod2width: {}'.format(fit.message))

def apod2sigma(apod, fwhm):
    """Return the broadening of the gaussian-sinc function in the
    spectrum for a given apodization level. Unit is that of the fwhm.

    :param apod: Apodization level (must be >= 1.)
    """
    broadening = 2. * (apod2width(apod) / (math.sqrt(2.) * math.pi)
                       / orb.utils.spectrum.compute_line_fwhm_pix(
                           oversampling_ratio=1))

    return broadening * fwhm

def sigma2apod(sigma, fwhm):
    """This is the inverse of apod2sigma.

    As the inverse is at least complicated to compute. This is done via
    minimization.
    """
    def diff(apod, sigma, fwhm):
        return apod2sigma(apod, fwhm) - sigma

    if sigma < 0: raise ValueError('sigma must be a positive float')
    if fwhm <= 0: raise ValueError('fwhm must be a strictly positive float')

    fit = optimize.least_squares(diff, 1, args=(sigma, fwhm))
    if fit.success:
        return fit.x[0]
    else:
        raise Exception('error when inverting apod2sigma: {}'.format(fit.message))
    
def gaussian_window(coeff, x):
    """Return a Gaussian apodization function for a given broadening
    factor.

    :param coeff: FWHM relative to the sinc function. Must be a float > 1.

    :param x: Must be an axis defined between -1 and 1 inclusively.
      x = np.linspace(-1., 1., n) for a symmetrical window.
    """
    coeff = float(coeff)
    #x = np.linspace(-1., 1., n)
    w = apod2width(coeff)
    return np.exp(-x**2 * w**2)

def learner95_window(x):
    """Return the apodization function described in Learner et al.,
    J. Opt. Soc. Am. A, 12, (1995).

    This function is closely related to the minimum four-term
    Blackman-Harris window.
    
    :param x: Must be an axis defnined between -1 and 1 inclusively.
      x = np.linspace(-1., 1., n) for a symmetrical window.
    """
    #
        
    return (0.355766
            + 0.487395 * np.cos(math.pi*x)
            + 0.144234 * np.cos(2.*math.pi*x)
            + 0.012605 * np.cos(3.*math.pi*x))

def border_cut_window(n, coeff=0.2):
    """Return a window function with only the edges cut by a nice
    gaussian shape function.
    
    :param n: Window length
    :param coeff: Border size in percentage of the total length.
    """
    window = np.ones(n)
    border_length = int(float(n)*coeff)
    if border_length <= 1:
        window[0] = 0.
        window[-1] = 0.
    else:
        borders = signal.get_window(("gaussian",border_length/3.),
                                    border_length*2+1)
        z = int(float(borders.shape[0])/2.)
        window[:z] = borders[:z]
        window[-z:] = borders[-z:]
    return window


def ndft(a, xk, vj):
    """Non-uniform Discret Fourier Tranform

    Compute the spectrum from an interferogram. Note that the axis can
    be irregularly sampled.

    If the spectral axis (output axis) is irregular the result is
    exact. But there is no magic: if the input axis (interferogram
    sampling) is irregular the output spectrum is not exact because
    the projection basis is not orthonormal.

    If the interferogram is the addition of multiple regularly sampled
    scans with a opd shift between each scan, the result will be good
    as long as there are not too much scans added one after the
    other. But if the interferogram steps are randomly distributed, it
    will be better to use a classic FFT because the resulting noise
    will be much lower.

    :param a: 1D interferogram
    
    :param xk: 1D sampling steps of the interferogram. Must have the
      same size as a and must be relative to the real step length,
      i.e. if the sampling is uniform xk = np.arange(a.size).
    
    :param vj: 1D frequency sampling of the output spectrum.
    """
    assert a.ndim == 1, 'a must be a 1d vector'
    assert vj.ndim == 1, 'vj must be a 1d vector'
    assert a.size == xk.size, 'size of a must equal size of xk'
    
    angle = np.inner((-2.j * np.pi * xk / xk.size)[:,None], vj[:,None])
    return np.dot(a, np.exp(angle))


def indft(a, x):
    """Inverse Non-uniform Discret Fourier Transform.

    Compute the irregularly sampled interferogram from a regularly
    sampled spectrum.

    :param a: regularly sampled spectrum.
    
    :param x: positions of the interferogram samples. If x =
      range(size(a)), this function is equivalent to an idft or a
      ifft. Note that the ifft is of course much faster to
      compute. This vector may have any length.
    """
    return orb.cutils.indft(a.astype(float), x.astype(float))

def spectrum_mean_energy(spectrum):
    """Return the mean energy of a spectrum by channel.

    :param spectrum: a 1D spectrum
    """
    return orb.cutils.spectrum_mean_energy(spectrum)

def interf_mean_energy(interf):
    """Return the mean energy of an interferogram by step.

    :param interf: an interferogram

    .. warning:: The mean of the interferogram is substracted to
      compute only the modulation energy. This is the modulation
      energy which must be conserved in the resulting spectrum. Note
      that the interferogram transformation function (see
      :py:meth:`utils.transform_interferogram`) remove the mean of the
      interferogram before computing its FFT.

    .. note:: NaNs are set to 0.
    """
    return orb.cutils.interf_mean_energy(interf)

def phase_model(sigma, costheta, *p):
    if len(p) != 2:
        raise Exception('p must have len 2, not {}'.format(len(p)))
    return np.polynomial.polynomial.polyval(sigma, p)
    #return np.polynomial.polynomial.polyval(sigma, p) * (1 - costheta)
    
