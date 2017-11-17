#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: sim.py

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
import math
import warnings

import orb.cutils
import orb.utils.fft
import orb.utils.vector
import orb.utils.spectrum
import orb.constants

def step_interf(sigma_min, sigma_max, step_nb, symm=False):
    """Simulate a step interferogram

    ZPD is on the first sample of the returned interferogram.

    :param sigma_ax: max frequency of the step (must be < step_nb/2)

    :param step_nb: Length of the interferogram

    :param symm: (Optional) If True, returned spectrum is symmetric,
      it has two times more steps - 1. Zpd position is equal to
      step_nb - 0.5.
   
    .. note:: results are much better with a symmetric interferogram
      but ZPD is not on the first sample and the spectrum must thus be
      phase corrected.
    """
    if sigma_max > step_nb / 2.: raise ValueError('sigma_max must be < step_nb/2')
    if sigma_min >= sigma_max: raise ValueError('sigma_min must be < sigma_max')
    if not 0 <= sigma_min < sigma_max: raise ValueError('sigma_min must be < sigma_max and > 0')

    x = np.arange(step_nb, dtype=float)
    x = x / (step_nb - 1.)
    
    fwhm = 1. / (float(sigma_max - sigma_min)) * orb.constants.FWHM_SINC_COEFF
    interf = orb.utils.spectrum.sinc1d(x, 0, 1, 0, fwhm)
    interf *= np.cos(2 * np.pi * x * (sigma_min + (sigma_max - sigma_min)/2.))

    if symm:
        return np.hstack((interf[::-1][:-1], interf))
    else:
        return interf


def line_interf(sigma, step_nb, phi=0, symm=False):
    """
    Simulate a simple line interferogram (a cosine)

    ZPD is on the first sample of the returned interferogram.

    :param sigma: line frequency (must be < step_nb/2)

    :param step_nb: Length of the interferogram

    :param phi: (Optional) Phase of the line (in radians) (default 0).

    :param symm: (Optional) If True, returned spectrum is symmetric,
      it has two times more steps - 1. Zpd position is equal to
      step_nb - 0.5.
    """
    if sigma > step_nb / 2.: raise ValueError('Sigma must be < step_nb/2')
    x = np.arange(step_nb, dtype=float) / (step_nb-1)
    a = np.cos(x*sigma*2.*math.pi + phi) / 2. + 0.5
    if symm:
        return np.hstack((a[::-1][:-1], a))
    else:
        return a

def fft(interf, zp_coeff=10, apod=None, phase=None):
    """
    Basic Fourier Transform with zero-padding.

    Useful to compute a quick assumption-less FFT.

    ZPD is assumed to be on the first sample of the interferogram
    
    :param a: interferogram

    :param zp_coeff: Zero-padding coefficient

    :param apod: Apodization function

    :return: axis, complex interferogram FFT
    """
    step_nb = interf.shape[0]

    # remove mean
    interf = np.copy(interf) - np.nanmean(interf)

    # apodization
    if apod is not None:        
        apodf = orb.utils.fft.gaussian_window(apod, step_nb*2)[step_nb:]
        interf *= apodf

        
    # zero padding
    zp_nb = step_nb * zp_coeff * 2
    zp_interf = np.zeros(zp_nb, dtype=float)
    zp_interf[:step_nb] = interf

    # dft
    interf_fft = np.fft.fft(zp_interf)
    if phase is not None:
        interf_fft *= np.exp(1j * orb.utils.vector.interpolate_size(phase, interf_fft.shape[0], 1))
    interf_fft = interf_fft[:interf_fft.shape[0]/2+1]
    axis = np.linspace(0, (step_nb - 1)/2., interf_fft.shape[0])
    return axis, interf_fft

