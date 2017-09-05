#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: sim.py

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

import logging
import numpy as np
import math
import warnings

import orb.cutils
import orb.utils.fft
import orb.utils.vector

def line_interf(sigma, step_nb, phi=0):
    """
    Simulate a simple line interferogram (a cosine)

    ZPD is on the first sample of the returned interferogram.

    :param sigma: line frequency (must be < step_nb/2)

    :param step_nb: Length of the interferogram

    :param phi: (Optional) Phase of the line (in radians) (default 0).
    """
    if sigma > step_nb / 2.: raise Exception('Sigma must be < step_nb/2')
    x = np.arange(step_nb, dtype=float) / (step_nb-1)
    a = np.cos(x*sigma*2.*math.pi + phi) / 2. + 0.5
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

