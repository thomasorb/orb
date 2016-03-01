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

import numpy as np
import math
import warnings

import orb.cutils



def line_interf(sigma, step_nb):
    """
    Simulate a simple line interferogram (a cosine)

    :param sigma: line frequency (must be < step_nb/2)

    :param step_nb: Length of the interferogram
    """
    if sigma > step_nb / 2.: raise Exception('Sigma must be < step_nb/2')
    x = np.arange(step_nb, dtype=float) / (step_nb-1)
    a = np.cos(x*sigma*2.*math.pi) / 2. + 0.5
    return a
    


def fft(interf, zp_coeff=10):
    """
    Basic Fourier Transform with zero-padding.

    Useful to compute a quick assumption-less FFT.
    
    :param a: interferogram

    :param zp_coeff: Zero-padding coefficient

    :return: axis, complex interferogram FFT
    """
    step_nb = interf.shape[0]
    zp_nb = step_nb * zp_coeff * 2
    #interf_fft = np.fft.fft(interf - np.nanmean(interf), n=zp_nb)
    zp_interf = np.zeros(zp_nb, dtype=float)
    zp_interf[:step_nb] = interf - np.nanmean(interf)
    interf_fft = np.fft.fft(zp_interf)
    interf_fft = interf_fft[:interf_fft.shape[0]/2+1]
    axis = np.linspace(0, (step_nb - 1)/2., interf_fft.shape[0])
    return axis, interf_fft

    
