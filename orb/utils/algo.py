#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: algo.py

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

"""Utils fit functions made for algopy."""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"       


import algopy
import numpy as np
import math

def fast_w2pix(w, axis_min, axis_step):
    """Fast conversion of wavelength/wavenumber to pixel

    :param w: wavelength/wavenumber
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    return algopy.absolute(w - axis_min) / axis_step

def fast_pix2w(pix, axis_min, axis_step):
    """Fast conversion of pixel to wavelength/wavenumber

    :param pix: position along axis in pixels
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    return pix * axis_step + axis_min


def gaussian1d(x, h, a, dx, fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: 1D array of float64 giving the positions where the
      gaussian is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    w = fwhm / (2. * algopy.sqrt(2. * math.log(2.)))
    return  h + a * algopy.exp(-(x - dx)**2. / (2. * w**2.))


def sinc1d(x, h, a, dx, fwhm):
    """Return a 1D sinc given a set of parameters.

    :param x: 1D array of float64 giving the positions where the
      sinc is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    
    X = ((x-dx)/(fwhm/1.20671))
    return h + a * algopy.special.hyp0f1(1.5, -(0.5 * math.pi * X)**2)

def sincgauss1d(x, h, a, dx, fwhm, sigma):
    """Return a 1D sinc convoluted with a gaussian of parameter sigma.

    If sigma == 0 returns a pure sinc.

    :param x: 1D array of float64 giving the positions where the
      sinc is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    :param sigma: Sigma of the gaussian.
    """
    # when sigma/fwhm is too high or too low, a pure sinc or gaussian
    # is returned (avoid overflow)
    if abs(sigma / fwhm) < 1e-2:
        return sinc1d(x, h, a, dx, fwhm)
    if abs(sigma / fwhm) > 1e2:
        return gaussian1d(x, h, a, dx, fwhm)

    sigma = abs(sigma)
    
    fwhm /= math.pi * 1.20671
    e = algopy.exp(-sigma**2. / 2.) / (math.sqrt(2.) * sigma * 1j)
    sig = np.ones_like(x) * 1j * sigma**2 
    dawson1 = (algopy.special.dawsn((sig - (x - dx) / fwhm)
                                   /(math.sqrt(2.) * sigma))
               * algopy.exp(-1j * (x - dx) / fwhm))
    dawson2 = (algopy.special.dawsn((-1. * sig - (x - dx) / fwhm)
                                   / (math.sqrt(2.) * sigma))
               * algopy.exp(1j *(x - dx) / fwhm))
    return algopy.real(h + a * e * (dawson1 - dawson2))



