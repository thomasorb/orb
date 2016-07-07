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
import orb.constants



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
    
    .. note:: the FWHM of the sinc is defined as the full width at
      half amplitude (i.e. maximum). A gaussian with the same
      parameters will cross the sinc at the half-max.
    """
    X = ((x-dx)/(fwhm/1.20671))
    return h + a * algopy.special.hyp0f1(1.5, -(0.5 * math.pi * X)**2)

