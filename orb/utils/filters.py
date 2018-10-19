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

