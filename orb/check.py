#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: check.py

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

"""Module containing functions used to check the internal coherency of
ORB.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"

import numpy as np

import orb.core
import orb.fit
import orb.utils.spectrum
import orb.fft

def fit(lines_cm1, filtername, step_nb, amp=None, snr=0, theta=None,
        sigma=0, vel=0, alpha=0., fmodel='sincgauss', instrument='sitelle', ratio=0.25):
    """Create a model and fit it.

    This is a good way to check the quality and the internal coherency
    of the fitting routine.

    :param lines_cm1: Lines rest wavenumber in cm-1

    :param amp: Amplitude of the lines

    :param step: Step size in nm

    :param oder: Folding order

    :param resolution: Resolution

    :param theta: Incident angle

    :param snr: SNR of the strongest line

    :param sigma: (Optional) Line broadening in km/s (default 0.)

    :param vel: (Optional) Velocity in km/s (default 0.)

    :param alpha: (Optional) Phase coefficient of the lines (default
      0.)
    """
    filt = orb.core.FilterFile(filtername).params

    lines_cm1 = np.array(lines_cm1)
    if amp is None:
        amp = np.ones_like(lines_cm1)
    else:
        assert len(amp) == len(lines_cm1), 'amp must have the same len as the number of lines'

    if theta is None:
        theta = orb.core.Tools(instrument=instrument).config['OFF_AXIS_ANGLE_CENTER']
        
    corr = orb.utils.spectrum.theta2corr(theta)
    zpd_index = int(step_nb * (1. - 1. / (1. + ratio)))

    spectrum = orb.fit.create_cm1_lines_model_raw(lines_cm1, amp, filt.step,
                                                  filt.order, step_nb, corr, zpd_index, 
                                                  sigma=sigma, vel=vel, alpha=alpha,
                                                  fmodel=fmodel)
    
    cm1_axis = orb.utils.spectrum.create_cm1_axis(step_nb, filt.step, filt.order, corr=corr)
    
    # add noise (note that this is not a true fft noise, for a true
    # fft noise use sim.py module)
    if snr > 0.:
        spectrum += np.random.standard_normal(
            spectrum.shape[0]) * np.nanmax(amp)/snr

    params = dict(filt)
    params['filter_name'] = filtername
    params['calib_coeff'] = corr
    params['calib_coeff_orig'] = corr
    params['step_nb'] = step_nb
    params['nm_laser'] = orb.core.Tools(instrument=instrument).config['CALIB_NM_LASER']
    params['zpd_index'] = zpd_index
    params['apodization'] = 1
    params['wavenumber'] = True
    
    spectrum = orb.fft.Spectrum(spectrum, axis=cm1_axis, params=params)

    kwargs = dict()
    if 'sincgauss' in fmodel:
        kwargs['sigma_def'] = '1'
        kwargs['sigma_cov'] = sigma
        
    if 'phased' in fmodel:
        kwargs['alpha_def'] = '1'
        kwargs['alpha_cov'] = alpha

    fwhm_def = 'fixed'
    if fmodel == 'gaussian':
        fwhm_def = '1'

    return spectrum, spectrum.fit(
        lines_cm1,
        pos_cov=vel,
        pos_def='1',
        fwhm_def=fwhm_def,
        fmodel=fmodel,
        **kwargs)
