#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fit.py

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
import scipy.signal
import orb.constants
import gvar
import orb.utils.validate
import orb.utils.spectrum
import orb.utils.stats
import copy

def vel2sigma(vel, lines, axis_step):
    """Convert a velocity in km/s to a broadening in channels.
    :param vel: velocity in km/s
    :param lines: line position in the unit of axis_step
    :param axis_step: axis step size
    """
    sigma = lines * vel / orb.constants.LIGHT_VEL_KMS
    sigma /= axis_step # convert sigma cm-1->pix
    return gvar.fabs(sigma)


def sigma2vel(sigma, lines, axis_step):
    """Convert a broadening in channels to a velocity in km/s
    :param sigma: broadening in channels
    :param lines: line position in the unit of axis_step
    :param axis_step: axis step size
    """
    vel = sigma * axis_step # convert sigma pix->cm-1
    # convert sigma cm-1-> km/s
    vel = orb.constants.LIGHT_VEL_KMS * vel / lines
    return gvar.fabs(vel)


def gvardict2pickdict(gvardict):
    """Convert a dictionary containing gvars into a nice pickable
    dictionary with couples of _mean / _sdev keys.

    Use the pickdict2gvardict to rerturn to the original dictionary.
    """
    if not isinstance(gvardict, dict):
        raise TypeError('gvardict must be a dict instance')
    idict = dict()    
    for ikey in gvardict:
        try:
            _mean = gvar.mean(gvardict[ikey])
            _sdev = gvar.sdev(gvardict[ikey])
            if not np.all(_sdev == 0.):
                idict[ikey + '_mean'] = _mean
                idict[ikey + '_sdev'] = _sdev
            else: raise TypeError
        except TypeError:
            idict[ikey] = gvardict[ikey]
        except AttributeError:
            idict[ikey] = gvardict[ikey]
    return idict

def pickdict2gvardict(pickdict):
    """Invert gvardict2pickdict."""
    if not isinstance(pickdict, dict):
        raise TypeError('pickdict must be a dict instance')
    pickdict = copy.copy(pickdict)
    idict = dict()
    _all_keys = list(pickdict.keys())
    for ikey in _all_keys:
        if ikey in pickdict:
            if '_mean' in ikey:
                prefix = ikey[:-len('_mean')]
                if prefix + '_sdev' in pickdict:
                    idict[prefix] = gvar.gvar(pickdict[ikey], pickdict[prefix + '_sdev'])
                    del pickdict[ikey]
                    del pickdict[prefix + '_sdev']
                else:
                    idict[ikey] = pickdict[ikey]
            else:
                idict[ikey] = pickdict[ikey]
    return idict

def paramslist2pick(paramslist):
    orb.utils.validate.is_iterable(paramslist, object_name='paramslist')
    pick = list()
    for idict in paramslist:
        pick.append(gvardict2pickdict(idict))
    return pick

def pick2paramslist(picklist):
    orb.utils.validate.is_iterable(picklist, object_name='picklist')
    paramslist = list()
    for idict in picklist:
        paramslist.append(pickdict2gvardict(idict))
    return paramslist


def get_comb(lines_cm1, vel, axis, oversampling_ratio):
    axis_step = axis[1] - axis[0]
    lines_cm1 = np.copy(lines_cm1)
    fwhm_pix = orb.utils.spectrum.compute_line_fwhm_pix(oversampling_ratio=oversampling_ratio)
    lines_cm1 += orb.utils.spectrum.line_shift(vel, lines_cm1, wavenumber=True, relativistic=False)
    lines_pix = (lines_cm1 - axis[0]) / axis_step
    
    comb = np.zeros_like(axis)
    x = np.arange(len(axis))
    flux = orb.utils.spectrum.gaussian1d_flux(1, fwhm_pix)
    for iline in lines_pix:
        comb += orb.utils.spectrum.gaussian1d(x, 0, 1, iline, fwhm_pix) / flux
    return comb / len(lines_pix)

def prepare_combs(lines_cm1, axis, vel_range, oversampling_ratio, precision):
    assert precision >= 1, 'precision must be >= 1'
    axis_step_vel = orb.utils.spectrum.compute_radial_velocity(lines_cm1[0] - axis[1] + axis[0], lines_cm1[0], wavenumber=True, relativistic=False)
    vels = np.linspace(vel_range[0], vel_range[1], max(3, int((vel_range[1] - vel_range[0]) / axis_step_vel * precision) + 1))
    
    combs = list()
    for i in range(len(vels)):
        combs.append(get_comb(lines_cm1, vels[i], axis, oversampling_ratio))
    return combs, vels

def estimate_velocity_prepared(spectrum, vels, combs, filter_range_pix, max_comps, threshold=2.5, return_score=False):
    """Provide a velocity estimate. Most of the input should be computed
    with a dedicated function such as
    fft.Spectrum.prepare_velocity_estimate.


    :param threshold: Detection threshold as a factor of the std
       of the calculated score.

    """
    spectrum = np.copy(spectrum)
    score = np.empty_like(vels)
    _spec = spectrum.real[filter_range_pix[0]:filter_range_pix[1]]
    back = np.nanmedian(_spec)
    _spec -= back
    std = orb.utils.stats.unbiased_std(_spec)
    for i in range(len(vels)):
        score[i] = np.nansum(combs[i][filter_range_pix[0]:filter_range_pix[1]] * _spec)
    score /= np.nanmax(score)
    score[np.isnan(score)] = 0
    peaks = scipy.signal.find_peaks(
        score,
        height=(threshold * orb.utils.stats.unbiased_std(score), 1))
    peaks = peaks[0][np.argsort(peaks[1]['peak_heights'])[::-1]]
    estimated_vels = list([vels[ipeak] for ipeak in peaks])
    if max_comps <= len(estimated_vels):
        estimated_vels = estimated_vels[:max_comps]
    else:
        estimated_vels += list([np.nan]) * (max_comps - len(estimated_vels))
    if return_score:
        return score
    return estimated_vels

def estimate_flux(spectrum, axis, lines_cm1, vel, filter_range_pix, oversampling_ratio):
    spectrum = np.copy(spectrum)
    axis_step = axis[1] - axis[0]
    lines_cm1 = np.copy(lines_cm1)
    fwhm_pix = orb.utils.spectrum.compute_line_fwhm_pix(oversampling_ratio=oversampling_ratio)
    lines_cm1 += orb.utils.spectrum.line_shift(vel, lines_cm1, wavenumber=True, relativistic=False)
    lines_pix = (lines_cm1 - axis[0]) / axis_step

    _spec = spectrum.real[int(filter_range_pix[0]):int(filter_range_pix[1])]
    back = np.nanmedian(_spec)
    _spec -= back

    fluxes = list()
    try:
        lines_pix[0]
    except IndexError:
        lines_pix = [lines_pix, ]
    for iline in lines_pix:
        if np.isnan(iline):
            fluxes.append(np.nan)
            continue
        iline -= int(filter_range_pix[0])
        fluxes.append(np.nansum(_spec[int(iline-fwhm_pix*3):int(iline+fwhm_pix*3)+1]))
    return fluxes

def BIC(residual, k):
    """
    Bayesian Information Criterion
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    
    :param residual: Fit residual (as a vector)
    
    :param k: Number of free parameters of the model
    """
    n = residual.size    
    return n * np.log(np.sum(residual**2)/n) + k * np.log(n)
