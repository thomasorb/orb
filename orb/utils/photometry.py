#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: photometry.py

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
import math
import numpy as np
import warnings
import orb.constants
import scipy.interpolate
import scipy.optimize
import orb.utils.spectrum
from orb.utils.astrometry import Gaussian
 
def flambda2ABmag(flambda, lam):
    """Return AB magnitude from flux in erg/cm2/s/A

    :param flambda: Flux in erg/cm2/s/A. Can be an array.

    :param lambda: Wavelength in A of the Flux. If flambda is an array
      lambda must have the same shape.
    """
    fnu = lam**2. / orb.constants.LIGHT_VEL_AAS * flambda
    ABmag = -2.5 * np.log10(fnu) - 48.60
    return ABmag

def ABmag2fnu(ABmag):
    """Return flux in erg/cm2/s/Hz from AB magnitude (Oke, ApJS, 27,
    21, 1974)

    ABmag = -2.5 * log10(f_nu) - 48.60
    f_nu = 10^(-0.4 * (ABmag + 48.60))

    :param ABmag: A magnitude in the AB magnitude system

    .. note:: Definition of the zero-point can change and be
      e.g. 48.59 for Oke standard stars (Hamuy et al., PASP, 104, 533,
      1992). This is the case for Spectrophotometric Standards given
      on the ESO website (https://www.eso.org/sci/observing/tools/standards/spectra/okestandards.html). Here the HST definition is used.
    """
    return 10**(-0.4*(ABmag + 48.60))

def fnu2flambda(fnu, nu):
    """Convert a flux in erg/cm2/s/Hz to a flux in erg/cm2/s/A

    :param fnu: Flux in erg/cm2/s/Hz
    :param nu: frequency in Hz
    """
    return fnu * nu**2. / orb.constants.LIGHT_VEL_AAS

def lambda2nu(lam):
    """Convert lambda in Ang to nu in Hz

    :param lam: Wavelength in angstrom
    """
    return orb.constants.LIGHT_VEL_AAS / lam

def ABmag2flambda(ABmag, lam):
    """Convert AB magnitude to flux in erg/cm2/s/A

    :param ABmag: A magnitude in the AB magnitude system

    :param lam: Wavelength in angstrom
    """
    return fnu2flambda(ABmag2fnu(ABmag), lambda2nu(lam))


def ext2trans(ext, airmass):
    """Convert extinction to transmission

    :param ext: extinction

    :param airmass: airmass
    """
    return 10**(-ext*airmass/2.5)

def compute_mean_star_flux(star_spectrum, filter_transmission):
    """Return mean star flux given a spectrum and the filter. Both
    files must be given along the same wavelength/wavenumber axis.

    :param star_spectrum: Spectrum of the star
    :param filter_transmission: Filter transmission curve
    """
    return (np.sum(star_spectrum * filter_transmission)
            / np.sum(filter_transmission))


def compute_mean_photon_energy(nm_axis, filter_transmission):
    """Return mean energy of the photons passing thourgh a given
    filter.
    
    :param nm_axis: Filter transmission axis in nm.
    :param filter_transmission: Filter transmission curve
    """
    ph_energy_spectrum = compute_photon_energy(nm_axis)
    return (np.sum(ph_energy_spectrum * filter_transmission)
            / np.sum(filter_transmission))

def compute_photon_energy(nm_axis):
    """Return the photon energy computed for all the wavelength along
    a given wavelength axis.

    :param nm_axis: Wavelength axis in nm
    """
    return (orb.constants.PLANCK
            * orb.constants.LIGHT_VEL_KMS * 1e12
            / nm_axis)

def compute_equivalent_bandwidth(nm_axis, filter_transmission):
    """Return the equivalent bandwidth of a given filter.

    :param nm_axis: Filter transmission axis in nm.
    :param filter_transmission: Filter transmission curve
    """
    return np.nansum(np.diff(nm_axis) * filter_transmission[:-1]
                     / np.nanmax(filter_transmission))


def compute_star_central_pixel_value(seeing, plate_scale):
    """Return the relative value of the pixel containing the greatest
    proportion of the flux (central pixel) of Gaussian star.

    :param seeing: Star FWHM in arcsec
    :param plate_scale: Size of the pixels in arcsec.
    """
    N = 100
    fwhm_pix = seeing / plate_scale
    star = Gaussian([0,1,N/2,N/2,fwhm_pix]).array2d(N,N)
    star /= np.nansum(star)
    return np.nanmax(star)

def compute_optimal_texp(star_flux, seeing, plate_scale,
                         saturation=30000):
    """Compute the optimal exposure time given the total flux of the
    star in ADU/s.

    :param star_flux: Total star flux in ADU/s
    :param seeing: Star FWHM in arcsec
    :param plate_scale: Size of 1 pixel in arcsec.
    :param saturation: (Optional) Saturation value (default 30000).
    """

    logging.info('Optimal exposure time is computed for a saturation value of: {} counts'.format(saturation))
    max_flux = compute_star_central_pixel_value(
        seeing, plate_scale) * star_flux

    return saturation/max_flux
    
def fit_std_spectrum(real_spectrum, std_spectrum, polydeg=2):
    """Fit a real spectrum multiplied by a polynomial over a standard
    spectrum.

    Return the polynomial which can be used directly as a calibration
    curve.

    :param real_spectrum: Observed spectrum
    :param std_spectrum: Standard spectrum
    :param polydeg: Degree of the polynomial
    """

    def model(p, x, real_spectrum):
        return (real_spectrum
                * np.polynomial.polynomial.polyval(x, p))


    def diff(p, x, std_spectrum, real_spectrum):
        res = model(p, x, real_spectrum) - std_spectrum
        return res[~np.isnan(res)]

    p = np.zeros((polydeg+1), dtype=float)
    x = np.arange(real_spectrum.shape[0], dtype=float)
    p[0] = np.nanmedian(std_spectrum / real_spectrum)
    
    p = scipy.optimize.leastsq(
        diff, p, args=(
            x, std_spectrum,
            real_spectrum))[0]
    
    return np.polynomial.polynomial.polyval(x, p)


def convert_cm1_flux2fluxdensity(a, cm1_axis):
    """ Convert a spectrum in X/s to X/s/A (X may be ADU or erg/cm^2)

    :param a: spectrum

    :param cm1_axis: Axis of the spectrum in cm-1
    """

    _cm1_axis_plus1 = np.hstack((cm1_axis, cm1_axis[-1]
                                 + np.diff(cm1_axis)[0]))
    _channel_A = np.abs(np.diff(1e7/_cm1_axis_plus1)) * 10.
    
    return np.copy(a) / _channel_A.astype(float)
    

def compute_flux_calibration_vector(re_spectrum, th_spectrum,
                                    std_step, std_order, std_exp_time,
                                    std_corr, filter_min_pix, filter_max_pix):
    """Compute the flux calibration vector from an observed spectrum
    and the standard spectrum.

    :param re_spectrum: Observed spectrum in wavenumber.

    :param th_spectrum: standard spectrum (in erg/cm2/s/A) in wavenumber.

    :param std_step: Standard step size (in nm)

    :param std_order: Standard folding order

    :param std_exp_time: Standard Exposition time

    :param std_corr: Standard Correction coeff.

    :param filter_min_pix: Filter min position in pixels

    :param filter_max_pix: Filter max position in pixels
    """
    
    std_step_nb = re_spectrum.shape[0]
    std_cm1_axis = orb.utils.spectrum.create_cm1_axis(
        std_step_nb, std_step, std_order, corr=std_corr)

    # Real spectrum is converted to ADU/s
    # We must divide by the total exposition time
    re_spectrum /= std_exp_time * std_step_nb # ADU -> ADU/s

    # Real spectrum is converted to ADU/A/s
    re_spectrum  = convert_cm1_flux2fluxdensity(
        re_spectrum, std_cm1_axis)

    # Remove portions outside the filter
    border = int(0.02 * np.size(re_spectrum))
    re_spectrum[:filter_min_pix + border] = np.nan
    re_spectrum[filter_max_pix - border:] = np.nan

    th_spectrum[np.nonzero(np.isnan(re_spectrum))] = np.nan

    # fit model * polynomial to adjust model and spectrum
    flux_calibf = orb.utils.photometry.fit_std_spectrum(
        re_spectrum, th_spectrum)


    ## import pylab as pl
    ## pl.plot(std_cm1_axis, re_spectrum * flux_calibf)
    ## pl.plot(std_cm1_axis, th_spectrum)
    ## pl.show()

    logging.info('Mean theoretical flux of the star: %e ergs/cm^2/A/s'%orb.utils.stats.robust_mean(th_spectrum))
    logging.info('Mean flux of the star in the cube: %e ADU/A/s'%orb.utils.stats.robust_mean(re_spectrum))
    logging.info('Mean Flambda calibration: %e ergs/cm^2/[ADU]'%np.nanmean(flux_calibf[~np.isnan(th_spectrum)]))

    return std_cm1_axis, flux_calibf


def modulation_efficiency_opd_jitter(cm1, opd_jitter):
    """Return the ME given the OPD jitter

    :param cm1: wavenumber in cm-1

    :param opd_jitter: OPD jitter in nm (standard deviation)
    """
    return 1. - (2. * np.pi**2 * opd_jitter**2 * (cm1/1e7)**2)

def modulation_efficiency_wavefront_error(cm1, wferr):
    """Return the ME given the OPD jitter

    :param cm1: wavenumber in cm-1

    :param wferr: wavefront error ratio (e.g. 1/30.)
    """
    def delta_sq(ratio_mirrors, ratio_bs):
        """total fw error in a plane mirror configuration"""
        V_mirrors = (632.8 * ratio_mirrors / 4.)**2.
        V_bs = (632.8 * ratio_bs / 4.)**2.
    
        return 8 * (V_mirrors + 2 * V_bs * np.cos(np.deg2rad(33.5))**2 + V_bs)
    
    return 1. - (2. * np.pi**2 * delta_sq(wferr, wferr) * (cm1/1e7)**2)

