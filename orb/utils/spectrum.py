#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: spectrum.py

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

import numpy as np
import math
from scipy import interpolate, special, fftpack
import gvar

import orb.constants
import orb.cutils
import orb.cgvar


def create_nm_axis(n, step, order, corr=1.):
    """Create a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order (cannot be 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    
    nm_min = orb.cutils.get_nm_axis_min(int(n), float(step),
                                        int(order), corr=float(corr))
    if (order > 0): 
        nm_max = orb.cutils.get_nm_axis_max(int(n), float(step),
                                            int(order), corr=float(corr))
        return np.linspace(nm_min, nm_max, n, dtype=np.longdouble)
    else:
        raise Exception("order must be > 0")
    
def create_cm1_axis(n, step, order, corr=1.):
    """Create a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    cm1_min = orb.cutils.get_cm1_axis_min(int(n), float(step),
                                          int(order), corr=float(corr))
    cm1_max = orb.cutils.get_cm1_axis_max(int(n), float(step),
                                          int(order), corr=float(corr))
    return np.linspace(cm1_min, cm1_max, n, dtype=np.longdouble) 
    
    
def create_nm_axis_ireg(n, step, order, corr=1.):
    """Create an irregular wavelength axis from the regular wavenumber
    axis in cm-1.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order (must be > 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if order > 0:
        return (1. / create_cm1_axis(n, step, order, corr=corr) * 1e7)
    else:
        raise Exception("Order must be > 0")
        
    
def pix2nm(nm_axis, pix):
     """Convert a pixel position to a wavelength in nm given an axis
     in nm

     .. warning:: Slow because of interpolation : using
       fast_pix2w is much faster.

     :param nm_axis: Axis in nm
     
     :param pix: Pixel position
     """  
     f = interpolate.interp1d(np.arange(nm_axis.shape[0]), nm_axis,
                              bounds_error=False, fill_value=np.nan)
     return f(pix)
   

def nm2pix(nm_axis, nm):
     """Convert a wavelength in nm to a pixel position given an axis
     in nm

     .. warning:: Slow because of interpolation : using
       fast_w2pix is much faster.

     :param nm_axis: Axis in nm
     
     :param nm: Wavelength in nm
     """
     x = np.arange(nm_axis.shape[0])
     inverted = False
     if nm_axis[0] > nm_axis[-1]:
         nm_axis = np.copy(nm_axis[::-1])
         x = x[::-1]
         inverted = True
     f = interpolate.interp1d(nm_axis, x, bounds_error=False, fill_value=np.nan)
     if not inverted:
         return f(nm)
     else:
         return f(nm)[::-1]

def nm2cm1(nm):
    """Convert a wavelength in nm to a wavenumber in cm-1.

    :param nm: wavelength in nm
    """
    try:
        return 1e7 / np.array(nm).astype(float)
    except TypeError:
        return 1e7 / nm

def cm12nm(cm1):
    """Convert a wavenumber in cm-1 to a wavelength in nm.

    :param cm1: wavenumber in cm-1
    """
    try:
        return 1e7 / np.array(cm1).astype(float)
    except TypeError:
        return 1e7 / cm1

def pix2cm1(cm1_axis, pix):
     """Convert a wavenumber in cm-1 to a pixel position given an axis
     in cm-1.

     :param cm1_axis: Axis in cm-1
     
     :param pix: Pixel position
     """
     f = interpolate.interp1d(np.arange(cm1_axis.shape[0]), cm1_axis,
                              bounds_error=False, fill_value=np.nan)

     return f(pix)
 
def cm12pix(cm1_axis, cm1):
     """Convert a wavenumber in cm-1 to a pixel position given an axis
     in cm-1.

     :param cm1_axis: Axis in cm-1
     
     :param cm1: Wavenumber in cm-1
     """
     f = interpolate.interp1d(cm1_axis, np.arange(cm1_axis.shape[0]),
                              bounds_error=False, fill_value=np.nan)
     return f(cm1)

def fwhm_nm2cm1(fwhm_nm, nm):
    """Convert a FWHM in nm to a FWHM in cm-1.
    
    The central wavelength in nm of the line must also be given

    :param fwhm_nm: FWHM in nm
    
    :param nm: Wavelength in nm where the FWHM is evaluated
    """
    return 1e7 * fwhm_nm / nm**2.

def fwhm_cm12nm(fwhm_cm1, cm1):
    """Convert a FWHM in cm-1 to a FWHM in nm.
    
    The central wavelength in cm-1 of the line must also be given

    :param fwhm_cm1: FWHM in cm-1
    
    :param cm1: Wavelength in cm-1 where the FWHM is evaluated
    """
    return 1e7 * fwhm_cm1 / cm1**2.


def velocity_relativistic2classic(vel):
    """Convert a relativistic velocity to a classic velocity

    :param vel: Relativistic velocity in km/s
    """
    line = 1
    shift = line_shift(vel, line, wavenumber=False, relativistic=True)
    return orb.utils.spectrum.compute_radial_velocity(
        line + shift, line, wavenumber=False, relativistic=False)

def velocity_classic2relativistic(vel):
    """Convert a classic velocity to a relativistic velocity

    :param vel: Classic velocity in km/s
    """
    line = 1
    shift = line_shift(vel, line, wavenumber=False, relativistic=False)
    return orb.utils.spectrum.compute_radial_velocity(
        line + shift, line, wavenumber=False, relativistic=True)

def compute_radial_velocity(line, rest_line, wavenumber=False, relativistic=True):
    """
    Return relativistic radial velocity in km.s-1

    V [km.s-1] = c [km.s-1]* (Lambda^2 / Lambda_0^2 - 1) / (Lambda^2 / Lambda_0^2 + 1)

    :param line: Emission line wavelength/wavenumber (can be a numpy
      array)
    
    :param rest_line: Rest-frame wavelength/wavenumber (can be a numpy
      array but must have the same size as line)

    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    if wavenumber:
        line = cm12nm(line)
        rest_line = cm12nm(rest_line)
    if relativistic:
        ratio = (line / rest_line)**2.
        vel = orb.constants.LIGHT_VEL_KMS * (ratio - 1) / (ratio + 1)
    else:
        vel = (line - rest_line) / rest_line * orb.constants.LIGHT_VEL_KMS
    return vel


def line_shift(velocity, line, wavenumber=False, relativistic=True):
    """Return the line shift (in nm or in cm-1) given its relativistic velocity 

    beta = v / c

    gamma = sqrt((1 + beta) / (1 - beta))

    lambda - lambda_0 = lambda_0  * (gamma - 1)

    :param velocity: Line velocity in km.s-1

    :param line: Wavelength/wavenumber of the line. Must be in cm-1 if
      wavenumber is True, must be in nm otherwise.

    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    if wavenumber:
        line = cm12nm(line)
        
    beta = velocity / orb.constants.LIGHT_VEL_KMS
    if relativistic:
        gamma = gvar.sqrt((1. + beta) / (1. - beta))
        shift = line * (gamma - 1.)
    else:
        shift = line * beta

    if wavenumber:
        shift = nm2cm1(line + shift) - nm2cm1(line)
        
    return shift

def compute_line_fwhm(step_nb, step, order, apod_coeff=1., corr=1.,
                      wavenumber=False):
    """Return the expected FWHM (in nm or in cm-1) of a line given the
    observation parameters.

    :param step_nb: Number of steps from the zpd to the longest side
      of the interferogram.
    
    :param step: Step size in nm
    
    :param order: Folding order
    
    :param apod_coeff: (Optional) Apodization coefficient. 1. stands
      for no apodization and gives the FWHM of the central lobe of the
      sinc (default 1.)

    :param corr: (Optional) Coefficient of correction (default 1.)
    
    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    opd_max = step_nb * step / corr
    if not wavenumber:
        nm_axis = create_nm_axis(step_nb, step, order)
        nm_mean = (nm_axis[-1] + nm_axis[0])/2.
        return (nm_mean**2. * orb.constants.FWHM_SINC_COEFF
                / (2 * opd_max) * apod_coeff)
    else:
        return orb.constants.FWHM_SINC_COEFF / (2 * opd_max) * apod_coeff * 1e7
        
def compute_line_fwhm_pix(oversampling_ratio=1.):
    """Return the expected FWHM of an unapodized sinc line in pixels.

    :oversampling_ratio: Ratio of the real number of steps of the
      spectrum vs step_nb (must be > 1.) For a two sided interferogram
      the oversampling ratio is 2.
    """
    return orb.constants.FWHM_SINC_COEFF * oversampling_ratio
    
def compute_mean_shift(velocity, step_nb, step, order, wavenumber=False):
    """Return the mean shift at the central wavelength of the band
    defined by step and order parameters given its velocity in nm or
    in cm-1.

    :param velocity: Line velocity in km.s-1
    
    :param step_nb: Number of steps

    :param step: Step size in nm

    :param order: Folding order

    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    if not wavenumber:
        nm_axis = create_nm_axis(step_nb, step, order)
        mean = (nm_axis[-1] + nm_axis[0])/2.
    else:
        cm1_axis = create_cm1_axis(step_nb, step, order)
        mean = (cm1_axis[-1] + cm1_axis[0])/2.
        
    return line_shift(velocity, mean, wavenumber=wavenumber)
        
def compute_step_nb(resolution, step, order):
    """Return the number of steps on the longest side of the
    interferogram given the resolution and the observation
    parameters.

    :param resolution: Resolution
    
    :param step: Step size (in nm)
    
    :param order: Folding order
    """
    med_cm1 = (order + 0.5) / (2.* step) * 1e7
    return orb.constants.FWHM_SINC_COEFF * resolution / (2 * med_cm1 * step * 1e-7)

def compute_resolution(step_nb, step, order, corr=1, sigma=None):
    """Return the theoretical resolution of a given scan

    :param step_nb: Number of steps of the longest side of the
      interferogram.
    
    :param step: Step size (in nm)
    
    :param order: Folding order

    :param corr: Correction coefficient for the incident angle.

    :param sigma: Wavenumber at which the resolution is computed. If
      None given, the central wavenumber in the bandpass is used. Note
      that, when the incident angle changes, the bandpass is shifted
      so that the computed resolution is the same at any inceident
      angle if sigma is not fixed.
    """
    fwhm_cm1 = compute_line_fwhm(
        step_nb, step, order, wavenumber=True, corr=corr)
    if sigma is None:
        sigma = (order + 0.5) / (2.* step) * corr * 1e7
    return sigma / fwhm_cm1

    
def theta2corr(theta):
    """Convert the incident angle to a correction coefficient.
    
    :param theta: Incident angle in degrees
    """
    return 1./np.cos(np.deg2rad(theta))
    

def corr2theta(corr):
    """Convert the correction coefficient to an incident angle.
    
    :param corr: Correction coefficient
    """
    if np.all(corr >= 1):
        return np.rad2deg(np.arccos(1./corr))
    else:
        raise ValueError("the correction coeff must be between >= 1")


def lorentzian1d(x, h, a, dx, fwhm):
    """Return a 1D lorentzian
    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM
    """
    return h + (a / (1. + ((x-dx)/(fwhm/2.))**2.))

def gaussian1d(x,h,a,dx,fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: Array giving the positions where the gaussian is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    w = fwhm / (2. * gvar.sqrt(2. * gvar.log(2.)))
    return  h + a * gvar.exp(-(x - dx)**2. / (2. * w**2.))

def gaussian1d_complex(x,h,a,dx,fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: Array giving the positions where the gaussian is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    w = fwhm / (2. * gvar.sqrt(2. * gvar.log(2.)))
    b = (x - dx) / (np.sqrt(2) * w)
    return  (h + a * gvar.exp(-b**2.), h + a * 2 / np.sqrt(np.pi) * special.dawsn(b))

def sinc1d(x, h, a, dx, fwhm):
    """Return a 1D sinc 
    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        
    X = ((x - dx) / (fwhm / orb.constants.FWHM_SINC_COEFF))
    return h + a * orb.cgvar.sinc1d(X)

def sinc1d_complex(x, h, a, dx, fwhm):
    """The "complex" version of the sinc (understood as the Fourier
    Transform of a boxcar function from 0 to MPD).

    This is the real sinc function when ones wants to fit both the real
    part and the imaginary part of the spectrum.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    """
    width = gvar.fabs(fwhm) / orb.constants.FWHM_SINC_COEFF
    width /= np.pi
    width /= 2.###
    X = (x-dx) / (2*width)

    s1dc_re = h + a * np.sin(X) / (X)
    s1dc_re[X == 0] = h + a
    
    s1dc_im = - h + a * (np.cos(X) - 1.) / (X)
    s1dc_im[X == 0] = 0
    return (s1dc_re, s1dc_im)

def mertz1d(x, h, a, dx, fwhm, ratio):
    """Complex ILS when Mertz ramp is used during the Fourier transform.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc

    :param ratio: Ratio of the shortest side over the longest side of
      the interferogram
    """
    width = gvar.fabs(fwhm) / orb.constants.FWHM_SINC_COEFF
    width /= np.pi
    width /= 2.###
    X = (x-dx) / (2*width)

    f_real = h + a * np.sin(X) / X
    f_imag = h + a * (np.cos(X) / X - 1./((X**2)*ratio) * np.sin(ratio*X))
    return (f_real, f_imag)


def sinc1d_phased(x, h, a, dx, fwhm, alpha):
    """The phased version of the sinc function when that can be used to
    fit a spectrum with a non perfect correction of the order 0 of the
    phase.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    :param alpha: Mixing coefficient (in radians).
    """
    if np.all(np.isclose(gvar.mean(alpha), 0)):
        return sinc1d(x, h, a, dx, fwhm)
    _sinc_re, _sinc_im = sinc1d_complex(x, h, a, dx, fwhm)
    return _sinc_re * np.cos(alpha) + _sinc_im * np.sin(alpha)

def dsinc1d(x, h, a, dx, fwhm, sigma):
    """Return a 1D double sinc 

    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM
    :param sigma: broadening
    """
    return sinc1d(x, h, a/2., dx - sigma, fwhm) + sinc1d(x, h, a/2., dx + sigma, fwhm)

def sincgauss1d(x, h, a, dx, fwhm, sigma):
    """Return a 1D sinc convoluted with a gaussian of parameter sigma.

    If sigma == 0 returns a pure sinc.

    :param x: 1D array of float64 giving the positions where the
    sinc is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    :param sigma: Sigma of the gaussian.
    """
    if np.size(sigma) > 1:
        if np.any(sigma != sigma[0]):
            raise Exception('Only one value of sigma can be passed')
        else:
            sigma = sigma[0]

    sigma = np.fabs(sigma)
    fwhm = np.fabs(fwhm)

    broadening_ratio = np.fabs(sigma / fwhm)
    
    if broadening_ratio < 1e-2:
        return sinc1d(x, h, a, dx, fwhm)

    if np.isclose(gvar.mean(sigma), 0.):
        return sinc1d(x, h, a, dx, fwhm)

    width = gvar.fabs(fwhm) / orb.constants.FWHM_SINC_COEFF
    width /= np.pi ###    
    
    a_ = sigma / math.sqrt(2) / width
    b_ = (x - dx) / math.sqrt(2) / sigma

    return h + a * orb.cgvar.sincgauss1d(a_, b_)

def sincgauss1d_complex_erf(x, h, a, dx, fwhm, sigma):
    """The "complex" version of the sincgauss (erf formulation).

    This is the real sinc*gauss function when ones wants to fit both the real
    part and the imaginary part of the spectrum.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    :param sigma: Sigma of the gaussian.
    """
    width = abs(fwhm) / orb.constants.FWHM_SINC_COEFF
    width /= np.pi ###

    a_ = sigma / np.sqrt(2) / width
    b_ = ((x - dx) / np.sqrt(2) / sigma)
    if b_.dtype == np.longdouble: b_ = b_.astype(float)


    erf1 = special.erf(a_ - 1j*b_)
    erf2 = special.erf(1j*b_)
    erf3 = special.erf(a_)
    
    return np.exp(-b_**2.) * (erf1 + erf2) / (erf3)


def sincgauss1d_complex(x, h, a, dx, fwhm, sigma):
    """The "complex" version of the sincgauss (dawson definition).

    This is the real sinc*gauss function when ones wants to fit both the real
    part and the imaginary part of the spectrum.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    :param sigma: Sigma of the gaussian.
    """
    if np.size(sigma) > 1:
        if np.any(sigma != sigma[0]):
            raise Exception('Only one value of sigma can be passed')
        else:
            sigma = sigma[0]

    sigma = np.fabs(sigma)
    fwhm = np.fabs(fwhm)

    broadening_ratio = np.fabs(sigma / fwhm)
    max_broadening_ratio = gvar.mean(broadening_ratio) + gvar.sdev(broadening_ratio)

    if broadening_ratio < 1e-2:
        return sinc1d_complex(x, h, a, dx, fwhm)

    if np.isclose(gvar.mean(sigma), 0.):
        return sinc1d_complex(x, h, a, dx, fwhm)

    if max_broadening_ratio > 7:
        return gaussian1d_complex(x, h, a, dx, sigma * (2. * gvar.sqrt(2. * gvar.log(2.))))

    width = gvar.fabs(fwhm) / orb.constants.FWHM_SINC_COEFF
    width /= np.pi ###

    a_ = sigma / np.sqrt(2) / width
    b_ = ((x - dx) / np.sqrt(2) / sigma)
    if b_.dtype == np.longdouble: b_ = b_.astype(float)

    sg1c = orb.cgvar.sincgauss1d_complex(a_, b_)
    return (h + a * sg1c[0], h + a * sg1c[1])

    
def sincgauss1d_phased(x, h, a, dx, fwhm, sigma, alpha):
    """The phased version of the sinc*gauss function when that can be
    used to fit a spectrum with a non perfect correction of the order
    0 of the phase.

    :param x: 1D array of float64 giving the positions where the
      function is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param fwhm: FWHM of the sinc
    :param sigma: Sigma of the gaussian.
    :param alpha: Mixing coefficient (in radians).
    """
    if np.all(np.isclose(gvar.mean(alpha), 0)):
        return sincgauss1d(x, h, a, dx, fwhm, sigma)

    if np.all(np.isclose(gvar.mean(sigma), 0)):
        return sinc1d_phased(x, h, a, dx, fwhm, alpha)

    sc_re, sc_im = sincgauss1d_complex(x, h, a, dx, fwhm, sigma)
    return np.cos(alpha) * sc_re + np.sin(alpha) * sc_im

def gaussian1d_flux(a, fwhm):
    """Compute flux of a 1D Gaussian.

    :param a: Amplitude
    :param fwhm: FWHM
    """
    width = fwhm / orb.constants.FWHM_COEFF
    return a * gvar.fabs(math.sqrt(2*math.pi) * width)

def sinc1d_flux(a, fwhm):
    """Compute flux of a 1D sinc.

    :param a: Amplitude
    :param fwhm: FWHM
    """
    width = fwhm / orb.constants.FWHM_SINC_COEFF
    return a * gvar.fabs(width)

def sinc21d_flux(a,fwhm):
    """Compute flux of a 1D sinc2.
    THIS IS BOGUS WITH CURRENT DEFINITION OF SINC2 MODEL
    :param a: Amplitude
    :param fwhm: FWHM
    """
    width = fwhm / orb.constants.FWHM_SINC_COEFF
    return a * od.abs(fwhm)
    
def sincgauss1d_flux(a, fwhm, sigma):
    """Compute flux of a 1D sinc convoluted with a Gaussian of
    parameter sigma.

    :param a: Amplitude
    :param fwhm: FWHM of the sinc
    :param sigma: Sigma of the gaussian
    :param no_err: (Optional) No error is returned (default False)
    """
    if np.all(np.isnan(gvar.mean(sigma))) or np.all(np.isclose(gvar.mean(sigma), 0)):
        return sinc1d_flux(a, fwhm)
    
    width = fwhm / orb.constants.FWHM_SINC_COEFF
    width /= math.pi

    def compute_flux(ia, isig, iwid):
        idia = orb.cgvar.dawsni(isig / (math.sqrt(2) * iwid))
        expa2 = gvar.exp(isig**2./2./iwid**2.)
        if not np.isclose(gvar.mean(idia),0):
            return ia * math.pi / math.sqrt(2.) * isig * expa2 / idia
        else: return gvar.gvar(np.inf, np.inf)


    try:
        _A = sigma / (np.sqrt(2) * width)
        dia = special.dawsn(1j*_A)
        expa2 = np.exp(_A**2.)
    
        return (a * np.pi / np.sqrt(2.) * 1j * sigma * expa2 / dia).real
    except TypeError: pass

    if isinstance(a, np.ndarray):
        result = np.empty_like(a)
        for i in range(np.size(a)):
            result.flat[i] = compute_flux(
                a.flat[i], sigma.flat[i], width.flat[i])
    else: result = compute_flux(a, sigma, width)
    return result

def fast_w2pix(w, axis_min, axis_step):
    """Fast conversion of wavelength/wavenumber to pixel

    :param w: wavelength/wavenumber
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    w_ = (w - axis_min)
    if np.any(gvar.sdev(w_) != 0.):
        w_ = gvar.gvar(gvar.mean(w_), gvar.sdev(w_))
    return gvar.fabs(w_) / axis_step
        

def fast_pix2w(pix, axis_min, axis_step):
    """Fast conversion of pixel to wavelength/wavenumber

    :param pix: position along axis in pixels
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    return pix * axis_step + axis_min

def thermal_broadening_kms(wl, aw, T):
    """
    Return the width of the line due to thermal broadening in km/s.

    Equation can be refered to Harwit (Astrophysical concepts) but his
    definition gives the HWHM (Half-Width at Half-Maximum).
    
    :param wl: Wavelength of the line (in nm)
    :param aw: Atomic weight of the emitting atom
    :param T: Temperature in K
    """
    E = aw * orb.constants.ATOMIC_MASS * (orb.constants.LIGHT_VEL_KMS * 1e5) **2.
    width = wl * np.sqrt(orb.constants.K_BOLTZMANN * T / E) # nm
    return orb.constants.LIGHT_VEL_KMS *  width / wl # kms

def phase_shift_cm1_axis(step_nb, step, order, nm_laser_obs, nm_laser):
    """Compute phase shift on a given cm1 axis

    :param step_nb: Number of steps
    :param step: Step size in nm
    :param order: Folding order
    :param nm_laser_obs: Observed calibration laser wavelength (in nm)
    :param nm_laser: Calibration laser wavelength (in nm)
    """
    corr = nm_laser_obs / nm_laser
    cm1_min_corr = orb.cutils.get_cm1_axis_min(int(step_nb), float(step),
                                               int(order), corr=float(corr))
    cm1_min_base = orb.cutils.get_cm1_axis_min(int(step_nb), float(step),
                                               int(order))
    cm1_axis_step =  orb.cutils.get_cm1_axis_step(int(step_nb),
                                                  float(step),
                                                  corr=float(corr))
    delta_cm1 = cm1_min_corr - cm1_min_base
    delta_x = - (delta_cm1 / cm1_axis_step)
    return delta_x

def guess_snr(calib_spectrum, flambda, exp_time):
    """Guess calibrated spectrum snr

    :param calib_spectrum: Calibrated spectrum

    :param flambda: Calibration FLAMBDA

    :param exp_time: Exposure time by step
    """
    int_time = calib_spectrum.shape[0] * exp_time # total integration time in s
    spec_counts = calib_spectrum / flambda * int_time
    noise = np.sqrt(np.nansum(np.sqrt(spec_counts**2)))
    signal = np.nanmax(spec_counts) - np.nanmedian(spec_counts)
    return signal / noise

def amp_ratio_from_flux_ratio(line0_cm1, line1_cm1, flux_ratio):
    """Return the amplitude ratio (amp(line0) / amp(line1)) from the flux
        ratio (at constant fwhm and broadening).

    :param line0: Wavenumber of the line 0 (in cm-1).

    :param line1: Wavenumber of the line 1 (in cm-1).

    :param flux_ratio: Flux ratio: flux(line0) / flux(line1).

    """
    return line0_cm1**2 / line1_cm1**2 * flux_ratio

def detect_velocity(spec, axis_cm1, lines_cm1, ncomps=1, vrange=1000):
    """Detect velocity of different components in a spectrum.

    :param ncomps: maximum velocity components
    """

    FWHM = 2
    THRESHOLD = 2
    FINE_FACTOR = 3
    
    if spec.size != axis_cm1.size:
        raise TypeError('spec and axis must have same size')

    spec = np.copy(spec.real)
    spec /= np.nanstd(spec[spec < np.nanpercentile(spec, 90)])
    lines_cm1 = np.array(lines_cm1)
    
    axis_cm1_diff = np.diff(axis_cm1)
    if not np.all(np.isclose(axis_cm1_diff - axis_cm1_diff[0], 0)):
        raise TypeError('axis must be equally sampled')

    lines_pix = fast_w2pix(lines_cm1, axis_cm1[0], axis_cm1_diff[0])

    channel_step_cm1 = axis_cm1_diff[0]
    channel_step_kms = np.abs(
        compute_radial_velocity(np.mean(lines_cm1) + channel_step_cm1,
                                np.mean(lines_cm1), wavenumber=True))
    #pix_range = vrange/channel_step_kms
    #kernel_hr = np.linspace(-pix_range, vrange/channel_step_kms, )
    x_hr = np.linspace(0, axis_cm1.size - 1, axis_cm1.size * 10)

    kernel_hr = np.zeros_like(x_hr)
    total_flux = 0
    for iline in lines_pix:
        kernel_hr += gaussian1d(x_hr, 0, 1, iline, FWHM)
        total_flux += gaussian1d_flux(1, FWHM)
    kernel_hr /= total_flux
    kernel_hr_f = interpolate.interp1d(x_hr, kernel_hr, bounds_error=False)

    vels = np.linspace(-vrange, vrange, 2*vrange/channel_step_kms)
    x = np.arange(spec.size)

    large_conv = list()
    for ivel in vels:
        iker = kernel_hr_f(x + ivel / channel_step_kms)
        large_conv.append(np.nansum(iker * spec))
    large_conv = np.array(large_conv)

    comps = list()
    
    while True:
        _argmax = np.nanargmax(large_conv)
        _max = large_conv[_argmax]
        if _max < THRESHOLD:
            break
        if len(comps) >= ncomps:
            break
        comps.append(vels[_argmax])
        large_conv -= gaussian1d(np.arange(large_conv.size), 0, _max, _argmax, FWHM)

    vels = np.linspace(-FINE_FACTOR * channel_step_kms,
                       FINE_FACTOR * channel_step_kms, 4*FINE_FACTOR)

    fine_comps = list()
    for icomp in comps:
        fine_conv = list()
        for ivel in vels:
            iker = kernel_hr_f(x + (ivel + icomp)/channel_step_kms)
            fine_conv.append(np.nansum(iker * spec))
        fine_comps.append((vels + icomp)[np.nanargmax(fine_conv)])

    return fine_comps


def project(data, old_axis, new_axis, quality):
    """Project complex data on a new axis. 

    Note: handle not complex data for backward compatibility, but the
    projection is a simple linear interpolation.

    :param data: data to project

    :param old_axis: data original axis

    :param new_axis: new axis (must be contained into the old axis -
      no extrapolation)

    :param quality: an integer from 2 to infinity which gives the
      zero padding factor before interpolation. The more zero
      padding, the better will be the interpolation, but the
      slower too.

    :return: data projected onto the new axis

    .. warning:: Though much (much!) faster than pure resampling, this
      can be a little less precise for complex data. Errors should be
      negligible With a high enough quality factor. For non complex
      data, its nothing more than a linear interpolation.
    """
    if np.any(np.iscomplex(data)):
        quality = int(quality)
        if quality < 2: raise ValueError('quality must be an integer > 2')
        interf_complex = fftpack.ifft(data)
        best_n = orb.utils.fft.next_power_of_two(len(old_axis) * quality)
        zp_interf = np.zeros(best_n, dtype=complex)
        center = interf_complex.shape[0] // 2
        zp_interf[:center] = interf_complex[:center]
        zp_interf[
            -center-int(interf_complex.shape[0]&1):] = interf_complex[
            -center-int(interf_complex.shape[0]&1):]

        zp_spec = fftpack.fft(zp_interf)
        ax_ratio = float(old_axis.size) / float(zp_spec.size)
        zp_axis = (np.arange(zp_spec.size)
                   * (old_axis[1] - old_axis[0]) * ax_ratio
                   + old_axis[0])
        f = interpolate.interp1d(zp_axis,
                                       zp_spec,
                                       bounds_error=False)

    else:
        f = interpolate.interp1d(old_axis.astype(np.longdouble),
                                       data.real.astype(np.longdouble),
                                       bounds_error=False)

    return f(new_axis.data)
