#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: spectrum.py

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
from scipy import interpolate, optimize
import warnings
import time

import orb.constants
import orb.cutils


def create_nm_axis(n, step, order, corr=1.):
    """Create a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order (cannot be 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    
    nm_min = orb.cutils.get_nm_axis_min(step, order, corr=corr)
    if (order > 0): 
        nm_max = orb.cutils.get_nm_axis_max(step, order, corr=corr)
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
    cm1_min = orb.cutils.get_cm1_axis_min(step, order, corr=corr)
    cm1_max = orb.cutils.get_cm1_axis_max(step, order, corr=corr)
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
       orb.cutils.fast_pix2w is much faster.

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
       orb.cutils.fast_w2pix is much faster.

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
    return 1e7 / np.array(nm).astype(float)

def cm12nm(cm1):
    """Convert a wavenumber in cm-1 to a wavelength in nm.

    :param cm1: wavenumber in cm-1
    """
    return 1e7 / np.array(cm1).astype(float)

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

def line_shift(velocity, line, wavenumber=False):
    """Return the line shift given its velocity in nm or in cm-1.

    :param velocity: Line velocity in km.s-1

    :param line: Wavelength/wavenumber of the line. Must be in cm-1 if
      wavenumber is True, must be in nm otherwise.

    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    vel = (np.array(line, dtype=np.longdouble)
           * np.array(velocity, dtype=np.longdouble)
           / float(orb.constants.LIGHT_VEL_KMS))
    if wavenumber: return -vel
    else: return vel
    
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
    opd_max = step_nb * step * 2. / corr
    if not wavenumber:
        nm_axis = create_nm_axis(step_nb, step, order)
        nm_mean = (nm_axis[-1] + nm_axis[0])/2.
        return (nm_mean**2. * orb.constants.FWHM_SINC_COEFF
                / opd_max * apod_coeff)
    else:
        return orb.constants.FWHM_SINC_COEFF / opd_max * apod_coeff * 1e7
        
def compute_line_fwhm_pix(step_nb):
    """Return the expected FWHM of an unapodized sinc line in pixels.
    
    :param step_nb: Number of steps from the zpd to the longest side
      of the interferogram.
    """
    return (2. * ((float(step_nb) - 1.) / float(step_nb))
            * orb.constants.FWHM_SINC_COEFF)
    
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
        

def compute_radial_velocity(line, rest_line, wavenumber=False):
    """
    Return radial velocity in km.s-1

    V [km.s-1] = c [km.s-1]* (Lambda - Lambda_0) / Lambda_0

    :param line: Emission line wavelength/wavenumber (can be a numpy
      array)
    
    :param rest_line: Rest-frame wavelength/wavenumber (can be a numpy
      array but must have the same size as line)

    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    line = np.array(line, dtype=np.longdouble)
    rest_line = np.array(rest_line, dtype=np.longdouble)
    return orb.cutils.compute_radial_velocity(
        np.array(line, dtype=np.longdouble),
        np.array(rest_line, dtype=np.longdouble),
        wavenumber=wavenumber)

def flambda2ABmag(flambda, lam):
    """Return AB magnitude from flux in erg/cm2/s/A

    :param flambda: Flux in erg/cm2/s/A. Can be an array.

    :param lambda: Wavelength in A of the Flux. If flambda is an array
      lambda must have the same shape.
    """
    c = 2.99792458e18 # Ang/s
    fnu = lam**2./c*flambda
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
    c = 2.99792458e18 # Ang/s
    return fnu * nu**2. / c

def lambda2nu(lam):
    """Convert lambda in Ang to nu in Hz

    :param lam: Wavelength in angstrom
    """
    c = 2.99792458e18 # Ang/s
    return c / lam

def ABmag2flambda(ABmag, lam):
    """Convert AB magnitude to flux in erg/cm2/s/A

    :param ABmag: A magnitude in the AB magnitude system

    :param lam: Wavelength in angstrom
    """
    return fnu2flambda(ABmag2fnu(ABmag), lambda2nu(lam))

def lorentzian1d(x, h, a, dx, fwhm):
    """Return a 1D lorentzian
    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM
    """
    return h + (a / (1. + ((x-dx)/(fwhm/2.))**2.))

def sinc1d(x, h, a, dx, fwhm):
    """Return a 1D sinc 
    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != float:
        x = x.astype(float)
    return orb.cutils.sinc1d(
        x, float(h), float(a), float(dx), float(fwhm))

def gaussian1d(x,h,a,dx,fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: Array giving the positions where the gaussian is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != float:
        x = x.astype(float)
    return orb.cutils.gaussian1d(
        x, float(h), float(a), float(dx), float(fwhm))
                        

