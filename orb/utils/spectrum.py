#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: spectrum.py

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
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

import orb.constants
import orb.cutils
import orb.utils.fft
import orb.utils.stats
import orb.utils.vector

def create_nm_axis(n, step, order, corr=1.):
    """Create a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order (cannot be 0)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    
    nm_min = 2. * float(step) / float(order + 1.) / float(corr)
    if (order > 0): 
        nm_max = 2. * float(step) / float(order) / float(corr)
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
    cm1_min = float(order) / (2.* float(step)) * float(corr) * 1e7
    cm1_max = float(order + 1.) / (2. * float(step)) * float(corr) * 1e7
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

     :param nm_axis: Axis in nm
     
     :param pix: Pixel position
     """  
     f = interpolate.interp1d(np.arange(nm_axis.shape[0]), nm_axis,
                              bounds_error=False, fill_value=np.nan)
     return f(pix)
   

def nm2pix(nm_axis, nm):
     """Convert a wavelength in nm to a pixel position given an axis
     in nm

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

    :param nm: wavelength i nm
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
    
def compute_line_fwhm(step_nb, step, order, apod_coeff=1., wavenumber=False):
    """Return the expected FWHM (in nm or in cm-1) of a line given the
    observation parameters.

    :param step_nb: Number of steps from the zpd to the longest side
      of the interferogram.
    
    :param step: Step size in nm
    
    :param order: Folding order
    
    :param apod_coeff: (Optional) Apodization coefficient. 1. stands
      for no apodization and gives the FWHM of the central lobe of the
      sinc (default 1.)
    
    :param wavenumber: (Optional) If True the result is returned in cm-1,
      else it is returned in nm.
    """
    nm_axis = create_nm_axis(step_nb, step, order)
    nm_mean = (nm_axis[-1] + nm_axis[0])/2.
    opd_max = step_nb * step * 2.
    if not wavenumber:
        return nm_mean**2. * 1.20671 / opd_max * apod_coeff
    else:
        return 1.20671 / opd_max * apod_coeff * 1e7
    

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
    if wavenumber: delta = rest_line - line
    else: delta = line - rest_line
    return float(orb.constants.LIGHT_VEL_KMS) * (delta / rest_line) 
    

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
    

def robust_fit_sinc_lines_in_spectrum(spectrum, lines, step, order, nm_laser,
                                      nm_laser_obs, **kwargs):
    """Robust fit of multiple sinc shaped emission lines in a spectrum vector.

    Spectrum is apodized and fitted with gaussian shaped emission
    lines and the fitted params are used as initial guesses for the
    final fit on the original sinc spectrum.

    This version is by far more robust than a crude sinc fit if the
    line position and fwhm is not precisely known.
    """
    FWHM_COEFF = 1.20671
    kwargs['fmodel'] = 'gaussian'
    
    if 'fwhm_guess' in kwargs:
        kwargs['fwhm_guess'] = kwargs['fwhm_guess'] * FWHM_COEFF

    # apodization
    if 'signal_range' in kwargs:
        if kwargs['signal_range'] is not None:
            signal_range = kwargs['signal_range']
            correction_coeff = float(nm_laser_obs) / nm_laser

            if kwargs['wavenumber']:
                axis = create_cm1_axis(spectrum.shape[0], step, order,
                                       corr=correction_coeff)
                signal_range_pix = cm12pix(axis, signal_range)
            else:
                axis = create_nm_axis(spectrum.shape[0], step, order,
                                      corr=correction_coeff)
                signal_range_pix = nm2pix(axis, signal_range)

            minx = int(np.min(signal_range_pix))
            maxx = int(math.ceil(np.max(signal_range_pix)))
            apod_spectrum_temp = orb.utils.fft.apodize(spectrum[minx:maxx])
            apod_spectrum = np.zeros_like(spectrum, dtype=float)
            apod_spectrum[minx:maxx] = apod_spectrum_temp
        else:
            apod_spectrum = orb.utils.fft.apodize(spectrum)
    else:
        apod_spectrum = orb.utils.fft.apodize(spectrum)
        
    # fit on apodized data
    gaussian_fit = fit_lines_in_spectrum(
        apod_spectrum, lines, step, order, nm_laser,
        nm_laser_obs, **kwargs)
    
    # fit on sinc data
    if gaussian_fit != []:
        robust_vel = np.nanmean(gaussian_fit['velocity'])
        robust_cont = gaussian_fit['cont-params']
        robust_fwhm = np.nanmean(gaussian_fit['fwhm-wave']) / FWHM_COEFF
        if not np.any(np.isnan(robust_vel)):
            kwargs['shift_guess'] = robust_vel
            kwargs['fwhm_guess'] = robust_fwhm
            kwargs['fmodel'] = 'sinc'
            kwargs['cont_guess'] = robust_cont
            
            sinc_fit = fit_lines_in_spectrum(
                spectrum, lines, step, order, nm_laser,
                nm_laser_obs, **kwargs)

            if sinc_fit != []:
                # only the velocity parameter, best fitted with a sinc, is
                # kept. amp, fwhm and continnuum seems more robustly
                # determined on an apodized spectrum.
                lines_params = np.copy(gaussian_fit['lines-params'])
                lines_params[:,2] = sinc_fit['lines-params'][:,2]
                gaussian_fit['lines-params'] = lines_params
            
                if 'lines-params-err' in gaussian_fit:
                    lines_params_err = np.copy(gaussian_fit['lines-params-err'])
                    if 'lines-params-err' in sinc_fit:
                        lines_params_err[:,2] = sinc_fit[
                            'lines-params-err'][:,2]
                    else:
                        lines_params_err[:,2] = np.nan
                    gaussian_fit['lines-params-err'] = lines_params_err
            
                gaussian_fit['velocity'] = sinc_fit['velocity']
                if 'velocity-err' in sinc_fit:
                    gaussian_fit['velocity-err'] = sinc_fit['velocity-err']

                if 'fitted-vector' in sinc_fit:
                    gaussian_fit['fitted-vector'] = sinc_fit['fitted-vector']

            return gaussian_fit
        else: return []
    else:
        return gaussian_fit


def fit_lines_in_spectrum(spectrum, lines, step, order, nm_laser,
                          nm_laser_obs, wavenumber=True, **kwargs):

    def fast_w2pix(w, axis_min, axis_step):
        return abs(w - axis_min) / axis_step

    def fast_pix2w(pix, axis_min, axis_step):
        return pix * axis_step + axis_min
    
    def shift(lines_pos, vel, p):
        axis_min, axis_step, lines, wavenumber = p
        delta = line_shift(vel, lines, wavenumber=wavenumber)
        #print vel, fast_w2pix(lines + delta, axis_min, axis_step)
        return fast_w2pix(lines + delta, axis_min, axis_step) - fast_w2pix(lines, axis_min, axis_step) + lines_pos


    correction_coeff = float(nm_laser_obs) / nm_laser

    if wavenumber:
        axis = create_cm1_axis(spectrum.shape[0], step, order,
                               corr=correction_coeff)
  
    else:
        axis = create_nm_axis(spectrum.shape[0], step, order,
                              corr=correction_coeff)
        
    axis_min = axis[0]
    axis_step = axis[1] - axis[0]

    lines_pix = fast_w2pix(lines, axis_min, axis_step)

    if 'fwhm_guess' in kwargs:
        kwargs['fwhm_guess'] = kwargs['fwhm_guess'] / axis_step

    if 'signal_range' in kwargs:
        if kwargs['signal_range'] is not None:
            signal_range_pix = fast_w2pix(kwargs['signal_range'], axis_min, axis_step)
            minx = int(np.min(signal_range_pix))
            maxx = int(math.ceil(np.max(signal_range_pix)))
            kwargs['signal_range'] = [minx, maxx]
            
    fadd_shift_p = (axis_min, axis_step ,lines, wavenumber)

    ## import pylab as pl
    ## pl.plot(axis, spectrum)
    ## for iline in lines:
    ##     pl.axvline(x=iline)
    ## pl.show()
    ## quit()
    fit = fit_lines_in_vector(spectrum, lines_pix,
                              fadd_shift=(shift, fadd_shift_p),
                              **kwargs)
    if fit != []:
        fit['velocity'] = compute_radial_velocity(
            fast_pix2w(
                fit['lines-params'][:,2],
                axis_min, axis_step), lines, wavenumber=wavenumber)
        fit['fwhm-wave'] = fit['lines-params'][:,3] * axis_step
        if 'lines-params-err' in fit:
            fit['velocity-err'] = np.abs(compute_radial_velocity(
                fast_pix2w(
                    lines_pix + fit['lines-params-err'][:,2],
                    axis_min, axis_step), lines, wavenumber=wavenumber))

    return fit
    


def fit_lines_in_vector(vector, lines, fwhm_guess=3.5,
                        cont_guess=None, shift_guess=0.,
                        fix_cont=False,
                        fix_fwhm=False, cov_fwhm=True, cov_pos=True,
                        reguess_positions=False,
                        return_fitted_vector=False, fit_tol=1e-5,
                        no_absorption=False, poly_order=0,
                        fmodel='gaussian', sig_noise=None,
                        signal_range=None,
                        fadd_shift=None):

    """Fit multiple emission lines in a spectrum vector.

    :param vector: Vector to fit

    :param lines: Positions of the lines in channels

    :param fwhm_guess: (Optional) Initial guess on the lines FWHM
      (default 3.5).

    :param cont_guess: (Optional) Initial guess on the continuum
      (default None). Must be a tuple of poly_order + 1 values ordered
      with the highest orders first.

    :param shift_guess: (Optional) Initial guess on the global shift
      of the lines (default 0.).

    :param fix_cont: (Optional) If True, continuum is fixed to the
      initial guess (default False).

    :param fix_fwhm: (Optional) If True, FWHM value is fixed to the
      initial guess (default False).

    :param cov_fwhm: (Optional) If True FWHM is considered to be the
      same for all lines and become a covarying parameter (default
      True).

    :param cov_pos: (Optional) If True the estimated relative
      positions of the lines (the lines parameter) are considered to
      be exact and only need to be shifted. Positions are thus
      covarying. Very useful but the initial estimation of the line
      relative positions must be very precise. This parameter can also
      be a tuple of the same length as the number of lines to
      distinguish the covarying lines. Covarying lines must share the
      same number. e.g. on 4 lines, [NII]6548, Halpha, [NII]6584,
      [SII]6717, [SII]6731, if each ion has a different velocity
      cov_pos can be : [0,1,0,2,2]. (default False).

    :param reguess_positions: (Optional) If True, positions are
      guessed again. Useful if the given estimations really are rough
      ones. Note that this must not be used with cov_pos set to True
      (default False).

    :param return_fitted_vector: (Optional) If True Fitted vector is
      returned.

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-5).

    :param no_absorption: (Optional) If True, no negative amplitude
      are returned (default False).

    :param poly_order: (Optional) Order of the polynomial used to fit
      continuum. Use high orders carefully (default 0).

    :param fmodel: (Optional) Fitting model. Can be 'gaussian', 
      'sinc', 'sinc2' or 'lorentzian' (default 'gaussian').

    :param sig_noise: (Optional) Noise standard deviation guess. If
      None noise value is guessed but the gaussian FWHM must not
      exceed half of the sampling interval (default None).

    :param signal_range: (Optional) A tuple (x_min, x_max) giving the
      lowest and highest channel numbers containing signal.

    :return: a dictionary containing:
    
      * lines parameters [key: 'lines-params'] Lines parameters are
        given as an array of shape (lines_nb, 4). The order of the 4
        parameters for each lines is [height at the center of the
        line, amplitude, position, fwhm].
      
      * lines parameters errors [key: 'lines-params-err']

      * residual [key: 'residual']
      
      * chi-square [key: 'chi-square']

      * reduced chi-square [key: 'reduced-chi-square']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont-params']

      * and optionally the fitted vector [key: 'fitted-vector']
        depending on the option return_fitted_vector.

      
    """
    def params_arrays2vect(lines_p, lines_p_mask,
                           cov_p, cov_p_mask,
                           cont_p, cont_p_mask):
        
        free_p = list(lines_p[np.nonzero(lines_p_mask)])
        free_p += list(cov_p[np.nonzero(cov_p_mask)])
        free_p += list(cont_p[np.nonzero(cont_p_mask)])
        
        fixed_p = list(lines_p[np.nonzero(~lines_p_mask)])
        fixed_p += list(cov_p[np.nonzero(~cov_p_mask)])
        fixed_p += list(cont_p[np.nonzero(~cont_p_mask)])
        
        return free_p, fixed_p

    def params_vect2arrays(free_p, fixed_p, lines_p_mask,
                           cov_p_mask, cont_p_mask):

        cont_p = np.empty_like(cont_p_mask, dtype=float)
        free_cont_p_nb = np.sum(cont_p_mask)
        if free_cont_p_nb > 0:
            cont_p[np.nonzero(cont_p_mask)] = free_p[
                -free_cont_p_nb:]
            free_p = free_p[:-free_cont_p_nb]
        if free_cont_p_nb < np.size(cont_p_mask):
            cont_p[np.nonzero(~cont_p_mask)] = fixed_p[
                -(np.size(cont_p_mask) - free_cont_p_nb):]
            fixed_p = fixed_p[
                :-(np.size(cont_p_mask) - free_cont_p_nb)]
            
        cov_p = np.empty_like(cov_p_mask, dtype=float)
        free_cov_p_nb = np.sum(cov_p_mask)
        if free_cov_p_nb > 0:
            cov_p[np.nonzero(cov_p_mask)] = free_p[
                -free_cov_p_nb:]
            free_p = free_p[:-free_cov_p_nb]
        if free_cov_p_nb < np.size(cov_p_mask):
            
            cov_p[np.nonzero(~cov_p_mask)] = fixed_p[
                -(np.size(cov_p_mask) - free_cov_p_nb):]
            fixed_p = fixed_p[
                :-(np.size(cov_p_mask) - free_cov_p_nb)]


        lines_p = np.empty_like(lines_p_mask, dtype=float)
        lines_p[np.nonzero(lines_p_mask)] = free_p
        lines_p[np.nonzero(~lines_p_mask)] = fixed_p
        
        return lines_p, cov_p, cont_p
    
    def model(n, lines_p, cont_p, fmodel, pix_axis):
        mod = np.zeros(n, dtype=float)
        # continuum
        mod += np.polyval(cont_p, np.arange(n))
        for iline in range(lines_p.shape[0]):
            
            if fmodel == 'sinc':
                mod += sinc1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
                
            elif fmodel == 'lorentzian':
                mod += lorentzian1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
                
            elif fmodel == 'sinc2':
                mod += np.sqrt(sinc1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])**2.)
                
            else:
                mod += gaussian1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
        return mod

    def add_shift(lines_pos, shift, p):
        return lines_pos + shift

    def diff(free_p, fixed_p, lines_p_mask, cov_p_mask,
             cont_p_mask, data, sig, fmodel, pix_axis,
             cov_pos_mask_list, iter_list, fadd_shift, fadd_shiftp):
        lines_p, cov_p, cont_p = params_vect2arrays(free_p, fixed_p,
                                                    lines_p_mask,
                                                    cov_p_mask, cont_p_mask)
        # + FWHM
        lines_p[:,2] += cov_p[0]
        
        # + SHIFT
        if cov_p_mask[1]:
            for i in range(len(cov_pos_mask_list)):
                ilines = np.arange(lines_p.shape[0])[
                    np.nonzero(cov_pos_mask_list[i])]
                ifadd_shiftp = list(fadd_shiftp)
                ifadd_shiftp[2] = fadd_shiftp[2][ilines]
                lines_p[ilines,1] = fadd_shift(
                    lines_p[ilines,1], cov_p[1+i], ifadd_shiftp)
  
        data_mod = model(np.size(data), lines_p, cont_p, fmodel, pix_axis)
        ## import pylab as pl
        ## pl.plot(data)
        ## for i in range(lines_p.shape[0]):
        ##     pl.axvline(x=lines_p[i,1])
        ## pl.plot(data_mod)
        ## pl.show()
        ## quit()
        res = (data - data_mod) / sig
        iter_list.append((res, free_p))
        return res

    MAXFEV = 5000 # Maximum number of evaluation

    MIN_LINE_SIZE = 5 # Minimum size of a line whatever the guessed
                      # fwhm can be.

    if fadd_shift is None:
        fadd_shiftp = None
        fadd_shift = add_shift
    else:
        fadd_shiftp = fadd_shift[1]
        fadd_shift = fadd_shift[0]

    x = np.copy(vector)

    if np.all(np.isnan(x)):
        return []
    
    if np.any(np.iscomplex(x)):
        x = x.real
        warnings.warn(
            'Complex vector. Only the real part will be fitted')

    lines = np.array(lines, dtype=float)
    lines = lines[np.nonzero(lines > 0.)]

    lines_nb = np.size(lines)
    line_size = int(max(math.ceil(fwhm_guess * 3.5), MIN_LINE_SIZE))
    
    # only 3 params max for each line, cont is defined as an N order
    # polynomial (N+1 additional parameters)
    lines_p = np.zeros((lines_nb, 3), dtype=float)
        
    # remove parts out of the signal range
    x_size_orig = np.size(x)
    if signal_range is not None:
        signal_range = np.array(signal_range).astype(int)
        if np.min(signal_range) >= 0 and np.max(signal_range < np.size(x)):
            x = x[np.min(signal_range):np.max(signal_range)]
            lines -= np.min(signal_range)
        else:
            raise Exception('Signal range must be a tuple (min, max) with min >= 0 and max < {:d}'.format(np.size(x)))
    # axis
    pix_axis = np.arange(x.shape[0])

    # check nans
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or (np.min(x) == np.max(x)):
        return []
    
    ## Guess ##
    noise_vector = np.copy(x)
    for iline in range(lines_nb):
        line_center = lines[iline]
        iz_min = int(line_center - line_size/2.)
        iz_max = int(line_center + line_size/2.) + 1
        if iz_min < 0: iz_min = 0
        if iz_max > x.shape[0] - 1: iz_max = x.shape[0] - 1
        line_box = x[iz_min:iz_max]

        # max
        lines_p[iline, 0] = np.max(line_box) - np.median(x)

        # position
        if reguess_positions:
            lines_p[iline, 1] = (np.sum(line_box
                                        * np.arange(line_box.shape[0]))
                                 / np.sum(line_box)) + iz_min
        
        # remove line from noise vector
        noise_vector[iz_min:iz_max] = np.nan

    
    if not reguess_positions:
        lines_p[:, 1] = np.array(lines)

    lines_p[:, 2] = fwhm_guess
        
    # polynomial guess of the continuum
    if (cont_guess is None) or (np.size(cont_guess) != poly_order + 1):
        if (cont_guess is not None) and np.size(cont_guess) != poly_order + 1:
            warnings.warn('Bad continuum guess shape')
        if poly_order > 0:
            w = np.ones_like(noise_vector)
            nans = np.nonzero(np.isnan(noise_vector))
            w[nans] = 0.
            noise_vector[nans] = 0.
            cont_guess = np.polynomial.polynomial.polyfit(
                np.arange(noise_vector.shape[0]), noise_vector, poly_order, w=w)
            cont_fit = np.polynomial.polynomial.polyval(
                np.arange(noise_vector.shape[0]), cont_guess)
            cont_guess = cont_guess[::-1]
            noise_value = orb.utils.stats.robust_std(noise_vector - cont_fit)
        else:
            cont_guess = [orb.utils.stats.robust_median(noise_vector)]
            noise_value = orb.utils.stats.robust_std(noise_vector)
    else:
        cont_fit = np.polyval(cont_guess, np.arange(noise_vector.shape[0]))
        noise_value = orb.utils.stats.robust_std(noise_vector - cont_fit)
    
    if sig_noise is not None:
        noise_value = sig_noise

    #### PARAMETERS = LINES_PARAMS (N*(3-COV_PARAMS_NB)),
    ####              COV_PARAMS (FWHM_COEFF, SHIFT),
    ####              CONTINUUM COEFFS

    if np.size(cov_pos) > 1:
        if np.size(cov_pos) != lines_nb:
            raise Exception('If cov_pos is not True or False it must be a tuple of the same length as the lines number')
        cov_pos_list = list()
        cov_pos_mask_list = list()
        for i in cov_pos:
            if i not in cov_pos_list:
                cov_pos_list.append(i)
                cov_pos_mask_list.append(np.array(cov_pos) == i)
    else:
        cov_pos_list = [0]
        cov_pos_mask_list = [np.ones(lines_nb, dtype=bool)]

    # define cov params
    cov_p = np.empty(1 + np.size(cov_pos_list), dtype=float)
    cov_p[0] = 0. # COV FWHM COEFF
    for i in range(len(cov_pos_list)):
        cov_p[1+i] = shift_guess # COV SHIFT
    cov_p_mask = np.zeros_like(cov_p, dtype=bool)
    if cov_fwhm and not fix_fwhm: cov_p_mask[0] = True
    
    if np.size(cov_pos) == 1:
        if cov_pos: cov_p_mask[1:] = True
    else:
        cov_p_mask[1:] = True   
        
    # define lines params mask
    lines_p_mask = np.ones_like(lines_p, dtype=bool)
    if fix_fwhm or cov_fwhm: lines_p_mask[:,2] = False
    if np.size(cov_pos) == 1:
        if cov_pos: lines_p_mask[:,1] = False
    else:
        lines_p_mask[:,1] = False
    
    # define continuum params
    cont_p = np.array(cont_guess)
    cont_p_mask = np.ones_like(cont_p, dtype=bool)
    if fix_cont: cont_p_mask.fill(False)

    free_p, fixed_p = params_arrays2vect(
        lines_p, lines_p_mask,
        cov_p, cov_p_mask,
        cont_p, cont_p_mask)

    
    ### FIT ###
    iter_list = list()
    fit = optimize.leastsq(diff, free_p,
                           args=(fixed_p, lines_p_mask,
                                 cov_p_mask, cont_p_mask,
                                 x, noise_value, fmodel, pix_axis,
                                 cov_pos_mask_list, iter_list, fadd_shift, fadd_shiftp),
                           maxfev=MAXFEV, full_output=True,
                           xtol=fit_tol)
    
    ### CHECK FIT RESULTS ###
    if fit[-1] <= 4:
        if fit[2]['nfev'] >= MAXFEV: return [] # reject evaluation bounded fit
        returned_data = dict()
        last_diff = fit[2]['fvec']
        lines_p, cov_p, cont_p = params_vect2arrays(
            fit[0], fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)

            
        # add cov to lines params
        full_lines_p = np.empty((lines_nb, 4), dtype=float)
        full_lines_p[:,0] = np.polyval(cont_p, lines_p[:,1])
        full_lines_p[:,1:] = lines_p
        cov_velocities = np.zeros(lines_nb, dtype=float)
        if cov_p_mask[1]:
            for i in range(len(cov_pos_mask_list)):
                ilines = np.arange(lines_p.shape[0])[
                    np.nonzero(cov_pos_mask_list[i])]
                ifadd_shiftp = list(fadd_shiftp)
                ifadd_shiftp[2] = fadd_shiftp[2][ilines]
                full_lines_p[ilines,2] = fadd_shift(full_lines_p[ilines,2],
                                                    cov_p[1+i], ifadd_shiftp)
                cov_velocities[ilines] = cov_p[1+i]

        full_lines_p[:,3] += cov_p[0] # + FWHM_COEFF
        # check and correct
        for iline in range(lines_nb):
            if no_absorption:
                if full_lines_p[iline, 1] < 0.:
                    full_lines_p[iline, 1] = 0.
            if (full_lines_p[iline, 2] < 0.
                or full_lines_p[iline, 2] > x.shape[0]):
                full_lines_p[iline, :] = np.nan

        # compute fitted vector
        fitted_vector = np.empty(x_size_orig, dtype=float)
        fitted_vector.fill(np.nan)
        fitted_vector[np.min(signal_range):
                      np.max(signal_range)] = model(
            x.shape[0], full_lines_p[:,1:], cont_p, fmodel, pix_axis)
        if return_fitted_vector:
            returned_data['fitted-vector'] = fitted_vector
       
        # correct shift for signal range
        if signal_range is not None:            
            full_lines_p[:,2] += np.min(signal_range)
        
        returned_data['lines-params'] = full_lines_p
        returned_data['cont-params'] = cont_p
        returned_data['cov-velocity-params'] = cov_velocities
       
        # compute reduced chi square
        chisq = np.sum(last_diff**2.)
        red_chisq = chisq / (np.size(x) - np.size(free_p))
        returned_data['reduced-chi-square'] = red_chisq
        returned_data['chi-square'] = chisq
        returned_data['residual'] = last_diff * noise_value

        # compute least square fit errors
        cov_x = fit[1]
        if cov_x is not None:
            cov_x *= returned_data['reduced-chi-square']
            cov_diag = np.sqrt(np.abs(
                np.array([cov_x[i,i] for i in range(cov_x.shape[0])])))
            fixed_p = np.array(fixed_p)
            fixed_p.fill(0.)
            lines_err, cov_err, cont_err = params_vect2arrays(
                cov_diag, fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)
            ls_fit_errors = np.empty_like(full_lines_p)
            ls_fit_errors[:,1:] = lines_err
            if cov_p_mask[0]: ls_fit_errors[:,3] = cov_err[0]
            if cov_p_mask[1]: ls_fit_errors[:,2] = cov_err[1]
            ls_fit_errors[:,0] = math.sqrt(np.sum(cont_err**2.))

            returned_data['lines-params-err'] = ls_fit_errors
            returned_data['cont-params-err'] = cont_err
        
        # compute SNR
        # recompute noise value in case the continnum is not linear
        # (noise can be a lot overestimated in this case)
        noise_value = np.std(x - fitted_vector[np.min(signal_range):
                                               np.max(signal_range)])
        returned_data['snr'] = full_lines_p[:,1] / noise_value


        # compute FLUX
        if fmodel == 'gaussian':
            returned_data['flux'] = (
                full_lines_p[:,1] * full_lines_p[:,3]
                / (2.*math.sqrt(2.*math.log(2.)))*
                math.sqrt(2.*math.pi))
        elif fmodel == 'sinc':
            returned_data['flux'] = (
                full_lines_p[:,1] * full_lines_p[:,3] / 1.20671)
            
        # Compute analytical errors [from Minin & Kamalabadi, Applied
        # Optics, 2009]. Note that p4 = fwhm / (2*sqrt(ln(2))). These
        # errors must be near the least square errors except for the
        # background because the background modelized by the authors
        # has nothing to see with the model we use here. In general
        # lesat squares error are thus a better estimate.
    
        fit_errors = np.empty_like(full_lines_p)
        p4 = full_lines_p[:,3] / (2. * math.sqrt(math.log(2.)))
        Q = (3. * p4 * math.sqrt(math.pi)
             / (x.shape[0] * math.sqrt(2)))
        Qnans = np.nonzero(np.isnan(Q))
        Q[Qnans] = 0.
        
        if np.all(Q <= 1.):
            fit_errors[:,0] = (
                noise_value * np.abs((1./x.shape[0]) * 1./(1. - Q))**0.5)
            
            fit_errors[:,1] = (
                noise_value * np.abs((3./2.)
                               * math.sqrt(2)/(p4 * math.sqrt(math.pi))
                               * (1. - (8./9.)*Q)/(1. - Q))**0.5)
            fit_errors[:,2] = (
                noise_value * np.abs(p4 * math.sqrt(2.)
                               / (full_lines_p[:,1]**2.
                                  * math.sqrt(math.pi)))**0.5)
            fit_errors[:,3] = (
                noise_value * np.abs((3./2.)
                               * (4. * p4 * math.sqrt(2.)
                                  / (3.*full_lines_p[:,1]**2.
                                     * math.sqrt(math.pi)))
                               * ((1.-(2./3.)*Q)/(1.-Q)))**0.5)
            
            fit_errors[Qnans,:] = np.nan
        
        else:
            fit_errors.fill(np.nan)
            
        returned_data['lines-params-err-an'] = fit_errors
        
        # append list of iterations
        returned_data['iter-list'] = iter_list
        return returned_data
    else:
        return []

## def fit_lines_in_vector(vector, lines, fwhm_guess=3.5,
##                         cont_guess=None, shift_guess=0.,
##                         fix_cont=False,
##                         fix_fwhm=False, cov_fwhm=True, cov_pos=True,
##                         reguess_positions=False,
##                         return_fitted_vector=False, fit_tol=1e-5,
##                         no_absorption=False, poly_order=0,
##                         fmodel='gaussian', sig_noise=None,
##                         observation_params=None, signal_range=None,
##                         wavenumber=False):

##     """Fit multiple emission lines in a spectrum vector.

##     :param vector: Vector to fit

##     :param lines: Positions of the lines in channels

##     :param fwhm_guess: (Optional) Initial guess on the lines FWHM
##       (default 3.5).

##     :param cont_guess: (Optional) Initial guess on the continuum
##       (default None). Must be a tuple of poly_order + 1 values ordered
##       with the highest orders first.

##     :param shift_guess: (Optional) Initial guess on the global shift
##       of the lines (default 0.).

##     :param fix_cont: (Optional) If True, continuum is fixed to the
##       initial guess (default False).

##     :param fix_fwhm: (Optional) If True, FWHM value is fixed to the
##       initial guess (default False).

##     :param cov_fwhm: (Optional) If True FWHM is considered to be the
##       same for all lines and become a covarying parameter (default
##       True).

##     :param cov_pos: (Optional) If True the estimated relative
##       positions of the lines (the lines parameter) are considered to
##       be exact and only need to be shifted. Positions are thus
##       covarying. Very useful but the initial estimation of the line
##       relative positions must be very precise. This parameter can also
##       be a tuple of the same length as the number of lines to
##       distinguish the covarying lines. Covarying lines must share the
##       same number. e.g. on 4 lines, [NII]6548, Halpha, [NII]6584,
##       [SII]6717, [SII]6731, if each ion has a different velocity
##       cov_pos can be : [0,1,0,2,2]. (default False).

##     :param reguess_positions: (Optional) If True, positions are
##       guessed again. Useful if the given estimations really are rough
##       ones. Note that this must not be used with cov_pos set to True
##       (default False).

##     :param return_fitted_vector: (Optional) If True Fitted vector is
##       returned.

##     :param fit_tol: (Optional) Tolerance on the fit value (default
##       1e-2).

##     :param no_absorption: (Optional) If True, no negative amplitude
##       are returned (default False).

##     :param poly_order: (Optional) Order of the polynomial used to fit
##       continuum. Use high orders carefully (default 0).

##     :param fmodel: (Optional) Fitting model. Can be 'gaussian', 
##       'sinc', 'sinc2' or 'lorentzian' (default 'gaussian').

##     :param sig_noise: (Optional) Noise standard deviation guess. If
##       None noise value is guessed but the gaussian FWHM must not
##       exceed half of the sampling interval (default None).

##     :param observation_params: (Optional) Must be a tuple [step,
##       order]. Interpolate data before fitting when data has been
##       previously interpolated from an irregular wavelength axis to a
##       regular one.

##     :param signal_range: (Optional) A tuple (x_min, x_max) giving the
##       lowest and highest channel numbers containing signal.

##     :param wavenumber: (Optional) If True the spectrum is considered
##       to be in wavenumber. It will not be interpolated but the
##       observation params will be used to compute the real line
##       shift. If False, and if the observation params are given the
##       spectrum will be interpolated to a regular wavenumber scale (The
##       shape of the lines becomes symetric) (default False).

##     :return: a dictionary containing:
    
##       * lines parameters [key: 'lines-params'] Lines parameters are
##         given as an array of shape (lines_nb, 4). The order of the 4
##         parameters for each lines is [height at the center of the
##         line, amplitude, position, fwhm].
      
##       * lines parameters errors [key: 'lines-params-err']

##       * residual [key: 'residual']
      
##       * chi-square [key: 'chi-square']

##       * reduced chi-square [key: 'reduced-chi-square']

##       * SNR [key: 'snr']

##       * continuum parameters [key: 'cont-params']

##       * and optionally the fitted vector [key: 'fitted-vector']
##         depending on the option return_fitted_vector.

      
##     """
##     def params_arrays2vect(lines_p, lines_p_mask,
##                            cov_p, cov_p_mask,
##                            cont_p, cont_p_mask):
        
##         free_p = list(lines_p[np.nonzero(lines_p_mask)])
##         free_p += list(cov_p[np.nonzero(cov_p_mask)])
##         free_p += list(cont_p[np.nonzero(cont_p_mask)])
        
##         fixed_p = list(lines_p[np.nonzero(~lines_p_mask)])
##         fixed_p += list(cov_p[np.nonzero(~cov_p_mask)])
##         fixed_p += list(cont_p[np.nonzero(~cont_p_mask)])
        
##         return free_p, fixed_p

##     def params_vect2arrays(free_p, fixed_p, lines_p_mask,
##                            cov_p_mask, cont_p_mask):

##         cont_p = np.empty_like(cont_p_mask, dtype=float)
##         free_cont_p_nb = np.sum(cont_p_mask)
##         if free_cont_p_nb > 0:
##             cont_p[np.nonzero(cont_p_mask)] = free_p[
##                 -free_cont_p_nb:]
##             free_p = free_p[:-free_cont_p_nb]
##         if free_cont_p_nb < np.size(cont_p_mask):
##             cont_p[np.nonzero(~cont_p_mask)] = fixed_p[
##                 -(np.size(cont_p_mask) - free_cont_p_nb):]
##             fixed_p = fixed_p[
##                 :-(np.size(cont_p_mask) - free_cont_p_nb)]
            
##         cov_p = np.empty_like(cov_p_mask, dtype=float)
##         free_cov_p_nb = np.sum(cov_p_mask)
##         if free_cov_p_nb > 0:
##             cov_p[np.nonzero(cov_p_mask)] = free_p[
##                 -free_cov_p_nb:]
##             free_p = free_p[:-free_cov_p_nb]
##         if free_cov_p_nb < np.size(cov_p_mask):
            
##             cov_p[np.nonzero(~cov_p_mask)] = fixed_p[
##                 -(np.size(cov_p_mask) - free_cov_p_nb):]
##             fixed_p = fixed_p[
##                 :-(np.size(cov_p_mask) - free_cov_p_nb)]


##         lines_p = np.empty_like(lines_p_mask, dtype=float)
##         lines_p[np.nonzero(lines_p_mask)] = free_p
##         lines_p[np.nonzero(~lines_p_mask)] = fixed_p
        
##         return lines_p, cov_p, cont_p
    
##     def model(n, lines_p, cont_p, fmodel, pix_axis):
##         mod = np.zeros(n, dtype=float)
##         # continuum
##         mod += np.polyval(cont_p, np.arange(n))
##         for iline in range(lines_p.shape[0]):
            
##             if fmodel == 'sinc':
##                 mod += sinc1d(
##                     pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
##                     lines_p[iline, 2])
                
##             elif fmodel == 'lorentzian':
##                 mod += lorentzian1d(
##                     pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
##                     lines_p[iline, 2])
                
##             elif fmodel == 'sinc2':
##                 mod += np.sqrt(sinc1d(
##                     pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
##                     lines_p[iline, 2])**2.)
                
##             else:
##                 mod += gaussian1d(
##                     pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
##                     lines_p[iline, 2])
        
##         return mod

##     def add_shift(lines_pos, shift, axis):
##         if axis is not None:
##             center = int(axis.shape[0] / 2.)
##             # delta_lambda is evaluated at the center of the axis
##             # delta_v must be a constant but the corresponding
##             # delta_lambda is not: delta_v = delta_lambda / lambda
##             delta_lambda = (pix2nm(axis, center+shift)
##                             - axis[center])
##             delta_v = delta_lambda / axis[center]

##             lines_nm = pix2nm(axis, lines_pos)
##             lines_pos = nm2pix(axis, lines_nm + delta_v * lines_nm)
##         else:
##             lines_pos += shift
##         return lines_pos

##     def diff(free_p, fixed_p, lines_p_mask, cov_p_mask,
##              cont_p_mask, data, sig, fmodel, pix_axis, axis,
##              cov_pos_mask_list, iter_list):
##         lines_p, cov_p, cont_p = params_vect2arrays(free_p, fixed_p,
##                                                     lines_p_mask,
##                                                     cov_p_mask, cont_p_mask)
##         lines_p[:,2] += cov_p[0] # + FWHM
##         # + SHIFT
##         for i in range(len(cov_pos_mask_list)):
##             ilines = np.arange(lines_p.shape[0])[
##                 np.nonzero(cov_pos_mask_list[i])]
##             lines_p[ilines,1] = add_shift(lines_p[ilines,1], cov_p[1+i], axis)
        
##         data_mod = model(np.size(data), lines_p, cont_p, fmodel, pix_axis)

##         res = (data - data_mod) / sig
##         iter_list.append((res, free_p))
##         return res

##     MAXFEV = 5000 # Maximum number of evaluation

##     MIN_LINE_SIZE = 5 # Minimum size of a line whatever the guessed
##                       # fwhm can be.

##     x = np.copy(vector)

##     if np.all(np.isnan(x)):
##         return []
    
##     if np.any(np.iscomplex(x)):
##         x = x.real
##         warnings.warn(
##             'Complex vector. Only the real part will be fitted')

##     lines = np.array(lines, dtype=float)
##     lines = lines[np.nonzero(lines > 0.)]

##     lines_nb = np.size(lines)
##     line_size = int(max(math.ceil(fwhm_guess * 3.5), MIN_LINE_SIZE))
    
##     # only 3 params max for each line, cont is defined as an N order
##     # polynomial (N+1 additional parameters)
##     lines_p = np.zeros((lines_nb, 3), dtype=float)

##     interpolate = False
##     # vector interpolation
##     if observation_params is not None:
##         step = observation_params[0]
##         order = observation_params[1]
##         if not wavenumber:
##             interpolate = True
##             axis = (create_nm_axis_ireg(
##                 x.size, step, order)[::-1])
##             nm_axis = create_nm_axis(x.size, step, order)
##             nm_axis_orig = np.copy(nm_axis)
##             axis_orig = np.copy(axis)

##             x = orb.utils.vector.interpolate_axis(
##                 x, axis, 5, old_axis=nm_axis,
##                 fill_value=np.nan)

##             # convert lines position from a regular axis to an irregular one
##             lines = nm2pix(axis, pix2nm(nm_axis, lines))
            
##             if signal_range is not None:
##                 signal_range = nm2pix(axis, pix2nm(
##                     nm_axis, signal_range))
##                 signal_range_min = math.ceil(np.min(signal_range))
##                 signal_range_max = math.floor(np.max(signal_range))
##                 signal_range = [signal_range_min, signal_range_max]

##         else:
##             axis = create_cm1_axis(x.size, step, order)
##     else:
##         axis = None
        
##     # remove parts out of the signal range
##     x_size_orig = np.size(x)
##     if signal_range is not None:
##         signal_range = np.array(signal_range).astype(int)
##         if np.min(signal_range) >= 0 and np.max(signal_range < np.size(x)):
##             x = x[np.min(signal_range):np.max(signal_range)]
##             if interpolate:
##                 axis = axis[
##                     np.min(signal_range):np.max(signal_range)]
##                 nm_axis = nm_axis[np.min(signal_range):np.max(signal_range)]
##             lines -= np.min(signal_range)
##         else:
##             raise Exception('Signal range must be a tuple (min, max) with min >= 0 and max < {:d}'.format(np.size(x)))

##     # axis
##     pix_axis = np.arange(x.shape[0])

##     # check nans
##     if np.any(np.isinf(x)) or np.any(np.isnan(x)) or (np.min(x) == np.max(x)):
##         return []
    
##     ## Guess ##
##     noise_vector = np.copy(x)
##     for iline in range(lines_nb):
##         line_center = lines[iline]
##         iz_min = int(line_center - line_size/2.)
##         iz_max = int(line_center + line_size/2.) + 1
##         if iz_min < 0: iz_min = 0
##         if iz_max > x.shape[0] - 1: iz_max = x.shape[0] - 1
##         line_box = x[iz_min:iz_max]

##         # max
##         lines_p[iline, 0] = np.max(line_box) - np.median(x)

##         # position
##         if reguess_positions:
##             lines_p[iline, 1] = (np.sum(line_box
##                                         * np.arange(line_box.shape[0]))
##                                  / np.sum(line_box)) + iz_min
        
##         # remove line from noise vector
##         noise_vector[iz_min:iz_max] = np.nan

    
##     if not reguess_positions:
##         lines_p[:, 1] = np.array(lines)

##     lines_p[:, 2] = fwhm_guess
        
##     # polynomial guess of the continuum
##     if (cont_guess is None) or (np.size(cont_guess) != poly_order + 1):
##         if (cont_guess is not None) and np.size(cont_guess) != poly_order + 1:
##             warnings.warn('Bad continuum guess shape')
##         if poly_order > 0:
##             w = np.ones_like(noise_vector)
##             nans = np.nonzero(np.isnan(noise_vector))
##             w[nans] = 0.
##             noise_vector[nans] = 0.
##             cont_guess = np.polynomial.polynomial.polyfit(
##                 np.arange(noise_vector.shape[0]), noise_vector, poly_order, w=w)
##             cont_fit = np.polynomial.polynomial.polyval(
##                 np.arange(noise_vector.shape[0]), cont_guess)
##             cont_guess = cont_guess[::-1]
##             noise_value = orb.utils.stats.robust_std(noise_vector - cont_fit)
##         else:
##             cont_guess = [orb.utils.stats.robust_median(noise_vector)]
##             noise_value = orb.utils.stats.robust_std(noise_vector)
##     else:
##         cont_fit = np.polyval(cont_guess, np.arange(noise_vector.shape[0]))
##         noise_value = orb.utils.stats.robust_std(noise_vector - cont_fit)
    
##     if sig_noise is not None:
##         noise_value = sig_noise

##     #### PARAMETERS = LINES_PARAMS (N*(3-COV_PARAMS_NB)),
##     ####              COV_PARAMS (FWHM_COEFF, SHIFT),
##     ####              CONTINUUM COEFFS

##     if np.size(cov_pos) > 1:
##         if np.size(cov_pos) != lines_nb:
##             raise Exception('If cov_pos is not True or False it must be a tuple of the same length as the lines number')
##         cov_pos_list = list()
##         cov_pos_mask_list = list()
##         for i in cov_pos:
##             if i not in cov_pos_list:
##                 cov_pos_list.append(i)
##                 cov_pos_mask_list.append(np.array(cov_pos) == i)
##     else:
##         cov_pos_list = [0]
##         cov_pos_mask_list = [np.ones(lines_nb, dtype=bool)]

##     # define cov params
##     cov_p = np.empty(1 + np.size(cov_pos_list), dtype=float)
##     cov_p[0] = 0. # COV FWHM COEFF
##     for i in range(len(cov_pos_list)):
##         cov_p[1+i] = shift_guess # COV SHIFT
##     cov_p_mask = np.zeros_like(cov_p, dtype=bool)
##     if cov_fwhm and not fix_fwhm: cov_p_mask[0] = True
##     if cov_pos != False: cov_p_mask[1:] = True
        
##     # define lines params mask
##     lines_p_mask = np.ones_like(lines_p, dtype=bool)
##     if fix_fwhm or cov_fwhm: lines_p_mask[:,2] = False
##     if cov_pos != False: lines_p_mask[:,1] = False
    
##     # define continuum params
##     cont_p = np.array(cont_guess)
##     cont_p_mask = np.ones_like(cont_p, dtype=bool)
##     if fix_cont: cont_p_mask.fill(False)

##     free_p, fixed_p = params_arrays2vect(
##         lines_p, lines_p_mask,
##         cov_p, cov_p_mask,
##         cont_p, cont_p_mask)

    
##     ### FIT ###
##     iter_list = list()
##     fit = optimize.leastsq(diff, free_p,
##                            args=(fixed_p, lines_p_mask,
##                                  cov_p_mask, cont_p_mask,
##                                  x, noise_value, fmodel, pix_axis, axis,
##                                  cov_pos_mask_list, iter_list),
##                            maxfev=MAXFEV, full_output=True,
##                            xtol=fit_tol)
    
##     ### CHECK FIT RESULTS ###
##     if fit[-1] <= 4:
##         if fit[2]['nfev'] >= MAXFEV: return [] # reject evaluation bounded fit
##         returned_data = dict()
##         last_diff = fit[2]['fvec']
##         lines_p, cov_p, cont_p = params_vect2arrays(
##             fit[0], fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)
        
        
##         # add cov to lines params
##         full_lines_p = np.empty((lines_nb, 4), dtype=float)
##         full_lines_p[:,0] = np.polyval(cont_p, lines_p[:,1])
##         full_lines_p[:,1:] = lines_p

##         for i in range(len(cov_pos_mask_list)):
##             ilines = np.arange(lines_p.shape[0])[
##                 np.nonzero(cov_pos_mask_list[i])]
##             full_lines_p[ilines,2] = add_shift(full_lines_p[ilines,2],
##                                                cov_p[1+i], axis)

##         full_lines_p[:,3] += cov_p[0] # + FWHM_COEFF
##         # check and correct
##         for iline in range(lines_nb):
##             if no_absorption:
##                 if full_lines_p[iline, 1] < 0.:
##                     full_lines_p[iline, 1] = 0.
##             if (full_lines_p[iline, 2] < 0.
##                 or full_lines_p[iline, 2] > x.shape[0]):
##                 full_lines_p[iline, :] = np.nan

##         # compute fitted vector
##         fitted_vector = np.empty(x_size_orig, dtype=float)
##         fitted_vector.fill(np.nan)
##         fitted_vector[np.min(signal_range):
##                       np.max(signal_range)] = model(
##             x.shape[0], full_lines_p[:,1:], cont_p, fmodel, pix_axis)
##         if return_fitted_vector:
##             returned_data['fitted-vector'] = fitted_vector

##         # correct shift for signal range
##         if signal_range is not None:            
##             full_lines_p[:,2] += np.min(signal_range)
        
##         returned_data['lines-params'] = full_lines_p
##         returned_data['cont-params'] = cont_p
       
##         # compute reduced chi square
##         chisq = np.sum(last_diff**2.)
##         red_chisq = chisq / (np.size(x) - np.size(free_p))
##         returned_data['reduced-chi-square'] = red_chisq
##         returned_data['chi-square'] = chisq
##         returned_data['residual'] = last_diff * noise_value

##         # compute least square fit errors
##         cov_x = fit[1]
##         if cov_x is not None:
##             cov_x *= returned_data['reduced-chi-square']
##             cov_diag = np.sqrt(np.abs(
##                 np.array([cov_x[i,i] for i in range(cov_x.shape[0])])))
##             fixed_p = np.array(fixed_p)
##             fixed_p.fill(0.)
##             lines_err, cov_err, cont_err = params_vect2arrays(
##                 cov_diag, fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)
##             ls_fit_errors = np.empty_like(full_lines_p)
##             ls_fit_errors[:,1:] = lines_err
##             if cov_p_mask[0]: ls_fit_errors[:,3] = cov_err[0]
##             if cov_p_mask[1]: ls_fit_errors[:,2] = cov_err[1]
##             ls_fit_errors[:,0] = math.sqrt(np.sum(cont_err**2.))

##             returned_data['lines-params-err'] = ls_fit_errors
##             returned_data['cont-params-err'] = cont_err
        
##         # compute SNR
##         # recompute noise value in case the continnum is not linear
##         # (noise can be a lot overestimated in this case)
##         noise_value = np.std(x - fitted_vector[np.min(signal_range):
##                                                np.max(signal_range)])
##         returned_data['snr'] = full_lines_p[:,1] / noise_value


##         # compute FLUX
##         ## if wavenumber:
##         ##     amp = data.array(returned_data['lines-params'][:, 1],
##         ##                      returned_data['lines-params-err'][:, 1])
##         ##     fwhm = data.array(returned_data['lines-params'][:, 3],
##         ##                       returned_data['lines-params-err'][:, 3])
##         ##     flux = amp * fwhm / 1.20671
            

##         # Compute analytical errors [from Minin & Kamalabadi, Applied
##         # Optics, 2009]. Note that p4 = fwhm / (2*sqrt(ln(2))). These
##         # errors must be near the least square errors except for the
##         # background because the background modelized by the authors
##         # has nothing to see with the model we use here. In general
##         # lesat squares error are thus a better estimate.
    
##         fit_errors = np.empty_like(full_lines_p)
##         p4 = full_lines_p[:,3] / (2. * math.sqrt(math.log(2.)))
##         Q = (3. * p4 * math.sqrt(math.pi)
##              / (x.shape[0] * math.sqrt(2)))
##         Qnans = np.nonzero(np.isnan(Q))
##         Q[Qnans] = 0.
        
##         if np.all(Q <= 1.):
##             fit_errors[:,0] = (
##                 noise_value * np.abs((1./x.shape[0]) * 1./(1. - Q))**0.5)
            
##             fit_errors[:,1] = (
##                 noise_value * np.abs((3./2.)
##                                * math.sqrt(2)/(p4 * math.sqrt(math.pi))
##                                * (1. - (8./9.)*Q)/(1. - Q))**0.5)
##             fit_errors[:,2] = (
##                 noise_value * np.abs(p4 * math.sqrt(2.)
##                                / (full_lines_p[:,1]**2.
##                                   * math.sqrt(math.pi)))**0.5)
##             fit_errors[:,3] = (
##                 noise_value * np.abs((3./2.)
##                                * (4. * p4 * math.sqrt(2.)
##                                   / (3.*full_lines_p[:,1]**2.
##                                      * math.sqrt(math.pi)))
##                                * ((1.-(2./3.)*Q)/(1.-Q)))**0.5)
            
##             fit_errors[Qnans,:] = np.nan
        
##         else:
##             fit_errors.fill(np.nan)
            
##         returned_data['lines-params-err-an'] = fit_errors
        
##         # interpolate back parameters
##         if interpolate:
##             lines_dx = returned_data['lines-params'][:,2]
            
##             lines_dx = nm2pix(
##                 nm_axis_orig, pix2nm(axis_orig, lines_dx))
##             returned_data['lines-params'][:,2] = lines_dx

##             if return_fitted_vector:
##                 returned_data['fitted-vector'] = orb.utils.vector.interpolate_axis(
##                     returned_data['fitted-vector'], nm_axis_orig, 5,
##                     old_axis=axis_orig, fill_value=np.nan)

##         # append list of iterations
##         returned_data['iter-list'] = iter_list
        
##         return returned_data
##     else:
##         return []
