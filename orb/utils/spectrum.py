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
import time

import orb.constants
import orb.cutils
import orb.utils.fft
import orb.utils.stats
import orb.utils.vector
import orb.fit

import cspectrum

def get_nm_axis_min(n, step, order, corr=1.):
    """Return min wavelength of regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order (cannot be 0)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return 2. * float(step) / float(order + 1.) / float(corr)

def get_nm_axis_max(n, step, order, corr=1.):
    """Return max wavelength of regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order (cannot be 0)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return 2. * float(step) / float(order) / float(corr)

def get_nm_axis_step(n, step, order, corr=1.):
    """Return step size of a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order (cannot be 0)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if (order > 0): 
        return 2. * step / (order * (order + 1.) * corr) / (n - 1.)
    else: raise Exception("order must be > 0")

def create_nm_axis(n, step, order, corr=1.):
    """Create a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order (cannot be 0)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    
    nm_min = get_nm_axis_min(n, step, order, corr=corr)
    if (order > 0): 
        nm_max = get_nm_axis_max(n, step, order, corr=corr)
        return np.linspace(nm_min, nm_max, n, dtype=np.longdouble)
    else:
        raise Exception("order must be > 0")
        

def get_cm1_axis_min(n, step, order, corr=1.):
    """Return min wavenumber of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return float(order) / (2.* float(step)) * float(corr) * 1e7

def get_cm1_axis_max(n, step, order, corr=1.):
    """Return max wavenumber of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return float(order + 1.) / (2. * float(step)) * float(corr) * 1e7

def get_cm1_axis_step(n, step, corr=1.):
    """Return step size of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return corr / (2. * (n - 1.) * step) * 1e7
    
def create_cm1_axis(n, step, order, corr=1.):
    """Create a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    cm1_min = get_cm1_axis_min(n, step, order, corr=corr)
    cm1_max = get_cm1_axis_max(n, step, order, corr=corr)
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
        

def fast_w2pix(w, axis_min, axis_step):
    return np.abs(w - axis_min) / axis_step

def fast_pix2w(pix, axis_min, axis_step):
    return pix * axis_step + axis_min
    
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
                        
def fit_lines_in_spectrum(spectrum, lines, step, order, nm_laser,
                          nm_laser_obs,
                          wavenumber=True, fwhm_guess=3.5,
                          cont_guess=None,
                          shift_guess=0.,
                          #fix_cont=False,
                          fix_fwhm=False, cov_fwhm=True,
                          cov_pos=True,
                          fit_tol=1e-10,
                          poly_order=0,
                          fmodel='gaussian',
                          signal_range=None,
                          filter_file_path=None,
                          fix_filter=False):
    """Fit lines in spectrum
    """
    correction_coeff = float(nm_laser_obs) / nm_laser

    if wavenumber:
        axis_min = get_cm1_axis_min(spectrum.shape[0], step, order,
                                    corr=correction_coeff)
        axis_step = get_cm1_axis_step(spectrum.shape[0], step, 
                                      corr=correction_coeff)
  
    else:
        raise Exception('wavelength fit not implemented yet')
        axis_min = get_nm_axis_min(spectrum.shape[0], step, order,
                                   corr=correction_coeff)
        axis_step = get_nm_axis_step(spectrum.shape[0], step, order,
                                     corr=correction_coeff)

    if signal_range is not None:
        signal_range_pix = fast_w2pix(
            np.array(signal_range), axis_min, axis_step)
        minx = int(np.min(signal_range_pix))
        maxx = int(math.ceil(np.max(signal_range_pix)))
    else:
        signal_range_pix = None
        
    if fix_fwhm: fwhm_def = 'fixed'
    elif cov_fwhm: fwhm_def = '1'
    else: fwhm_def = 'free'

    if np.size(cov_pos) > 1:
        pos_def = cov_pos
    else:
        if not cov_pos: pos_def = 'free'
        else: pos_def = '1'

    ## import pylab as pl
    ## axis = create_cm1_axis(spectrum.shape[0], step, order, corr=nm_laser_obs/nm_laser)
    ## pl.plot(axis, spectrum)
    ## searched_lines = lines + orb.utils.spectrum.line_shift(shift_guess, lines, wavenumber=wavenumber)
    ## [pl.axvline(x=iline) for iline in searched_lines]
    ## [pl.axvline(x=iline, ls=':') for iline in signal_range]
    ## pl.show()

    if filter_file_path is not None:
        filter_function = orb.utils.filters.get_filter_function(
            filter_file_path, step, order, spectrum.shape[0],
            wavenumber=wavenumber,
            silent=True)[0]
        if wavenumber:
            filter_axis = create_cm1_axis(spectrum.shape[0], step, order, corr=1.)
            filter_axis_calib = create_cm1_axis(spectrum.shape[0], step, order,
                                                corr=nm_laser_obs/nm_laser)
        else:
            raise Exception('Not implemented yet')
        filter_function = orb.utils.vector.interpolate_axis(
            filter_function, filter_axis_calib, 1, old_axis=filter_axis)
        if fix_filter:
            filter_def = 'fixed'
        else:
            filter_def = 'free'
            
    else:
        filter_function = np.ones(spectrum.shape[0], dtype=float)
        filter_def = 'fixed'

    fs = orb.fit.FitVector(spectrum,
                           ((orb.fit.Cm1LinesModel, 'add'),
                            (orb.fit.ContinuumModel, 'add'),
                            (orb.fit.FilterModel, 'mult')),
                           ({'step-nb':spectrum.shape[0],
                             'step':step,
                             'order':order,
                             'nm-laser':nm_laser,
                             'nm-laser-obs':nm_laser_obs,
                             'line-nb':np.size(lines),
                             'amp-def':'free',
                             'fwhm-def':fwhm_def,
                             'pos-guess':lines,
                             'pos-cov':shift_guess,
                             'pos-def':pos_def,
                             'fmodel':fmodel,
                             'fwhm-guess':fwhm_guess,
                             'sigma-def':'1'},
                            {'poly-order':poly_order,
                             'cont-guess':cont_guess},
                            {'filter-function':filter_function,
                             'shift-def':filter_def}),
                           fit_tol=fit_tol,
                           signal_range=signal_range_pix)

    fit = fs.fit()
    if fit != []:

        # compute velocity
        line_params = fit['fit-params'][0]
        line_params = line_params.reshape(
            (3, line_params.shape[0]/3)).T
        pos_wave = line_params[:,1]
        
        fit['velocity'] = compute_radial_velocity(
            pos_wave, lines, wavenumber=wavenumber)
        
        if 'fit-params-err' in fit:
            # compute vel err
            line_params_err = fit['fit-params-err'][0]
            line_params_err = line_params_err.reshape(
                (3, line_params_err.shape[0]/3)).T
            pos_wave_err = line_params_err[:,1]
            
            fit['velocity-err'] = np.abs(compute_radial_velocity(
                pos_wave + pos_wave_err, pos_wave, wavenumber=wavenumber))
        
        ## create a formated version of the parameters:
        ## [N_LINES, (H, A, DX, FWHM)]
        cont_params = fit['fit-params'][1]
        
        # evaluate continuum level at each position
        pos_pix = fast_w2pix(
            pos_wave, axis_min, axis_step)
        cont_model = fs.models[1]
        cont_model.set_p_val(cont_params)
        cont_level = cont_model.get_model(pos_pix)
        all_params = np.append(cont_level, line_params.T)
        fit['lines-params'] = all_params.reshape((4, all_params.shape[0]/4)).T

        if 'fit-params-err' in fit:
            cont_params_err = fit['fit-params-err'][1]
            # evaluate error on continuum level at each position
            cont_model.set_p_val(cont_params + cont_params_err / 2.)
            cont_level_max = cont_model.get_model(pos_pix)
            cont_model.set_p_val(cont_params - cont_params_err / 2.)
            cont_level_min = cont_model.get_model(pos_pix)
            cont_level_err = np.abs(cont_level_max - cont_level_min)
            all_params_err = np.append(cont_level_err, line_params_err.T)
            fit['lines-params-err'] = all_params_err.reshape(
                (4, all_params_err.shape[0]/4)).T

            fit['snr'] = fit['lines-params'][:,1] / fit['lines-params-err'][:,1]

        return fit

    else: return []
    
def fit_lines_in_spectrum_old(spectrum, lines, step, order, nm_laser,
                              nm_laser_obs, wavenumber=True, **kwargs):


    correction_coeff = float(nm_laser_obs) / nm_laser

    if wavenumber:
        axis_min = get_cm1_axis_min(spectrum.shape[0], step, order,
                                    corr=correction_coeff)
        axis_step = get_cm1_axis_step(spectrum.shape[0], step, 
                                      corr=correction_coeff)
  
    else:
        axis_min = get_nm_axis_min(spectrum.shape[0], step, order,
                                   corr=correction_coeff)
        axis_step = get_nm_axis_step(spectrum.shape[0], step, order,
                                     corr=correction_coeff)
            
    lines_pix = cspectrum.fit_lines_in_spectrum().fast_w2pix(
        lines.astype(np.float64), axis_min, axis_step)

    if 'fwhm_guess' in kwargs:
        kwargs['fwhm_guess'] = kwargs['fwhm_guess'] / axis_step

    if 'signal_range' in kwargs:
        if kwargs['signal_range'] is not None:
            signal_range_pix = cspectrum.fit_lines_in_spectrum().fast_w2pix(
                np.array(kwargs['signal_range'], dtype=np.float64),
                float(axis_min), float(axis_step))
            minx = int(np.min(signal_range_pix))
            maxx = int(math.ceil(np.max(signal_range_pix)))
            kwargs['signal_range'] = [minx, maxx]
            
    fadd_shift_p = (axis_min, axis_step, lines.astype(np.float64), wavenumber)

    ## import pylab as pl
    ## pl.plot(spectrum)
    ## for iline in lines_pix:
    ##     pl.axvline(x=iline)
    ## pl.show()
    ## quit()
    fit = fit_lines_in_vector(spectrum, lines_pix,
                              fadd_shift=(
                                  cspectrum.fit_lines_in_spectrum().shift,
                                  fadd_shift_p),
                              **kwargs)
    if fit != []:
        fit['velocity'] = fit['cov-velocity']
        fit['fwhm-wave'] = fit['lines-params'][:,3] * axis_step
        if 'lines-params-err' in fit:
            fit['velocity-err'] = fit['cov-velocity-err']
            fit['fwhm-wave-err'] = fit['lines-params-err'][:,3] * axis_step
       
            
        ## import pylab as pl
        ## pl.plot(spectrum)
        ## pl.plot(fit['fitted-vector'])
        ## for iline in lines_pix:
        ##     pl.axvline(x=iline)
        ## pl.show()
        

    return fit
    


def fit_lines_in_vector(vector, lines, fwhm_guess=3.5,
                        cont_guess=None, shift_guess=0.,
                        fix_cont=False,
                        fix_fwhm=False, cov_fwhm=True, cov_pos=True,
                        reguess_positions=False,
                        return_fitted_vector=False, fit_tol=1e-10,
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
        
        fixed_p = list(lines_p[np.nonzero(lines_p_mask==0)])
        fixed_p += list(cov_p[np.nonzero(cov_p_mask==0)])
        fixed_p += list(cont_p[np.nonzero(cont_p_mask==0)])
        return np.array(free_p), np.array(fixed_p)

    

    def add_shift(lines_pos, shift, p):
        return lines_pos + shift


    MAXFEV = 5000 # Maximum number of evaluation

    MIN_LINE_SIZE = 5 # Minimum size of a line whatever the guessed
                      # fwhm can be.
    if fmodel == 'sinc' or fmodel == 'sinc2':             
        add_params_nb = 1 # Number of additional parameters
    else:
        add_params_nb = 0
    
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
    pix_axis = np.arange(x.shape[0]).astype(float)

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
    ####              COV_PARAMS (FWHM_COEFF, SHIFT, SIGMA),
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
        cov_pos_mask_list = [np.ones(lines_nb, dtype=np.uint8)]

    # define cov params
    # FWHM + ADD_PARAMS
    cov_p = np.empty(1 + add_params_nb + np.size(cov_pos_list),
                     dtype=np.float64)
    cov_p[0] = 0. # COV FWHM COEFF
    cov_p[1:1 + add_params_nb] = 1e-8  # COV ADD
    for i in range(len(cov_pos_list)):
        cov_p[1 + add_params_nb + i] = shift_guess # COV SHIFT
    
    cov_p_mask = np.zeros_like(cov_p, dtype=np.uint8)
    if cov_fwhm and not fix_fwhm: cov_p_mask[0] = 1 # COV FWHM COEFF
    
    cov_p_mask[1:1 + add_params_nb] = 1 # COV ADD
    
    # COV SHIFT
    if np.size(cov_pos) == 1:
        if cov_pos: cov_p_mask[1 + add_params_nb:] = 1
    else:
        cov_p_mask[1 + add_params_nb:] = 1 
        
    # define lines params mask
    lines_p_mask = np.ones_like(lines_p, dtype=np.uint8)
    if fix_fwhm or cov_fwhm: lines_p_mask[:,2] = 0
    if np.size(cov_pos) == 1:
        if cov_pos: lines_p_mask[:,1] = 0
    else:
        lines_p_mask[:,1] = 0
    
    # define continuum params
    cont_p = np.array(cont_guess)
    cont_p_mask = np.ones_like(cont_p, dtype=np.uint8)
    if fix_cont: cont_p_mask.fill(0)

    free_p, fixed_p = params_arrays2vect(
        lines_p, lines_p_mask,
        cov_p, cov_p_mask,
        cont_p, cont_p_mask)

    ### FIT ###
    iter_list = list()
    
    fit = optimize.leastsq(cspectrum.fit_lines_in_vector().diff,
                           free_p.astype(np.float64),
                           args=(fixed_p.astype(np.float64),
                                 lines_p_mask,
                                 cov_p_mask, cont_p_mask,
                                 x.astype(np.float64),
                                 noise_value, fmodel, pix_axis.astype(float),
                                 cov_pos_mask_list, iter_list, fadd_shift,
                                 fadd_shiftp),
                           maxfev=MAXFEV, full_output=True,
                           xtol=fit_tol)

    ### CHECK FIT RESULTS ###
    if fit[-1] <= 4:
        
        if fit[2]['nfev'] >= MAXFEV: return [] # reject maxfev bounded fit
        returned_data = dict()
        last_diff = fit[2]['fvec']
        
        ## compute fitted vector ###########
        fitted_vector = np.empty(x_size_orig, dtype=float)
        fitted_vector.fill(np.nan)
        fitted_vector[
            np.min(signal_range):
            np.max(signal_range)] = cspectrum.fit_lines_in_vector().model(
            fit[0], fixed_p, lines_p_mask, cov_p_mask,
            cont_p_mask, np.size(x), fmodel, pix_axis,
            cov_pos_mask_list,
            fadd_shift, fadd_shiftp)
        
        if return_fitted_vector:
            returned_data['fitted-vector'] = fitted_vector

        ## compute lines params ###########
        lines_p, cov_p, cont_p = (
            cspectrum.fit_lines_in_vector().params_vect2arrays(
                fit[0], fixed_p, lines_p_mask, cov_p_mask, cont_p_mask))
        
        # add cov to lines params
        full_lines_p = np.empty((lines_nb, 4), dtype=float)
        full_lines_p[:,0] = np.polyval(cont_p, lines_p[:,1])
        full_lines_p[:,1:] = lines_p

        full_lines_p[:,3] += cov_p[0] # COV FWHM
        cov_velocities = np.zeros(lines_nb, dtype=float)
        if cov_p_mask[1 + add_params_nb]: # COV SHIFT
            cov_velocities = np.empty(lines_p_mask.shape[0], dtype=np.float64)
            for i in range(len(cov_pos_mask_list)):
                cov_velocities[np.nonzero(cov_pos_mask_list[i])] = cov_p[
                    1 + add_params_nb + i]
            full_lines_p[:,2] = fadd_shift(full_lines_p[:,2],
                                           cov_velocities, fadd_shiftp)

        # check and correct
        for iline in range(lines_nb):
            if no_absorption:
                if full_lines_p[iline, 1] < 0.:
                    full_lines_p[iline, 1] = 0.
            if (full_lines_p[iline, 2] < 0.
                or full_lines_p[iline, 2] > x.shape[0]):
                full_lines_p[iline, :] = np.nan
                
        # correct shift for signal range
        if signal_range is not None:            
            full_lines_p[:,2] += np.min(signal_range)
        
        returned_data['lines-params'] = full_lines_p
        returned_data['cont-params'] = cont_p
        returned_data['cov-velocity'] = cov_velocities
        if fmodel == 'sinc' or fmodel == 'sinc2':
            returned_data['sigma'] = abs(cov_p[1])

        ## compute stats ###########
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
            lines_err, cov_err, cont_err = (
                cspectrum.fit_lines_in_vector().params_vect2arrays(
                    cov_diag, fixed_p, lines_p_mask, cov_p_mask, cont_p_mask))
            ls_fit_errors = np.empty_like(full_lines_p)
            ls_fit_errors[:,1:] = lines_err

            if cov_p_mask[0]: ls_fit_errors[:,3] = cov_err[0] # FWHM ERR

            if fmodel == 'sinc' or fmodel == 'sinc2':
                returned_data['sigma-err'] = abs(cov_err[1])
            
            if cov_p_mask[1 + add_params_nb]: # COV SHIFT ERR
                shift_err = np.empty(lines_p_mask.shape[0], dtype=np.float64)
                for i in range(len(cov_pos_mask_list)):
                    shift_err[np.nonzero(cov_pos_mask_list[i])] = cov_err[
                        1 + add_params_nb + i]
                
                ls_fit_errors[:,2] = (fadd_shift(full_lines_p[:,2],
                                                 shift_err, fadd_shiftp)
                                      - full_lines_p[:,2])
                returned_data['cov-velocity-err'] = shift_err
                
            ls_fit_errors[:,0] = math.sqrt(np.sum(cont_err**2.)) # CONT ERR

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
                full_lines_p[:,1] * full_lines_p[:,3]
                / orb.constants.FWHM_SINC_COEFF)
            
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
        ## returned_data['iter-list'] = iter_list
        return returned_data
    else:
        return []
