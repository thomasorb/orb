#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fft.py

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
## or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

import sys
import numpy as np
import math
import warnings
import scipy
from scipy import signal

import orb.utils.vector
import orb.utils.spectrum
import orb.utils.stats
import orb.cutils


def apodize(s, apodization_function='2.0'):
    """Apodize a spectrum

    :param s: Spectrum
    
    :param apodization_function: (Optional) A Norton-Beer apodization
      function (default 2.0)
    """
    s_ifft = np.fft.ifft(s)
    w = norton_beer_window(apodization_function, s.shape[0])
    w = np.roll(w, w.shape[0]/2)
    s_ifft *= w
    if np.iscomplexobj(s):
        return np.fft.fft(s_ifft)
    else:
        return np.fft.fft(s_ifft).real

def find_zpd(interf, step_number=None,
             return_zpd_shift=False):
    """Return the index of the ZPD along the z axis.

    :param step_number: (Optional) If the full number of steps is
      greater than the number of frames of the cube. Useful when
      the interferograms are non symetric (default None).

    :param return_zpd_shift: (Optional) If True return ZPD shift
      instead of ZPD index (default False).
    """
    if step_number is not None:
        dimz = step_number
    else:
        dimz = interf.shape[0]

    interf = np.copy(interf)
    # correct vector for zeros
    interf[
        np.nonzero(interf == 0)] = np.median(interf)

    # filtering vector to remove low and high frequency patterns (e.g. sunrise)
    interf = orb.utils.vector.fft_filter(interf, 0.5, filter_type='low_pass')
    interf = orb.utils.vector.fft_filter(interf, 0.3, filter_type='high_pass')
    
    full_interf = np.zeros(dimz, dtype=float)
    full_interf[:interf.shape[0]] = interf
    
    # vector is weighted so that the center part is prefered
    full_interf *= norton_beer_window(fwhm='1.5', n=dimz)
    
    # absolute value of the vector
    full_interf = np.sqrt(full_interf**2.)
    
    # ZPD is defined to be at the maximum of the vector
    zpd_index = np.argmax(full_interf)

    #zpd_shift = int(int(self.dimz/2.) - zpd_index + self.dimz%2)
    zpd_shift = int(int(dimz/2.) - zpd_index)
    
    if return_zpd_shift:
        return zpd_shift
    else:
        return zpd_index


def count_nonzeros(a):
    """Return the length of nonzeros parts in a vector as a vector of
    the same length with the length of each part at each occurence of
    a nonzero number.

    e.g. : if a = [0,0,0,1,1,0,1] this function returns: [0,0,0,2,2,0,1]

    :param a: A vector.
    """
    counts = np.zeros_like(a)
    for iz in range(a.shape[0]):
        if a[iz] != 0 and counts[iz] == 0:
            end_count = False
            il = 0
            while not end_count:
                if (iz+il >= a.shape[0]
                    or a[iz+il] == 0):
                    end_count = True
                else:
                    il += 1
            counts[iz:iz+il] = il
    return counts


def amplitude(a):
    """Return the amplitude of a complex number"""
    return np.sqrt(a.real**2. + a.imag**2)

def phase(a):
    """Return the phase of a complex number"""
    return np.arctan2(a.imag, a.real)

def real(amp, pha):
    """Return the real part from amplitude and phase"""
    return amp * np.cos(pha) 

def imag(amp, pha):
    """Return the imaginary part from amplitude and phase"""
    return amp * np.sin(pha)

def next_power_of_two(n):
    """Return the next power of two greater than n.
    
    :param n: The number from which the next power of two has to be
      computed. Can be an array of numbers.
    """
    return np.array(2.**np.ceil(np.log2(n))).astype(int)

def raw_fft(x, apod=None, inverse=False, return_complex=False,
            return_phase=False):
    """
    Compute the raw FFT of a vector.

    Return the absolute value of the complex vector by default.
    
    :param x: Interferogram.
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.norton_beer_window` (default None)

    :param inverse: (Optional) If True compute the inverse FFT
      (default False).

    :param return_complex: (Optional) If True, the complex vector is
      returned (default False).

    :param return_phase: (Optional) If True, the phase is
      returned.(default False)
    
    """
    x = np.copy(x)
    windows = ['1.1', '1.2', '1.3', '1.4', '1.5',
               '1.6', '1.7', '1.8', '1.9', '2.0']
    N = x.shape[0]
    
    # mean substraction
    x -= np.mean(x)
    
    # apodization
    if apod in windows:
        x *= norton_beer_window(apod, N)
    elif apod is not None:
        raise Exception("Unknown apodization function try %s"%
                        str(windows))
        
    # zero padding
    zv = np.zeros(N*2, dtype=x.dtype)
    zv[int(N/2):int(N/2)+N] = x

    # zero the centerburst
    zv = np.roll(zv, zv.shape[0]/2)
    
    # FFT
    if not inverse:
        x_fft = (np.fft.fft(zv))[:N]
    else:
        x_fft = (np.fft.ifft(zv))[:N]
        
    if return_complex:
        return x_fft
    elif return_phase:
        return np.unwrap(np.angle(x_fft))
    else:
        return np.abs(x_fft)
     
def cube_raw_fft(x, apod=None):
    """Compute the raw FFT of a cube (the last axis
    beeing the interferogram axis)

    :param x: Interferogram cube
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.norton_beer_window` (default None)
    """
    x = np.copy(x)
    windows = ['1.1', '1.2', '1.3', '1.4', '1.5',
               '1.6', '1.7', '1.8', '1.9', '2.0']
    N = x.shape[-1]
    # mean substraction
    x = (x.T - np.mean(x, axis=-1)).T
    # apodization
    if apod in windows:
        x *= norton_beer_window(apod, N)
    elif apod is not None:
        raise Exception("Unknown apodization function try %s"%
                             str(windows))
    # zero padding
    zv_shape = np.array(x.shape)
    zv_shape[-1] = N*2
    zv = np.zeros(zv_shape)
    zv[:,int(N/2):int(N/2)+N] = x
    # FFT
    return np.abs((np.fft.fft(zv))[::,:N])

def norton_beer_window(fwhm='1.6', n=1000):
    """
    Return an extended Norton-Beer window function (see [NAY2007]_).

    Returned window is symmetrical.
    
    :param fwhm: FWHM relative to the sinc function. Must be: 1.1,
       1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 or 2.0. (default '1.6')
       
    :param n: Number of points (default 1000)

    .. note:: Coefficients of the extended Norton-Beer functions
       apodizing functions [NAY2007]_ :
    
       ==== ======== ========= ======== ======== ======== ======== 
       FWHM    C0       C1        C2       C4       C6       C8
       ---- -------- --------- -------- -------- -------- -------- 
       1.1  0.701551 -0.639244 0.937693 0.000000 0.000000 0.000000
       1.2  0.396430 -0.150902 0.754472 0.000000 0.000000 0.000000
       1.3  0.237413 -0.065285 0.827872 0.000000 0.000000 0.000000
       1.4  0.153945 -0.141765 0.987820 0.000000 0.000000 0.000000
       1.5  0.077112 0.000000  0.703371 0.219517 0.000000 0.000000
       1.6  0.039234 0.000000  0.630268 0.234934 0.095563 0.000000
       1.7  0.020078 0.000000  0.480667 0.386409 0.112845 0.000000
       1.8  0.010172 0.000000  0.344429 0.451817 0.193580 0.000000
       1.9  0.004773 0.000000  0.232473 0.464562 0.298191 0.000000
       2.0  0.002267 0.000000  0.140412 0.487172 0.256200 0.113948
       ==== ======== ========= ======== ======== ======== ========

    .. [NAY2007] Naylor, D. A., & Tahic, M. K. (2007). Apodizing
       functions for Fourier transform spectroscopy. Journal of the
       Optical Society of America A.
    """
    
    norton_beer_coeffs = [
        [1.1, 0.701551, -0.639244, 0.937693, 0., 0., 0., 0., 0., 0.],
        [1.2, 0.396430, -0.150902, 0.754472, 0., 0., 0., 0., 0., 0.],
        [1.3, 0.237413, -0.065285, 0.827872, 0., 0., 0., 0., 0., 0.],
        [1.4, 0.153945, -0.141765, 0.987820, 0., 0., 0., 0., 0., 0.],
        [1.5, 0.077112, 0., 0.703371, 0., 0.219517, 0., 0., 0., 0.],
        [1.6, 0.039234, 0., 0.630268, 0., 0.234934, 0., 0.095563, 0., 0.],
        [1.7, 0.020078, 0., 0.480667, 0., 0.386409, 0., 0.112845, 0., 0.],
        [1.8, 0.010172, 0., 0.344429, 0., 0.451817, 0., 0.193580, 0., 0.],
        [1.9, 0.004773, 0., 0.232473, 0., 0.464562, 0., 0.298191, 0., 0.],
        [2.0, 0.002267, 0., 0.140412, 0., 0.487172, 0., 0.256200, 0., 0.113948]]

    fwhm_list = ['1.1', '1.2', '1.3', '1.4', '1.5',
                 '1.6', '1.7', '1.8', '1.9', '2.0']
    if fwhm in fwhm_list:
        fwhm_index = fwhm_list.index(fwhm)
    else:
        raise Exception("Bad extended Norton-Beer window FWHM. Must be in : " + str(fwhm_list))

    x = np.linspace(-1., 1., n)

    nb = np.zeros_like(x)
    for index in range(9):
        nb += norton_beer_coeffs[fwhm_index][index+1]*(1. - x**2)**index
    return nb

def learner95_window(n):
    """Return the apodization function described in Learner et al.,
    J. Opt. Soc. Am. A, 12, (1995).

    This function is closely related to the minimum four-term
    Blackman-Harris window.

    Returned window is symmetrical.
    
    :param n: Number of points.
    """
    x = np.linspace(-1., 1., n)
    return (0.355766
            + 0.487395 * np.cos(math.pi*x)
            + 0.144234 * np.cos(2.*math.pi*x)
            + 0.012605 * np.cos(3.*math.pi*x))



def border_cut_window(n, coeff=0.2):
    """Return a window function with only the edges cut by a nice
    gaussian shape function.
    
    :param n: Window length
    :param coeff: Border size in percentage of the total length.
    """
    window = np.ones(n)
    border_length = int(float(n)*coeff)
    if border_length <= 1:
        window[0] = 0.
        window[-1] = 0.
    else:
        borders = signal.get_window(("gaussian",border_length/3.),
                                    border_length*2+1)
        z = int(float(borders.shape[0])/2.)
        window[:z] = borders[:z]
        window[-z:] = borders[-z:]
    return window


def get_lr_phase(interf, n_phase=None, return_lr_spectrum=False):
    """Return a low resolution phase from a given interferogram vector.

    :param interf: Interferogram vector
    
    :param n_phase: (Optional) Number of points for phase
      computation. Of course it can be no greater than the number of
      points of the interferogram. If None, this is set to 50% of the
      interferogram length (Default None).

    :param return_lr_spectrum: (Optional) If True return also the low
      resolution spectrum from which phase is computed (Default False).
    """
    LOW_RES_COEFF = 0.5 # Ratio of the number of points for phase
                        # computation over the number of points of the
                        # interferogram
                        
    dimz = interf.shape[0]
    # define the number of points for phase computation
    if n_phase is None:
        n_phase = int(LOW_RES_COEFF * float(dimz))
            
    elif n_phase > dimz:
        warnings.warn("The number of points for phase computation is too high (it can be no greater than the interferogram length). Phase is computed with the maximum number of points available")
        n_phase = dimz

    if n_phase != dimz:
        lr_interf = np.copy(interf[
            int((dimz - n_phase)/2.):
            int((dimz - n_phase)/2.) + n_phase])
    else:
        lr_interf = np.copy(interf)
        
    # apodization
    lr_interf *= learner95_window(n_phase)
  
    # zero padding
    zp_phase_len = next_power_of_two(2 * n_phase)
    zp_border = int((zp_phase_len - n_phase) / 2.)
    temp_vector = np.zeros(zp_phase_len, dtype=float)
    temp_vector[zp_border:(zp_border + n_phase)] = lr_interf
    lr_interf = temp_vector
    
    # centerburst
    lr_interf = np.roll(
        lr_interf, zp_phase_len/2 - int((dimz&1 and not n_phase&1)))
    
    # fft
    lr_spectrum = np.fft.fft(lr_interf)[:zp_phase_len/2]
    lr_phase = np.unwrap(np.angle(lr_spectrum))
    if not return_lr_spectrum:
        return orb.utils.vector.interpolate_size(lr_phase, n_phase, 1)
    else:
        return (orb.utils.vector.interpolate_size(lr_phase, n_phase, 1),
                orb.utils.vector.interpolate_size(np.abs(lr_spectrum), n_phase, 1))

def compute_phase_coeffs_vector(phase_maps,
                                res_map=None):
    """Return a vector containing the mean of the phase
    coefficients for each given phase map.

    :param phase_maps: Tuple of phase maps. Coefficients are
      sorted in the same order as the phase maps.

    :param res_map: (Optional) If given this map is used to
      get only the well fitted coefficients in order to compute a
      more precise mean coefficent.
    """
    BEST_RATIO = 0.2 # Max ratio of coefficients considered as good
    
    print "Computing phase coefficients of order > 0"
    res_map[np.nonzero(res_map == 0)] = np.nanmax(res_map)
    res_map[np.nonzero(np.isnan(res_map))] = np.nanmax(res_map)
    res_distrib = res_map[np.nonzero(~np.isnan(res_map))].flatten()
    # residuals are sorted and sigma-cut filtered 
    best_res_distrib = orb.utils.stats.sigmacut(
        np.partition(
            res_distrib,
            int(BEST_RATIO * np.size(res_distrib)))[
            :int(BEST_RATIO * np.size(res_distrib))], sigma=2.5)
    res_map_mask = np.ones_like(res_map, dtype=np.bool)
    res_map_mask[np.nonzero(
        res_map > orb.utils.stats.robust_median(best_res_distrib))] = 0


    print "Number of well fitted phase vectors used to compute phase coefficients: %d"%len(np.nonzero(res_map_mask)[0])

    phase_coeffs = list()
    order = 1
    for phase_map in phase_maps:
        # Only the pixels with a good residual coefficient are used 
        clean_phase_map = phase_map[np.nonzero(res_map_mask)]
        median_coeff = np.median(clean_phase_map)
        std_coeff = np.std(clean_phase_map)

        # phase map is sigma filtered to remove bad pixels
        clean_phase_map = [coeff for coeff in clean_phase_map
                           if ((coeff < median_coeff + 2. * std_coeff)
                               and (coeff > median_coeff - 2.* std_coeff))]

        phase_coeffs.append(np.mean(clean_phase_map))
        print "Computed phase coefficient of order %d: %f (std: %f)"%(order, np.mean(clean_phase_map), np.std(clean_phase_map))
        
        if np.std(clean_phase_map) >= abs(np.mean(clean_phase_map)):
            warnings.warn("Phase map standard deviation (%f) is greater than its mean value (%f) : the returned coefficient is not well determined and phase correction might be uncorrect"%(np.std(clean_phase_map), np.mean(clean_phase_map)))
        order += 1

    return phase_coeffs
    
def transform_interferogram(interf, nm_laser, 
                            calibration_coeff, step, order, 
                            window_type, zpd_shift, phase_correction=True,
                            return_phase=False, ext_phase=None,
                            weights=None, polyfit_deg=1,
                            balanced=True, bad_frames_vector=None,
                            smoothing_deg=2, return_complex=False,
                            final_step_nb=None, wavenumber=False,
                            low_order_correction=False,
                            conserve_energy=False):
    
    """Transform an interferogram into a spectrum.
    
    :param interf: Interferogram to transform.
    
    :param nm_laser: Wavelength of the laser used for calibration.
    
    :param calibration_coeff: Wavelength of the laser emission line
      corresponding to the computed interferogram.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order (if 0 the result cannot be projected
      on an axis in nm, i.e. wavenumber option is automatically set to
      True).

    :param window_type: Name of the apodization function.

    :param zpd_shift: Shift of the interferogram to center the ZPD.

    :param bad_frames_vector: (Optional) Mask-like vector containing
      ones for bad frames. Bad frames are replaced by zeros using a
      special function that smoothes transition between good parts and
      zeros (default None). This vector must be uncorrected for ZPD
      shift

    :param phase_correction: (Optional) If False, no phase correction will
      be done and the resulting spectrum will be the absolute value of the
      complex spectrum (default True).

    :param ext_phase: (Optional) External phase vector. If given this
      phase vector is used instead of a low-resolution one. It must be
      as long as the interferogram.
      
    :param return_phase: (Optional) If True, compute only the phase of
      the interferogram and return it. If polyfit_deg is >= 0, return
      the coefficients of the fitted phase (default False). Note that
      this option is not compatible with ext_phase. You must set
      ext_phase to None to set return_phase to True.

    :param weights: (Optional) A vector of the same length as the
      interferogram giving the weight of each point for interpolation
      (Must be a float between 0. and 1.). If none is given, the
      weights are defined by the amplitude of the vector.

    :param polyfit_deg: (Optional) Degree of the polynomial fit to the
      computed phase. If < 0, no fit will be performed (Default 1).

    :param smoothing_deg: (Optional) Degree of zeros smoothing. A
      higher degree means a smoother transition from zeros parts (bad
      frames) to non-zero parts (good frames) of the
      interferogram. Good parts on the other side of the ZPD in
      symmetry with zeros parts are multiplied by 2. The same
      transition is used to multiply interferogram points by zero and
      2 (default 2). This operation is not done if smoothing_deg is
      set to 0.

    :param balanced: (Optional) If False, the interferogram is
      considered as unbalanced. It is flipped before its
      transformation to get a positive spectrum. Note
      that a merged interferogram is balanced (default True).

    :param return_complex: (Optional) If True and if phase is
      corrected the returned spectrum will be complex. In False only
      the real part is returned (default False)

    :param final_step_nb: (Optional) Number of samples of the
      resulting spectrum. If None, the number of samples of the
      spectrum will be the same as the interferogram (default None).

    :param wavenumber: (Optional) If True, the returned spectrum is
      projected onto its original wavenumber axis (emission lines and
      especially unapodized sinc emission lines are thus symetric
      which is not the case if the spectrum is projected onto a, more
      convenient, regular wavelength axis) (default False).

    :param low_order_correction: (Optional) If True substract a low
      order polynomial to remove low frequency noise. Useful for
      unperfectly corrected interferograms (default False).

    :param conserve_energy: (Optional) If True the energy is conserved
      in the transformation (default False).

    .. note:: Interferogram can be complex
    """
    MIN_ZEROS_LENGTH = 8 # Minimum length of a zeros band to smooth it
    interf = np.copy(interf)
    interf_orig = np.copy(interf)
    if order == 0 and not wavenumber:
        warnings.warn("order 0: Wavenumber output automatically set to True. Please set manually wavenumber option to True ifyou don't want this warning message to be printed.")
        wavenumber = True
   
    if return_phase and phase_correction:
        raise Exception("phase correction and return_phase cannot be all set to True")
    if return_phase and ext_phase is not None:
        raise Exception("return_phase=True and ext_phase != None options are not compatible. Set the phase or get it !")
    
    dimz = interf.shape[0]

    if final_step_nb is None:
        final_step_nb = dimz
    
    # discard zeros interferogram
    if len(np.nonzero(interf)[0]) == 0:
        if return_phase:
            return None
        else:
            return interf

    # discard interferograms with a bad phase vector
    if ext_phase is not None:
        if np.any(np.isnan(ext_phase)):
            return None

    if conserve_energy:
        interf_energy = interf_mean_energy(interf)

    # replace NaN and Inf values by zeros
    interf[np.nonzero(np.isnan(interf))] = 0.
    interf[np.nonzero(np.isinf(interf))] = 0.

    # reverse unbalanced vector
    if not balanced:
        interf = -interf
   
    #####
    # 1 - substraction of the mean of the interferogram where the
    # interferogram is not 0
    nonzero_pix = np.nonzero(interf != 0.)
    if len(nonzero_pix[0])>0:
        interf[nonzero_pix] -= np.mean(interf[nonzero_pix])
        
    #####
    # 2 - low order polynomial substraction to suppress 
    # low frequency noise
    if low_order_correction:
        interf[nonzero_pix] -= orb.utils.vector.polyfit1d(
            interf, 3)[nonzero_pix]
    
    #####
    # 3 - ZPD shift to center the spectrum
    if zpd_shift != 0:
        temp_vector = np.zeros(interf.shape[0] + 2 * abs(zpd_shift),
                               dtype=interf.dtype)
        temp_vector[abs(zpd_shift):abs(zpd_shift) + dimz] = interf
        interf = np.copy(temp_vector)
        interf = np.roll(interf, zpd_shift)
        
        if bad_frames_vector is not None:
            if np.any(bad_frames_vector > 0):
                temp_vector[
                    abs(zpd_shift):abs(zpd_shift) + dimz] = bad_frames_vector
                bad_frames_vector = np.copy(temp_vector)
                bad_frames_vector = np.roll(bad_frames_vector, zpd_shift)
            else:
                bad_frames_vector = None
    
    #####
    # 4 - Zeros smoothing
    #
    # Smooth the transition between good parts and 'zeros' parts. We
    # use here a concept from Learner et al. (1995) Journal of the
    # Optical Society of America A, 12(10), 2165
    zeros_vector = np.ones_like(interf)
    zeros_vector[np.nonzero(interf == 0)] = 0
    zeros_vector = zeros_vector.real # in case interf is complex
    if bad_frames_vector is not None:
        zeros_vector[np.nonzero(bad_frames_vector)] = 0
    if len(np.nonzero(zeros_vector == 0)[0]) > 0:
        # correct only 'bands' of zeros:
        zcounts = count_nonzeros(-zeros_vector + 1)
        zeros_mask = np.nonzero(zcounts >= MIN_ZEROS_LENGTH)
        if len(zeros_mask[0]) > 0 and smoothing_deg > 0.:
            for izero in zeros_mask[0]:
                if (izero > smoothing_deg
                    and izero < interf.shape[0] - 1 - smoothing_deg):
                    zeros_vector[izero - smoothing_deg:
                                 izero + smoothing_deg + 1] = 0
            zeros_vector = orb.utils.vector.smooth(
                np.copy(zeros_vector), deg=smoothing_deg,
                kind='cos_conv')
            zeros_vector = zeros_vector * (- zeros_vector[::-1] + 2)
            interf *= zeros_vector
    

    #####
    # 5 - Phase determination (Mertz method)
    #
    # We use the method described by Learner et al. (1995) Journal of
    # the Optical Society of America A, 12(10), 2165
    #
    # The low resolution interferogram is a small part of the real
    # interferogram taken symmetrically around ZPD
    if phase_correction:
        if ext_phase is None:
            
            lr_phase2, lr_spectrum2 = get_lr_phase(interf, n_phase=dimz,
                                                   return_lr_spectrum=True)
            lr_spectrum = transform_interferogram(
                interf_orig, 1, 1, step, order, 'learner95', zpd_shift,
                phase_correction=False, return_phase=False, polyfit_deg=-1,
                wavenumber=True, ext_phase=None, final_step_nb=dimz,
                return_complex=True)
            if int(order) & 1: # must be inversed again if order is
                               # even because the output is inversed
                lr_spectrum = lr_spectrum[::-1]
                    
            lr_phase = np.unwrap(np.angle(lr_spectrum))
        
            # fit
            if polyfit_deg >= 0:
                # polynomial fitting must be weigthed in case of a spectrum
                # without enough continuum.
                if weights is None or not np.any(weights):
                    weights = np.abs(lr_spectrum)
                    # suppress noise on spectrum borders
                    weights *= border_cut_window(lr_spectrum.shape[0])
                    if np.max(weights) != 0.:
                        weights /= np.max(weights)
                    else:
                        weights = np.ones_like(lr_spectrum)
                    # remove parts with a bad signal to noise ratio
                    weights[np.nonzero(weights < 0.25)] = 0.
                else:
                    if weights.shape[0] != lr_phase.shape[0]:
                        weights = orb.utils.vector.interpolate_size(
                            weights, lr_phase.shape[0], 1)

                lr_phase, lr_phase_coeffs = orb.utils.vector.polyfit1d(
                    lr_phase, polyfit_deg,
                    w=weights, return_coeffs=True)
            
        else:
            lr_phase = ext_phase

    #####
    # 6 - Apodization of the real interferogram
    if window_type is not None and window_type != '1.0':
        if window_type in ['1.1', '1.2', '1.3', '1.4', '1.5',
                           '1.6', '1.7', '1.8', '1.9', '2.0']:
            window = norton_beer_window(window_type, interf.shape[0])
        elif window_type == 'learner95':
            window = learner95_window(interf.shape[0])
        else:
            window = signal.get_window((window_type), interf.shape[0])
            
        interf *= window

    #####
    # 7 - Zero padding
    #
    # Define the size of the zero padded vector to have at
    # least 2 times more points than the initial vector to
    # compute its FFT. FFT computation is faster for a vector
    # size equal to a power of 2.
    #zero_padded_size = next_power_of_two(2*final_step_nb)
    zero_padded_size = 2 * final_step_nb
    
    temp_vector = np.zeros(zero_padded_size, dtype=interf.dtype)
    zeros_border = int((zero_padded_size - interf.shape[0]) / 2.)
    temp_vector[zeros_border:(zeros_border + interf.shape[0])] = interf
    zero_padded_vector = temp_vector

    #####
    # 8 - Zero the centerburst
    zero_padded_vector = np.roll(zero_padded_vector,
                                 zero_padded_vector.shape[0]/2)
    
    #####
    # 9 - Fast Fourier Transform of the interferogram
    center = zero_padded_size / 2
    interf_fft = np.fft.fft(zero_padded_vector)[:center]

    # normalization of the vector to take into account zero-padding 
    if np.iscomplexobj(interf):
        interf_fft /= (zero_padded_size / dimz)
    else:
        interf_fft /= (zero_padded_size / dimz) / 2.
        
    #####
    # 10 - Phase correction
    if phase_correction:
        lr_phase = orb.utils.vector.interpolate_size(
            lr_phase, interf_fft.shape[0], 1)
        spectrum_corr = np.empty_like(interf_fft)
        spectrum_corr.real = (interf_fft.real * np.cos(lr_phase)
                              + interf_fft.imag * np.sin(lr_phase))
        spectrum_corr.imag = (interf_fft.imag * np.cos(lr_phase)
                              - interf_fft.real * np.sin(lr_phase))

    else:
        spectrum_corr = interf_fft
        
    #####
    # 11 - Off-axis effect correction with maxima map   
    # Irregular wavelength axis creation
    correction_coeff = float(calibration_coeff) / nm_laser
    if not wavenumber:
        base_axis = orb.utils.spectrum.create_nm_axis_ireg(
            spectrum_corr.shape[0], step, order,
            corr=correction_coeff)
    else:
        base_axis = orb.utils.spectrum.create_cm1_axis(
            spectrum_corr.shape[0], step, order, corr=correction_coeff)
    
    # Spectrum is returned if folding order is even
    if int(order) & 1:
        spectrum_corr = spectrum_corr[::-1]

    # Interpolation (order 5) of the spectrum from its irregular axis
    # to the regular one
    
    # regular axis creation (in nm, if step is in nm)
    if not wavenumber:
        final_axis = orb.utils.spectrum.create_nm_axis(
            final_step_nb, step, order)
    else:
        final_axis = orb.utils.spectrum.create_cm1_axis(
            final_step_nb, step, order, corr=1.)
 
    # spectrum interpolation
    if not (wavenumber and correction_coeff == 1.):
        spectrum = orb.utils.vector.interpolate_axis(
            spectrum_corr, final_axis, 5, old_axis=base_axis)
    else:
        spectrum = spectrum_corr

    # Extrapolated parts of the spectrum are set to NaN
    spectrum[np.nonzero(final_axis > np.max(base_axis))] = np.nan
    spectrum[np.nonzero(final_axis < np.min(base_axis))] = np.nan

    if conserve_energy:
        # Spectrum is rescaled to the modulation energy of the interferogram
        spectrum = spectrum / spectrum_mean_energy(spectrum) * interf_energy

    if return_phase:
        return np.copy(np.unwrap(np.angle(spectrum)))
    elif return_complex:
        return np.copy(spectrum)
    else:
        if phase_correction:
            return np.copy(spectrum.real)
        else:    
            return np.copy(np.abs(spectrum))


def transform_spectrum(spectrum, nm_laser, calibration_coeff,
                       step, order, window_type, zpd_shift,
                       ext_phase=None, return_complex=False, wavenumber=False,
                       final_step_nb=None, sampling_vector=None,
                       zero_padding=False):
    """Transform a spectrum into an interferogram.

    This function is the inverse of :py:meth:`utils.transform_interferogram`.

    So that to get the initial interferogram, the same options used in
    transform interferogram must be passed to this function. The
    spectrum must also be the complex form (use return_complex option
    in :py:meth:`utils.transform_interferogram`)

    :param spectrum: Spectrum to transform

    :param nm_laser: Wavelength of the laser used for calibration.
    
    :param calibration_coeff: Wavelength of the laser emission line
      corresponding to the computed interferogram.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order (can be 0 but the input must be in
      wavenumber).

    :param window_type: Name of the apodization function.

    :param zpd_shift: Shift of the interferogram to decenter the ZPD.
      
    :param ext_phase: (Optional) External phase vector. If given this
      phase vector is used in place of the original phase of the
      spectrum. Useful to add a phase to an interferogram. Note that
      this phase is intended to be used to inverse transform an
      already transformed interferogram. The computed phase correction
      can thus be used directly. As the phase vector given by
      :py:meth:`utils.transform_interferogram` is not reversed for
      even orders, it is reversed here in this function.

    :param return_complex: (Optional) If True return a complex
      interferogram. Else return the real part of it (default False).

    :param wavenumber: (Optional) If True the spectrum axis is in
      cm-1. In this case, and if no wavelength correction has to be
      applied (calibration_coeff == nm_laser) there will be no
      interpolation of the original spectrum (better precision)
      (default False).

    :param final_step_nb: (Optional) Final size of the
      interferogram. Must be less than the size of the original
      spectrum. If None the final size of the interferogram is the
      same as the size of the original spectrum (default None).

    :sampling_vector: (Optional) If the samples of the interferogram
      are not uniformly distributed, a vector giving the positions of
      the samples can be passed. In this case an inverse NDFT is
      computed which may be really slow for long vectors. A uniformly
      sampled vector would be range(final_step_nb). The size of the
      vector must be equal to final_step_nb (default None).

    :zero_padding: (Optional) If True and if final_step_nb is >
      spectrum step number, ouput is zero padded. Can be used to
      compare a high resolution interferogram to a low resolution
      interferogram.
    
    .. note:: Interferogram can be complex
    """
    if order ==0 and not wavenumber:
        wavenumber = True
        warnings.warn("Order 0: spectrum input automatically set to wavenumber. Please set manually wavenumber option to True if you don't want this warning message to be printed.")
    spectrum = np.copy(spectrum)
    spectrum = spectrum.astype(np.complex)
    step_nb = spectrum.shape[0]
    if final_step_nb is not None:
        if final_step_nb > step_nb:
            raise Exception('final_step_nb must be less than the size of the original spectrum')
    else:
        final_step_nb = step_nb

    # On-axis -> Off-axis [nm - > cm-1]
    correction_coeff = calibration_coeff / nm_laser
    
    if not wavenumber:
        base_axis = orb.utils.spectrum.create_nm_axis(
            step_nb, step, order)
    else:
        base_axis = orb.utils.spectrum.create_nm_axis_ireg(
            step_nb, step, order,corr=1.)
    
    nm_axis_ireg = orb.utils.spectrum.create_nm_axis_ireg(
        step_nb, step, order, corr=correction_coeff)
    if not (wavenumber and correction_coeff == 1.):
        spectrum = orb.utils.vector.interpolate_axis(
            spectrum, nm_axis_ireg[::-1], 5,
            old_axis=base_axis, fill_value=0.)
    else:
        spectrum = spectrum[::-1]
    
    # Add phase to the spectrum (Re-phase)
    if ext_phase is not None:
        if np.any(ext_phase != 0.):
            if not order&1:
                ext_phase = ext_phase[::-1]
            ext_phase = orb.utils.vector.interpolate_size(ext_phase, step_nb, 5)
            spectrum_real = np.copy(spectrum.real)
            spectrum_imag = np.copy(spectrum.imag)
            spectrum.real = (spectrum_real * np.cos(ext_phase)
                             - spectrum_imag * np.sin(ext_phase))
            spectrum.imag = (spectrum_real * np.sin(ext_phase)
                             + spectrum_imag * np.cos(ext_phase))

    
    # Zero-filling
    zeros_spectrum = np.zeros(step_nb * 2, dtype=spectrum.dtype)
    if order&1:
        zeros_spectrum[:spectrum.shape[0]] += spectrum
    else:
        zeros_spectrum[:spectrum.shape[0]] += spectrum[::-1]
    spectrum = zeros_spectrum

    # IFFT and re-shift + center burst
    if sampling_vector is None:
        interf = np.fft.ifft(spectrum)
        interf = np.roll(interf, step_nb - zpd_shift)
        interf = interf[
            step_nb-(final_step_nb/2) - final_step_nb%2:
            step_nb+(final_step_nb/2)]
        
    else:
        if sampling_vector.shape[0] != final_step_nb: raise Exception(
            'Sampling vector size must be equal to the final_step_nb')
        interf = indft(spectrum.real, sampling_vector)
    
    interf = np.array(interf)

    # De-apodize
    if window_type is not None:
        window = norton_beer_window(window_type, final_step_nb)
        interf /= window

    # Normalization to remove zero filling effect on the mean energy
    interf *= step_nb / float(final_step_nb) * 2.

    # Zero-padding of the output
    if zero_padding and final_step_nb < step_nb:
        print interf.shape, final_step_nb, step_nb
        zp_interf = np.zeros(step_nb, dtype=complex)
        zp_interf[
            int(math.ceil(step_nb/2.))-(final_step_nb/2) - final_step_nb%2:
            int(math.ceil(step_nb/2.))+(final_step_nb/2)] = interf
        interf = zp_interf
        
    if return_complex:
        return interf
    else:
        return interf.real

def indft(a, x):
    """Inverse Non-uniform Discret Fourier Transform.

    Compute the irregularly sampled interferogram from a regularly
    sampled spectrum.

    :param a: regularly sampled spectrum.
    
    :param x: positions of the interferogram samples. If x =
      range(size(a)), this function is equivalent to an idft or a
      ifft. Note that the ifft is of course much faster to
      compute. This vector may have any length.
    """
    return orb.cutils.indft(a.astype(float), x.astype(float))

def spectrum_mean_energy(spectrum):
    """Return the mean energy of a spectrum by channel.

    :param spectrum: a 1D spectrum
    """
    return orb.cutils.spectrum_mean_energy(spectrum)


def interf_mean_energy(interf):
    """Return the mean energy of an interferogram by step.

    :param interf: an interferogram

    .. warning:: The mean of the interferogram is substracted to
      compute only the modulation energy. This is the modulation
      energy which must be conserved in the resulting spectrum. Note
      that the interferogram transformation function (see
      :py:meth:`utils.transform_interferogram`) remove the mean of the
      interferogram before computing its FFT.

    .. note:: NaNs are set to 0.
    """
    return orb.cutils.interf_mean_energy(interf)
    

def variable_me(n, params):
     """Return a sinusoidal function representing a variable
     modulation efficiency.

     This function is used to correct for fringes.

     :param params: A tuple of floats [frequency, amplitude,
       phase]. The frequency gives the number of repetition of a sinus
       over the vector. The amplitude must be between 0. (returns a
       vector of 1) and 1. (returns a sinus going from 0 to 1). Phase
       can be a single float or a vector of size n
     """
     f = params[0]
     a = params[1]
     phi = params[2]
     me_real = np.cos(np.arange(n, dtype=float)
                      / float(n - 1.) * 2. * math.pi * f)
     me_imag = np.sin(np.arange(n, dtype=float)
                      / float(n - 1.) * 2. * math.pi * f)
     me = np.empty_like(me_real, dtype=complex)
     me.real = (me_real * np.cos(phi) - me_imag * np.sin(phi)) * a + (1. - a)
     me.imag = (me_imag * np.cos(phi) + me_real * np.sin(phi)) * a
     return me

def find_phase_coeffs_brute_force(interf, step, order, zpd_shift,
                                  div_nb=400):
    """Return phase coefficients (order0, order1) based on a brute
    force method which finds where the standard deviation of the
    imaginary part is minimized.

    Especially useful to get a good guess on the order 1.

    :param interf: Interferogram
    
    :param step: Step size in nm.
    
    :param order: Folding order.
    
    :param zpd_shift: ZPD shift.
    
    :param div_nb: (Optional) Number of division along each axis of
      the search matrix (the more divisions, the more precise will be
      the result) (default 400).
    """

    x0 = np.linspace(-math.pi/4., math.pi/4., div_nb)
    x1 = np.linspace(-math.pi/2., math.pi/2., div_nb)
    
    matrix = np.empty((div_nb, div_nb), dtype=float)
    matrix.fill(np.nan)
    print '> Searching phase coefficients by brute force'
    for i0 in range(div_nb):
        sys.stdout.write('\r {}/{}'.format(i0, div_nb))
        sys.stdout.flush()
        for i1 in range(div_nb):
            ext_phase = np.polyval([x1[i1], x0[i0]], np.arange(np.size(interf)))
            a_fft = transform_interferogram(
                interf, 1., 1., step, order, '2.0', zpd_shift,
                wavenumber=True,
                ext_phase=ext_phase,
                return_complex=True,
                phase_correction=True)
            matrix[i0,i1] = np.nanstd(a_fft.imag)

    matrix_min = np.unravel_index(np.argmin(matrix), matrix.shape)
    a0 = x0[matrix_min[0]]
    a1 = x1[matrix_min[1]]
    print ' > order0: {}, order1: {}'.format(a0, a1)

    return a0, a1

def optimize_phase(interf, step, order, zpd_shift,
                   guess=[0,0], return_coeffs=False,
                   fixed_params=[0, 0]):
    """Return an optimized phase vector based on the minimization of
    the imaginary part.

    :param interf: Interferogram
    :param step: Step size (in nm)
    :param order: Alisasing order
    :param zpd_shift: ZPD shift
    
    :param return_coeffs: (Optional) If True, coeffs and residual are
      returned instead of the phase vector (default False).
      
    """
    def diff(vp, interf, step, order, zpd_shift, fp, findex):
        p = np.empty_like(findex, dtype=float)
        p[np.nonzero(findex==0)] = vp
        p[np.nonzero(findex)] = fp
        ext_phase = np.polyval(p, np.arange(np.size(interf)))
        a_fft = transform_interferogram(
            interf, 1., 1., step, order, '2.0', zpd_shift,
            wavenumber=True,
            ext_phase=ext_phase,
            return_complex=True)
        return a_fft.imag

    guess = np.array(guess, dtype=float)
    fixed_params = np.array(fixed_params, dtype=bool)
    vguess = guess[np.nonzero(fixed_params==0)]
    fguess = guess[np.nonzero(fixed_params)]
            
    optim = scipy.optimize.leastsq(
        diff, vguess, args=(
            interf, step, order, zpd_shift, fguess, fixed_params),
        full_output=True)
    
    if optim[-1] < 5:
        p = np.empty_like(fixed_params, dtype=float)
        p[np.nonzero(fixed_params==0)] = optim[0]
        p[np.nonzero(fixed_params)] = guess[np.nonzero(fixed_params)]
        res = (np.sqrt(np.nanmean(optim[2]['fvec']**2.))
               /interf_mean_energy(interf))
        if not return_coeffs:
            return np.polyval(p, np.arange(np.size(interf)))
        else:
            return p, res
    else: return None


def create_phase_vector(order_0, other_orders, n):
    """Create a phase vector.

    :param order_0: Order 0 coefficient
    
    :param other_orders: Other orders coefficients (in ascending
      order).

    :param n: Length of the phase vector
    """
    coeffs_list = list()
    coeffs_list.append(order_0)
    coeffs_list += other_orders
    return np.polynomial.polynomial.polyval(
        np.arange(n), coeffs_list)
    
    
