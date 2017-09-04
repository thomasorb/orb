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

import time
import sys
import numpy as np
import math
import warnings
import scipy
import scipy.special as ss
from scipy import signal, interpolate, optimize
import gvar

import orb.utils.vector
import orb.utils.spectrum
import orb.utils.stats
import orb.utils.filters
import orb.cutils
import orb.constants

def apodize(s, apodization_function=2.0):
    """Apodize a spectrum

    :param s: Spectrum
    
    :param apodization_function: (Optional) A Norton-Beer apodization
      function (default 2.0)
    """
    s_ifft = np.fft.ifft(s)
    w = gaussian_window(apodization_function, s.shape[0])
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
    full_interf *= gaussian_window(1.5, dimz)
    
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
        x *= gaussian_window(apod, N)
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
      :py:meth:`utils.gaussian_window` (default None)
    """
    x = np.copy(x)
    N = x.shape[-1]
    # mean substraction
    x = (x.T - np.mean(x, axis=-1)).T
    # apodization
    if apod is not None:
        x *= gaussian_window(apod, N)

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

def apod2width(apod):
    """Return the width of the gaussian window for a given apodization level.

    :param apod: Apodization level (must be >= 1.)

    The apodization level is the broadening factor of the line (an
    apodization level of 2 mean that the line fwhm will be 2 times
    wider).
    """
    if apod < 1.: raise Exception(
        'Apodization level (broadening factor) must be > 1')

    return apod - 1. + (gvar.erf(math.pi / 2. * gvar.sqrt(apod - 1.))
                        * orb.constants.FWHM_SINC_COEFF)

def apod2sigma(apod, fwhm):
    """Return the broadening of the gaussian-sinc function in the
    spectrum for a given apodization level. Unit is that of the fwhm.

    :param apod: Apodization level (must be >= 1.)
    """
    broadening = 2. * (apod2width(apod) / (math.sqrt(2.) * math.pi)
                       / orb.utils.spectrum.compute_line_fwhm_pix(
                           oversampling_ratio=1))

    return broadening * fwhm

def gaussian_window(coeff, n):
    """Return a Gaussian apodization function for a given broadening
    factor.

    :param coeff: FWHM relative to the sinc function. Must be a float > 1.
       
    :param n: Number of points.
    """
    coeff = float(coeff)
    x = np.linspace(-1., 1., n)
    w = apod2width(coeff)
    return np.exp(-x**2 * w**2)

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
        print "Computed phase coefficient: %f (std: %f)"%(np.mean(clean_phase_map), np.std(clean_phase_map))
        
        if np.std(clean_phase_map) >= abs(np.mean(clean_phase_map)):
            warnings.warn("Phase map standard deviation (%f) is greater than its mean value (%f) : the returned coefficient is not well determined and phase correction might be uncorrect"%(np.std(clean_phase_map), np.mean(clean_phase_map)))
        order += 1

    return phase_coeffs
    
def transform_interferogram(interf, nm_laser, 
                            calibration_nm_laser, step, order, 
                            window_type, zpd_shift, phase_correction=True,
                            wave_calibration=True,
                            return_phase=False, ext_phase=None,
                            balanced=True, bad_frames_vector=None,
                            smoothing_coeff=0.04, return_complex=False,
                            final_step_nb=None, wavenumber=False,
                            low_order_correction=False,
                            high_order_phase=None,
                            return_zp_vector=False,
                            sampling_steps=None):
    
    """Transform an interferogram into a spectrum.
    
    :param interf: Interferogram to transform.
    
    :param nm_laser: Wavelength of the laser used for calibration.
    
    :param calibration_nm_laser: Wavelength of the laser emission line
      corresponding to the computed interferogram.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order (if 0 the result cannot be projected
      on an axis in nm, i.e. wavenumber option is automatically set to
      True).

    :param window_type: Name of the apodization function (can be
      learner95 or a float > 1.).

    :param zpd_shift: Shift of the interferogram to center the ZPD.

    :param bad_frames_vector: (Optional) Mask-like vector containing
      ones for bad frames. Bad frames are replaced by zeros using a
      special function that smoothes transition between good parts and
      zeros (default None). This vector must be uncorrected for ZPD
      shift

    :param phase_correction: (Optional) If False, no phase correction
      will be done and the resulting spectrum will be the absolute
      value of the complex spectrum. Else the ext_phase vector will be
      used for phase correction. If ext_phase is set to None,
      ext_phase will be replaced by a vector of 0 (default True).

    :param wave_calibration: (Optional) If True wavenumber/wavelength
      calibration is done (default True).

    :param ext_phase: (Optional) External phase vector. If given this
      phase vector is used instead of a low-resolution one. It must be
      as long as the interferogram.
      
    :param return_phase: (Optional) If True, compute only the phase of
      the interferogram and return it. If polyfit_deg is >= 0, return
      the coefficients of the fitted phase (default False). Note that
      this option is not compatible with ext_phase. You must set
      ext_phase to None to set return_phase to True.

    :param smoothing_coeff: (Optional) Coefficient of zeros smoothing
      in proportion of the total interferogram size. A higher
      coefficient means a smoother transition from zeros parts (bad
      frames) to non-zero parts (good frames) of the
      interferogram. Good parts on the other side of the ZPD in
      symmetry with zeros parts are multiplied by 2 to keep a constant
      amount of energy. The same transition is used to multiply
      interferogram points by zero and 2. This operation is not done
      if smoothing_coeff is set to 0. must be between 0. and 0.2
      (default 0.04).

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

    :param high_order_phase: (Optional) High order phase to be added
      to the phase computed via a low order polynomial (generally 1
      order). Note that it must be a orb.core.PhaseFile instance or a
      scipy.interpolate.UnivariateSpline instance to accelerate the
      process.

    :param sampling_steps: (Optional) If the sampling steps are not
      uniform, the real sampling function can be given. It must have
      the exact same size as the interferogram. Note that a NDFT will
      be performed which is much slower than a FFT.

    .. note:: Interferogram can be complex

    .. note:: Only NANs or INFs are interpreted as bad values
    
    """
    interf = np.copy(interf)
    interf_orig = np.copy(interf)
        
    if order == 0 and not wavenumber:
        warnings.warn("order 0: Wavenumber output automatically set to True. Please set manually wavenumber option to True if you don't want to see this warning message.")
        wavenumber = True
   
    if return_phase and phase_correction:
        raise ValueError("phase correction and return_phase cannot be all set to True")
    
    if return_phase and ext_phase is not None:
        raise ValueError("return_phase=True and ext_phase != None options are not compatible. Set the phase or get it !")

    if (smoothing_coeff < 0. or smoothing_coeff > 0.2):
        raise ValueError('smoothing coeff must be between 0. and 0.2')

    if sampling_steps is not None:
        sampling_steps = np.array(sampling_steps)
        assert sampling_steps.size == interf.size, 'sampling_steps must have the same size as interf'

    dimz = interf.shape[0]

    smoothing_deg = int(math.ceil(dimz * smoothing_coeff))
    min_zeros_length = smoothing_deg * 2 # Minimum length of a zeros band to smooth it    

    if final_step_nb is None:
        final_step_nb = dimz

    # discard zeros and nans interferogram
    if len(np.nonzero(interf)[0]) == 0:
        if return_phase:
            return None
        else:
            return interf

    if np.all(np.isnan(interf)): return interf*np.nan

    # discard interferograms with a bad phase vector
    if ext_phase is not None:
        if np.any(np.isnan(ext_phase)):
            return None

    # replace Inf by Nan
    interf[np.nonzero(np.isinf(interf))] = np.nan

    # reverse unbalanced vector
    if not balanced:
        interf = -interf

    #####
    # 1 - substraction of the mean of the interferogram where the
    # interferogram is not nan
    interf[~np.isnan(interf)] -= np.nanmean(interf)
        
    #####
    # 2 - low order polynomial substraction to suppress 
    # low frequency noise
    if low_order_correction:
        interf[~np.isnan(interf)] -= orb.utils.vector.polyfit1d(
            interf, 3)[~np.isnan(interf)]
        
    #####
    # 3 - ZPD shift to center the spectrum
    
    # Check phase vector to guess a better ZPD shift
    if ext_phase is not None:
        order1 = np.median(np.diff(ext_phase)) * ext_phase.shape[0]
        phase_shift = int(round(order1 / np.pi))
        ext_phase_corr = (ext_phase - np.arange(ext_phase.shape[0])
                          * phase_shift * np.pi / ext_phase.shape[0])
        zpd_shift_corr = int(zpd_shift) + phase_shift
    else:
        ext_phase_corr = None
        zpd_shift_corr = int(zpd_shift)

    if zpd_shift_corr != 0:
        if zpd_shift_corr > interf.shape[0] / 2 + 1:
            raise Exception('Bad zpd shift (must be <= {})'.format(interf.shape[0]/2 + 1))
        temp_vector = np.zeros(interf.shape[0] + 2 * abs(zpd_shift_corr),
                               dtype=interf.dtype)
        temp_vector[abs(zpd_shift_corr):abs(zpd_shift_corr) + dimz] = interf
        interf = np.copy(temp_vector)
        interf = np.roll(interf, zpd_shift_corr)            
        if bad_frames_vector is not None or np.size(bad_frames_vector) > 0:
            warnings.warn('bad frames handling not implemented')
            if np.any(bad_frames_vector > 0):
                temp_vector[
                    abs(zpd_shift_corr):abs(zpd_shift_corr) + dimz] = bad_frames_vector
                bad_frames_vector = np.copy(temp_vector)
                bad_frames_vector = np.roll(bad_frames_vector, zpd_shift_corr)
            else:
                bad_frames_vector = None
                
    ### Replace Nans by zeros in interf vector
    interf[np.isnan(interf)] = 0.
    

    #####
    # 4 - Ramp-like truncation function from Mertz (1967) Infrared
    # Physics, 7, 17-23
    
    
    # count zeros to detect which side is the truncation
    zpd_pos = interf.shape[0]/2
    
    if interf[0] == 0.:
        left_zeros_nb = np.argmax(np.diff(interf[:zpd_pos]) > 0)
    else: left_zeros_nb = 0
    
    if interf[-1] == 0.:
        right_zeros_nb = np.argmax(np.diff(interf[zpd_pos:][::-1]) > 0)
    else: right_zeros_nb = 0

    # create ramp
    start_pos = interf.shape[0] - dimz
    sym_len = abs(zpd_pos - start_pos) * 2
    
    zeros_vector = np.zeros_like(interf)
    zeros_vector[start_pos:start_pos+sym_len] = np.linspace(0,2,sym_len)
    zeros_vector[start_pos+sym_len:] = 2.

    if left_zeros_nb < right_zeros_nb:
        zeros_vector = zeros_vector[::-1]
        
    interf *= zeros_vector

    #####
    # 5 - Apodization of the real interferogram
    if window_type is not None and window_type != '1.0':
        if window_type == 'learner95':
            window = learner95_window(interf.shape[0])
        else:
            window = gaussian_window(window_type, interf.shape[0])
            
        interf *= window

    #####
    # 6 - Zero padding
    #
    # Define the size of the zero padded vector to have at
    # least 2 times more points than the initial vector to
    # compute its FFT. FFT computation is faster for a vector
    # size equal to a power of 2. ???
    #zero_padded_size = next_power_of_two(2*final_step_nb)
    zero_padded_size = 2 * final_step_nb
    
    temp_vector = np.zeros(zero_padded_size, dtype=interf.dtype)
    zeros_border = int((zero_padded_size - interf.shape[0]) / 2.)
    temp_vector[zeros_border:(zeros_border + interf.shape[0])] = interf
    zero_padded_vector = temp_vector

    #####
    # 7 - ZPD rolled at the beginning of the interferogram
    zero_padded_vector = np.roll(zero_padded_vector,
                                 zero_padded_vector.shape[0]/2)
    
    #####
    # 8 - Fast Fourier Transform of the interferogram
    center = zero_padded_size / 2
    # spectrum is cropped at zero_padded_size / 2 instead of
    # zero_padded_size / 2 + 1 which would output a spectrum with 1
    # more sample than the input length. Computed axis must be
    # cropped accordingly.
    if return_zp_vector:
        return zero_padded_vector

    if sampling_steps is None:
        interf_fft = np.fft.fft(zero_padded_vector)[:center]
    else:
        sampling_steps = np.hstack((sampling_steps,
                                    np.arange(zero_padded_vector.size - sampling_steps.size)
                                    + sampling_steps.size))        
        interf_fft = ndft(zero_padded_vector,
                          sampling_steps,
                          np.arange(zero_padded_vector.size))[:center]
        
    # normalization of the vector to take into account zero-padding
    # and mimic a dispersive instrument: if the same energy is
    # dispersed over more channels (more zeros) then you get less
    # counts/channel
    if np.iscomplexobj(interf):
        interf_fft /= (zero_padded_size / dimz)
    else:
        interf_fft /= (zero_padded_size / dimz) / 2.
                            

    #### Create spectrum original cm-1 axis
    if wave_calibration:
        correction_coeff = float(calibration_nm_laser) / nm_laser
    else:
        correction_coeff = 1.

    if not wavenumber:
        base_axis = orb.utils.spectrum.create_nm_axis_ireg(
            interf_fft.shape[0], step, order,
            corr=correction_coeff)
    else:
        base_axis = orb.utils.spectrum.create_cm1_axis(
            interf_fft.shape[0], step, order, corr=correction_coeff)
        
    #####
    # 9 - Phase correction
    if phase_correction:
        if ext_phase_corr is None:
            ext_phase_corr = np.zeros(dimz, dtype=float)

        if high_order_phase is not None:
            phase_axis = orb.utils.spectrum.create_cm1_axis(
                dimz, step, order,
                corr=float(calibration_nm_laser) / nm_laser).astype(np.float64)
            try: # if high_order_phase is a PhaseFile instance
                high_order_phase = high_order_phase.get_improved_phase(
                    float(calibration_nm_laser))
            except AttributeError: pass
            ext_phase_corr += high_order_phase(phase_axis)

        # interpolation of the phase to zero padded size
        ext_phase_corr = orb.utils.vector.interpolate_size(
            ext_phase_corr, interf_fft.shape[0], 1)

    
        spectrum_corr = np.empty_like(interf_fft)
        spectrum_corr.real = (interf_fft.real * np.cos(ext_phase_corr)
                              + interf_fft.imag * np.sin(ext_phase_corr))
        spectrum_corr.imag = (interf_fft.imag * np.cos(ext_phase_corr)
                              - interf_fft.real * np.sin(ext_phase_corr))
    else:
        spectrum_corr = interf_fft
        
    #####
    # 10 - Off-axis effect correction with maxima map   
    # Irregular wavelength axis creation
    
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
        # Extrapolated parts of the spectrum are set to NaN
        spectrum[np.nonzero(final_axis > np.max(base_axis))] = np.nan
        spectrum[np.nonzero(final_axis < np.min(base_axis))] = np.nan

    else:
        spectrum = spectrum_corr

    
    if return_phase:
        return np.copy(np.unwrap(np.angle(spectrum)))
    elif return_complex:
        return np.copy(spectrum)
    else:
        if phase_correction:
            return np.copy(spectrum.real)
        else:    
            return np.copy(np.abs(spectrum))


def transform_spectrum(spectrum, nm_laser, calibration_nm_laser,
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
    
    :param calibration_nm_laser: Wavelength of the laser emission line
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
      applied (calibration_nm_laser == nm_laser) there will be no
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
    if not isinstance(zpd_shift, int):
        warnings.warn("ZPD shift must be an integer and will be converted")
        zpd_shift = int(zpd_shift)

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
    correction_coeff = calibration_nm_laser / nm_laser
    
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
        interf = np.roll(interf, int(step_nb - zpd_shift))
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
        window = gaussian_window(window_type, final_step_nb)
        interf /= window

    # Normalization to remove zero filling effect on the mean energy
    interf *= step_nb / float(final_step_nb) * 2.

    # Zero-padding of the output
    if zero_padding and final_step_nb < step_nb:
        zp_interf = np.zeros(step_nb, dtype=complex)
        zp_interf[
            int(math.ceil(step_nb/2.))-(final_step_nb/2) - final_step_nb%2:
            int(math.ceil(step_nb/2.))+(final_step_nb/2)] = interf
        interf = zp_interf
        
    if return_complex:
        return interf
    else:
        return interf.real

def ndft(a, xk, vj):
    """Non-uniform Discret Fourier Tranform

    Compute the spectrum from an interferogram. Noth axis can be
    irregularly sampled.

    If the spectral axis (output axis) is irregular the result is
    exact. But there is no magic: if the input axis (interferogram
    sampling) is irregular the output spectrum is not exact because
    the projection basis is not orthonormal.

    If the interferogram is the addition of multiple regularly sampled
    scans with a opd shift between each scan, the result will be good
    as long as there are not too much scans added one after the
    other. But if the interferogram steps are randomly distributed, it
    will be better to use a classic FFT because the resulting noise
    will be much lower.

    :param a: 1D interferogram
    
    :param xk: 1D sampling steps of the interferogram. Must have the
      same size as a and must be relative to the real step length,
      i.e. if the sampling is uniform xk = np.arange(a.size).
    
    :param vj: 1D frequency sampling of the output spectrum.
    """
    assert a.ndim == 1, 'a must be a 1d vector'
    assert vj.ndim == 1, 'vj must be a 1d vector'
    assert a.size == xk.size, 'size of a must equal size of xk'
    
    angle = np.inner((-2.j * np.pi * xk / xk.size)[:,None], vj[:,None])
    return np.dot(a, np.exp(angle))



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


def optimize_phase(interf, step, order, zpd_shift,
                   calib, nm_laser,
                   guess=[0.0001, 0.0001], return_coeffs=False,
                   fixed_params=[0, 0], weights=None,
                   high_order_phase=None):
    """Return an optimized phase vector based on the minimization of
    the imaginary part.

    :param interf: Interferogram
    
    :param step: Step size (in nm)
    
    :param order: Alisasing order
    
    :param zpd_shift: ZPD shift

    :param calib: Calibration laser observed wavelength

    :param nm_laser: Calibration laser real wavelength

    :param guess: (Optional) First guess. The number of values defines the order
      of the polynomial used used to fit (default [0,0]).
    
    :param return_coeffs: (Optional) If True, coeffs and residual are
      returned instead of the phase vector (default False).

    :param fixed_params: (Optional) Define free and fixed parameters
      (1 for fixed, 0 for free, default [0,0])

    :param weights: (Optional) spectrum weighting (a vector with
      values ranging from 0 to 1, 1 being the maximum weight)

    :param high_order_phase: (Optional) High order phase to be
      subtracted during the optimization process. Must be a
      orb.core.PhaseFile instance or a
      scipy.interpolate.UnivariateSpline instance to accelerate the
      process (as returned by
      :py:meth:`orb.utils.fft.read_phase_file`).
    """
    def diff(vp, interf_fft, fp, findex, weights, high_phase):
        p = np.empty_like(findex, dtype=float)
        p[np.nonzero(findex==0)] = vp
        p[np.nonzero(findex)] = fp
        ext_phase = np.polynomial.polynomial.polyval(
            np.arange(np.size(interf)), p)
        if high_phase is not None:
            ext_phase += high_phase
        # phase correction
        a_fft = np.zeros_like(interf_fft)
        ## a_fft.real = (interf_fft.real * np.cos(ext_phase)
        ##               + interf_fft.imag * np.sin(ext_phase))
        a_fft.imag = (interf_fft.imag * np.cos(ext_phase)
                      - interf_fft.real * np.sin(ext_phase))
        return a_fft.imag * weights

    guess = np.array(guess, dtype=float)
    fixed_params = np.array(fixed_params, dtype=bool)
    vguess = guess[np.nonzero(fixed_params==0)]
    fguess = guess[np.nonzero(fixed_params)]

    if weights is None:
        weights = np.ones(interf.shape[0], dtype=float)
    
    interf_fft = transform_interferogram(
        interf, 1., 1., step, order, '2.0', zpd_shift,
        wavenumber=True,
        ext_phase=np.zeros_like(interf),
        phase_correction=True,
        return_complex=True)

    if high_order_phase is not None:
        cm1_axis = orb.utils.spectrum.create_cm1_axis(
            interf.shape[0], step, order,
            corr=calib/nm_laser).astype(np.float64)
        try: # if high_order_phase is a PhaseFile instance
            high_order_phase = high_order_phase.get_improved_phase(calib)
        except AttributeError: pass
            
        high_phase = high_order_phase(cm1_axis)
    else:
        high_phase = None
    
    optim = scipy.optimize.leastsq(
        diff, vguess, args=(
            interf_fft, fguess, fixed_params, weights,
            high_phase),
        full_output=True)
    
    if optim[-1] < 5:
        p = np.empty_like(fixed_params, dtype=float)
        p[np.nonzero(fixed_params==0)] = optim[0]
        p[np.nonzero(fixed_params)] = guess[np.nonzero(fixed_params)]
        res = (np.sqrt(np.nanmean(optim[2]['fvec']**2.))
               /interf_mean_energy(interf))
        if not return_coeffs:
            optim_phase = np.polynomial.polynomial.polyval(
                np.arange(np.size(interf)), p)
            if high_phase is not None:
                return optim_phase + high_phase
            else:
                return optim_phase
        else:
            return p, res
    else: return None


def optimize_phase3d(interf_cube, step, order, zpd_shift,
                     calib_map, nm_laser,
                     high_order_phase, pm0=None, pm1=None):
    """Return the 3 coefficents that define the linear phase terms (p0
    + p1 * x).

    The optimization is based on the maximization of the real part of
    the spectrum. It is done on the complete interferogram cube (a
    binned version is much faster) with a given calibration map. In
    this process the calibration map is considered to be exact and the
    order 0 phase map is directly calculated from it unless a phase
    map is given.

    This method is very efficient for cubes with poor phase
    informations in each pixel: e.g. extended galactic nebula that
    covers the whole field of view.

    :param interf_cube: Interferogram cube
    
    :param step: Step size
    
    :param order: Folding order
    
    :param zpd_shift: ZPD shift
    
    :param calib_map: Calibration laser map
    
    :param nm_laser: Calibration laser wavelength
    
    :param high_order_phase: A scipy.Spline instance of the high order
      phase or an orb.core.PhaseFile instance.

    :param pm0: (Optional) Order 0 phase map. This map is adjusted
      instead of calculating a phase map from a calibration laser
      frame (default None).
      
    :param pm1: (Optional) Order 1 phase map. This map is adjusted
      instead of considering a single order 1 coefficient (default
      None). WARNING: this phase map must be in a "portable format",
      i.e.: it must have been multiplied by the number of steps of the
      original cube.
  
    :return: Phase map coefficients (a0, a1, p1) if the phase map has
      to be calculated from the calibration map (pm0 set to None) or
      (a0, p1) if the order 0 phase map is given
    """

    def model(p, interf_cube_fft, calib_map, nm_laser,
              high_order_cube, Z, real, pm0, pm1):
        if pm1 is None:
            _pm1 = np.ones_like(calib_map)
        else:
            _pm1 = np.copy(pm1) / interf_cube_fft.shape[-1]
            
        if pm0 is None:
            _pm0 = calib_map2phase_map0([p[0], p[1]], np.copy(calib_map), nm_laser)
            _pm1 = _pm1 + p[2]
        else:
            _pm0 = p[0] + pm0
            _pm1 = _pm1 + p[1]
    
        ext_phase = (_pm0.T + (_pm1.T * Z.T)).T + high_order_cube
        if real:
            return (interf_cube_fft.imag * np.sin(ext_phase)
                    + interf_cube_fft.real * np.cos(ext_phase))
        else:
            return (interf_cube_fft.imag * np.cos(ext_phase)
                    - interf_cube_fft.real * np.sin(ext_phase))
        

    def objf(pfree, pfixed, interf_cube_fft, 
             calib_map, nm_laser, Z, high_order_cube, w3d,
             real, pm0, pm1):
        pfixed = np.array(pfixed)
        p = np.copy(pfixed)
        p[np.isnan(pfixed)] = np.array(pfree)
        
        fft_3d = np.copy(model(p, interf_cube_fft, calib_map,
                               nm_laser, high_order_cube, Z, real, pm0, pm1))
        fft_3d *= w3d
        if real:
            return -np.nanmean(fft_3d)
        else:
            return np.nanstd(fft_3d)

    def brute(gridsz, guess_min, guess_max, use_pm1):

        if r_pm1 is not None:
            if use_pm1: _r_pm1 = np.copy(r_pm1)
            else: _r_pm1 = None
            
        axes = list()
        slices = list()
        p_fixed = list()
        steps = list()
        
        print 'Brute force exploration space:'
        
        for i in range(len(guess_min)):
            slices.append(slice(guess_min[i], guess_max[i], gridsz*1j))
            p_fixed.append(np.nan)
            axes.append(np.linspace(guess_min[i], guess_max[i], gridsz))
            steps.append(axes[i][1] - axes[i][0])
            print ' a{}: {} to {} [{}]'.format(
                i, guess_min[i], guess_max[i], np.diff(axes[i])[0])

        start_brute_time = time.time()
        brute = optimize.brute(
            objf, slices,
            args=(p_fixed,
                  r_interf_cube_fft, r_calib_map, nm_laser,
                  r_Z, r_high_order_cube, r_w3d, True, r_pm0, _r_pm1),
            full_output=True, finish=None)
        print 'Brute force exploration time: {} s'.format(
            time.time() - start_brute_time)
        print 'Brute force guess: {} [min value: {:.4e}]'.format(
            brute[0], brute[1])

        ## brute_gridv = brute[3]
        ## best_index = np.unravel_index(np.argmin(brute_gridv),
        ##                               brute_gridv.shape)        
        ## orb.utils.io.write_fits('brute_gridv.fits', brute_gridv, overwrite=True)
        ## import pylab as pl
        ## pl.figure(0)
        ## pl.plot(axes[0], brute_gridv[:,best_index[1]]) # x
        ## pl.figure(1)
        ## pl.plot(axes[1], brute_gridv[best_index[0],:]) # y        
        ## pl.show()
        ## import pylab as pl
        ## pl.figure(0)
        ## pl.plot(axes[0], np.nanmin(np.nanmin(brute_gridv, axis=2), axis=1)) # x
        ## pl.figure(1)
        ## pl.plot(axes[1], np.nanmin(np.nanmin(brute_gridv, axis=2), axis=0)) # y
        ## pl.figure(2)
        ## pl.plot(axes[2], np.nanmin(np.nanmin(brute_gridv, axis=0), axis=0)) # z
        ## pl.show()

        return np.array(brute[0]), np.array(steps)
                
    BORDER = 0.1 # Relative size of the borders
    ZPD_RANGE = 5 # range checked around the real ZPD
    RND_COEFF = 0.2 # ratio of the randomly picked interferograms
    GRIDSZ = 40 # Brute force grid size
    SUB_GRIDSZ = GRIDSZ / 2 # Brute force subgrid size
    REFINE_COEFF = 2 # number of steps of the grid used for the subgrid

    dimx, dimy, dimz = interf_cube.shape

    # compute ZPD shift
    if pm1 is None:
        zpd_pos = dimz / 2 - zpd_shift
        zpd_check_range = zpd_pos - ZPD_RANGE, zpd_pos + ZPD_RANGE + 1
        zpd_check_list = list()
        for i in range(min(zpd_check_range), max(zpd_check_range)):
            frame = interf_cube[:,:,i]
            zpd_check_list.append(
                np.nanpercentile(frame, 0.9) - np.nanpercentile(frame, 0.1))

        new_zpd_pos = np.nanargmax(zpd_check_list) + min(zpd_check_range)
        new_zpd_shift = dimz/2 - new_zpd_pos
        print 'Init ZPD shift: {}, real ZPD shift: {}'.format(
            zpd_shift, new_zpd_shift)
    else:
        new_zpd_shift = int(zpd_shift)
    
    # compute complex fft of interf cube
    interf_cube_fft = np.empty_like(interf_cube, dtype=complex)
    high_order_cube = np.empty_like(interf_cube)
    for ii in range(dimx):
        sys.stdout.write('\rTransforming column {}/{}'.format(ii, dimx-1))
        sys.stdout.flush()
        for ij in range(dimy):
            interf_cube_fft[ii,ij] = transform_interferogram(
                interf_cube[ii,ij], 1., 1., step, order, '1.0', new_zpd_shift,
                wavenumber=True,
                ext_phase=np.zeros(dimz, dtype=float),
                phase_correction=True,
                return_complex=True)
            phase_axis = orb.utils.spectrum.create_cm1_axis(
                dimz, step, order,
                corr=float(calib_map[ii,ij]) / nm_laser).astype(np.float64)
            try: # if high_order_phase is a PhaseFile instance
                high_order_phase = high_order_phase.get_improved_phase(
                    float(calib_map[ii,ij]))
            except AttributeError: pass
            high_order_cube[ii,ij] = high_order_phase(phase_axis)
    sys.stdout.write('\n')
    ## orb.utils.io.write_fits('interf_cube_fft.real.fits', interf_cube_fft.real,
    ##                         overwrite=True)
    ## orb.utils.io.write_fits('interf_cube_fft.imag.fits', interf_cube_fft.imag,
    ##                         overwrite=True)
    ## orb.utils.io.write_fits('high_order_cube.fits', high_order_cube,
    ##                         overwrite=True)
    ## interf_cube_fft.real = orb.utils.io.read_fits('interf_cube_fft.real.fits')
    ## interf_cube_fft.imag = orb.utils.io.read_fits('interf_cube_fft.imag.fits')
    ## high_order_cube = orb.utils.io.read_fits('high_order_cube.fits')

    # compute weights
    w3d = np.ones_like(interf_cube_fft, dtype=float)
    w3d[:int(BORDER*dimx),:,:] = 0.
    w3d[-int(BORDER*dimx):,:,:] = 0.
    w3d[:,:int(BORDER*dimy),:] = 0.
    w3d[:,-int(BORDER*dimy):,:] = 0.
    
    Z = np.mgrid[:dimx, :dimy, :dimz][2,:,:,:]

    # create randomized cubes to accelerate the process
    rndsize = int(dimx * dimy * RND_COEFF)
    rndx = np.random.randint(0, dimx, 2*rndsize)
    rndy = np.random.randint(0, dimy, 2*rndsize)

    r_interf_cube_fft = np.empty((rndsize, dimz), dtype=complex)
    r_calib_map = np.empty(rndsize, dtype=float)
    r_Z = np.empty((rndsize, dimz), dtype=float)
    r_high_order_cube = np.empty((rndsize, dimz), dtype=float)
    r_w3d = np.empty((rndsize, dimz), dtype=float)
    r_pm0 = np.empty(rndsize, dtype=float)
    r_pm1 = np.empty(rndsize, dtype=float)

    for i in range(rndsize):
        ix = rndx[i]
        iy = rndy[i]
        if np.nanmax(np.abs(interf_cube_fft[ix, iy, :])) < 45000:
            r_interf_cube_fft[i, :] = interf_cube_fft[ix, iy, :]
            r_calib_map[i] = calib_map[ix, iy]
            r_Z[i, :] = Z[ix, iy, :]
            r_high_order_cube[i, :] = high_order_cube[ix, iy, :]
            r_w3d[i, :] = w3d[ix, iy, :]
            if pm0 is not None: r_pm0[i] = pm0[ix, iy]
            else: r_pm0 = None
            if pm1 is not None: r_pm1[i] = pm1[ix, iy]
            else: r_pm1 = None
        
    # brute force exploration
    if pm1 is None:
        order1_max = 1.5 * np.pi / dimz
        order1_min = -order1_max
    else:
        order1_min = -0.2
        order1_max = 0.2

    if pm0 is None:
        a0_min = - np.pi
        a0_max = np.pi
        a1_min = 100
        a1_max = 250
        guess_min = [a0_min, a1_min, order1_min]
        guess_max = [a0_max, a1_max, order1_max]
    else:
        a0_min = 0
        a0_max = 2 * np.pi
        guess_min = [a0_min, order1_min]
        guess_max = [a0_max, order1_max]

    best_coeffs, steps = brute(
        GRIDSZ, guess_min, guess_max, True)

    for _ in range(6):
        guess_min = best_coeffs - REFINE_COEFF * steps
        guess_max = best_coeffs + REFINE_COEFF * steps
        best_coeffs, steps = brute(
            SUB_GRIDSZ, guess_min, guess_max, True)
    
    ## orb.utils.io.write_fits(
    ##     'fftreal3d.fits',
    ##     model(best_coeffs,
    ##           interf_cube_fft,
    ##           calib_map, nm_laser, high_order_cube, Z, True, pm0, pm1),
    ##     overwrite=True)

    if pm1 is None:
        # order 1 is corrected for the new zpd shift
        best_coeffs[-1] += (new_zpd_shift - zpd_shift) * math.pi / dimz
    
        print 'Order 1 fitted value corrected for ZPD init shift ({} steps): {}'.format(new_zpd_shift - zpd_shift, best_coeffs[-1])
        
    return best_coeffs
    
def calib_map2phase_map0(p, calib_map, nm_laser):
    """Compute order 0 phase map from calibration laser map 

    :param p: Transformation parameters [a0, a1]
    :param calib_map: Calibration laser map
    :param nm_laser: Calibration laser wavelength
    """
    theta_inv = 1 - (calib_map / nm_laser)
    #theta_inv = calib_map
    return np.polynomial.polynomial.polyval(theta_inv, p)

def phase_map02calib_map(p, phase_map0, nm_laser):
    """Compute calibration laser map from order 0 phase map.

    :param p: Transformation parameters [a0, a1]
    :param calib_map: Order 0 phase map
    :param nm_laser: Calibration laser wavelength
    """
    return nm_laser * (1 - ((phase_map0 - p[0])/p[1]))

## def create_mean_phase_vector(phase_cube, step, order,
##                              calib_map, nm_laser,
##                              filter_file_path,
##                              size_coeff=0.45,
##                              border_coeff=0.05):

##     """Compute mean phase vector of order > 1.

##     :param phase_cube: Phase cube (generally binned) as computed by
##       ORBS Interferogram class.

##     :param step: Step size in nm

##     :param order: Folding order

##     :param filter_file_path: Filter file path

##     :param size_coeff: Relative size of the central part used to compute mean
##       phase vector.

##     :param border_coeff: Relative size of the border inside the filter
##       edges. This part is not considered as good.

##     :returns: a tuple (Phase axis [cm-1], Phase [radians])
##     """
    
##     x_min, x_max, y_min, y_max = orb.utils.image.get_box_coords(
##         phase_cube.shape[0]/2,
##         phase_cube.shape[1]/2,
##         int(float(min(phase_cube.shape))*size_coeff),
##         0, phase_cube.shape[0],
##         0, phase_cube.shape[1])
    
##     cm1_axis_base = orb.utils.spectrum.create_cm1_axis(
##                 phase_cube.shape[2], step, order, corr=1.)
    
##     phase = np.zeros(phase_cube.shape[2], dtype=float)

##     counts = 0
##     range_axis = np.arange(phase_cube.shape[2])

##     z_min, z_max = orb.utils.filters.get_filter_edges_pix(
##         filter_file_path, 1., step, order,
##         phase.shape[0])

##     border_pix = int(float(z_max - z_min) * border_coeff)
##     z_min += border_pix
##     z_max -= border_pix
    
##     for ii in range(x_min, x_max):
##         phase_col = list()
##         for ij in range(y_min, y_max):
##             # interpolate phase to 0deg axis
##             corr =  calib_map[ii,ij] / nm_laser
##             cm1_axis = orb.utils.spectrum.create_cm1_axis(
##                 phase_cube.shape[2], step, order, corr=corr)
##             iphase = phase_cube[ii,ij,:]
##             iphase = orb.utils.vector.interpolate_axis(
##                 iphase, cm1_axis_base, 1, old_axis=cm1_axis)
##             if int(order) & 1: iphase = iphase[::-1]
##             # remove parts outside filter
##             iphase[:z_min] = np.nan
##             iphase[z_max:] = np.nan
            
##             # remove a 1 order polynomial to avoid order 0 and order 1
##             # variable contribution.
##             range_nans = range_axis[~np.isnan(iphase)]
##             iphase_nans = iphase[~np.isnan(iphase)]
            
##             iphase -= np.polyval(np.polyfit(
##                 range_nans, iphase_nans, 1), range_axis)
##             phase_col.append(iphase)
        
##         # each column of phase vectors is reduced to one phase vector
##         # with a sigmacut
##         phase_col = np.array(phase_col, dtype=float)
##         iphase = np.empty(phase_col.shape[1], dtype=float)
##         for ik in range(iphase.shape[0]):
##             iphase[ik] = np.nanmean(orb.utils.stats.sigmacut(
##                 phase_col[:,ik], sigma=2.0))
##         # computed phase vector for the column is added to the final
##         # phase vector
##         phase += iphase
##         counts += 1
        
  
##     phase /= counts
##     phase -= phase[~np.isnan(phase)][0] # first sample at 0.
            
    
##     return cm1_axis_base, phase

def create_phase_file(file_path, phase_vector, cm1_axis):
    """Write a phase vector in a phase file.

    Phase vector is interpolated via a cubic spline.

    :param file_path: Path to the output phase file.

    :param phase_vector: Phase vector

    :param cm1_axis: Phase vector axis in cm-1
    """
    
    STEP_NB = 2000

    # interpolate phase
    cm1_axis = cm1_axis.astype(float)
    cm1_axis_nans = cm1_axis[~np.isnan(phase_vector)]
    phase_vector_nans = phase_vector[~np.isnan(phase_vector)]
    phase_finterp = scipy.interpolate.UnivariateSpline(
        cm1_axis_nans, phase_vector_nans,
        k=3, s=0)
    cm1_axis_interp = np.linspace(np.min(cm1_axis_nans),
                                  np.max(cm1_axis_nans),
                                  STEP_NB)
    phase_interp = phase_finterp(cm1_axis_interp)

    # write phase in a phase file
    with open(file_path, 'w') as f:
        for i in range(STEP_NB):
            f.write('{} {}\n'.format(
                cm1_axis_interp[i],
                phase_interp[i]))


def read_phase_file(file_path, return_spline=False):
    """Read a basic phase file and return its content.

    :param file_path: Path to the phase file

    :param return_spline: If True a cubic spline
      (scipy.interpolate.UnivariateSpline instance) is returned
      instead of a tuple.

    :returns: a tuple (Phase axis [cm-1], Phase [radians]) or a
      scipy.interpolate.UnivariateSpline instance if return_spline is
      True.
    """
    cm1_axis = list()
    phase = list()
    with open(file_path, 'r') as f:
        for line in f:
            if '#' not in line and len(line) > 2:
                cm1, ph = line.strip().split()
                cm1_axis.append(float(cm1))
                phase.append(float(ph))
    cm1_axis = np.array(cm1_axis, dtype=float)
    phase = np.array(phase, dtype=float)
    if not return_spline:
        return cm1_axis, phase
    else:
        return interpolate.UnivariateSpline(
            cm1_axis, phase, k=3, s=0, ext=0)
