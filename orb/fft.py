#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fft.py

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
## or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy as np
import warnings

import utils.validate
import utils.fft
import utils.vector
import utils.err
import core
import cutils
import scipy

class Interferogram(core.Vector1d):
    """Interferogram class.
    """
    needed_params = 'step', 'order', 'zpd_index', 'calib_coeff', 'filter_file_path'
    optional_params = ('filter_cm1_min', 'filter_cm1_max')
        
    def __init__(self, interf, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray interferogram.

        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are 'step', 'order', 'zpd_index', 'calib_coeff' (default
          None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        core.Vector1d.__init__(self, interf, params=params, **kwargs)
        if self.params.zpd_index < 0 or self.params.zpd_index >= self.step_nb:
            raise ValueError('zpd must be in the interferogram')

    def __getitem__(self, key):
        """Implement __getitem__ special method and return a valid
        inteferogram class (with its parameters changed to reflect the
        slicing)
        """
        axis = np.zeros(self.step_nb)
        axis[self.params.zpd_index] = 1
        axis = axis.__getitem__(key)
        if axis.size <= 1:
            raise ValueError('slicing cannot return an interferogram with less than 2 samples. Use self.data[index] instead of self[index].')
        if np.max(axis) != 1:
            raise RuntimeError('ZPD is not anymore in the returned interferogram')
        zpd_index = np.argmax(axis)
        params = self.params
        params['zpd_index'] = zpd_index
        return self.__class__(self.data.__getitem__(key), params=params)

        
    def subtract_mean(self):
        """substraction of the mean of the interferogram where the
        interferogram is not nan
        """
        self.data[~np.isnan(self.data)] -= np.nanmean(self.data)


    def subtract_low_order_poly(self, order=3):
        """ low order polynomial substraction to suppress low
        frequency noise

        :param order: (Optional) Polynomial order (beware of high
          order polynomials, default 3).
        """
        self.data[~np.isnan(self.data)] -= utils.vector.polyfit1d(
            self.data, order)[~np.isnan(self.data)]


    def apodize(self, window_type):
        """Apodization of the interferogram

        :param window_type: Name of the apodization function (can be
          'learner95' or a float > 1.)
        """
        self.assert_params()
        
        if not (0 <= self.params.zpd_index <= self.step_nb):
            raise ValueError('zpd index must be >= 0 and <= interferogram size')
        
        x = np.arange(self.step_nb, dtype=float) - self.params.zpd_index
        x /= max(np.abs(x[0]), np.abs(x[-1]))
        
        if window_type is None: return
        elif window_type == '1.0' : return
        elif window_type == 'learner95':
            window = utils.fft.learner95_window(x)
        else:
            window = utils.fft.gaussian_window(window_type, x)

        self.data *= window

    def symmetric(self):
        """Return an interferogram which is symmetric around the zpd"""
        if self.params.zpd_index < self.step_nb / 2:
            return self[:self.params.zpd_index * 2 - 1]
        else:
            shortlen = self.step_nb - self.params.zpd_index
            return self[max(self.params.zpd_index - shortlen, 0):]
        
    def transform(self):
        """zero padded fft.
          
        :return: A Spectrum instance (or a core.Vector1d instance if
          interferogram is full of zeros or nans)

        .. note:: no phase correction is made here.
        """
        if self.anynan:
            logging.debug('Nan detected in interferogram')
            return core.Vector1d(np.zeros(
                self.step_nb, dtype=self.data.dtype) * np.nan)
        if self.allzero:
            logging.debug('interferogram is filled with zeros')
            return core.Vector1d(np.zeros(
                self.step_nb, dtype=self.data.dtype))
        
        # zero padding
        zp_nb = self.step_nb * 2
        zp_interf = np.zeros(zp_nb, dtype=float)
        zp_interf[:self.step_nb] = np.copy(self.data)

        # dft
        interf_fft = np.fft.fft(zp_interf)
        #interf_fft = interf_fft[:interf_fft.shape[0]/2+1]
        interf_fft = interf_fft[:self.step_nb]
        

        # normalization of the vector to take into account zero-padding
        # and mimic a dispersive instrument: if the same energy is
        # dispersed over more channels (more zeros) then you get less
        # counts/channel
        if np.iscomplexobj(self.data):
            interf_fft /= (zp_nb / float(self.step_nb))
        else:
            interf_fft /= (zp_nb / float(self.step_nb)) / 2.
                            
        # create axis
        if self.has_params():
            axis = core.Axis(utils.spectrum.create_cm1_axis(
                self.step_nb, self.params.step, self.params.order,
                corr=self.params.calib_coeff))

        else:
            axis_step = (self.step_nb - 1) / 2. / self.step_nb
            axis_max = (self.step_nb - 1) * axis_step
            axis = core.Axis(np.linspace(0, axis_max, self.step_nb))

        spec = Spectrum(interf_fft, axis, params=self.params)

        # zpd shift phase correction
        if self.has_params():
            spec.zpd_shift(self.params.zpd_index)

        # spectrum is reversed if order is even
        if self.has_params():
            if int(self.params.order)&1:
                spec.reverse()
            
        return spec

    def get_spectrum(self):
        """Classical spectrum computation method. Returns a Spectrum instance."""
        new_interf = self.copy()
        new_interf.subtract_mean()
        return new_interf.transform()

    def get_phase(self):
        """Classical phase computation method. Returns a Phase instance."""
        new_interf = self.copy()
        new_interf = new_interf.symmetric()
        new_interf.subtract_mean()
        new_spectrum = new_interf.transform()
        return new_spectrum.get_phase().cleaned()

class Cm1Vector1d(core.Vector1d):
    """General 1d vector class for data projected on a cm-1 axis
    (e.g. complex spectrum, phase)
    """
    needed_params = ('filter_file_path', )
    optional_params = ('filter_cm1_min', 'filter_cm1_max')
    
    def __init__(self, spectrum, axis, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray vector.
        
        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are 'step', 'order', 'zpd_index', 'calib_coeff' (default
          None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        core.Vector1d.__init__(self, spectrum, params=params, **kwargs)

        self.axis = core.Axis(axis)

        if self.axis.step_nb != self.step_nb:
            raise ValueError('axis must have the same size as spectrum')

    def copy(self):
        """Return a copy of the instance"""
        return self.__class__(
            np.copy(self.data),
            np.copy(self.axis.data),
            params=self.params.convert())

    def reverse(self):
        """Reverse data. Do not reverse the axis.
        """
        self.data = self.data[::-1]

    def get_filter_bandpass_cm1(self):
        """Return filter bandpass in cm-1"""
        if 'filter_cm1_min' not in self.params or 'filter_cm1_max' not in self.params:
            nm_min, nm_max = utils.filters.get_filter_bandpass(self.params.filter_file_path)
            warnings.warn('Uneffective call to get filter bandpass. Please provide filter_cm1_min and filter_cm1_max in the parameters.')
            cm1_min, cm1_max = utils.spectrum.nm2cm1((nm_max, nm_min))
            self.params['filter_cm1_min'] = cm1_min
            self.params['filter_cm1_max'] = cm1_max
            
        return self.params.filter_cm1_min, self.params.filter_cm1_max

    def get_filter_bandpass_pix(self):
        """Return filter bandpass in channels"""
        return (self.axis(self.get_filter_bandpass_cm1()[0]),
                self.axis(self.get_filter_bandpass_cm1()[1]))
    
class Phase(Cm1Vector1d):
    """Phase class
    """
    def cleaned(self):
        """Return a cleaned phase vector with values out of the filter set to
        nan and a median around 0 (modulo pi).
        """
        zmin, zmax = np.array(self.get_filter_bandpass_pix()).astype(int)
        data = np.empty_like(self.data)
        data.fill(np.nan)
        ph = utils.vector.robust_unwrap(self.data[zmin:zmax], 2*np.pi)
        if np.any(np.isnan(ph)):
            ph.fill(np.nan)
        else:
            # set the first sample at the smallest positive modulo pi
            # value (order 0 is modulo pi)
            new_orig = np.fmod(ph[0], np.pi)
            while new_orig < 0:
                new_orig += np.pi
            if np.abs(new_orig) > np.abs(new_orig - np.pi):
                new_orig -= np.pi
            elif np.abs(new_orig) > np.abs(new_orig + np.pi):
                new_orig += np.pi
                
                
            ph -= ph[0]
            ph += new_orig
            
        data[zmin:zmax] = ph
        
        return Phase(data, self.axis, params=self.params)
        
    def polyfit(self, deg, coeffs=None, return_coeffs=False):
        """Polynomial fit of the phase
   
        :param deg: Degree of the fitting polynomial. Must be >= 0.

        :param coeffs: Used to fix some coefficients to a given
          value. If not None, must be a list of length = deg. set a
          coeff to a np.nan or a None to let the parameter free.

        :param return_coeffs: If True return (fit coefficients, error
          on coefficients) else return a Phase instance
          representing the fitted phase.
        """
        RANGE_BORDER_COEFF = 0.1

        self.assert_params()
        deg = int(deg)
        if deg < 0: raise ValueError('deg must be >= 0')
            
        cm1_min, cm1_max = self.get_filter_bandpass_cm1()
        
        cm1_border = np.abs(cm1_max - cm1_min) * RANGE_BORDER_COEFF
        cm1_min += cm1_border
        cm1_max -= cm1_border

        weights = np.ones(self.step_nb, dtype=float) * 1e-35
        weights[int(self.axis(cm1_min)):int(self.axis(cm1_max))+1] = 1.
        
        
        phase = np.copy(self.data)
        ok_phase = phase[int(self.axis(cm1_min)):int(self.axis(cm1_max))+1]
        if np.any(np.isnan(ok_phase)):
            raise utils.err.FitError('phase contains nans in the filter passband')
        
        phase[np.isnan(phase)] = 0.
        
        # create guess
        guesses = list()
        guess0 = np.nanmean(ok_phase)
        guess1 = np.nanmean(np.diff(ok_phase))
        guesses.append(guess0)
        guesses.append(guess1)        
        if deg > 1:
            for i in range(deg - 1):
                guesses.append(0)
        guesses = np.array(guesses)
        
        if coeffs is not None:
            utils.validate.has_len(coeffs, deg + 1)
            coeffs = np.array(coeffs, dtype=float) # change None by nan
            new_guesses = list()
            for i in range(guesses.size):
                if np.isnan(coeffs[i]):
                    new_guesses.append(guesses[i])
            guesses = np.array(new_guesses)

        # polynomial fit
        def format_guess(p):
            if coeffs is not None:
                all_p = list()
                ip = 0
                for icoeff in coeffs:
                    if not np.isnan(icoeff):
                        all_p.append(icoeff)
                    else:
                        all_p.append(p[ip])
                        ip += 1
            else:
                all_p = p
            return np.array(all_p)
        
        def model(x, *p):
            p = format_guess(p)            
            return np.polynomial.polynomial.polyval(x, p)

        def diff(p, x, y, w):
            res = model(x, *p) - y
            return res * w
        
        try:            
            # pfit, pcov = scipy.optimize.curve_fit(
            #     model, self.axis.data.astype(float), phase,
            #     guesses,
            #     1./weights)
            # perr = np.sqrt(np.diag(pcov))
            _fit = scipy.optimize.leastsq(
                diff, guesses,
                args=(
                    self.axis.data.astype(float),
                    phase, weights),
                full_output=True)
            pfit = _fit[0]
            pcov = _fit[1]
            perr = np.sqrt(np.diag(pcov) * np.std(_fit[2]['fvec'])**2)
            
        except Exception, e:
            logging.debug('Exception occured during phase fit: {}'.format(e))
            return None

        all_pfit = format_guess(pfit)
        all_perr = format_guess(perr)
        if coeffs is not None:
            all_perr[np.nonzero(~np.isnan(coeffs))] = np.nan
        
        logging.debug('fitted coeffs: {} ({})'.format(all_pfit, all_perr))
        if return_coeffs:
            return all_pfit, all_perr
        else:
            return self.__class__(model(self.axis.data.astype(float), *pfit),
                                  self.axis, params=self.params)

class Spectrum(Cm1Vector1d):
    """Spectrum class
    """    
    def __init__(self, spectrum, axis, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray vector.
        
        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are 'step', 'order', 'zpd_index', 'calib_coeff' (default
          None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        Cm1Vector1d.__init__(self, spectrum, axis, params=params, **kwargs)
     
        if not np.iscomplexobj(self.data):
            raise TypeError('input spectrum is not complex')

                   
    def get_phase(self):
        """return phase"""
        nans = np.isnan(self.data)
        _data = np.copy(self.data)
        _data[nans] = 0
        _phase = np.unwrap(np.angle(_data))
        _phase[nans] = np.nan
        return Phase(_phase, self.axis, params=self.params)

    def get_amplitude(self):
        """return amplitude"""
        return np.abs(self.data)

    def get_real(self):
        """Return the real part"""
        return np.copy(self.data.real)

    def get_imag(self):
        """Return the imaginary part"""
        return np.copy(self.data.imag)

    def zpd_shift(self, shift):
        """correct spectrum phase from shifted zpd"""
        self.correct_phase(
            np.arange(self.step_nb, dtype=float)
            * shift * np.pi / (self.step_nb)) # not (self.step_nb - 1) !
        
    def correct_phase(self, phase):
        """Correct spectrum phase"""
        phase = core.Vector1d(phase).data
        if phase.shape[0] != self.step_nb:
            warnings.warn('phase does not have the same size as spectrum. It will be interpolated.')
            phase = utils.vector.interpolate_size(phase, self.dimz, 1)
            
        self.data *= np.exp(1j * phase)
        
    def resample(self, axis):
        """Resample spectrum over the given axis

        :return: A new Spectrum instance

        .. warning:: Resampling is done via a DFT which is much slower
          than a FFT. But it gives perfect results with respect to
          interpolation.
        """
        raise NotImplementedError('Not implemented yet')
        ### should be done in the interferogram class instead (much easier to implement) ###
        interf = self.transform().data
        real_axis_step = float(self.axis[1] - self.axis[0])
        samples = (axis - self.axis[0]) / real_axis_step
        ok_samples = (0 <= samples) * (samples < self.step_nb)
        if not np.all(ok_samples):
            logging.debug('at least some samples are not in the axis range {}-{}'.format(
                self.axis[0], self.axis[-1]))
            dft_samples = samples[np.nonzero(ok_samples)]
        else:
            dft_samples = samples
            
        dft_spec = cutils.complex_dft(interf, dft_samples.astype(float))
        if not np.all(ok_samples):
            spec = np.empty_like(axis, dtype=complex)
            spec.fill(np.nan)
            spec[np.nonzero(ok_samples)] = dft_spec
        else:
            spec = dft_spec
            
        return Spectrum(spec, axis, params=self.params)


    def interpolate(self, axis, quality=10):
        """Resample spectrum by interpolation over the given axis

        :param quality: an integer from 2 to infinity which gives the
          zero padding factor before interpolation. The more zero
          padding, the better will be the interpolation, but the
          slower too.

        :return: A new Spectrum instance

        .. warning:: Though much faster than pure resampling, this can
          be a little less precise.
        """
        if isinstance(axis, core.Axis):
            axis = np.copy(axis.data)
        
        quality = int(quality)
        if quality < 2: raise ValueError('quality must be an interger > 2')
   
        interf_complex = np.fft.ifft(self.data)
        zp_interf = np.zeros(self.step_nb * quality, dtype=complex)
        center = interf_complex.shape[0] / 2
        zp_interf[:center] = interf_complex[:center]
        zp_interf[
            -center-int(interf_complex.shape[0]&1):] = interf_complex[
            -center-int(interf_complex.shape[0]&1):]

        zp_spec = np.fft.fft(zp_interf)
        zp_axis = (np.arange(zp_spec.size)
                   * (self.axis.data[1] - self.axis.data[0])  / float(quality)
                   + self.axis.data[0])
        f = scipy.interpolate.interp1d(zp_axis, zp_spec, bounds_error=False)
        return Spectrum(f(axis), axis, params=self.params)


#################################################
#### CLASS InteferogramCube #####################
#################################################
class InterferogramCube(core.OCube):
    """Provide additional methods for an interferogram cube when
    observation parameters are known.
    """

    def get_interferogram(self, x, y, zpd_index):
        """Return an orb.fft.Interferogram instance
        
        :param x: x position
        :param y: y position
        :param zpd_index: position of the ZPD of the interferogram
        """
        self.validate()
        x = self.validate_x_index(x, clip=False)
        y = self.validate_x_index(y, clip=False)
        
        calib_coeff = self.get_calibration_coeff_map()[x, y]
        return Interferogram(self[x, y, :], self.params,
                             zpd_index=zpd_index, calib_coeff=calib_coeff)

    def get_mean_interferogram(self, xmin, xmax, ymin, ymax, zpd_index):
        """Return mean interferogram in a box [xmin:xmax, ymin:ymax, :]
        along z axis
        
        :param xmin: min boundary along x axis
        :param xmax: max boundary along x axis
        :param ymin: min boundary along y axis
        :param ymax: max boundary along y axis
        :param zpd_index: position of the ZPD of the interferogram
        """
        self.validate()
        xmin, xmax = np.sort(self.validate_x_index([xmin, xmax], clip=False))
        ymin, ymax = np.sort(self.validate_y_index([ymin, ymax], clip=False))

        if xmin == xmax or ymin == ymax:
            raise ValueError('Boundaries badly defined, please check xmin, xmax, ymin, ymax')
        
        calib_coeff = np.nanmean(self.get_calibration_coeff_map()[xmin:xmax, ymin:ymax])
        interf = np.nanmean(np.nanmean(self[xmin:xmax, ymin:ymax, :], axis=0), axis=0)
        return Interferogram(interf, self.params, zpd_index=zpd_index,
                             calib_coeff=calib_coeff)
