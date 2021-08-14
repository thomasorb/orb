#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fft.py

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
## or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

import logging
import numpy as np
import warnings
import time
import scipy
import gvar

import orb.utils.validate
import orb.utils.fft
import orb.utils.vector
import orb.utils.err
import orb.utils.stats
import orb.core
import orb.cutils
import orb.fit
import orb.photometry



#################################################
#### CLASS Interferogram ########################
#################################################

class Interferogram(orb.core.Vector1d):

    """Interferogram class.
    """
    needed_params = 'step', 'order', 'zpd_index', 'calib_coeff', 'filter_name', 'exposure_time'
        
    def __init__(self, interf, err=None, axis=None, params=None, **kwargs):
        """Init method.


        :param interf: A 1d numpy.ndarray interferogram in counts (not
          counts/s)

        :param err: (Optional) Error vector. A 1d numpy.ndarray (default None).

        :param axis: (optional) A 1d numpy.ndarray axis (default None)
          with the same size as vector.

        :param params: (Optional) A dict containing observation
          parameters (default None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.

        """       
        orb.core.Vector1d.__init__(self, interf, axis=axis, err=err, params=params, **kwargs)

        if self.params.zpd_index < 0 or self.params.zpd_index >= self.dimx:
            raise ValueError('zpd must be in the interferogram')

        # opd axis (in cm) is automatically computed from the parameters
        opdaxis = 1e-7 * (np.arange(self.dimx) * self.params.step
                - (self.params.step * self.params.zpd_index)) * self.params.calib_coeff
        if self.axis is None:
            self.axis = orb.core.Axis(opdaxis)
        elif np.any(opdaxis != self.axis.data):
            raise Exception('provided axis is inconsistent with the opd axis computed from the observation parameters')
        
        if self.axis.dimx != self.dimx:
            raise ValueError('axis must have the same size as the interferogram')
        
    def crop(self, xmin, xmax):
        """Crop data. see Vector1d.crop()"""
        out = orb.core.Vector1d.crop(self, xmin, xmax, returned_class=orb.core.Vector1d)
        
        if out.axis.data.size <= 1:
            raise ValueError('cropping cannot return an interferogram with less than 2 samples. Use self.data[index] instead.')
        
        if out.params.zpd_index < xmin or out.params.zpd_index >= xmax:
            raise RuntimeError('ZPD is not anymore in the returned interferogram')

        zpd_index = out.params.zpd_index - xmin
        out.params.reset('zpd_index', zpd_index)
        return self.__class__(out)
        
        
    def subtract_mean(self, inplace=False):
        """substraction of the mean of the interferogram where the
        interferogram is not nan
        """
        if not inplace:
            spec = self.copy()
        else:
            spec = self
        spec.data[~np.isnan(spec.data)] -= np.nanmean(spec.data)
        return spec

    def subtract_low_order_poly(self, order=3, inplace=False):
        """ low order polynomial substraction to suppress low
        frequency noise

        :param order: (Optional) Polynomial order (beware of high
          order polynomials, default 3).
        """
        if not inplace:
            spec = self.copy()
        else:
            spec = self
        
        spec.data[~np.isnan(spec.data)] -= orb.utils.vector.polyfit1d(
            spec.data, order)[~np.isnan(spec.data)]
        return spec


    def apodize(self, window_type, inplace=False):
        """Apodization of the interferogram

        :param window_type: Name of the apodization function (can be
          'learner95' or a float > 1.)
        """
        self.assert_params()
        
        if not (0 <= self.params.zpd_index <= self.dimx):
            raise ValueError('zpd index must be >= 0 and <= interferogram size')
        
        x = np.arange(self.dimx, dtype=float) - self.params.zpd_index
        x /= max(np.abs(x[0]), np.abs(x[-1]))

        if window_type is None: return
        elif window_type == '1.0' : return
        elif window_type == 'learner95':
            window = orb.utils.fft.learner95_window(x)
        else:
            window = orb.utils.fft.gaussian_window(window_type, x)

        if not inplace:
            spec = self.copy()
        else:
            spec = self
        
        spec.data *= window
        if spec.has_err():
            spec.err *= window

        return spec


    def is_right_sided(self):
        """Check if interferogram is right sided (left side wrt zpd
        shorter than right side)
        """
        return (self.params.zpd_index < self.dimx / 2) # right sided
        

    def symmetric(self):
        """Return an interferogram which is symmetric around the zpd"""
        if self.is_right_sided():
            return self.crop(0, self.params.zpd_index * 2 - 1)
        else:
            shortlen = self.dimx - self.params.zpd_index
            return self.crop(max(self.params.zpd_index - shortlen, 0), self.dimx)

    def multiply_by_mertz_ramp(self, inplace=False):
        """Multiply by Mertz (1976) ramp function to avoid counting
        symmetric samples twice and reduce emission lines contrast wrt
        the background.
        """
        # create ramp
        zeros_vector = np.zeros(self.dimx, dtype=self.data.dtype)

        if self.is_right_sided():        
            sym_len = self.params.zpd_index * 2
            zeros_vector[:sym_len] = np.linspace(0,2,sym_len)
            zeros_vector[sym_len:] = 2.
        else:
            sym_len = (self.dimx - self.params.zpd_index) * 2
            zeros_vector[-sym_len:] = np.linspace(0,2,sym_len)
            zeros_vector[:-sym_len] = 2.

        if sym_len > self.dimx / 2.:
            logging.debug('interferogram is mostly symmetric. The use of Mertz ramp should be avoided.')

        if not inplace:
            spec = self.copy()
        else:
            spec = self
        
        spec.data *= zeros_vector
        if spec.has_err():
            spec.err *= zeros_vector

        return spec

    def transform(self):
        """zero padded fft.
          
        :return: A Spectrum instance (or a core.Vector1d instance if
          interferogram is full of zeros or nans)

        .. note:: no phase correction is made here.
        """
        if np.any(np.isnan(self.data)):
            logging.debug('Nan detected in interferogram')
            return orb.core.Vector1d(np.zeros(
                self.dimx, dtype=self.data.dtype) * np.nan)
        if len(np.nonzero(self.data)[0]) == 0:
            logging.debug('interferogram is filled with zeros')
            return orb.core.Vector1d(np.zeros(
                self.dimx, dtype=self.data.dtype))
        
        # zero padding
        zp_nb = self.dimx * 2
        zp_interf = np.zeros(zp_nb, dtype=float)
        zp_interf[:self.dimx] = np.copy(self.data)

        # dft
        #interf_fft = np.fft.fft(zp_interf)
        interf_fft = scipy.fft.fft(zp_interf)
        
        #interf_fft = interf_fft[:interf_fft.shape[0]/2+1]
        interf_fft = interf_fft[:self.dimx]
                                    
        # create axis
        if self.has_params():
            axis = orb.core.Axis(orb.utils.spectrum.create_cm1_axis(
                self.dimx, self.params.step, self.params.order,
                corr=self.params.calib_coeff))

        else:
            axis_step = (self.dimx - 1) / 2. / self.dimx
            axis_max = (self.dimx - 1) * axis_step
            axis = orb.core.Axis(np.linspace(0, axis_max, self.dimx))

        # compute err (photon noise)
        if self.has_err():
            err = np.ones_like(self.data, dtype=float) * np.sqrt(np.sum(self.err**2))
        else: err = None
            
        spec = Spectrum(interf_fft, err=err, axis=axis.data, params=self.params)

        # spectrum is flipped if order is even
        if self.has_params():
            if int(self.params.order)&1:
                spec.reverse()

        # zpd shift phase correction. The sign depends on even or odd order.
        if self.has_params():
            if int(self.params.order)&1:
                spec.zpd_shift(-self.params.zpd_index)
            else:
                spec.zpd_shift(self.params.zpd_index)

        return spec

    def get_spectrum(self, mertz=True):
        """Classical spectrum computation method. Returns a Spectrum instance.

        :param mertz: If True, multiply by Mertz ramp. Must be used for assymetric interferograms.
        """
        new_interf = self.copy()
        new_interf.subtract_mean(inplace=True)
        if mertz:
            new_interf.multiply_by_mertz_ramp(inplace=True)
        return new_interf.transform()

    def get_phase(self):
        """Classical phase computation method. Returns a Phase instance."""
        new_interf = self.copy()
        new_interf = new_interf.symmetric()
        new_interf.subtract_mean(inplace=True)
        new_interf.apodize('1.5', inplace=True)
        new_spectrum = new_interf.transform()
        return new_spectrum.get_phase().cleaned()

#################################################
#### CLASS RealInterferogram ####################
#################################################

class RealInterferogram(Interferogram):
    """Class used for an observed interferogram in counts"""
    
    def __init__(self, *args, **kwargs):
        """.. warning:: in principle data unit should be counts (not
          counts / s) so that computed noise value and helpful methods
          keep working. The sky interferogram should not be subtracted
          from the input interferogram. It can be done once the
          interferogram is initialized with the method subtract_sky().

        parameters are the same as the parent Interferogram
        class. Note that, if not supplied in the arguments, the error
        vector (err) which gives the photon noise will be computed as
        sqrt(data)

        important parameters that must supplied in params:

        - pixels: number of integrated pixels (default 1). Must be an
          integer.

        - source_counts: total number of counts in the source. Must be
          a float. Used to estimate the modulation
          efficiency. It the given interferogram is raw (no sky
          subtraction, no combination, no stray light removal...) the
          total number of counts of the source is bascally
          np.sum(data). This raw number will be actualized when
          sky/stray light will be subtracted.

        """
        Interferogram.__init__(self, *args, **kwargs)

        # check if integrated
        if 'pixels' not in self.params:
            self.params.reset('pixels', 1)
        elif not isinstance(self.params.pixels, int):
            logging.debug('pixels converted to an integer')
            self.params['pixels'] = int(self.params.pixels)

        # check source_counts
        if 'source_counts' not in self.params:
            if np.nanmin(self.data) < 0:
                logging.warning('interferogram may be a combined interferogram. source_counts can be wrong')
            self.params.reset('source_counts', np.sum(np.abs(self.data)))
        elif not isinstance(self.params.source_counts, float):
            raise TypeError('source_counts must be a float')
            
        # compute photon noise
        if not self.has_err(): # supplied err argument should have been
                               # read during Vector1d init. Only
                               # self.err is tested
            if np.nanmin(self.data) < 0:
                logging.warning('interferogram may be a combined interferogram. photon noise can be wrong.')
            
            self.err = np.sqrt(np.abs(self.data))
            

    def math(self, opname, arg=None):
        """Do math operations and update the 'source_counts' value.

        :param opname: math operation, must be a numpy.ufuncs.

        :param arg: If None, no argument is supplied. Else, can be a
          float or a Vector1d instance.
        """
        out = Interferogram.math(self, opname, arg=arg)

        if arg is None:
            source_counts = getattr(np, opname)(self.params.source_counts)
        else:
            try:
                _arg = arg.params.source_counts
            except Exception:
                _arg = arg    
            source_counts = getattr(np, opname)(self.params.source_counts, _arg)
            
        out.params.reset('source_counts', source_counts)
            
    def subtract_sky(self, sky):
        """Subtract sky interferogram (or any background interferogram). 
        
        The values of the parameter 'pixels' in both this
        interferogram and the sky interferogram should be set to the
        number of integrated pixels.
        """
        sky = sky.copy()
        sky.data *= float(self.params.pixels) / float(sky.params.pixels)
        return self.subtract(sky)
        
        
    def combine(self, interf, transmission=None, ratio=None):
        """Combine two interferograms (one from each camera) and return the result

        :param interf: Another Interferogram instance coming from the
          complementary camera.

        :param transmission: (Optional) Must be a Vector1d instance. if None
          supplied, transmission is computed from the combined
          interferogram which may add some noise (default None).

        :param ratio: (Optional) optical transmission ratio self /
          interf. As the beamsplitter and the differential gain of the
          cameras produce a difference in the number of counts
          collected in one camera or the other, the interferograms
          must be corrected for this ratio before being combined. If
          None, this ratio is computed from the provided
          interferograms.
        """
        # project interf
        interf = interf.project(self.axis)

        # compute ratio
        if ratio is None:
            ratio = np.mean(self.data) / np.mean(interf.data)

        corr = 1. - (1. - ratio) / 2.

        comb = self.copy()

        # compute transmission and correct for it
        if transmission is None:
            transmission = self.compute_transmission(interf)

        # combine interferograms
        _comb = (self.get_gvar() / corr - interf.get_gvar() * corr) / transmission.get_gvar()
        
        comb.data = gvar.mean(_comb)
        comb.err = gvar.sdev(_comb)

        del _comb

        # compute photon_noise
        comb.params.reset('source_counts', interf.params.source_counts
                          + self.params.source_counts)
        
        return comb

    def compute_transmission(self, interf):
        """Return the transmission vector computed from the combination of two
        complementary interferograms.

        :param interf: Another Interferogram instance coming from the
          complementary camera.
        """
        NORM_PERCENTILE = 99

        transmission = self.add(interf)
        transmission.data /= np.nanpercentile(transmission.data, NORM_PERCENTILE)
        return transmission                

    def transform(self):
        """Zero padded fft. See Interferogram.transform()
        """
        out = Interferogram.transform(self)

        return RealSpectrum(out)        


#################################################
#### CLASS Phase ################################
#################################################

class Phase(orb.core.Cm1Vector1d):
    """Phase class
    """
    def cleaned(self, border_ratio=0.):
        """Return a cleaned phase vector with values out of the filter set to
        nan and a median around 0 (modulo pi).
        
        :param border_ratio: (Optional) Relative portion of the phase
          in the filter range removed (can be a negative float,
          default 0.)

        :return: A Phase instance
        """
        zmin, zmax = self.get_filter_bandpass_pix(border_ratio=border_ratio)
        data = np.empty_like(self.data)
        data.fill(np.nan)
        data[zmin:zmax] = orb.utils.fft.clean_phase(self.data[zmin:zmax])
                    
        return Phase(data, axis=self.axis, params=self.params)        

        
    def polyfit(self, deg,
                amplitude=None,
                coeffs=None,
                return_coeffs=False,
                border_ratio=0.1):
        """Polynomial fit of the phase
   
        :param deg: Degree of the fitting polynomial. Must be >= 0.

        :param amplitude: (Optional) Amplitude of the spectrum. Used
          to weight the phase data (default None).

        :param coeffs: (Optional) Used to fix some coefficients to a
          given value. If not None, must be a list of length =
          deg. set a coeff to a np.nan or a None to let the parameter
          free (default None).

        :param return_coeffs: (Optional) If True return (fit
          coefficients, error on coefficients) else return a Phase
          instance representing the fitted phase (default None).

        :param border_ratio: (Optional) relative width on the
          borders of the filter range removed from the fitted values
          (default 0.1)

        """
        self.assert_params()

        if amplitude is not None:
            assert len(amplitude) == self.dimx, 'amplitude vector must have same shape as phase vector'
            
        sigmaref = orb.core.FilterFile(self.params.filter_name).get_phase_fit_ref()
        
        deg = int(deg)
        if deg < 0: raise ValueError('deg must be >= 0')

        if not 0 <= border_ratio < 0.5:
            raise ValueError(
                'border_ratio must be between 0 and 0.5')
            
        cm1_min, cm1_max = self.get_filter_bandpass_cm1()
        
        cm1_border = np.abs(cm1_max - cm1_min) * border_ratio
        cm1_min += cm1_border
        cm1_max -= cm1_border
        
        phase = np.copy(self.data).astype(float)
        ok_phase = phase[int(self.axis(cm1_min)):int(self.axis(cm1_max))]
        if np.any(np.isnan(ok_phase)):
            raise orb.utils.err.FitError('phase contains nans in the filter passband')
        
        ok_axis = self.axis.data.astype(float)[int(self.axis(cm1_min)):int(self.axis(cm1_max))]
        if amplitude is not None:
            ok_amp = amplitude[int(self.axis(cm1_min)):int(self.axis(cm1_max))]
            if np.any(np.isnan(ok_amp)):
                raise orb.utils.err.FitError('amplitude contains nans in the filter passband')
        else:
            ok_amp = np.full_like(ok_phase, 1)
        
            
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
            orb.utils.validate.has_len(coeffs, deg + 1)
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
        
        def model(x, p):
            p = format_guess(p)
            return orb.utils.fft.phase_model(x, sigmaref, p)

        def diff(p):
            # weight = snr = amplitude because in this case noise is
            # distributed, i.e. snr is propto signal
            return (ok_phase - model(ok_axis, p)) * ok_amp

        try:
            _fit = scipy.optimize.leastsq(
                diff, guesses,
                full_output=True)
            pfit = _fit[0]
            pcov = _fit[1]
            perr = np.sqrt(np.diag(pcov) * np.std(_fit[2]['fvec'])**2)
            
        except Exception as e:
            raise orb.utils.err.FitError('Exception occured during phase fit: {}'.format(e))
        
        all_pfit = format_guess(pfit)
        all_perr = format_guess(perr)
        if coeffs is not None:
            all_perr[np.nonzero(~np.isnan(coeffs))] = np.nan
        
        logging.debug('fitted coeffs: {} ({})'.format(all_pfit, all_perr))
        if return_coeffs:
            return all_pfit, all_perr
        else:
            return self.__class__(model(self.axis.data.astype(float), pfit),
                                  self.axis, params=self.params)

    def subtract_low_order_poly(self, deg, border_ratio=0.1):
        """ low order polynomial substraction to suppress low
        frequency noise

        :param deg: Degree of the fitting polynomial. Must be >= 0.

        :param border_ratio: (Optional) relative width on the
          borders of the filter range removed from the fitted values
          (default 0.1)
        """
        return self.subtract(orb.core.Vector1d(self.polyfit(deg, border_ratio=border_ratio).project(self.axis)))

#################################################
#### CLASS Spectrum #############################
#################################################

class Spectrum(orb.core.Cm1Vector1d):
    """Spectrum class
    """
    def __init__(self, spectrum, err=None, axis=None, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray vector.

        :param axis: (optional) A 1d numpy.ndarray axis (default None)
          with the same size as vector.
        
        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are 'step', 'order', 'zpd_index', 'calib_coeff' (default
          None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        orb.core.Cm1Vector1d.__init__(self, spectrum, err=err, axis=axis,
                                      params=params, **kwargs)
            
        if not np.iscomplexobj(self.data):
            logging.debug('input spectrum is not complex')
            self.data = self.data.astype(complex)

                   
    def get_phase(self):
        """return phase"""
        nans = np.isnan(self.data)
        _data = np.copy(self.data)
        _data[nans] = 0
        _phase = np.unwrap(np.angle(_data))
        _phase[nans] = np.nan
        return Phase(_phase, axis=self.axis, params=self.params)

    def get_amplitude(self):
        """return amplitude"""
        return np.abs(self.data)

    def get_real(self):
        """Return the real part"""
        return np.copy(self.data.real)

    def get_imag(self):
        """Return the imaginary part"""
        return np.copy(self.data.imag)

    def apodize(self, coeff):
        """Return an apodized spectrum (works well only if spectrum is complex)"""
        spec = self.copy()
        if not np.any(np.iscomplex(self.data)):
            logging.warning('spectrum is not complex. Apodizing will not give ideal results')
        spec.data[np.isnan(spec.data)] = 0.
        zp_spec = np.concatenate([spec.data, spec.data[::-1]])
        spec_ifft = scipy.fft.ifft(zp_spec)
        x = np.linspace(0, 1, spec.dimx)
        x = np.concatenate([x, x[::-1]])
        w = orb.utils.fft.gaussian_window(coeff, x)
        spec.data = scipy.fft.fft(spec_ifft * w)[:spec.dimx]
        if not np.any(np.iscomplex(self.data)):
            spec.data.imag.fill(0.)
        return spec

    
    def inverse_transform(self):
        """Return the inverse transform of the spectrum. This should be the
        exact inverse transform of Interferogram.transform()
        """
        spec = self.copy()
        spec.data[np.isnan(spec.data)] = 0.
        if np.any(np.iscomplex(spec.data)):
            iscomplex = True
            raise NotImplementedError()
        else:
            iscomplex = False

        # spectrum is flipped if order is even
        if self.has_params():
            if int(self.params.order)&1:
                spec.reverse()

        spec.data = spec.data.astype(np.complex128)
        
        a = np.concatenate((spec.data, np.zeros(spec.dimx)))
        a_ifft = scipy.fft.ifft(a)
        a_interf = np.concatenate(
            (a_ifft[-self.params.zpd_index:], 
             a_ifft[:self.params.step_nb - self.params.zpd_index]))

        # compensate energy lost in the imaginary part !
        a_interf = a_interf.real.astype(float) * 2.
        
        interf = Interferogram(a_interf, params=self.params)
        return interf 
                
        
    def zpd_shift(self, shift):
        """correct spectrum phase from shifted zpd"""
        self.correct_phase(
            np.arange(self.dimx, dtype=float)
            * -1. * shift * np.pi / self.dimx)
        
    def correct_phase(self, phase):
        """Correct spectrum phase

        :param phase: can be a 1d array or a Phase instance.
        """
        if isinstance(phase, Phase):
            phase = phase.project(self.axis).data
        else:
            orb.utils.validate.is_1darray(phase, object_name='phase')
            phase = orb.core.Vector1d(phase, axis=self.axis).data
            
        if phase.shape[0] != self.dimx:
            logging.warning('phase does not have the same size as spectrum. It will be interpolated.')
            phase = orb.utils.vector.interpolate_size(phase, self.dimx, 1)
            
        self.data *= np.exp(-1j * phase)

    def interpolate(self, axis, quality=10, timing=False):
        """Resample spectrum by interpolation over the given axis

        This is a simple wrapper around core.Vector1d.project which
        now integrates all the fft interpolation method when data is
        complex.

        It has been kept for backward compatibility. Please use the
        project method.
        """
        return self.project(axis, quality=quality, timing=timing)        

    def prepare_fit(self, lines, fmodel='sinc', nofilter=True, **kwargs):
        """Return the input parameters, which can be reused to accelerate
        similar fits.
        """
        input_params_start_time = time.time()

        try:
            lines = list(lines)
        except Exception:
            raise TypeError("lines should be a list of lines, e.g. ['Halpha'] or [15534.25] but has type {}".format(type(lines)))

        # compute incident angle theta
        theta = orb.utils.spectrum.corr2theta(
            self.params['calib_coeff'])

        theta_orig = orb.utils.spectrum.corr2theta(
            self.params['calib_coeff_orig'])

        # fmodel
        kwargs['fmodel'] = fmodel

        # signal_range
        if nofilter:
            filter_name = None
            if 'signal_range' in kwargs:
                signal_range = kwargs['signal_range']
                del kwargs['signal_range']
            else:
                signal_range = self.get_filter_bandpass_cm1()

        else:
            filter_name = self.params.filter_name
            if 'signal_range' in kwargs:
                signal_range = kwargs['signal_range']
                del kwargs['signal_range']
            else:
                signal_range = None

        # check kwargs formatting
        def to_tuple(ikey, length=1):
            if isinstance(kwargs[ikey], str):
                kwargs[ikey] = tuple([kwargs[ikey],])
            elif np.isscalar(kwargs[ikey]):
                kwargs[ikey] = tuple([kwargs[ikey],])
            

        for ikey in kwargs:
            if '_def' in ikey:
                to_tuple(ikey)
                if len(kwargs[ikey]) <= 1:
                    kwargs[ikey] = kwargs[ikey] * len(lines)
                
            if '_cov' in ikey:
                to_tuple(ikey)
                
                
        inputparams = orb.fit._prepare_input_params(
            self.params.step_nb,
            lines,
            self.params.step,
            self.params.order,
            self.params.nm_laser,
            theta,
            self.params.zpd_index,
            theta_orig=theta_orig,
            wavenumber=True,
            filter_name=filter_name,
            apodization=self.params.apodization,
            signal_range=signal_range,
            **kwargs)
        
        logging.debug('prepare_fit timing: {}'.format(time.time() - input_params_start_time))
        
        return inputparams.convert(), kwargs

    def prepared_fit(self, inputparams, max_iter=None, nogvar=False, **kwargs):
        """Run a fit already prepared with prepare_fit() method.
        """
        start_time = time.time()
        kwargs_orig = dict(kwargs)
        
        # compute incident angle theta
        theta = orb.utils.spectrum.corr2theta(
            self.params['calib_coeff'])

        theta_orig = orb.utils.spectrum.corr2theta(
            self.params['calib_coeff_orig'])
        
        # recompute the fwhm guess
        if 'calib_coeff_orig' not in self.params:
            self.params['calib_coeff_orig'] = self.params['calib_coeff']
            
        fwhm_guess_cm1 = orb.utils.spectrum.compute_line_fwhm(
            self.dimx - self.params['zpd_index'],
            self.params['step'], self.params['order'],
            self.params['calib_coeff_orig'],
            wavenumber=self.params['wavenumber'])

        fwhm_guess = [fwhm_guess_cm1] * inputparams['allparams']['line_nb']

        if not 'fwhm_guess' in kwargs:
            kwargs['fwhm_guess'] = fwhm_guess
        else:
            if 'sinc' in kwargs['fmodel']:
                raise ValueError('fwhm_guess must not be in kwargs. It must be set via theta_orig parameter.')
            if isinstance(kwargs['fwhm_def'], str):
                kwargs['fwhm_def'] = list([kwargs['fwhm_def']]) * inputparams['allparams']['line_nb']
            if np.size(np.squeeze(kwargs['fwhm_guess'])) <= 1:
                kwargs['fwhm_guess'] = np.squeeze(list([np.squeeze(kwargs['fwhm_guess'])]) * inputparams['allparams']['line_nb'])
                
        logging.debug('recomputed fwhm guess: {}'.format(kwargs['fwhm_guess']))
        
        if max_iter is None:
            max_iter = max(100 * inputparams['allparams']['line_nb'], 1000)

        spectrum = np.copy(self.data)
        spectrum[np.isnan(spectrum)] = 0

        err = None
        if self.has_err():
            err = np.copy(self.err)
            err[np.isnan(err)] = 0.
        try:
            warnings.simplefilter('ignore')
            _fit = orb.fit._fit_lines_in_spectrum(
                spectrum, inputparams,
                fit_tol=1e-10,
                compute_mcmc_error=False,
                max_iter=max_iter,
                nogvar=nogvar,
                vector_err=err,
                **kwargs)
            warnings.simplefilter('default')

        except Exception as e:
            logging.warning('Exception occured during fit: {}'.format(e))
            import traceback
            print((traceback.format_exc()))

            return []

        # handle sincgauss unstability when broadening is too small or SNR is too low

        # check if model is sincgauss
        is_sincgauss = False
        if 'fmodel' in kwargs_orig:
            if kwargs_orig['fmodel'] == 'sincgauss':
                is_sincgauss = True
        else:                        
            for imodel in range(len(inputparams['models'])):
                if inputparams['models'][imodel][0] == orb.fit.Cm1LinesModel:                    
                    if inputparams['params'][imodel]['fmodel'] == 'sincgauss':
                        is_sincgauss = True

        # check fixed sigma parameters
        is_fixed = False
        if is_sincgauss:
            for imodel in range(len(inputparams['models'])):
                if inputparams['models'][imodel][0] == orb.fit.Cm1LinesModel:
                    is_fixed = np.array(inputparams['params'][imodel]['sigma_def']) == 'fixed'
                        
        if (_fit != []
            and is_sincgauss
            and np.any(np.isnan(_fit['broadening_err']) * ~is_fixed)):
            logging.info('bad sigma value for sincgauss model, fit recomputed with a sinc model')

            # clean kwargs from sigma related params
            new_kwargs = dict(kwargs_orig)
            for ikey in list(new_kwargs.keys()):
                if 'sigma_' in ikey:
                    del new_kwargs[ikey]
                    
            new_kwargs['fmodel'] = 'sinc'

            try:
                new_inputparams = inputparams.convert()
            except AttributeError:
                import copy
                new_inputparams = copy.deepcopy(inputparams)

            # clean inputparams
            for imodel in range(len(new_inputparams['models'])):    
                if new_inputparams['models'][imodel][0] == orb.fit.Cm1LinesModel:
                    for ikey in list(new_inputparams['params'][imodel].keys()):
                        if 'sigma_' in ikey:
                            del new_inputparams['params'][imodel][ikey]

            return self.prepared_fit(
                new_inputparams, max_iter=max_iter,
                nogvar=nogvar, **new_kwargs)


        del spectrum
                
        logging.debug('total fit timing: {}'.format(time.time() - start_time))
        
        return _fit
        
    def fit(self, lines, fmodel='sinc', nofilter=True,
            max_iter=None, nogvar=False,
            **kwargs):
        """Fit lines in a spectrum

        Wrapper around orb.fit.fit_lines_in_spectrum.

        :param lines: lines to fit.
        
        :param max_iter: (Optional) Maximum number of iterations (default None)
        
        :param kwargs: kwargs used by orb.fit.fit_lines_in_spectrum.
        """
        if 'snr_guess' in kwargs:
            logging.warning(
                "snr_guess is deprecated. It's value is not used anymore")
            kwargs.pop('snr_guess')
        
        kwargs_orig = dict(kwargs)

        # prepare input params
        kwargs_orig = dict(kwargs)
        inputparams, kwargs = self.prepare_fit(
            lines, fmodel=fmodel, nofilter=nofilter, **kwargs)

        fit = self.prepared_fit(
            inputparams, max_iter=max_iter, nogvar=nogvar,
            **kwargs)
        
        return fit

    def prepare_velocity_estimate(self, lines, vel_range, precision=10):
        lines_cm1 = orb.core.Lines().get_line_cm1(lines)
        oversampling_ratio = (self.params.zpd_index
                              / (self.params.step_nb - self.params.zpd_index) + 1)
        combs, vels = orb.utils.fit.prepare_combs(lines_cm1, self.axis.data, vel_range, oversampling_ratio, precision)
        return combs, vels, self.axis(self.params.filter_range).astype(int), lines_cm1, oversampling_ratio

    def estimate_velocity_prepared(self, combs, vels, filter_range_pix):
        return orb.utils.fit.estimate_velocity_prepared(
            self.data.real, vels, combs, filter_range_pix)

    def estimate_parameters(self, lines, vel_range, precision=10):
        (combs, vels, filter_range_pix,
         lines_cm1, oversampling_ratio) = self.prepare_velocity_estimate(
             lines, vel_range, precision=precision)  
        vel = self.estimate_velocity_prepared(combs, vels, filter_range_pix)
        fluxes = self.estimate_flux(lines, vel)
        return vel, fluxes
    
    def estimate_flux(self, lines, vel):
        lines_cm1 = orb.core.Lines().get_line_cm1(lines)
        oversampling_ratio = (self.params.zpd_index
                              / (self.params.step_nb - self.params.zpd_index) + 1)
        return orb.utils.fit.estimate_flux(
            self.data.real, self.axis.data, lines_cm1, vel,
            self.axis(self.params.filter_range).astype(int), oversampling_ratio)

        if fit != [] and fmodel == 'sincgauss' and np.all(np.isnan(fit['broadening'])):
            logging.info('bad sigma value for sincgauss model, fit recomputed with a sinc model')
            
            new_kwargs = dict(kwargs_orig)
            for ikey in list(new_kwargs.keys()):
                if 'sigma_' in ikey:
                    del new_kwargs[ikey]
                    
            return self.fit(lines, fmodel='sinc', nofilter=nofilter,
                            snr_guess=snr_guess, max_iter=max_iter, **new_kwargs)
        
        return fit

#################################################
#### CLASS RealSpectrum #########################
#################################################

class RealSpectrum(Spectrum):
    """Spectrum class computed from real interferograms (in counts)
    """
    def __init__(self, *args, **kwargs):
        """Init method.

        important parameters that must be supplied in params (if the
          spectrum does not come from Interfrogram.transform()):

        - pixels: number of integrated pixels (default 1)

        - source_counts: total number of counts in the source. Must be
          a float. Used to estimate the modulation
          efficiency.
        """
        Spectrum.__init__(self, *args, **kwargs)
        
        # check if integrated
        if 'pixels' not in self.params:
            self.params.reset('pixels', 1)

        # recompute counts in the original interferogram if needed
        if 'source_counts' not in self.params:
            logging.debug('source_counts not set, computed from spectrum, me computation will be wrong')
            self.params['source_counts'] = self.compute_counts_in_spectrum()
            
        # compute photon noise
        if not self.has_err():
            logging.debug('err vector not supplied, this RealSpectrum, is not real :)')
            #np.ones_like(self.data) * np.sqrt(self.params.source_counts)
            
    def compute_counts_in_spectrum(self):
        """Return the number of counts in the spectrum
        """
        if self.has_err():
            _data = gvar.gvar(np.abs(self.data), np.abs(self.err))
        else:
            _data = np.abs(self.data)
            
        xmin, xmax = self.get_filter_bandpass_pix(border_ratio=-0.05)
        _data[:xmin] = 0
        _data[xmax:] = 0
        counts =  np.sum(_data)
        del _data
        return counts
    
    def compute_me(self):
        """Return the modulation efficiency, computed from the ratio between
        the number of counts in the original interferogram and the
        number of counts in the spectrum.
        """
        _source_counts = gvar.gvar(self.params.source_counts,
                                   np.sqrt(self.params.source_counts))
        _counts_in_spec = self.compute_counts_in_spectrum()
        me = _counts_in_spec / _source_counts
        del _counts_in_spec
        del _source_counts
        return me

    
    def subtract_sky(self, sky):
        """Subtract spectrum interferogram (or any background spectrum).
        
        :param sky: A spectrum instance.

        The values of the parameter 'pixels' in both this
        interferogram and the sky interferogram should be set to the
        number of integrated pixels.
        """
        sky = sky.copy()
        sky.data *= float(self.params.pixels) / float(sky.params.pixels)
        return self.subtract(sky)

#################################################
#### CLASS StandardSpectrum #####################
#################################################

class StandardSpectrum(RealSpectrum):
    """Spectrum class for standard spectrum computed from real
    interferograms (in counts).
    
    Spectrum unit should be in counts.
    """

    convert_params = {'AIRMASS':'airmass',
                      'EXPTIME':'exposure_time',
                      'FILTER':'filter_name',
                      'INSTRUME':'instrument',
                      'CAMERA': 'camera'}
    
    def __init__(self, *args, **kwargs):
        RealSpectrum.__init__(self, *args, **kwargs)
        self.params.reset('object_name', ''.join(
            self.params.OBJECT.strip().split()).upper())
        self.params.reset('instrument', self.params.instrument.lower())
        
        self.params.reset(
            'camera', orb.utils.misc.convert_camera_parameter(
                self.params.camera))

        if not self.has_param('airmass'):
            logging.debug('airmass not set, automatically set to 1')
            self.params['airmass'] = 1.
    

    def get_standard(self):
        """Return the standard spectrum
        """
        std = orb.photometry.Standard(self.params.object_name,
                                      instrument=self.params.instrument)
        airmass = self.params.airmass
        if np.size(airmass) > 1:
            airmass = np.nanmedian(airmass)
        else:
            airmass = float(airmass)
            
        sim = std.simulate_measured_flux(
            self.params.filter_name, self.axis,
            camera_index=self.params.camera,
            modulated=True, airmass=airmass)

        assert np.all(np.isclose(sim.axis.data - self.axis.data, 0)), 'axes must be equal'
        
        sim.data[np.isnan(sim.data)] = 0

        return sim

    def to_counts_s(self):
        """Return spectrum in counts/s
        """
        spe = self.copy()
        spe.data /= self.dimx * self.params.exposure_time # data unit changed to counts/s
        spe.data[np.isnan(spe.data)] = 0
        return spe

    def compute_flux_correction_vector(self, deg=None, resolution=None, return_residual=False):
        """Compute flux correction vector by fitting a simulated model of the
        standard star on the observed spectrum.
        """
        def model(_sim, *p):
            m = _sim * np.polynomial.polynomial.polyval(np.arange(_sim.size), p)
            return np.array(m, dtype=float)

        ff = orb.core.FilterFile(self.params.filter_name)
        
        if deg is None:
            deg = int(ff.params.flux_correction_order)
        logging.info('flux correction fitted with an order {} polynomial'.format(deg))
        
        if resolution is None:
            resolution = int(ff.params.flux_correction_resolution)
        logging.info('flux correction fitted with a resolution R={}'.format(resolution))
            
        sim = self.get_standard().change_resolution(resolution) # standard flux in counts/s
               
        spe = self.to_counts_s().change_resolution(resolution)

        
        xmin, xmax = self.get_filter_bandpass_pix(
            border_ratio=0.1)
        
        sim.data[:xmin] = sim.data[xmin]
        sim.data[xmax:] = sim.data[xmax]
        spe.data[:xmin] = sim.data[xmin]
        spe.data[xmax:] = sim.data[xmax]

        sigma = np.ones_like(spe.data, dtype=float)
        sigma[:xmin] = 1e7
        sigma[xmax:] = 1e7
        
        fit = scipy.optimize.curve_fit(
            model, sim.data, spe.data,
            p0=np.ones(deg+1, dtype=float)/10.,
            sigma=sigma)
        
        poly = np.polynomial.polynomial.polyval(
            np.arange(sim.dimx), fit[0])

        # this is not a real residual but the fitted data so that both
        # the residual and the fitted polynomial can be plotted
        # together directly.
        if return_residual: 
            residual = spe.copy()
            residual.data /= sim.data
            residual.data[:xmin] = np.nan
            residual.data[xmax:] = np.nan
            residual.data /= np.nanmean(poly[xmin:xmax])

        poly /= np.nanmean(poly[xmin:xmax])
        poly[:xmin] = poly[xmin]
        poly[xmax:] = poly[xmax]

        # smooth polynomial on the borders
        poly = orb.utils.vector.smooth(poly, deg=int(0.10*poly.size))
        
        eps = orb.core.Cm1Vector1d(
            poly, axis=self.axis, params=self.params)

        if not return_residual:
            return eps
        else:
            return eps, residual

#################################################
#### CLASS PhaseMaps ############################
#################################################
class PhaseMaps(orb.core.Tools):

    phase_maps = None

    def __init__(self, phase_maps_path,
                 overwrite=False, indexer=None, **kwargs):
        """Initialize PhaseMaps class.

        :param phase_maps_path: path to the hdf5 file containing the
          phase maps.
      
        :param overwrite: (Optional) If True existing FITS files will
          be overwritten (default False).

        :param indexer: (Optional) Must be a :py:class:`core.Indexer`
          instance. If not None created files can be indexed by this
          instance.    

        :param kwargs: Kwargs are :meth:`core.Tools` properties.
        """
        with orb.utils.io.open_hdf5(phase_maps_path, 'r') as f:
            if 'instrument' not in f.attrs:
                raise StandardError('instrument not in cube attributes')
            
            instrument = f.attrs['instrument']
            if not isinstance(instrument, str):
                instrument = instrument.decode()
                
            kwargs['instrument'] = instrument

        orb.core.Tools.__init__(self, **kwargs)
        self.params = orb.core.ROParams()
        
        self.overwrite = overwrite
        self.indexer = indexer

        self.dimx_unbinned = self.config['CAM1_DETECTOR_SIZE_X']
        self.dimy_unbinned = self.config['CAM1_DETECTOR_SIZE_Y']

        self.phase_maps = list()
        self.phase_maps_err = list()
        with orb.utils.io.open_hdf5(phase_maps_path, 'r') as f:
            self.phase_maps_path = phase_maps_path
            if 'calibration_coeff_map' in f:
                self.calibration_coeff_map = f['calibration_coeff_map'][:]
            else: 
                self.calibration_coeff_map = f['phase_maps_coeff_map'][:]
            if 'cm1_axis' in f:
                self.axis = f['cm1_axis'][:]
            else:
                self.axis = f['phase_maps_cm1_axis'][:]
                
            self.theta_map = orb.utils.spectrum.corr2theta(
                self.calibration_coeff_map)
            
            loaded = False
            iorder = 0
            while not loaded:
                ipm_path ='phase_map_{}'.format(iorder)
                ipm_path_err ='phase_map_err_{}'.format(iorder)
                if ipm_path in f:
                    ipm_mean = f[ipm_path][:]
                    if ipm_path_err in f:
                        ipm_sdev = f[ipm_path_err][:]
                        self.phase_maps.append(ipm_mean)
                        self.phase_maps_err.append(ipm_sdev)
                        
                    else: raise ValueError('Badly formatted phase maps file')
                else:
                    loaded = True
                    continue
                iorder += 1

            if len(self.phase_maps) == 0: raise ValueError('No phase maps in phase map file')

            # add params
            for ikey in list(f.attrs.keys()):
                self.params[ikey] = f.attrs[ikey]


        self.sigmaref = orb.core.FilterFile(self.params.filter_name).get_phase_fit_ref()
        
        # detect binning
        self.dimx = self.phase_maps[0].shape[0]
        self.dimy = self.phase_maps[0].shape[1]
        
        binx = int(self.dimx_unbinned/self.dimx)
        biny = int(self.dimy_unbinned/self.dimy)
        if binx != biny: raise Exception('Binning along x and y axes is different ({} != {})'.format(binx, biny))
        else: self.binning = binx

        logging.info('Phase maps loaded : order {}, shape ({}, {}), binning {}'.format(
            len(self.phase_maps) - 1, self.dimx, self.dimy, self.binning))

        self._compute_unbinned_maps()

    def _compute_unbinned_maps(self):
        """Compute unbinnned maps"""
        # unbin maps
        self.unbinned_maps = list()
        self.unbinned_maps_err = list()
        
        for iorder in range(len(self.phase_maps)):
            self.unbinned_maps.append(orb.cutils.unbin_image(
                gvar.mean(self.phase_maps[iorder]),
                self.dimx_unbinned, self.dimy_unbinned))
            self.unbinned_maps_err.append(orb.cutils.unbin_image(
                gvar.sdev(self.phase_maps[iorder]),
                self.dimx_unbinned, self.dimy_unbinned))

        self.unbinned_calibration_coeff_map = orb.cutils.unbin_image(
            self.calibration_coeff_map,
            self.dimx_unbinned, self.dimy_unbinned)
        
    def _isvalid_order(self, order):
        """Validate order
        
        :param order: Polynomial order
        """
        if not isinstance(order, int): raise TypeError('order must be an integer')
        order = int(order)
        if order in range(len(self.phase_maps)):
            return True
        else:
            raise ValueError('order must be between 0 and {}'.format(len(self.phase_maps)))

    def get_map(self, order):
        """Return map of a given order

        :param order: Polynomial order
        """
        if self._isvalid_order(order):
            return np.copy(self.phase_maps[order])

    def get_map_err(self, order):
        """Return map uncertainty of a given order

        :param order: Polynomial order
        """
        if self._isvalid_order(order):
            return np.copy(self.phase_maps_err[order])

    def get_mapped_model(self, order):
        """Return mapped model"""
        import orb.utils.stats
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _phase_map = self.get_map(order)
        _phase_map_err = self.get_map_err(order)
        _phase_map_err[_phase_map_err > np.nanmedian(_phase_map_err)
                       + 3 * orb.utils.stats.unbiased_std(
                           _phase_map_err)] = np.nan
        _phase_map[np.isnan(_phase_map_err)] = np.nan
        _phase_map_err[~np.isnan(_phase_map_err)] = 1
        _phase_map_err.fill(1.)

        model, err, _ = orb.utils.image.fit_map_zernike(_phase_map, 1/_phase_map_err, 5)
        
        return model, err

    def modelize(self):
        """Replace phase maps by their model inplace
        """
        model, err = self.get_mapped_model(0)
        self.phase_maps[0] = model
        
        model, err = self.get_mapped_model(1)
        self.phase_maps[1] = model

        for iorder in range(2, len(self.phase_maps)):
            self.phase_maps[iorder] = (np.ones_like(self.phase_maps[iorder])
                                       * np.nanmean(self.phase_maps[iorder]))
        self._compute_unbinned_maps()


    def reverse_polarity(self):
        """Add pi to the order 0 phase map to reverse polarity of the
        corrected spectrum.
        """
        self.phase_maps[0] += np.pi
        self._compute_unbinned_maps()


    def get_coeffs(self, x, y, unbin=False):
        """Return coeffs at position x, y in the maps. x, y are binned
        position by default (set unbin to True to get real positions
        on the detector)

        :param x: X position (dectector position)
        :param y: Y position (dectector position)

        :param unbin: If True, positions are unbinned position
          (i.e. real positions on the detector) (default False).
        """
        x, y = self.validate_xy(x, y, unbin=unbin)
        
        coeffs = list()
        for iorder in range(len(self.phase_maps)):
            if unbin:
                coeffs.append(self.unbinned_maps[iorder][x, y])
            else:
                coeffs.append(self.phase_maps[iorder][x, y])
                
        return coeffs

    def validate_xy(self, x, y, unbin=False):
        x = int(x)
        y = int(y)
        if unbin:
            orb.utils.validate.index(x, 0, self.dimx_unbinned, clip=False)
            orb.utils.validate.index(y, 0, self.dimy_unbinned, clip=False)
        else:
            orb.utils.validate.index(x, 0, self.dimx, clip=False)
            orb.utils.validate.index(y, 0, self.dimy, clip=False)
        return x, y
    
    def get_phase(self, x, y, unbin=False, coeffs=None):
        """Return a phase instance at position x, y in the maps. x, y are
        binned position by default (set unbin to True to get real
        positions on the detector)
        
        :param x: X position (dectector position)
        :param y: Y position (dectector position)

        :param unbin: If True, positions are unbinned position
          (i.e. real positions on the detector) (default False).

        :param coeffs: Used to set some coefficients to a given
          value. If not None, must be a list of length = order. set a
          coeff to a np.nan to use the phase map value.
        """
        _coeffs = self.get_coeffs(x, y, unbin=unbin)
        if coeffs is not None:
            orb.utils.validate.has_len(coeffs, len(self.phase_maps))
            for i in range(len(coeffs)):
                if coeffs[i] is not None:
                    if not np.isnan(coeffs[i]):
                        _coeffs[i] = coeffs[i]

        x, y = self.validate_xy(x, y, unbin=unbin)
        if unbin:
            ctheta = 1. / self.unbinned_calibration_coeff_map[x, y]
        else:
            ctheta = 1. / self.calibration_coeff_map[x, y]


        model = orb.utils.fft.phase_model(self.axis, self.sigmaref, _coeffs)
        return Phase(
            model.astype(float),
            axis=self.axis, params=self.params)
        
    def generate_phase_cube(self, path, coeffs=None, x_range=None, y_range=None, silent=False):
        """Generate a phase cube from the given phase maps.

        :param coeffs: Used to set some coefficients to a given
          value. If not None, must be a list of length = order. set a
          coeff to a np.nan to use the phase map value.

        """
        
        if x_range is None:
            x_range = (0, self.dimx)
        if y_range is None:
            y_range = (0, self.dimy)

        phase_cube = np.empty((x_range[1] - x_range[0],
                               y_range[1] - y_range[0],
                               self.axis.size), dtype=float)
        phase_cube.fill(np.nan)


        if not silent:
            progress = orb.core.ProgressBar(self.dimx)
            
        for ii in range(x_range[0],  x_range[1]):
            if not silent:
                progress.update(
                    ii, info="computing column {}/{}".format(ii, self.dimx))
                
            for ij in range(y_range[0],  y_range[1]):
                phase_cube[ii - x_range[0], ij - y_range[0], :] = self.get_phase(
                    ii, ij, coeffs=coeffs).data
                
        if not silent:
            progress.end()

        if path is not None:
            orb.utils.io.write_fits(path, phase_cube, overwrite=True)
            
        return phase_cube
        
    
    def unwrap_phase_map_0(self):
        """Unwrap order 0 phase map.


        Phase is defined modulo pi/2. The Unwrapping is a
        reconstruction of the phase so that the distance between two
        neighboor pixels is always less than pi/4. Then the real phase
        pattern can be recovered and fitted easily.
    
        The idea is the same as with np.unwrap() but in 2D, on a
        possibly very noisy map, where a naive 2d unwrapping cannot be
        done.
        """
        self.phase_map_order_0_unwraped = orb.utils.image.unwrap_phase_map0(
            np.copy(self.phase_maps[0]))
        
        # Save unwraped map
        phase_map_path = self._get_phase_map_path(0, phase_map_type='unwraped')

        orb.utils.io.write_fits(phase_map_path,
                            orb.cutils.unbin_image(
                                np.copy(self.phase_map_order_0_unwraped),
                                self.dimx_unbinned,
                                self.dimy_unbinned), 
                            fits_header=self._get_phase_map_header(
                                0, phase_map_type='unwraped'),
                            overwrite=True)
        if self.indexer is not None:
            self.indexer['phase_map_unwraped_0'] = phase_map_path


#################################################
#### CLASS HighOrderPhaseMaps ###################
#################################################

class HighOrderPhaseCube(orb.core.Data):
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = orb.core.Tools(self.params.instrument).config
        self.full_shape = (self.config['CAM1_DETECTOR_SIZE_X'],
                           self.config['CAM1_DETECTOR_SIZE_Y'])
        
        self.x_map = np.linspace(0, self.full_shape[0], self.dimx, endpoint=False)
        self.x_map += np.diff(self.x_map)[0] / 2.
        self.y_map = np.linspace(0, self.full_shape[1], self.dimy, endpoint=False)
        self.y_map += np.diff(self.y_map)[0] / 2.
        self.maps = list()
        for i in range(self.dimz):
            self.maps.append(scipy.interpolate.interp2d(
                self.x_map, self.y_map, self.data[:,:,i]))
            
        
    def get_phase_coeffs(self, x, y):
        return [imap(x, y) for imap in self.maps]
            
    def get_phase(self, x, y, asarr=False, axis=None):
        if axis is None:
            axis = self.params.phase_axis.astype(float)
        else:
            if isinstance(axis, orb.core.Axis):
                axis = axis.data
            assert isinstance(axis, np.ndarray), 'axis must be a numpy.ndarray or a orb.core.Axis instance'
            axis = axis.astype(float)
            
        arr = np.polyval(self.get_phase_coeffs(x, y), axis).astype(float)
        if asarr:
            return arr
        else:
            return orb.fft.Phase(arr, axis=axis, params=self.params)
        
    def get_map(self, order):
        return self.maps[-(order+1)](self.x_map, self.y_map)
    
    def generate_phase_cube(self, path, dimx, dimy, axis=None):
        if axis is None:
            axis = self.params.phase_axis.astype(float)
            
        x_map = np.linspace(0, self.full_shape[0], dimx, endpoint=False)
        x_map += np.diff(x_map)[0] / 2.
        y_map = np.linspace(0, self.full_shape[1], dimy, endpoint=False)
        y_map += np.diff(y_map)[0] / 2.
        
        phase_cube = np.empty((x_map.size, y_map.size, axis.shape[0]), dtype=float)
        phase_cube.fill(np.nan)
        progress = orb.core.ProgressBar(x_map.size)
        for ii in range(x_map.size):
            progress.update(ii)
            for ij in range(y_map.size):
                phase_cube[ii,ij] = self.get_phase(x_map[ii], y_map[ij], asarr=True, axis=axis)
        progress.end()
        if path is not None:
            orb.utils.io.write_fits(path, phase_cube, overwrite=True)
        return phase_cube
