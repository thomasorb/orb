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
import scipy
import gvar

import fit


#################################################
#### CLASS Interferogram ########################
#################################################

class Interferogram(core.Vector1d):

    """Interferogram class.
    """
    needed_params = 'step', 'order', 'zpd_index', 'calib_coeff', 'filter_file_path', 'exposure_time'
        
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
        core.Vector1d.__init__(self, interf, axis=axis, err=err, params=params, **kwargs)

        if self.params.zpd_index < 0 or self.params.zpd_index >= self.dimx:
            raise ValueError('zpd must be in the interferogram')

        # opd axis (in cm) is automatically computed from the parameters
        opdaxis = 1e-7 * (np.arange(self.dimx) * self.params.step
                - (self.params.step * self.params.zpd_index))
        if self.axis is None:
            self.axis = core.Axis(opdaxis)
        elif np.any(opdaxis != self.axis.data):
            raise StandardError('provided axis is inconsistent with the opd axis computed from the observation parameters')
        
        if self.axis.dimx != self.dimx:
            raise ValueError('axis must have the same size as the interferogram')
        
    def crop(self, xmin, xmax):
        """Crop data. see Vector1d.crop()"""
        out = core.Vector1d.crop(self, xmin, xmax, returned_class=core.Vector1d)
        
        if out.axis.data.size <= 1:
            raise ValueError('cropping cannot return an interferogram with less than 2 samples. Use self.data[index] instead.')
        
        if out.params.zpd_index < xmin or out.params.zpd_index >= xmax:
            raise RuntimeError('ZPD is not anymore in the returned interferogram')

        zpd_index = out.params.zpd_index - xmin
        out.params.reset('zpd_index', zpd_index)
        return self.__class__(out)
        
        
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
        
        if not (0 <= self.params.zpd_index <= self.dimx):
            raise ValueError('zpd index must be >= 0 and <= interferogram size')
        
        x = np.arange(self.dimx, dtype=float) - self.params.zpd_index
        x /= max(np.abs(x[0]), np.abs(x[-1]))
        
        if window_type is None: return
        elif window_type == '1.0' : return
        elif window_type == 'learner95':
            window = utils.fft.learner95_window(x)
        else:
            window = utils.fft.gaussian_window(window_type, x)

        self.data *= window
        if self.has_err():
            self.err *= window


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

    def multiply_by_mertz_ramp(self):
        """Multiply by Mertz (1976) ramp function to avoid counting
        symmetric samples twice and reduce emission lines contrast wrt
        the background.

        :return: Mertz ramp as a 1d np.ndarray
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
            warnings.warn('interferogram is mostly symmetric. The use of Mertz ramp should be avoided.')
            
        self.data *= zeros_vector
        if self.has_err():
            self.err *= zeros_vector

        return zeros_vector

    def transform(self):
        """zero padded fft.
          
        :return: A Spectrum instance (or a core.Vector1d instance if
          interferogram is full of zeros or nans)

        .. note:: no phase correction is made here.
        """
        if np.any(np.isnan(self.data)):
            logging.debug('Nan detected in interferogram')
            return core.Vector1d(np.zeros(
                self.dimx, dtype=self.data.dtype) * np.nan)
        if len(np.nonzero(self.data)[0]) == 0:
            logging.debug('interferogram is filled with zeros')
            return core.Vector1d(np.zeros(
                self.dimx, dtype=self.data.dtype))
        
        # zero padding
        zp_nb = self.dimx * 2
        zp_interf = np.zeros(zp_nb, dtype=float)
        zp_interf[:self.dimx] = np.copy(self.data)

        # dft
        interf_fft = np.fft.fft(zp_interf)
        #interf_fft = interf_fft[:interf_fft.shape[0]/2+1]
        interf_fft = interf_fft[:self.dimx]
                                    
        # create axis
        if self.has_params():
            axis = core.Axis(utils.spectrum.create_cm1_axis(
                self.dimx, self.params.step, self.params.order,
                corr=self.params.calib_coeff))

        else:
            axis_step = (self.dimx - 1) / 2. / self.dimx
            axis_max = (self.dimx - 1) * axis_step
            axis = core.Axis(np.linspace(0, axis_max, self.dimx))

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
        new_interf.subtract_mean()
        if mertz:
            new_interf.multiply_by_mertz_ramp()
        return new_interf.transform()

    def get_phase(self):
        """Classical phase computation method. Returns a Phase instance."""
        new_interf = self.copy()
        new_interf = new_interf.symmetric()
        new_interf.subtract_mean()
        new_spectrum = new_interf.transform()
        return new_spectrum.get_phase().cleaned()

#################################################
#### CLASS Interferogram ########################
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
            raise TypeError('pixels must be an integer')

        # check source_counts
        if 'source_counts' not in self.params:
            if np.nanmin(self.data) < 0:
                warnings.warn('interferogram may be a combined interferogram. source_counts can be wrong')
            self.params.reset('source_counts', np.sum(np.abs(self.data)))
        elif not isinstance(self.params.source_counts, float):
            raise TypeError('source_counts must be a float')
            
        # compute photon noise
        if self.has_err(): # supplied err argument should have been
                           # read during Vector1d init. Only
                           # self.err is tested
            if np.nanmin(self.data) < 0:
                warnings.warn('interferogram may be a combined interferogram. photon noise can be wrong.')
            
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
        """Subtract sky interferogram. 
        
        The values of the parameter 'pixels' in both this
        interferogram and the sky interferogram should be set to the
        number of integrated pixels.
        """
        sky = sky.copy()
        sky.data *= self.params.pixels / sky.params.pixels
        self = self.math('subtract', sky)
        
        
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

class Phase(core.Cm1Vector1d):
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
        
        return Phase(data, axis=self.axis, params=self.params)        
        
    def polyfit(self, deg, coeffs=None, return_coeffs=False,
                border_ratio=0.1):
        """Polynomial fit of the phase
   
        :param deg: Degree of the fitting polynomial. Must be >= 0.

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
        deg = int(deg)
        if deg < 0: raise ValueError('deg must be >= 0')

        if not 0 <= border_ratio < 0.5:
            raise ValueError(
                'border_ratio must be between 0 and 0.5')
            
        cm1_min, cm1_max = self.get_filter_bandpass_cm1()
        
        cm1_border = np.abs(cm1_max - cm1_min) * border_ratio
        cm1_min += cm1_border
        cm1_max -= cm1_border

        weights = np.ones(self.dimx, dtype=float) * 1e-35
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

    def subtract_low_order_poly(self, deg, border_ratio=0.1):
        """ low order polynomial substraction to suppress low
        frequency noise

        :param deg: Degree of the fitting polynomial. Must be >= 0.

        :param border_ratio: (Optional) relative width on the
          borders of the filter range removed from the fitted values
          (default 0.1)
        """
        self = self.subtract(self.polyfit(deg, border_ratio=border_ratio))

#################################################
#### CLASS Spectrum #############################
#################################################

class Spectrum(core.Cm1Vector1d):
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
        core.Cm1Vector1d.__init__(self, spectrum, err=err, axis=axis,
                                  params=params, **kwargs)
        
        if not np.iscomplexobj(self.data):
            warnings.warn('input spectrum is not complex')
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
            utils.validate.is_1darray(phase, object_name='phase')
            phase = core.Vector1d(phase, axis=self.axis).data
            
        if phase.shape[0] != self.dimx:
            warnings.warn('phase does not have the same size as spectrum. It will be interpolated.')
            phase = utils.vector.interpolate_size(phase, self.dimx, 1)
            
        self.data *= np.exp(-1j * phase)
        
    def resample(self, axis):
        """Resample spectrum over the given axis

        :return: A new Spectrum instance

        .. warning:: Resampling is done via a DFT which is much slower
          than a FFT. But it gives perfect results with respect to
          interpolation.
        """
        raise NotImplementedError('Not implemented yet')
        ### should be done in the interferogram class instead (much easier to implement) ###
        ## interf = self.transform().data
        ## real_axis_step = float(self.axis[1] - self.axis[0])
        ## samples = (axis - self.axis[0]) / real_axis_step
        ## ok_samples = (0 <= samples) * (samples < self.dimx)
        ## if not np.all(ok_samples):
        ##     logging.debug('at least some samples are not in the axis range {}-{}'.format(
        ##         self.axis[0], self.axis[-1]))
        ##     dft_samples = samples[np.nonzero(ok_samples)]
        ## else:
        ##     dft_samples = samples
            
        ## dft_spec = cutils.complex_dft(interf, dft_samples.astype(float))
        ## if not np.all(ok_samples):
        ##     spec = np.empty_like(axis, dtype=complex)
        ##     spec.fill(np.nan)
        ##     spec[np.nonzero(ok_samples)] = dft_spec
        ## else:
        ##     spec = dft_spec
            
        ## return Spectrum(spec, axis=axis, params=self.params)


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
        zp_interf = np.zeros(self.dimx * quality, dtype=complex)
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
        return Spectrum(f(axis), axis=axis, params=self.params)


    def fit(self, lines, fmodel='sinc', nofilter=True, **kwargs):
        """Fit lines in a spectrum

        Wrapper around orb.fit.fit_lines_in_spectrum.

        :param lines: lines to fit.
        
        :param kwargs: kwargs used by orb.fit.fit_lines_in_spectrum.
        """
        if not isinstance(lines, list): raise TypeError("lines should be a list of lines, e.g. ['Halpha'] or [15534.25]")
        theta = utils.spectrum.corr2theta(
            self.params.calib_coeff)
        spectrum = np.copy(self.data)
        spectrum[np.isnan(spectrum)] = 0
        if nofilter: filter_file_path = None
        else: filter_file_path = self.params.filter_file_path
        return fit.fit_lines_in_spectrum(
            spectrum, lines, self.params.step, self.params.order,
            self.params.nm_laser, theta, self.params.zpd_index,
            filter_file_path=filter_file_path,
            fmodel=fmodel, **kwargs)

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

        # compute photon noise
        if not self.has_err():
            raise StandardError('err vector (photon noise) must be supplied')

        # recompute counts in the original interferogram if needed
        if 'source_counts' not in self.params:
            raise StandardError('source_counts must be supplied in the parameters')

    def compute_me(self):
        """Return the modulation efficiency, computed from the ratio between
        the number of counts in the original interferogram and the
        number of counts in the spectrum.
        """
        _data = gvar.gvar(np.abs(self.data), np.abs(self.err))
        xmin, xmax = self.get_filter_bandpass_pix(border_ratio=-0.05)
        _data[:xmin] = 0
        _data[xmax:] = 0
        _source_counts = gvar.gvar(self.params.source_counts,
                                   np.sqrt(self.params.source_counts))
        return np.sum(_data) / _source_counts
