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

import utils.fft
import utils.vector
import core
import cutils
import scipy

class Vector1d(object):
    """Basic 1d vector management class.

    Useful for checking purpose.
    """
    needed_params = ()
    optional_params = ()
    
    def __init__(self, vector, params=None, **kwargs):
        """Init method.

        :param vector: A 1d numpy.ndarray vector.

        :param params: (Optional) A dict containing additional
          parameters giving access to more methods. The needed params
          are stored in self.needed_params (default None).

        :param kwargs: (Optional) Keyword arguments, can be used to
          supply observation parameters not included in the params
          dict. These parameters take precedence over the parameters
          supplied in the params dictionnary.    
        """
        if isinstance(vector, self.__class__):
            vector = np.copy(vector.data)

        # checking
        if not isinstance(vector, np.ndarray):
            raise TypeError('input vector is a {} but must be a numpy.ndarray'.format(type(vector)))
        if vector.ndim != 1:
            vector = np.squeeze(vector)
            if vector.ndim != 1:
                raise TypeError('input vector has {} dims but must have only one dimension'.format(vector.ndim))
        if len(np.nonzero(vector)[0]) == 0:
            self.allzero = True
        else:
            self.allzero = False

        if np.all(np.isnan(vector)):
            self.allnan = True
        else:
            self.allnan = False

        if np.any(np.isnan(vector)):
            self.anynan = True
        else:
            self.anynan = False


        self.data = np.copy(vector)
        self.data_orig = np.copy(vector)
        self.step_nb = self.data.shape[0]

        # load parameters
        self.params = None        

        if params is not None:
            if not isinstance(params, dict):
                raise TypeError('params must be a dict')
            self.params = core.ROParams()
            for iparam in self.needed_params:
                try:
                    self.params[iparam] = kwargs[iparam]
                except KeyError: 
                    try:
                        self.params[iparam] = params[iparam]
                    except KeyError:
                        raise KeyError('param {} needed in params dict'.format(iparam))
            for iparam in self.optional_params:
                try:
                    self.params[iparam] = kwargs[iparam]
                except KeyError: 
                    try:
                        self.params[iparam] = params[iparam]
                    except KeyError: pass

        # check keyword arguments validity
        for iparam in kwargs:
            if iparam not in self.needed_params and iparam not in self.optional_params:
                raise KeyError('supplied keyword argument {} not understood'.format(iparam))

    def has_params(self):
        """Check the presence of observation parameters"""
        if self.params is None:
            return False
        else: return True

    def assert_params(self):
        """Assert the presence of observation parameters"""
        if self.params is None: raise StandardError('Parameters not supplied, please give: {} at init'.format(self.needed_params))

    def __getitem__(self, key):
        """Getitem special method"""
        return self.data.__getitem__(key)
    


class Axis(Vector1d):
    """Axis class"""

    def __init__(self, axis):
        """Init class with an axis vector

        :param axis: Regularly samples and naturally ordered axis.
        """
        Vector1d.__init__(self, axis)

        # check that axis is regularly sampled
        diff = np.diff(self.data)
        if np.any(~np.isclose(diff - diff[0], 0.)):
            raise StandardError('axis must be regularly sampled')
        if self[0] > self[-1]:
            raise StandardError('axis must be naturally ordered')

        self.axis_step = diff[0]

    def __call__(self, pos):
        """return the position in channels from an input in axis unit

        :param pos: Postion in the axis in the axis unit

        :return: Position in index
        """
        pos_index = (pos - self[0]) / float(self.axis_step)
        if pos_index < 0 or pos_index >= self.step_nb:
            warnings.warn('requested position is off axis')
        return pos_index

class Interferogram(Vector1d):
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
        Vector1d.__init__(self, interf, params=params, **kwargs)
                      
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

    def transform(self):
        """zero padded fft.
          
        :return: A Spectrum instance (or a Vector1d instance if
          interferogram is full of zeros or nans)

        .. note:: no phase correction is made here.
        """
        if self.anynan:
            logging.debug('Nan detected in interferogram')
            return Vector1d(np.zeros(
                self.step_nb, dtype=self.data.dtype) * np.nan)
        if self.allzero:
            logging.debug('interferogram is filled with zeros')
            return Vector1d(np.zeros(
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
            axis = Axis(utils.spectrum.create_cm1_axis(
                self.step_nb, self.params.step, self.params.order,
                corr=self.params.calib_coeff))

        else:
            axis_step = (self.step_nb - 1) / 2. / self.step_nb
            axis_max = (self.step_nb - 1) * axis_step
            axis = Axis(np.linspace(0, axis_max, self.step_nb))

        spec = Spectrum(interf_fft, axis, params=self.params)

        # zpd shift phase correction
        if self.has_params():
            spec.zpd_shift(self.params.zpd_index)

        # spectrum is reversed if order is even
        if self.has_params():
            if int(self.params.order)&1:
                spec.reverse()
            
        return spec

class Spectrum(Vector1d):
    """Spectrum class
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
        Vector1d.__init__(self, spectrum, params=params, **kwargs)

        self.axis = Axis(axis)

        if self.axis.step_nb != self.step_nb:
            raise ValueError('axis must have the same size as spectrum')
            
        if not np.iscomplexobj(self.data):
            raise TypeError('input spectrum is not complex')

                   
    def get_phase(self):
        """return phase"""
        nans = np.isnan(self.data)
        _data = np.copy(self.data)
        _data[nans] = 0
        _phase = np.unwrap(np.angle(_data))
        _phase[nans] = np.nan
        return _phase

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
            np.arange(self.step_nb, dtype=float) * shift * np.pi / (self.step_nb - 1))
        
    def correct_phase(self, phase):
        """Correct spectrum phase"""
        phase = Vector1d(phase).data
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
          be less precise.
        """
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
                   * (self.axis[1] - self.axis[0])  / float(quality)
                   + self.axis[0])
        f = scipy.interpolate.interp1d(zp_axis, zp_spec, bounds_error=False)
        return Spectrum(f(axis), axis, params=self.params)

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
        return self.axis(self.get_filter_bandpass_cm1()[0]), self.axis(self.get_filter_bandpass_cm1()[1])
    
    def polyfit_phase(self, return_coeffs=True, deg=1):
        """Polynomial fit of the phase
    
        :param return_coeffs: If True return (fit coefficients, error
          on coefficients) else return a Vector1d instance
          representing the fitted phase.

        :param deg: (Optional) Degree of the fitting polynomial. Must be > 0.
          (default 1).

        """
        self.assert_params()

        deg = int(deg)
        if deg < 1: raise ValueError('deg must be > 0')
    
        RANGE_BORDER_COEFF = 0.1

        cm1_min, cm1_max = self.get_filter_bandpass_cm1()
        
        cm1_border = np.abs(cm1_max - cm1_min) * RANGE_BORDER_COEFF
        cm1_min += cm1_border
        cm1_max -= cm1_border

        weights = np.ones(self.step_nb, dtype=float) * 1e-35
        weights[int(self.axis(cm1_min)):int(self.axis(cm1_max))+1] = 1.
        
        # polynomial fit
        def model(x, *p):
            return np.polynomial.polynomial.polyval(x, p)

        x= np.arange(self.step_nb)
        phase = self.get_phase()
        phase[np.isnan(phase)] = 0.
        ok_phase = phase[int(self.axis(cm1_min)):int(self.axis(cm1_max))+1]

        guesses = list()
        guess0 = np.nanmean(ok_phase)
        guess1 = np.nanmean(np.diff(ok_phase))
        guesses.append(guess0)
        guesses.append(guess1)
        
        if deg > 1:
            for i in range(deg - 1):
                guesses.append(0)
        try:
            
            pfit, pcov = scipy.optimize.curve_fit(
                model, x, phase,
                guesses,
                1./weights)
            perr = np.sqrt(np.diag(pcov))
        except Exception, e:
            logging.debug('Exception occured during phase fit: {}'.format(e))
            return None

        if return_coeffs:
            return pfit, perr
        else:
            return Vector1d(model(np.arange(self.step_nb), *pfit))
        
