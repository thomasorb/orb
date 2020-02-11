#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: sim.py

## Copyright (c) 2010-2018 Thomas Martin <thomas.martin.1@ulaval.ca>
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

import orb.utils.validate
import orb.utils.filters
import orb.utils.sim
import orb.utils.spectrum
import orb.core
import orb.fft
import orb.constants
import scipy.interpolate
import scipy.stats

class RawSimulator(object):

    def __init__(self, step_nb, params, instrument='sitelle'):
        """
        :param params: Can be a parameters dict or the name of a filter.
        """
        self.tools = orb.core.Tools(instrument=instrument)

        if isinstance(params, str):
            filter_name = params
            self.params = orb.core.ROParams()
            if step_nb <= 0: raise ValueError('step_nb must be > 0')
        
            self.params['step_nb'] = int(step_nb)
            self.params['filter_name'] = str(filter_name)
            self.filterfile = orb.core.FilterFile(self.params.filter_name)
            self.params['filter_file_path'] = self.filterfile.basic_path
            self.params['step'] = self.filterfile.params.step
            self.params['order'] = self.filterfile.params.order
            self.params['zpd_index'] = self.params.step_nb // 5
            self.params['calib_coeff'] = orb.utils.spectrum.theta2corr(
                self.tools.config['OFF_AXIS_ANGLE_CENTER'])
            self.params['nm_laser'] = self.tools.config['CALIB_NM_LASER']
        elif isinstance(params, dict):
            self.params = params
            if 'calib_coeff' not in self.params:
                if 'axis_corr' in self.params:
                    self.params['calib_coeff'] = self.params['axis_corr']
        else:
            raise TypeError('params must be a filter name (str) or a parameter dictionary')
            
        self.data = np.zeros(self.params.step_nb)

        cm1_axis = orb.utils.spectrum.create_cm1_axis(
            self.params.step_nb, self.params.step, self.params.order,
            corr=self.params.calib_coeff)

        self.spectrum_axis = orb.core.Axis(cm1_axis, params=self.params)
        
    def get_interferogram(self):
        return orb.fft.Interferogram(np.copy(self.data), params=self.params, exposure_time=1)

    def add_line(self, wave, vel=0, flux=1, sigma=0, jitter=0):
        """

        :param vel: Velocity in km/s

        :param jitter: Std of an OPD jitter. Must be given in nm.
        """
        if isinstance(wave, str):
            wave = orb.core.Lines().get_line_cm1(wave)

        if vel != 0:
            wave += orb.utils.spectrum.line_shift(
                vel, wave, wavenumber=True)

            
        RESOLV_COEFF = self.params.order * 10
        opd_axis = (np.arange(self.params.step_nb) * self.params.step
                    - (self.params.step * self.params.zpd_index)) * 1e-7 / self.params.calib_coeff

        ratio = self.params.step_nb / float(self.params.step_nb - self.params.zpd_index)

        if jitter == 0:
            interf = np.cos(2. * np.pi * wave * opd_axis)
        else:
            jitter_range = np.linspace(-jitter * 3, jitter * 3, RESOLV_COEFF) * 1e-7
            highres_opd_axis = np.concatenate([iopd + jitter_range for iopd in opd_axis])
            highres_interf = np.cos(2 * np.pi * wave * highres_opd_axis) 
            
            kernel = np.array(orb.utils.spectrum.gaussian1d(
                jitter_range / jitter * 1e7, 0., 1., 0,
                orb.constants.FWHM_COEFF))
            kernel /= np.sum(kernel)

            interf = np.array([np.sum(sub * kernel)
                               for sub in np.split(highres_interf, self.params.step_nb)])
            
        if sigma != 0:
            sigma_pix = orb.utils.fit.vel2sigma(sigma, wave, orb.cutils.get_cm1_axis_step(
                self.params.step_nb, self.params.step, self.params.calib_coeff))
            fwhm_pix = orb.utils.spectrum.compute_line_fwhm_pix(oversampling_ratio=ratio)
        
            window = orb.utils.fft.gaussian_window(
                orb.utils.fft.sigma2apod(sigma_pix, fwhm_pix),
                opd_axis/np.max(opd_axis))

            interf *= window

        # compute line flux for normalization
        fwhm_cm1 = orb.utils.spectrum.compute_line_fwhm(
            self.params.step_nb - self.params.zpd_index,
            self.params.step, self.params.order,
            self.params.calib_coeff, wavenumber=True)
        fwhm_nm = orb.utils.spectrum.fwhm_cm12nm(fwhm_cm1, wave) * 10.

        line_flux = orb.utils.spectrum.sinc1d_flux(
            self.params.step_nb / ratio, fwhm_nm)

        interf /= line_flux / flux
        
        self.data += interf
        
    def add_background(self):
        QUALITY_COEFF = 100
        
        a = self.filterfile.get_transmission(
            self.params.step_nb * QUALITY_COEFF,
            corr=self.params.calib_coeff)

        return self.add_spectrum(a)
    
    def add_spectrum(self, spectrum):
        """
        :param spectrum: Spectrum instance which must be defined on the filter range.
        """
        a = spectrum.project(self.spectrum_axis)
        if np.any(np.isnan(a.data)):
            raise ValueError('spectrum must be defined at least on the whole filter bandpass')
        
        a = np.concatenate((a.data, np.zeros(a.dimx))).astype(float)

        a_ifft = np.fft.ifft(a)

        a_interf = np.concatenate(
            (a_ifft[-self.params.zpd_index:], 
             a_ifft[:self.params.step_nb - self.params.zpd_index]))

        a_interf = a_interf.real.astype(float)
        self.data += a_interf


        
        
        
class SourceSpectrum(orb.core.Vector1d, orb.core.Tools):


    needed_params = ('instrument', 'filter_name', 'exposure_time', 'step_nb', 'airmass')
    
    def __init__(self, spectrum, axis, params, data_prefix="./", **kwargs):

        
        orb.core.Tools.__init__(self, instrument=params['instrument'],
                       data_prefix=data_prefix,
                       config=None)
        
        orb.core.Vector1d.__init__(self, spectrum, axis=axis, params=params, **kwargs)

        ff = orb.core.FilterFile(self.params['filter_name'])
        self.params.update(ff.params)
        self.params['zpd_index'] = int(0.25 * self.params['step_nb'])
        self.params['nm_laser'] = self.config.CALIB_NM_LASER
        self.params['apodization'] = 1.
        self.params['wavenumber'] = True

    def get_interferogram(self, camera=0, theta=None, binning=1, me_factor=1.):

        if theta is None:
            theta = self.config.OFF_AXIS_ANGLE_CENTER

        corr = orb.utils.spectrum.theta2corr(theta)
        self.params['calib_coeff_orig'] = corr
        
        
        regular_cm1_axis = np.linspace(1e7/self.axis.data[-1], 1e7/self.axis.data[0], self.dimx)
        
        spectrum = scipy.interpolate.interp1d(self.axis.data,
                                              self.data.astype(np.float128),
                                              bounds_error=False)(1e7/regular_cm1_axis)
        
        
        spectrum = orb.fft.Spectrum(spectrum, axis=regular_cm1_axis, params=self.params)
        
        
        photom = orb.photometry.Photometry(self.params.filter_name,
                                           camera, instrument=self.params.instrument,
                                           airmass=self.params.airmass)
        
        
        
        cm1_axis = orb.core.Axis(orb.utils.spectrum.create_cm1_axis(
            self.params.step_nb, self.params.step, self.params.order, corr=corr))

        #spectrum = spectrum.project(cm1_axis)
        # flux conservative projection
        bins = np.array(list(cm1_axis.data[:-1] - np.diff(cm1_axis.data) / 2.) + list((cm1_axis.data[-2:] + np.diff(cm1_axis.data[-3:-1]) / 2.)))

        spectrum.data = scipy.stats.binned_statistic(
            spectrum.axis.data.astype(float),
            spectrum.data.astype(float),
            bins=bins, statistic='mean')[0].astype(complex)
        spectrum.axis = cm1_axis

        spectrum = photom.flux2counts(spectrum, modulated=True)

        spectrum.data *= self.params.step_nb * self.params.exposure_time * binning**2.
        
        spectrum = orb.fft.Spectrum(spectrum)
        
        spectrum.data = spectrum.data.real
        spectrum.params['calib_coeff'] = corr
        
        interf = spectrum.inverse_transform()

        interf.data = interf.data.real

        # modulation efficiency loss with OPD
        mean_ = np.mean(interf.data)
        interf.data -= mean_
        interf.data *= orb.utils.fft.gaussian_window(
            me_factor, interf.axis.data/np.max(interf.axis.data))
        interf.data += mean_

        # poisson noise
        interf.data = np.random.poisson(interf.data.astype(int)).astype(float)
        return interf
