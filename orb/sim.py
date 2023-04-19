#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: sim.py

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

import orb.utils.validate
import orb.utils.filters
import orb.utils.sim
import orb.utils.photometry
import orb.utils.spectrum
import orb.core
import orb.fft
import orb.constants
import scipy.interpolate
import scipy.stats

import pandas as pd




class SkyModel(object):
    """Very basic sky model which generate a spectrum of the sky.
    
    It includes continuum brightness, sky lines and moon brightness.

    most data comes from https://www.gemini.edu/observing/telescopes-and-sites/sites
    """
    
    def __init__(self, airmass=1, instrument='sitelle'):

        self.airmass = float(airmass)

        # www.cfht.hawaii.edu/Instruments/ObservatoryManual/CFHT_ObservatoryManual_(Sec_2).html
        # self.sky_brightness = pd.DataFrame(
        #     {'band': ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K'],
        #      'nm': [365, 445, 551, 658, 806, 1220, 1630, 2190], # nm
        #      'mag': [21.6, 22.3, 21.1, 20.3, 19.2, 14.8, 13.4, 12.6]}) # mag/arsec^2
        
        # self.sky_brightness['flam'] = orb.utils.photometry.ABmag2flambda(
        #     self.sky_brightness.mag, self.sky_brightness.nm*10.) # erg/cm2/s/A/arcsec^2

        self.skybg = dict()
        self.tools = orb.core.Tools(instrument=instrument)
        filepath = self.tools._get_orb_data_file_path('skybg_50_10.dat')
        with open(filepath) as f:
            for line in f:
                if '#' in line: continue
                if len(self.skybg) == 0:
                    keys = line.strip().split()
                    [self.skybg.__setitem__(ikey, list()) for ikey in keys]
                    continue
                vals = line.strip().split()
                if float(vals[1]) == 0: continue
                for ikey, ival in zip(keys, vals):
                    self.skybg[ikey].append(float(ival))

        self.skybg = pd.DataFrame(self.skybg)
        self.skybg['flam'] = self.skybg['phot/s/nm/arcsec^2/m^2'].values
        self.skybg['flam'] *= orb.utils.photometry.compute_photon_energy(self.skybg.nm.values) # erg/s/nm/arcsec2/m2
        self.skybg['flam'] /= 10 # erg/s/m2/A/arcsec2
        self.skybg['flam'] /= 1e4 # erg/s/cm2/A/arcsec2

        y = self.skybg['flam'].values
        x = self.skybg.nm.values
        BIN = 31
        diffuse = np.array([((x[min(i+BIN, x.size-1)]+x[i])/2, np.min(y[i:i+BIN]))
                            for i in range(0, x.size, BIN)]).T
        self.diffusef = scipy.interpolate.UnivariateSpline(diffuse[0], diffuse[1], k=2, s=np.mean(diffuse[1])**2 * 0.05)
        self.skybg['diffuse'] = self.diffusef(x)

        BIN = 200
        #lines = np.array([((x[min(i+BIN, x.size-1)]+x[i])/2, np.mean(y[i:i+BIN]) - np.min(y[i:i+BIN]))
        #                       for i in range(0, x.size, BIN)]).T
        #self.linesf = scipy.interpolate.UnivariateSpline(lines[0], lines[1], k=2, s=0)
        self.linesf = scipy.interpolate.UnivariateSpline(x, y - self.diffusef(x), k=2, s=0)
        
        atm_ext_cm1 = orb.core.Vector1d(
            self.tools._get_atmospheric_extinction_file_path(), instrument=instrument)
        
        atm_ext_cm1.data = orb.utils.photometry.ext2trans(
            atm_ext_cm1.data, self.airmass)
        
        self.atm_transf = scipy.interpolate.UnivariateSpline(
            1e7/atm_ext_cm1.axis.data[::-1], atm_ext_cm1.data[::-1], k=3, s=0, ext=3)

    def get_spectrum(self, cm1_axis):

        assert np.all(np.isclose(np.diff(cm1_axis), np.diff(cm1_axis)[0]), 0), 'cm1_axis must be evenly spaced'
        sky_spectrum = np.zeros_like(cm1_axis)
        sky_lines = orb.core.Lines().air_sky_lines_nm
        axis_step = cm1_axis[1] - cm1_axis[0]
        fwhm = axis_step*4
        
        for iline in sky_lines:
    
            iline_cm1 = 1e7/sky_lines[iline][0]
            iline_amp = sky_lines[iline][1]
            if ((iline_cm1 > cm1_axis[0])
                and(iline_cm1 < cm1_axis[-1])):
                iiline = orb.utils.spectrum.gaussian1d(
                    cm1_axis, 0, iline_amp, iline_cm1, fwhm)
                sky_spectrum += iiline
            
        # lines are scaled to the brightness of the reference lines        
        SMOOTH = 2
        sky_spectrum /= orb.utils.vector.smooth(sky_spectrum, deg=sky_spectrum.size/SMOOTH)
        sky_spectrum *= orb.utils.vector.smooth(self.linesf(1e7/cm1_axis), deg=sky_spectrum.size/SMOOTH)
                
        # adding diffuse brightness
        sky_spectrum += self.diffusef(1e7/cm1_axis)

        # convert to erg/cm2/s/A
        pixel_surf = (self.tools.config.FIELD_OF_VIEW_1 * 60
                      / self.tools.config.CAM1_DETECTOR_SIZE_X)**2
        sky_spectrum *= pixel_surf

        
        return orb.core.Vector1d(sky_spectrum, axis=cm1_axis)
    
        
class Base(object):

    def __init__(self, step_nb, params, instrument='sitelle', **kwargs):
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

        self.params.update(kwargs)

        self.params['calib_coeff_orig'] = self.params['calib_coeff']
        self.params['apodization'] = 1
        self.params['wavenumber'] = True
        
        self.data = None
        self.axis = None

class Spectrum(Base):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def add_component(self, lines, amp, noise_std=0, vel=0, sigma=0):
    
        lines = orb.core.Lines().get_line_cm1(lines)
        amp = np.array(amp).reshape((np.size(amp)))
        ax, sp = orb.fit.create_cm1_lines_model_raw(
            np.array(lines).reshape((np.size(lines))), amp, self.params.step, self.params.order, self.params.step_nb, 
            self.params.calib_coeff, self.params.zpd_index,
            vel=vel, sigma=sigma, fmodel='sincgauss')
        if self.data is None:
            self.data = np.copy(sp)
        else:
            self.data += sp
        if self.axis is not None:
            assert np.all(self.axis == ax)
        else:
            self.axis = np.copy(ax)
    
    def get_spectrum(self):
        if self.data is None:
            raise Exception('add at least one component with add_component()')
        return orb.fft.Spectrum(self.data, axis=self.axis, params=self.params)  

        
class Interferogram(Base):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        cm1_axis = orb.utils.spectrum.create_cm1_axis(
            self.params.step_nb, self.params.step, self.params.order,
            corr=self.params.calib_coeff)

        self.spectrum_axis = orb.core.Axis(cm1_axis, params=self.params)
        self.data = np.zeros_like(cm1_axis)
        
    def get_interferogram(self):
        return orb.fft.Interferogram(np.copy(self.data), params=self.params, exposure_time=1)

    def add_line(self, wave, vel=0, flux=1, sigma=0, jitter=0):
        """
        :param wave: The name of the line or the line wavenumber in cm-1
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

        # compensate for energy lost in the imaginary part !
        a_interf = a_interf.real.astype(float) * 2. 
        self.data += a_interf


        
        
        
class SourceSpectrum(orb.core.Vector1d, orb.core.Tools):


    
    def __init__(self, spectrum, axis, instrument='sitelle', data_prefix="./", **kwargs):
        """Init

        :param spectrum: Spectrum of the source, may be vector of
          zeros.  Must have the same size as axis. Must be calibrated
          in erg/cm2/s/A.

        :param axis: Axis in cm-1. Resolution must be much higher than
          the simulated spectrum (e.g. np.linspace(10000, 25000,
          30000))

        """
        assert np.size(axis) > 0, 'axis size must be > 0'
        assert np.size(spectrum) == np.size(axis), 'axis must have same size as spectrum'
        orb.core.Tools.__init__(
            self, instrument=instrument,
            data_prefix=data_prefix,
            config=None)
        
        orb.core.Vector1d.__init__(
            self, spectrum, axis=axis,
            params={'instrument':instrument}, **kwargs)

        
    def add_line(self, wave, flux, vel=None, sigma=None):
        """add a line to the spectrum

        :param wave: Name of the line or its wavenumber in cm-1 (can be a tuple)

        :param vel: Velocity in km/s

        :param flux: Flux in erg/cm2/s

        :param sigma: Broadening in km/s
        """
        if np.size(wave) == 1:
            wave = [wave,]
        wave = np.array(wave)
        if not np.issubdtype(wave.dtype, np.number):
            wave = orb.core.Lines().get_line_cm1(wave)
        
        if np.size(flux) == 1:
            flux = [flux,]
        flux = np.array(flux)

        assert flux.size == wave.size, 'flux must have same size as wave'

        
        if vel is not None:
            vel = np.array(vel)
            assert vel.size == wave.size, 'vel must have same size as wave'
            wave += orb.utils.spectrum.line_shift(
                vel, wave, wavenumber=True)
            
        if sigma is not None:
            sigma = np.array(sigma)
            assert sigma.size == wave.size, 'sigma must have same size as wave'
        else:
            sigma = np.zeros_like(wave)

        axis_step = self.axis.data[1] - self.axis.data[0]
        
        for iwave, isigma, iflux in zip(wave, sigma, flux):
            isigma = orb.utils.fit.vel2sigma(isigma, iwave, axis_step) * axis_step
            isigma = max(4 * axis_step, isigma)
            iline = orb.utils.spectrum.gaussian1d(
                self.axis.data, 0, 1, iwave, isigma * orb.constants.FWHM_COEFF)
            iline /= orb.utils.spectrum.gaussian1d_flux(1, isigma) # flux is normalized to 1 
            iline *= iflux
            self.data += iline
            

    def add_spectrum(self, spectrum):
        """Spectrum

        :param spectrum: spectrum vector, must be in erg/cm2/s/A
        """
        spectrum = np.array(spectrum)
        assert spectrum.size == self.axis.data.size, 'spectrum must have same size has the axis provided at init'
        self.data += spectrum

    def get_interferogram(self, params, camera=0, theta=None, binning=1, me_factor=1., x=None,
                          bypass_flux2counts=False, noiseless=True):

        """:param x: scanning vector (to simulate a non-uniform
           scan). Must be given in step fraction. e.g. x =
           (np.arange(params['step_nb']) - params['zpd_index']) will
           simulate the default scanning sequence.
        """
        needed_params = ('instrument', 'filter_name', 'exposure_time', 'step_nb', 'airmass')
        for ipar in needed_params:
            if ipar not in params:
                raise Exception('parameter {} needed'.format(ipar))


        ff = orb.core.FilterFile(params['filter_name'])
        params.update(ff.params)
        params['zpd_index'] = int(0.25 * params['step_nb'])
        params['nm_laser'] = self.config.CALIB_NM_LASER
        params['apodization'] = 1.
        params['wavenumber'] = True
        if theta is None:
            theta = self.config.OFF_AXIS_ANGLE_CENTER

        corr = orb.utils.spectrum.theta2corr(theta)
        params['calib_coeff_orig'] = corr
        params['calib_coeff'] = corr
        params = orb.core.Params(params)

        spectrum = orb.fft.Spectrum(np.copy(self.data), axis=np.copy(self.axis.data), params=params)
        
        photom = orb.photometry.Photometry(params.filter_name,
                                           camera, instrument=params.instrument,
                                           airmass=params.airmass)

        # spectrum.data is in erg/cm2/s/A
        # it should be transformed to counts
        if not bypass_flux2counts:
            spectrum = orb.fft.Spectrum(photom.flux2counts(spectrum, modulated=True))
            spectrum.data *= params.step_nb * params.exposure_time * binning**2.
            spectrum.data[np.nonzero(np.isnan(spectrum.data))] = 0.
            

        # compute total flux of input spectrum
        # axis_steps = (1e7/spectrum.axis.data[:-1] - 1e7/spectrum.axis.data[1:]) * 10
        # axis_steps = np.concatenate((axis_steps, [axis_steps[-1],]))
        # print(np.sum(spectrum.data * axis_steps) )

        # decalibrate spectrum
        decal_cm1_axis = orb.core.Axis(orb.utils.spectrum.create_cm1_axis(
            self.dimx, params.step, params.order, corr=corr))
        spectrum = spectrum.project(decal_cm1_axis)
        spectrum.data[np.nonzero(np.isnan(spectrum.data))] = 0.
        
        # spectrum is flipped if order is even
        if int(params.order)&1:
            spectrum.reverse()

        a = np.concatenate((spectrum.data, np.zeros(spectrum.dimx)))
        if x is None:
            a_ifft = scipy.fft.ifft(a)
            a_interf = np.concatenate(
                (a_ifft[-params.zpd_index:], 
                 a_ifft[:params.step_nb - params.zpd_index]))
        else:
            a_interf = orb.utils.fft.indft(spectrum.data, x/2)/2
              
        # compensate energy lost in the imaginary part !
        a_interf = a_interf.real.astype(float) * 2.            
        interf = orb.fft.Interferogram(a_interf, params=params)
                
        unmod_spectrum = spectrum.math('divide', photom.get_modulation_efficiency())
        unmod_spectrum.data[np.nonzero(np.isnan(unmod_spectrum.data))] = 0.
        
        interf.data += np.mean(unmod_spectrum.data).astype(np.longdouble)
        interf.data = interf.data.real
        
        # modulation efficiency loss with OPD
        if me_factor > 1:
            mean_ = np.mean(interf.data)
            interf.data -= mean_
            interf.data *= orb.utils.fft.gaussian_window(
                me_factor, interf.axis.data/np.max(interf.axis.data))
            interf.data += mean_

        # poisson noise
        if not noiseless:
            interf.data = np.random.poisson(interf.data).astype(int).astype(float)
        return interf
