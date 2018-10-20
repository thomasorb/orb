#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: photometry.py

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
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ORB.  If not, see <http://www.gnu.org/licenses/>.

import core
import utils.photometry
import utils.filters
import utils.spectrum
import utils.vector
import numpy as np
import logging

#################################################
#### CLASS Photometry ###########################
#################################################
class Photometry(object):

    STEP_NB = 2000
    transmission_terms = ['atmosphere', 'mirror', 'optics', 'filter', 'telescope']
    cameras = [1,2]
    def __init__(self, filter_name, camera_index, instrument='sitelle', airmass=1):
        
        self.tools = core.Tools(instrument=instrument)
        self.filter_name = filter_name
        if not camera_index in self.cameras:
            raise ValueError('camera_index must be in {}'.format(self.cameras))
        self.camera_index = camera_index
        self.filter_file = core.FilterFile(self.filter_name)
        self.order = self.filter_file.params.order
        self.step = self.filter_file.params.step

        self.airmass = float(airmass)

        self.corr = utils.spectrum.theta2corr(self.tools.config['OFF_AXIS_ANGLE_CENTER'])
        self.cm1_axis = core.Axis(utils.spectrum.create_cm1_axis(
            self.STEP_NB, self.step, self.order, corr=self.corr))
        self.filter_trans = self.filter_file.project(self.cm1_axis)

        self.params = {'step': self.step, 'order':self.order, 'calib_coeff':self.corr,
                       'filter_file_path':self.filter_file.basic_path}
        
    def get_transmission(self, tterm):
        if not tterm in self.transmission_terms:
            raise ValueError('tterm must be in {}'.format(self.transmission_terms))
        if tterm == 'atmosphere':
            atm = core.Cm1Vector1d(self.tools._get_atmospheric_extinction_file_path(),
                                   params=self.params).project(self.cm1_axis)
            atm.data = utils.photometry.ext2trans(atm.data, self.airmass)
            return atm
        elif tterm == 'mirror':
            return core.Cm1Vector1d(self.tools._get_mirror_transmission_file_path(),
                                   params=self.params).project(self.cm1_axis)
        elif tterm == 'optics':
            return core.Cm1Vector1d(self.tools._get_optics_file_path(self.filter_name),
                                   params=self.params).project(self.cm1_axis)
        elif tterm == 'filter':
            return self.filter_trans.copy()
        elif tterm == 'telescope':
            mtrans = self.get_transmission('mirror')
            mtrans = mtrans.math('power', 2)
            return mtrans

        else: raise NotImplementedError('{} not defined'.format(tterm))
        
    def get_unmodulated_transmission(self):
        trans = self.get_transmission('atmosphere')
        trans = trans.multiply(self.get_transmission('telescope'))
        trans = trans.multiply(self.get_transmission('optics'))
        trans = trans.multiply(self.get_transmission('filter'))
        trans = trans.multiply(self.get_qe())
        return trans

    def get_modulated_transmission(self, opd_jitter=None, wf_error=None):
        trans = self.get_unmodulated_transmission()
        trans = trans.multiply(self.get_modulation_efficiency(
            opd_jitter=opd_jitter, wf_error=wf_error))
        return trans
    
    def get_qe(self):
        return core.Cm1Vector1d(self.tools._get_quantum_efficiency_file_path(
            self.camera_index), params=self.params).project(self.cm1_axis)

    def get_ccd_gain(self):
        return self.tools.config['CAM{}_GAIN'.format(self.camera_index)]

    def get_modulation_efficiency(self, opd_jitter=None, wf_error=None):
        """Return modulation efficiency

        :param opd_jitter: OPD jitter in nm (standard deviation)

        :param wf_error: wavefront error ratio (e.g. 1/30.)
        """
        
        if opd_jitter is None: opd_jitter = self.tools.config['OPD_JITTER']
        if wf_error is None: wf_error = self.tools.config['WF_ERROR']
        rt4 = core.Cm1Vector1d(self.tools._get_4rt_file_path(),
                               params=self.params).project(self.cm1_axis)

        me_opd_jitter= utils.photometry.modulation_efficiency_opd_jitter(
            self.cm1_axis.data, opd_jitter)
        me_wf= utils.photometry.modulation_efficiency_wavefront_error(
            self.cm1_axis.data, wf_error)

        # opd jitter me is squared because tip-tilt jitter gives the same me loss
        rt4.data *= me_opd_jitter**2. * me_wf
        return rt4
    
    def modulated_flux2counts(self, flux):
        """
        convert a flux in erg/cm2/s/A to a flux in counts/s in both cameras
        
        :param flux: Flux must be in erg/cm2/s/A
        
        :return: a flux in counts/s
        """
        if isinstance(flux, core.Cm1Vector1d):
            cm1_axis = flux.axis.data
            params = flux.params
            flux = flux.data
            
        elif np.size(flux) != 1:
            raise TypeError('If flux is a vector it must be passed as a Cm1Vector1d instance')
        else:
            flux = float(flux)

            
        flux /= utils.photometry.compute_photon_energy(1e7/cm1_axis) # photons/cm2/s/A
        flux *= self.tools.config.MIR_SURFACE # photons/s/A
        flux *= self.get_modulated_transmission().project(core.Axis(cm1_axis)).data # electrons/s/A
        flux *= self.get_ccd_gain() # counts/s/A

        delta_cm1 = np.diff(cm1_axis)[0] # in cm-1, channels have the same width
        nm_bins = utils.spectrum.fwhm_cm12nm(delta_cm1, cm1_axis) # bins in nm

        flux *= nm_bins * 10. # counts/s in each channel
  
        if np.size(flux) > 1:
            return core.Cm1Vector1d(flux, axis=cm1_axis, params=params)
        else:
            return flux
        
#################################################
#### CLASS Standard #############################
#################################################
class Standard(core.Tools):
    """Manage standard files and photometrical calibration"""

    ang = None # Angstrom axis of the standard file
    flux = None # Flux of the standard in erg/cm2/s/Ang

    def __init__(self, std_name, **kwargs):
        """Initialize Standard class.

        :param std_name: Name of the standard.

        :param kwargs: Kwargs are :py:class:`~core.Tools` properties.
        """
        core.Tools.__init__(self, **kwargs)
             
        std_file_path, std_type = self._get_standard_file_path(std_name)
        if std_type == 'MASSEY' or std_type == 'MISC':
            self.ang, self.flux = self._read_massey_dat(std_file_path)
        elif std_type == 'CALSPEC':
            self.ang, self.flux = self._read_calspec_fits(std_file_path)
        elif std_type == 'OKE':
            self.ang, self.flux = self._read_oke_dat(std_file_path)
        else:
            raise StandardError(
                "Bad type of standard file. Must be 'MASSEY', 'CALSPEC', 'MISC' or 'OKE'")
       

    def _get_data_prefix(self):
        return (os.curdir + os.sep + 'STANDARD' + os.sep
                + 'STD' + '.')

    def _read_oke_dat(self, file_path):
        """Read a data file from Oke J. B., Faint spectrophotometric
        standards, AJ, (1990) and return a tuple of arrays (wavelength,
        flux).
        
        Returned wavelength axis is in A. Returned flux is converted
        in erg/cm^2/s/A.

        :param file_path: Path to the Oke dat file ('fXX.dat').
        """
        std_file = self.open_file(file_path, 'r')
        
        spec_ang = list()
        spec_flux = list()
        for line in std_file:
             line = line.split()
             spec_ang.append(line[0])
             spec_flux.append(line[1])

        spec_ang = np.array(spec_ang, dtype=float)
        spec_flux = np.array(spec_flux, dtype=float) * 1e-16

        return spec_ang, spec_flux



    def _read_massey_dat(self, file_path):
        """Read a data file from Massey et al., Spectrophotometric
        Standards (1988) and return a tuple of arrays (wavelength,
        flux).
        
        Returned wavelength axis is in A. Returned flux is converted
        in erg/cm^2/s/A.

        :param file_path: Path to the Massey dat file (generally
          'spXX.dat').
        """
        std_file = self.open_file(file_path, 'r')
        
        spec_ang = list()
        spec_mag = list()
        for line in std_file:
             line = line.split()
             spec_ang.append(line[0])
             spec_mag.append(line[1])

        spec_ang = np.array(spec_ang, dtype=float)
        spec_mag = np.array(spec_mag, dtype=float)
        
        # convert mag to flux in erg/cm^2/s/A
        spec_flux = utils.photometry.ABmag2flambda(spec_mag, spec_ang)

        return spec_ang, spec_flux

    def _read_calspec_fits(self, file_path):
        """Read a CALSPEC fits file containing a standard spectrum and
          return a tuple of arrays (wavelength, flux).

        Returned wavelength axis is in A. Returned flux is in
        erg/cm^2/s/A.
        
        :param file_path: Path to the Massey dat file (generally
          'spXX.dat').
        """
        hdu = self.read_fits(file_path, return_hdu_only=True)
        hdr = hdu[1].header
        data = hdu[1].data

        logging.info('Calspec file flux unit: %s'%hdr['TUNIT2'])
        
        # wavelength is in A
        spec_ang = np.array([data[ik][0] for ik in range(len(data))])

        # flux is in erg/cm2/s/A
        spec_flux = np.array([data[ik][1] for ik in range(len(data))])

        return spec_ang, spec_flux

    def get_spectrum(self, filter_name, n, corr=None):
        """Return part of the standard spectrum corresponding to the
        observation parameters.

        Returned spectrum is calibrated in erg/cm^2/s/A

        :param filter_name: Filter name        
        
        :param n: Number of steps
                  
        :param corr: (Optional) Correction coefficient related to the incident
          angle (default None, taken at the center of the field).
        """
        ff = core.FilterFile(filter_name)
        if corr is None: corr = utils.spectrum.theta2corr(
                self.config['OFF_AXIS_ANGLE_CENTER'])
        axis = utils.spectrum.create_cm1_axis(
            n, ff.params.step, ff.params.order, corr=corr)
        old_axis = utils.spectrum.nm2cm1(self.ang / 10.)
        
        params = {'step': ff.params.step, 'order':ff.params.order, 'calib_coeff':corr,
                  'filter_file_path':ff.basic_path}

        return core.Cm1Vector1d(utils.vector.interpolate_axis(
            self.flux, axis, 3, old_axis=old_axis), axis=axis, params=params)

    def simulate_observed_spectrum(self, filter_name, n, airmass=1, corr=None, camera_index=1,
                                   opd_jitter=None, wf_error=None):
        """Return the simulate observed spectrum in counts/s

        :param filter_name: Filter name        
        
        :param n: Number of steps
                  
        :param corr: (Optional) Correction coefficient related to the incident
          angle (default None, taken at the center of the field).

        :param opd_jitter: OPD jitter in nm (standard deviation)

        :param wf_error: wavefront error ratio (e.g. 1/30.)
        """
        spe = self.get_spectrum(filter_name, n, corr=corr)
        
        photom = Photometry(filter_name, camera_index,
                            instrument=self.params.instrument,
                            airmass=airmass)

        trans = photom.get_modulated_transmission(
            opd_jitter=opd_jitter, wf_error=wf_error)
        spe = spe.multiply(trans)
        
        spe = photom.modulated_flux2counts(spe)
        
        return spe
        
    
    def compute_optimal_texp(self, step, order, seeing, filter_name,
                             camera_number,
                             saturation=30000, airmass=1.):

        """Compute the optimal exposition time given the total flux of
        the star in ADU/s.

        :param step: Step size in nm
        
        :param order: Folding order
        
        :param seeing: Star's FWHM in arcsec
        
        :param filter_name: Name of the filter
        
        :param camera_number: Number of the camera
        
        :param saturation: (Optional) Saturation value of the detector
          (default 30000).

        :param airmass: (Optional) Airmass (default 1)    
        """
        raise NotImplementedError('must be reimplemented')
        star_flux = self.compute_star_flux_in_frame(
            step, order, filter_name,
            camera_number, airmass=airmass)
        
        dimx = self.config['CAM{}_DETECTOR_SIZE_X'.format(camera_number)]
        dimy = self.config['CAM{}_DETECTOR_SIZE_Y'.format(camera_number)]
        fov = self.config['FIELD_OF_VIEW_{}'.format(camera_number)]
        plate_scale = fov / max(dimx, dimy) * 60 # arcsec
        return utils.photometry.compute_optimal_texp(
            star_flux, seeing,
            plate_scale,
            saturation=saturation)
