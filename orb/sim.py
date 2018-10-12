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

class Simulator(object):

    def __init__(self, step_nb, filter_name, instrument='sitelle'):

        self.tools = orb.core.Tools(instrument=instrument, silent=True)
        
        self.params = orb.core.ROParams()
        if step_nb <= 0: raise ValueError('step_nb must be > 0')
        
        self.params['step_nb'] = int(step_nb)
        self.params['filter_name'] = str(filter_name)
        self.params['filter_file_path'] = self.tools._get_filter_file_path(self.params.filter_name)
        self.params['step'] = orb.utils.filters.get_key(self.params.filter_file_path, 'STEP', float)
        self.params['order'] = orb.utils.filters.get_key(self.params.filter_file_path, 'ORDER', int)
        self.params['zpd_index'] = self.params.step_nb // 5
        self.params['calib_coeff'] = orb.utils.spectrum.theta2corr(
            self.tools.config['OFF_AXIS_ANGLE_CENTER'])
        self.params['nm_laser'] = self.tools.config['CALIB_NM_LASER']
        
        self.data = np.zeros(self.params.step_nb)

        cm1_axis = orb.utils.spectrum.create_cm1_axis(
            self.params.step_nb, self.params.step, self.params.order,
            corr=self.params.calib_coeff)

        self.spectrum_axis = orb.core.Axis(cm1_axis, params=self.params)
        


    def get_interferogram(self):
        return orb.fft.Interferogram(np.copy(self.data), params=self.params)

    def add_line(self, sigma, jitter=0):
        """
        :param jitter: Std of an OPD jitter. Must be given in nm.
        """
        if isinstance(sigma, str):
            sigma = orb.core.Lines().get_line_cm1(sigma)

        line_interf = orb.utils.sim.line_interf(
            self.spectrum_axis(sigma) / 2.,
            self.params.step_nb, symm=True,
            jitter = jitter / self.params.step)
        
        line_interf = line_interf[-self.params.step_nb-self.params.zpd_index:
                                  -self.params.zpd_index]

        self.data += line_interf
        
    def add_background(self):
        QUALITY_COEFF = 100
        
        a, _, _ = orb.utils.filters.get_filter_function(
            self.params.filter_file_path, self.params.step,
            self.params.order, self.params.step_nb * QUALITY_COEFF,
            wavenumber=True, silent=True, corr=self.params.calib_coeff)

        a *= self.params.step_nb
        
        a = np.concatenate((a, np.zeros(a.size)))
        
        a_ifft = np.fft.ifft(a)

        a_interf = np.concatenate(
            (a_ifft[-self.params.zpd_index:], 
             a_ifft[:self.params.step_nb - self.params.zpd_index]))
        
        print np.argmax(a_interf), self.params.zpd_index
        self.data += a_interf.real.astype(float)
