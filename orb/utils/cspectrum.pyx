# cython: embedsignature=True
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: cspectrum.pyx

## Copyright (c) 2010-2015 Thomas Martin <thomas.martin.1@ulaval.ca>
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


"""
cspectrum is a set of C functions coded in Cython_ to improve their speed.

.. note:: This file must be compiled before it can be used::

     cython cspectrum.pyx
     
     gcc -c -fPIC -I/usr/include/python2.7 cspectrum.c
     
     gcc -shared cspectrum.o -o cspectrum.so

.. _Cython: http://cython.org/

"""
cimport cython
import numpy as np
cimport numpy as np
import time
import scipy.ndimage.filters
import scipy.optimize
import scipy.interpolate
import orb.cutils
import orb.constants
import scipy.special
import bottleneck as bn # https://pypi.python.org/pypi/Bottleneck
from cpython cimport bool

## Import functions from math.h (faster than python math.py)
cdef extern from "math.h" nogil:
    double cos(double theta)
    double sin(double theta)
    double exp(double x)
    double sqrt(double x)
    double log(double x)
    double ceil(double x)
    double floor(double x)
    double M_PI
    double isnan(double x)
    double abs(double x)

ctypedef long double float128_t

    
class fit_lines_in_spectrum:
    
    def fast_w2pix(self, np.ndarray[np.float64_t, ndim=1] w,
                   double axis_min,
                   double axis_step):
        return np.abs(w - axis_min) / axis_step

    def fast_pix2w(self, np.ndarray[np.float64_t, ndim=1] pix,
                   double axis_min,
                   double axis_step):
        return pix * axis_step + axis_min
    
    def shift(self, np.ndarray[np.float64_t, ndim=1] lines_pos,
              np.ndarray[np.float64_t, ndim=1] vel,
              tuple p):
        
        cdef double axis_min, axis_step
        cdef np.ndarray[np.float64_t, ndim=1] lines
        cdef bool wavenumber
        cdef np.ndarray[np.float64_t, ndim=1] delta
        cdef np.ndarray[float128_t, ndim=1] lines_128
        
        axis_min, axis_step, lines, wavenumber = p
        lines_128 = lines.astype(np.float128)
        
        delta = (lines_128 * vel.astype(np.float128)
                 / <double> orb.constants.LIGHT_VEL_KMS).astype(np.float64)
        if wavenumber:
            delta = -delta
        
        return (self.fast_w2pix(lines + delta, axis_min, axis_step)
                - self.fast_w2pix(lines, axis_min, axis_step)
                + lines_pos)
        


class fit_lines_in_vector:

    def params_vect2arrays(self, np.ndarray[np.float64_t, ndim=1] free_p,
                           np.ndarray[np.float64_t, ndim=1] fixed_p,
                           np.ndarray[np.uint8_t, ndim=2] lines_p_mask,
                           np.ndarray[np.uint8_t, ndim=1] cov_p_mask,
                           np.ndarray[np.uint8_t, ndim=1] cont_p_mask):

        cdef np.ndarray[np.float64_t, ndim=1] cont_p = np.empty_like(
            cont_p_mask, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] cov_p = np.empty_like(
            cov_p_mask, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] lines_p = np.empty_like(
            lines_p_mask, dtype=np.float64)
        cdef int free_n, fixed_n, free_cont_p_index, free_cov_p_index
        cdef int fixed_cont_p_index, fixed_cov_p_index
        cdef int free_cont_p_nb, free_cov_p_nb

        free_n = free_p.shape[0]
        fixed_n = fixed_p.shape[0]
        free_cont_p_nb = np.sum(cont_p_mask)
        free_cont_p_index = free_n - free_cont_p_nb
        fixed_cont_p_index = fixed_n - (np.size(cont_p_mask) - free_cont_p_nb)
        
        if free_cont_p_nb > 0:
            cont_p[np.nonzero(cont_p_mask)] = free_p[
                free_cont_p_index:]
        if free_cont_p_nb < np.size(cont_p_mask):
            cont_p[np.nonzero(cont_p_mask==0)] = fixed_p[
                fixed_cont_p_index:]
            
        free_cov_p_nb = np.sum(cov_p_mask)
        free_cov_p_index = free_cont_p_index - free_cov_p_nb
        fixed_cov_p_index = fixed_cont_p_index - (np.size(cov_p_mask) - free_cov_p_nb)
        if free_cov_p_nb > 0:
            cov_p[np.nonzero(cov_p_mask)] = free_p[
                free_cov_p_index:free_cont_p_index]
            
        if free_cov_p_nb < np.size(cov_p_mask):
            cov_p[np.nonzero(cov_p_mask==0)] = fixed_p[
                fixed_cov_p_index:fixed_cont_p_index]
      
        lines_p[np.nonzero(lines_p_mask)] = free_p[:free_cov_p_index]
        lines_p[np.nonzero(lines_p_mask==0)] = fixed_p[:fixed_cov_p_index]
        
        return lines_p, cov_p, cont_p
    
    def model(self,
              np.ndarray[np.float64_t, ndim=1] free_p,
              np.ndarray[np.float64_t, ndim=1] fixed_p,
              np.ndarray[np.uint8_t, ndim=2] lines_p_mask,
              np.ndarray[np.uint8_t, ndim=1] cov_p_mask,
              np.ndarray[np.uint8_t, ndim=1] cont_p_mask,
              int n,
              str fmodel,
              np.ndarray[np.float64_t, ndim=1] pix_axis,
              list cov_pos_mask_list,
              fadd_shift,
              tuple fadd_shiftp):
        
        cdef np.ndarray[np.float64_t, ndim=1] cont_p = np.empty_like(
            cont_p_mask, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] cov_p = np.empty_like(
            cov_p_mask, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] lines_p = np.empty_like(
            lines_p_mask, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] mod = np.zeros(
            n, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] vel = np.zeros(
            lines_p_mask.shape[0], dtype=np.float64)
        cdef int iline

        lines_p, cov_p, cont_p = self.params_vect2arrays(
            free_p, fixed_p,
            lines_p_mask,
            cov_p_mask, cont_p_mask)

        cov_shift_index = cov_p_mask.shape[0] - np.array(cov_pos_mask_list).shape[0]        
        # + SHIFT
        if cov_p_mask[cov_shift_index]:
            for i in range(len(cov_pos_mask_list)):
                vel[np.nonzero(cov_pos_mask_list[i])] = cov_p[cov_shift_index+i]
            lines_p[:,1] = fadd_shift(lines_p[:,1], vel, fadd_shiftp)
            
        # continuum
        mod += np.polyval(cont_p, np.arange(n))

        # fwhm
        lines_p[:,2] += cov_p[0]
        for iline in range(lines_p.shape[0]):
            
            if fmodel == 'sinc':
                mod += orb.cutils.sincgauss1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2], cov_p[1])
                
            elif fmodel == 'lorentzian':
                mod += orb.cutils.lorentzian1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
                
            elif fmodel == 'sinc2':
                mod += np.sqrt(orb.cutils.sincgauss1d(
                    pix_axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2], cov_p[1])**2.)
                
            elif fmodel == 'gaussian':
                mod += orb.cutils.gaussian1d(
                    pix_axis, 0., lines_p[iline, 0],
                    lines_p[iline, 1], lines_p[iline, 2])
            else:
                raise ValueError("fmodel must be set to 'sinc', 'gaussian' or 'sinc2'")
        
        return mod
    
    def diff(self, np.ndarray[np.float64_t, ndim=1] free_p,
        np.ndarray[np.float64_t, ndim=1] fixed_p,
        np.ndarray[np.uint8_t, ndim=2] lines_p_mask,
        np.ndarray[np.uint8_t, ndim=1] cov_p_mask,
        np.ndarray[np.uint8_t, ndim=1] cont_p_mask,
        np.ndarray[np.float64_t, ndim=1] data,
        double sig,
        str fmodel,
        np.ndarray[np.float64_t, ndim=1] pix_axis,
        list cov_pos_mask_list,
        list iter_list,
        fadd_shift,
        fadd_shiftp):

        cdef np.ndarray[np.float64_t, ndim=1] data_mod = np.empty_like(
            data, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=1] res = np.empty_like(
            data, dtype=np.float64)
        cdef int i

        st = time.time() ##
        
        data_mod = self.model(free_p, fixed_p, lines_p_mask, cov_p_mask,
                              cont_p_mask, np.size(data), fmodel, pix_axis,
                              cov_pos_mask_list,
                              fadd_shift, fadd_shiftp)


     
        res = (data - data_mod) / sig
        return res



