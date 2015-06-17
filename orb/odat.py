#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: odat.py

## Copyright (c) 2010-2014 Thomas Martin <thomas.martin.1@ulaval.ca>
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
With thoses classes uncertainty are ensured to be propagated all along
operations with 1d and 2d data.
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'


from orb.core import Tools
import numpy as np
import math



#################################################
#### CLASS Data1D ###############################
#################################################
class Data1D(Tools):

    ## properties
    # data
    # err
    # snr

    _dat = None
    _err = None
    _snr = None
    _dimx = None
    _snr_NA = False
    
    def __init__(self, dat, err=None, snr=None):
        dat = np.array(dat).astype(float)
        if len(dat.shape) > 1: self._print_error('Data must be a 1D vector')
        self._dat = dat
        self._dimx = np.size(dat)
        
        if err is not None:
            self._err = self._check_data(err)
        else:
            self._err = None

        if snr is not None:
            self._snr = self._check_data(snr)
            self._snr_NA = True
        else:
            self._snr = None
            self._snr_NA = False

    def __str__(self):
        return '{} ({})'.format(str(self.dat), str(self.err))

    def __mul__(self, b):
        result = self.copy()
        result *= b
        return result
    
    __rmul__ = __mul__ # commutativity

    def __imul__(self, b):
        if isinstance(b, self.__class__):
            result = self._dat * b.dat
            # error propagation
            if self._err is not None:
                if b.err is not None:
                    self._err = result * np.sqrt((self._err/self._dat)**2.
                                                 + (b.err/b.dat)**2.)
                else:
                    self._err *= b
            self._dat = result
            
        else:
            self._dat *= b
            if self._err is not None:
                self._err *= b
                
        return self

    def __div__(self, b):
        result = self.copy()
        result /= b
        return result

    def __rdiv__(self, b):
        result = self.__class__(b)
        if result.err is None: result.err = np.zeros_like(result.dat)
        a = self.copy()
        result /= a
        return result

    def __idiv__(self, b):
        if isinstance(b, self.__class__):
            result = self._dat / b.dat
            # error propagation
            if self._err is not None:
                if b.err is not None:
                    self._err = result * np.sqrt((self._err/self._dat)**2.
                                                 + (b.err/b.dat)**2.)
                else:
                    self._err /= b
            self._dat = result
            
        else:
            self._dat /= b
            if self._err is not None:
                self._err /= b
                
        return self

    def __add__(self, b):
        result = self.copy()
        result += b
        return result

    __radd__ = __add__ # commutativity

    def __iadd__(self, b):
        if isinstance(b, self.__class__):
            result = self._dat + b.dat
            # error propagation
            if self._err is not None and b.err is not None:
                self._err = np.sqrt((self._err)**2.
                                    + (b.err)**2.)
            self._dat = result
        else:
            self._dat += b
            
        return self

    def __sub__(self, b):
        result = self.copy()
        result -= b
        return result

    def __rsub__(self, b):
        return - (self - b)

    def __isub__(self, b):
        if isinstance(b, self.__class__):
            result = self._dat - b.dat
            # error propagation
            if self._err is not None and b.err is not None:
                self._err = np.sqrt((self._err)**2.
                                    + (b.err)**2.)
            self._dat = result
        else:
            self._dat -= b

        return self

    def __pow__(self,b):
        a = self.copy()
        if isinstance(b, self.__class__):
            result = a.dat ** b.dat
            # error propagation
            if a.err is not None and b.err is not None:
                a.err = result * np.sqrt(
                    (b.dat / a.dat * a.err)**2.
                    + (np.log(a.dat) * b.err)**2.)
        else:
            result = a.dat ** b
            
        a.dat = result
        return a

    def __neg__(self):
        result = self.copy()
        result.dat = - result.dat
        return result

    def __pos__(self):
        return self.copy()
        
    def _check_data(self, a):
        a = np.array(a).astype(float)
        if a.shape != self._dat.shape: self._print_error('Vector must have the same size as Data')
        return a
    

    def _get_snr(self):
        if self._snr_NA:
            return self._snr
        elif self._err is not None:
            return self._dat / self._err
        else:
            return None

    def _set_snr(self, a):
        self._snr = self._check_data(a)
        self._snr_NA = True

    def _get_dat(self):
        if np.size(self._dat) > 1:
            return np.squeeze(self._dat)
        elif isinstance(self._dat, np.ndarray):
            if self._dat.ndim > 0:
                return float(self._dat[0])
            else:
                return float(self._dat)
        else:
            return float(self._dat)

    def _set_dat(self, a):
        self._dat = self._check_data(a)

    def _get_err(self):
        if np.size(self._err) > 1:
            return np.abs(np.squeeze(self._err))
        elif isinstance(self._err, np.ndarray):
            if self._err.ndim > 0:
                return abs(float(self._err[0]))
            else:
                return abs(float(self._err))
        else:
            return abs(float(self._err))
        

    def _set_err(self, a):
        self._err = self._check_data(a)
        
    snr = property(_get_snr, _set_snr)
    dat = property(_get_dat, _set_dat)
    err = property(_get_err, _set_err)


    def copy(self):
        dat_copy = np.copy(self._dat)
        
        if self._err is not None:
            err_copy = np.copy(self._err)
        else:
            err_copy = None

        if self._snr is not None and self._snr_NA:
            snr_copy = np.copy(self._snr)
        else:
            snr_copy = None
            
        return self.__class__(dat_copy, err=err_copy,
                              snr=snr_copy)


#################################################
#### CLASS Data2D ###############################
#################################################
class Data2D(Data1D):
    """Manage data map, associated error map.

    It is also possible to manage beam map and surface map created
    after snr smoothing.
    
    Compute error propagation on basic data operations (addition,
    substraction, multiplication, division, power)
    """
    
    ## properties:
    ## snr
    ## dat
    ## err
    ## beam
    ## surf
    ## shape
    
    _dat = None
    _err = None
    _snr = None
    _beam = None
    _surf = None
    _snr_NA = False
    
    dimx = None
    dimy = None
    
    def __init__(self, dat, err=None, beam=None, surf=None, snr=None):
        """
        Init of Data2D class.
        
        :param dat: Data map.
        
        :param err: Absolute error map. Must have the same shape as dat.

        :param beam: Beam map. Size of the beam (FWHM of the Gaussian
          smoothing applied) at each pixel. If None given, the beam
          size is considered to be of 1 pixel everywhere.

        :param surf: Number of pixels used to get the SNR.
        
        :param snr: If not None the SNR map will not be computed from
          dat/err. Useful when dat/err has no meaning (e.g. velocity
          map)
        """
        if isinstance(dat, np.ndarray):
            if len(dat.shape) == 2:
                self._dat = dat
                self.dimx = self._dat.shape[0]
                self.dimy = self._dat.shape[1]
            else:
                self._print_error('Data must be a 2d array')
        else:
            self._print_error('Data must be a numpy array')

        self._err = self._check_data(err)
        self._beam = self._check_data(beam)
        self._surf = self._check_data(surf)
        self._snr = self._check_data(snr)
        if self._snr is not None:
            self._snr_NA = True
        else:
            self._snr_NA = False
            
        
    def __getitem__(self, key):
        dat_slice = self._dat[key]
        if self._err is not None: err_slice = self._err[key]
        else: err_slice = None
        if self._beam is not None: beam_slice = self._beam[key]
        else: beam_slice = None
        if self._surf is not None: surf_slice = self._surf[key]
        else: surf_slice = None
        
        return self.__class__(dat_slice, err=err_slice,
                              beam=beam_slice, surf=surf_slice)

    def _check_data(self, m):
        if m is not None:
            if isinstance(m, np.ndarray):
                if len(m.shape) == 2:
                    if (m.shape[0] == self.dimx
                        and m.shape[1] == self.dimy):
                        return m
                    else:
                        self._print_error(
                            'Bad map shape. Must be (%d, %d)'%(
                                self.dimx, self.dimy))
                else:
                    self._print_error('Map must be a 2d array')
            else:
                self._print_error('Map must be a numpy array')
        else:
            return None

    def _get_snr(self):
        if self._snr_NA:
            return np.copy(self._snr)
        else:
            return self._dat / self._err

    def _set_snr(self, m):
        self._snr = self._check_data(m)
        self._snr_NA = True

    def _get_dat(self):
        return np.copy(self._dat)

    def _set_dat(self, m):
        self._dat = self._check_data(m)

    def _get_err(self):
        if self._err != None:
            self._err = np.copy(np.abs(self._err))
        return self._err

    def _set_err(self, m):
        self._err = self._check_data(m)

    def _get_beam(self):
        if self._beam is not None:
            return np.copy(self._beam)
        else:
            return np.ones(self.shape, dtype=float)

    def _set_beam(self, m):
        self._beam = self._check_data(m)

    def _get_surf(self):
        if self._surf is not None:
            return np.copy(self._surf)
        else:
            return np.ones(self.shape, dtype=float)

    def _set_surf(self, m):
        self._surf = self._check_data(m)

    def _get_shape(self):
        return (self.dimx, self.dimy)

    snr = property(_get_snr, _set_snr)
    dat = property(_get_dat, _set_dat)
    err = property(_get_err, _set_err)
    beam = property(_get_beam, _set_beam)
    surf = property(_get_surf, _set_surf)
    shape = property(_get_shape)
    
    def copy(self):
        dat_copy = np.copy(self._dat)
        
        if self._err is not None:
            err_copy = np.copy(self._err)
        else:
            err_copy = None
            
        if self._beam is not None:
            beam_copy = np.copy(self._beam)
        else:
            beam_copy = None

        if self._surf is not None:
            surf_copy = np.copy(self._surf)
        else:
            surf_copy = None

        if self._snr is not None and self._snr_NA:
            snr_copy = np.copy(self._snr)
        else:
            snr_copy = None
            
        return self.__class__(dat_copy, err=err_copy,
                              beam=beam_copy, surf=surf_copy,
                              snr=snr_copy)

    
#################################################
#### Operations on Data #########################
#################################################
# Replaces numpy operations on data with error
# propagation.

def log10(a):
    if isinstance(a, Data1D) or isinstance(a, Data2D):
        result = a.copy()
        result.dat = np.log10(a.dat)
        result.err = (a.err / a.dat) * math.log10(math.exp(1))
        return result
    else:
        return np.log10(a)

def sum(a):
    if isinstance(a, Data1D) or isinstance(a, Data2D):
        return Data1D(np.sum(a.dat), math.sqrt(np.sum(a.err**2.)))
    else:
        return np.sum(a)

def nansum(a):
    if isinstance(a, Data1D) or isinstance(a, Data2D):
        return Data1D(np.nansum(a.dat), math.sqrt(np.nansum(a.err**2.)))
    else:
        return np.nansum(a)

def mean(a):
    return sum(a) / a.dat.size

def nanmean(a):
    return nansum(a) / np.sum(~np.isnan(a.dat))
