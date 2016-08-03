#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: data.py

## Copyright (c) 2010-2016 Thomas Martin <thomas.martin.1@ulaval.ca>
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

import numpy as np
import scipy.special as ss
import math

#################################################
#### CLASS Data #################################
#################################################
class Data(object):
    """This class can be used as a substitute to a numpy.ndarray to
    store data.

    Data can be stored with its error and general operations can be
    used to propagate the uncertainty with no effort, e.g.:

    .. code-block:: python

      import orb.data as od
      import numpy as np

      x_dat = np.ones(5, dtype=float)
      x_err = np.random.standard_normal(5)

      a = od.array(x_dat, x_err)
      b = od.array(x_dat, x_err)
      c = a * b
      print a
      print od.mean(a)
      print od.sum(od.log10(c))
      print od.nanmean(a*b**2.)
    
    """
    ## properties
    ## data
    ## err
    ## shape
    ## real
    ## imag
    
    _dat = None
    _err = None
    _shape = None
    dtype = None

    
    def __init__(self, dat, err=None, dtype=None):
        """
        Init of Data class.
        
        :param dat: Data map.
        
        :param err: Absolute error map. Must have the same shape as
          dat.
        """
        if isinstance(dat, np.ndarray) or np.isscalar(dat):
            self._dat = np.array(dat, dtype=dtype)
            self.dtype = self._dat.dtype
        else:
            raise ValueError('Data must be a numpy array (actual type: {})'.format(type(dat)))

        self._err = self._check_data(err, err=True)

    def __getitem__(self, key):
        """Implement self[key]

        :param key: data slice.

        :return: a :py:class:`data.Data` instance.
        """
        dat_slice = self._dat[key]
        if self._err is not None: err_slice = self._err[key]
        else: err_slice = None
        return self.__class__(dat_slice, err=err_slice)
        
    def __str__(self):
        """Informal string representation, called by print"""
        return '{} ({})'.format(self.dat, self.err)

    def __mul__(self, b):
        """Implement ``self * b``.

        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        result = self.copy()
        result *= b
        return result
    
    __rmul__ = __mul__ # commutativity

    def __imul__(self, b):
        """Implement ``self *= b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
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
            if np.any(np.iscomplex(b)): self._ascomplex()    
            self._dat *= b
            if self._err is not None:
                self._err *= b
        return self

    def __div__(self, b):
        """Implement ``self / b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        result = self.copy()
        result /= b
        return result

    def __rdiv__(self, b):
        """Implement ``b / self``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        result = array(b)
        if result.err is None: result.err = np.zeros_like(result.dat)
        a = self.copy()
        result /= a
        return result

    def __idiv__(self, b):
        """Implement ``self /= b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
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
            if np.any(np.iscomplex(b)): self._ascomplex()
            self._dat /= b
            if self._err is not None:
                self._err /= b
                
        return self

    def __add__(self, b):
        """Implement ``self + b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        result = self.copy()
        result += b
        return result

    __radd__ = __add__ # commutativity

    def __iadd__(self, b):
        """Implement ``self += b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        if isinstance(b, self.__class__):
            result = self._dat + b.dat
            # error propagation
            if self._err is not None and b.err is not None:
                self._err = np.sqrt((self._err)**2.
                                    + (b.err)**2.)
            self._dat = result
        else:
            if np.any(np.iscomplex(b)): self._ascomplex()
            self._dat += np.array(b)
            
        return self

    def __sub__(self, b):
        """Implement ``self - b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        result = self.copy()
        result -= b
        return result

    def __rsub__(self, b):
        """Implement ``b - self``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        return - (self - b)

    def __isub__(self, b):
        """Implement ``self -= b``
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        if isinstance(b, self.__class__):
            result = self._dat - b.dat
            # error propagation
            if self._err is not None and b.err is not None:
                self._err = np.sqrt((self._err)**2.
                                    + (b.err)**2.)
            self._dat = result
        else:
            if np.any(np.iscomplex(b)): self._ascomplex()
            self._dat -= b
            
        return self

    def __pow__(self, b):
        """Implement ``self ** b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        a = self.copy()
        b = array(b)
        
        result = a.dat ** b.dat
        # error propagation
        if a.err is not None or b.err is not None:
            a.err = np.abs(result) * np.sqrt(
                (b.dat / a.dat * a.err)**2.
                + (np.log(a.dat) * b.err)**2.)
        
        a.dat = result
        return a

    def __neg__(self):
        """Implement ``- b``."""
        result = self.copy()
        result._dat = -result._dat
        return result

    def __pos__(self):
        """Implement ``+ b``."""
        return self.copy()

    def _ascomplex(self):
        """Change data type to complex type"""
        if not np.all(np.iscomplex(self._dat)):
            self._dat = self._dat.astype(np.complex)
            self._err = self._err.astype(np.complex)
            
    def _check_data(self, m, err=False):
        """Check passed data

        :param m: data (a Data array)

        :param err: (Optional) True if data is an uncertainty (default
          False).
        """
        if m is not None:
            if not isinstance(m, np.ndarray):
                m = np.array(m)         
            
            if (m.shape == self.shape):
                return np.array(m, dtype=self.dtype)
            elif np.size(m) == np.size(self._dat):
                m = np.broadcast_to(m, self.shape)
            else:
                raise ValueError(
                    'Bad shape. Must be {}'.format(
                        self.shape))
        else:
            if err:
                _err = np.zeros(self.shape, dtype=self.dtype)
                return _err
            else:
                raise Exception('data value cannot be None')


    def _get_dat(self):
        """Getter for dat property.

        Called by ``a = self.dat``.
        """
        return np.copy(self._dat)

    def _set_dat(self, m):
        """Setter for dat property.

        Called by ``self.dat = m``.

        :param m: data map
        """
        self._dat = self._check_data(m, err=False)

    def _get_err(self):
        """Getter for err property.

        Called by ``a = self.err``.
        """
        if self._err is not None:
            self._err = np.copy(self._err)
        else:
            self._err = np.zeros(self.shape, dtype=self.dtype)
        return np.copy(self._err)

    def _set_err(self, m):
        """Setter for err property.

        Called by ``self.err = m``.

        :param m: uncertainty map
        """
        self._err = self._check_data(m, err=True)


    def _get_shape(self):
        """Getter for shape property.

        Called by ``a = self.shape``.
        """
        return self._dat.shape

    def _get_real(self):
        """Getter for real property. Implement ndarray.real

        Called by ``a = self.real``.
        """
        if self._err is not None:
            return Data(np.copy(self._dat.real),
                        np.copy(self._err.real))
        else:
            return Data(np.copy(self._dat.real),
                        None)

    def _get_imag(self):
        """Getter for imag property. Implement ndarray.imag

        Called by ``a = self.imag``.
        """
        if self._err is not None:
            return Data(np.copy(self._dat.imag),
                        np.copy(self._err.imag))
        else:
            return Data(np.copy(self._dat.imag),
                        None)

    dat = property(_get_dat, _set_dat)
    err = property(_get_err, _set_err)
    shape = property(_get_shape)
    real = property(_get_real)
    imag = property(_get_imag)

    def copy(self):
        """Return a copy of self."""
        dat_copy = np.copy(self._dat)
        
        if self._err is not None:
            err_copy = np.copy(self._err)
        else:
            err_copy = None
            
        return self.__class__(dat_copy, err=err_copy)

    def astype(self, dtype):
        """Implement ndarray.astype

        :param dtype: numpy dtype
        """
        if self._err is not None:
            return Data(np.copy(self._dat.astype(dtype)),
                        np.copy(self._err.astype(dtype)))
        else:
            return Data(np.copy(self._dat.astype(dtype)),
                        None)


#################################################
#### Convenience functions ######################
#################################################
def array(a, e=None, dtype=None):
    """Return a :py:class:`data.Data` instance

    :param a: Estimation. Must be a :py:class:`data.Data` instance or
      must be convertible to a numpy.ndarray.

    :param e: (Optional) Uncertainty. Must be convertible to a numpy.ndarray and
      must have the same shape as a (default None).

    :param dtype: (Optional) numpy dtype (default None)
    """
    # convert a numy array of Data objects to a Data array
    if isnpdata(a):
        if e is not None: raise Exception('a seems to be already an array of Data objects')
        _a_dat = list()
        _a_err = list()
        for iobj in a:
            _a_dat.append(iobj.dat)
            _a_err.append(iobj.err)
        a = np.array(_a_dat)
        e = np.array(_a_err)
        
    if isinstance(a, Data):
        return a

    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except Exception, err:
            raise Exception('Estimation array cannot be converted to a numpy.ndarray: {}'.format(err))
    
    if e is not None:
        if not isinstance(e, np.ndarray):
            try:
                e = np.array(e)
            except Exception, err:
                raise Exception('Uncertainty array cannot be converted to a numpy.ndarray: {}'.format(err))
    
        if a.shape != e.shape:
            raise ValueError('Estimation array and Error array must have the same shape: {} != {}'.format(a.shape, e.shape))
    
    return Data(a, e, dtype=dtype)


def isdata(a):
    """Check if object is an instance of Data or a numpy array of Data
    objects. Return True in both cases.

    :param a: object
    """
    if isinstance(a, Data): return True
    elif isnpdata(a): return True
    else: return False

def isnpdata(a):
    """Return True if object is a numpy array of Data objects.

    :param a: object
    """
    if isinstance(a, np.ndarray):
        if a.dtype == object:
            if isinstance(a[0], Data):
                return True
    return False

#################################################
#### Operations on Data #########################
#################################################
# Replaces numpy operations on data with error
# propagation.


def abs(a):
    """Absolute value of a Data array"""
    if isnpdata(a): a = array(a)
        
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.abs(a.dat)
        result.err = a.err
        return result
    else:
        return np.abs(a)

def erf(a):
    """Error function.

    Uncertainty obtained via :math:`err = \sigma_x \frac{d}{dx}erf(x)`
    """
    if isnpdata(a): a = array(a)
        
    if isinstance(a, Data):
        result = a.copy()
        result.dat = ss.erf(a.dat)
        result.err = 2. * np.exp(-a.dat**2) / math.sqrt(math.pi) * a.err
        return result
    else:
        return ss.erf(a)

def dawsn(a):
    """Dawson function

    Uncertainty obtained via :math:`err = \sigma_x \frac{d}{dx}daws(x)`
    """
    
    if isnpdata(a): a = array(a)
        
    if isinstance(a, Data):
        result = a.copy()
        result.dat = ss.dawsn(a.dat)
        result.err = (1. - 2. * a.dat * ss.dawsn(a.dat)) * a.err
        return result
    else:
        return ss.dawsn(a)


def log10(a):
    """Log10 of a Data array"""
    if isnpdata(a): a = array(a)
    
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.log10(a.dat)
        result.err = (a.err / a.dat) * math.log10(math.exp(1))
        return result
    else:
        return np.log10(a)

def exp(a):
    """exponential of a Data array"""
    if isnpdata(a): a = array(a)
    
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.exp(a.dat)
        result.err = result.dat * a.err
        return result
    else:
        return np.exp(a)

def sqrt(a):
    """exponential of a Data array"""
    if isnpdata(a): a = array(a)
    
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.sqrt(a.dat)
        result.err = (result.dat * 0.5 * (a.err)) / a.dat
        return result
    else:
        return np.sqrt(a)

def sum(a):
    """Sum of a Data array"""
    if isnpdata(a): a = array(a)
    
    if isinstance(a, Data):
        return array(np.sum(a.dat), math.sqrt(np.sum(a.err**2.)))
    else:
        return np.sum(a)

def nansum(a):
    """Nansum of a Data array"""
    if isnpdata(a): a = array(a)
    
    if isinstance(a, Data):
        return array(np.nansum(a.dat), math.sqrt(np.nansum(a.err**2.)))
    else:
        return np.nansum(a)

def mean(a):
    """Mean of a Data array"""
    if isnpdata(a): a = array(a)
    
    return sum(a) / a.dat.size

def nanmean(a):
    """Nanmean of a Data array"""
    if isnpdata(a): a = array(a)
    
    return nansum(a) / np.sum(~np.isnan(a.dat))

