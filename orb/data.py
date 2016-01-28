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
import math

#################################################
#### CLASS Data #################################
#################################################
class Data(object):
    """Empty class that must be implemented by a particular Data class
    depending on the data dimensions (e.g. :py:class:`data.Data1D`)"""
    pass
    

#################################################
#### CLASS Data1D ###############################
#################################################
class Data1D(Data):
    """This class can be used as a substitute to a 1D numpy.ndarray to
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
    # data
    # err

    _dat = None
    _err = None
    _dimx = None
    
    def __init__(self, dat, err=None):
        """Init class
        :param dat: Estimation array
        :param err: (Optional) Uncertainty array (default None)
        """
        dat = np.array(dat).astype(float)
        if len(dat.shape) > 1: raise ValueError('Data must be a 1D vector')
        self._dat = dat
        self._dimx = np.size(dat)

        if err is not None:
            self._err = self._check_data(err)
        else:
            self._err = np.zeros_like(dat, dtype=float)

    def __str__(self):
        """Informal string representation, called by print"""
        return '{} ({})'.format(str(self.dat), str(self.err))

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
        result = self.__class__(b)
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
            self._dat += b
            
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
            self._dat -= b

        return self

    def __pow__(self, b):
        """Implement ``self ** b``.
        
        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
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
        """Implement ``- b``."""
        result = self.copy()
        result.dat = - result.dat
        return result

    def __pos__(self):
        """Implement ``+ b``."""
        return self.copy()
        
    def _check_data(self, a):
        """Check passed data

        :param a: data
        """
        a = np.array(a).astype(float)
        if a.shape != self._dat.shape: raise ValueError('Vector must have the same size as Data')
        return a

    def _get_dat(self):
        """Getter for dat property.

        Called by ``a = self.dat``.
        """
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
        """Setter for dat property.

        Called by ``self.dat = a``.

        :param a: data
        """
        self._dat = self._check_data(a)

    def _get_err(self):
        """Getter for err property.

        Called by ``a = self.err``.
        """
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
        """Setter for err property.

        Called by ``self.err = a``.

        :param a: data
        """
        self._err = self._check_data(a)

    def _get_shape(self):
        """Getter for shape property.

        Called by ``a = self.shape``.
        """
        return self._dat.shape
    
    dat = property(_get_dat, _set_dat)
    err = property(_get_err, _set_err)
    shape = property(_get_shape)

    def copy(self):
        """Return a copy of self."""
        dat_copy = np.copy(self._dat)
        
        if self._err is not None:
            err_copy = np.copy(self._err)
        else:
            err_copy = None
            
        return self.__class__(dat_copy, err=err_copy)


#################################################
#### CLASS Data2D ###############################
#################################################
class Data2D(Data1D):
    """This class can be used as a substitute to a 2D numpy.ndarray to
    store data.

    Data can be stored with its error and general operations can be
    used to propagate the uncertainty with no effort, e.g.:


    .. code-block:: python

      import orb.data as od
      import numpy as np

      x_dat = np.random.standard_normal((5, 5))
      x_err = np.ones((5, 5), dtype=float)

      a = od.array(x_dat, x_err)
      b = od.array(x_dat, x_err)
      c = a * b
      print a
      print od.mean(a)
      print od.sum(od.log10(c))
      print od.nanmean(a*b**2.)
      
    """
    
    ## properties:
    ## dat
    ## err
    ## shape
    
    dimy = None
    
    def __init__(self, dat, err=None):
        """
        Init of Data2D class.
        
        :param dat: Data map.
        
        :param err: Absolute error map. Must have the same shape as dat.    
        """
        if isinstance(dat, np.ndarray):
            if len(dat.shape) == 2:
                self._dat = dat
                self.dimx = self._dat.shape[0]
                self.dimy = self._dat.shape[1]
            else:
                raise ValueError('Data must be a 2D array')
        else:
            raise ValueError('Data must be a numpy array')

        self._err = self._check_data(err)
     
            
        
    def __getitem__(self, key):
        """Implement self[key]

        :param key: data slice.

        :return: a :py:class:`data.Data2D` instance.
        """
        dat_slice = self._dat[key]
        if self._err is not None: err_slice = self._err[key]
        else: err_slice = None
      
        return self.__class__(dat_slice, err=err_slice)


    def _check_data(self, m):
        """Check passed data map

        :param m: data map (a 2D array)
        """
        if m is not None:
            if isinstance(m, np.ndarray):
                if len(m.shape) == 2:
                    if (m.shape[0] == self.dimx
                        and m.shape[1] == self.dimy):
                        return m
                    else:
                        raise ValueError(
                            'Bad map shape. Must be (%d, %d)'%(
                                self.dimx, self.dimy))
                else:
                    raise ValueError('Map must be a 2D array')
            else:
                raise ValueError('Map must be a numpy array')
        else:
            return None


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
        self._dat = self._check_data(m)

    def _get_err(self):
        """Getter for err property.

        Called by ``a = self.err``.
        """
        if self._err is not None:
            self._err = np.copy(np.abs(self._err))
        return self._err

    def _set_err(self, m):
        """Setter for err property.

        Called by ``self.err = m``.

        :param m: uncertainty map
        """
        self._err = self._check_data(m)

    dat = property(_get_dat, _set_dat)
    err = property(_get_err, _set_err)


    def copy(self):
        """Return a copy of self."""
        dat_copy = np.copy(self._dat)
        
        if self._err is not None:
            err_copy = np.copy(self._err)
        else:
            err_copy = None
            
        return self.__class__(dat_copy, err=err_copy)


#################################################
#### Convenience functions ######################
#################################################
def array(a, e):
    """Return a :py:class:`data.Data1D` or :py:class:`data.Data2D`
    instance depending on the dimensions of the passed arrays.

    :param a: Estimation. Must be convertible to a numpy.ndarray.

    :param e: Uncertainty. Must be convertible to a numpy.ndarray and
      must have the same shape as a.
    """
    if not isinstance(a, np.ndarray):
        try:
            a = np.array(a)
        except Exception, err:
            raise Exception('Estimation array cannot be converted to a numpy.ndarray: {}'.format(err))
        
    if not isinstance(e, np.ndarray):
        try:
            e = np.array(e)
        except Exception, err:
            raise Exception('Uncertainty array cannot be converted to a numpy.ndarray: {}'.format(err))

    if a.shape != e.shape:
        raise ValueError('Estimation array and Error array must have the same shape: {} != {}'.format(a.shape, e.shape))
    
    if np.ndim(a) <= 1:
        return Data1D(a, e)

    if np.ndim(a) == 2:
        return Data2D(a, e)

    else:
        raise ValueError('Array must be a scalar, 1D or 2D')

#################################################
#### Operations on Data #########################
#################################################
# Replaces numpy operations on data with error
# propagation.

def log10(a):
    """Log10 of a Data array"""
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.log10(a.dat)
        result.err = (a.err / a.dat) * math.log10(math.exp(1))
        return result
    else:
        return np.log10(a)



def exp(a):
    """exponential of a Data array"""
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.exp(a.dat)
        result.err = result.dat * a.err
        return result
    else:
        return np.exp(a)

def sqrt(a):
    """exponential of a Data array"""
    if isinstance(a, Data):
        result = a.copy()
        result.dat = np.sqrt(a.dat)
        result.err = (result.dat * 0.5 * (a.err)) / a.dat
        return result
    else:
        return np.sqrt(a)

def sum(a):
    """Sum of a Data array"""
    if isinstance(a, Data):
        return array(np.sum(a.dat), math.sqrt(np.sum(a.err**2.)))
    else:
        return np.sum(a)

def nansum(a):
    """Nansum of a Data array"""
    if isinstance(a, Data):
        return array(np.nansum(a.dat), math.sqrt(np.nansum(a.err**2.)))
    else:
        return np.nansum(a)

def mean(a):
    """Mean of a Data array"""
    return sum(a) / a.dat.size

def nanmean(a):
    """Nanmean of a Data array"""
    return nansum(a) / np.sum(~np.isnan(a.dat))

