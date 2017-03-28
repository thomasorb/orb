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
Manage data with associated uncertainties
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
    dtypes =  [np.float, np.float16, np.float32, np.float64,
               np.complex64, np.complex128, np.longdouble]
    
    def __init__(self, dat, err=None, dtype=None):
        """
        Init of Data class.
        
        :param dat: Data map.
        
        :param err: Absolute error map. Must have the same shape as
          dat.
        """
        if isinstance(dat, np.ndarray) or np.isscalar(dat):
            if dat.dtype not in self.dtypes:
                dtype = np.float
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
                    if self._has_complexity(b):
                       self._err = self._compute_complex_err(
                           lambda x1, x2: x1 * x2, b)
                    else:
                        self._err = result * np.sqrt(
                            (self._err/self._dat)**2.
                            + (b.err/b.dat)**2.)
                else:
                    self._err *= b
            self._dat = result
            
        else:
            if np.any(np.iscomplexobj(b)): self._ascomplex()    
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
                    if self._has_complexity(b):
                        self._err = self._compute_complex_err(
                            lambda x1, x2: x1 / x2, b)
                    else:
                        self._err = result * np.sqrt(
                            (self._err/self._dat)**2.
                            + (b.err/b.dat)**2.)
                else:
                    self._err /= b
            self._dat = result
            
        else:
            if np.any(np.iscomplexobj(b)): self._ascomplex()
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
                if self._has_complexity(b):
                    self._err = self._compute_complex_err(
                        lambda x1, x2: x1 + x2, b)
                else:
                    self._err = np.sqrt((self._err)**2.
                                        + (b.err)**2.)
            self._dat = result
        else:
            if np.any(np.iscomplexobj(b)): self._ascomplex()
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
                if self._has_complexity(b):
                    self._err = self._compute_complex_err(
                        lambda x1, x2: x1 - x2, b)
                else:
                    self._err = np.sqrt((self._err)**2.
                                        + (b.err)**2.)
            self._dat = result
        else:
            if np.any(np.iscomplexobj(b)): self._ascomplex()
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
            if self._has_complexity(b):
                a.err = self._compute_complex_err(
                    lambda x1, x2: x1 ** x2, b, a=a)
            else:
                a.err = np.abs(result) * np.sqrt(
                    (b.dat / a.dat * a.err)**2.
                    + (np.log(a.dat) * b.err)**2.)
        
        a.dat = result
        return a

    def __rpow__(self, b):
        """Implement ``b ** self``

        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        b = array(b)
        a = self.copy()
        result = b.dat ** a.dat
        
        # error propagation
        if a.err is not None or b.err is not None:
            if self._has_complexity(b):
                self._err = self._compute_complex_err(
                    lambda x1, x2: x2 ** x1, b, a=a)
            else:
                b.err = np.abs(result) * np.sqrt(
                    (a.dat / b.dat * b.err)**2.
                    + (np.log(b.dat) * a.err)**2.)
        
        b.dat = result
        return b


    def __neg__(self):
        """Implement ``- b``."""
        result = self.copy()
        result._dat = -result._dat
        return result

    def __pos__(self):
        """Implement ``+ b``."""
        return self.copy()

    def _has_complexity(self, b):
        """Check if an operation present any complexity that involves a
        Monte-Carlo simulation
        """
        if _has_complex_error(self._err + b.err):
            return True
        if (np.any(np.iscomplex(self._err))
            or np.any(np.iscomplex(b.err))):
            if (np.iscomplexobj(self.dat)):
                return True
            if np.any(self._dat == 0.) or np.any(b.dat == 0.):
                return True
        return False
                    

    def _compute_complex_err(self, f, b, a=None):
        """Compute error in the case of complex uncertainties via Monte-Carlo.

        :param f: function to apply

        :param b: A numpy array or a Data instance. Must be
          broadcastable with self.
        """
        if a is None: a = self
        _err = mcu(f,
                   [array(np.copy(a._dat),
                          np.copy(a._err)),
                    array(np.copy(b._dat),
                          np.copy(b._err))])
        return np.copy(_err)

    def _ascomplex(self):
        """Change data type to complex type"""
        if not np.all(np.iscomplexobj(self._dat)):
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
            if (np.any(np.iscomplexobj(m)) 
                and not np.any(np.iscomplexobj(self._dat))):
                raise Exception('Error is complex and data is not !')
            if (m.shape == self.shape):
                return np.array(m, dtype=self.dtype)
            elif np.size(m) == np.size(self._dat):
                return np.broadcast_to(m, self.shape)
            elif np.size(m) == 1:
                return np.array(m, dtype=self.dtype)
            elif np.size(self._dat) == 1:
                return np.array(m, dtype=self.dtype)
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
            self._err.real = np.abs(self._err.real)
            if np.any(np.iscomplexobj(self._err)):
                self._err.imag = np.abs(self._err.imag)

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

def _has_complex_error(a_err):
    """Check if the error is complex and different from 0 in both the
    real and the imaginary part.

    :param a_err: Error array
    """
    if np.any(np.iscomplexobj(a_err)):
        if np.any(a_err.real) and np.any(a_err.imag):
            return True
    return False

def _compute_err(fdat, ferr, _a):
    if isnpdata(_a): _a = array(_a)
        
    if isinstance(_a, Data):
        _result = _a.copy()
        _result.dat = fdat(_a.dat)
        if _has_complex_error(_a.err):
            _result.err = mcu(fdat, _a)
            return _result

        _result.err = ferr(_a.dat, _a.err)
        return _result
    else:
        return fdat(_a)
    
def abs(a):
    """Absolute value of a Data array"""
    ferr = lambda a_dat, a_err: np.abs(a_err)
    return _compute_err(np.abs, ferr, a)

def erf(a):
    """Error function.

    Uncertainty obtained via :math:`err = \sigma_x \frac{d}{dx}erf(x)`
    """
    ferr = lambda a_dat, a_err: 2. * np.exp(-a_dat**2) / math.sqrt(math.pi) * a_err
    return _compute_err(ss.erf, ferr, a)

def dawsn(a):
    """Dawson function

    Uncertainty obtained via :math:`err = \sigma_x \frac{d}{dx}daws(x)`
    """
    ferr = lambda a_dat, a_err:  (1. - 2. * a_dat * ss.dawsn(a_dat)) * a_err
    return _compute_err(ss.dawsn, ferr, a)

def log10(a):
    """Log10 of a Data array"""
    ferr = lambda a_dat, a_err: (a_err / a_dat) * math.log10(math.exp(1))
    return _compute_err(np.log10, ferr, a)


def exp(a):
    """exponential of a Data array"""
    ferr = lambda a_dat, a_err: np.exp(a_dat) * a_err
    return _compute_err(np.exp, ferr, a)

def sqrt(a):
    """exponential of a Data array"""
    ferr = lambda a_dat, a_err: (np.sqrt(a_dat) * 0.5 * (a_err)) / a_dat
    return _compute_err(np.sqrt, ferr, a)

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




##############################################
#### Monte-Carlo approximation ###############
##############################################

def mcu_one_val(f, val, err, **kwargs):
    """Monte-Carlo approximation of the uncertainty of one value
    through a function f.

    :param f: function
    
    :param val: Value (or a list of values if the function needs more
      than one variable)
    
    :param err: Uncertainty on the value (or a list of uncertainties
      if the function needs more than one variable)

    :param kwargs: keyword arguments passed to f

    :return: Uncertainty on f(val)
    """
    N = 10000
    if not isinstance(val, list):
        if not np.isscalar(val):
            raise Exception('val must be scalar (real or complex)')
        val = list([val])
        
    if not isinstance(err, list):
        if not np.isscalar(err):
            raise Exception('err must be scalar (real or complex)')
        err = list([err])

    if len(val) != len(err):
        raise Exception('err and val must have the same length')

    dists = list()

    any_complex = False
    for i in range(len(val)):
        if np.iscomplexobj(val[i]) or np.iscomplexobj(err[i]):
            any_complex = True
            idist = val[i] + (np.random.standard_normal(N) * err[i].real
                              + 1j * np.random.standard_normal(N) * err[i].imag)
        else:
            idist = val[i] + np.random.standard_normal(N) * err[i]
        dists.append(idist)

    errdist = f(*dists, **kwargs)

    if any_complex:
        return np.nanstd(errdist.real) + 1j*np.nanstd(errdist.imag)
    else:
        return np.nanstd(errdist)
    
def mcu(f, arr, **kwargs):
    """Monte-Carlo approximation of the uncertainty on an array
    through a function f.

    :param f: function
    
    :param arr: orb.Data instance or a list of orb.Data instances (for
      a function with mutiple variables, note that all arrays must
      have the same shape)


    :param kwargs: keyword arguments passed to f

    :return: Uncertainty on f(arr)
    """
    if isinstance(arr, list):
        arrs = list()
        for iarr in arr:
            arrs.append(array(iarr))
        err = np.empty(arrs[0].shape, dtype=arrs[0].dtype)
        if arrs[0].dat.size > 1:
            for i in range(arrs[0].dat.size):
                ivals = list()
                ierrs = list()
                for j in range(len(arrs)):
                    ivals.append(arrs[j].dat[i])
                    ierrs.append(arrs[j].err[i])
                err[i] = mcu_one_val(f, ivals, ierrs, **kwargs)
        else:
            _dat = [arrs[i].dat for i in range(len(arrs))]
            _err = [arrs[i].err for i in range(len(arrs))]
            err = mcu_one_val(f, _dat, _err, **kwargs)
            
    else:
        arr = array(arr)
        err = np.empty(arr.shape, dtype=arr.dtype)
        if arr.dat.size > 1:
            for i in range(arr.dat.size):
                err[i] = mcu_one_val(f, arr.dat[i], arr.err[i])
        else:
            err = mcu_one_val(f, arr.dat, arr.err, **kwargs)
            
        
    return err
        



##############################################
#### Testing functions #######################
##############################################


def uncertainty_testing_one_val(f, val, err):
    N = int(1e4)
    odval = array(val, err)
    yerr_sim = f(*odval)

    yerr = list()
    for j in range(N):
        val_rnd = list()
        for i in range(len(val)):
            val_rnd.append(val[i] + np.random.standard_normal()*err[i])
        yerr.append(f(*val_rnd))
    yerr = np.array(yerr)
    if np.any(np.iscomplexobj(yerr)):
        yerr_std = np.nanstd(yerr.real) + 1j*np.nanstd(yerr.imag)
    else:
        yerr_std = np.nanstd(yerr)
    print "simulated value: ", yerr_sim
    print "value obtained from randomized guess: ", np.nanmean(yerr), yerr_std
    return (yerr_sim.dat - np.nanmean(yerr))/ np.nanmean(yerr), (yerr_sim.err - yerr_std)/yerr_std



def uncertainty_testing(f, val_min, val_max, err, show=True):
    """Test returned uncertainty.

    :param f: function to test
    :param val_min: Min value, can be complex
    :param val_max: Max value, can be complex
    :param err: uncertainty on the value, can be complex
    
    :param show: (Optional) If True show the results (default True).
    """

    N = 1e4
    PARTS = 101

    if (np.iscomplexobj(val_min)
        or np.iscomplexobj(val_max)
        or np.iscomplexobj(err)):
        IS_COMPLEX = True
        val_min = complex(val_min)
        val_max = complex(val_max)
        err = complex(err)
    else:
        IS_COMPLEX = False
        

    if IS_COMPLEX:
        vals = np.zeros(int(N))
        if val_min.real != 0. and val_max.real != 0.:
            vals += 10**np.linspace(log10(val_min.real), log10(val_max.real), N)
        if val_min.imag != 0. and val_max.imag != 0.:
            vals = vals.astype(complex)
            vals += 1j * 10**np.linspace(log10(val_min.imag), log10(val_max.imag), N)
             
        vals_rnd = vals + (np.random.standard_normal(vals.shape[0]) * err.real
                           + 1j * (np.random.standard_normal(vals.shape[0]) * err.imag))
        
    else:
        vals = 10**np.linspace(log10(val_min), log10(val_max), N)
        vals_rnd = vals + np.random.standard_normal(vals.shape[0]) * err

    y = f(vals)
    yrnd = f(vals_rnd)
    yerr = yrnd - y
    yerr_sim = f(array(vals, np.ones_like(vals) * err)).err

    part = np.linspace(0, 100, PARTS+1)
    yerr_std_list = list()
    vals_bin_list = list()
    for i in range(part.shape[0] - 1):
        if IS_COMPLEX:
            percmin = (np.nanpercentile(vals.real, part[i])
                       + 1j * np.nanpercentile(vals.imag, part[i]))
            percmax = (np.nanpercentile(vals.real, part[i+1])
                       + 1j * np.nanpercentile(vals.imag, part[i+1]))
            nz_real = np.nonzero((vals.real > percmin.real) * (vals.real <= percmax.real))
            nz_imag = np.nonzero((vals.imag > percmin.imag) * (vals.imag <= percmax.imag))    
            vals_bin_list.append(np.nanmedian(vals.real[nz_real])
                                 + 1j * np.nanmedian(vals.imag[nz_imag]))
            yerr_std_list.append(np.nanstd(yerr.real[nz_real])
                                 + 1j * np.nanstd(yerr.imag[nz_imag]))

        else:
            percmin = np.nanpercentile(vals, part[i])
            percmax = np.nanpercentile(vals, part[i+1])
            nz = np.nonzero((vals > percmin) * (vals <= percmax))
            vals_bin_list.append(np.nanmedian(vals[nz]))
            yerr_std_list.append(np.nanstd(yerr[nz]))

                
    yerr_std_list = np.array(yerr_std_list)
    vals_bin_list = np.array(vals_bin_list)

    if show:
        import pylab as pl
        if IS_COMPLEX:
            pl.figure()
            pl.scatter(vals.real, yerr.real, marker='+', color='green')
            pl.plot(vals.real, yerr_sim.real, color='yellow')
            pl.scatter(vals_bin_list.real, yerr_std_list.real, color = 'red')
            pl.title('Real part')
            pl.xscale('log')
            pl.yscale('log')

            pl.figure()
            pl.scatter(vals.imag, yerr.imag, marker='+', color='cyan')
            pl.plot(vals.imag, yerr_sim.imag, color='orange')
            pl.scatter(vals_bin_list.imag, yerr_std_list.imag, color = 'blue')
            pl.title('Imaginary part')
            pl.xscale('log')
            pl.yscale('log')

        else:
            pl.scatter(vals, yerr, marker='+', color='cyan')
            pl.plot(vals, yerr_sim, color='orange')
            pl.scatter(vals_bin_list, yerr_std_list, color = 'red')
            pl.xscale('log')
            pl.yscale('log')
        pl.show()

        


    
    
