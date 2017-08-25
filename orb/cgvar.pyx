# cython: embedsignature=True
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: gvarc.pyx

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


cimport numpy as np
cimport cython
import numpy as np
import scipy.special
cimport gvar
import gvar

cdef extern from "math.h":
    double c_pow "pow" (double x,double y)
    double c_sin "sin" (double x)
    double c_cos "cos" (double x)
    double c_tan "tan" (double x)
    double c_sinh "sinh" (double x)
    double c_cosh "cosh" (double x)
    double c_tanh "tanh" (double x)
    double c_log "log" (double x)
    double c_exp "exp" (double x)
    double c_sqrt "sqrt" (double x)
    double c_asin "asin" (double x)
    double c_acos "acos" (double x)
    double c_atan "atan" (double x)


def sinc_dfdx(np.ndarray[np.float64_t, ndim=1] x):
    """Derivative of the sinc function"""
    cdef np.ndarray[np.float64_t, ndim=1] pi_x = np.pi * x
    return np.where(x != 0, (pi_x * np.cos(pi_x) - np.sin(pi_x)) / (pi_x * x), 0)

    
def sinc1d(x):
    """sinc function"""
    cdef int i

    if not isinstance(x[0], gvar.GVar):
        return np.sinc(x)

    cdef np.ndarray[object, ndim=1] ans = np.empty(x.shape[0], x.dtype)
    cdef np.ndarray[np.float64_t, ndim=1] f = np.sinc(gvar.mean(x))
    cdef np.ndarray[np.float64_t, ndim=1] dfdx = sinc_dfdx(gvar.mean(x))
        
    for i in range(x.size):
        ans[i] = gvar.gvar_function(x[i], f[i], dfdx[i])
    return ans



def dawsni_dfdx(double im):
    """Derivative of the dawson function for an imaginary input

    :param im: Imaginary input

    :return: an imaginary float
    """
    return (2 * im * scipy.special.dawsn(1j * im) + 1j).imag

def dawsni(gvar.GVar im):
    """Dawson function for an imaginary float input

    :param im: Imaginary gvar.GVar input

    :return: an imaginary gvar.GVar scalar
    """
    cdef double f = scipy.special.dawsn(1j * im.mean).imag
    cdef double dfdx = dawsni_dfdx(im.mean)
    return gvar.gvar_function(im, f, dfdx)
    


## def dawsn_dfdx(double re, double im):
##     """Derivative of the Dawson function for complex input

##     :param re: real part
##     :param im: imaginary part
##     """
##     cdef double x
##     return 1. - 2. * x * scipy.special.dawsn(x)

## def dawsn(re, im):
##     """Dawson function for complex input

##     :param re: real part
##     :param im: imaginary part
##     """
##     cdef int i
##     cdef float xi, f, dfdx
    
##     if isinstance(x, gvar.GVar):
##         f = scipy.special.dawsn(re.mean + 1j * im.mean)
##         dfdx = dawsn_dfdx(x.mean)
##         return gvar.gvar_function(x, f, dfdx)
##     else:
##         x = np.asarray(x)
##         ans = np.empty(x.shape, x.dtype)
##         for i in range(x.size):
##             try:
##                 ans.flat[i] = scipy.special.dawsn(x.flat[i])
##             except TypeError:
##                 xi = x.flat[i]
##                 f = scipy.special.dawsn(xi.mean)
##                 dfdx = dawsn_dfdx(xi.mean)
##                 ans.flat[i] = gvar.gvar_function(xi, f, dfdx)
##         return ans


## def exp(x):
##     if np.iscomplexobj()


def sincgauss_real(double a,
                   np.ndarray[np.float64_t, ndim=1] b):
    """Sincgauss function of a and b which returns a real vector.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] dawson1
    cdef np.ndarray[np.complex128_t, ndim=1] dawson2
    cdef complex dawson3
    
    dawson1 = scipy.special.dawsn(1j * a + b) * np.exp(2.* 1j * a *b)
    dawson2 = scipy.special.dawsn(1j * a - b) * np.exp(-2. * 1j * a *b)
    dawson3 = 2. * scipy.special.dawsn(1j * a)
    
    return ((dawson1 + dawson2)/dawson3).real

def sincgauss_real_dfdx(double a,
                        np.ndarray[np.float64_t, ndim=1] b):
    """Partial derivatives wrt. a and b of the sincgauss function.

    Returns real derivatives.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] cos1
    cdef np.ndarray[np.complex128_t, ndim=1] sin1
    cdef complex dia
    cdef np.ndarray[np.complex128_t, ndim=1] dfda
    cdef np.ndarray[np.complex128_t, ndim=1] dfdb

    dia = scipy.special.dawsn(1j * a)

    cos1 = 1j * np.cos(2. * a * b)
    dfda = (cos1 - 1j * sincgauss_real(a, b)) / dia
    
    sin1 = 1j * np.sin(2. * a * b) / dia
    dfdb = sin1 - 2. * b * sincgauss_real(a, b)
    return (dfda.real, dfdb.real)

def sincgauss1d(a, b):
    """sincgauss function"""
    cdef int i

    assert (np.size(a)) == 1

    if not isinstance(a, gvar.GVar) and not isinstance(b[0], gvar.GVar):
        return sincgauss_real(a, b)
    
    cdef np.ndarray[object, ndim=1] ans = np.empty(b.shape[0], b.dtype)
    cdef np.ndarray[np.float64_t, ndim=1] f = np.copy(sincgauss_real(gvar.mean(a), gvar.mean(b)))
    cdef np.ndarray[np.float64_t, ndim=1] dfda
    cdef np.ndarray[np.float64_t, ndim=1] dfdb

    dfda, dfdb = sincgauss_real_dfdx(gvar.mean(a), gvar.mean(b))

    if not isinstance(a, gvar.GVar):
        for i in range(b.size):
            ans.flat[i] = gvar.gvar_function(
                b[i], f[i], dfdb[i])
        
    else:
        for i in range(b.size):
            ans.flat[i] = gvar.gvar_function(
                (a, b[i]), f[i], (dfda[i], dfdb[i]))

    return ans


## def sincgauss_real_dfdb(a_, b_):
##     """Partial derivative wrt. b of the sincgauss function of a and b. Returns
##     a real derivative"""
##     a_ = np.array(a_, dtype=float)
##     b_ = np.array(b_, dtype=float)
    
##     cdef sin1 = a_ * 1j * np.sin(2.*a_*b_) / scipy.special.dawsn(1j*a_)
    
##     cdef dfdb = sin1 - 2 * b_ * sincgauss_real(a_, b_)
##     return dfdb.real

## def sincgauss_real_dfdx(a_, b_):
##     """Derivatives wrt. a and b of the sincgauss function. Returns a
##     tuple of partial derivatives along a and b.
##     """
##     return (sincgauss_real_dfda(a_, b_), sincgauss_real_dfdb(a_, b_))

## def sincgauss1d(ab):
##     """Sincgauss function for a tuple of parameters (a, b). Returns a
##     real vector"""
##     cdef int i
    
##     if isinstance(ab[0], gvar.GVar) and isinstance(ab[1], gvar.GVar):
##         if ab[0].sdev == 0. and ab[1].sdev == 0.:
##             return sincgauss_real(ab[0].mean, ab[1].mean)
##         f = sincgauss_real(gvar.mean(ab[0]), gvar.mean(ab[1]))
##         dfdx = sincgauss_real_dfdx(gvar.mean(ab[0]), gvar.mean(ab[1]))
##         return gvar.gvar_function(ab, f, dfdx)
##     else:
##         a = ab[0]
##         b = np.asarray(ab[1])

##         ans = np.empty_like(b)
##         for i in range(b.size):
##             try:
##                 ans.flat[i] = sincgauss_real(a, b.flat[i])
##             except TypeError:
##                 bi = b.flat[i]
##                 f = sincgauss_real(a.mean, bi.mean)
##                 dfdx = sincgauss_real_dfdx(a.mean, bi.mean)
##                 ans.flat[i] = gvar.gvar_function((a, bi), f, dfdx)
                
##         return ans
