# cython: embedsignature=True
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: gvarc.pyx

## Copyright (c) 2010-2017 Thomas Martin <thomas.martin.1@ulaval.ca>
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
import warnings

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


def sincgauss_real_erf(double a,
                       np.ndarray[np.float64_t, ndim=1] b):
    """Sincgauss function (erf formulation) of a and b which returns a
    real vector.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] erf1
    cdef np.ndarray[np.complex128_t, ndim=1] erf2
    cdef complex erf3

    erf1 = scipy.special.erf(a - 1j * b)
    erf2 = scipy.special.erf(a + 1j * b)
    erf3 = scipy.special.erf(a)

    return (np.exp(-b**2.) * (erf1 + erf2) / (2 * erf3)).real

def sincgauss_real_dfdx(double a,
                        np.ndarray[np.float64_t, ndim=1] b,
                        erf=False, f_ab=None):
    """Partial derivatives wrt. a and b of the sincgauss function.

    Returns real derivatives.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] cos1
    cdef np.ndarray[np.complex128_t, ndim=1] sin1
    cdef complex dia
    cdef np.ndarray[np.complex128_t, ndim=1] dfda
    cdef np.ndarray[np.complex128_t, ndim=1] dfdb

    dia = scipy.special.dawsn(1j * a)

    if erf: sgr = sincgauss_real_erf
    else: sgr = sincgauss_real

    if f_ab is None:
        SGR_AB = sgr(a, b)
    else:
        SGR_AB = f_ab
        
    cos1 = 1j * np.cos(2. * a * b)
    dfda = (cos1 - 1j * SGR_AB) / dia
    
    sin1 = 1j * np.sin(2. * a * b) / dia
    dfdb = sin1 - 2. * b * SGR_AB
    return (dfda.real, dfdb.real)


def sincgauss1d(a, b, erf=False):
    """sincgauss function"""
    cdef int i

    assert (np.size(a)) == 1

    if erf: sgr = sincgauss_real_erf
    else: sgr = sincgauss_real

    if not isinstance(a, gvar.GVar) and not isinstance(b[0], gvar.GVar):
        return sgr(a, b)
    
    cdef np.ndarray[object, ndim=1] ans = np.empty(b.shape[0], b.dtype)
    cdef np.ndarray[np.float64_t, ndim=1] f = np.copy(sgr(gvar.mean(a), gvar.mean(b)))

    cdef np.ndarray[np.float64_t, ndim=1] dfda
    cdef np.ndarray[np.float64_t, ndim=1] dfdb

        
    dfda, dfdb = sincgauss_real_dfdx(gvar.mean(a), gvar.mean(b), erf=erf, f_ab=f)

    if not isinstance(a, gvar.GVar):
        for i in range(b.size):
            ans.flat[i] = gvar.gvar_function(
                b[i], f[i], dfdb[i])
        
    else:
        for i in range(b.size):
            ans.flat[i] = gvar.gvar_function(
                (a, b[i]), f[i], (dfda[i], dfdb[i]))

    return ans



#########################################
####### PHASED VERSIONS #################
#########################################

def sincgausscomplex_real(double a,
                          np.ndarray[np.float64_t, ndim=1] b):
    """Sincgausscomplex function of a and b which returns two real
    vectors instead of a complex vector.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] dawson1
    cdef np.ndarray[np.float64_t, ndim=1] dawson2
    cdef complex dawson3
    
    dawson1 = scipy.special.dawsn(1j * a + b) * np.exp(2.* 1j * a * b)
    dawson2 = scipy.special.dawsn(b) * np.exp(a ** 2.)
    dawson3 = scipy.special.dawsn(1j * a)

    sgc = (dawson1 - dawson2) / dawson3
    return (sgc.real, sgc.imag)

def sincgausscomplex_real_erf(double a,
                              np.ndarray[np.float64_t, ndim=1] b):
    """Sincgausscomplex function (erf formulation) of a and b which returns two real
    vectors instead of a complex vector.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] erf1
    cdef np.ndarray[np.complex128_t, ndim=1] erf2
    cdef complex erf3
    warnings.simplefilter("ignore")
    erf1 = scipy.special.erf(a - 1j * b)
    erf2 = scipy.special.erf(1j * b)
    erf3 = scipy.special.erf(a)
    sgc = np.exp(-(b**2)) * ((erf1 + erf2)/erf3)
    sgc[np.isnan(sgc)] = 0.
    warnings.simplefilter("default")
    
    return (sgc.real, sgc.imag)


def sincgausscomplex_real_dfdx(double a,
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
    diab = scipy.special.dawsn(1j * a + b)
    db = scipy.special.dawsn(b)
    exp2iab = np.exp(2.* 1j * a * b)
    expa2 = np.exp(a ** 2.)
    
    dfda = 1j * (exp2iab * (dia - diab) + db * expa2) / (dia ** 2.)
    dfdb = ((1 - 2 * b * diab) * exp2iab - (1 - 2 * b * db) * expa2) / dia
    return dfda.real, dfda.imag, dfdb.real, dfdb.imag

def sincgausscomplex_real_dfdx_erf(double a,
                                   np.ndarray[np.float64_t, ndim=1] b):
    """Partial derivatives (erf formulation) wrt. a and b of the
    sincgauss function.

    Returns real derivatives.
    """
    cdef np.ndarray[np.complex128_t, ndim=1] cos1
    cdef np.ndarray[np.complex128_t, ndim=1] sin1
    cdef complex dia
    cdef np.ndarray[np.complex128_t, ndim=1] dfda
    cdef np.ndarray[np.complex128_t, ndim=1] dfdb
    warnings.simplefilter("ignore")
    erfa = scipy.special.erf(-a)
    exp2iab = np.exp(2.* 1j * a * b)
    expmb2 = np.exp(-(b**2.))
    erf2 = scipy.special.erf(1j * b - a)
    erf3 = scipy.special.erf(1j * b)
    expma2 = np.exp(-(a**2.))
    beta = -1j * np.sqrt(np.pi) / 2.
    
    dfda =  - 2. / np.sqrt(np.pi) * expma2 * (exp2iab * erfa + expmb2 * (-erf2 + erf3)) /  (erfa**2.)
    dfdb = ((np.exp(2j * a * b - a**2) - 1)/beta + 2 * b * expmb2 * (- erf2 + erf3)) / erfa

    dfda[np.isnan(dfda.real)] = 0.
    dfda[np.isnan(dfda.imag)] = 0.
    dfdb[np.isnan(dfdb.real)] = 0.
    dfdb[np.isnan(dfdb.imag)] = 0.
    warnings.simplefilter("default")
    return dfda.real, dfda.imag, dfdb.real, dfdb.imag


def sincgauss1d_complex(a, b, erf=False):
    """The "complex" version of the sincgauss (dawson definition).

    This is the real sinc*gauss function when ones wants to fit both the real
    part and the imaginary part of the spectrum."""

    cdef int i

    assert (np.size(a)) == 1

    if not erf:
        sgc = sincgausscomplex_real
        sgc_dfdx = sincgausscomplex_real_dfdx
    else:
        sgc = sincgausscomplex_real_erf
        sgc_dfdx = sincgausscomplex_real_dfdx_erf
    

    if not isinstance(a, gvar.GVar) and not isinstance(b[0], gvar.GVar):
        return sgc(a, b)
    
    cdef np.ndarray[object, ndim=1] ans_re = np.empty(b.shape[0], b.dtype)
    cdef np.ndarray[object, ndim=1] ans_im = np.empty(b.shape[0], b.dtype)
    
    cdef np.ndarray[np.float64_t, ndim=1] f_re
    cdef np.ndarray[np.float64_t, ndim=1] f_im
    cdef np.ndarray[np.float64_t, ndim=1] dfda_re
    cdef np.ndarray[np.float64_t, ndim=1] dfdb_re
    cdef np.ndarray[np.float64_t, ndim=1] dfda_im
    cdef np.ndarray[np.float64_t, ndim=1] dfdb_im


    f_re, f_im = np.copy(sgc(gvar.mean(a), gvar.mean(b)))
    
    dfda_re, dfda_im, dfdb_re, dfdb_im = sgc_dfdx(gvar.mean(a), gvar.mean(b))

    # real part
    if not isinstance(a, gvar.GVar):
        for i in range(b.size):
            ans_re.flat[i] = gvar.gvar_function(
                b[i], f_re[i], dfdb_re[i])
        
    else:
        for i in range(b.size):
            ans_re.flat[i] = gvar.gvar_function(
                (a, b[i]), f_re[i], (dfda_re[i], dfdb_re[i]))

    # imag part
    if not isinstance(a, gvar.GVar):
        for i in range(b.size):
            ans_im.flat[i] = gvar.gvar_function(
                b[i], f_im[i], dfdb_im[i])
        
    else:
        for i in range(b.size):
            ans_im.flat[i] = gvar.gvar_function(
                (a, b[i]), f_im[i], (dfda_im[i], dfdb_im[i]))


    return ans_re, ans_im
