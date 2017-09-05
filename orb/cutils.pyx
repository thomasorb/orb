# cython: embedsignature=True
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: cutils.pyx

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
CUtils is a set of C functions coded in Cython_ to improve their speed.

.. note:: This file must be compiled before it can be used::

     cython cutils.pyx
     
     gcc -c -fPIC -I/usr/include/python2.7 cutils.c
     
     gcc -shared cutils.o -o cutils.so

.. _Cython: http://cython.org/

"""
cimport cython
import numpy as np
cimport numpy as np
import time
import scipy.ndimage.filters
import scipy.optimize
import scipy.interpolate
import scipy.special
import constants

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

# define long double for numpy arrays
ctypedef long double float128_t

def radians(double deg):
    """Convert degrees to radians
    """
    return deg * M_PI / 180.


def transform_B_to_A(double x1, double y1, double dx, double dy, double dr,
                     double da, double db, double xrc, double yrc, double zx,
                     double zy):
    """Transform star positions in camera B to the same star position
       in camera A given the transformation parameters.

    .. warning:: this function is meant to be the inverse of
      transform_A_to_B. Given the output of transform_A_to_B and the
      **SAME transformation parameters** the initial positions must be
      returned by this function within the numerical error. i.e. if
      (Xb, Yb) = A_to_B(Xi,Yi,p) and (Xa, Ya) = B_to_A(Xb, Yb, p) then
      Xa - Xi ~ 1e-13 and Ya - Yi ~ 1e-13 (for float64 numbers)

    :param x1: x coordinate to transform
    :param y1: y coordinate to transform
    :param dx: translation along x
    :param dy: translation along y
    :param dr: rotation in the plane of the image
    :param da: tip angle
    :param db: tilt angle
    :param xrc: x coordinate of the rotation center
    :param yrc: y coordinate of the rotation center
    :param zx: zoom coefficient along x
    :param zy: zoom coefficient along y
    """
    cdef double x1_orig = x1
    cdef double y1_orig = y1
    cdef double x2_err = 0.
    cdef double y2_err = 0.
    cdef double a, b, c, d, a_err, b_err, c_err, d_err
    cdef double x2 = 0.
    cdef double y2 = 0.
    
    dr = radians(dr)
    da = radians(da)
    db = radians(db)

    # rotation
    x2 = (x1 - xrc) * cos(dr) - (y1 - yrc) * sin(dr) + xrc
    y2 = (y1 - yrc) * cos(dr) + (x1 - xrc) * sin(dr) + yrc

    # da, db, zx, zy
    x2 = x2 + dx
    y2 = y2 + dy
    
    x2 = x2 * cos(da) * zx
    y2 = y2 * cos(db) * zy

    return x2, y2

def transform_A_to_B(double x1, double y1, double dx, double dy, double dr,
                     double da, double db, double xrc, double yrc, double zx,
                     double zy,
                     double x1_err=0.,
                     double y1_err=0.,
                     double dx_err=0.,
                     double dy_err=0.,
                     double dr_err=0.,
                     double zx_err=0.,
                     double zy_err=0.,
                     bool return_err=False):
    """Transform star positions in camera A to the same star position
    in camera B given the transformation parameters

    :param x1: x coordinate to transform
    :param y1: y coordinate to transform
    :param dx: translation along x
    :param dy: translation along y
    :param dr: rotation in the plane of the image
    :param da: tip angle
    :param db: tilt angle
    :param xrc: x coordinate of the rotation center
    :param yrc: y coordinate of the rotation center
    :param zx: zoom coefficient along x
    :param zy: zoom coefficient along y
    :param x1_err: (Optional) Error on x1 estimate (default 0.)
    :param y1_err: (Optional) Error on y1 estimate (default 0.)
    :param dx_err: (Optional) Error on dx estimate (default 0.)
    :param dy_err: (Optional) Error on dy estimate (default 0.)
    :param dr_err: (Optional) Error on dr estimate (default 0.)
    :param zx_err: (Optional) Error on zx estimate (default 0.)
    :param zy_err: (Optional) Error on zy estimate (default 0.)  
    :param return_err: (Optional) If True, the error on the estimate
      of x2 and y2 is returned.

    :return: (x2, y2) = f(x1, y1). (x2, y2, x2_err, y2_err) if
      return_err is True.
    """
    cdef double x1_orig = x1
    cdef double y1_orig = y1
    cdef double x2_err = 0.
    cdef double y2_err = 0.
    cdef double a, b, c, d, a_err, b_err, c_err, d_err
    cdef double x2 = 0.
    cdef double y2 = 0.
    
    dr = radians(dr)
    da = radians(da)
    db = radians(db)

    # da, db, zx, zy
    x1 = x1 / cos(da) / zx
    y1 = y1 / cos(db) / zy


    if return_err:
        x1_err = x1 * sqrt((zx_err / zx)**2. + (x1_err / x1_orig)**2.)
        y1_err = y1 * sqrt((zy_err / zy)**2. + (y1_err / y1_orig)**2.)
        
    x1 = x1 - dx
    y1 = y1 - dy
    
    if return_err:
        x1_err = sqrt(x1_err**2. + dx_err**2.)
        y1_err = sqrt(y1_err**2. + dy_err**2.)

    # rotation
    x2 = (x1 - xrc) * cos(-dr) - (y1 - yrc) * sin(-dr) + xrc
    y2 = (y1 - yrc) * cos(-dr) + (x1 - xrc) * sin(-dr) + yrc
    
    if return_err:
        dr_err = radians(dr_err)
        a = (x1 - xrc) ; a_err = x1_err
        b = cos(-dr) ; b_err = dr_err * sin(-dr)
        c = (y1 - yrc) ; c_err = y1_err
        d = sin(-dr) ; d_err = dr_err * cos(-dr)
        x2_err = sqrt((sqrt((a_err*b)**2. + (b_err*a)**2.))**2.
                      + (sqrt((c_err*d)**2. + (d_err*c)**2.))**2.)

        a = (y1 - yrc) ; a_err = y1_err
        b = cos(-dr) ; b_err = dr_err * sin(-dr)
        c = (x1 - xrc) ; c_err = x1_err
        d = sin(-dr) ; d_err = dr_err * cos(-dr)
        y2_err = sqrt((sqrt((a_err*b)**2. + (b_err*a)**2.))**2.
                      + (sqrt((c_err*d)**2. + (d_err*c)**2.))**2.)
     
        return x2, y2, x2_err, y2_err
    
    return x2, y2

def sip_im2pix(np.ndarray[np.float64_t, ndim=2] im_coords, sip,
               tolerance=1e-8):
    """Transform perfect pixel positions to distorded pixels positions 

    :param im_coords: perfect pixel positions as an Nx2 array of floats.
    :param sip: pywcs.WCS() instance containing SIP parameters.
    :param tolerance: tolerance on the iterative method.

    .. warning:: SIP.foc2pix must be used instead
    """
    cdef np.ndarray[np.float64_t, ndim=1] xcoord = np.empty(
        im_coords.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] ycoord = np.empty(
        im_coords.shape[0], dtype=np.float64)
    
    xcoord, ycoord = sip.wcs_pix2world(im_coords[:,1], im_coords[:,0], 0)
    xcoord, ycoord = sip.all_world2pix(xcoord, ycoord, 0, tolerance=tolerance)
   
    return np.array([ycoord, xcoord]).T

def sip_pix2im(np.ndarray[np.float64_t, ndim=2] pix_coords, sip):
    """Transform distorded pixel positions to perfect pixels positions 

    :param pix_coords: distorded pixel positions as an Nx2 array of floats.
    :param sip: pywcs.WCS() instance containing SIP parameters.

    .. warning:: SIP.pix2foc must be used instead
    """
    cdef np.ndarray[np.float64_t, ndim=1] xcoord = np.empty(
        pix_coords.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] ycoord = np.empty(
        pix_coords.shape[0], dtype=np.float64)
    
    xcoord, ycoord = sip.all_pix2world(pix_coords[:,1], pix_coords[:,0], 0)
    xcoord, ycoord = sip.wcs_world2pix(xcoord, ycoord, 0)
    
    return np.array([ycoord, xcoord]).T

@cython.boundscheck(False)
@cython.wraparound(False)
def create_transform_maps(int nx, int ny, double dx, double dy,
                          double dr, double da, double db, double xrc,
                          double yrc, double zx, double zy, sip_A, sip_B):
    """Create the 2 transformation maps used to compute the
    geometrical transformation.

    :param nx: size of the frame along x axis
    :param ny: size of the frame along y axis
    :param dx: translation along x
    :param dy: translation along y
    :param dr: rotation in the plane of the image
    :param da: tip angle
    :param db: tilt angle
    :param xrc: x coordinate of the rotation center
    :param yrc: y coordinate of the rotation center
    :param zx: zoom coefficient along x
    :param zy: zoom coefficient along y
    
    :param sip_A: pywcs.WCS() instance containing SIP parameters of
      the output image.
    :param sip_B: pywcs.WCS() instance containing SIP parameters of
      the input image.
    """

    cdef np.ndarray[np.float64_t, ndim=2] outx = np.zeros(
        (nx, ny), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] outy = np.zeros(
        (nx, ny), dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=2] coords = np.zeros(
        (nx*ny, 2), dtype=np.float64)
    
    cdef int ii, ij

            
    if sip_A is not None:  
        outx, outy = np.mgrid[0:nx:1, 0:ny:1].astype(np.float64)
        coords[:,0] = outx.flatten()
        coords[:,1] = outy.flatten()
        coords = sip_pix2im(coords, sip_A)
        outx = np.reshape(coords[:,0], [outx.shape[0], outx.shape[1]])
        outy = np.reshape(coords[:,1], [outy.shape[0], outy.shape[1]])
        
        
    for ii in range(nx):
        for ij in range(ny):
            if sip_A is None:
                outx[ii,ij], outy[ii,ij] = transform_A_to_B(
                    ii, ij, dx, dy, dr, da, db, xrc, yrc, zx, zy)
            else:
                outx[ii,ij], outy[ii,ij] = transform_A_to_B(
                    outx[ii,ij], outy[ii,ij],
                    dx, dy, dr, da, db, xrc, yrc, zx, zy)

    if sip_B is not None:
        coords[:,0] = outx.flatten()
        coords[:,1] = outy.flatten()
        coords = sip_im2pix(coords, sip_B)
        outx = np.reshape(coords[:,0], [outx.shape[0], outx.shape[1]])
        outy = np.reshape(coords[:,1], [outy.shape[0], outy.shape[1]])
        
    return outx, outy


@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_array2d(double h, double a, double dx, double dy, double fwhm,
                     int nx, int ny):
    """Return the 2D profile of a gaussian

    :param h: Height
    :param a: Amplitude
    :param dx: X position
    :param dy: Y position
    :param fwhm: FWHM
    :param nx: X dimension of the output array
    :param ny: Y dimension of the output array
    """
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.zeros(
        (nx, ny), dtype=np.float64)
    cdef double r = 0.
    cdef double w = fwhm / (2. * sqrt(2. * log(2.)))
    cdef int ii, ij
    
    with nogil:
        for ii in range(nx):
            for ij in range(ny):
                r = sqrt(((<double> ii) - dx)**2. + ((<double> ij) - dy)**2.)
                arr[ii,ij] = h + a * exp((-r**2.)/(2.*(w**2.)))
    
    return arr

def moffat_array2d(double h, double a, double dx, double dy,
                   double fwhm, double beta, int nx, int ny):
    
    """Return the 2D profile of a moffat

    :param h: Height
    :param a: Amplitude
    :param dx: X position
    :param dy: Y position
    :param fwhm: FWHM
    :param beta: Beta
    :param nx: X dimension of the output array
    :param ny: Y dimension of the output array
    """
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.zeros((nx, ny))
    cdef double r = 0.
    cdef double w = fwhm / (2. * sqrt(2. * log(2.)))
    cdef double alpha

    if beta > 0.:
        alpha = fwhm / (2. * sqrt(2**(1. / beta) - 1.))

        for ii in range(nx):
            for ij in range(ny):
                r = sqrt((ii - dx)**2. + (ij - dy)**2.)
                arr[ii,ij] = h + a * (1. + (r/alpha)**2.)**(-beta)
    else:
        arr.fill(np.nan)
    
    return arr

@cython.boundscheck(False)
@cython.wraparound(False)
def surface_value(int dimx, int dimy, double xc, double yc, double rmin,
                  double rmax, long sub_div):
    """Return an approximation of the surface value of a pixel given
    the min and max radius of an annulus in pixels.

    :param dimx: dimension of the box along x
    
    :param dimy: dimension of the box along y
    
    :param xc: center of the annulus along x
    
    :param yc: center of the annulus along y
    
    :param rmin: min radius of the annulus
    
    :param rmax: max radius of the annulus
    
    :param sub_div: Number of subdivisions to make (the higher,the
      better but the longer too)
    """
    cdef double dsub_div = <double> sub_div
    cdef double value = 1. / dsub_div**2
    cdef int isub
    cdef int jsub
    cdef double r
    cdef int sub_dimx = dimx * sub_div
    cdef int sub_dimy = dimy * sub_div
    cdef np.ndarray[np.float64_t, ndim=2] S = np.zeros((dimx, dimy))
    
    xc += 0.5
    yc += 0.5

    with nogil:
        for isub in range(sub_dimx):
            for jsub in range(sub_dimy):
                r = sqrt(((<double> isub) - (xc * dsub_div - 0.5))**2.
                         + ((<double> jsub) - (yc * dsub_div - 0.5))**2.)
                if r <= rmax * dsub_div and r >= rmin * dsub_div:           
                    S[<int>((<double>isub)/dsub_div),
                      <int>((<double>jsub)/dsub_div)] += value
                
    return S


def sigmacut(np.ndarray[np.float64_t, ndim=1] x, double central_value,
             int use_central_value, double sigma, int min_values,
             return_index_list=False):
    """Return a distribution after a sigma cut rejection
    of the too deviant values.

    :param x: The distribution to cut
    
    :param sigma: Number of sigma above which values are considered as
      deviant

    :param min_values: Minimum number of values to return

    :param central_value: If not none, this value is used as the
      central value of the cut. Else the median of the distribution is
      used as the central value

    :param use_central_value: If True central value is used instead of
      the median.

    :param return_index_list: (Optional) If True the list of the non
      rejected values is returned also (default False).
    """
    cdef int still_rejection = 1
    cdef double central_x
    cdef int new_sz
    cdef int sz
    
    if return_index_list:
        index_list = np.arange(np.size(x))
        
    if bn.anynan(x):
        x = x[np.nonzero(~np.isnan(x))]
        if return_index_list:
            index_list = index_list[np.nonzero(~np.isnan(x))]
    
    while still_rejection:
        sz = np.size(x)
        if not use_central_value:
            central_x = bn.nanmedian(x)
        else:
            central_x = central_value
            
        std_x = bn.nanstd(x)
        new_x = x[np.nonzero((x < central_x + sigma * std_x)
                             * (x > central_x - sigma * std_x))]
        if return_index_list:
            index_list = index_list[
                np.nonzero((x < central_x + sigma * std_x)
                           * (x > central_x - sigma * std_x))]
        
        new_sz = np.size(new_x)
        if new_sz == sz or new_sz <= min_values:
            still_rejection = 0
        else:
            x = new_x
    if not return_index_list:
        return x
    else:
        return x, index_list

def meansigcut2d(np.ndarray[np.float64_t, ndim=2] x, double sigma=3,
                 int min_values=3, int axis=0):
    """Return the sigma cut mean of a 2d array along a given axis.

    :param x: The 2d array
    
    :param sigma: Number of sigma above which values are considered as
      deviant

    :param min_values: Minimum number of values to return

    :param axis: Axis number. Must be 0 or 1.
    """
    cdef int i = 0
    cdef int coaxis
    if axis == 0: coaxis = 1
    else: coaxis = 0
    cdef int n = x.shape[coaxis]
    
    cdef np.ndarray[np.float64_t, ndim=1] line = np.empty(n, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] iline = np.empty(x.shape[axis],
                                                           dtype=float)
    
    for i in range(n):
        if axis == 0:
            iline = x[:,i]
        else:
            iline = x[i,:]
        line[i] = bn.nanmean(sigmacut(iline, 0, False,
                                      sigma, min_values,
                                      return_index_list=False))
    return line
    

def sigmaclip(np.ndarray[np.float64_t, ndim=1] x, double sigma,
              int min_values):
    """Return a distribution after a sigma clipping rejection
    of the too deviant values.

    :param x: The distribution to sigmaclip

    :param sigma: Number of sigma above which values are considered as
      deviant
      
    :param min_values: Minimum number of values to return
    """
    cdef int still_rejection = 1
    cdef double central_x
    cdef int sz, new_sz
    cdef double med, std
    cdef int min_mask = 0
    cdef int max_mask = x.shape[0] - 1
    
    if bn.anynan(x):
        x = x[np.nonzero(~np.isnan(x))]

    # sort array once and for all
    x = np.sort(x)

    # compute first median without min and max
    med = bn.median(x[1:-1])
    std = bn.nanstd(x[1:-1])

    while still_rejection:
        # put min and max at first and last index
        sz = max_mask - min_mask + 1

        while x[min_mask] <= med - sigma * std:
            min_mask += 1
        while x[max_mask] >= med + sigma * std:
            max_mask -= 1

        new_sz = max_mask - min_mask + 1
        
        if new_sz == sz or new_sz <= min_values:
            still_rejection = 0
        else:
            med = bn.median(x[min_mask:max_mask+1])
            std = bn.nanstd(x[min_mask:max_mask+1])
            
    return x[min_mask:max_mask+1]

def master_combine(np.ndarray[np.float64_t, ndim=3] frames, double sigma,
                   int nkeep, int combine_mode, int reject_mode,
                   return_std_frame=False):
    """
    Create a master frame from a set a frames.

    This method has been inspired by the **IRAF** function
    combine.

    :param frames: Frames to be combined.

    :param sigma: Sigma factor for pixel rejection.

    :param nkeep: Minimum number of values to keep before
      combining operation
    
    :param combine_mode: 0, mean ; 1, median.
    
    :param reject_mode: 0, avsigclip ; 1, sigclip ; 2, minmax.

    :param return_std: If True, the std frame is also returned
      (default False).

    .. note:: Rejection operations:

      * **sigclip**: A Sigma Clipping algorithm is applied for
        each pixel. Min and max values are rejected to estimate
        the mean and the standard deviation at each pixel. Then
        all values over (median + sigma * std) or below (median -
        sigma * std) are rejected. Those steps are repeated (this
        time not excluding the extreme values) while no other
        value is rejected or the minimum number of values to keep
        is reached. Work best with at least 10 frames.

      * **avsigclip**: Average Sigma Clipping algorithm is the
        same as Sigma Clipping algorithm but the standard
        deviation at each pixel is estimated using an averaged
        value of the std over the lines. This work best than sigma
        clipping for a small number of frames. This algorithm is a
        little more time consuming than the others. Works best with
        at least 5 frames.

      * **minmax**: Minimum and maximum values at each pixel are
        rejected.

    .. warning:: No rejection operation can be performed with less
      than 3 frames.
    """

    cdef int stop_rejection = 0
    cdef int dimx = frames.shape[0]
    cdef int dimy = frames.shape[1]
    cdef int dimz = frames.shape[2]
    if dimz < 3: raise Exception('There must be more than 2 frames to combine')

    cdef np.ndarray[np.float64_t, ndim=3] framesdiff = np.zeros(
        (dimx, dimy, dimz), dtype=np.float64)
    
    cdef np.ndarray[np.int64_t, ndim=2] argmax2d = np.zeros(
        (dimx, dimy), dtype=np.int64)
    cdef np.ndarray[np.float64_t, ndim=2] max2d = np.zeros(
        (dimx, dimy), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] sqrtmean2d = np.zeros(
        (dimx, dimy), dtype=np.float64)
    cdef np.ndarray[np.uint8_t, ndim=2] rejects2d = np.zeros(
        (dimx, dimy), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=2] new_rejects2d = np.zeros(
        (dimx, dimy), dtype=np.uint8)
    
    
    cdef np.ndarray[np.float64_t, ndim=2] mean2d = np.zeros((dimx, dimy))
    cdef np.ndarray[np.float64_t, ndim=2] std2d = np.zeros((dimx, dimy))
       
    frames = np.sort(frames, axis=2)
    mean2d = bn.nanmean(frames[:,:,1:-1], axis=2)
    std2d = bn.nanstd(frames[:,:,1:-1], axis=2)

    ## Rejection
    # sigclip or avsigclip
    if reject_mode == 0 or reject_mode == 1 :
        while not stop_rejection:
            if reject_mode == 0:
                sqrtmean2d = np.sqrt(np.abs(mean2d))
                sqrtmean2d[np.nonzero(mean2d == 0)] = np.nan
                std2d = (sqrtmean2d.T
                         * bn.nanmean(std2d / sqrtmean2d, axis=1)).T

            framesdiff = np.abs((frames.T - mean2d.T).T)
            max2d = bn.nanmax(framesdiff, axis=2)
            
            # remove all-NaN slices
            nans = np.nonzero(np.isnan(max2d))
            # NaNs in framesdiff are set to 0 to avoid a ValueError
            framesdiff[nans] = 0.
            # All Nan slices are set to a value which will always reject them
            max2d[nans] = std2d[nans] * sigma * 2
            
            argmax2d = bn.nanargmax(framesdiff, axis=2)
            mask2d = np.nonzero(np.logical_and(max2d > std2d * sigma,
                                               rejects2d < dimz - nkeep))
            rejpix = (mask2d[0], mask2d[1], argmax2d[mask2d])
            frames[rejpix] = np.nan
            
            new_rejects2d[mask2d] += 1
            
            if np.all(new_rejects2d == rejects2d):
                stop_rejection = 1

            rejects2d = np.copy(new_rejects2d)
            mean2d = bn.nanmean(frames, axis=2)
            std2d = bn.nanstd(frames, axis=2)
        
    # minmax
    elif reject_mode == 2:
        frames = frames[:,:,1:-1]
        rejects2d.fill(2)
    else: raise Exception('Bad rejection mode. Must be 0, 1 or 2.')

    ## Combination
    if combine_mode == 0:
        result = bn.nanmean(frames, axis=2)
    elif combine_mode == 1:
        result = bn.nanmedian(frames, axis=2)
    else: raise Exception('Bad combining mode. Must be 0 or 1')
    if not return_std_frame:
        return result, rejects2d
    else:
        return result, rejects2d, bn.nanstd(frames, axis=2)


def _robust_format(np.ndarray x):
    """Format data and check if it can be computed by bottleneck module.

    :param x: a numpy ndarray

    :return: (formatted_data, flag, complexflag). If the flag is True,
      data can be computed by the bottleneck module. Else it must be
      computed with Numpy. The complexflag tells if the data is
      complex.
    """
    cdef bool flag = True
    cdef bool complexflag = False
    
    if x.ndim < 1:
        flag = False

    if x.dtype == np.dtype(complex):
        complexflag = True

    if x.dtype.byteorder != '|' or '=':
        x = x.astype(x.dtype.newbyteorder('N'))

    if x.dtype == np.dtype(float) or x.dtype == np.dtype(complex):
        if np.any(np.isinf(x)):
            x[np.nonzero(np.isinf(x))] = np.nan
        
    return x, flag, complexflag
    
def robust_mean(np.ndarray x):
    """Compute robust mean of a numpy ndarray (NaNs are skipped)

    :param x: a numpy ndarray 
    """
    cdef bool flag
    cdef bool complexflag
    cdef complex complexresult
    
    (x, flag, complexflag) = _robust_format(x)

    if flag:
        if complexflag:
            complexresult.real = bn.nanmean(x.real)
            complexresult.imag = bn.nanmean(x.imag)
            return complexresult
        else:
            return bn.nanmean(x)
    else:
        return np.nanmean(x)


def robust_median(np.ndarray x):
    """Compute robust median of a numpy ndarray (NaNs are skipped)

    :param x: a numpy ndarray 
    """
    cdef bool flag
    cdef bool complexflag
    cdef complex complexresult
    
    (x, flag, complexflag) = _robust_format(x)

    if flag:
        if complexflag:
            complexresult.real = bn.nanmedian(x.real)
            complexresult.imag = bn.nanmedian(x.imag)
            return complexresult
        else:
            return bn.nanmedian(x)
    else:
        return np.median(x)

def robust_sum(np.ndarray x):
    """Compute robust sum of a numpy ndarray (NaNs are skipped)

    :param x: a numpy ndarray 
    """
    cdef bool flag
    cdef bool complexflag
    cdef complex complexresult
    
    (x, flag, complexflag) = _robust_format(x)
 
    if flag:
        if complexflag:
            complexresult.real = bn.nansum(x.real)
            complexresult.imag = bn.nansum(x.imag)
            return complexresult
        else:
            return bn.nansum(x)
    else:
        return np.nansum(x)

def robust_std(np.ndarray x):
    """Compute robust std of a numpy ndarray (NaNs are skipped)

    :param x: a numpy ndarray 
    """
    cdef bool flag
    cdef bool complexflag
    cdef complex complexresult
    
    (x, flag, complexflag) = _robust_format(x)

    if flag:
        if complexflag:
            complexresult.real = bn.nanstd(x.real)
            complexresult.imag = bn.nanstd(x.imag)
            return complexresult
        else:
            return bn.nanstd(x)
    else:
        return np.nanstd(x)

def robust_average(np.ndarray x,
                   np.ndarray w):
    """Compute robust average of a numpy ndarray (NaNs are skipped)

    :param x: a numpy ndarray
    :param w: a numpy ndarray of weigths

    .. note:: To get a weighted average the MEAN of the weights must
      be equal to 1.
    """
    cdef bool flagx
    cdef bool flagw
    cdef bool complexflagx
    cdef bool complexflagw
    cdef complex complexresult
    cdef int i
    
    (x, flagx, complexflagx) = _robust_format(x)
    (w, flagw, complexflagw) = _robust_format(w)

    if x.ndim == w.ndim:
        for i in range(x.ndim):
            if x.shape[i] != w.shape[i]:
                raise Exception('Array and weights must have same shape')
    else:
        raise Exception('Array and weights must have same number of dimensions')
        
    if flagx and flagw:
        if complexflagx or complexflagw:
            complexresult.real = bn.nanmean(x.real * w.real)
            complexresult.imag = bn.nanmean(x.imag * w.imag)
            return complexresult
        else:
            return bn.nanmean(x * w)
    else:
        return np.nanmean(x * w)

def gaussian1d(np.ndarray[np.float64_t, ndim=1] x,
               double h, double a, double dx, double fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: 1D array of float64 giving the positions where the
      gaussian is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    cdef double w = fwhm / (2. * sqrt(2. * log(2.)))
    return  h + a * np.exp(-(x - dx)**2. / (2. * w**2.))

def sinc1d(np.ndarray[np.float64_t, ndim=1] x,
           double h, double a, double dx, double fwhm):
    """Return a 1D sinc given a set of parameters.

    :param x: 1D array of float64 giving the positions where the
      sinc is evaluated
    
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    cdef np.ndarray[np.float64_t, ndim=1] X = ((x-dx)/(fwhm/1.20671))
    return h + a * np.sinc(X)


## def sincgauss1d(np.ndarray[np.float64_t, ndim=1] x,
##                 double h, double a, double dx, double fwhm, double sigma):
##     """Return a 1D sinc convoluted with a gaussian of parameter sigma.

##     If sigma == 0 returns a pure sinc.

##     :param x: 1D array of float64 giving the positions where the
##       sinc is evaluated
    
##     :param h: Height
##     :param a: Amplitude
##     :param dx: Position of the center
##     :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
##     :param sigma: Sigma of the gaussian.
##     """
##     # when sigma/fwhm is too high or too low, a pure sinc or gaussian
##     # is returned (avoid overflow)
##     if abs(sigma / fwhm) < 1e-2:
##         return sinc1d(x, h, a, dx, fwhm)
##     if abs(sigma / fwhm) > 1e2:
##         return gaussian1d(x, h, a, dx, fwhm)
    
##     sigma = abs(sigma)
    
##     fwhm /= M_PI * 1.20671
##     cdef double complex e = exp(-sigma**2. / 2.) / (sqrt(2.) * sigma * 1j)
##     cdef np.ndarray[np.complex128_t, ndim=1] dawson1, dawson2
##     dawson1 = (scipy.special.dawsn((1j * sigma**2 - (x - dx) / fwhm)
##                                    /(sqrt(2.) * sigma))
##                * np.exp(-1j * (x - dx) / fwhm))
##     dawson2 = (scipy.special.dawsn((-1j * sigma**2 - (x - dx) / fwhm)
##                                    / (sqrt(2.) * sigma))
##                * np.exp(1j *(x - dx) / fwhm))
    
##     return (h + a * e * (dawson1 - dawson2)).real

def interf_mean_energy(np.ndarray interf):
    """Return the mean energy of an interferogram by step.

    :param interf: an interferogram

    .. warning:: The mean of the interferogram is substracted to
      compute only the modulation energy. This is the modulation
      energy which must be conserved in the resulting spectrum. Note
      that the interferogram transformation function (see
      :py:meth:`utils.transform_interferogram`) remove the mean of the
      interferogram before computing its FFT.

    .. note:: NaNs are counted as zeros.
    """
    cdef double energy_sum

    if not np.iscomplexobj(interf):
        energy_sum = bn.nansum((interf - bn.nanmean(interf))**2.)
    else:
        energy_sum = bn.nansum(
            (interf.real - bn.nanmean(interf.real))**2.
            + (interf.imag - bn.nanmean(interf.imag))**2.)
    
    return sqrt(energy_sum / <float> np.size(interf))

def spectrum_mean_energy(np.ndarray spectrum):
    """Return the mean energy of a spectrum by channel.

    :param spectrum: a 1D spectrum

    .. note:: NaNs are counted as zeros.
    """
    cdef double energy_sum

    if not np.iscomplexobj(spectrum):
        energy_sum = bn.nansum(spectrum**2.)
    else:
        energy_sum = bn.nansum(spectrum.real**2. + spectrum.imag**2.)
        
    return sqrt(energy_sum) / <float> np.size(spectrum)

def fft_filter(np.ndarray[np.float64_t, ndim=1] a,
                double cutoff, double width, bool lowpass):
    """
    Simple lowpass or highpass FFT filter (high pass or low pass)

    Filter shape is a gaussian.
    
    :param a: a 1D float64 vector

    :param cutoff_coeff: Coefficient defining the position of the cutoff
      frequency (Cutoff frequency = cutoff_coeff * vector length)

    :param width_coeff: Coefficient defining the width of
      the smoothed part of the filter (width = width_coeff * vector
      length) 

    :param lowpass: If True filter will be 'low_pass' and 'high_pass'
      if False.
    """
    if cutoff > 1. or cutoff < 0.:
        raise ValueError('cutoff must be between 0. and 1.')
    if width > 1. or width < 0.:
        raise ValueError('width must be between 0. and 1.')

    cdef int n = a.shape[0]
    cdef int fn = <int> floor(<double> n/2.)
    cdef np.ndarray[np.float64_t, ndim=1] hwindow = np.zeros(
        fn, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] window = np.zeros(
        n, dtype=np.float64)
    cdef int icut = max(0, <int> floor(<double> n * cutoff))
    cdef double wlen = <double> fn * width
    cdef int wsize = <int> ceil(wlen) * 2
    cdef np.ndarray[np.float64_t, ndim=1] w = np.zeros(
        wsize, dtype=np.float64)
    cdef double dx
    cdef int minx, maxx
    cdef double median_a
    
    # half-window design
    dx = <double> fn * cutoff - <double> icut
    w = gaussian1d(np.arange(wsize, dtype=np.float64),
                   0., 1., dx, wlen / 1.5)
    w[0] = 1.
    
    minx = icut - <int> floor(wlen / 2.)
    maxx = minx + w.shape[0]
    if minx < 0:
        w = w[-minx:]
        minx = 0
    if maxx >= fn:
        w = w[:fn-maxx]
        maxx = fn
    hwindow[minx:maxx] = w
    if minx > 0:
        hwindow[:minx+1] = 1.

    if not lowpass:
        hwindow = - hwindow + 1.

    if not n%2:
        window = np.hstack((hwindow, hwindow[::-1]))
    else:
        window = np.hstack((hwindow, hwindow[-1], hwindow[::-1]))

    if bn.anynan(a):
        median_a = robust_median(a)
        a[np.nonzero(np.isnan(a))] = median_a

    # FFT and IFFT
    return (np.fft.ifft(np.fft.fft(a) * window)).real


def low_pass_image_filter(np.ndarray[np.float64_t, ndim=2] im, int deg):
    """Low pass image filter by convolution with a gaussian kernel.

    :param im: Image to filter
    
    :param deg: Kernel degree
    """
    cdef np.ndarray[np.int8_t, ndim=2] real_nans = (
        np.isnan(im).astype(np.int8))
    cdef np.ndarray[np.int8_t, ndim=2] new_nans = np.zeros_like(real_nans)
    
    cdef np.ndarray[np.float64_t, ndim=2] final_im = np.zeros_like(im)
    cdef np.ndarray[np.float64_t, ndim=2] kernel = gaussian_kernel(deg)
    cdef np.ndarray[np.float64_t, ndim=2] kernel_mod = np.zeros_like(kernel)
    cdef np.ndarray[np.float64_t, ndim=2] box = np.zeros_like(kernel)
    
    cdef int inan
    cdef int ix
    cdef int iy

    if np.any(np.isinf(im)):
        im[np.nonzero(np.isinf(im))] = np.nan

    final_im = scipy.ndimage.filters.convolve(im, kernel, 
                                              mode='nearest')
    
    new_nans = (np.isnan(final_im).astype(np.int8)
                +  np.isinf(final_im).astype(np.int8) - real_nans)
    
    nans = np.nonzero(new_nans > 0)
    
    for inan in range(len(nans[0])):
        ix = nans[0][inan]
        iy = nans[1][inan]
        if (ix >= deg and iy >= deg and ix < im.shape[0] - deg
            and  iy < im.shape[1] - deg):
            box = np.copy(im[ix-deg:ix+deg+1, iy-deg:iy+deg+1])
            if not np.isnan(bn.nansum(box)):
                kernel_mod = kernel * (~np.isnan(box))
                kernel_mod = (kernel_mod / bn.nansum(kernel_mod)
                              * bn.nansum(kernel))
                box[np.nonzero(np.isnan(box))] = 0.
                final_im[ix,iy] = bn.nansum(box * kernel_mod)
                
    return np.copy(final_im)
    
def fast_gaussian_kernel(int deg):
    """Return a fast gaussian kernel.

    The value of each pixel is just the value of the gaussian at the
    center of the pixel.

    The degree gives the size of the kernel's side : size = 2 * deg + 1

    :param deg: The degree of the kernel. Must be an integer >= 0.
    """
    cdef double ddeg = <double> deg
    cdef int sz = 2 * deg + 1
    cdef double fwhm = ddeg/2. * (2. * sqrt(2. * log(2.)))
    cdef np.ndarray[np.float64_t, ndim=2] kernel = np.zeros(
        (sz, sz), dtype=np.float64)
    
    if deg < 0: raise ValueError('deg must be >= 0')
    if deg > 0:
        kernel = gaussian_array2d(0., 1., ddeg, ddeg, fwhm, sz, sz)
        kernel / bn.nansum(kernel)
        return kernel
    else:
        kernel.fill(1.)
        return kernel

    
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_kernel(double deg):
    """Return a gaussian kernel.

    The value of each pixel is the integral of the gaussian over the
    whole pixel because the shape of the gaussian is not linear at
    all. The estimation is done by subdividing each pixel in 9
    sub-pixels.

    The degree gives the size of the kernel's side : size = 2 * deg + 1

    :param deg: The degree of the kernel. Must be an integer >= 0.
    """
    cdef int ideg = <int> floor(deg)
    cdef int sz = 2 * ideg + 1
    cdef int PRECISION_COEFF = 9 # must be odd and >= 3
    cdef int large_sz = PRECISION_COEFF * sz
    cdef np.ndarray[np.float64_t, ndim=2] large_kernel = np.zeros(
        (large_sz, large_sz), dtype=np.float64)
    
    cdef double fwhm = deg/2. * (2. * sqrt(2. * log(2.)))
    cdef np.ndarray[np.float64_t, ndim=2] kernel = np.zeros(
        (sz, sz), dtype=float)
    cdef int ii, ij, i, j

    
    if deg < 0: raise Exception('deg must be >= 0')

    large_kernel = gaussian_array2d(0., 1.,
                                    <double> large_sz / 2. - 0.5,
                                    <double> large_sz / 2. - 0.5,
                                    fwhm * <double> PRECISION_COEFF,
                                    large_sz, large_sz)

    with nogil:
        for ii in range(large_sz):
            for ij in range(large_sz):
                i = <int> (<double> ii / <double> PRECISION_COEFF)
                j = <int> (<double> ij / <double> PRECISION_COEFF)
                kernel[i,j] += large_kernel[ii,ij]
            
    return kernel / bn.nansum(kernel)


def get_box_coords(int ix, int iy, int box_size,
                   int x_lim_min, int x_lim_max,
                   int y_lim_min, int y_lim_max):
    """Return the coordinates of a box given the center of the box,
    its size and the limits of the range along x and y axes.

    :param ix: center of the box along x axis
    :param iy: center of the box along y axis
    
    :param box_size: Size of the box. The final size of the box will
      generally be the same if box_size is odd. Note that the final
      size of the box cannot be guaranteed.

    :param x_lim_min: Minimum limit of the range along x.
    :param x_lim_max: Maximum limit of the range along x.
    :param y_lim_min: Minimum limit of the range along y.
    :param y_lim_max: Maximum limit of the range along y.

    :return: x_min, x_max, y_min, y_max
    """
    cdef int x_min, x_max, y_min, y_max

    x_min = <int> (ix - <int> (box_size/2.))
    x_max = <int> (ix + <int> (box_size/2.)) + 1
    y_min = <int> (iy - <int> (box_size/2.))
    y_max = <int> (iy + <int> (box_size/2.)) + 1
    if x_min < x_lim_min: x_min = x_lim_min
    if y_min < y_lim_min: y_min = y_lim_min
    if x_max >= x_lim_max: x_max = x_lim_max
    if y_max >= y_lim_max: y_max = y_lim_max
    if x_max - x_min < 1:
        x_max = x_min + 1
    if y_max - y_min < 1:
        y_max = y_min + 1
        
    return x_min, x_max, y_min, y_max


def point_inside_polygon(double x, double y, list poly):
    """ Determine if a point is inside a given polygon or not
    Polygon is a list of (x,y) pairs.

    This function has been taken from
    http://www.ariel.com.au/a/python-point-int-poly.html and cythonized.
    """
    cdef int n = len(poly)
    cdef bool inside = False
    cdef int i
    cdef double p1x, p1y, p2x, p2y
    cdef double xinters
    
    plx, ply = poly[0]
    
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def multi_fit_stars(np.ndarray[np.float64_t, ndim=2] frame,
                    np.ndarray[np.float64_t, ndim=2] pos,
                    int box_size,
                    double height_guess=np.nan,
                    np.ndarray[np.float64_t, ndim=1] fwhm_guess=np.array(
                        [np.nan], dtype=float),
                    bool cov_height=False,
                    bool cov_pos=True,
                    bool cov_fwhm=True,
                    bool fix_height=False,
                    bool fix_pos=False,
                    bool fix_fwhm=False,
                    double fit_tol=1e-3,
                    double ron=np.nan,
                    double dcl=np.nan,
                    bool enable_zoom=False,
                    bool enable_rotation=False,
                    bool estimate_local_noise=True,
                    double saturation=0,
                    sip=None):
    """Fit multiple stars at the same time.

    Useful if the relative positions of the stars are well known. In
    this case the pattern of position can be shifted, zoomed and
    rotated in order to be adjusted to the stars in the frame.

    Other covarying parameters can be the height and the FWHM.

    :param frame: Frame
    
    :param pos: array of stars positions of shape [[x1, y1], [x2, y2] ...]
    
    :param box_size: Size of the box around a single star
    
    :param height_guess: (Optional) Initial guess on the height
      parameter (default NaN).
      
    :param fwhm_guess: (Optional) Initial guess on the FWHM parameter
      must a numpy.ndarray.

    :param cov_height: (Optional) If True, height is considered to be
      the same for all the stars. It is then a covarying parameter
      (default False).

    :param cov_pos: (Optional) If True, shift along x and y is
      considered as the same for all the stars. dx and dy are thus 2
      covarying parameters (default True).

    :param cov_fwhm: (Optional) If True, FWHM is considered to be the
      same for all the stars. It is then a covarying parameter
      (default True).

    :param fix_height: (Optional) If True, height is fixed to its
      guess (default False).

    :param fix_pos: (Optional) If True, x and y are fixed to the position
      guess given by pos (default False).

    :param fix_fwhm: (Optional) If True, FWHM is fixed to its guess
      (default False).

    :param fit_tol: (Optional) Tolerance on the fit (default 1e-3).
      
    :param ron: (Optional) Readout noise. If given and if
      estimate_local_noise is set to False the readout noise is fixed
      to the given value. If not given the ron is guessed from the
      background around the stars (default NaN).

    :param dcl: (Optional) Dark current level. If given and if
      estimate_local_noise is set to False the dark current level is
      fixed to the given value. If not given the dark current level is
      fixed to 0. (default NaN)

    :param enable_zoom: (Optional) If True the position pattern can be
      zoomed (default False).

    :param enable_rotation: (Optional) If True the position pattern
      can be rotated (default False).

    :param estimate_local_noise: (Optional) If True, the level of
      noise is computed from the background pixels around the
      stars. ron and dcl parameters are thus not used (default True).

    :param saturation: (Optional) If not 0, all pixels above the
      saturation level are removed from the fit (default 0).

    :param sip: (Optional) A pywcs.WCS instance containing SIP
      distorsion correction (default None).
    """    
    def params_arrays2vect(np.ndarray[np.float64_t, ndim=2] stars_p,
                           np.ndarray[np.float64_t, ndim=1] stars_p_mask,
                           np.ndarray[np.float64_t, ndim=1] cov_p,
                           np.ndarray[np.float64_t, ndim=1] cov_p_mask):

        cdef int stars_free_p_nb = np.sum(stars_p_mask)
        cdef int cov_free_p_nb = np.sum(cov_p_mask)
        cdef np.ndarray[np.float64_t, ndim=1] free_p = np.zeros(
            stars_p.shape[0] * stars_free_p_nb
            + cov_free_p_nb, dtype=float)
        cdef np.ndarray[np.float64_t, ndim=1] fixed_p = np.zeros(
            np.size(stars_p) + np.size(cov_p) - np.size(free_p),
            dtype=float)
        cdef int ip, i, j

        i = 0
        j = 0
        
        for ip in range(stars_p.shape[1]):
            if stars_p_mask[ip] == 1:
                free_p[i*stars_p.shape[0]:
                       (i+1)*stars_p.shape[0]] = stars_p[:,ip]
                i += 1
            else:        
                fixed_p[j*stars_p.shape[0]:
                       (j+1)*stars_p.shape[0]] = stars_p[:,ip]
                j += 1

        if bn.nansum(cov_p_mask > 0):
            free_p[-cov_free_p_nb:] = cov_p[np.nonzero(cov_p_mask)]
        if bn.nansum(cov_p_mask > 0) != np.size(cov_p_mask):
            fixed_p[-(np.size(cov_p)-cov_free_p_nb):] = cov_p[
                np.nonzero(cov_p_mask == 0)]
            
       
        return free_p, fixed_p

    def params_vect2arrays(np.ndarray[np.float64_t, ndim=1] free_p,
                           np.ndarray[np.float64_t, ndim=1] fixed_p,
                           np.ndarray[np.float64_t, ndim=1] stars_p_mask,
                           np.ndarray[np.float64_t, ndim=1] cov_p_mask,
                           int star_nb):

        cdef int stars_free_p_nb = bn.nansum(stars_p_mask)
        cdef int cov_free_p_nb = bn.nansum(cov_p_mask)
        cdef np.ndarray[np.float64_t, ndim=2] stars_p = np.zeros(
            (star_nb, np.size(stars_p_mask)), dtype=float)
        cdef np.ndarray[np.float64_t, ndim=1] cov_p = np.zeros(
            np.size(cov_p_mask), dtype=float)
        cdef int ip, i, j

        i = 0
        j = 0
        for ip in range(np.size(stars_p_mask)):
            if stars_p_mask[ip] == 1:
                stars_p[:,ip] = free_p[i*stars_p.shape[0]:
                                       (i+1)*stars_p.shape[0]]
                i += 1
            else:
                stars_p[:,ip] =  fixed_p[j*stars_p.shape[0]:
                                         (j+1)*stars_p.shape[0]]
                j += 1
                
        if bn.nansum(cov_p_mask > 0):
            cov_p[np.nonzero(cov_p_mask)] = free_p[-cov_free_p_nb:]
            cov_p[np.nonzero(cov_p_mask == 0)] = fixed_p[
                -(np.size(cov_p)-cov_free_p_nb):]
            
        return stars_p, cov_p

    def sigma(np.ndarray[np.float64_t, ndim=2] data,
              double noise, double dcl):
        """guess sigma as sqrt(photon noise + readout noise^2 + dark
        current level)"""
        return np.sqrt(np.abs(data) + (noise)**2. + dcl)

    def model_diff(int dimx, int dimy, np.ndarray[np.float64_t, ndim=2] params,
                   int box_size, np.ndarray[np.float64_t, ndim=2] frame,
                   np.ndarray[np.float64_t, ndim=1] noise, double dcl,
                   double saturation, transpose=False, normalize=False):
        """params is an array (star_nb, 5) with the 5 parameters of
        each star in this order : height, amplitude, posx, posy, fwhm
        """
        cdef int int_posx, int_posy, istar
        cdef np.ndarray[np.float64_t, ndim=2] star = np.zeros(
            (box_size, box_size), dtype=float)
        cdef np.ndarray[np.float64_t, ndim=2] data = np.zeros_like(
            star, dtype=float)
        cdef np.ndarray[np.float64_t, ndim=2] res
        cdef double dx, dy
        cdef int x_min, x_max, y_min, y_max
        cdef int hsz

        if not transpose:
            res = np.zeros(
                (box_size * params.shape[0], box_size), dtype=float)
        else:
            res = np.zeros(
                (box_size, box_size * params.shape[0]), dtype=float)

        res.fill(np.nan)
        
        for istar in range(params.shape[0]):
            int_posx = <int> params[istar,2]
            int_posy = <int> params[istar,3]
            dx = (params[istar,2] - <double> int_posx
                  + (<double> box_size / 2.) - 0.5)
            dy = (params[istar,3] - <double> int_posy
                  + (<double> box_size / 2.) - 0.5)

            # star
            star = gaussian_array2d(0.,
                                    params[istar,1],
                                    dx, dy,
                                    params[istar,4],
                                    box_size, box_size)
            # background
            #xx, xy = np.mgrid[0 - dx: star.shape[0] - dx,
            #                  0 - dy: star.shape[1] - dy]
            #star += params[istar,0] + params[istar,5] * xy
            star += params[istar,0]

            x_min, x_max, y_min, y_max = get_box_coords(
                int_posx, int_posy, box_size,
                0, dimx, 0, dimy)
            hsz = <int> (<double> box_size / 2.)
            if int_posx - x_min != hsz:
                star = star[hsz - (int_posx - x_min):,:]
            if x_max - int_posx != hsz + 1:
                star = star[:-(hsz + 1 - (x_max - int_posx)), :]
            if int_posy - y_min != hsz:
                star = star[:, hsz - (int_posy - y_min):]
            if y_max - int_posy != hsz + 1:
                star = star[:, :-(hsz + 1 - (y_max - int_posy))]

            if star.shape[0] > 1 and star.shape[1] > 1:
                data = frame[<int> x_min:<int> x_max, <int> y_min:<int> y_max]
                star = (star - data) #/ sigma(data, noise[istar], dcl)
                if saturation > 0.:
                    star[np.nonzero(data >= saturation)] = np.nan
            
                if normalize:
                    star /= np.nanmax(star)
                 
                if not transpose:
                    res[istar * box_size:
                        istar * box_size + star.shape[0],
                        0: star.shape[1]] = star
                else:
                    res[0: star.shape[0],
                        istar * box_size:
                        istar * box_size + star.shape[1]] = star
        
        return res
    
    def diff(np.ndarray[np.float64_t, ndim=1] free_p,
             np.ndarray[np.float64_t, ndim=1] fixed_p,
             np.ndarray[np.float64_t, ndim=1] stars_p_mask,
             np.ndarray[np.float64_t, ndim=1] cov_p_mask,
             np.ndarray[np.float64_t, ndim=2] frame,
             int box_size, int star_nb,
             np.ndarray[np.float64_t, ndim=1] noise, double dcl,
             double saturation, sip):

        cdef np.ndarray[np.float64_t, ndim=2] params = np.zeros(
            (star_nb, np.size(stars_p_mask)), dtype=float)
        cdef np.ndarray[np.float64_t, ndim=2] stars_p = np.zeros_like(params)
        cdef np.ndarray[np.float64_t, ndim=1] cov_p = np.zeros_like(cov_p_mask)
        cdef np.ndarray[np.float64_t, ndim=2] res = np.zeros(
            (box_size * star_nb, box_size), dtype=float)
        cdef double rcx, rcy
        cdef int istar

        rcx = <double> frame.shape[0] / 2.
        rcy = <double> frame.shape[1] / 2.
        
        stars_p, cov_p = params_vect2arrays(
            free_p, fixed_p, stars_p_mask, cov_p_mask, star_nb)
        
        params = np.copy(stars_p)
        params[:,0] += cov_p[0] # COV HEIGHT

        # COV FWHM: here FWHM covariance is best interpreted as the
        # convolution of the original gaussian with another gaussian
        # psf. In this case, the fwhm of two convoluted gaussian is
        # the quadratic sum of the psf's.
        #params[:,4] += cov_p[3] # COV FWHM
        params[:,4] = np.sqrt(params[:,4]**2. + cov_p[3]**2.) 

        # dx, dy & zoom & rotation
        for istar in range(stars_p.shape[0]):
            params[istar,2],  params[istar,3] = transform_A_to_B(
                params[istar,2], params[istar,3], -cov_p[1], -cov_p[2],
                cov_p[6], 0., 0., rcx, rcy, cov_p[4], cov_p[5])

        if sip is not None:
            params[:,2:4] = sip_im2pix(params[:,2:4], sip)
        
        res = model_diff(frame.shape[0], frame.shape[1],
                         params, box_size, frame, noise, dcl,
                         saturation)
        
        return res[np.nonzero(~np.isnan(res))]
    
    ## filter pos list
    cdef np.ndarray[np.uint8_t, ndim=1] pos_mask = np.zeros(
        pos.shape[0], dtype=np.uint8)
    cdef int istar
    for istar in range(pos.shape[0]):
        if (pos[istar,0] > 0. and pos[istar,0] < <double> frame.shape[0]
            and pos[istar,1] > 0. and pos[istar,1] < <double> frame.shape[1]):
            pos_mask[istar] = 1
            
    cdef np.ndarray[np.float64_t, ndim=2] new_pos = np.zeros(
        (np.sum(pos_mask), 2), dtype=float)

    new_pos = pos[np.nonzero(pos_mask)]


    # stop here if no star positions are in the frame
    if np.size(new_pos) == 0:
        return []

    cdef int PARAMS_NB = 5
    cdef int star_nb = new_pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] stars_p = np.zeros(
        (star_nb, PARAMS_NB), dtype=float) # [STARS_NB * (H,A,DX,DY,FWHM)]

    cdef np.ndarray[np.float64_t, ndim=1] fwhm_pix = np.zeros(
        (star_nb), dtype=float)

    cdef np.ndarray[np.float64_t, ndim=2] stars_err = np.zeros_like(
        stars_p, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] new_stars_p = np.zeros(
        (pos.shape[0], stars_p.shape[1]), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] new_stars_err = np.zeros_like(
        new_stars_p, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] stars_p_mask = np.ones(
        PARAMS_NB, dtype=float)
    
    cdef np.ndarray[np.float64_t, ndim=2] test_p = np.zeros_like(
        stars_p, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] cov_p = np.zeros(
        7, dtype=float) # COV_P: HEIGHT, POSX, POSY, FWHM, ZOOMX, ZOOMY, ROT
    cdef np.ndarray[np.float64_t, ndim=1] cov_err = np.zeros_like(
        cov_p, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] cov_p_mask = np.zeros_like(
        cov_p, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] frame_mask = np.zeros_like(
        frame, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] amp_guess = np.ones(
        (star_nb), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] free_p
    cdef np.ndarray[np.float64_t, ndim=1] fixed_p
    cdef np.ndarray[np.float64_t, ndim=1] test_x = np.zeros(
        box_size, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] test_y = np.zeros(
        box_size, dtype=float)
    cdef int x_min, x_max, y_min, y_max
    cdef double rcx, rcy
    cdef np.ndarray[np.float64_t, ndim=1] noise_guess = np.zeros(
        (star_nb), dtype=float)
    cdef np.ndarray[np.float64_t, ndim=2] cov_matrix, box
    cdef double dx_guess, dy_guess, dx_guess_arg, dy_guess_arg
    cdef added_height = 0.
    cdef frame_min = 0.

    cdef double FWHM_SKY_COEFF = 1.5
    cdef int SUB_DIV = 10

    # no fit on pure NaN frame
    if np.all(np.isnan(frame)):
        return []

    # avoid negative values by adding a height level to the frame
    frame_min = np.nanmin(frame)
    if frame_min < 0.:
        added_height = -frame_min
        frame += added_height
    
    # box size must be odd
    if box_size % 2 == 0: box_size += 1

    # fwhm guess
    if np.any(np.isnan(fwhm_guess)):
        fwhm_guess[np.isnan(fwhm_guess)] = <double> box_size / 3.

    # initial parameters    
    stars_p[:,4] = fwhm_guess
    stars_p[:,2:4] = new_pos
    
    # precise determination of the initial shift from the marginal
    # distribution of the psf [e.g. Howell 2006]
    test_p = np.copy(stars_p)
    test_p[:,1] = 0.
    test_x = np.abs(bn.nansum(model_diff(frame.shape[0], frame.shape[1],
                                         test_p, box_size, frame,
                                         noise_guess * 0.,
                                         0., saturation, transpose=True,
                                         normalize=True), axis=1))
    test_y = np.abs(bn.nansum(model_diff(frame.shape[0], frame.shape[1],
                                         test_p, box_size, frame,
                                         noise_guess * 0.,
                                         0., saturation, transpose=False,
                                         normalize=True), axis=0))
    test_x = test_x - np.min(test_x)
    test_y = test_y - np.min(test_y)
    
    dx_guess_arg = <double> np.argmax(test_x) - <double> box_size / 2.
    dy_guess_arg = <double> np.argmax(test_y) - <double> box_size / 2.
    
    test_x *= gaussian1d(np.arange(test_x.shape[0]).astype(float),
                         0., 1., <double> test_x.shape[0] / 2. - 0.5,
                         <double> test_x.shape[0] / 2.)
    test_y *= gaussian1d(np.arange(test_y.shape[0]).astype(float),
                         0., 1., <double> test_y.shape[0] / 2. - 0.5,
                         <double> test_y.shape[0] / 2.)

    nzi = np.nonzero(test_x > 0.)
    nzj = np.nonzero(test_y > 0.)

    dx_guess = (np.sum((test_x[nzi]) * np.arange(box_size)[nzi])
                / np.sum(test_x[nzi])
                - (<double> box_size / 2. - 0.))
    dy_guess = (np.sum((test_y[nzj]) * np.arange(box_size)[nzj])
                /np.sum(test_y[nzj])
                - (<double> box_size / 2. - 0.))

    # far from the center the arg-based guess is generally better
    if (abs(dx_guess) > <double> box_size / 4.
        or np.isnan(dx_guess)): dx_guess = dx_guess_arg
        
    if (abs(dy_guess) > <double> box_size / 4.
        or np.isnan(dy_guess)): dy_guess = dy_guess_arg
        
    if cov_pos:
        cov_p[1] = dx_guess
        cov_p[2] = dy_guess
    else:
        stars_p[:,2] += dx_guess
        stars_p[:,3] += dy_guess

    # background and noise guess
    for istar in range(star_nb):
        x_min, x_max, y_min, y_max = get_box_coords(
            stars_p[istar,2] + cov_p[1], stars_p[istar,3] + cov_p[2], box_size,
            0, frame.shape[0], 0, frame.shape[1])
        frame_mask[x_min:x_max, y_min:y_max] = 1
        box = frame[x_min:x_max, y_min:y_max]
        # amplitude guess
        # the 3 most intense pixels are skipped to determine the max
        # in the box to avoid errors due to a cosmic ray.
        amp_guess[istar] = bn.nanmax(
            (np.sort(box.flatten()))[:-3])
        
        # define 'sky pixels'
        S_sky = surface_value(
            box.shape[0], box.shape[1],
            stars_p[istar,2] + cov_p[1] - <double> x_min,
            stars_p[istar,3] + cov_p[2] - <double> y_min,
            FWHM_SKY_COEFF * np.nanmedian(
                fwhm_guess),
            np.max([box.shape[0], box.shape[1]]), SUB_DIV)
       
        sky_pixels = box * S_sky
        sky_pixels = np.sort(sky_pixels[np.nonzero(sky_pixels)])[1:-1]
        # guess background
        stars_p[istar,0] = bn.nanmedian(sky_pixels)
        if np.isnan(stars_p[istar,0]): stars_p[istar,0] = 0.
        
        # guess noise
        noise_guess[istar] = bn.nanstd(sky_pixels) #- sqrt(stars_p[istar,0])

    if not np.isnan(height_guess):
        stars_p[:,0] = height_guess

    # height guess
    amp_guess[np.isnan(amp_guess)] = 1.
    stars_p[:,1] = amp_guess - stars_p[:,0]

    if bn.anynan(stars_p): return []
    
    # guess ron and dcl
    if not np.isnan(ron) and not estimate_local_noise:
        noise_guess.fill(ron)
        
    if np.isnan(dcl) or estimate_local_noise:
        dcl = 0.

    # define masks and init cov_p
    cov_p_mask.fill(0)
    stars_p_mask.fill(1)
    cov_p[3] = 1. # cov fwhm starts at 1 for a quadratic sum
    cov_p[4] = 1.
    cov_p[5] = 1.
    cov_p[6] = 0.
    
    if cov_height and not fix_height:
        cov_p_mask[0] = 1
        stars_p_mask[0] = 0
    if cov_pos and not fix_pos:
        cov_p_mask[1:3] = 1
        stars_p_mask[2:4] = 0
    if cov_fwhm and not fix_fwhm:
        cov_p_mask[3] = 1
        stars_p_mask[4] = 0
    if fix_height:  stars_p_mask[0] = 0
    if fix_pos: stars_p_mask[2:4] = 0
    if fix_fwhm: stars_p_mask[4] = 0
    if enable_zoom: cov_p_mask[4:6] = 1
    if enable_rotation: cov_p_mask[6] = 1
        
    free_p, fixed_p = params_arrays2vect(stars_p, stars_p_mask,
                                         cov_p, cov_p_mask)
    ### FIT ###
    try:
        fit = scipy.optimize.leastsq(diff, free_p,
                                     args=(fixed_p, stars_p_mask, cov_p_mask,
                                           np.copy(frame), box_size, star_nb,
                                           noise_guess, dcl, saturation,
                                           sip),
                                     maxfev=500, full_output=True,
                                     xtol=fit_tol)
    except Exception, e:
        print 'Exception raised during least square fit of cutils.multi_fit_stars:', e
        fit = [5]

    ### CHECK FIT RESULTS ###
    if fit[-1] <= 4:
        returned_data = dict()
        last_diff = fit[2]['fvec']
        stars_p, cov_p = params_vect2arrays(
            fit[0], fixed_p, stars_p_mask, cov_p_mask, star_nb)
       
        # compute reduced chi square
        chisq = np.sum(last_diff**2.)
        red_chisq = chisq / (np.sum(frame_mask) - np.size(free_p))
        returned_data['reduced-chi-square'] = red_chisq
        returned_data['chi-square'] = chisq
        returned_data['cov_angle'] = cov_p[6]
        returned_data['cov_zx'] = cov_p[4]
        returned_data['cov_zy'] = cov_p[5]
        returned_data['cov_dx'] = cov_p[1]
        returned_data['cov_dy'] = cov_p[2]
        
        # cov height and fwhm
        stars_p[:,0] += cov_p[0] - added_height # HEIGHT
        stars_p[:,4] = np.sqrt(stars_p[:,4]**2. + cov_p[3]**2.) # FWHM
        # compute least square fit errors and add cov dx, dy and zoom
        rcx = <double> frame.shape[0] / 2.
        rcy = <double> frame.shape[1] / 2.
        cov_matrix = fit[1]
        
        if cov_matrix is None: # no covariance : no error estimation
            stars_err.fill(np.nan)
            # compute transformed postitions
            for istar in range(stars_p.shape[0]):
                stars_p[istar,2], stars_p[istar,3] = transform_A_to_B(
                    stars_p[istar,2], stars_p[istar,3], -cov_p[1], -cov_p[2],
                    cov_p[6], 0., 0., rcx, rcy, cov_p[4], cov_p[5])
            if sip is not None:
                stars_p[:,2:4] = sip_im2pix(stars_p[:,2:4], sip)

            
        else:
            cov_matrix *= returned_data['reduced-chi-square']
            cov_diag = np.sqrt(np.abs(
                np.array([cov_matrix[i,i]
                          for i in range(cov_matrix.shape[0])])))

            fixed_p.fill(0.)
            stars_err, cov_err = params_vect2arrays(
                cov_diag, fixed_p, stars_p_mask, cov_p_mask, star_nb)

            stars_err[:,0] += cov_err[0] # HEIGHT
            stars_err[:,4] += cov_err[3] # FWHM

            # compute transformed positions + error
            for istar in range(stars_p.shape[0]):
                (stars_p[istar,2], stars_p[istar,3],
                 stars_err[istar,2], stars_err[istar,3]) = transform_A_to_B(
                    stars_p[istar,2], stars_p[istar,3], -cov_p[1], -cov_p[2],
                    cov_p[6], 0., 0., rcx, rcy, cov_p[4], cov_p[5],
                    x1_err=stars_err[istar,2],
                    y1_err=stars_err[istar,3],
                    dx_err=cov_err[1],
                    dy_err=cov_err[2],
                    zx_err=cov_err[4],
                    zy_err=cov_err[5],
                    dr_err=cov_err[6],
                    return_err=True)
            if sip is not None:
                stars_p[:,2:4] = sip_im2pix(stars_p[:,2:4], sip)
            
        # put nan in place of filtered stars
        new_stars_p.fill(np.nan)
        new_stars_err.fill(np.nan)
        i = 0
        for istar in range(pos.shape[0]):
            if pos_mask[istar] == 1:
                new_stars_p[istar,:] = stars_p[i]
                new_stars_err[istar,:] = stars_err[i]
                i += 1
        stars_err = np.copy(new_stars_err)
        stars_p = np.copy(new_stars_p)
        
        returned_data['stars-params-err'] = stars_err
        returned_data['stars-params'] = stars_p
        return returned_data
        
    else:
        return []
        

def part_value(np.ndarray[np.float64_t, ndim=1] distrib, double coeff):
    """Return the value lying between two parts of a partition 

    The partition process is nan robusts. It is made over a
    distribution cleaned from nans.
    
    :param distrib: A 1D array of floats.
    
    :param coeff: Partition coefficient (must be >= 0. and <= 1.). If
      0 return the min of the distribution and if 1 return the max.
    """
    cdef np.ndarray[np.float64_t, ndim=1] cleaned_distrib = distrib[
        np.nonzero(~np.isnan(distrib))]
    cdef int k

    coeff = max(0., min(1., coeff)) # coeff is coerced between 0 and 1
    
    if coeff == 0:
        return bn.nanmin(distrib)
    if coeff == 1:
        return bn.nanmax(distrib)
    
    k = max(1, (min(<int> (coeff * np.size(cleaned_distrib)),
                    np.size(cleaned_distrib) - 1)))
    
    return bn.partition(cleaned_distrib, k)[k]

def indft(np.ndarray[np.float64_t, ndim=1] a,
          np.ndarray[np.float64_t, ndim=1] x):
    """Inverse Non-uniform Discret Fourier Transform.

    Compute the irregularly sampled interferogram from a regularly
    sampled spectrum.

    :param a: regularly sampled spectrum.
    
    :param x: positions of the interferogram samples. If x =
      range(size(a)), this function is equivalent to an idft or a
      ifft. Note that the ifft is of course much faster to
      compute. This vector may have any length.
    """
    cdef int N = a.shape[0]
    cdef int M = x.shape[0]
    cdef float angle = 0.
    cdef int m, n
    
    f = np.zeros(M, dtype=complex)
    cdef np.ndarray[np.float64_t, ndim=1] freal = np.zeros(M, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] fimag = np.zeros(M, dtype=float)
    for m in xrange(M):
        for n in xrange(N):
            angle = 2. * M_PI * x[m] * n / N
            freal[m] += a[n] * cos(angle)
            fimag[m] += a[n] * sin(angle)
    f.real = freal
    f.imag = fimag
    return f / N

def dft(np.ndarray[np.float64_t, ndim=1] a,
        np.ndarray[np.float64_t, ndim=1] x):
    """Discret Fourier Transform.

    Compute an irregularly sampled spectrum from a regularly
    sampled interferogram.

    :param a: regularly sampled interferogram.
    
    :param x: positions of the spectrum samples. If x =
      range(size(a)), this function is equivalent to an fft. Note that
      the fft is of course much faster to compute. This vector may
      have any length.
    """
    cdef int N = a.shape[0]
    cdef int M = x.shape[0]
    cdef float angle = 0.
    cdef int m, n
    
    f = np.zeros(M, dtype=complex)
    cdef np.ndarray[np.float64_t, ndim=1] freal = np.zeros(M, dtype=float)
    cdef np.ndarray[np.float64_t, ndim=1] fimag = np.zeros(M, dtype=float)
    for m in xrange(M):
        for n in xrange(N):
            angle = -2. * M_PI * x[m] * n / N
            freal[m] += a[n] * cos(angle)
            fimag[m] += a[n] * sin(angle)
    f.real = freal
    f.imag = fimag
    return f

          
def map_me(np.ndarray[np.float64_t, ndim=2] frame):
    """Create a map of the modulation efficiency from a laser frame.

    The more fringes the best are the results.

    :param frame: laser frame.
    """

    cdef np.ndarray[np.float64_t, ndim=2] me = np.empty_like(frame)
    cdef np.ndarray[np.float64_t, ndim=1] icol = np.empty(frame.shape[1],
                                                          dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] sign = np.empty_like(icol)
    cdef np.ndarray[np.float64_t, ndim=1] diff = np.empty(frame.shape[1] - 1,
                                                          dtype=np.float64)
    
    cdef int ii, ij, imin, imax
    cdef double vmin, vmax
    
    me.fill(np.nan)
    
    for ii in range(frame.shape[0]):
        icol = frame[ii,:]
        sign = np.sign(np.gradient(icol))
        diff = np.diff(sign)
        nans = diff[np.nonzero(np.isnan(diff))] = 0
        maxs = np.nonzero(np.abs(diff) > 0)[0]
        
        maxs = np.concatenate([maxs, np.array([frame.shape[1]-1])])

        for ij in range(maxs.shape[0] - 1):
            imin = maxs[ij]
            imax = maxs[ij+1]
            vmin = np.nanmin(icol[imin:imax])
            vmax = np.nanmax(icol[imin:imax])
            if vmax != 0.:
                me[ii, imin:imax] = (vmax-vmin)/vmax
        
    return me



def nanbin_image(np.ndarray[np.float64_t, ndim=2] im, int binning): 
    """Mean image binning robust to NaNs.

    :param im: Image to bin
    :param binning: Binning factor (must be an integer)
    """     
    cdef np.ndarray[np.float64_t, ndim=2] out 
    cdef np.ndarray[np.int64_t, ndim=1] x_range
    cdef np.ndarray[np.int64_t, ndim=1] y_range
    cdef int ii, ij, xmin, xmax, ymin, ymax
    
    x_range = np.arange(0, im.shape[0]/binning*binning, binning)
    y_range = np.arange(0, im.shape[1]/binning*binning, binning)
    out = np.empty((x_range.shape[0], y_range.shape[0]), dtype=im.dtype)
    
    for ii in range(x_range.shape[0]):
        for ij in range(y_range.shape[0]):
            xmin = x_range[ii]
            xmax = xmin + binning
            if xmax > im.shape[0]: xmax=im.shape[0]
            ymin = y_range[ij]
            ymax = ymin + binning
            if ymax > im.shape[1]: ymax=im.shape[1]
            
            out[ii,ij] = bn.nanmean(im[xmin:xmax,ymin:ymax])
            
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
def unbin_image(np.ndarray[np.float64_t, ndim=2] im,
                int nx, int ny):
    """Unbin a binned image (restore the image binned with the
    function :py:func:`~orb.cutils.nanbin_image`).

    :param im: Image to unbin.

    :param nx: X dimension of the unbinned image.
    
    :param ny: Y dimension of the unbinned image.
    """

    cdef np.ndarray[np.float64_t, ndim=2] out
    cdef np.ndarray[np.float64_t, ndim=1] xaxis
    cdef np.ndarray[np.float64_t, ndim=1] yaxis
    cdef int binx, biny
    cdef int ii, ij, x1, x2, y1, y2
    cdef double q11, q12, q21, q22, x, y
    cdef int dimx = im.shape[0]
    cdef int dimy = im.shape[1]
    cdef double x1d, y1d, x2d, y2d
    
    out = np.empty((nx, ny), dtype=float)
    out.fill(np.nan)
    
    binx = nx / dimx
    biny = ny / dimy

    xaxis = (np.arange(dimx * binx) - (<double> binx / 2. - 0.5)) / (<double> binx)
    yaxis = (np.arange(dimy * biny) - (<double> biny / 2. - 0.5)) / (<double> biny)
    with nogil:
        for ii in range(dimx * binx):
            for ij in range(dimy * biny):
                x = xaxis[ii]
                y = yaxis[ij]
                if x >= 0. and y >= 0.:
                    x1 = <int> x
                    x2 = x1 + 1
                    y1 = <int> y
                    y2 = y1 + 1
                    x1d = <double> x1
                    x2d = <double> x2
                    y1d = <double> y1
                    y2d = <double> y2
                    
                    if x1 >= 0 and y1 >= 0 and x2 < dimx and y2 < dimy:
                        q11 = <double> im[x1, y1]
                        q12 = <double> im[x1, y2]
                        q21 = <double> im[x2, y1]
                        q22 = <double> im[x2, y2]
                        out[ii,ij] = (q11 * (x2d - x) * (y2d - y)
                                      + q21 * (x - x1d) * (y2d - y)
                                      + q12 * (x2d - x) * (y - y1d)
                                      + q22 * (x - x1d) * (y - y1d))
            
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def im2rgba(np.ndarray[np.float64_t, ndim=2] im,
            mpl_colorbar,
            double vmin, double vmax,
            int xmin, int xmax, int ymin, int ymax,
            np.ndarray[np.uint8_t, ndim=2] computed_pixels,
            last_arr8,
            int res=1000):

    """Compute RGBA from image given a matplotlib colorbar instance.

    This is a function used by :py:class:`orb.visual.ImageCanvas`. It
    is not a generalist function. It has been written to accelerate
    matplotlib function colorbar.to_rgba().

    :param im: Image
    :param mpl_colorbar: A matplotlib colorbar instance
    :param vmin: min value of the colorbar
    :param vmax: max value of the colorbar
    :param xmin: min x index of the region to compute
    :param xmax: max x index of the region to compute
    :param ymin: min y index of the region to compute
    :param ymax: max y index of the region to compute
    :param computed_pixels: Array giving the already computed pixels
    :param last_arr8: If not None, last computed array.
    :param res: (Optional) Lookup table resolution (default 1000)
    """

    cdef np.ndarray[np.float64_t, ndim=1] color_values
    cdef np.ndarray[np.uint8_t, ndim=2] color_mapper
    cdef np.ndarray[np.uint8_t, ndim=3] arr8
    cdef np.ndarray[np.uint16_t, ndim=1] x_range, y_range
    cdef int ii, ij, index, ir, n
    
    color_values = np.linspace(vmin, vmax, res)
    color_mapper = mpl_colorbar.to_rgba(
        color_values, alpha=None, bytes=True)
    if last_arr8 is None:
        arr8 = np.ones((im.shape[1], im.shape[0], 4),
                       dtype=np.uint8) * 255
    else:
        arr8 = last_arr8

    pix_to_compute = np.nonzero(computed_pixels[xmin:xmax, ymin:ymax] < 1)
    x_range = pix_to_compute[0].astype(np.uint16)
    y_range = pix_to_compute[1].astype(np.uint16)
    n = <int> len(x_range)
    
    with nogil:
        for ir in range(n):
            ii = <int> x_range[ir] + xmin
            ij = <int> y_range[ir] + ymin
            if computed_pixels[ii,ij] < 1:
                if isnan(im[ii,ij]): index = -1
                elif im[ii,ij] <= vmin: index = 0
                elif im[ii,ij] >= vmax: index = res - 1
                else:
                    index = <int> (((im[ii,ij] - vmin) / (vmax - vmin)) * (res - 1))
                for ik in range(3):
                    if index < 0:
                        arr8[ij,ii,ik] = 255
                    else:
                        arr8[ij,ii,ik] = color_mapper[index, ik]
                    
    return arr8

@cython.boundscheck(False)
@cython.wraparound(False)
def brute_photometry(np.ndarray[np.float64_t, ndim=2] im,
                     np.ndarray[np.float64_t, ndim=2] star_list,
                     np.ndarray[np.float64_t, ndim=2] kernel,
                     int box_size):

    cdef double total_flux = 0.
    cdef int star_nb = star_list.shape[0]
    cdef int istar
    cdef int x_min, x_max, y_min, y_max
    cdef int dimx = im.shape[0]
    cdef int dimy = im.shape[1]
    cdef int ii, ij
    cdef double ix, iy
    cdef double val

    with nogil:
        for istar in range(star_nb):
            ix = star_list[istar,0]
            iy = star_list[istar,1]
            if not isnan(ix) and not isnan(iy):
                x_min = <int> (ix - box_size/2)
                y_min = <int> (iy - box_size/2)

                if (ix + box_size < dimx
                    and iy + box_size < dimy
                    and x_min > 0
                    and y_min > 0):
                    for ii in range(box_size):
                        for ij in range(box_size):
                            val = (im[x_min+ii, y_min+ij]
                                   * kernel[ii, ij])
                            if not isnan(val):
                                total_flux += val
    return total_flux

@cython.boundscheck(False)
@cython.wraparound(False)
def detect_cosmic_rays(np.ndarray[np.float64_t, ndim=2] frame,
                       crs_list, int box_size, double detect_coeff):
    """Check if a given pixel is a cosmic ray (classic detection).
    
    classic detection: pixel value is checked against standard
    deviation of values in a box around the pixel.

    :param frame: Frame to check
    
    :param crs_list: List of pixels to check
    
    :param box_size: Size of the box in pixels
    
    :param detect_coeff: Coefficient of detection (number of sigmas
      threshold)
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] workframe = np.copy(frame)
    cdef np.ndarray[np.uint8_t, ndim=2] cr_map = np.zeros_like(
        frame, dtype=np.uint8)
    cdef int cr_list_len = len(crs_list[0])
    cdef int icr, ix, iy, xmin, xmax, ymin, ymax
    cdef double boxmed, boxstd
    cdef np.ndarray[np.float64_t, ndim=2] box
    
    workframe[crs_list] = np.nan
    
    for icr in range(cr_list_len):
        ix = crs_list[0][icr]
        iy = crs_list[1][icr]

        xmin, xmax, ymin, ymax = get_box_coords(
            ix, iy, box_size,
            0, workframe.shape[0], 0, workframe.shape[1])

        if xmax - xmin == box_size and ymax - ymin == box_size:
        
            box = np.copy(workframe[xmin:xmax, ymin:ymax])
        
            if not np.all(np.isnan(box)):
                boxstd = bn.nanstd(box)
                boxmed = bn.nanmedian(box)
                if frame[ix,iy] > boxmed + detect_coeff * boxstd:
                    cr_map[ix,iy] = 1
                
    return cr_map

@cython.boundscheck(False)
@cython.wraparound(False)
def check_cosmic_rays_neighbourhood(
    np.ndarray[np.float64_t, ndim=2] frame,
    np.ndarray[np.uint8_t, ndim=2] cr_map,
    int box_size, double detect_coeff):
    """Check the neighbourhood around detected cosmic rays in a frame.

    :param frame: Frame to check
    
    :param cr_map: Map of the cosimic-rays positions (boolean map, 1
      is a cosmic ray)

    :param box_size: Size of the box checked around each cr.

    :param detect_coeff: Coefficient of detection (number of sigmas
      threshold)
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] workframe = np.copy(frame)
    cdef int inewcr, icr, ix, iy, xmin, xmax, ymin, ymax, ii, ij
    cdef np.ndarray[np.float64_t, ndim=2] box
    cdef double boxmed, boxstd

    crs_list = np.nonzero(cr_map)
    workframe[crs_list] = np.nan
    
    for icr in range(len(crs_list[0])):
        ix = crs_list[0][icr]
        iy = crs_list[1][icr]
        xmin, xmax, ymin, ymax = get_box_coords(
            ix, iy, box_size,
            0, workframe.shape[0], 0, workframe.shape[1])
        
        if xmax - xmin == box_size and ymax - ymin == box_size:
            box = np.copy(workframe[xmin:xmax, ymin:ymax])
            if not np.all(np.isnan(box)):
                boxstd = bn.nanstd(box)
                boxmed = bn.nanmedian(box)
                for ii in range(xmin, xmax):
                    for ij in range(ymin, ymax):
                        if frame[ii, ij] > boxmed + detect_coeff * boxstd:
                            cr_map[ii,ij] = 1
            
    return cr_map


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_w2pix(np.ndarray[np.float64_t, ndim=1] w,
               double axis_min, double axis_step):
    """Fast conversion of wavelength/wavenumber to pixel

    :param w: wavelength/wavenumber
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    return np.abs(w - axis_min) / axis_step

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_pix2w(np.ndarray[np.float64_t, ndim=1] pix,
               double axis_min, double axis_step):
    """Fast conversion of pixel to wavelength/wavenumber

    :param pix: position along axis in pixels
    
    :param axis_min: min axis wavelength/wavenumber
    
    :param axis_step: axis step size in wavelength/wavenumber
    """
    return pix * axis_step + axis_min


def get_cm1_axis_min(int n, double step, int order, double corr=1.):
    """Return min wavenumber of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis

    :param step: Step size in nm
    
    :param order: Folding order
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    # last sample of the axis is removed because this sample is also
    # removed in orb.utils.fft.transform_interferogram (after fft
    # spectrum is cropped to step_nb/2 instead of step_nb/2 + 1)
    cdef double cm1_min = <double> order / (2.* step) * corr * 1e7
    if order & 1:
        cm1_min += get_cm1_axis_step(n, step, corr=corr)
    return cm1_min

def get_cm1_axis_max(int n, double step, int order, double corr=1.):
    """Return max wavenumber of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis

    :param step: Step size in nm
    
    :param order: Folding order
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    # last sample of the axis is removed because this sample is also
    # removed in orb.utils.fft.transform_interferogram (after fft
    # spectrum is cropped to step_nb/2 instead of step_nb/2 + 1)
    cdef double cm1_max = <double> (order + 1.) / (2. * step) * corr * 1e7
    if not order & 1:
        cm1_max -= get_cm1_axis_step(n, step, corr=corr)
    return cm1_max

def get_cm1_axis_step(int n, double step, double corr=1.):
    """Return step size of a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    return corr / (2. * <double> n * step) * 1e7

def get_nm_axis_min(int n, double step, int order, double corr=1.):
    """Return min wavelength of regular wavelength axis in nm.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order (cannot be 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if order == 0: raise ValueError('Order cannot be 0 for a nm axis (minimum wavelength is infinite), use a cm-1 axis instead')
    return 1. / get_cm1_axis_max(n, step, order, corr=corr) * 1e7

def get_nm_axis_max(int n, double step, int order, double corr=1.):
    """Return max wavelength of regular wavelength axis in nm.

    :param n: Number of steps on the axis

    :param step: Step size in nm
    
    :param order: Folding order (cannot be 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if order == 0: raise ValueError('Order cannot be 0 for a nm axis (minimum wavelength is infinite), use a cm-1 axis instead')
    return 1. / get_cm1_axis_min(n, step, order, corr=corr) * 1e7

def get_nm_axis_step(int n, double step, int order, double corr=1.):
    """Return step size of a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    
    :param step: Step size in nm
    
    :param order: Folding order (cannot be 0)
    
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if (order > 0): 
        return 2. * step / (<double> order * <double> (order + 1) * corr) / <double> n
    else: raise ValueError('Order cannot be 0 for a nm axis (stepsize is infinite), use a cm-1 axis instead')


@cython.boundscheck(False)
@cython.wraparound(False)
def filter_background(np.ndarray[np.float64_t, ndim=2] frame,
                      int box_size, int big_box_coeff):
    """Replace each pixel by the value in a box around it minus the
    median of the baclground.
    
    :param frame: Frame to filter
    
    :param box_size: Size of the box
    
    :param big_box_coeff: Coeff by which the bo size is multiplied to
      get the background box size.
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] workframe = np.copy(frame)
    cdef np.ndarray[np.float64_t, ndim=2] box
    cdef np.ndarray[np.float64_t, ndim=2] back
    
    
    cdef int ii, ij, dimx, dimy, back_size, nans, ik
    cdef float mean_box, median_back
    cdef int xmin, xmax, ymin, ymax, xminb, xmaxb, yminb, ymaxb

    back_size = big_box_coeff * box_size

    dimx = frame.shape[0]
    dimy = frame.shape[1]
    
    
    for ii in range(dimx):
        for ij in range(dimy):

            xmin, xmax, ymin, ymax = get_box_coords(
                ii, ij, box_size,
                0, dimx, 0, dimy)
            box = frame[xmin:xmax, ymin:ymax]
            mean_box = mean2d(box)
            xminb, xmaxb, yminb, ymaxb = (
                get_box_coords(
                    ii, ij, back_size, 0, dimx,
                    0, dimy))
            back = np.copy(frame[xminb:xmaxb, yminb:ymaxb])
            back[xmin - xminb:xmax - xminb,
                 ymin - yminb:ymax - yminb] = np.nan
                             
            median_back = median2d(back)
            workframe[ii, ij] = mean_box - median_back
                                  
    return workframe


@cython.boundscheck(False)
@cython.wraparound(False)
def mean2d(np.ndarray[np.float64_t, ndim=2] box):
    """Return the mean of a 2d box with no GIL

    :param box: 2d array
    """
    cdef int ii, ij
    cdef int dimx, dimy, nb
    cdef double val
    dimx = box.shape[0]
    dimy = box.shape[1]
    val = 0
    nb = 0
    with nogil:
        for ii in range(dimx):
            for ij in range(dimy):
                if not isnan(box[ii,ij]):
                    val += box[ii,ij]
                    nb += 1
    if nb == 0: return np.nan
    return val / <float> nb
                
        
@cython.boundscheck(False)
@cython.wraparound(False)
def median2d(np.ndarray[np.float64_t, ndim=2] box):
    """Return the median of a 2d box with no GIL

    :param box: 2d array
    """
    cdef int ii, ij
    cdef int dimx, dimy, nb
    cdef np.ndarray[np.float64_t, ndim=1] box_s = np.empty(
        np.size(box), dtype=float)

    box_s = np.sort(box.flatten())
    dimx = box_s.shape[0]
    with nogil:
        for ii in range(dimx):
            if not isnan(box_s[dimx-ii-1]):
                nb = ii
                break
    return box_s[(dimx-nb)/2]
            
