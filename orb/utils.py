#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: utils.py

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
Utils module contains functions that are used by the processing
classes of ORBS
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'

import version
__version__ = version.__version__

import warnings
import os
import subprocess

from core import Tools
import cutils

import numpy as np
from scipy import interpolate, signal, ndimage, optimize
import scipy.ndimage.filters
import math
import bottleneck as bn

##################################################
#### MISC  #######################################
##################################################

def get_mask_from_ds9_region_line(reg_line, x_range=None, y_range=None):
    """Read one line of a ds9 region file and return the list of
    pixels in the region.

    :param reg_line: Line of the ds9 region file
    
    :param x_range: (Optional) Range of x image coordinates
        considered as valid. Pixels outside this range are
        rejected. If None, no validation is done (default None).

    :param y_range: (Optional) Range of y image coordinates
        considered as valid. Pixels outside this range are
        rejected. If None, no validation is done (default None).

    .. note:: The returned array can be used like a list of
        indices returned by e.g. numpy.nonzero().

    .. note:: Coordinates can be image coordinates (x,y) or sky
        coordinates in degrees (ra, dec)
    """
    x_list = list()
    y_list = list()
    
    if len(reg_line) <= 3:
        Tools._print_warning('Bad region line')
        return None
        
    if reg_line[:3] == 'box':
        reg_line = reg_line.split('#')[0]
        reg_line = reg_line[4:]
        if '"' in reg_line:
            reg_line = reg_line[:-3]
        else:
            reg_line = reg_line[:-2]

        if ',' in reg_line:
            box_coords = np.array(reg_line.split(","), dtype=float)
        else:
            Tools()._print_error('Bad coordinates, check if coordinates are in pixels')

        x_min = round(box_coords[0] - (box_coords[2] / 2.) - 1.5)
        x_max = round(box_coords[0] + (box_coords[2] / 2.) + .5)
        y_min = round(box_coords[1] - (box_coords[3] / 2.) - 1.5) 
        y_max = round(box_coords[1] + (box_coords[3] / 2.) + .5)        
        if x_range != None:
            if x_min < np.min(x_range) : x_min = np.min(x_range)
            if x_max > np.max(x_range) : x_max = np.max(x_range)
        if y_range != None:
            if y_min < np.min(y_range) : y_min = np.min(y_range)
            if y_max > np.max(y_range) : y_max = np.max(y_range)

        for ipix in range(int(x_min), int(x_max)):
            for jpix in range(int(y_min), int(y_max)):
                x_list.append(ipix)
                y_list.append(jpix)

    if reg_line[:6] == 'circle':
        reg_line = reg_line.split('#')[0]
        reg_line = reg_line[7:]
        if '"' in reg_line:
            reg_line = reg_line[:-3]
        else:
            reg_line = reg_line[:-2]
        cir_coords = np.array(reg_line.split(","), dtype=float)
        x_min = round(cir_coords[0] - (cir_coords[2]) - 1.5)
        x_max = round(cir_coords[0] + (cir_coords[2]) + .5)
        y_min = round(cir_coords[1] - (cir_coords[2]) - 1.5)
        y_max = round(cir_coords[1] + (cir_coords[2]) + .5)
        if x_range != None:
            if x_min < np.min(x_range) : x_min = np.min(x_range)
            if x_max > np.max(x_range) : x_max = np.max(x_range)
        if y_range != None:
            if y_min < np.min(y_range) : y_min = np.min(y_range)
            if y_max > np.max(y_range) : y_max = np.max(y_range)


        for ipix in range(int(x_min), int(x_max)):
            for jpix in range(int(y_min), int(y_max)):
                if (math.sqrt((ipix - cir_coords[0] + 1.)**2
                             + (jpix - cir_coords[1] + 1.)**2)
                    <= round(cir_coords[2])):
                    x_list.append(ipix)
                    y_list.append(jpix)

    if reg_line[:7] == 'polygon':
        reg_line = reg_line.split('#')[0]
        reg_line = reg_line[8:-2]
        reg_line = np.array(reg_line.split(',')).astype(float)
        if np.size(reg_line) > 0:
            poly = [(reg_line[i], reg_line[i+1]) for i in range(
                0,np.size(reg_line),2)]
            poly_x = np.array(poly)[:,0]
            poly_y = np.array(poly)[:,1]
            x_min = np.min(poly_x)
            x_max = np.max(poly_x)
            y_min = np.min(poly_y)
            y_max = np.max(poly_y)
            if x_range != None:
                if x_min < np.min(x_range) : x_min = np.min(x_range)
                if x_max > np.max(x_range) : x_max = np.max(x_range)
            if y_range != None:
                if y_min < np.min(y_range) : y_min = np.min(y_range)
                if y_max > np.max(y_range) : y_max = np.max(y_range)

            for ipix in range(int(x_min), int(x_max)):
                for jpix in range(int(y_min), int(y_max)):
                    if cutils.point_inside_polygon(ipix, jpix, poly):
                         x_list.append(ipix)
                         y_list.append(jpix)

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    return list([x_list, y_list])

                        
def get_mask_from_ds9_region_file(reg_path, x_range=None,
                                  y_range=None):
    """Return the indices of the elements inside 'box', 'circle' and
    'polygon' regions.

    :param reg_path: Path to a ds9 region file

    :param x_range: (Optional) Range of x image coordinates
        considered as valid. Pixels outside this range are
        rejected. If None, no validation is done (default None).

    :param y_range: (Optional) Range of y image coordinates
        considered as valid. Pixels outside this range are
        rejected. If None, no validation is done (default None).

    .. note:: The returned array can be used like a list of
        indices returned by e.g. numpy.nonzero().

    .. note:: Coordinates can be image coordinates (x,y) or sky
        coordinates in degrees (ra, dec)
    """
    f = Tools().open_file(reg_path, 'r')
    x_list = list()
    y_list = list()
                    
    for iline in f:
        if len(iline) > 3:
            mask = get_mask_from_ds9_region_line(iline, x_range=x_range,
                                                 y_range=y_range)
            x_list += list(mask[0])
            y_list += list(mask[1])
                            
    x_list = np.array(x_list)
    y_list = np.array(y_list)

    return list([x_list, y_list])

def compute_obs_params(nm_min_filter, nm_max_filter,
                       theta_min=5.01, theta_max=11.28):
    """Compute observation parameters (order, step size) given the
    filter bandpass.

    :param nm_min_filter: Min wavelength of the filter in nm.

    :param nm_max_filter: Max wavelength of the filter in nm.
    
    :param theta_min: (Optional) Min angle of the detector (default
      5.01).

    :param theta_max: (Optional) Max angle of the detector (default
      11.28).

    :return: A tuple (order, step size, max wavelength)
    """
    def get_step(nm_min, n, cos_min):
        return int(nm_min * ((n+1)/(2*cos_min)))

    def get_nm_max(step, n, cos_max):
        return 2. * step * cos_max / float(n)
    cos_min = math.cos(math.radians(theta_min))
    cos_max = math.cos(math.radians(theta_max))

    n = 1
    order_found = False
    while n < 100 and not order_found:
        step = get_step(nm_min_filter, n, cos_min)
        nm_max = get_nm_max(step, n, cos_max)
        if nm_max <= nm_max_filter:
            order_found = True
            order = n - 1
        n += 1

    step = get_step(nm_min_filter, order, cos_min)
    nm_max = get_nm_max(step, order, cos_max)
    
    return order, step, nm_max

def ABmag2fnu(ABmag):
    """Return flux in erg/cm2/s/Hz from AB magnitude (Oke, ApJS, 27,
    21, 1974)

    ABmag = -2.5 * log10(f_nu) - 48.60
    f_nu = 10^(-0.4 * (ABmag + 48.60))

    :param ABmag: A magnitude in the AB magnitude system

    .. note:: Definition of the zero-point can change and be
      e.g. 48.59 for Oke standard stars (Hamuy et al., PASP, 104, 533,
      1992). This is the case for Spectrophotometric Standards given
      on the ESO website (https://www.eso.org/sci/observing/tools/standards/spectra/okestandards.html). Here the HST definition is used.
    """
    return 10**(-0.4*(ABmag + 48.60))

def fnu2flambda(fnu, nu):
    """Convert a flux in erg/cm2/s/Hz to a flux in erg/cm2/s/A

    :param fnu: Flux in erg/cm2/s/Hz
    :param nu: frequency in Hz
    """
    c = 2.99792458e18 # Ang/s
    return fnu * nu**2. / c

def lambda2nu(lam):
    """Convert lambda in Ang to nu in Hz

    :param lam: Wavelength in angstrom
    """
    c = 2.99792458e18 # Ang/s
    return c / lam

def ABmag2flambda(ABmag, lam):
    """Convert AB magnitude to flux in erg/cm2/s/A

    :param ABmag: A magnitude in the AB magnitude system

    :param lam: Wavelength in angstrom
    """
    return fnu2flambda(ABmag2fnu(ABmag), lambda2nu(lam))

def ra2deg(ra):
     ra = np.array(ra, dtype=float)
     if (ra.shape == (3,)):
          return (ra[0] + ra[1]/60. + ra[2]/3600.)*(360./24.)
     else:
          return None

def dec2deg(dec):
     dec = np.array(dec, dtype=float)
     if (dec.shape == (3,)):
         if dec[0] > 0:
             return dec[0] + dec[1]/60. + dec[2]/3600.
         else:
             return dec[0] - dec[1]/60. - dec[2]/3600.
     else:
         return None   

def deg2ra(deg, string=False):
     deg=float(deg)
     ra = np.empty((3), dtype=float)
     deg = deg*24./360.
     ra[0] = int(deg)
     deg = deg - ra[0]
     deg *= 60.
     ra[1] = int(deg)
     deg = deg - ra[1]
     deg *= 60.
     ra[2] = deg
     if not string:
          return ra
     else:
          return "%d:%d:%.2f" % (ra[0], ra[1], ra[2])

def deg2dec(deg, string=False):
     deg=float(deg)
     dec = np.empty((3), dtype=float)
     dec[0] = int(deg)
     deg = deg - dec[0]
     deg *= 60.
     dec[1] = int(deg)
     deg = deg - dec[1]
     deg *= 60.
     dec[2] = deg
     if (float("%.2f" % (dec[2])) == 60.0):
          dec[2] = 0.
          dec[1] += 1
     if dec[1] == 60:
          dec[0] += 1
     if dec[0] < 0:
         dec[1] = -dec[1]
         dec[2] = -dec[2]
     if not string:
          return dec
     else:
        return "+%d:%d:%.2f" % (dec[0], dec[1], dec[2])

def correct_bad_frames_vector(bad_frames_vector, dimz):
    """Remove bad indexes of the bad frame vector.

    :param bad_frames_vector: The vector of indexes to correct
    :param dimz: Dimension of the cube along the 3rd axis.
    """
    if (bad_frames_vector == None
        or np.size(bad_frames_vector) == 0):
        return bad_frames_vector
    
    bad_frames_vector= np.array(np.copy(bad_frames_vector))
    bad_frames_vector = [bad_frames_vector[badindex]
                         for badindex in range(bad_frames_vector.shape[0])
                         if (bad_frames_vector[badindex] >= 0
                             and bad_frames_vector[badindex] < dimz)]
    return bad_frames_vector


def find_zpd(interf, step_number=None,
             return_zpd_shift=False):
    """Return the index of the ZPD along the z axis.

    :param step_number: (Optional) If the full number of steps is
      greater than the number of frames of the cube. Useful when
      the interferograms are non symetric (default None).

    :param return_zpd_shift: (Optional) If True return ZPD shift
      instead of ZPD index (default False).
    """
    if step_number != None:
        dimz = step_number
    else:
        dimz = interf.shape[0]

    interf = np.copy(interf)
    # correct vector for zeros
    interf[
        np.nonzero(interf == 0)] = np.median(interf)

    # filtering vector to remove low and high frequency patterns (e.g. sunrise)
    interf = fft_filter(interf, 0.3, filter_type='high_pass')
    interf = fft_filter(interf, 0.5, filter_type='low_pass')
    
    full_interf = np.zeros(dimz, dtype=float)
    full_interf[:interf.shape[0]] = interf
    
    # vector is weighted so that the center part is prefered
    full_interf *= norton_beer_window(fwhm='2.0', n=dimz)
    # absolute value of the vector
    full_interf = np.sqrt(full_interf**2.)

    # ZPD is defined to be at the maximum of the vector
    zpd_index = np.argmax(full_interf)

    #zpd_shift = int(int(self.dimz/2.) - zpd_index + self.dimz%2)
    zpd_shift = int(int(dimz/2.) - zpd_index)
    
    if return_zpd_shift:
        return zpd_shift
    else:
        return zpd_index


def count_nonzeros(a):
    """Return the length of nonzeros parts in a vector as a vector of
    the same length with the length of each part at each occurence of
    a nonzero number.

    e.g. : if a = [0,0,0,1,1,0,1] this function returns: [0,0,0,2,2,0,1]

    :param a: A vector.
    """
    counts = np.zeros_like(a)
    for iz in range(a.shape[0]):
        if a[iz] != 0 and counts[iz] == 0:
            end_count = False
            il = 0
            while not end_count:
                if (iz+il >= a.shape[0]
                    or a[iz+il] == 0):
                    end_count = True
                else:
                    il += 1
            counts[iz:iz+il] = il
    return counts


def robust_mean(a, weights=None, warn=True):
    """Compute the mean of a distribution even with NaN values

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param weights: Weights of each value of a (Must have the same
      length as a). If None, weights are all considered equal to 1
      (default None).
      
    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    
    if weights == None:
        result = cutils.robust_mean(a)
    else:
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
            
        result = cutils.robust_average(a, weights)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        Tools()._print_warning('Only NaN values found in the given array')
        
    return result

def robust_std(a, warn=True):
    """Compute the std of a distribution even with NaN values
    
    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = cutils.robust_std(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        Tools()._print_warning('Only NaN values found in the given array')
        
    return result

def robust_sum(a, warn=True):
    """Compute the sum of a distribution (skip NaN values)

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck
    
    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = cutils.robust_sum(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        Tools()._print_warning('Only NaN values found in the given array')
        
    return result

def robust_median(a, warn=True):
    """Compute the median of a distribution (skip NaN values).

    This is based on bottleneck module. See:
    https://pypi.python.org/pypi/Bottleneck

    :param a: A distribution of values

    :param warn: If True, warnings are raised.
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
        
    result = cutils.robust_median(a)
    
    if np.isnan(result.imag) and np.isnan(result.real) and warn:
        Tools()._print_warning('Only NaN values found in the given array')
        
    return result

def sigmacut(x, sigma=3., min_values=3, central_value=None, warn=False):
    """Return a distribution after a sigma cut rejection
    of the too deviant values.

    :param x: The distribution to cut
    
    :param sigma: (Optional) Number of sigma above which values are
      considered as deviant (default 3.)

    :param min_values: (Optional) Minimum number of values to return
      (default 3)

    :param central_value: (Optional) If not none, this value is used as
      the central value of the cut. Else the median of the
      distribution is used as the central value (default None)

    :param warn: (Optional) If False no warning message is printed
      (default False).
    """
    if central_value == None:
        central_value = 0.
        use_central_value = False
    else:
        use_central_value = True

    if np.size(x) <= min_values:
        if warn:
            Tools()._print_warning("No sigma-cut done because the number of values (%d) is too low"%np.size(x))
        return x
        
    return cutils.sigmacut(np.array(x).astype(float).flatten(), central_value,
                           use_central_value, sigma, min_values)
    

def pp_create_master_frame(frames, combine='average', reject='avsigclip',
                           sigma=3.):
    """
    Run a parallelized version of :py:meth:`utils.create_master_frame`.

    Use it only for big data set because it can be much slower for a
    small data set (< 500 x 500 x 10).

    :param frames: Frames to combine.

    :param reject: (Optional) Rejection operation. Can be 'sigclip',
      'minmax', 'avsigclip' (default 'avsigclip')

    :param combine: (Optional) Combining operation. Can be
      'average' or 'median' (default 'average')

    :param sigma: (Optional) Sigma factor for pixel rejection
      (default 3.).

    .. seealso:: :py:meth:`utils.create_master_frame`
    """
    to = Tools()
    job_server, ncpus = to._init_pp_server()
    divs = np.linspace(0, frames.shape[0], ncpus + 1).astype(int)
    result = np.empty((frames.shape[0], frames.shape[1]), dtype=float)
    
    frames = check_frames(frames)
    
    jobs = [(ijob, job_server.submit(
        create_master_frame, 
        args=(frames[divs[ijob]:divs[ijob+1],:,:],
              combine, reject, sigma, True, False),
        modules=("import numpy as np",
                 "from orbs.utils import create_master_frame",
                 "from orbs.cutils import master_combine")))
            for ijob in range(ncpus)]
    
    for ijob, job in jobs:
        result[divs[ijob]:divs[ijob+1],:] = job()
        
    to._close_pp_server(job_server)
    
    return result


def check_frames(frames, sigma_reject=2.):
    """Check and reject deviating frames based on their median level.

    Frames with a too deviant median level are discarded. This
    function is used by :py:meth:`utils.create_master_frame`.

    :param frames: Set of frames to check
    
    :param sigma_reject: (Optional) Rejection coefficient (default 2.)
    
    """
    z_median = np.array([bn.nanmedian(frames[:,:,iframe])
                         for iframe in range(frames.shape[2])])
    z_median_cut = sigmacut(z_median, sigma=sigma_reject)
    bad_frames = (z_median > (robust_median(z_median_cut)
                         + sigma_reject * robust_std(z_median_cut)))
    if np.any(bad_frames):
        Tools()._print_warning('Some frames (%d) appear to be much different from the others. They have been removed before being combined. Please check the frames.'%np.sum(bad_frames))
        Tools()._print_msg('Median levels: %s'%str(z_median))
        Tools()._print_msg('Rejected: %s'%str(bad_frames))
        frames = np.dstack([frames[:,:,iframe]
                            for iframe in range(frames.shape[2])
                            if not bad_frames[iframe]])
    return frames
        

def create_master_frame(frames, combine='average', reject='avsigclip',
                        sigma=3., silent=False, check=True):
    """
    Create a master frame from a set a frames.

    This method has been inspired by the **IRAF** function
    combine.

    :param frames: Frames to combine.

    :param reject: (Optional) Rejection operation. Can be 'sigclip',
      'minmax', 'avsigclip' (default 'avsigclip')

    :param combine: (Optional) Combining operation. Can be
      'average' or 'median' (default 'average')

    :param sigma: (Optional) Sigma factor for pixel rejection
      (default 3.).

    :param silent: (Optional) If True no information message are
      displayed.

    :param check: (Optional) If True deviating frames are rejected
      before combination (default True).

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

    NKEEP = 2 # Minimum number of values to keep for each pixel

    frames = np.array(frames)

    if len(frames.shape) == 2: # only one image
        if not silent: Tools()._print_warning("Only one image to create a master frame. No combining method can be used.")
        return frames

    if frames.shape[2] < 3:
        if not silent: Tools()._print_warning("Not enough frames to use a rejection method (%d < 3)"%frames.shape[2])
        reject = None

    if reject not in ['sigclip', 'minmax', 'avsigclip']:
        Tools()._print_error("Rejection operation must be 'sigclip', 'minmax' or None")
    if combine not in ['median', 'average']:
        Tools()._print_error("Combining operation must be 'average' or 'median'")

    if not silent: Tools()._print_msg("Rejection operation: %s"%reject)
    if not silent: Tools()._print_msg("Combining operation: %s"%combine)

    if reject == 'avsigclip':
        reject_mode = 0
    if reject == 'sigclip':
        reject_mode = 1
    if reject == 'minmax':
        reject_mode = 2
        
    if combine == 'average':
        combine_mode = 0
    if combine == 'median':
        combine_mode = 1

    if check:
        frames = check_frames(frames)
    
    master, reject_count_frame = cutils.master_combine(
        frames, sigma, NKEEP, combine_mode, reject_mode)

    if reject in ['sigclip', 'avsigclip']:
        if not silent: Tools()._print_msg("Maximum number of rejected pixels: %d"%np.max(reject_count_frame))
        if not silent: Tools()._print_msg("Mean number of rejected pixels: %f"%np.mean(reject_count_frame))

    
    return master


def get_box_coords(ix, iy, box_size,
                   x_lim_min, x_lim_max,
                   y_lim_min, y_lim_max):
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

    """
    return cutils.get_box_coords(int(ix), int(iy), int(box_size),
                                 int(x_lim_min), int(x_lim_max),
                                 int(y_lim_min), int(y_lim_max))

def transform_star_position_A_to_B(star_list_A, params, rc, zoom_factor):
    """Transform star positions in camera A to the same star position
    in camera B given the transformation parameters

    :param star_list_A: List of star coordinates in the cube A.
    :param params: Transformation parameters [dx, dy, dr, da, db].
    :param rc: Rotation center coordinates.
    :param zoom_factor: Zooming factor between the two cameras.
    """
    if not isinstance(star_list_A, np.ndarray):
        star_list_A = np.array(star_list_A)
    if star_list_A.dtype != np.dtype(float):
        star_list_A.astype(float)
        
    star_list_B = np.empty_like(star_list_A)
    for istar in range(star_list_A.shape[0]):
        star_list_B[istar,:] = cutils.transform_A_to_B(
            star_list_A[istar,0], star_list_A[istar,1],
            params[0], params[1], params[2], params[3], params[4],
            rc[0], rc[1], zoom_factor)
        
    return star_list_B

def restore_error_settings(old_settings):
    """Restore old floating point error settings of numpy.
    """
    np.seterr(divide = old_settings["divide"])
    np.seterr(over = old_settings["over"])
    np.seterr(under = old_settings["under"])
    np.seterr(invalid = old_settings["invalid"])
        
def amplitude(a):
    """Return the amplitude of a complex number"""
    return np.sqrt(a.real**2. + a.imag**2)

def phase(a):
    """Return the phase of a complex number"""
    return np.arctan2(a.imag, a.real)

def real(amp, pha):
    """Return the real part from amplitude and phase"""
    return amp * np.cos(pha) 

def imag(amp, pha):
    """Return the imaginary part from amplitude and phase"""
    return amp * np.sin(pha)

def next_power_of_two(n):
    """Return the next power of two greater than n.
    
    :param n: The number from which the next power of two has to be
      computed. Can be an array of numbers.
    """
    return np.array(2.**np.ceil(np.log2(n))).astype(int)

def smooth(a, deg=2, kind='gaussian', keep_sides=True):
    """Smooth a given vector.

    :param a: Vector to smooth
    
    :param deg: (Optional) Smoothing degree (or kernel
      radius) Must be an integer (default 2).
    
    :param kind: Kind of smoothing function. 'median' or 'mean' are
      self-explanatory. 'gaussian' uses a gaussian function for a
      weighted average. 'gaussian_conv' and 'cos_conv' make use of
      convolution with a gaussian kernel or a cosine
      kernel. Convolution is much faster but less rigorous on the
      edges of the vector (default 'gaussian').

    :params keep_sides: If True, the vector is seen as keeping its
      side values above its real boudaries (If False, the values
      outside the vector are 0. and this creates an undesirable border
      effect when convolving).
    """
    def cos_kernel(deg):
        x = (np.arange((deg*2)+1, dtype=float) - float(deg))/deg * math.pi
        return (np.cos(x) + 1.)/2.

    if keep_sides:
        a_large = np.empty(a.shape[0]+2*deg, dtype=float)
        a_large[deg:a.shape[0]+deg] = a
        a_large[:deg+1] = a[0]
        a_large[-deg:] = a[-1]
        a = np.copy(a_large)
        
    if (kind=="gaussian") or (kind=="gaussian_conv"):
        kernel = np.array(gaussian1d(
            np.arange((deg*2)+1),0.,1.,deg,
            deg/4.*abs(2*math.sqrt(2. * math.log(2.)))))

    if (kind== 'cos_conv'):
        kernel = cos_kernel(deg)

    if (kind=="gaussian_conv") or (kind== 'cos_conv'):
        kernel /= np.sum(kernel)
        smoothed_a = signal.convolve(a, kernel, mode='same')
        if keep_sides:
            return smoothed_a[deg:-deg]
        else:
            return smoothed_a
    
    else:
        smoothed_a = np.copy(a)
        for ii in range(a.shape[0]):
            weights = np.copy(kernel)
            x_min = ii-deg
            if x_min < 0:
                if (kind=="gaussian"):
                    weights = np.copy(kernel[abs(x_min):])
                x_min = 0
            x_max = ii+deg+1
            if x_max > a.shape[0]:
                if (kind=="gaussian"):
                    weights = np.copy(kernel[:-(x_max-a.shape[0])])
                x_max = a.shape[0]
            box = a[x_min:x_max]

            if (kind=="median"):
                smoothed_a[ii] = np.median(box)
            elif (kind=="mean"):
                smoothed_a[ii] = np.mean(box) 
            elif (kind=="gaussian"):
                smoothed_a[ii] = np.average(box, weights=weights)
            else: Tools()._print_error("kind parameter must be 'median', 'mean', 'gaussian', 'gaussian_conv' or 'cos_conv'")
    if keep_sides:
        return smoothed_a[deg:-deg]
    else:
        return smoothed_a


def correct_vector(vector, bad_value=np.nan, deg=3,
                   polyfit=False):
    """Correct a given vector for non valid values by interpolation or
    polynomial fit.

    :param vector: The vector to be corrected.
    
    :param bad_value: (Optional) Bad value to correct (default np.nan)

    :param deg: (Optional) Spline degree or polyfit degree (default 3)

    :param polyfit: (Optional) If True non valid values are guessed
      using a polynomial fit to the data instead of an spline
      interpolation (default False)
    """
  
    n = np.size(vector)
    vector = vector.astype(float)
    if not np.isnan(bad_value):
        vector[np.nonzero(vector == bad_value)] = np.nan
        
    if np.all(np.isnan(vector)):
        Tools()._print_error("The given vector has only bad values")

    # create vectors containing only valid values for interpolation
    x = np.arange(n)[np.nonzero(~np.isnan(vector))]
    new_vector = vector[np.nonzero(~np.isnan(vector))]
 
    if not polyfit:
        finterp = interpolate.UnivariateSpline(x, new_vector, k=deg, s=0)
        return finterp(np.arange(n))
    else:
        coeffs = np.polynomial.polynomial.polyfit(
            x, new_vector, deg, w=None, full=True)
        fit = np.polynomial.polynomial.polyval(
            np.arange(n), coeffs[0])
        vector[np.nonzero(np.isnan(vector))] = fit[np.nonzero(np.isnan(vector))]
        return vector
    

def read_filter_file(filter_file_path):
    """
    Read a file containing a the filter transmission function.

    :param filter_file_path: Path to the filter file.

    :returns: (list of filter wavelength, list of corresponding
      transmission coefficients, minimum edge of the filter, maximum
      edge of the filter) (Both min and max edges can be None if they
      were not recorded in the file)
      
    .. note:: The filter file used must have two colums separated by a
      space character. The first column contains the wavelength axis
      in nm. The second column contains the transmission
      coefficients. Comments are preceded with a #.  Filter edges can
      be specified using the keywords : FILTER_MIN and FILTER_MAX::

        ## ORBS filter file 
        # Author: Thomas Martin <thomas.martin.1@ulaval.ca>
        # Filter name : SpIOMM_R
        # Wavelength in nm | Transmission percentage
        # FILTER_MIN 648
        # FILTER_MAX 678
        1000 0.001201585284
        999.7999878 0.009733387269
        999.5999756 -0.0004460749624
        999.4000244 0.01378122438
        999.2000122 0.002538740868

    """
    filter_file = Tools().open_file(filter_file_path, 'r')
    filter_trans_list = list()
    filter_nm_list = list()
    filter_min = None
    filter_max = None
    for line in filter_file:
        if len(line) > 2:
            line = line.split()
            if '#' not in line[0]: # avoid comment lines
                filter_nm_list.append(float(line[0]))
                filter_trans_list.append(float(line[1]))
            else:
                if "FILTER_MIN" in line:
                    filter_min = float(line[line.index("FILTER_MIN") + 1])
                if "FILTER_MAX" in line:
                    filter_max = float(line[line.index("FILTER_MAX") + 1])
    filter_nm = np.array(filter_nm_list)
    filter_trans = np.array(filter_trans_list)
    # sort coefficients the correct way
    if filter_nm[0] > filter_nm[1]:
        filter_nm = filter_nm[::-1]
        filter_trans = filter_trans[::-1]
        
    return filter_nm, filter_trans, filter_min, filter_max


def get_filter_edges_pix(filter_file_path, correction_factor, step, order,
                         n, filter_min=None, filter_max=None):
    """Return the position in pixels of the edges of a filter
    corrected for the off-axis effect.

    Note that the axis is assumed to be uncalibrated. Spectra are
    generally calibrated but phase vectors are not. So this function
    is best used with phase vectors.
    
    :param filter_file_path: Path to the filter file. If None,
      filter_min and filter_max must be specified.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order.

    :param correction_factor: Correction factor
      (i.e. calibration_map_value / laser_wavelength)
      
    :param n: Number of points of the interpolation axis.

    :param filter_min: (Optional) Edge min of the filter in nm
      (default None).

    :param filter_max: (Optional) Edge max of the filter in nm
      (default None).

    .. seealso:: :py:meth:`utils.read_filter_file`
    """
    if filter_file_path != None:
        (filter_nm, filter_trans,
         filter_min, filter_max) = read_filter_file(filter_file_path)
    elif (filter_min ==  None or filter_max == None):
        Tools()._print_error("filter_min and filter_max must be specified if filter_file_path is None")
    
    nm_axis_ireg = create_nm_axis_ireg(n, step, order,
                                       corr=correction_factor)

    # note that nm_axis_ireg is an reversed axis
    fpix_axis = interpolate.UnivariateSpline(nm_axis_ireg[::-1],
                                             np.arange(n))
    
    try:
        filter_min_pix = int(fpix_axis(filter_min)) # axis is reversed
        filter_max_pix = int(fpix_axis(filter_max)) # axis is reversed
    except Exception:
        filter_min_pix = np.nan
        filter_max_pix = np.nan

    if (filter_min_pix <= 0. or filter_min_pix > n
        or filter_max_pix <= 0. or filter_max_pix > n
        or filter_min_pix >= filter_max_pix):
        filter_min_pix == np.nan
        filter_max_pix == np.nan 

    return filter_min_pix, filter_max_pix

def get_filter_function(filter_file_path, step, order, n):
    """Read a filter file and return its function interpolated over
    the desired number of points. Return also the edges position over
    its axis in pixels.
    
    :param filter_file_path: Path to the filter file.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order.

    :param n: Number of points of the interpolation axis.

    :returns: (interpolated filter function, min edge, max edge). Min
      and max edges are given in pixels over the interpolation axis.

    .. seealso:: :py:meth:`utils.read_filter_file`
    """

    THRESHOLD_COEFF = 0.7

    # Preparing filter function
    (filter_nm, filter_trans,
     filter_min, filter_max) = read_filter_file(filter_file_path)

    # Spectrum wavelength axis creation.
    spectrum_nm_axis = create_nm_axis(n, step, order)

    fnm_axis = interpolate.UnivariateSpline(np.arange(n),
                                            spectrum_nm_axis)
    fpix_axis = interpolate.UnivariateSpline(spectrum_nm_axis,
                                             np.arange(n))

    # Interpolation of the filter function
    interpol_f = interpolate.UnivariateSpline(filter_nm, filter_trans, 
                                              k=5, s=0)
    filter_function = interpol_f(spectrum_nm_axis)

    # Filter function is expressed in percentage. We want it to be
    # between 0 and 1.
    filter_function /= 100.

    # If filter edges were not specified in the filter file look
    # for parts with a transmission coefficient higher than
    # 'threshold_coeff' of the maximum transmission
    if (filter_min == None) or (filter_max == None):
        filter_threshold = ((np.max(filter_function) 
                             - np.min(filter_function)) 
                            * THRESHOLD_COEFF + np.min(filter_function))
        ok_values = np.nonzero(filter_function > filter_threshold)
        filter_min = np.min(ok_values)
        filter_max = np.max(ok_values)
        Tools()._print_warning("Filter edges (%f -- %f nm) determined automatically using a threshold of %f %% transmission coefficient"%(fnm_axis(filter_min), fnm_axis(filter_max), filter_threshold*100.))
    else:
        Tools()._print_msg("Filter edges read from filter file: %f -- %f"%(filter_min, filter_max))
        # filter edges converted to index of the filter vector
        filter_min = int(fpix_axis(filter_min))
        filter_max = int(fpix_axis(filter_max))

    return filter_function, filter_min, filter_max

def correct_map2d(map2d, bad_value=np.nan):
    """Correct a map of values by interpolation along columns.

    The bad value must be specified.

    :param map2d: The map to correct
    
    :param bad_value: (Optional) Value considered as bad (default
      np.nan).
    """
    map2d = np.copy(np.array(map2d).astype(float))
    if bad_value != np.nan:
        map2d[np.nonzero(map2d == bad_value)] = np.nan

    for icol in range(map2d.shape[1]):
        column = np.copy(map2d[:,icol])
        good_vals = np.nonzero(~np.isnan(column))[0]
        bad_vals = np.nonzero(np.isnan(column))[0]
        interp = interpolate.UnivariateSpline(good_vals, column[good_vals], k=5)
        column[bad_vals] = interp(bad_vals)
        map2d[:,icol] = np.copy(column)
        
    return map2d

def polar_map2d(f, n, corner=False, circle=True):
    """
    Map a function over a square matrix in polar coordinates. The
    origin is placed at the center of the map by default.

    :param f: The function to map.

    :param n: Matrix size. Can be a couple of integers (nx, ny).

    :param corner: (Optional) If True, the origin of the coordinates
      becomes the corner (0,0) of the map (default False)

    :param circle: (Optional) If False and if the matrix is not
      squared, the coordinates are those of an ellipsis of the same
      shape as the matrix (default True).
    """
    if np.size(n) == 2:
        nx = n[0]
        ny = n[1]
        n = max(nx, ny)
    else:
        nx = n
        ny = n
    hdimx = nx/2. - 0.5
    hdimy = ny/2. - 0.5
    if not corner:
        X,Y = np.mgrid[-hdimx:hdimx+1, -hdimy:hdimy+1]
    else:
        X,Y = np.mgrid[0:nx, 0:ny]
    if circle:
        R = np.sqrt(X**2+Y**2)
    else:
        R = np.sqrt((X/(float(nx)/n))**2+(Y/(float(ny)/n))**2)
    return np.array(map(f,R))
    

##################################################
#### FITTING  ####################################
##################################################
def sinc1d(x, h, a, dx, fwhm):
    """Return a 1D sinc 
    :param x: Array giving the positions where the function is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != float:
        x = x.astype(float)
    return cutils.sinc1d(x, float(h), float(a), float(dx), float(fwhm))

def gaussian1d(x,h,a,dx,fwhm):
    """Return a 1D gaussian given a set of parameters.

    :param x: Array giving the positions where the gaussian is evaluated
    :param h: Height
    :param a: Amplitude
    :param dx: Position of the center
    :param w: FWHM, :math:`\\text{FWHM} = \\text{Width} \\times 2 \\sqrt{2 \\ln 2}`
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != float:
        x = x.astype(float)
    return cutils.gaussian1d(x, float(h), float(a), float(dx), float(fwhm))

def fit_lines_in_vector(vector, lines, fwhm_guess=3.5,
                        cont_guess=None, shift_guess=0.,
                        fix_cont=False,
                        fix_fwhm=False, cov_fwhm=True, cov_pos=True,
                        reguess_positions=False,
                        return_fitted_vector=False, fit_tol=1e-3,
                        no_absorption=False, poly_order=0,
                        fmodel='gaussian', sig_noise=None,
                        interpolation_params=None, signal_range=None):

    """Fit multiple gaussian shaped emission lines in a spectrum vector.

    :param vector: Vector to fit

    :param lines: Positions of the lines in channels

    :param fwhm_guess: (Optional) Initial guess on the lines FWHM
      (default 3.5).

    :param cont_guess: (Optional) Initial guess on the continuum
      (default None). Must be a tuple of poly_order + 1 values ordered
      with the highest orders first.

    :param shift_guess: (Optional) Initial guess on the global shift
      of the lines (default 0.).

    :param fix_cont: (Optional) If True, continuum is fixed to the
      initial guess (default False).

    :param fix_fwhm: (Optional) If True, FWHM value is fixed to the
      initial guess (default False).

    :param cov_fwhm: (Optional) If True FWHM is considered to be the
      same for all lines and become a covarying parameter (default
      True).

    :param cov_pos: (Optional) If True the estimated relative
      positions of the lines (the lines parameter) are considered to
      be exact and only need to be shifted. Positions are thus
      covarying. Very useful but the initial estimation of the line
      relative positions must be very precise (default False).

    :param reguess_positions: (Optional) If True, positions are
      guessed again. Useful if the given estimations really are rough
      ones. Note that this must not be used with cov_pos set to True
      (default False).

    :param return_fitted_vector: (Optional) If True Fitted vector is
      returned.

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-2).

    :param no_absorption: (Optional) If True, no negative amplitude
      are returned (default False).

    :param poly_order: (Optional) Order of the polynomial used to fit
      continuum. Use high orders carefully (default 0).

    :param fmodel: (Optional) Fitting model. Can be 'gaussian', 
      'sinc' or 'sinc2' (default 'gaussian').

    :param sig_noise: (Optional) Noise standard deviation guess. If
      None noise value is guessed but the gaussian FWHM must not
      exceed half of the sampling interval (default None).

    :param interpolation_params: (Optional) Must be a tuple [step,
      order]. Interpolate data before fitting when data has been
      previously interpolated from an irregular wavelength axis to a
      regular one.

    :param signal_range: (Optional) a tuple (x_min, x_max) giving the
      lowest and highest channel numbers containing signal.

    :return: a dictionary containing:
    
      * lines parameters [key: 'lines-params'] Lines parameters are
        given as an array of shape (lines_nb, 4). The order of the 4
        parameters for each lines is [height at the center of the
        line, ampitude, position, fwhm].
      
      * lines parameters errors [key: 'lines-params-err']

      * residual [key: 'residual']
      
      * chi-square [key: 'chi-square']

      * reduced chi-square [key: 'reduced-chi-square']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont-params']

      * and optionally the fitted vector [key: 'fitted-vector']
        depending on the option return_fitted_vector.

      
    """
    def params_arrays2vect(lines_p, lines_p_mask,
                           cov_p, cov_p_mask,
                           cont_p, cont_p_mask):
        
        free_p = list(lines_p[np.nonzero(lines_p_mask)])
        free_p += list(cov_p[np.nonzero(cov_p_mask)])
        free_p += list(cont_p[np.nonzero(cont_p_mask)])
        
        fixed_p = list(lines_p[np.nonzero(~lines_p_mask)])
        fixed_p += list(cov_p[np.nonzero(~cov_p_mask)])
        fixed_p += list(cont_p[np.nonzero(~cont_p_mask)])
        
        return free_p, fixed_p

    def params_vect2arrays(free_p, fixed_p, lines_p_mask,
                           cov_p_mask, cont_p_mask):

        cont_p = np.empty_like(cont_p_mask, dtype=float)
        free_cont_p_nb = np.sum(cont_p_mask)
        if free_cont_p_nb > 0:
            cont_p[np.nonzero(cont_p_mask)] = free_p[
                -free_cont_p_nb:]
            free_p = free_p[:-free_cont_p_nb]
        if free_cont_p_nb < np.size(cont_p_mask):
            cont_p[np.nonzero(~cont_p_mask)] = fixed_p[
                -(np.size(cont_p_mask) - free_cont_p_nb):]
            fixed_p = fixed_p[
                :-(np.size(cont_p_mask) - free_cont_p_nb)]
            
        cov_p = np.empty_like(cov_p_mask, dtype=float)
        free_cov_p_nb = np.sum(cov_p_mask)
        if free_cov_p_nb > 0:
            cov_p[np.nonzero(cov_p_mask)] = free_p[
                -free_cov_p_nb:]
            free_p = free_p[:-free_cov_p_nb]
        if free_cov_p_nb < np.size(cov_p_mask):
            
            cov_p[np.nonzero(~cov_p_mask)] = fixed_p[
                -(np.size(cov_p_mask) - free_cov_p_nb):]
            fixed_p = fixed_p[
                :-(np.size(cov_p_mask) - free_cov_p_nb)]


        lines_p = np.empty_like(lines_p_mask, dtype=float)
        lines_p[np.nonzero(lines_p_mask)] = free_p
        lines_p[np.nonzero(~lines_p_mask)] = fixed_p
        
        return lines_p, cov_p, cont_p
    
    def model(n, lines_p, cont_p, fmodel, axis):
        mod = np.zeros(n, dtype=float)
        # continuum
        mod += np.polyval(cont_p, np.arange(n))
        for iline in range(lines_p.shape[0]):
            if fmodel == 'sinc':
                mod += sinc1d(
                    axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
            elif fmodel == 'sinc2':
               mod += np.sqrt(sinc1d(
                   axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                   lines_p[iline, 2])**2.)
            else:
                mod += gaussian1d(
                    axis, 0., lines_p[iline, 0], lines_p[iline, 1],
                    lines_p[iline, 2])
        
        return mod

    def add_shift(lines_pos, shift, nm_axis_ireg):
        if nm_axis_ireg is not None:
            center = int(nm_axis_ireg.shape[0] / 2.)
            # delta_lambda is evaluated at the center of the axis
            # delta_v must be a constant but the corresponding
            # delta_lambda is not: delta_v = delta_lambda / lambda
            delta_lambda = (pix2nm(nm_axis_ireg, center+shift)
                            - nm_axis_ireg[center])
            delta_v = delta_lambda / nm_axis_ireg[center]

            lines_nm = pix2nm(nm_axis_ireg, lines_pos)
            lines_pos = nm2pix(nm_axis_ireg, lines_nm + delta_v * lines_nm)
        else:
            lines_pos += shift
        return lines_pos

    def diff(free_p, fixed_p, lines_p_mask, cov_p_mask,
             cont_p_mask, data, sig, fmodel, axis, nm_axis_ireg):
        lines_p, cov_p, cont_p = params_vect2arrays(free_p, fixed_p,
                                                    lines_p_mask,
                                                    cov_p_mask, cont_p_mask)
        lines_p[:,2] += cov_p[0] # + FWHM
        
        # + SHIFT
        lines_p[:,1] = add_shift(lines_p[:,1], cov_p[1], nm_axis_ireg)
        
        data_mod = model(np.size(data), lines_p, cont_p, fmodel, axis)
        return (data - data_mod) / sig

    MIN_LINE_SIZE = 5 # Minimum size of a line whatever the guessed
                      # fwhm can be.

    x = np.copy(vector)
    
    if np.all(np.isnan(x)):
        return []
    
    if np.any(np.iscomplex(x)):
        x = x.real
        Tools()._print_warning(
            'Complex vector. Only the real part will be fitted')

    lines = np.array(lines, dtype=float)
    lines = lines[np.nonzero(lines > 0.)]

    lines_nb = np.size(lines)
    line_size = int(max(math.ceil(fwhm_guess * 3.5), MIN_LINE_SIZE))
    
    # only 3 params max for each line, cont is defined as a N order
    # polynomial (N+1 additional parameters)
    lines_p = np.zeros((lines_nb, 3), dtype=float)

    # vector interpolation
    if interpolation_params is not None:
        step = interpolation_params[0]
        order = interpolation_params[1]
        nm_axis_ireg = (create_nm_axis_ireg(
            x.size, step, order)[::-1])
        nm_axis = create_nm_axis(x.size, step, order)
        nm_axis_orig = np.copy(nm_axis)
        nm_axis_ireg_orig = np.copy(nm_axis_ireg)

        x = interpolate_axis(x, nm_axis_ireg, 5,
                             old_axis=nm_axis,
                             fill_value=np.nan)

        # convert lines position from regular axis to irregular axis
        lines = nm2pix(nm_axis_ireg, pix2nm(nm_axis, lines))
        if signal_range is not None:
            signal_range = nm2pix(nm_axis_ireg, pix2nm(
                nm_axis, signal_range))
    else:
        nm_axis_ireg = None

    # remove parts out of signal range
    x_size_orig = np.size(x)
    if signal_range is not None:
        signal_range = np.array(signal_range).astype(int)
        if np.min(signal_range) >= 0 and np.max(signal_range < np.size(x)):
            x = x[np.min(signal_range):np.max(signal_range)]
            if interpolation_params is not None:
                nm_axis_ireg = nm_axis_ireg[
                    np.min(signal_range):np.max(signal_range)]
                nm_axis = nm_axis[np.min(signal_range):np.max(signal_range)]
            lines -= np.min(signal_range)
        else:
            raise Exception('Signal range must be a tuple (min, max) with min >= 0 and max < {:d}'.format(np.size(x)))

    # axis
    axis = np.arange(x.shape[0])

    # check nans
    if np.any(np.isinf(x)) or np.any(np.isnan(x)) or (np.min(x) == np.max(x)):
        return []
    
    ## Guess ##
    noise_vector = np.copy(x)
    for iline in range(lines_nb):
        line_center = lines[iline]
        iz_min = int(line_center - line_size/2.)
        iz_max = int(line_center + line_size/2.) + 1
        if iz_min < 0: iz_min = 0
        if iz_max > x.shape[0] - 1: iz_max = x.shape[0] - 1
        line_box = x[iz_min:iz_max]

        # max
        lines_p[iline, 0] = np.max(line_box) - np.median(x)

        # position
        if reguess_positions:
            lines_p[iline, 1] = (np.sum(line_box
                                             * np.arange(line_box.shape[0]))
                                      / np.sum(line_box)) + iz_min
        
        # remove line from noise vector
        noise_vector[iz_min:iz_max] = np.nan

    
    if not reguess_positions:
        lines_p[:, 1] = np.array(lines)

    lines_p[:, 2] = fwhm_guess
        
    # polynomial guess of the continuum
    if (cont_guess == None) or (np.size(cont_guess) != poly_order + 1):
        if (cont_guess != None) and np.size(cont_guess) != poly_order + 1:
            Tools()._print_warning('Bad continuum guess shape')
        if poly_order > 0:
            w = np.ones_like(noise_vector)
            nans = np.nonzero(np.isnan(noise_vector))
            w[nans] = 0.
            noise_vector[nans] = 0.
            cont_guess = np.polynomial.polynomial.polyfit(
                np.arange(noise_vector.shape[0]), noise_vector, poly_order, w=w)
            cont_fit = np.polynomial.polynomial.polyval(
                np.arange(noise_vector.shape[0]), cont_guess)
            cont_guess = cont_guess[::-1]
            noise_value = robust_std(noise_vector - cont_fit)
        else:
            cont_guess = [robust_median(noise_vector)]
            noise_value = robust_std(noise_vector)
    else:
        cont_fit = np.polyval(cont_guess, np.arange(noise_vector.shape[0]))
        noise_value = robust_std(noise_vector - cont_fit)
    
    if sig_noise != None:
        noise_value = sig_noise

    #### PARAMETERS = LINES_PARAMS (N*(3-COV_PARAMS_NB)),
    ####              COV_PARAMS (FWHM_COEFF, SHIFT),
    ####              CONTINUUM COEFFS

    # define cov params
    cov_p = np.empty(2, dtype=float)
    cov_p[0] = 0. # COV FWHM COEFF
    cov_p[1] = shift_guess # COV SHIFT
    cov_p_mask = np.zeros(2, dtype=bool)
    if cov_fwhm and not fix_fwhm: cov_p_mask[0] = True
    if cov_pos: cov_p_mask[1] = True
        
    # define lines params mask
    lines_p_mask = np.ones_like(lines_p, dtype=bool)
    if fix_fwhm or cov_fwhm: lines_p_mask[:,2] = False
    if cov_pos: lines_p_mask[:,1] = False
    
    # define continuum params
    cont_p = np.array(cont_guess)
    cont_p_mask = np.ones_like(cont_p, dtype=bool)
    if fix_cont: cont_p_mask.fill(False)

    free_p, fixed_p = params_arrays2vect(
        lines_p, lines_p_mask,
        cov_p, cov_p_mask,
        cont_p, cont_p_mask)
    
    ### FIT ###
    fit = optimize.leastsq(diff, free_p,
                           args=(fixed_p, lines_p_mask,
                                 cov_p_mask, cont_p_mask,
                                 x, noise_value, fmodel, axis, nm_axis_ireg),
                           maxfev=5000, full_output=True,
                           xtol=fit_tol)

    ### CHECK FIT RESULTS ###
    if fit[-1] <= 4:
        returned_data = dict()
        last_diff = fit[2]['fvec']
        lines_p, cov_p, cont_p = params_vect2arrays(
            fit[0], fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)

        # add cov to lines params
        full_lines_p = np.empty((lines_nb, 4), dtype=float)
        full_lines_p[:,0] = np.polyval(cont_p, lines_p[:,1])
        full_lines_p[:,1:] = lines_p
        full_lines_p[:,2] = add_shift(full_lines_p[:,2], cov_p[1], nm_axis_ireg)
        full_lines_p[:,3] += cov_p[0] # + FWHM_COEFF
        
        # check and correct
        for iline in range(lines_nb):
            if no_absorption:
                if full_lines_p[iline, 1] < 0.:
                    full_lines_p[iline, 1] = 0.
            if (full_lines_p[iline, 2] < 0.
                or full_lines_p[iline, 2] > x.shape[0]):
                full_lines_p[iline, :] = np.nan

        # compute fitted vector
        fitted_vector = np.empty(x_size_orig, dtype=float)
        fitted_vector.fill(np.nan)
        fitted_vector[np.min(signal_range):
                      np.max(signal_range)] = model(
            x.shape[0], full_lines_p[:,1:], cont_p, fmodel, axis)
        if return_fitted_vector:
            returned_data['fitted-vector'] = fitted_vector

        # correct shift for signal range
        if signal_range is not None:            
            full_lines_p[:,2] += np.min(signal_range)
        
        returned_data['lines-params'] = full_lines_p
        returned_data['cont-params'] = cont_p
       
        # compute reduced chi square
        chisq = np.sum(last_diff**2.)
        red_chisq = chisq / (np.size(x) - np.size(free_p))
        returned_data['reduced-chi-square'] = red_chisq
        returned_data['chi-square'] = chisq
        returned_data['residual'] = last_diff * noise_value

        # compute least square fit errors
        cov_x = fit[1]
        if cov_x is None: # fit has stopped because of max iteration
            return []
        cov_x *= returned_data['reduced-chi-square']
        cov_diag = np.sqrt(np.abs(
            np.array([cov_x[i,i] for i in range(cov_x.shape[0])])))
        fixed_p = np.array(fixed_p)
        fixed_p.fill(0.)
        lines_err, cov_err, cont_err = params_vect2arrays(
            cov_diag, fixed_p, lines_p_mask, cov_p_mask, cont_p_mask)
        ls_fit_errors = np.empty_like(full_lines_p)
        ls_fit_errors[:,1:] = lines_err
        if cov_p_mask[0]: ls_fit_errors[:,3] = cov_err[0]
        if cov_p_mask[1]: ls_fit_errors[:,2] = cov_err[1]
        ls_fit_errors[:,0] = math.sqrt(np.sum(cont_err**2.))

        returned_data['lines-params-err'] = ls_fit_errors
        returned_data['cont-params-err'] = cont_err

        # compute SNR
        # recompute noise value in case the continnum is not linear
        # (noise can be a lot overestimated in this case)
        noise_value = np.std(x - fitted_vector[np.min(signal_range):
                                               np.max(signal_range)])
        returned_data['snr'] = full_lines_p[:,1] / noise_value

        # Compute analytical errors [from Minin & Kamalabadi, Applied
        # Optics, 2009]. Note that p4 = fwhm / (2*sqrt(ln(2))). These
        # errors must be near the least square errors except for the
        # background because the background modelized by the authors
        # has nothing to see with the model we use here. In general
        # lesat squares error are thus a better estimate.
    
        fit_errors = np.empty_like(full_lines_p)
        p4 = full_lines_p[:,3] / (2. * math.sqrt(math.log(2.)))
        Q = (3. * p4 * math.sqrt(math.pi)
             / (x.shape[0] * math.sqrt(2)))
        Qnans = np.nonzero(np.isnan(Q))
        Q[Qnans] = 0.
        
        if np.all(Q <= 1.):
            fit_errors[:,0] = (
                noise_value * np.abs((1./x.shape[0]) * 1./(1. - Q))**0.5)
            
            fit_errors[:,1] = (
                noise_value * np.abs((3./2.)
                               * math.sqrt(2)/(p4 * math.sqrt(math.pi))
                               * (1. - (8./9.)*Q)/(1. - Q))**0.5)
            fit_errors[:,2] = (
                noise_value * np.abs(p4 * math.sqrt(2.)
                               / (full_lines_p[:,1]**2.
                                  * math.sqrt(math.pi)))**0.5)
            fit_errors[:,3] = (
                noise_value * np.abs((3./2.)
                               * (4. * p4 * math.sqrt(2.)
                                  / (3.*full_lines_p[:,1]**2.
                                     * math.sqrt(math.pi)))
                               * ((1.-(2./3.)*Q)/(1.-Q)))**0.5)
            
            fit_errors[Qnans,:] = np.nan
        
        else:
            fit_errors.fill(np.nan)
            
        returned_data['lines-params-err-an'] = fit_errors
        
        # interpolate back parameters
        if interpolation_params is not None:
            lines_dx = returned_data['lines-params'][:,2]
            
            lines_dx = nm2pix(
                nm_axis_orig, pix2nm(nm_axis_ireg_orig, lines_dx))
            returned_data['lines-params'][:,2] = lines_dx

            if return_fitted_vector:
                returned_data['fitted-vector'] = interpolate_axis(
                    returned_data['fitted-vector'], nm_axis_orig, 5,
                    old_axis=nm_axis_ireg_orig, fill_value=np.nan)
                   
        return returned_data 
    else:
        return []

def get_open_fds():
    """Return the number of open file descriptors

    .. warning:: Only works on UNIX-like OS

    .. note:: This is a useful debugging function that has been taken from: http://stackoverflow.com/questions/2023608/check-what-files-are-open-in-python
    """
    import resource
    pid = os.getpid()
    procs = subprocess.check_output(
        ["lsof", '-w', '-Ff', "-p", str( pid )])
    return len(filter( 
        lambda s: s and s[0] == 'f' and s[1:].isdigit(),
        procs.split( '\n' ))), resource.getrlimit(resource.RLIMIT_NOFILE)
    
##################################################
#### IMAGE TOOLS #################################
##################################################

def transform_frame(frame, x_min, x_max, y_min, y_max, 
                    d, rc, zoom_factor, interp_order, 
                    mask=None, fill_value=np.nan):
    """Transform one frame or a part of it using transformation
    coefficients.

    :param frame: Frame to transform
    
    :param x_min: Lower x boundary of the frame to transform
    
    :param x_max: Upper x boundary of the frame to transform
    
    :param y_min: Lower y boundary of the frame to transform
    
    :param y_max: Upper y boundary of the frame to transform
    
    :param d: Transformation coefficients [dx, dy, dr, da, db]
    
    :param rc: Rotation center of the frame [rc_x, rc_y]
    
    :param zoom_factor: Zoom on the image
    
    :param interp_order: Interpolation order
    
    :param mask: (Optional) If a mask frame is passed is is also
      transformed (default None).

    :param fill_value: (Optional) Fill value for extrapolated points
      (default np.nan).
    """
    def trans(coords, d, rc, zoom_factor):
        return cutils.transform_A_to_B(
            coords[0], coords[1],
            d[0], d[1], d[2], d[3], d[4], rc[0],
            rc[1], zoom_factor)
       
    if frame.dtype != np.dtype(float):
        frame = frame.astype(float)
    
    if mask != None:
        mask = mask.astype(float)
    
    frame = ndimage.interpolation.geometric_transform(
        frame, trans, extra_arguments=(d, rc, zoom_factor),
        order=interp_order, mode='constant', cval=fill_value)

    if mask != None:
        mask = ndimage.interpolation.geometric_transform(
            mask, trans, extra_arguments=(d, rc, zoom_factor),
            order=interp_order, mode='constant', cval=fill_value)

    if mask != None:
        return (frame[x_min:x_max, y_min:y_max],
                mask[x_min:x_max, y_min:y_max])
    else:
        return frame[x_min:x_max, y_min:y_max]


def shift_frame(frame, dx, dy, x_min, x_max, 
                y_min, y_max, order, fill_value=np.nan):
    """Return a shifted frame wit the same dimensions.

    :param frame: Two dimensions array to be shifted
    
    :param dx: Shift value along the axis 0
    
    :param dy: Shift value along the axis 1

    :param x_min, x_max, y_min, y_max: Boundaries of the region to be
      shifted.

    :param order: interpolation order.

    :param fill_value (Optional): Value of the extrapolated points
      (default np.nan).

    .. note:: To avoid spline interpolation defects around
       stars use order 1 (linear interpolation).
    """
    z = frame[x_min:x_max, y_min:y_max]
    interp = ndimage.interpolation.shift(z, [-dx, -dy], order=order, 
                                         mode='constant', cval=fill_value, 
                                         prefilter=True)
    return interp
        
def high_pass_image_filter(im):
    """Return a high pass filtered image.

    :param im: Image to filter
    """
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]
    kernel = np.array(kernel, dtype=float) / 9.
    return signal.convolve(im, kernel, mode='same')

def high_pass_diff_image_filter(im, deg=1):
    """Return a high pass filtered image using the method of low pass
    diffrence filtering given by Mighell (1999).

    :param im: Image to filter

    :param deg: (Optional) Radius of the kernel of the low pass
      filter. Must be > 0 (default 2).
    """
    lp_im = low_pass_image_filter(im, deg=deg)
    return lp_im - scipy.ndimage.filters.median_filter(lp_im, size=(5,5))

def low_pass_image_filter(im, deg):
    """Return a low pass filtered image using a gaussian kernel.
    
    :param im: Image to filter
    
    :param deg: Radius of the kernel. Must be > 0.
    """
    if not deg > 0:
        Tools()._print_error('Kernel degree must be > 0')

    if 2 * deg >= max(im.shape):
        Tools()._print_error('Kernel degree is too high given the image size')

    return cutils.low_pass_image_filter(np.copy(im).astype(float), int(deg))


def fft_filter(a, cutoff_coeff, width_coeff=0.2, filter_type='high_pass'):
    """
    Simple lowpass or highpass FFT filter (high pass or low pass)

    Filter shape is a gaussian.
    
    :param a: Vector to filter

    :param cutoff_coeff: Coefficient defining the position of the cut
      frequency (Cut frequency = cut_coeff * vector length)

    :param width_coeff: (Optional) Coefficient defining the width of
      the smoothed part of the filter (width = width_coeff * vector
      length) (default 0.2)

    :param filter_type: (Optional) Type of filter to use. Can be
      'high_pass' or 'low_pass'.
    """
    if cutoff_coeff < 0. or cutoff_coeff > 1.:
        Tools()._print_error('cutoff_coeff must be between 0. and 1.')
    if width_coeff < 0. or width_coeff > 1.:
        Tools()._print_error('width_coeff must be between 0. and 1.')
        
    if filter_type == 'low_pass':
        lowpass = True
    elif filter_type == 'high_pass':
        lowpass = False
    else:
        Tools()._print_error(
            "Bad filter type. Must be 'low_pass' or 'high_pass'")
    
    return cutils.fft_filter(a, cutoff_coeff, width_coeff, lowpass)

##################################################
#### SPECTRUM COMPUTATION  #######################
##################################################

def raw_fft(x, apod=None, inverse=False):
    """Compute the raw FFT of a vector
    
    :param x: Interferogram.
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.norton_beer_window` (default None)

    :param inverse: (Optional) If True compute the inverse FFT
      (default False).
    """
    x = np.copy(x)
    windows = ['1.1', '1.2', '1.3', '1.4', '1.5',
               '1.6', '1.7', '1.8', '1.9', '2.0']
    N = x.shape[0]
    
    # mean substraction
    x -= np.mean(x)
    
    # apodization
    if apod in windows:
        x *= norton_beer_window(apod, N)
    elif apod != None:
        Tools()._print_error("Unknown apodization function try %s"%
                             str(windows))
        
    # zero padding
    zv = np.zeros(N*2, dtype=float)
    zv[int(N/2):int(N/2)+N] = x

    # zero the centerburst
    zv = np.roll(zv, zv.shape[0]/2)
    
    # FFT
    if not inverse:
        x_fft = np.abs((np.fft.fft(zv))[:N])
    else:
        x_fft = np.abs((np.fft.ifft(zv))[:N])
    return x_fft
     
def cube_raw_fft(x, apod=None):
    """Compute the raw FFT of a cube (the last axis
    beeing the interferogram axis)

    :param x: Interferogram cube
    
    :param apod: (Optional) Apodization function used. See
      :py:meth:`utils.norton_beer_window` (default None)
    """
    x = np.copy(x)
    windows = ['1.1', '1.2', '1.3', '1.4', '1.5',
               '1.6', '1.7', '1.8', '1.9', '2.0']
    N = x.shape[-1]
    # mean substraction
    x = (x.T - np.mean(x, axis=-1)).T
    # apodization
    if apod in windows:
        x *= norton_beer_window(apod, N)
    elif apod != None:
        Tools()._print_error("Unknown apodization function try %s"%
                             str(windows))
    # zero padding
    zv_shape = np.array(x.shape)
    zv_shape[-1] = N*2
    zv = np.zeros(zv_shape)
    zv[:,int(N/2):int(N/2)+N] = x
    # FFT
    return np.abs((np.fft.fft(zv))[::,:N])

def norton_beer_window(fwhm='1.6', n=1000):
    """
    Return an extended Norton-Beer window function (see [NAY2007]_).

    Returned window is symmetrical.
    
    :param fwhm: FWHM relative to the sinc function. Must be: 1.1,
       1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 or 2.0. (default '1.6')
       
    :param n: Number of points (default 1000)

    .. note:: Coefficients of the extended Norton-Beer functions
       apodizing functions [NAY2007]_ :
    
       ==== ======== ========= ======== ======== ======== ======== 
       FWHM    C0       C1        C2       C4       C6       C8
       ---- -------- --------- -------- -------- -------- -------- 
       1.1  0.701551 -0.639244 0.937693 0.000000 0.000000 0.000000
       1.2  0.396430 -0.150902 0.754472 0.000000 0.000000 0.000000
       1.3  0.237413 -0.065285 0.827872 0.000000 0.000000 0.000000
       1.4  0.153945 -0.141765 0.987820 0.000000 0.000000 0.000000
       1.5  0.077112 0.000000  0.703371 0.219517 0.000000 0.000000
       1.6  0.039234 0.000000  0.630268 0.234934 0.095563 0.000000
       1.7  0.020078 0.000000  0.480667 0.386409 0.112845 0.000000
       1.8  0.010172 0.000000  0.344429 0.451817 0.193580 0.000000
       1.9  0.004773 0.000000  0.232473 0.464562 0.298191 0.000000
       2.0  0.002267 0.000000  0.140412 0.487172 0.256200 0.113948
       ==== ======== ========= ======== ======== ======== ========

    .. [NAY2007] Naylor, D. A., & Tahic, M. K. (2007). Apodizing
       functions for Fourier transform spectroscopy. Journal of the
       Optical Society of America A.
    """
    
    norton_beer_coeffs = [
        [1.1, 0.701551, -0.639244, 0.937693, 0., 0., 0., 0., 0., 0.],
        [1.2, 0.396430, -0.150902, 0.754472, 0., 0., 0., 0., 0., 0.],
        [1.3, 0.237413, -0.065285, 0.827872, 0., 0., 0., 0., 0., 0.],
        [1.4, 0.153945, -0.141765, 0.987820, 0., 0., 0., 0., 0., 0.],
        [1.5, 0.077112, 0., 0.703371, 0., 0.219517, 0., 0., 0., 0.],
        [1.6, 0.039234, 0., 0.630268, 0., 0.234934, 0., 0.095563, 0., 0.],
        [1.7, 0.020078, 0., 0.480667, 0., 0.386409, 0., 0.112845, 0., 0.],
        [1.8, 0.010172, 0., 0.344429, 0., 0.451817, 0., 0.193580, 0., 0.],
        [1.9, 0.004773, 0., 0.232473, 0., 0.464562, 0., 0.298191, 0., 0.],
        [2.0, 0.002267, 0., 0.140412, 0., 0.487172, 0., 0.256200, 0., 0.113948]]

    fwhm_list = ['1.1', '1.2', '1.3', '1.4', '1.5',
                 '1.6', '1.7', '1.8', '1.9', '2.0']
    if fwhm in fwhm_list:
        fwhm_index = fwhm_list.index(fwhm)
    else:
        Tools()._print_error("Bad extended Norton-Beer window FWHM. Must be in : " + str(fwhm_list))

    x = np.linspace(-1., 1., n)

    nb = np.zeros_like(x)
    for index in range(9):
        nb += norton_beer_coeffs[fwhm_index][index+1]*(1. - x**2)**index
    return nb

def learner95_window(n):
    """Return the apodization function described in Learner et al.,
    J. Opt. Soc. Am. A, 12, (1995).

    This function is closely related to the minimum four-term
    Blackman-Harris window.

    Returned window is symmetrical.
    
    :param n: Number of points.
    """
    x = np.linspace(-1., 1., n)
    return (0.355766
            + 0.487395 * np.cos(math.pi*x)
            + 0.144234 * np.cos(2.*math.pi*x)
            + 0.012605 * np.cos(3.*math.pi*x))

def create_nm_axis(n, step, order, nm_max=None):
    """Create a regular wavelength axis in nm.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param nm_max: (Optional) Must be given if order is 0 (default None)
    
    .. warning:: If the order is equal to zero, nm_max (the maximum
      wavelength observed) must be specified because in cannot be
      defined using only the the step and order arguments.
    """
    nm_min = 2. * step / (order + 1.)
    if (order > 0): 
        nm_max = 2. * step / order
    elif (nm_max == None):
        raise Exception("If order is 0, 'nm_max' must be specified")
    return np.linspace(nm_min, nm_max, n)

def create_cm1_axis(n, step, order, corr=1.):
    """Create a regular wavenumber axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    cm1_min = float(order) / (2.* step) * corr * 1e7
    cm1_max = float(order + 1) / (2. * step) * corr * 1e7
    return np.linspace(cm1_min, cm1_max, n)
    
    
def create_nm_axis_ireg(n, step, order, nm_max=None, corr=1.):
    """Create an irregular wavelength axis from the regular wavenumber
    axis in cm-1.

    :param n: Number of steps on the axis
    :param step: Step size in nm
    :param order: Folding order
    :param nm_max: (Optional) Must be given if order is 0 (default None)
    :param corr: (Optional) Coefficient of correction (default 1.)
    """
    if nm_max == None:
        if order > 0:
            cm1_min = float(order) / (2.* step) * corr
        else:
            raise Exception("If order is 0, 'nm_max' must be specified")
    else:
        cm1_min = 1. / nm_max * corr
        
    cm1_max = float(order + 1) / (2. * step) * corr
    cm1_axis = np.linspace(cm1_min, cm1_max, n)
    return (1. / cm1_axis)
    
def pix2nm(nm_axis, pix):
     """Convert a pixel position to a wavelength in nm given an axis
     in nm

     :param nm_axis: Axis in nm
     
     :param pix: Pixel position
     """
     f = interpolate.interp1d(np.arange(nm_axis.shape[0]), nm_axis,
                              bounds_error=False, fill_value=np.nan)
     
     return f(pix)

def nm2pix(nm_axis, nm):
     """Convert a wavelength in nm to a pixel position given an axis
     in nm

     :param nm_axis: Axis in nm
     
     :param nm: Wavelength in nm
     """
     f = interpolate.interp1d(nm_axis, np.arange(nm_axis.shape[0]),
                              bounds_error=False, fill_value=np.nan)
     return f(nm)

def nm2cm1(nm):
    """Convert a wavelength in nm to a wavenumber in cm-1.

    :param nm: wavelength i nm
    """
    return 1e7 / np.float(nm)

def cm12nm(cm1):
    """Convert a wavenumber in cm-1 to a wavelength in nm.

    :param cm1: wavenumber in cm-1
    """
    return 1e7 / np.float(cm1)

def cm12pix(cm1_axis, cm1):
     """Convert a wavenumber in cm-1 to a pixel position given an axis
     in cm-1.

     :param cm1_axis: Axis in cm-1
     
     :param cm1: Wavenumber in cm-1
     """
     f = interpolate.interp1d(cm1_axis, np.arange(cm1_axis.shape[0]),
                              bounds_error=False, fill_value=np.nan)
     return f(cm1)

def fwhm_nm2cm1(fwhm_nm, nm):
    """Convert a FWHM in nm to a FWHM in cm-1.
    
    The central wavelength in nm of the line must also be given

    :param fwhm_nm: FWHM in nm
    
    :param nm: Wavelength in nm where the FWHM is evaluated
    """
    return 1e7 * fwhm_nm / nm**2.

def polyfit1d(a, deg, w=None, return_coeffs=False):
    """Fit a polynomial to a 1D vector.
    
    :param a: Vector to fit
    
    :param deg: Fit degree
    
    :param return_coeffs: (Optional) If True return fit coefficients
      as returned by numpy.polynomial.polynomial.polyfit() (default
      False).

    :param w: (Optional) If not None, weights to apply to the
      fit. Must have the same shape as the vector to fit (default
      None)
    """
    n = a.shape[0]
    coeffs = np.polynomial.polynomial.polyfit(
        np.arange(n), a, deg, w=w, full=True)
    fit = np.polynomial.polynomial.polyval(
        np.arange(n), coeffs[0])
    if not return_coeffs:
        return fit
    else:
        return fit, coeffs

def interpolate_map(calibration_map, dimx, dimy):
    """Interpolate 2D data map.
    
    :param calibration_map: Map
    
    :param dimx: X dimension of the result
    
    :param dimy: Y dimension of the result
    """
    x_map = np.arange(calibration_map.shape[0])
    y_map = np.arange(calibration_map.shape[1])
    x_int = np.linspace(0, calibration_map.shape[0], dimx,
                        endpoint=False)
    y_int = np.linspace(0, calibration_map.shape[1], dimy,
                        endpoint=False)
    interp = interpolate.RectBivariateSpline(x_map, y_map,
                                             calibration_map)
    return interp(x_int, y_int)
    
def interpolate_size(a, size, deg):
    """Change size of a vector by interpolation

    :param a: vector to interpolate
    
    :param size: New size of the vector
    
    :param deg: Interpolation degree
    """
    if a.dtype != np.complex:
        f = interpolate.UnivariateSpline(np.arange(a.shape[0]), a, k=deg, s=0)
        return f(np.arange(size)/float(size - 1L) * float(a.shape[0] - 1L))
    else:
        result = np.empty(size, dtype=complex)
        f_real = interpolate.UnivariateSpline(
            np.arange(a.shape[0]), a.real, k=deg, s=0)
        result.real = f_real(
            np.arange(size)/float(size - 1L) * float(a.shape[0] - 1L))
        f_imag = interpolate.UnivariateSpline(
            np.arange(a.shape[0]), a.imag, k=deg, s=0)
        result.imag = f_imag(
            np.arange(size)/float(size - 1L) * float(a.shape[0] - 1L))
        return result
        
def interpolate_axis(a, new_axis, deg, old_axis=None, fill_value=np.nan):
    """Interpolate a vector along a new axis.

    :param a: vector to interpolate
    
    :param new_axis: Interpolation axis
    
    :param deg: Interpolation degree
    
    :param old_axis: (Optional) Original vector axis. If None,
      a regular range axis is assumed (default None).

    :param fill_value: (Optional) extrapolated points are filled with
      this value (default np.nan)
    """
    returned_vector=False
    if old_axis == None:
        old_axis = np.arange(a.shape[0])
    elif old_axis[0] > old_axis[-1]:
        old_axis = old_axis[::-1]
        a = a[::-1]
        new_axis = new_axis[::-1]
        returned_vector=True

    if np.sum(np.isnan(a)) > 0:
            nonnans = np.nonzero(~np.isnan(a))
            old_axis = old_axis[nonnans]
            a = a[nonnans]

    if a.dtype != np.complex:
        f = interpolate.UnivariateSpline(old_axis, a, k=deg, s=0)
        result = f(new_axis)
    else:
        result = np.empty(new_axis.shape[0], dtype=complex)
        f_real = interpolate.UnivariateSpline(old_axis, np.real(a), k=deg, s=0)
        result.real = f_real(new_axis)
        f_imag = interpolate.UnivariateSpline(old_axis, np.imag(a), k=deg, s=0)
        result.imag = f_imag(new_axis)
            
    if returned_vector:
        result = result[::-1]
        old_axis = old_axis[::-1]
        new_axis = new_axis[::-1]

    # extrapolated parts are set to fil__value
    result[np.nonzero(new_axis > np.max(old_axis))] = fill_value
    result[np.nonzero(new_axis < np.min(old_axis))] = fill_value
    return result

def border_cut_window(n, coeff=0.2):
    """Return a window function with only the edges cut by a nice
    gaussian shape function.
    
    :param n: Window length
    :param coeff: Border size in percentage of the total length.
    """
    window = np.ones(n)
    border_length = int(float(n)*coeff)
    if border_length <= 1:
        window[0] = 0.
        window[-1] = 0.
    else:
        borders = signal.get_window(("gaussian",border_length/3.),
                                    border_length*2+1)
        z = int(float(borders.shape[0])/2.)
        window[:z] = borders[:z]
        window[-z:] = borders[-z:]
    return window


def get_lr_phase(interf, n_phase=None, return_lr_spectrum=False):
    """Return a low resolution phase from a given interferogram vector.

    :param interf: Interferogram vector
    
    :param n_phase: (Optional) Number of points for phase
      computation. Of course it can be no greater than the number of
      points of the interferogram. If None, this is set to 50% of the
      interferogram length (Default None).

    :param return_lr_spectrum: (Optional) If True return also the low
      resolution spectrum from which phase is computed (Default False).
    """
    LOW_RES_COEFF = 0.5 # Ratio of the number of points for phase
                        # computation over the number of points of the
                        # interferogram
                        
    dimz = interf.shape[0]
    # define the number of points for phase computation
    if n_phase == None:
        n_phase = int(LOW_RES_COEFF * float(dimz))
            
    elif n_phase > dimz:
        warnings.warn("The number of points for phase computation is too high (it can be no greater than the interferogram length). Phase is computed with the maximum number of points available")
        n_phase = dimz

    if n_phase != dimz:
        lr_interf = np.copy(interf[
            int((dimz - n_phase)/2.):
            int((dimz - n_phase)/2.) + n_phase])
    else:
        lr_interf = np.copy(interf)
        
    # apodization
    lr_interf *= learner95_window(n_phase)
  
    # zero padding
    zp_phase_len = next_power_of_two(2 * n_phase)
    zp_border = int((zp_phase_len - n_phase) / 2.)
    temp_vector = np.zeros(zp_phase_len, dtype=float)
    temp_vector[zp_border:(zp_border + n_phase)] = lr_interf
    lr_interf = temp_vector
    
    # centerburst
    lr_interf = np.roll(lr_interf,
                        zp_phase_len/2 - int((dimz&1 and not n_phase&1)))
    
    # fft
    lr_spectrum = np.fft.fft(lr_interf)[:zp_phase_len/2]
    lr_phase = np.unwrap(np.angle(lr_spectrum))
    if not return_lr_spectrum:
        return interpolate_size(lr_phase, n_phase, 1)
    else:
        return (interpolate_size(lr_phase, n_phase, 1),
                interpolate_size(np.abs(lr_spectrum), n_phase, 1))


def transform_interferogram(interf, nm_laser, 
                            calibration_coeff, step, order, 
                            window_type, zpd_shift, n_phase=None,
                            return_phase=False, ext_phase=None,
                            nm_max=None, weights=None, polyfit_deg=1,
                            balanced=True, bad_frames_vector=None,
                            smoothing_deg=2, return_complex=False,
                            final_step_nb=None, return_ireg_axis=False,
                            low_order_correction=True,
                            conserve_energy=False):
    
    """Transform an interferogram into a spectrum.
    
    :param interf: Interferogram to transform.
    
    :param nm_laser: Wavelength of the laser used for calibration.
    
    :param calibration_coeff: Wavelength of the laser emission line
      corresponding to the computed interferogram.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order (can be 0 but nm_max must be specified).

    :param window_type: Name of the apodization function.

    :param zpd_shift: Shift of the interferogram to center the ZPD.

    :param bad_frames_vector: (Optional) Mask-like vector containing
      ones for bad frames. Bad frames are replaced by zeros using a
      special function that smoothes transition between good parts and
      zeros (default None). This vector must be uncorrected for ZPD
      shift

    :param n_phase: (Optional) Number of points to use for phase
      correction. It can be no greater than interferogram length. If
      0, no phase correction will be done and the resulting spectrum
      will be the absolute value of the complex spectrum. If None, the
      number of points is set to 20 percent of the interferogram
      length (default None).

    :param ext_phase: (Optional) External phase vector. If given this
      phase vector is used instead of a low-resolution one. It must be
      as long as the interferogram.
    
    :param nm_max: (Optional) Maximum wavelength of the spectrum. Must be
      specified if order is equal to 0 (default None).
      
    :param return_phase: (Optional) If True, compute only the phase of
      the interferogram and return it. If polyfit_deg is >= 0, return
      the coefficients of the fitted phase (default False). Note that
      this option is not compatible with ext_phase. You must set
      ext_phase to None to set return_phase to True.

    :param weights: (Optional) A vector of the same length as the
      number of points used to compute the phase (n_phase) giving the
      weight of each point for interpolation (Must be a float between
      0. and 1.). If none is given, the weights are defined by the
      amplitude of the vector.

    :param polyfit_deg: (Optional) Degree of the polynomial fit to the
      computed phase. If < 0, no fit will be performed (Default 1).

    :param smoothing_deg: (Optional) Degree of zeros smoothing. A
      higher degree means a smoother transition from zeros parts (bad
      frames) to non-zero parts (good frames) of the
      interferogram. Good parts on the other side of the ZPD in
      symmetry with zeros parts are multiplied by 2. The same
      transition is used to multiply interferogram points by zero and
      2 (default 2). This operation is not done if smoothing_deg is
      set to 0.

    :param balanced: (Optional) If False, the interferogram is
      considered as unbalanced. It is flipped before its
      transformation to get a positive spectrum. Note
      that a merged interferogram is balanced (default True).

    :param return_complex: (Optional) If True and if phase is
      corrected the returned spectrum will be complex. In False only
      the real part is returned (default False)

    :param final_step_nb: (Optional) Number of samples of the
      resulting spectrum. If None, the number of samples of the
      spectrum will be the same as the interferogram (default None).

    :param return_ireg_axis: (Optional) If True, return spectrum along
      an irregular wavelength axis corresponding to its regular cm-1
      wavenumber axis (emission lines and especially unapodized sinc
      emission lines are symetric) (default False).

    :param low_order_correction: (Optional) If True substract a low
      order polynomial to remove low frequency noise. Useful for
      unperfectly corrected interferograms (default True).

    :param conserve_energy: (Optional) If True the energy is conserved
      in the transformation (default False).

    .. note:: Interferogram can be complex
    """
    MIN_ZEROS_LENGTH = 8 # Minimum length of a zeros band to smooth it
    interf = np.copy(interf)
   
    if return_phase and n_phase == 0:
        raise Exception("Phase cannot be computed with 0 points, return_phase=True and n_phase=0 options are not compatible !")
    if return_phase and ext_phase != None:
        raise Exception("return_phase=True and ext_phase != None options are not compatible. Set the phase or get it !")
    
    dimz = interf.shape[0]

    if final_step_nb == None:
        final_step_nb = dimz

    
    # discard zeros interferogram
    if len(np.nonzero(interf)[0]) == 0:
        if return_phase:
            return None
        else:
            return interf

    if conserve_energy:
        interf_energy = interf_mean_energy(interf)

    # replace NaN and Inf values by zeros
    interf[np.nonzero(np.isnan(interf))] = 0.
    interf[np.nonzero(np.isinf(interf))] = 0.

    # reverse unbalanced vector
    if not balanced:
        interf = -interf
   
    #####
    # 1 - substraction of the mean of the interferogram where the
    # interferogram is not 0
    nonzero_pix = np.nonzero(interf != 0.)
    if len(nonzero_pix[0])>0:
        interf_mean = np.mean(interf[nonzero_pix])
        interf[nonzero_pix] -= interf_mean
        
    #####
    # 2 - low order polynomial substraction to suppress 
    # low frequency noise
    if low_order_correction:
        low_order_fit = polyfit1d(interf, 3)
        for inonzero in nonzero_pix[0]:
            interf[inonzero] -= low_order_fit[inonzero]

    #####
    # 3 - ZPD shift to center the spectrum
    if zpd_shift != 0:
        temp_vector = np.zeros(interf.shape[0] + 2 * abs(zpd_shift),
                               dtype=interf.dtype)
        temp_vector[abs(zpd_shift):abs(zpd_shift) + dimz] = interf
        interf = np.copy(temp_vector)
        interf = np.roll(interf, zpd_shift)
        
        if bad_frames_vector is not None:
            temp_vector[
                abs(zpd_shift):abs(zpd_shift) + dimz] = bad_frames_vector
            bad_frames_vector = np.copy(temp_vector)
            bad_frames_vector = np.roll(bad_frames_vector, zpd_shift)
    
    #####
    # 4 - Zeros smoothing
    #
    # Smooth the transition between good parts and 'zeros' parts. We
    # use here a concept from Learner et al. (1995) Journal of the
    # Optical Society of America A, 12(10), 2165
    zeros_vector = np.ones_like(interf)
    zeros_vector[np.nonzero(interf == 0)] = 0
    zeros_vector = zeros_vector.real # in case interf is complex
    if bad_frames_vector != None:
        zeros_vector[np.nonzero(bad_frames_vector)] = 0
    if len(np.nonzero(zeros_vector == 0)[0]) > 0:
        # correct only 'bands' of zeros:
        zcounts = count_nonzeros(-zeros_vector + 1)
        zeros_mask = np.nonzero(zcounts >= MIN_ZEROS_LENGTH)
        if len(zeros_mask[0]) > 0 and smoothing_deg > 0.:
            for izero in zeros_mask[0]:
                if (izero > smoothing_deg
                    and izero < interf.shape[0] - 1 - smoothing_deg):
                    zeros_vector[izero - smoothing_deg:
                                 izero + smoothing_deg + 1] = 0
            zeros_vector = smooth(np.copy(zeros_vector), deg=smoothing_deg,
                                  kind='cos_conv')
            zeros_vector = zeros_vector * (- zeros_vector[::-1] + 2)
            interf *= zeros_vector
    

    #####
    # 5 - Phase determination (Mertz method)
    #
    # We use the method described by Learner et al. (1995) Journal of
    # the Optical Society of America A, 12(10), 2165
    #
    # The low resolution interferogram is a small part of the real
    # interferogram taken symmetrically around ZPD
    if (ext_phase is None) and (n_phase != 0):
        lr_phase, lr_spectrum = get_lr_phase(interf, n_phase=n_phase,
                                             return_lr_spectrum=True)
        
        # fit
        if polyfit_deg >= 0:
            # polynomial fitting must be weigthed in case of a spectrum
            # without enough continuum.
            if weights == None or not np.any(weights):
                weights = np.abs(lr_spectrum)
                # suppress noise on spectrum borders
                weights *= border_cut_window(lr_spectrum.shape[0])
                if np.max(weights) != 0.:
                    weights /= np.max(weights)
                else:
                    weights = np.ones_like(lr_spectrum)
                # remove parts with a bad signal to noise ratio
                weights[np.nonzero(weights < 0.25)] = 0.
            else:
                if weights.shape[0] != lr_phase.shape[0]:
                    weights = interpolate_size(weights, lr_phase.shape[0], 1)

            lr_phase, lr_phase_coeffs = polyfit1d(lr_phase, polyfit_deg,
                                                  w=weights, return_coeffs=True)
            
            
    elif ext_phase is not None:
        lr_phase = ext_phase

    if return_phase:
        if polyfit_deg < 0:
            return lr_phase
        else:
            return lr_phase_coeffs

    #####
    # 6 - Apodization of the real interferogram
    if window_type != None and window_type != '1.0':
        if window_type in ['1.1', '1.2', '1.3', '1.4', '1.5',
                           '1.6', '1.7', '1.8', '1.9', '2.0']:
            window = norton_beer_window(window_type, interf.shape[0])
        else:
            window = signal.get_window((window_type), interf.shape[0])
            
        interf *= window

    #####
    # 7 - Zero padding
    #
    # Define the size of the zero padded vector to have at
    # least 2 times more points than the initial vector to
    # compute its FFT. FFT computation is faster for a vector
    # size equal to a power of 2.
    #zero_padded_size = next_power_of_two(2*final_step_nb)
    zero_padded_size = 2 * final_step_nb
    
    temp_vector = np.zeros(zero_padded_size, dtype=interf.dtype)
    zeros_border = int((zero_padded_size - interf.shape[0]) / 2.)
    temp_vector[zeros_border:(zeros_border + interf.shape[0])] = interf
    zero_padded_vector = temp_vector

    #####
    # 8 - Zero the centerburst
    zero_padded_vector = np.roll(zero_padded_vector,
                                 zero_padded_vector.shape[0]/2)

    #####
    # 9 - Fast Fourier Transform of the interferogram
    center = zero_padded_size / 2
    interf_fft = np.fft.fft(zero_padded_vector)[:center]

    # normalization of the vector to take into account zero-padding 
    if np.iscomplexobj(interf):
        interf_fft /= (zero_padded_size / dimz)
    else:
        interf_fft /= (zero_padded_size / dimz) / 2.
        
    #####
    # 10 - Phase correction
    if n_phase != 0:
        lr_phase = interpolate_size(lr_phase, interf_fft.shape[0], 1)
        spectrum_corr = np.empty_like(interf_fft)
        spectrum_corr.real = (interf_fft.real * np.cos(lr_phase)
                              + interf_fft.imag * np.sin(lr_phase))
        spectrum_corr.imag = (interf_fft.imag * np.cos(lr_phase)
                              - interf_fft.real * np.sin(lr_phase))
    else:
        spectrum_corr = np.abs(interf_fft)

    #####
    # 11 - Off-axis effect correction with maxima map   
    # Irregular wavelength axis creation
    correction_coeff = float(calibration_coeff) / nm_laser
    nm_axis_ireg = create_nm_axis_ireg(spectrum_corr.shape[0], step, order,
                                       nm_max=nm_max, corr=correction_coeff)
    
    # Spectrum is returned if folding order is even
    if int(order) & 1:
        spectrum_corr = spectrum_corr[::-1]

    # Interpolation (order 5) of the spectrum from its irregular axis
    # to the regular one
    
    # regular axis creation (in nm, if step is in nm)
    if not return_ireg_axis:
        final_axis = create_nm_axis(final_step_nb, step, order, nm_max=nm_max)
    else:
        final_axis = create_nm_axis_ireg(final_step_nb, step, order,
                                         nm_max=nm_max, corr=1.)
 
    # spectrum interpolation
    if not (return_ireg_axis and correction_coeff == 1.):
        spectrum = interpolate_axis(spectrum_corr, final_axis, 5,
                                    old_axis=nm_axis_ireg)
    else:
        spectrum = spectrum_corr


    # Extrapolated parts of the spectrum are set to NaN
    spectrum[np.nonzero(final_axis > np.max(nm_axis_ireg))] = np.nan
    spectrum[np.nonzero(final_axis < np.min(nm_axis_ireg))] = np.nan


    if conserve_energy:
        # Spectrum is rescaled to the modulation energy of the interferogram
        spectrum = spectrum / spectrum_mean_energy(spectrum) * interf_energy
    
    if n_phase != 0:
        if return_complex:
            return np.copy(spectrum)
        else:
            return np.copy(spectrum.real)
    else:
        return np.copy(spectrum)


def transform_spectrum(spectrum, nm_laser, calibration_coeff,
                       step, order, window_type, zpd_shift, nm_max=None,
                       ext_phase=None, return_complex=False, wavenumber=False,
                       final_step_nb=None):
    """Transform a spectrum into an interferogram.

    This function is the inverse of :py:meth:`utils.transform_interferogram`.

    So that to get the initial interferogram, the same options used in
    transform interferogram must be passed to this function. The
    spectrum must also be the complex form (use return_complex option
    in :py:meth:`utils.transform_interferogram`)

    :param spectrum: Spectrum to transform

    :param nm_laser: Wavelength of the laser used for calibration.
    
    :param calibration_coeff: Wavelength of the laser emission line
      corresponding to the computed interferogram.

    :param step: Step size of the moving mirror in nm.

    :param order: Folding order (can be 0 but nm_max must be specified).

    :param window_type: Name of the apodization function.

    :param zpd_shift: Shift of the interferogram to decenter the ZPD.

    :param nm_max: (Optional) Maximum wavelength of the spectrum. Must be
      specified if order is equal to 0 (default None).
      
    :param ext_phase: (Optional) External phase vector. If given this
      phase vector is used in place of the original phase of the
      spectrum. Useful to add a phase to an interferogram. Note that
      this phase is intended to be used to inverse transform an
      already transformed interferogram. The computed phase correction
      can thus be used directly. As the phase vector given by
      :py:meth:`utils.transform_interferogram` is not reversed for
      even orders, it is reversed here in this function.

    :param return_complex: (Optional) If True return a complex
      interferogram. Else return the real part of it (default False).

    :param wavenumber: (Optional) If True the spectrum axis is in
      cm-1. In this case, and if no wavelength correction has to be
      applied (calibration_coeff == nm_laser) there will be no
      interpolation of the original spectrum (better precision)
      (default False).

    :param final_step_nb: (Optional) Final size of the
      interferogram. Must be less than the size of the original
      spectrum. If None the final size of the interferogram is the
      same as the size of the original spectrum (default None).

    .. note:: Interferogram can be complex
    """
    INTERPOLATION_COEFF = 1 # must be an interger >= 1

    spectrum = np.copy(spectrum)
    spectrum = spectrum.astype(np.complex)
    step_nb = spectrum.shape[0]
    if final_step_nb is not None:
        if final_step_nb > step_nb:
            self._print_error('final_step_nb must be less than the size of the original spectrum')
    else:
        final_step_nb = step_nb

    # On-axis -> Off-axis [nm - > cm-1]
    correction_coeff = calibration_coeff / nm_laser
    
    if not wavenumber:
        base_axis = create_nm_axis(step_nb, step, order, nm_max=nm_max)
    else:
        base_axis = create_nm_axis_ireg(step_nb, step, order,
                                        corr=1., nm_max=nm_max)
    
    nm_axis_ireg = create_nm_axis_ireg(step_nb, step, order,
                                       corr=correction_coeff,
                                       nm_max=nm_max)
    if not (wavenumber and correction_coeff == 1.):
        spectrum = interpolate_axis(spectrum, nm_axis_ireg[::-1], 5,
                                    old_axis=base_axis, fill_value=0.)
    else:
        spectrum = spectrum[::-1]
    
    # Add phase to the spectrum (Re-phase)
    if ext_phase is not None:
        if np.any(ext_phase != 0.):
            if not order&1:
                ext_phase = ext_phase[::-1]
            ext_phase = interpolate_size(ext_phase, step_nb, 5)
            spectrum_real = np.copy(spectrum.real)
            spectrum_imag = np.copy(spectrum.imag)
            spectrum.real = (spectrum_real * np.cos(ext_phase)
                             - spectrum_imag * np.sin(ext_phase))
            spectrum.imag = (spectrum_real * np.sin(ext_phase)
                             + spectrum_imag * np.cos(ext_phase))

    
    # Zero-filling
    zeros_spectrum = np.zeros(step_nb * 2, dtype=spectrum.dtype)
    if order&1:
        zeros_spectrum[:spectrum.shape[0]] += spectrum
    else:
        zeros_spectrum[:spectrum.shape[0]] += spectrum[::-1]
    spectrum = zeros_spectrum

    # IFFT and re-shift + center burst
    interf = np.fft.ifft(spectrum)
    interf = np.roll(interf, step_nb - zpd_shift)
    interf = interf[
        step_nb-(final_step_nb/2) - final_step_nb%2:
        step_nb+(final_step_nb/2)]
    
    interf = np.array(interf)

    # De-apodize
    if window_type is not None:
        window = norton_beer_window(window_type, final_step_nb)
        interf /= window
    
    # Normalization to remove zero filling effect on the mean energy
    interf *= step_nb / float(final_step_nb) * 2.
    
    if return_complex:
        return interf
    else:
        return interf.real


def spectrum_mean_energy(spectrum):
    """Return the mean energy of a spectrum by channel.

    :param spectrum: a 1D spectrum
    """
    s = np.array(spectrum[np.nonzero(~np.isnan(spectrum))])
    n = np.size(spectrum)
    return (1./float(n)) * math.sqrt(np.sum(np.abs(s)**2.))


def interf_mean_energy(interf):
    """Return the mean energy of an interferogram by step.

    :param interf: an interferogram

    .. warning:: The mean of the interferogram is substracted to
      compute only the modulation energy. This is the modulation
      energy which must be conserved in the resulting spectrum. Note
      that the interferogram transformation function (see
      :py:meth:`utils.transform_interferogram`) remove the mean of the
      interferogram before computing its FFT.

    .. note:: NaNs are set to 0.
    """
    modulation_interf = interf - robust_mean(interf)
    modulation_interf[np.nonzero(np.isnan(modulation_interf))] = 0.
    return math.sqrt(robust_mean(np.abs(modulation_interf)**2.))

def variable_me(n, params):
     """Return a sinusoidal function representing a variable
     modulation efficiency.

     This function is used to correct for fringes.

     :param params: A tuple of floats [frequency, amplitude,
       phase]. The frequency gives the number of repetition of a sinus
       over the vector. The amplitude must be between 0. (returns a
       vector of 1) and 1. (returns a sinus going from 0 to 1). Phase
       can be a single float or a vector of size n
     """
     f = params[0]
     a = params[1]
     phi = params[2]
     me_real = np.cos(np.arange(n, dtype=float)
                      / float(n - 1.) * 2. * math.pi * f)
     me_imag = np.sin(np.arange(n, dtype=float)
                      / float(n - 1.) * 2. * math.pi * f)
     me = np.empty_like(me_real, dtype=complex)
     me.real = (me_real * np.cos(phi) - me_imag * np.sin(phi)) * a + (1. - a)
     me.imag = (me_imag * np.cos(phi) + me_real * np.sin(phi)) * a
     return me
