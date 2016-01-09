#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: image.py

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

import numpy as np
import math
import scipy
import warnings
from scipy import interpolate, optimize, ndimage, signal
import bottleneck as bn

from orb.core import Tools

import orb.utils.stats
import orb.utils.vector
import orb.cutils

def compute_binning(image_shape, detector_shape):
    """Return binning along both axis given the image shape and the
    detector shape.

    :param image_size: Tuple [x,y] giving the image shape
    
    :param detector_shape: Tuple [x,y] giving the detector shape
      (i.e. maximum numbers of pixels along the x and y axis.)
    """
    binning = np.floor(
        np.array(detector_shape, dtype=float)
        / np.array(image_shape, dtype=float)).astype(int)
    if np.all(binning > 0): return binning
    else: raise Exception('Bad binning value (must be > 0)')


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
                 "import orb.cutils as cutils",
                 "from orb.core import Tools",
                 "from orb.utils import *",
                 "import warnings",
                 "import orb")))
            for ijob in range(ncpus)]
    
    for ijob, job in jobs:
        result[divs[ijob]:divs[ijob+1],:] = job()
        
    to._close_pp_server(job_server)
    return result


def check_frames(frames, sigma_reject=2.5):
    """Check and reject deviating frames based on their median level.

    Frames with a too deviant median level are discarded. This
    function is used by :py:meth:`utils.create_master_frame`.

    :param frames: Set of frames to check
    
    :param sigma_reject: (Optional) Rejection coefficient (default 2.5)
    
    """
    z_median = np.array([bn.nanmedian(frames[:,:,iframe])
                         for iframe in range(frames.shape[2])])
    z_median_cut = orb.utils.stats.sigmacut(
        z_median, sigma=sigma_reject)
    bad_frames = (z_median > (orb.utils.stats.robust_median(z_median_cut)
                         + sigma_reject * orb.utils.stats.robust_std(
                                  z_median_cut)))
    if np.any(bad_frames):
        warnings.warn('Some frames (%d) appear to be much different from the others. They have been removed before being combined. Please check the frames.'%np.sum(bad_frames))
        print 'Median levels: %s'%str(z_median)
        print 'Rejected: %s'%str(bad_frames)
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
    warnings.simplefilter('ignore', RuntimeWarning)
    
    NKEEP = 2 # Minimum number of values to keep for each pixel

    frames = np.array(frames)

    if len(frames.shape) == 2: # only one image
        if not silent: warnings.warn("Only one image to create a master frame. No combining method can be used.")
        return frames

    if frames.shape[2] < 3:
        if frames.shape[2] == 1:
            warnings.warn("Only one image to create a master frame. No combining method can be used.")
            return np.squeeze(frames)
        
        if not silent: warnings.warn("Not enough frames to use a rejection method (%d < 3)"%frames.shape[2])
        reject = None

    if reject not in ['sigclip', 'minmax', 'avsigclip']:
        raise Exception("Rejection operation must be 'sigclip', 'minmax' or None")
    if combine not in ['median', 'average']:
        raise Exception("Combining operation must be 'average' or 'median'")

    if not silent: print "Rejection operation: %s"%reject
    if not silent: print "Combining operation: %s"%combine

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
    
    master, reject_count_frame, std_frame = orb.cutils.master_combine(
        frames, sigma, NKEEP, combine_mode, reject_mode, return_std_frame=True)

    if reject in ['sigclip', 'avsigclip']:
        if not silent: print "Maximum number of rejected pixels: %d"%np.max(reject_count_frame)
        if not silent: print "Mean number of rejected pixels: %f"%np.mean(reject_count_frame)

    print "median std of combined frames: {}".format(
        orb.utils.stats.robust_median(std_frame))
    
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

    :return: x_min, x_max, y_min, y_max
    """
    return orb.cutils.get_box_coords(int(ix), int(iy), int(box_size),
                                     int(x_lim_min), int(x_lim_max),
                                     int(y_lim_min), int(y_lim_max))


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


def correct_hot_pixels(im, hp_map, box_size=3, std_filter_coeff=1.5):
    """Correct hot pixels in an image given a map of their position.

    The algorithm used replaces a hot pixel value by the median of the
    pixels in a box around it. Pixels values which are not too much
    different from the values around are not modified.

    :param im: Image to correct
    
    :param hp_map: Hot pixels map (1 for hot pixels, 0 for normal
      pixel)

    :param box_size: (Optional) Size of the correction box (default
      3).

    :param std_filter_coeff: (Optional) Coefficient on the std used to
      check if the value of a hot pixel must be changed (default 1.5).
    """
    im_temp = np.copy(im)
    hp_list = np.nonzero(hp_map)
    
    for i in range(len(hp_list[0])):
        ix, iy = hp_list[0][i], hp_list[1][i]
        x_min, x_max, y_min, y_max = get_box_coords(ix, iy, box_size,
                                    0, im.shape[0], 0, im.shape[1])
        im_temp[ix,iy] = np.nan
        box = im_temp[x_min:x_max,y_min:y_max]
        std = orb.utils.stats.robust_std(box)
        med = orb.utils.stats.robust_median(box)
        if (im[ix,iy] > med + std_filter_coeff * std
            or im[ix,iy] <  med - std_filter_coeff * std):
            im[ix,iy] = med
    return im

def transform_frame(frame, x_min, x_max, y_min, y_max, 
                    d, rc, zoom_factor, interp_order, 
                    mask=None, fill_value=np.nan,
                    sip_A=None, sip_B=None):
    """Transform one frame or a part of it using transformation
    coefficients.

    :param frame: Frame to transform
    
    :param x_min: Lower x boundary of the transformed section (can be
      a tuple in order to get multiple sections)
    
    :param x_max: Upper x boundary of the transformed section (can be
      a tuple in order to get multiple sections)
    
    :param y_min: Lower y boundary of the transformed section (can be
      a tuple in order to get multiple sections)
    
    :param y_max: Upper y boundary of the transformed section (can be
      a tuple in order to get multiple sections)
    
    :param d: Transformation coefficients [dx, dy, dr, da, db]
    
    :param rc: Rotation center of the frame [rc_x, rc_y]
    
    :param zoom_factor: Zoom on the image. Can be a couple (zx, zy).
    
    :param interp_order: Interpolation order
    
    :param mask: (Optional) If a mask frame is passed it is
      transformed also (default None).

    :param fill_value: (Optional) Fill value for extrapolated points
      (default np.nan).

    :param sip_A: (Optional) pywcs.WCS() instance containing SIP parameters of
      the output image (default None).
      
    :param sip_B: (Optional) pywcs.WCS() instance containing SIP parameters of
      the input image (default None).
    """

    def mapping(coords, transx, transy, x_min, y_min):
        if (x_min + coords[0] < transx.shape[0]
            and y_min + coords[1] < transx.shape[1]):
            return (transx[x_min + coords[0],
                           y_min + coords[1]],
                    transy[x_min + coords[0],
                           y_min + coords[1]])
        else:
            return (np.nan, np.nan)

    if np.size(zoom_factor) == 2:
        zx = zoom_factor[0]
        zy = zoom_factor[1]
    else:
        zx = float(zoom_factor)
        zy = float(zoom_factor)
        
    ## create transform maps for mapping function
    transx, transy = orb.cutils.create_transform_maps(
        frame.shape[0], frame.shape[1], d[0], d[1], d[2], d[3], d[4], rc[0],
        rc[1], zx, zy, sip_A, sip_B)

    if frame.dtype != np.dtype(float):
        frame = frame.astype(float)
    
    if mask is not None:
        mask = mask.astype(float)

    if not (isinstance(x_min, tuple) or isinstance(x_min, list)):
        x_min = [x_min]
        y_min = [y_min]
        x_max = [x_max]
        y_max = [y_max]

    if not ((len(x_min) == len(x_max))
            and (len(y_min) == len(y_max))
            and (len(x_min) == len(y_min))):
        raise Exception('x_min, y_min, x_max, y_max must have the same length')
    sections = list()
    sections_mask = list()
    for i in range(len(x_min)):
        x_size = x_max[i] - x_min[i]
        y_size = y_max[i] - y_min[i]
        output_shape = (x_size, y_size) # accelerate the process (unused
                                        # data points are not computed)

        sections.append(ndimage.interpolation.geometric_transform(
            frame, mapping, extra_arguments=(transx, transy,
                                             x_min[i], y_min[i]),
            output_shape=output_shape,
            order=interp_order, mode='constant', cval=fill_value))

        if mask is not None:
            sections_mask.append(ndimage.interpolation.geometric_transform(
                mask, mapping, extra_arguments=(transx, transy,
                                                x_min[i], y_min[i]),
                output_shape=output_shape,
                order=interp_order, mode='constant', cval=fill_value))
            

    if mask is not None:
        if len(x_min) == 1:
            return (sections[0], sections_mask[0])
        else:
            return sections, sections_mask
    else:
        if len(x_min) == 1:
            return sections[0]
        else:
            return sections

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
      filter. Must be > 0 (default 1).
    """
    lp_im = low_pass_image_filter(im, deg=deg)
    return lp_im - scipy.ndimage.filters.median_filter(lp_im, size=(5,5))

def low_pass_image_filter(im, deg):
    """Return a low pass filtered image using a gaussian kernel.
    
    :param im: Image to filter
    
    :param deg: Radius of the kernel. Must be > 0.
    """
    if not deg > 0:
        raise Exception('Kernel degree must be > 0')

    if 2 * deg >= max(im.shape):
        raise Exception('Kernel degree is too high given the image size')

    return orb.cutils.low_pass_image_filter(np.copy(im).astype(float), int(deg))

def fit_map(data_map, err_map, smooth_deg):
    """Fit map with low order polynomials

    :param data_map: data map

    :param err_map: error map

    :param smooth_deg: Degree of fit smoothing (beware of high
      smoothing degrees)

    :return: (fitted data map, error map, fit error)
    """

    def smooth_fit_parameters(coeffs_list, err_list, order, smooth_deg):
        """
        Smooth the fitting parameters for a particular order of the
        polynomial fit.
        """
        coeffs = coeffs_list[:,order]
        w = np.squeeze((1./err_list)**2.)
        return orb.utils.vector.polyfit1d(coeffs, smooth_deg, w=w,
                                          return_coeffs=True)

    def model(fit_coeffs, smooth_deg, dimx, dimy):
        fit_coeffs = fit_coeffs.reshape((smooth_deg + 1, smooth_deg + 1))
        fitted_data_map = np.zeros((dimx, dimy), dtype=float)
        params_fit_list = list()
        for ideg in range(smooth_deg + 1):
            params_fit_list.append(np.polynomial.polynomial.polyval(
                np.arange(dimy), fit_coeffs[ideg,:]))
        params_fit_list = np.array(params_fit_list)
        for ij in range(dimy):
            fitted_data_map[:,ij] = np.polynomial.polynomial.polyval(
                np.arange(dimx), params_fit_list[:, ij])
        return fitted_data_map
    
    def diff(fit_coeffs, data_map, smooth_deg, err_map):
        fitted_data_map = model(fit_coeffs, smooth_deg,
                                data_map.shape[0],
                                data_map.shape[1])
        
        res = (data_map - fitted_data_map)/err_map
        res = res[np.nonzero(~np.isnan(res))]
        res = res[np.nonzero(~np.isinf(res))]
        return res.flatten()
  

    coeffs_list = list()
    err_list = list()
    dimx, dimy = data_map.shape
    
    ## Phase map fit
    for ij in range(dimy):
        imap = data_map[:,ij]
        ierr = err_map[:,ij]
        w = (1./ierr)**2.
        
        w[np.nonzero(np.isinf(w))] = np.nan
        
        
        # reject columns with too much NaNs
        if np.sum(~np.isnan(imap)* ~np.isnan(w)) > dimy/3.:
            vect, coeffs = orb.utils.vector.polyfit1d(
                imap, smooth_deg, w=w, return_coeffs=True)
            coeffs_list.append(coeffs[0])
            err_list.append(coeffs[1][0])
            
        else:
            bad_coeffs = np.empty(smooth_deg + 1, dtype=float)
            bad_coeffs.fill(np.nan)
            coeffs_list.append(bad_coeffs)
            err_list.append(np.nan)
            

    coeffs_list = np.array(coeffs_list)
    err_list = np.array(err_list)

    if np.all(np.isnan(coeffs_list)):
        raise Exception('All fit coeffs are NaNs !')
        
    ## Smooth fit parameters 
    params_fit_list = list()
    fit_coeffs_list = list()
    for ideg in range(smooth_deg + 1):
        fit, coeffs = smooth_fit_parameters(
            coeffs_list, err_list, ideg, smooth_deg)
        params_fit_list.append(fit)
        fit_coeffs_list.append(coeffs[0])
        
    params_fit_list = np.array(params_fit_list)
    fit_coeffs_list = np.array(fit_coeffs_list)
    
    
    ## smooth optimization
    fit = optimize.leastsq(diff, fit_coeffs_list.flatten(),
                           args=(data_map, smooth_deg, err_map),
                           maxfev=1000, full_output=True,
                           xtol=1e-6)
    
    if fit[-1] <= 4:
        fitted_data_map = model(fit[0], smooth_deg, dimx, dimy)
    
    else:
        raise Exception('Fit could not be optimized')

    ## Error computation
    # Creation of the error map: The error map gives the 
    # Squared Error for each point used in the fit point. 
    error_map = data_map - fitted_data_map

    # The square root of the mean of this map is then normalized
    # by the range of the values fitted. This gives the Normalized
    # root-mean-square deviation
    fit_error =(np.nanmean(np.sqrt(error_map**2.))
                / (np.nanmax(data_map) - np.nanmin(data_map)))

    return fitted_data_map, error_map, fit_error


def tilt_calibration_laser_map(cmap, calib_laser_nm, phi_x, phi_y, phi_r):
    """Tilt and rotate a calibration laser map.

    :param cmap: calibration laser map.
    :param calib_laser_nm: Calibration laser wavelength in nm.
    :param phi_x: tilt angle along X axis (degrees).
    :param phi_y: tilt angle along Y axis (degrees).
    :param phi_r: Rotation angle (degrees).
    """
    phi_x = phi_x / 180. * math.pi
    phi_y = phi_y / 180. * math.pi
    phi_r = phi_r / 180. * math.pi
    
    xc, yc = cmap.shape[0]/2, cmap.shape[1]/2
    theta_map = np.arccos(calib_laser_nm / cmap)
    theta_c = theta_map[xc, yc]
    theta_map -= theta_c
    
    X, Y = np.mgrid[0:cmap.shape[0], 0:cmap.shape[1]].astype(float)
    X -= xc
    Y -= yc
    
    alpha = math.pi/2. - theta_map

    if phi_r != 0:
        Xr = X * math.cos(phi_r) - Y * math.sin(phi_r)
        Yr = X * math.sin(phi_r) + Y * math.cos(phi_r)
        X = Xr
        Y = Yr
    
    if phi_x != 0.:
        alpha2 = math.pi/2. - theta_map - phi_x
        ratio = np.sin(alpha) / np.sin(alpha2)
        X *= ratio
        
    if phi_y != 0.:
        alpha2 = math.pi/2. - theta_map - phi_y
        ratio = np.sin(alpha) / np.sin(alpha2)
        Y *= ratio
    
    f_theta = interpolate.RectBivariateSpline(
        np.arange(cmap.shape[0], dtype=float)-xc,
        np.arange(cmap.shape[1], dtype=float)-yc,
        theta_map, kx=1, ky=1, s=0)
    
    new_calib_map = calib_laser_nm / np.cos(f_theta.ev(X, Y) + theta_c)
    # reject extrapolated values
    new_calib_map[np.nonzero(
        (X < -xc) + (X >= cmap.shape[0] - xc)
        + (Y < -yc) + (Y >= cmap.shape[1] - yc))] = np.nan
    
    return new_calib_map


def simulate_calibration_laser_map(nx, ny, pixel_size, calib_laser_wl, mirror_distance,
                                   theta_cx, theta_cy, phi_x, phi_y, phi_r):
    """Simulate a calibration laser map from optical and mechanical parameters

    :param nx: Number of pixels along X
    
    :param ny: Number of pixels along Y
    
    :param pixel_size: Size of a pixel in microns
    
    :param calib_laser_wl: Calibration laser wavelength in nm
    
    :param mirror_distance: Distance to the mirror in microns
    
    :param theta_cx: Angle along X from the optical axis to the mirror
      center in degrees
      
    :param theta_cy: Angle along Y from the optical axis to the mirror
      center in degrees

    :param phi_x: Tilt of the mirror along X in degrees
    
    :param phi_y: Tilt of the mirror along Y in degrees
    
    :param phi_r: Rotation angle of the mirror in degrees
    """
    phi_x = phi_x / 180. * math.pi
    phi_y = phi_y / 180. * math.pi
    phi_r = phi_r / 180. * math.pi
    theta_cx = theta_cx/ 180. * math.pi
    theta_cy = theta_cy/ 180. * math.pi

    X, Y = np.mgrid[0:nx, 0:ny].astype(float)
    X -= nx / 2.
    Y -= ny / 2.
    X *= pixel_size
    Y *= pixel_size
    
    # rotation
    if phi_r != 0:
        Xr = X * math.cos(phi_r) - Y * math.sin(phi_r)
        Yr = X * math.sin(phi_r) + Y * math.cos(phi_r)
        X = Xr
        Y = Yr
    
    xc = mirror_distance * math.tan(theta_cx)
    yc = mirror_distance * math.tan(theta_cy)
    dmap = np.sqrt((X+xc)**2. + (Y+yc)**2.)
    theta_map = np.arctan(dmap/mirror_distance)
    
    
    if phi_x != 0. or phi_y != 0:
        
        theta_c = math.acos(math.cos(theta_cx)*math.cos(theta_cy))
        alpha = math.pi/2. - (theta_map - theta_c)
        if phi_x != 0.:
            alpha2 = math.pi/2. - (theta_map - theta_c) - phi_x
            ratio = np.sin(alpha) / np.sin(alpha2)
            X *= ratio
            
        if phi_y != 0.:
            alpha2 = math.pi/2. - (theta_map - theta_c) - phi_y
            ratio = np.sin(alpha) / np.sin(alpha2)
            Y *= ratio
            
        dmap = np.sqrt((X+xc)**2.+(Y+yc)**2.)
        theta_map = np.arctan(dmap/mirror_distance)
            
    calib_map = calib_laser_wl/np.cos(theta_map)
    return calib_map

def nanbin_image(im, binning):
    """Mean image binning robust to NaNs.

    :param im: Image to bin
    :param binning: Binning factor (must be an integer)
    """     
    return orb.cutils.nanbin_image(im.astype(np.float64), int(binning))


def fit_calibration_laser_map(calib_laser_map, calib_laser_nm, pixel_size=15.,
                              binning=4, mirror_distance_guess=2.2e5,
                              center_angle_guess=15):
    """
    Fit a calibration laser map.

    The model is based on optical parameters.
    
    :param calib_laser_map: Reference calibration laser map.
    
    :param calib_laser_nm: Wavelength of the calibration laser in nm.
    
    :param pixel_size: (Optional) Size of the CCD pixels in um
      (default 15).
    
    :param binning: (Optional) Maps are binned to accelerate the
      process. Set the binning factor (default 4).

    :param mirror_distance_guess: (Optional) Guess on the mirror
      distance in um (default 2.2e5).

    :param center_angle_guess: (Optional) Guess on the angle at the
      center of the frame in degrees (default 15)
    """

    def model_laser_map(p, calib, calib_laser_nm, pixel_size):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        return simulate_calibration_laser_map(
            calib.shape[0], calib.shape[1], pixel_size,
            calib_laser_nm, p[0], p[1], p[2], p[3], p[4], p[5])

    def diff_laser_map(p_var, p_fix, p_ind, calib, calib_laser_nm, pixel_size):
        p = get_p(p_var, p_fix, p_ind)
        res = model_laser_map(p, calib, calib_laser_nm, pixel_size)
        res -= calib
        res = res[np.nonzero(~np.isnan(res))]
        res = orb.utils.stats.sigmacut(res)
        return res

    def get_p(p_var, p_fix, p_ind):
        p = np.empty_like(p_ind, dtype=float)
        p.fill(np.nan)
        p[np.nonzero(p_ind == 0)] = p_var
        p[np.nonzero(p_ind == 1)] = p_fix
        return p
        

    def print_params(params, fvec):
        print ('    > Calibration laser map fit parameters:\n'
               + '    distance to mirror: {} cm\n'.format(params[0]*1e-4)
               + '    X angle from the optical axis to the center: {} degrees\n'.format(params[1])
               + '    Y angle from the optical axis to the center: {} degrees\n'.format(params[2])
               + '    X tilt of the mirror: {} degrees\n'.format(params[3])
               + '    Y tilt of the mirror: {} degrees\n'.format(params[4])
               + '    Rotation angle of the mirror: {} degrees\n'.format(params[5])
               + '    Error on fit: mean {}, std {} (in nm)\n'.format(
                   np.nanmean(fvec), np.nanstd(fvec))
               + '    Error on fit: mean {}, std {} (in km/s)'.format(
                   np.nanmean(fvec)/calib_laser_nm*3e5, np.nanstd(fvec)/calib_laser_nm*3e5))

        

    CENTER_COEFF = 0.2
    LARGE_COEFF = 0.95

    print '> Binning calibration map'

    # remove obvious bad points
    calib_laser_map[calib_laser_map > calib_laser_nm * 2.] = np.nan
    calib_laser_map[calib_laser_map < calib_laser_nm / 2.] = np.nan

    binning = int(binning)
    if binning > 1:
        calib_laser_map_bin = nanbin_image(calib_laser_map, binning)
    else:
        calib_laser_map_bin = calib_laser_map

    print '> Calibration laser map fit'

    # mirror_dist, X theta, Y theta, X tilt, Y tilt, R
    # p_ind = 0: variable parameter, index=1: fixed parameter

    print '  > First fit on the central portion of the calibration laser map ({:.1f}% of the total size)'.format(CENTER_COEFF*100)
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(CENTER_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_center = calib_laser_map_bin[xmin:xmax,ymin:ymax]

    p_var = np.array([mirror_distance_guess, 0, center_angle_guess])
    p_ind = np.array([0, 0, 0, 1, 1, 1])
    p_fix = np.array([0, 0, 0])
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_center,
                                       calib_laser_nm,
                                       float(pixel_size*binning)),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, fit[2]['fvec'])

    print '  > Second fit on all the map ({:.1f}% of the total size)'.format(LARGE_COEFF*100)
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(LARGE_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_large = calib_laser_map_bin[xmin:xmax,ymin:ymax]

    p_fix = np.array([params[0], params[1], params[2]])
    p_ind = np.array([1, 1, 1, 0, 0, 0])
    p_var = np.array([0, 0, 0])
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_large,
                                       calib_laser_nm,
                                       float(pixel_size*binning)),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, fit[2]['fvec'])
    
    new_calib_laser_map = model_laser_map(
        params, calib_laser_map, calib_laser_nm, pixel_size)


    # map fit of the diff map
    print '  > residual map fit ({:.1f}% of the total size)'.format(
        LARGE_COEFF*100)
    res_map = calib_laser_map - new_calib_laser_map
    
    xmin,xmax,ymin,ymax = get_box_coords(
        res_map.shape[0]/2,
        res_map.shape[1]/2,
        int(LARGE_COEFF*res_map.shape[0]),
        0, res_map.shape[0],
        0, res_map.shape[1])

    err_map = np.ones_like(res_map)
    err_map[xmin:xmax, ymin:ymax] = 1e-35
    res_map_fit, err_map, fit_error = fit_map(
        res_map.T, err_map.T, 5)
    new_calib_laser_map += res_map_fit.T
    
    # final error
    std_err = np.nanstd(
        (calib_laser_map - new_calib_laser_map)[
            xmin:xmax, ymin:ymax])
    
    
    print '> final error (std on {:.1f}% of the total size): {:.3e} nm, {:.3e} km/s'.format(LARGE_COEFF*100., std_err, std_err/calib_laser_nm*3e5)
        
    return params, new_calib_laser_map

def fit_sitelle_phase_map(phase_map, phase_map_err, calib_laser_map,
                          calib_laser_nm, pixel_size=15., binning=4):

    """Fit a SITELLE phase map (order 0 map of the phase) using a
    model based on a simulated calibration laser map..

    An real calibration laser map is needed first to get a first guess
    on the parameters of the fit. Then the whole phase map is modeled
    to fit the real phase map.

    The modeled calibration laser map obtained from the fit is also
    returned.

    :param phase_map: Phase map to fit.
    
    :param phase_map_err: Error on the phase map values.
    
    :param calib_laser_map: Reference calibration laser map.
    
    :param calib_laser_nm: Wavelength of the calibration laser in nm.
    
    :param pixel_size: (Optional) Size of the CCD pixels in um
      (default 15).
    
    :param binning: (Optional) Maps are binned to accelerate the
      process. Set the binning factor (default 4).

    :return: a tuple (fitted phase map, error map, fit error, new calibration laser map)
    """
    def model_laser_map(p, calib, calib_laser_nm, pixel_size):
        return simulate_calibration_laser_map(
            calib.shape[0], calib.shape[1], pixel_size,
            calib_laser_nm, p[0], p[1], p[2], p[3], p[4], p[5])

    def model_phase_map(p, calib, calib_laser_nm, pixel_size):
        return p[6] + p[7] * model_laser_map(
            p[:6], calib, calib_laser_nm, pixel_size)

    def get_p(p_var, p_fix, p_ind):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        p_all = np.empty_like(p_ind, dtype=float)
        p_all[np.nonzero(p_ind == 0.)] = p_var
        p_all[np.nonzero(p_ind > 0.)] = p_fix
        return p_all

    def diff_phase_map(p_var, calib, calib_laser_nm, pixel_size, pm, pm_err, p_fix, p_ind):
        p_all = get_p(p_var, p_fix, p_ind)
        model_map = model_phase_map(p_all, calib, calib_laser_nm, pixel_size)
        result = (model_map - pm) / pm_err
        result = result[np.nonzero(~np.isnan(result))]
        return result

    def print_params(params):
        print ('> Phase map fit parameters:\n'
               + 'distance to mirror: {} cm (fixed)\n'.format(params[0]*1e-4)
               + 'X angle from the optical axis to the center: {} degrees (fixed)\n'.format(params[1])
               + 'Y angle from the optical axis to the center: {} degrees (fixed)\n'.format(params[2])
               + 'X tilt of the mirror: {} degrees\n'.format(params[3])
               + 'Y tilt of the mirror: {} degrees\n'.format(params[4])
               + 'Rotation angle of the mirror: {} degrees\n'.format(params[5])
               + 'order 0: {} radians\n'.format(params[6])
               + 'order 1: {} radians\n'.format(params[7]))


    # Data is binned to accelerate the fit
    if binning > 1:
        print '> Binning phase maps'
        phase_map_bin = nanbin_image(phase_map, binning)
        phase_map_err_bin = nanbin_image(phase_map_err, binning)
        calib_laser_map_bin = nanbin_image(calib_laser_map, binning)
    else:
        phase_map_bin = phase_map
        phase_map_err_bin = phase_map_err
        calib_laser_map_bin = calib_laser_map

    # ref calibration laser map fit
    params, new_calib_map = fit_calibration_laser_map(calib_laser_map_bin, calib_laser_nm,
                                                      pixel_size=pixel_size*binning,
                                                      binning=1)
    
    print '> Phase map fit'
    # first fit of the linear parameters
    p_ind = np.array([1,1,1,1,1,1,0,0])
    p_fix = params
    fit = scipy.optimize.leastsq(diff_phase_map,
                                 [0, 0],
                                 args=(calib_laser_map_bin,
                                       calib_laser_nm,
                                       float(pixel_size*binning),
                                       phase_map_bin,
                                       phase_map_err_bin,
                                       p_fix,
                                       p_ind),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params)

    # second fit (x and y tilt + rotation angle)
    p_fix = [params[0], params[1], params[2]]
    p_ind = np.array([1,1,1,0,0,0,0,0])
    fit = scipy.optimize.leastsq(diff_phase_map,
                                 [params[3], params[4], params[5], params[6], params[7]],
                                 args=(calib_laser_map_bin,
                                       calib_laser_nm,
                                       float(pixel_size*binning),
                                       phase_map_bin,
                                       phase_map_err_bin,
                                       p_fix, p_ind),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params)

    fitted_phase_map = model_phase_map(
        params, calib_laser_map, calib_laser_nm, pixel_size)
    
    ## Error computation
    # Creation of the error map: The error map gives the 
    # Squared Error for each point used in the fit point. 
    error_map = phase_map - fitted_phase_map

    # The square root of the mean of this map is then normalized
    # by the range of the values fitted. This gives the Normalized
    # root-mean-square deviation
    fit_error =(np.nanmean(np.sqrt(error_map**2.))
                / (np.nanmax(phase_map) - np.nanmin(phase_map)))

    new_calib_laser_map = simulate_calibration_laser_map(
        calib_laser_map.shape[0], calib_laser_map.shape[1], pixel_size,
        calib_laser_nm, params[0], params[1], params[2],
        params[3], params[4], params[5])

    return fitted_phase_map, error_map, fit_error, new_calib_laser_map
    
def interpolate_map(m, dimx, dimy):
    """Interpolate 2D data map.

    This function is robust to Nans.
    
    .. warning:: The interpolation process is much longer if Nans are
       present in the map.
    
    :param m: Map
    
    :param dimx: X dimension of the result
    
    :param dimy: Y dimension of the result
    """
    x_int = np.linspace(0, m.shape[0], dimx,
                        endpoint=False)
    y_int = np.linspace(0, m.shape[1], dimy,
                        endpoint=False)

    x_map = np.arange(m.shape[0])
    y_map = np.arange(m.shape[1])
    interp = interpolate.RectBivariateSpline(x_map, y_map, m)
    return interp(x_int, y_int)