#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: image.py

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

import numpy as np
import math
import scipy
import warnings
from scipy import interpolate, optimize, ndimage, signal
import bottleneck as bn

import orb.utils.stats
import orb.utils.vector
import orb.utils.parallel
import orb.utils.io
import orb.cutils

import orb.ext.zern

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
    job_server, ncpus = orb.utils.parallel.init_pp_server()
    divs = np.linspace(0, frames.shape[0], ncpus + 1).astype(int)
    result = np.empty((frames.shape[0], frames.shape[1]), dtype=float)
    
    frames = check_frames(frames)
    
    jobs = [(ijob, job_server.submit(
        create_master_frame, 
        args=(frames[divs[ijob]:divs[ijob+1],:,:],
              combine, reject, sigma, True, False),
        modules=("import numpy as np",
                 "import orb.cutils as cutils",
                 "import orb.utils.stats",
                 "import warnings")))
            for ijob in range(ncpus)]
    
    for ijob, job in jobs:
        result[divs[ijob]:divs[ijob+1],:] = job()
        
    orb.utils.parallel.close_pp_server(job_server)
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

        if combine == 'average':
            return np.nanmean(frames, axis=2)
        else:
            return np.nanmedian(frames, axis=2)
        
    if reject not in ['sigclip', 'minmax', 'avsigclip']:
        raise Exception("Rejection operation must be 'sigclip', 'minmax' or 'avsigclip'")
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
        if len(good_vals) > 0.25 * np.size(column):
            interp = interpolate.UnivariateSpline(
                good_vals, column[good_vals], k=5)
            column[bad_vals] = interp(bad_vals)
            map2d[:,icol] = np.copy(column)
        else:
            map2d[:,icol] = np.nan
        
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

def fit_map_zernike(data_map, weights_map, nmodes):
    """
    Fit a map with Zernike polynomials.

    Bad values must be set to NaN (not 0.)

    :param data_map: Data map to fit

    :param weights_map: weights map (high weight value stands for high
      precision data)

    :param nmodes: Number of zernike modes to use for fitting.
    
    :return: (fitted data map, error map, fit error)
    
    .. note:: Zernike polynomial fit routine has been written by Tim
      van Werkhoven (werkhoven@strw.leidenuniv.nl) as a part of
      libtim. It can be found in ORB module in ./ext/zern.py.
    """
    # bigger version used to fit corners
    data_map_big = np.zeros(np.array(data_map.shape) * math.sqrt(2.) + 1,
                           dtype=float)
    borders = (np.array(data_map_big.shape) - np.array(data_map.shape))/2.
    data_map_big[borders[0]:borders[0]+data_map.shape[0],
                 borders[1]:borders[1]+data_map.shape[1]] = np.copy(data_map)
    mask = np.ones_like(data_map_big, dtype=float)
    weights_map = np.abs(weights_map)
    weights_map /= np.nanmax(weights_map) # error map is normalized
    mask[borders[0]:borders[0]+data_map.shape[0],
         borders[1]:borders[1]+data_map.shape[1]] = weights_map

    # nans and 0s are masked
    mask[np.nonzero(np.isnan(data_map_big))] = 0.
    mask[np.nonzero(data_map_big == 0.)] = 0.
    
    # nans are replaced by zeros in the fitted map
    data_map_big[np.nonzero(np.isnan(data_map_big))] = 0. 

    (wf_zern_vec, wf_zern_rec, fitdiff) = orb.ext.zern.fit_zernike(
        data_map_big, fitweight=mask, startmode=1, nmodes=nmodes)

    data_map_fit = wf_zern_rec[borders[0]:borders[0]+data_map.shape[0],
                               borders[1]:borders[1]+data_map.shape[1]]

    res_map = data_map - data_map_fit
    fit_error_map = np.abs(res_map) / np.abs(data_map)
    fit_error_map[np.isinf(fit_error_map)] = np.nan
    fit_res_std = np.nanstd(res_map)
    fit_error = np.nanmedian(fit_error_map)
    print 'Standard deviation of the residual: {}'.format(fit_res_std)
    print 'Median relative error (err/val)): {:.2f}%'.format(
        fit_error * 100.)

    return data_map_fit, res_map, fit_error

def fit_map(data_map, err_map, smooth_deg):
    """Fit map with low order polynomials

    :param data_map: data map

    :param err_map: error map

    :param smooth_deg: Degree of fit smoothing (beware of high
      smoothing degrees)

    :return: a tuple: (fitted data map, residual map, fit RMS error)
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

    def model(fit_coeffs, coeffs_smooth_deg, dimx, dimy):
        fit_coeffs = fit_coeffs.reshape((np.size(fit_coeffs)
                                         /(coeffs_smooth_deg + 1),
                                         coeffs_smooth_deg + 1))
        fitted_data_map = np.zeros((dimx, dimy), dtype=float)
        params_fit_list = list()
        # compute coeffs 
        for ideg in range(coeffs_smooth_deg + 1):
            params_fit_list.append(np.polynomial.polynomial.polyval(
                np.arange(dimy), fit_coeffs[ideg,:]))
        params_fit_list = np.array(params_fit_list)
        # compute map
        for ij in range(dimy):
            fitted_data_map[:,ij] = np.polynomial.polynomial.polyval(
                np.arange(dimx), params_fit_list[:, ij])
        return fitted_data_map
    
    def diff(fit_coeffs, data_map, coeffs_smooth_deg, err_map):
        fitted_data_map = model(fit_coeffs, coeffs_smooth_deg,
                                data_map.shape[0],
                                data_map.shape[1])
        
        res = (data_map - fitted_data_map)/err_map
        res = res[np.nonzero(~np.isnan(res))]
        res = res[np.nonzero(~np.isinf(res))]
        return res.flatten()
  

    coeffs_smooth_deg = smooth_deg + 1
    coeffs_list = list()
    err_list = list()
    dimx, dimy = data_map.shape
    
    ## 1st pass: fit lines independantly with a polynomial
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
            
    ## first fit returns smooth_deg + 1 coefficient for each line
    coeffs_list = np.array(coeffs_list)
    err_list = np.array(err_list)

    if np.all(np.isnan(coeffs_list)):
        raise Exception('All fit coeffs are NaNs !')
        
    ## 2nd pass: fit coeffs are smoothed with a polynomial also (which
    ## gives a list of coefficients to recover the first coefficients)
    coeffs_coeffs_list = list()
    for ideg in range(smooth_deg + 1):
        fitted_coeffs, coeffs_coeffs = smooth_fit_parameters(
            coeffs_list, err_list, ideg, coeffs_smooth_deg)
        coeffs_coeffs_list.append(coeffs_coeffs[0])
    
    coeffs_coeffs_list = np.array(coeffs_coeffs_list)
    
    ## coeffs optimization over the real data map
    fit = optimize.leastsq(diff, coeffs_coeffs_list.flatten(),
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
    res_map = data_map - fitted_data_map

    # The square root of the mean of this map is then normalized
    # by the range of the values fitted. This gives the Normalized
    # root-mean-square deviation
    fit_rms_error =(np.nanmean(np.sqrt(res_map**2.))
                / (np.nanmax(data_map) - np.nanmin(data_map)))

    return fitted_data_map, res_map, fit_rms_error


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


def simulate_calibration_laser_map(nx, ny, pixel_size, calib_laser_wl,
                                   mirror_distance,
                                   theta_c, phi_x, phi_y, phi_r):
    """Simulate a calibration laser map from optical and mechanical parameters

    :param nx: Number of pixels along X
    
    :param ny: Number of pixels along Y
    
    :param pixel_size: Size of a pixel in microns
    
    :param calib_laser_wl: Calibration laser wavelength in nm
    
    :param mirror_distance: Distance to the mirror in microns on the
      optical axis.
    
    :param theta_c: Angle from the optical axis to the mirror
      center in degrees

    :param phi_x: Tilt of the mirror along X in degrees
    
    :param phi_y: Tilt of the mirror along Y in degrees
    
    :param phi_r: angle from the center of the frame to the optical
      axis along X axis in degrees (equivalent to a rotation of the
      camera)
    """
    
    pixel_scale = math.atan(pixel_size/mirror_distance)
    phi_x = phi_x / 180. * math.pi
    phi_y = phi_y / 180. * math.pi
    phi_r = phi_r / 180. * math.pi
    theta_c = theta_c / 180. * math.pi

    X, Y = np.mgrid[0:nx, 0:ny].astype(float)
    X -= nx / 2. - 0.5
    Y -= ny / 2. - 0.5
    X *= pixel_scale * mirror_distance
    Y *= pixel_scale * mirror_distance

    X *= math.cos(phi_x)
    Y *= math.cos(phi_y)
    
    xc = phi_r * mirror_distance
    
    yc = mirror_distance * theta_c
    
    d_map = np.sqrt((X+xc)**2. + (Y+yc)**2.)
    theta_map = d_map / mirror_distance
    calib_map = calib_laser_wl/np.cos(theta_map)
    
    return calib_map


def nanbin_image(im, binning):
    """Mean image binning robust to NaNs.

    :param im: Image to bin
    :param binning: Binning factor (must be an integer)
    """     
    return orb.cutils.nanbin_image(im.astype(np.float64), int(binning))


def fit_calibration_laser_map(calib_laser_map, calib_laser_nm, pixel_size=15.,
                              binning=4, mirror_distance_guess=2.4e5):
    """
    Fit a calibration laser map.

    Fit a classic optical model first and uses Zernike polynomials to
    fit the residual.

    The model is based on optical parameters.
    
    :param calib_laser_map: Reference calibration laser map.
    
    :param calib_laser_nm: Wavelength of the calibration laser in nm.
    
    :param pixel_size: (Optional) Size of the CCD pixels in um
      (default 15).
    
    :param binning: (Optional) Maps are binned to accelerate the
      process. Set the binning factor (default 4).

    :param mirror_distance_guess: (Optional) Guess on the mirror
      distance in um (default 2.2e5).

    .. note:: Zernike polynomial fit routine has been written by Tim
      van Werkhoven (werkhoven@strw.leidenuniv.nl) as a part of
      libtim. It can be found in ORB module in ./ext/zern.py.
    """

    def model_laser_map(p, calib, calib_laser_nm, pixel_size, theta_c):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        return simulate_calibration_laser_map(
            calib.shape[0], calib.shape[1], pixel_size,
            calib_laser_nm, p[0], theta_c, 0., 0., p[1])

    def diff_laser_map(p_var, p_fix, p_ind, calib, calib_laser_nm, pixel_size,
                       theta_c):
        p = get_p(p_var, p_fix, p_ind)
        res = model_laser_map(p, calib, calib_laser_nm, pixel_size, theta_c)
        res -= calib
        res = res[np.nonzero(~np.isnan(res))]
        #res = orb.utils.stats.sigmacut(res)
        return res

    def get_p(p_var, p_fix, p_ind):
        p = np.empty_like(p_ind, dtype=float)
        p.fill(np.nan)
        p[np.nonzero(p_ind == 0)] = p_var
        p[np.nonzero(p_ind == 1)] = p_fix
        return p
        

    def print_params(params, fvec, p_ind):
        def print_fix(index):
            if p_ind[index] : return '(Fixed)'
            else: return ''
        print ('    > Calibration laser map fit parameters:\n'
               + '    distance to mirror: {} cm {}\n'.format(
                   params[0]*1e-4, print_fix(0))
               + '    X angle from the optical axis to the center: {} degrees {}\n'.format(math.fmod(float(params[1]),2.*math.pi), print_fix(1))
               + '    Error on fit: mean {}, std {} (in nm)\n'.format(
                   np.nanmean(fvec), np.nanstd(fvec))
               + '    Error on fit: mean {}, std {} (in km/s)'.format(
                   np.nanmean(fvec)/calib_laser_nm*3e5,
                   np.nanstd(fvec)/calib_laser_nm*3e5))

    CENTER_COEFF = 0.5
    LARGE_COEFF = 0.95
    ZERN_MODES = 20 # number of Zernike modes to fit 
    BORDER_SIZE = 10 # in pixels
    ANGLE_RANGE = 4 # in degrees

    # compute angle at the exact center of the map
    cx = calib_laser_map.shape[0]/2.
    cy = calib_laser_map.shape[1]/2.
    center_calib_nm = np.nanmean(
        calib_laser_map[int(cx-0.5):math.ceil(cx-0.5+1),
                        int(cy-0.5):math.ceil(cy-0.5+1)])
    
    theta_c = math.acos(calib_laser_nm/center_calib_nm) / math.pi * 180.
    print 'Angle at the center of the frame: {}'.format(theta_c)

    # filter calibration laser map
    value_min = calib_laser_nm / math.cos((theta_c - ANGLE_RANGE)/180.*math.pi)
    value_max = calib_laser_nm / math.cos((theta_c + ANGLE_RANGE)/180.*math.pi)
    calib_laser_map[np.nonzero(calib_laser_map > value_max)] = np.nan
    calib_laser_map[np.nonzero(calib_laser_map < value_min)] = np.nan

    # remove borders
    calib_laser_map[:BORDER_SIZE,:] = np.nan
    calib_laser_map[-BORDER_SIZE:,:] = np.nan
    calib_laser_map[:,:BORDER_SIZE] = np.nan
    calib_laser_map[:,-BORDER_SIZE:] = np.nan
    
    print '> Binning calibration map'

    binning = int(binning)
    if binning > 1:
        calib_laser_map_bin = nanbin_image(calib_laser_map, binning)
    else:
        calib_laser_map_bin = calib_laser_map

    print '> Calibration laser map fit'

    # mirror_dist, Y angle
    # p_ind = 0: variable parameter, index=1: fixed parameter

    print '  > First fit on the central portion of the calibration laser map ({:.1f}% of the total size)'.format(CENTER_COEFF*100)
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(CENTER_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_center = calib_laser_map_bin[xmin:xmax,ymin:ymax]

    p_var = np.array([mirror_distance_guess, 0.])
    p_ind = np.array([0, 0])
    p_fix = np.array([])
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_center,
                                       calib_laser_nm,
                                       float(pixel_size*binning), theta_c),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, fit[2]['fvec'], p_ind)

    print '  > Second fit on all the map ({:.1f}% of the total size)'.format(LARGE_COEFF*100)
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(LARGE_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_large = calib_laser_map_bin[xmin:xmax,ymin:ymax]
    p_var = np.array([params[0], params[1]])
    p_ind = np.array([0, 0])
    p_fix = np.array([])
    
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_large,
                                       calib_laser_nm,
                                       float(pixel_size*binning), theta_c),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, fit[2]['fvec'], p_ind)
    
    new_calib_laser_map = model_laser_map(
        params, calib_laser_map, calib_laser_nm, pixel_size, theta_c)
    
    # Zernike fit of the diff map
    print '  > Zernike polynomials fit of the residual'
    res_map = calib_laser_map - new_calib_laser_map
    
    res_map[np.nonzero(res_map == 0.)] = np.nan
    res_map_fit, _err_map, _fit_error = fit_map_zernike(
        res_map, np.ones_like(res_map), ZERN_MODES)

    new_calib_laser_map += res_map_fit
    
    # final error
    std_err = np.nanstd(
        (calib_laser_map - new_calib_laser_map)[
            xmin:xmax, ymin:ymax])
    
    print '> final error (std on {:.1f}% of the total size): {:.3e} nm, {:.3e} km/s'.format(LARGE_COEFF*100., std_err, std_err/calib_laser_nm*3e5)

    params = np.array(list(params) + list([theta_c]))
    return params, new_calib_laser_map



def fit_highorder_phase_map(phase_map, err_map, nmodes=10):
    """Robust fit phase maps of order > 1

    First fit pass is made with a polynomial of order 1.
    
    Second pass is a Zernike fit of the residual.

    :param phase_map: Phase map to fit

    :param err_map: Error map of phase map values

    :param nmodes: (Optional) Number of Zernike modes (default 10).

    :return: A tuple: (Fitted map, residual map)
    """
    # order 1 fit
    
    CROP_COEFF = 0.85 # proportion of the phase map to keep when
                      # cropping

    # bad values are filtered and phase map is cropped to remove
    # borders with erroneous phase values.
    phase_map[np.nonzero(phase_map==0)] = np.nan
    
    xmin,xmax,ymin,ymax = get_box_coords(
        phase_map.shape[0]/2,
        phase_map.shape[1]/2,
        int(CROP_COEFF*phase_map.shape[0]),
        0, phase_map.shape[0],
        0, phase_map.shape[1])
    phase_map[:xmin,:] = np.nan
    phase_map[xmax:,:] = np.nan
    phase_map[:,:ymin] = np.nan
    phase_map[:,ymax:] = np.nan
    
    err_map[np.nonzero(np.isnan(phase_map))] = np.nan
    
    phase_map_fit, res_map, rms_error = fit_map(phase_map, err_map, 1)
    print ' > Residual STD after 1st order fit: {}'.format(np.nanstd(res_map))
    
    # residual fit with zernike
    w_map = np.copy(err_map)
    w_map = np.abs(w_map)
    w_map = 1./w_map
    w_map /= orb.cutils.part_value(w_map.flatten(), 0.95)
    w_map[w_map > 1.] = 1.
    res_map_fit, res_res_map, fit_error = fit_map_zernike(res_map, w_map, nmodes)
    print ' > Residual STD after Zernike fit of the residual: {}'.format(np.nanstd(res_res_map))
    full_fit = phase_map_fit + res_map_fit
    
    return full_fit, phase_map - full_fit
    
    


def fit_sitelle_phase_map(phase_map, phase_map_err, calib_laser_map,
                          calib_laser_nm, pixel_size=15., binning=4,
                          return_coeffs=False):

    """Fit a SITELLE phase map (order 0 map of the phase) using a
    model based on a simulated calibration laser map..

    A real calibration laser map is needed first to get a first guess
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

    :param return_coeffs: (Optional) If True, transformation
      coefficients are returned also (default False).

    :return: a tuple (fitted phase map, error map, fit error, new
      calibration laser map) + a tuple of transformation coefficients
      (a0 and a1) if return_coeffs is True.
    """
    def model_laser_map(p, calib, calib_laser_nm, pixel_size, theta_c):
        return simulate_calibration_laser_map(
            calib.shape[0], calib.shape[1], pixel_size,
            calib_laser_nm, p[0], theta_c, 0., 0., p[1])

    def model_phase_map(p, calib, calib_laser_nm, pixel_size, theta_c):
        return orb.utils.fft.calib_map2phase_map0([p[2], p[3]],model_laser_map(
            p[:2], calib, calib_laser_nm, pixel_size, theta_c), calib_laser_nm)

    def get_p(p_var, p_fix, p_ind):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        p_all = np.empty_like(p_ind, dtype=float)
        p_all[np.nonzero(p_ind == 0.)] = p_var
        p_all[np.nonzero(p_ind > 0.)] = p_fix
        return p_all

    def diff_phase_map(p_var, calib, calib_laser_nm, pixel_size, pm,
                       pm_err, p_fix, p_ind, theta_c):
        p_all = get_p(p_var, p_fix, p_ind)
        model_map = model_phase_map(p_all, calib, calib_laser_nm, pixel_size,
                                    theta_c)
        result = (model_map - pm) / pm_err
        result = result[np.nonzero(~np.isnan(result))]
        return result

    def print_params(params):
        print ('> Phase map fit parameters:\n'
               + 'distance to mirror: {} cm (fixed)\n'.format(params[0]*1e-4)
               + 'X angle from the optical axis to the center: {} degrees (fixed)\n'.format(math.fmod(float(params[1]),2.*math.pi))
               + 'a0: {} radians\n'.format(params[2])
               + 'a1: {} radians\n'.format(params[3]))

    ZERN_MODES = 30 # number of Zernike modes to fit

    CROP_COEFF = 0.85 # proportion of the phase map to keep when
                      # cropping

    # bad values are filtered and phase map is cropped to remove
    # borders with erroneous phase values.
    phase_map[np.nonzero(phase_map==0)] = np.nan
    
    xmin,xmax,ymin,ymax = get_box_coords(
        phase_map.shape[0]/2,
        phase_map.shape[1]/2,
        int(CROP_COEFF*phase_map.shape[0]),
        0, phase_map.shape[0],
        0, phase_map.shape[1])
    phase_map[:xmin,:] = np.nan
    phase_map[xmax:,:] = np.nan
    phase_map[:,:ymin] = np.nan
    phase_map[:,ymax:] = np.nan
    
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
    params, new_calib_map = fit_calibration_laser_map(
        calib_laser_map_bin, calib_laser_nm,
        pixel_size=pixel_size*binning,
        binning=1)

    # get theta_c
    theta_c = params[-1]
    
    print '> Phase map fit'
    # first fit of the linear parameters
    p_ind = np.array([1,1,0,0])
    p_fix = params
    fit = scipy.optimize.leastsq(diff_phase_map,
                                 [0., 0.],
                                 args=(calib_laser_map_bin,
                                       calib_laser_nm,
                                       float(pixel_size*binning),
                                       phase_map_bin,
                                       phase_map_err_bin,
                                       p_fix, p_ind, theta_c),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params)

    # second fit (all params)
    p_fix = np.array([params[0]])
    p_ind = np.array([1,0,0,0])
    fit = scipy.optimize.leastsq(diff_phase_map,
                                 [params[1], params[2], params[3]],
                                 args=(calib_laser_map_bin,
                                       calib_laser_nm,
                                       float(pixel_size*binning),
                                       phase_map_bin,
                                       phase_map_err_bin,
                                       p_fix, p_ind, theta_c),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params)

    fitted_phase_map = model_phase_map(
        params, calib_laser_map, calib_laser_nm, pixel_size, theta_c)


    ### Zernike fit: Instead of fitting the residuals of the phase map,
    ## the residuals of the laser map are fitted (because it is a simple
    ## linear conversion of the laser map). This way a best fitted
    ## laser map is also obtained along with the best fitted phase
    ## map. the obtained laser map might be used for better precision.
    
    print '> Phase map residuals fit with Zernike polynomials ({} modes)'.format(ZERN_MODES)
    real_calib_laser_map = orb.utils.fft.phase_map02calib_map(
        [params[2], params[3]], phase_map, calib_laser_nm)
    
    mod_calib_laser_map = model_laser_map(
        params, calib_laser_map, calib_laser_nm, pixel_size, theta_c)
    res_map = real_calib_laser_map - mod_calib_laser_map
    
    res_map[np.nonzero(phase_map == 0.)] = np.nan # 0s are replaced
                                                  # with nans
    
    res_map_fit, _err_map, _fit_error = fit_map_zernike(
        res_map, np.ones_like(res_map), ZERN_MODES)

    fitted_laser_map = mod_calib_laser_map + res_map_fit
    fitted_phase_map = orb.utils.fft.calib_map2phase_map0(
        [params[2], params[3]], fitted_laser_map, calib_laser_nm)
    
    ## Error computation
    # Creation of the error map: The error map gives the 
    # Squared Error for each point used in the fit point. 
    error_map = phase_map - fitted_phase_map
    error_map[np.nonzero(phase_map == 0)] = np.nan
    

    # The square root of the mean of this map is then normalized
    # by the range of the values fitted. This gives the Normalized
    # root-mean-square deviation
    fit_error_rms =(np.nanmean(np.sqrt(error_map**2.))
                / (np.nanmax(phase_map) - np.nanmin(phase_map)))

    fit_error = np.nanstd(error_map)

    print '> Final fit std: {} radians'.format(fit_error)

    if not return_coeffs:
        return fitted_phase_map, error_map, fit_error_rms, fitted_laser_map
    else:
        return fitted_phase_map, error_map, fit_error_rms, fitted_laser_map, [params[2], params[3]]
    
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



def on_ellipse(x, y, x0, y0, rX, rY, theta, e=0.5):
    """Tell whether a pixel is on the ellipse or not.

    :param x: X position of the point
    :param y: Y position of the point
    :param x0: X position of the center
    :parma y0: Y position of the center
    :param theta: Angle of the ellipse (in deg)
    :param rX: Radius of the X axis
    :param rY: Radius of the Y axis
    :param e: (Optional) Precision in pixels (default 0.5).
    """
    a = theta / 180 * math.pi
    e = float(rX) / float(rY) # ellipticity
    X = (x - x0) * np.cos(a) + (y - y0) * np.sin(a)
    Y = (x - x0) * np.sin(a) - (y - y0) * np.cos(a)
    return np.abs(np.sqrt(X**2. + (e*Y)**2.) - rX) <= e


def in_ellipse(x, y, x0, y0, rX, rY, theta):
    """Tell whether a pixel is in the ellipse or not.

    :param x: X position of the point
    :param y: Y position of the point
    :param x0: X position of the center
    :parma y0: Y position of the center
    :param theta: Angle of the ellipse (in deg)
    :param rX: Radius of the X axis
    :param rY: Radius of the Y axis
    """
    a = theta / 180. * math.pi
    e = float(rX) / float(rY) # ellipticity
    X = (x - x0) * np.cos(a) + (y - y0) * np.sin(a)
    Y = (x - x0) * np.sin(a) - (y - y0) * np.cos(a)
    return np.sqrt(X**2. + (e*Y)**2.) < rX

def extract_elliptical_profile(im, x0, y0, rX, rY, theta, n=20, percentile=None):
    """Extract the elliptical profile of a source

    :param im: Image
    :param x0: X position of the center
    :parma y0: Y position of the center
    :param theta: Angle of the ellipse (in deg)
    :param rX: Radius of the X axis
    :param rY: Radius of the Y axis
    :param n: (Optional) Number of divisions (default 20)
    
    :param percentile: (Optional) percentile instead of std. Return
      (r, lmedian, [lmin, lmax]). Remember that the 1-sigma percentile
      is 15.865 for a gaussian distribution (default None).
    

    :return: a tuple (r, l, lerr) where r is the list of radiuses
      along rX, l the mean luminosity in the ellipse portion
      corresponding to the radius, lerr the standard deviation of the
      luminosity in the same portion of the ellipse.
    """
    im = im
    X, Y = np.mgrid[0:im.shape[0],0:im.shape[1]]
    l = list()
    lerr = list()
    r = list()
    coeffs = np.linspace(0, 1, n+1)[1:]
    for i in range(n):
        mask = in_ellipse(X, Y, x0, y0, coeffs[i]*rX, coeffs[i]*rY, theta)
        if i > 0:
            mask -= in_ellipse(X, Y, x0, y0, coeffs[i-1]*rX, coeffs[i-1]*rY, theta)
            r.append((coeffs[i] + coeffs[i-1])/2. * rX)
        else: r.append(coeffs[i]/2. * rX)
        l.append(np.nanmedian(im[mask]))
        if percentile is None:
            lerr.append(np.nanstd(im[mask]))
        else:
            lerr.append([
                np.nanpercentile(im[mask], percentile),
                np.nanpercentile(im[mask], 100. - percentile)])
    return np.array(r), np.array(l), np.array(lerr)
      
    
