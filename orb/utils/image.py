#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: image.py

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

import logging
import numpy as np
import scipy
import warnings
from scipy import interpolate, optimize, ndimage, signal
import bottleneck as bn

import orb.utils.validate
import orb.utils.stats
import orb.utils.vector
import orb.utils.parallel
import orb.utils.io
import orb.utils.fft
import orb.cutils

import orb.ext.zern


def correct_cosmic_rays(frame, cr_map):
    """correct cosmic rays in an image

    CRs are replaced by a weighted average of the neighbouring
    region. Weights are calculated from a 2d gaussian kernel.

    :param frame: image

    :param cr_map: cosmic ray map (1 = CR, 0 = NO_CR). must be of
      boolean type.

    """
    MEDIAN_DEG = 2
    x_range = range(MEDIAN_DEG, int(frame.shape[0] - MEDIAN_DEG - 1L))
    y_range = range(MEDIAN_DEG, int(frame.shape[1] - MEDIAN_DEG - 1L))
    x_min = np.min(x_range)
    x_max = np.max(x_range) + 1L
    y_min = np.min(y_range)
    y_max = np.max(y_range) + 1L

    
    orb.utils.validate.is_2darray(frame, object_name='frame')
    orb.utils.validate.is_2darray(cr_map, object_name='cr_map')
    if frame.shape != cr_map.shape:
        raise TypeError('frame and cr_map should have same shape')
    if cr_map.dtype != np.bool:
        raise TypeError('cr_map must be of boolean type')
    
    bad_pixels = np.nonzero(cr_map)
    
    for ibp in range(len(bad_pixels[0])):
        ix = bad_pixels[0][ibp]
        iy = bad_pixels[1][ibp]
        if (ix < x_max and iy < y_max
            and ix >= x_min and iy >= y_min):
            (med_x_min, med_x_max,
             med_y_min, med_y_max) = orb.utils.image.get_box_coords(
                ix, iy, MEDIAN_DEG*2+1,
                x_min, x_max, y_min, y_max)
            box = frame[med_x_min:med_x_max,
                        med_y_min:med_y_max]

            # definition of the kernel. It must be
            # adjusted to the real box
            ker = orb.cutils.gaussian_kernel(MEDIAN_DEG)
            if (box.shape[0] != MEDIAN_DEG*2+1
                or box.shape[1] != MEDIAN_DEG*2+1):
                if ix - med_x_min != MEDIAN_DEG:
                    ker = ker[MEDIAN_DEG - (ix - med_x_min):,:]
                if iy - med_y_min != MEDIAN_DEG:
                    ker = ker[:,MEDIAN_DEG - (iy - med_y_min):]
                if med_x_max - ix != MEDIAN_DEG + 1:
                    ker = ker[:- (MEDIAN_DEG + 1
                                  - (med_x_max - ix)),:]
                if med_y_max - iy != MEDIAN_DEG + 1:
                    ker = ker[:,:- (MEDIAN_DEG + 1
                                    - (med_y_max - iy))]

            # cosmic rays are removed from the
            # weighted average (their weight is set to
            # 0)
            ker *=  1 - cr_map[med_x_min:med_x_max,
                               med_y_min:med_y_max]

            # pixel is replaced by the weighted average
            if np.sum(ker) != 0:
                frame[ix, iy] = np.sum(box * ker)/np.sum(ker)
            else:
                # if no good pixel around can be found
                # the pixel is replaced by the median
                # of the whole box
                frame[ix, iy] = np.median(box)
                
    return frame

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
                           sigma=3., ncpus=0):
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
    job_server, ncpus = orb.utils.parallel.init_pp_server(ncpus=ncpus)
    divs = np.linspace(0, frames.shape[0], ncpus + 1).astype(int)
    result = np.empty((frames.shape[0], frames.shape[1]), dtype=float)
    
    frames = check_frames(frames)
    
    jobs = [(ijob, job_server.submit(
        create_master_frame, 
        args=(frames[divs[ijob]:divs[ijob+1],:,:],
              combine, reject, sigma, True, False),
        modules=("import logging",
                 "import numpy as np",
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
        logging.info('Median levels: %s'%str(z_median))
        logging.info('Rejected: %s'%str(bad_frames))
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

    if not silent: logging.info("Rejection operation: %s"%reject)
    if not silent: logging.info("Combining operation: %s"%combine)

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
        if not silent: logging.info("Maximum number of rejected pixels: %d"%np.max(reject_count_frame))
        if not silent: logging.info("Mean number of rejected pixels: %f"%np.mean(reject_count_frame))

    logging.info("median std of combined frames: {}".format(
        orb.utils.stats.robust_median(std_frame)))
    
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

def fit_map_theta(data_map, err_map, theta_map):
    """fit any data map with respect to theta. Given the corresponding theta map.

    :param data_map: Data map to fit
    :param err_map: Uncertainty on the data map
    :param theta_map: Theta map (in degree)
    """
    orb.utils.validate.is_2darray(data_map)
    orb.utils.validate.have_same_shape((data_map, err_map, theta_map))
    KNOTS_NB = 100
    thetas = np.linspace(np.nanpercentile(theta_map,.1),
                         np.nanpercentile(theta_map,99.9),
                         KNOTS_NB)
    okpix = np.zeros_like(data_map, dtype=bool)
    okpix[np.nonzero(data_map)] = True
    okpix[np.nonzero(np.isnan(data_map))] = False
    okpix[np.nonzero(np.isinf(data_map))] = False

    means = list()
    sdevs = list()
    okthetas = list()
    
    pixmap = np.zeros_like(data_map, dtype=bool)
    for i in range(thetas.size-1):
        ipix = np.nonzero((theta_map > thetas[i])
                          * (theta_map <= thetas[i+1])
                          * okpix.astype(float))

        pixmap[ipix] = True

        dist = data_map[ipix]
        dist = orb.utils.stats.sigmacut(dist)
        med = np.nanmedian(dist)
        std = np.nanstd(dist)
        if not np.isnan(med) and not np.isnan(std):
            means.append(med)
            sdevs.append(std)
            okthetas.append(np.nanmedian(theta_map[ipix]))
    
    w = 1. / np.array(sdevs)
    w /= np.nanmax(w)
    model = interpolate.UnivariateSpline(okthetas, means, w=w, ext=0, k=3, s=None)
    model_err = interpolate.UnivariateSpline(okthetas, sdevs, w=w, ext=0, k=3, s=None)   

    err = (model(theta_map) - data_map)[np.nonzero(pixmap)]
    err = orb.utils.stats.sigmacut(err, sigma=2.5)
    logging.info('modeling error: {} (uncertainty on data: {})'.format(
        np.nanstd(err),
        np.nanmedian(sdevs)))


    return okthetas, model, model_err

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
    data_map_big = np.zeros(list((np.array(data_map.shape) * np.sqrt(2.) + 1).astype(int)),
                            dtype=float)
    borders = ((np.array(data_map_big.shape) - np.array(data_map.shape))/2.).astype(int)
    data_map_big[borders[0]:borders[0]+data_map.shape[0],
                 borders[1]:borders[1]+data_map.shape[1]] = np.copy(data_map)
    mask = np.zeros_like(data_map_big, dtype=float)
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
    logging.info('Standard deviation of the residual: {}'.format(fit_res_std))
    logging.info('Median relative error (err/val)): {:.2f}%'.format(
        fit_error * 100.))

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
    phi_x = phi_x / 180. * np.pi
    phi_y = phi_y / 180. * np.pi
    phi_r = phi_r / 180. * np.pi
    
    xc, yc = cmap.shape[0]/2, cmap.shape[1]/2
    theta_map = np.arccos(calib_laser_nm / cmap)
    theta_c = theta_map[xc, yc]
    theta_map -= theta_c
    
    X, Y = np.mgrid[0:cmap.shape[0], 0:cmap.shape[1]].astype(float)
    X -= xc
    Y -= yc
    
    alpha = np.pi/2. - theta_map

    if phi_r != 0:
        Xr = X * np.cos(phi_r) - Y * np.sin(phi_r)
        Yr = X * np.sin(phi_r) + Y * np.cos(phi_r)
        X = Xr
        Y = Yr
    
    if phi_x != 0.:
        alpha2 = np.pi/2. - theta_map - phi_x
        ratio = np.sin(alpha) / np.sin(alpha2)
        X *= ratio
        
    if phi_y != 0.:
        alpha2 = np.pi/2. - theta_map - phi_y
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

def simulate_theta_map(nx, ny, pixel_size,
                       mirror_distance,
                       theta_cx, theta_cy,
                       phi_x, phi_y, phi_r):
    """Simulate incident angle (theta) map from optical and mechanical
    parameters

    :param nx: Number of pixels along X
    
    :param ny: Number of pixels along Y
    
    :param pixel_size: Size of a pixel in microns
    
    :param mirror_distance: Distance to the mirror in microns on the
      optical axis.
    
    :param theta_cx: Angle from the optical axis to the mirror
      center in degrees along X axis (in degrees)

    :param theta_cy: Angle from the optical axis to the mirror
      center in degrees along Y axis (in degrees)

    :param phi_x: Tilt of the mirror along X in degrees
    
    :param phi_y: Tilt of the mirror along Y in degrees
    
    :param phi_r: Rotation angle of the camera in degrees
    """
    
    def x2theta(x, D, alpha):
        return (np.arctan(
            (x - D) / (x + D)
            / np.tan((alpha + np.pi / 2.)/2.))
                - alpha / 2. + np.pi / 4.)

    def central_angle(phiX, phiY):
        """from Vincenty's formula"""
        return np.arctan2(
            np.sqrt(
                (np.cos(phiY) * np.sin(phiX))**2.
                + np.sin(phiY)**2.),
            np.cos(phiY) * np.cos(phiX))
    
    theta_cx = np.deg2rad(theta_cx)
    theta_cy = np.deg2rad(theta_cy)
    phi_x = np.deg2rad(phi_x)
    phi_y = np.deg2rad(phi_y)
    phi_r = np.deg2rad(phi_r)

    X, Y = np.mgrid[:nx,:ny].astype(float)
    X -= nx / 2. - 0.5
    Y -= ny / 2. - 0.5
    X *= pixel_size
    Y *= pixel_size
    
    Xr = X * np.cos(phi_r) + Y * np.sin(phi_r)
    Yr = Y * np.cos(phi_r) - X * np.sin(phi_r)

    thetax = theta_cx + x2theta(Xr, mirror_distance, phi_x)
    thetay = theta_cy + x2theta(Yr, mirror_distance, phi_y)

    return np.rad2deg(central_angle(thetax, thetay))

    #return np.rad2deg(np.sqrt(thetax**2 + thetay**2))

def simulate_calibration_laser_map(nx, ny, pixel_size,
                                   mirror_distance,
                                   theta_cx, theta_cy,
                                   phi_x, phi_y, phi_r,
                                   calib_laser_nm):

    """Simulate a calibration laser map from optical and mechanical
    parameters

    :param nx: Number of pixels along X
    
    :param ny: Number of pixels along Y
    
    :param pixel_size: Size of a pixel in microns
    
    :param mirror_distance: Distance to the mirror in microns on the
      optical axis.
    
    :param theta_cx: Angle from the optical axis to the mirror
      center in degrees along X axis (in degrees)

    :param theta_cy: Angle from the optical axis to the mirror
      center in degrees along Y axis (in degrees)

    :param phi_x: Tilt of the mirror along X in degrees
    
    :param phi_y: Tilt of the mirror along Y in degrees
    
    :param phi_r: Rotation angle of the camera in degrees
    
    :param calib_laser_nm: Calibration laser wavelength in nm
    """
    return calib_laser_nm / np.cos(np.deg2rad(simulate_theta_map(
        nx, ny, pixel_size, mirror_distance,
        theta_cx, theta_cy, phi_x, phi_y, phi_r)))

def nanbin_image(im, binning):
    """Mean image (or cube) binning robust to NaNs.

    :param im: Image or cube to bin
    :param binning: Binning factor (must be an integer)

    .. note:: adapted from https://stackoverflow.com/questions/6163334/binning-data-in-python-with-scipy-numpy.
    """
    if not isinstance(im, np.ndarray): raise ValueError('Image must be a numpy.ndarray')
    if not im.ndim in [2, 3]: raise ValueError('Array dimensions must be 2 or 3')
    s0 = int(im.shape[0]//binning)
    s1 = int(im.shape[1]//binning)
    im_view = np.copy(im[:s0 * binning, :s1 * binning, ...])
    if im_view.size > 0:
        im_view = im_view.reshape(int(im.shape[0]//binning), binning,
                                  int(im.shape[1]//binning), binning,
                                  -1)
        return np.squeeze(np.nanmean(np.nanmean(im_view, axis=3), axis=1))
    else:
        return np.nanmean(im).reshape((1,1))
    #return orb.cutils.nanbin_image(im.astype(np.float64), int(binning))


def fit_calibration_laser_map(calib_laser_map, calib_laser_nm, pixel_size=15.,
                              binning=4, mirror_distance_guess=2.4e5,
                              return_model_fit=False):
    """
    Fit a calibration laser map.

    Fit an opto-mechanical model first and uses Zernike polynomials to
    fit the residual wavefront error.

    The model is based on optical parameters.
    
    :param calib_laser_map: Reference calibration laser map.
    
    :param calib_laser_nm: Wavelength of the calibration laser in nm.
    
    :param pixel_size: (Optional) Size of the CCD pixels in um
      (default 15).
    
    :param binning: (Optional) Maps are binned to accelerate the
      process. Set the binning factor (default 4).

    :param mirror_distance_guess: (Optional) Guess on the mirror
      distance in um (default 2.2e5).

    :param return_model_fit: (Optional) If True the optical model fit
      is also returned (i.e. without the wavefront modeling with Zernike
      polynomials) (default False).

    .. note:: Zernike polynomial fit routine has been written by Tim
      van Werkhoven (werkhoven@strw.leidenuniv.nl) as a part of
      libtim. It can be found in ORB module in ./ext/zern.py.
    """

    def model_laser_map(p, nx, ny, pixel_size):
        """
        0: mirror_distance
        1: theta_cx
        2: theta_cy
        3: phi_x
        4: phi_y
        5: phi_r    
        """
        return simulate_calibration_laser_map(
            nx, ny, pixel_size,
            p[0], p[1], p[2], p[3], p[4], p[5], p[6])

    def diff_laser_map(p_var, p_fix, p_ind, calib, pixel_size):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """

        p = get_p(p_var, p_fix, p_ind)
        res = model_laser_map(p, calib.shape[0], calib.shape[1], pixel_size)
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
        logging.info(('    > Calibration laser map fit parameters:\n'
               + '    distance to mirror: {} cm {}\n'.format(
                   params[0] * 1e-4, print_fix(0))
               + '    X angle from the optical axis to the center: {} degrees {}\n'.format(
                   np.fmod(float(params[1]),360), print_fix(1))
               + '    Y angle from the optical axis to the center: {} degrees {}\n'.format(
                   np.fmod(float(params[2]),360), print_fix(2))
               + '    Tip-tilt angle of the detector along X: {} degrees {}\n'.format(
                   np.fmod(float(params[3]),360), print_fix(3))
               + '    Tip-tilt angle of the detector along Y: {} degrees {}\n'.format(
                   np.fmod(float(params[4]),360), print_fix(4))
               + '    Rotation angle of the detector: {} degrees {}\n'.format(
                   np.fmod(float(params[5]),360), print_fix(5))
               + '    Calibration laser wavelength: {} nm {}\n'.format(
                   params[6], print_fix(6))
               + '    Error on fit: mean {}, std {} (in nm)\n'.format(
                   np.nanmean(fvec), np.nanstd(fvec))
               + '    Error on fit: mean {}, std {} (in km/s)'.format(
                   np.nanmean(fvec)/calib_laser_nm*3e5,
                   np.nanstd(fvec)/calib_laser_nm*3e5)))

    CENTER_COEFF = 0.3
    LARGE_COEFF = 0.5 # 0.95
    ZERN_MODES = 20 # number of Zernike modes to fit 
    BORDER_SIZE = 10 # in pixels
    ANGLE_RANGE = 4 # in degrees

    # compute angle at the exact center of the map
    cx = calib_laser_map.shape[0]/2.
    cy = calib_laser_map.shape[1]/2.
    center_calib_nm = np.nanmean(
        calib_laser_map[int(cx-0.5):int(np.ceil(cx-0.5+1)),
                        int(cy-0.5):int(np.ceil(cy-0.5+1))])
    
    theta_c = np.acos(calib_laser_nm/center_calib_nm) / np.pi * 180.
    logging.info('Angle at the center of the frame: {}'.format(theta_c))

    # filter calibration laser map
    value_min = calib_laser_nm / np.cos((theta_c - ANGLE_RANGE)/180.*np.pi)
    value_max = calib_laser_nm / np.cos((theta_c + ANGLE_RANGE)/180.*np.pi)
    calib_laser_map[np.nonzero(calib_laser_map > value_max)] = np.nan
    calib_laser_map[np.nonzero(calib_laser_map < value_min)] = np.nan

    # remove borders
    calib_laser_map[:BORDER_SIZE,:] = np.nan
    calib_laser_map[-BORDER_SIZE:,:] = np.nan
    calib_laser_map[:,:BORDER_SIZE] = np.nan
    calib_laser_map[:,-BORDER_SIZE:] = np.nan
    
    logging.info('> Binning calibration map')

    binning = int(binning)
    if binning > 1:
        calib_laser_map_bin = nanbin_image(calib_laser_map, binning)
    else:
        calib_laser_map_bin = calib_laser_map

    logging.info('> Calibration laser map fit')

    ## mirror_dist, Y angle
    ## p_ind = 0: variable parameter, index=1: fixed parameter
    ## 0: mirror_distance
    ## 1: theta_cx
    ## 2: theta_cy
    ## 3: phi_x
    ## 4: phi_y
    ## 5: phi_r    


    logging.info('  > First fit on the central portion of the calibration laser map ({:.1f}% of the total size)'.format(CENTER_COEFF*100))
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(CENTER_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_center = calib_laser_map_bin[xmin:xmax,ymin:ymax]

    p_var = np.array([mirror_distance_guess, theta_c])
    p_ind = np.array([0, 1, 0, 1, 1, 1, 1])
    p_fix = np.array([0., 0., 0., 0., calib_laser_nm])
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_center,
                                       float(pixel_size*binning)),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
   
    print_params(params, fit[2]['fvec'], p_ind)

    logging.info('  > Second fit on the central portion of the calibration laser map ({:.1f}% of the total size)'.format(CENTER_COEFF*100))
    ## p_var = np.array([params[0], 0., params[2], 0., 0., 0.])
    ## p_ind = np.array([0, 0, 0, 0, 0, 0, 1])
    ## p_fix = np.array([calib_laser_nm])
    p_var = np.array([params[0], 0., params[2], 0., 0.])
    p_ind = np.array([0, 0, 0, 0, 0, 1, 1])
    p_fix = np.array([0., calib_laser_nm])

    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_center,
                                       float(pixel_size*binning)),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
   
    print_params(params, fit[2]['fvec'], p_ind)


    logging.info('  > Third fit on a larger portion of the map ({:.1f}% of the total size)'.format(LARGE_COEFF*100))
    xmin,xmax,ymin,ymax = get_box_coords(
        calib_laser_map_bin.shape[0]/2,
        calib_laser_map_bin.shape[1]/2,
        int(LARGE_COEFF*calib_laser_map_bin.shape[0]),
        0, calib_laser_map_bin.shape[0],
        0, calib_laser_map_bin.shape[1])
    calib_laser_map_bin_large = calib_laser_map_bin[xmin:xmax,ymin:ymax]
    ## p_var = np.array(params[:-1])
    ## p_ind = np.array([0, 0, 0, 0, 0, 0, 1])
    ## p_fix = np.array([calib_laser_nm])
    p_var = np.array([params[0], 0., params[2], 0., 0.])
    p_ind = np.array([0, 0, 0, 0, 0, 1, 1])
    p_fix = np.array([0., calib_laser_nm])

    
    fit = scipy.optimize.leastsq(diff_laser_map, p_var,
                                 args=(p_fix, p_ind, calib_laser_map_bin_large,
                                       float(pixel_size*binning)),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, fit[2]['fvec'], p_ind)

    ## logging.info('  > Third fit on all the map ({:.1f}% of the total size)'.format(LARGE_COEFF*100))
    ## xmin,xmax,ymin,ymax = get_box_coords(
    ##     calib_laser_map_bin.shape[0]/2,
    ##     calib_laser_map_bin.shape[1]/2,
    ##     int(LARGE_COEFF*calib_laser_map_bin.shape[0]),
    ##     0, calib_laser_map_bin.shape[0],
    ##     0, calib_laser_map_bin.shape[1])
    ## calib_laser_map_bin_large = calib_laser_map_bin[xmin:xmax,ymin:ymax]
    ## p_var = np.array(params[0:3])
    ## p_ind = np.array([0, 0, 0, 1, 1, 1, 1])
    ## p_fix = np.array([0,0,0,calib_laser_nm])
    
    ## fit = scipy.optimize.leastsq(diff_laser_map, p_var,
    ##                              args=(p_fix, p_ind, calib_laser_map_bin_large,
    ##                                    float(pixel_size*binning)),
    ##                              full_output=True)
    ## params = get_p(fit[0], p_fix, p_ind)
    ## print_params(params, fit[2]['fvec'], p_ind)

    
    new_calib_laser_map = model_laser_map(
        params, calib_laser_map.shape[0], calib_laser_map.shape[1], pixel_size)
    
    model_fit_calib_laser_map = np.copy(new_calib_laser_map)
    
    # Zernike fit of the diff map
    logging.info('  > Zernike polynomials fit of the residual wavefront')
    res_map = calib_laser_map - new_calib_laser_map    
    res_map[np.nonzero(res_map == 0.)] = np.nan
    res_map_fit, _err_map, _fit_error = fit_map_zernike(
        res_map, np.ones_like(res_map), ZERN_MODES)

    new_calib_laser_map += res_map_fit
    
    # final error
    std_err = np.nanstd(
        (calib_laser_map - new_calib_laser_map)[
            xmin:xmax, ymin:ymax])
    
    logging.info('> final error (std on {:.1f}% of the total size): {:.3e} nm, {:.3e} km/s'.format(LARGE_COEFF*100., std_err, std_err/calib_laser_nm*3e5))

    params = np.array(list(params) + list([theta_c]))
    if return_model_fit:
        return params, new_calib_laser_map, model_fit_calib_laser_map
    else:
        return params, new_calib_laser_map


def fit_highorder_phase_map(phase_map, err_map, calib_map, nm_laser, knb=10):
    """Robust fit phase maps of order > 1

    Uses a theta dependant fit model base on a spline. See
    py:meth:`utils.fit_map_cos`.
    
    :param phase_map: Phase map to fit

    :param err_map: Error map of phase map values

    :param calib_map: Calibration laser map.

    :param nm_laser: Calibration laser wavelength in nm.

    :return: A tuple: (Fitted map, residual map)
    """

    # WARNING: CROP_COEFF must correspond to the CROP_COEFF used in
    # fit_sitelle_phase_map

    CROP_COEFF = 0.98 # proportion of the phase map to keep when
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
    
    phase_map_fit, res_map, rms_error = fit_map_cos(phase_map, err_map, calib_map, nm_laser, knb=knb)
    logging.info(' > Residual STD after cos theta fit: {}'.format(np.nanstd(res_map)))

    return phase_map_fit, phase_map - phase_map_fit
    
    


def fit_sitelle_phase_map(phase_map, phase_map_err, calib_laser_map,
                          calib_laser_nm, pixel_size=15., binning=4,
                          return_coeffs=False, wavefront_map=None):

    """
    Fit a SITELLE phase map (order 0 map of the phase) using a
    model based on a simulated calibration laser map.

    A real calibration laser map is needed first to get an initial guess
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

    :param wavefront_map: (Optional) Residual between the modeled
      calibration laser map and the real laser map. This residual can
      generally be fitted with Zernike polynomials. If given, the
      wavefront is considered stable and is removed before the model
      is fitted (default None).

    :return: a tuple (fitted phase map, error map, fit error, new
      calibration laser map) + a tuple of transformation coefficients
      (a0 and a1) if return_coeffs is True.
      
    """
    def model_laser_map(p, calib, calib_laser_nm, pixel_size):
        return simulate_calibration_laser_map(
            calib.shape[0], calib.shape[1], pixel_size,
            p[0], p[1], p[2], p[3], p[4], p[5], calib_laser_nm)

    def model_phase_map(p, calib, calib_laser_nm, pixel_size, poly_deg, wf_map):
        _model_calib_map = model_laser_map(
            p[poly_deg + 1:], calib, calib_laser_nm, pixel_size)
        _model_calib_map += wf_map
        return orb.utils.fft.calib_map2phase_map0(
            p[:poly_deg + 1],
            _model_calib_map,
            calib_laser_nm)

    def get_p(p_var, p_fix, p_ind):
        """p_ind = 0: variable parameter, index=1: fixed parameter
        """
        p_all = np.empty_like(p_ind, dtype=float)
        p_all[np.nonzero(p_ind == 0.)] = p_var
        p_all[np.nonzero(p_ind > 0.)] = p_fix
        return p_all

    def diff_phase_map(p_var, calib, calib_laser_nm, pixel_size, pm,
                       pm_err, p_fix, p_ind, poly_deg, wf_map):
        p_all = get_p(p_var, p_fix, p_ind)
        model_map = model_phase_map(p_all, calib, calib_laser_nm, pixel_size,
                                    poly_deg, wf_map)
        result = (model_map - pm) / pm_err
        result[np.isinf(result)] = np.nan
        result = result[np.nonzero(~np.isnan(result))]
        return result

    def print_params(params, p_ind, poly_deg):
        def str_fix(i):
            if p_ind[i] == 1: return '(fixed)'
            else: return ''
        def ang(_a):
            return np.fmod(float(_a),360.)

        poly_str = ''.join(['a{}: {} radians {}\n'.format(
            i, params[i], str_fix(i)) for i in range(poly_deg + 1)])
        _i = poly_deg + 1
        logging.info(('> Phase map fit parameters:\n'
               + poly_str
               + 'distance to mirror: {} cm {}\n'.format(
                   params[_i]*1e-4, str_fix(_i))
               + 'X angle from the optical axis: {} degrees {}\n'.format(
                   ang(params[_i+1]), str_fix(_i+1))
               + 'Y angle from the optical axis: {} degrees {}\n'.format(
                   ang(params[_i+2]), str_fix(_i+2))
               + 'Tilt along X: {} degrees {}\n'.format(
                   ang(params[_i+3]), str_fix(_i+3))
               + 'Tilt along Y: {} degrees {}\n'.format(
                   ang(params[_i+4]), str_fix(_i+4))
               + 'Rotation angle: {} degrees {}\n'.format(
                   ang(params[_i+5]), str_fix(_i+5))))

    # WARNING: CROP_COEFF must correspond to the CROP_COEFF used in
    # fit_highorder_phase_map

    CROP_COEFF = 0.98 # proportion of the phase map to keep when
                      # cropping

    POLY_DEG = 2 # degree of the polyomial used to transform a
                 # calibration map in a phase map

    # bad values are filtered and phase map is cropped to remove
    # borders with erroneous phase values.
    phase_map[np.nonzero(phase_map==0)] = np.nan
    uncropped_phase_map = np.copy(phase_map)

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

    if wavefront_map is None:
        wavefront_map = np.zeros_like(phase_map)
    
    # Data is binned to accelerate the fit
    if binning > 1:
        logging.info('> Binning phase maps')
        phase_map_bin = nanbin_image(phase_map, binning)
        phase_map_err_bin = nanbin_image(phase_map_err, binning)
        calib_laser_map_bin = nanbin_image(calib_laser_map, binning)
        wavefront_map_bin = nanbin_image(wavefront_map, binning)
    else:
        phase_map_bin = phase_map
        phase_map_err_bin = phase_map_err
        calib_laser_map_bin = calib_laser_map
        wavefront_map_bin = wavefront_map

    calib_laser_map_bin -= wavefront_map_bin

    # ref calibration laser map fit
    calib_fit_params, _, _ = fit_calibration_laser_map(
        calib_laser_map_bin, calib_laser_nm,
        pixel_size=pixel_size*binning,
        binning=1, return_model_fit=True)
    
    ## Output of fit_calibration_laser_map
    ## 0: mirror_distance
    ## 1: theta_cx
    ## 2: theta_cy
    ## 3: phi_x
    ## 4: phi_y
    ## 5: phi_r
    ## 6: calib_laser_nm (unused)
    ## 7: theta c (unused, control value, not a fit parameter)
    
    logging.info('> Phase map fit')
    ## 0: a0
    ## 1: a1
    ## 2: mirror_distance
    ## 3: theta_cx
    ## 4: theta_cy
    ## 5: phi_x
    ## 6: phi_y
    ## 7: phi_r

    # first fit of the linear parameters
    p_ind = np.array(list([0])*(POLY_DEG+1) + [1,1,1,1,1,1])
    p_fix = calib_fit_params[:-2]
    fit = scipy.optimize.leastsq(diff_phase_map,
                                 [0., 0., 0.],
                                 args=(calib_laser_map_bin,
                                       calib_laser_nm,
                                       float(pixel_size*binning),
                                       phase_map_bin,
                                       phase_map_err_bin,
                                       p_fix, p_ind, POLY_DEG,
                                       wavefront_map_bin),
                                 full_output=True)
    params = get_p(fit[0], p_fix, p_ind)
    print_params(params, p_ind, POLY_DEG)
    res_phase_map = phase_map - model_phase_map(
        params, calib_laser_map, calib_laser_nm, pixel_size, POLY_DEG,
        wavefront_map)
    logging.info('residual std: {} (flux error: {}%)'.format(
        np.nanstd(res_phase_map),
        100 * (1. - np.cos(np.nanstd(res_phase_map)))))


    ## # second fit
    ## p_ind = np.array(list([1])*(POLY_DEG+1) + [1,0,1,1,1,0])
    ## p_fix = params[p_ind.astype(bool)]
    ## p_var = params[~p_ind.astype(bool)]
    ## fit = scipy.optimize.leastsq(diff_phase_map,
    ##                              p_var,
    ##                              args=(calib_laser_map_bin,
    ##                                    calib_laser_nm,
    ##                                    float(pixel_size*binning),
    ##                                    phase_map_bin,
    ##                                    phase_map_err_bin,
    ##                                    p_fix, p_ind, POLY_DEG,
    ##                                    wavefront_map_bin),
    ##                              full_output=True)
    ## params = get_p(fit[0], p_fix, p_ind)
    ## print_params(params, p_ind, POLY_DEG)
    ## res_phase_map = phase_map - model_phase_map(
    ##     params, calib_laser_map, calib_laser_nm, pixel_size, POLY_DEG,
    ##     wavefront_map)
    ## logging.info('residual std: {} (flux error: {}%)'.format(
    ##     np.nanstd(res_phase_map),
    ##     100 * (1. - np.cos(np.nanstd(res_phase_map)))))

    ## # third fit
    ## p_ind = np.array(list([0])*(POLY_DEG+1) + [1,0,0,1,1,0])   
    ## p_fix = params[p_ind.astype(bool)]
    ## p_var = params[~p_ind.astype(bool)]
    ## fit = scipy.optimize.leastsq(diff_phase_map,
    ##                              p_var,
    ##                              args=(calib_laser_map_bin,
    ##                                    calib_laser_nm,
    ##                                    float(pixel_size*binning),
    ##                                    phase_map_bin,
    ##                                    phase_map_err_bin,
    ##                                    p_fix, p_ind, POLY_DEG,
    ##                                    wavefront_map_bin),
    ##                              full_output=True)
    ## params = get_p(fit[0], p_fix, p_ind)
    ## print_params(params, p_ind, POLY_DEG)
    ## res_phase_map = phase_map - model_phase_map(
    ##     params, calib_laser_map, calib_laser_nm, pixel_size, POLY_DEG,
    ##     wavefront_map)
    ## logging.info('residual std: {} (flux error: {}%)'.format(
    ##     np.nanstd(res_phase_map),
    ##     100 * (1. - np.cos(np.nanstd(res_phase_map)))))

    
    fitted_phase_map = model_phase_map(
        params, calib_laser_map, calib_laser_nm, pixel_size, POLY_DEG,
        wavefront_map)

    ## computed calibration laser map from instrumental parameters
    ## deduced from the phase map fit. If the wavefront is added, this
    ## calibration laser map might be used for a better wavelength
    ## calibration.
    new_calib_laser_map = (model_laser_map(
        params[POLY_DEG +1:], calib_laser_map, calib_laser_nm, pixel_size)
                           + wavefront_map)
    
    # Residual fit
    logging.info('> Phase map residuals fit with cos theta fit')
   
    res_phase_map = uncropped_phase_map - fitted_phase_map
    
    res_phase_map_fit, _err_map, _fit_error = fit_map_cos(
        res_phase_map, np.ones_like(res_phase_map),
        new_calib_laser_map, calib_laser_nm, knb=5)
    
    fitted_phase_map += res_phase_map_fit

    ## Error computation
    # Creation of the error map: The error map gives the 
    # Squared Error for each point used in the fit point. 
    error_map = phase_map - fitted_phase_map
    error_map[np.nonzero(phase_map == 0)] = np.nan
    

    # The square root of the mean of this map is then normalized
    # by the range of the values fitted. This gives the Normalized
    # root-mean-square deviation
    fit_error_rms =(np.nanmean(np.sqrt(error_map**2.))
                / (np.nanpercentile(phase_map, 84)
                   - np.nanpercentile(phase_map, 16)))

    fit_error = np.nanstd(error_map)

    logging.info('> Final fit std: {} radians'.format(fit_error))

    if not return_coeffs:
        return fitted_phase_map, error_map, fit_error_rms, new_calib_laser_map
    else:
        return fitted_phase_map, error_map, fit_error_rms, new_calib_laser_map, [params[2], params[3]]


def fit_phase_map02calib_map(calib, pm0, nm_laser):
    """Return the best transformation parameters that permit to
    compute an order 0 phase map from a calibration laser map

    :param calib: Calibration laser map

    :param pm0: Order 0 phase map

    :param nm_laser: Calibration laser wavelength in nm.
    """
    def diff(p, calib, pm0, nm_laser):
        
        res = orb.utils.fft.calib_map2phase_map0(
            p, calib, nm_laser) - pm0
        return res[np.nonzero(~np.isnan(res))]

    p0 = [np.pi, 100]

    fit = scipy.optimize.leastsq(
        diff, p0, args=(calib, pm0, nm_laser),
        full_output=True)
    if fit[-1] < 5:
        return fit[0]
    else:
        warnings.warn('Phase map 2 calibration laser map fit failed: {}'.format(fit[-2]))
        return None

def unwrap_phase_map0(phase_map):
    """
    Phase is defined modulo pi/2. The Unwrapping is a
    reconstruction of the phase so that the distance between two
    neighbour pixels is always less than pi/4. Then the real
    phase pattern can be recovered and fitted easily.
    
    The idea is the same as with np.unwrap() but in 2D, on a
    possibly very noisy map, where a naive 2d unwrapping cannot
    be done.

    :param phase_map: Order 0 phase map.
    """

    BIN_SIZE = 20
    LINE_SIZE = 30

    def unwrap(val, target):
        while abs(val - target) > np.pi / 2.:
            if val  - target > 0. :
                val -= np.pi
            else:
                val += np.pi
        return val

    def unwrap_columns(pm0, bin_size):
        for ii in range(pm0.shape[0]):
            for ij in range(pm0.shape[1]):
                colbin = pm0[ii,ij:ij+bin_size+1]
                colbin_med = np.nanmedian(colbin)
                for ik in range(colbin.shape[0]):
                    colbin[ik] = unwrap(colbin[ik], colbin_med)
                pm0[ii,ij:ij+bin_size+1] = colbin
        return pm0

    def unwrap_all(pm0, bin_size):
        test_line = np.nanmedian(
            pm0[:, pm0.shape[1]/2-LINE_SIZE/2:pm0.shape[1]/2+LINE_SIZE/2],
            axis=1)
        test_line_init = np.copy(test_line)
        for ii in range(0, test_line.shape[0]-bin_size/2):
            linebin = test_line[ii:ii+bin_size+1]
            linebin_med = np.nanmedian(orb.utils.stats.sigmacut(
                linebin, sigma=2))
            for ik in range(linebin.shape[0]):
                linebin[ik] = unwrap(linebin[ik], linebin_med)
            test_line[ii:ii+bin_size+1] = linebin
        diff = test_line - test_line_init
        pm0 = (pm0.T + diff.T).T
        return pm0

    phase_map = np.fmod(phase_map, np.pi)

    # unwrap pixels along columns
    phase_map = unwrap_columns(phase_map, BIN_SIZE)
    # unwrap columns along a line
    phase_map = unwrap_all(phase_map, BIN_SIZE)

    phase_map[np.nonzero(np.isnan(phase_map))] = 0.

    return phase_map

        
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
    a = theta / 180 * np.pi
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
    a = theta / 180. * np.pi
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
      
    
def bf_laser_aligner(im1, im2, init_dx, init_dy, init_angle, zf,  binning=4):
    """Align two complementary laser frames (i.e. cam1 and cam2) with
    a brute force algorithm.

    :param im1: frame 1
    
    :param im2: frame 2
    
    :param init_dx: Initial alignement parameter along X axis
    
    :param init_dy: Initial alignement parameter along Y axis
    
    :param init_angle: Initial angle

    :param zf: Zoom factor

    :param binning: Binning of the data for the first pass

    .. warning:: This function returns parameters much different that
      the alignement parameters we get from stars... I don't know why
      !
    """

    def model(p, im2, xmin, xmax, ymin, ymax, zf):
        d = np.array([p[0], p[1], p[2], 0, 0])
        
        iarr = orb.utils.image.transform_frame(
            im2, xmin, xmax, ymin, ymax,
            d, (im2.shape[0]/2., im2.shape[1]/2.), zf, 1)
        im2_t = np.empty_like(im2, dtype=float)
        im2_t.fill(np.nan)
        if np.size(xmin) < 2:
            im2_t[xmin:xmax, ymin:ymax] = iarr
        else:
            for i in range(len(iarr)):
                im2_t[xmin[i]:xmax[i], ymin[i]:ymax[i]] = iarr[i]
                    
        return im2_t


    def diff(p, *args):
        im1, im2, xmin, xmax, ymin, ymax, zf = args
        res = np.nanstd(im1 + model(p, im2, xmin, xmax, ymin, ymax, zf))
        return res
    
    def get_coords(grid_len, box_size, im1):
        coords = list()

        ixs = np.linspace(0, im1.shape[0], grid_len + 2)
        ixs = ixs[1:-1]
        iys = np.linspace(0, im1.shape[1], grid_len + 2)
        iys = iys[1:-1]

        for ix in ixs:
            for iy in iys:
                coords.append(
                    orb.utils.image.get_box_coords(
                        ix, iy, box_size,
                        0, im1.shape[0],
                        0, im1.shape[1]))
        coords = np.array(coords)
        xmin = list(coords[:,0])
        xmax = list(coords[:,1])
        ymin = list(coords[:,2])
        ymax = list(coords[:,3])
        return xmin, xmax, ymin, ymax

    def bf_by_angle(im1_mod, im2, init_dx, init_dy, iangle, xmin, xmax, ymin, ymax, bf_range, zf):

        _xmin, _xmax, _ymin, _ymax = orb.utils.image.get_box_coords(
            im2.shape[0]/2, im2.shape[0]/2, min(im2.shape) - 2 * bf_range -1,
            0, im2.shape[0],
            0, im2.shape[1])

        _res = list()        
        for idx in range(-bf_range, bf_range+1):
            test = list()
            for idy in range(-bf_range, bf_range+1):
                imod = model([init_dx + idx, init_dy + idy, iangle],
                     im2, xmin, xmax, ymin, ymax, zf)
                _imod = imod[_xmin:_xmax, _ymin:_ymax]
                _std = np.nanstd(im1_mod +_imod)
                _res.append((_std, idx, idy, iangle))
                test.append(_std)
        return _res    
        
    BF_RANGE = int(10 / float(binning)) + 1
    BF_R_RANGE = 1.
    BF_R_STEPS = (int(BF_R_RANGE) + 1) * 32

    BF2_RANGE = 1 * binning
    BF2_R_RANGE = 2 * BF_R_RANGE / float(BF_R_STEPS)
    BF2_R_STEPS = 16

    GRID_LEN = 5
    BOX_SIZE = 100
    XY_RANGE = 1
    XY_STEPS = 3 * XY_RANGE
    R_RANGE = 0.2
    R_STEPS = 5

    # first pass on binned data
    im1_full = np.copy(im1)
    im1 = nanbin_image(im1, binning)
    im2_full = np.copy(im2)
    im2 = nanbin_image(im2, binning)
    
    xmin, xmax, ymin, ymax = 0, im2.shape[0], 0, im2.shape[1]
    
    angles = np.linspace(init_angle - BF_R_RANGE, init_angle + BF_R_RANGE, BF_R_STEPS)
    
    job_server, ncpus = orb.utils.parallel.init_pp_server()

    _xmin, _xmax, _ymin, _ymax = orb.utils.image.get_box_coords(
        im1.shape[0]/2, im1.shape[0]/2, min(im1.shape) - 2 * BF_RANGE - 1,
        0, im1.shape[0],
        0, im1.shape[1])
    im1_mod = im1[_xmin:_xmax, _ymin:_ymax]

    res = list()
    for ik in range(0, len(angles), ncpus):
        
        # no more jobs than frames to compute
        if (ik + ncpus >= len(angles)):
            ncpus = len(angles) - ik

        logging.info('computing angles from {} to {}'.format(
            angles[ik], angles[ik + ncpus - 1]))
        logging.info(' > computing dx from {} to {}'.format(
            (init_dx / float(binning) - BF_RANGE) * binning,
            (init_dx / float(binning) + BF_RANGE + 1) * binning))
        logging.info(' > computing dy from {} to {}'.format(
            (init_dy / float(binning) - BF_RANGE) * binning,
            (init_dy / float(binning) + BF_RANGE + 1) * binning))

        jobs = [(ijob, job_server.submit(
            bf_by_angle, 
            args=(im1_mod, im2, init_dx / float(binning),
                  init_dy / float(binning),
                  angles[ik + ijob], xmin, xmax, ymin, ymax, BF_RANGE, zf),
            modules=("import logging",
                     "numpy as np", 
                     "import orb.utils.image"),
            depfuncs=(model,)))
                for ijob in range(ncpus)]
        
        for ijob, job in jobs:
            res += job()
   
    orb.utils.parallel.close_pp_server(job_server)
    
    res = np.array(res)
    
    _min = np.nanargmin(res[:,0])
    best_init = res[_min,1:]

    init_dx = init_dx / float(binning) + best_init[0]
    init_dy = init_dy / float(binning) + best_init[1]
    init_angle = best_init[2]

    im1 = im1_full
    im2 = im2_full
    init_dx *= binning
    init_dy *= binning

    logging.info('first pass best init parameters: ', init_dx, init_dy, init_angle)

    # second pass on non-binnned data    
    xmin, xmax, ymin, ymax = 0, im2.shape[0], 0, im2.shape[1]
    
    angles = np.linspace(init_angle - BF2_R_RANGE, init_angle + BF2_R_RANGE, BF2_R_STEPS)
    
    job_server, ncpus = orb.utils.parallel.init_pp_server()

    _xmin, _xmax, _ymin, _ymax = orb.utils.image.get_box_coords(
        im1.shape[0]/2, im1.shape[0]/2, min(im1.shape) - 2 * BF2_RANGE - 1,
        0, im1.shape[0],
        0, im1.shape[1])
    im1_mod = im1[_xmin:_xmax, _ymin:_ymax]

    res = list()
    for ik in range(0, len(angles), ncpus):
        
        # no more jobs than frames to compute
        if (ik + ncpus >= len(angles)):
            ncpus = len(angles) - ik

        logging.info('computing angles from {} to {}'.format(
            angles[ik], angles[ik + ncpus - 1]))
        logging.info(' > computing dx from {} to {}'.format(
            init_dx - BF2_RANGE, init_dx + BF2_RANGE + 1))
        logging.info(' > computing dy from {} to {}'.format(
            init_dy - BF2_RANGE, init_dy + BF2_RANGE + 1))

        jobs = [(ijob, job_server.submit(
            bf_by_angle, 
            args=(im1_mod, im2, init_dx,
                  init_dy,
                  angles[ik + ijob], xmin, xmax, ymin, ymax, BF2_RANGE, zf),
            modules=("import logging",
                     "numpy as np", 
                     "import orb.utils.image"),
            depfuncs=(model,)))
                for ijob in range(ncpus)]
        
        for ijob, job in jobs:
            res += job()
    orb.utils.parallel.close_pp_server(job_server)
    
    res = np.array(res)
    
    _min = np.nanargmin(res[:,0])
    best_init = res[_min,1:]
    
    init_dx += best_init[0]
    init_dy += best_init[1]
    init_angle = best_init[2]

    logging.info('second pass best init parameters: ', init_dx, init_dy, init_angle)
    
    # finer pass
    xmin, xmax, ymin, ymax = get_coords(GRID_LEN, BOX_SIZE, im1)
    logging.info('finer brute force optimization')

    return scipy.optimize.brute(
        diff,
        (slice(init_dx - XY_RANGE, init_dx + XY_RANGE, XY_STEPS * 1j),
         slice(init_dy - XY_RANGE, init_dy + XY_RANGE, XY_STEPS * 1j),
         slice(init_angle - R_RANGE, init_angle + R_RANGE, R_STEPS * 1j)),
        args=(im1, im2, xmin, xmax, ymin, ymax, zf))

            
def crop_pixel_positions(pixs, xmin, xmax, ymin, ymax):
    """Returned a croped list of pixels position
    """
    bounds = (int(xmin), int(xmax), int(ymin), int(ymax))
    pixs = [np.copy(pixs[0]), np.copy(pixs[1])]
    pixs[1] = pixs[1][(pixs[0] >= bounds[0]) * (pixs[0] < bounds[1])]
    pixs[0] = pixs[0][(pixs[0] >= bounds[0]) * (pixs[0] < bounds[1])]
    pixs[0] = pixs[0][(pixs[1] >= bounds[2]) * (pixs[1] < bounds[3])]
    pixs[1] = pixs[1][(pixs[1] >= bounds[2]) * (pixs[1] < bounds[3])]
    pixs[0] -= bounds[0]
    pixs[1] -= bounds[2]
    return pixs
