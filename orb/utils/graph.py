#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: image.py

## Copyright (c) 2010-2020 Thomas Martin <thomas.martin.1@ulaval.ca>
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


import pylab as pl
import astropy.wcs
import matplotlib.cm
import matplotlib.colors
import numpy as np
import orb.utils.io
import orb.utils.stats
import orb.utils.validate

def imshow(data, figsize=(7,7), perc=99, cmap='viridis', wcs=None, alpha=1, ncolors=None,
           vmin=None, vmax=None, autofit=False, fig=None, interpolation=None, **kwargs):
    """Convenient image plotting function

    :param data: 2d array to show. Can be a path to a fits file.

    :param figsize: size of the figure (same as pyplot.figure's figsize keyword)

    :param perc: percentile of the data distribution used to scale
      the colorbar. Can be a tuple (min, max) or a scalar in which
      case the min percentile will be 100-perc.

    :param cmap: colormap

    :param wcs: if a astropy.WCS instance show celestial coordinates,
      Else, pixel coordinates are shown.

    :param alpha: image opacity (if another image is displayed above)

    :param ncolors: if an integer is passed, the colorbar is
      discretized to this number of colors.

    :param vmin: min value used to scale the colorbar. If set the
      perc parameter is not used.

    :param vmax: max value used to scale the colorbar. If set the
      perc parameter is not used.

    :param fig: You can pass an instance of matplotlib.Figure instead
      of creating a new one. Note that figsize will not be used in
      this case.

    :param interpolation: Interpolation method (same as pyplot.figure
      parameter)

    """
    if isinstance(data, str):
        data = orb.utils.io.read_fits(data)
    assert data.ndim == 2, 'array must have 2 dimensions'

    if np.iscomplexobj(data):
        data = data.real
    
    try:
        iter(perc)
    except Exception:
        perc = np.clip(float(perc), 50, 100)
        perc = 100-perc, perc

    else:
        if len(list(perc)) != 2:
            raise Exception('perc should be a tuple of len 2 or a single float')

    if vmin is None: vmin = np.nanpercentile(data, perc[0])
    if vmax is None: vmax = np.nanpercentile(data, perc[1])

    if ncolors is not None:
        cmap = getattr(matplotlib.cm, cmap)
        norm = matplotlib.colors.BoundaryNorm(np.linspace(vmin, vmax, ncolors),
                                              cmap.N, clip=True)
        vmin = None # must be set to None if norm is passed
        vmax = None # must be set to None if norm is passed
    else:
        norm = None

    if fig is None:
        fig = pl.figure(figsize=figsize)
        
    if wcs is not None:
        assert isinstance(wcs, astropy.wcs.WCS), 'wcs must be an astropy.wcs.WCS instance'
        
        ax = fig.add_subplot(111, projection=wcs)
        ax.coords[0].set_major_formatter('d.dd')
        ax.coords[1].set_major_formatter('d.dd')
    pl.imshow(data.T, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower', alpha=alpha,norm=norm,
              interpolation=interpolation, **kwargs)

    if autofit:
        xbounds = np.arange(data.shape[0]) * np.where(np.any(~np.isnan(data), axis=1), 1, np.nan) # x
        ybounds = np.arange(data.shape[1]) * np.where(np.any(~np.isnan(data), axis=0), 1, np.nan) # y
        xmin = np.nanmin(xbounds)
        xmax = np.nanmax(xbounds)+1
        ymin = np.nanmin(ybounds)
        ymax = np.nanmax(ybounds)+1
        pl.xlim(xmin, xmax)
        pl.ylim(ymin, ymax)

def moments(a, plot=True, median=True, **kwargs):
    if median:
        mean = orb.utils.stats.unbiased_mean(a)
    else:
        mean = np.nanmean(a)
    std = orb.utils.stats.unbiased_std(a)
    if plot:
        pl.hist(a, **kwargs)
        pl.axvline(mean, c='red', alpha=1)
        pl.axvline(mean+std, c='red', alpha=0.5)
        pl.axvline(mean-std, c='red', alpha=0.5)
        pl.text()
        print(mean, std)
    return mean, std

def scatter(x, y, c=None, vmin=None, vmax=None, perc=95, **kwargs):
    if c is not None:
        if orb.utils.validate.is_iterable(c, raise_exception=False):
            if vmin is None: vmin = np.nanpercentile(c, 100-perc)
            if vmax is None: vmax = np.nanpercentile(c, perc)
    pl.scatter(x, y, c=c, vmin=vmin, vmax=vmax)
