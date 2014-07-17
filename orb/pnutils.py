#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: pnutils.py

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
Utils functions derived from pyNeb.

pyNeb must be installed
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"                      
__docformat__ = 'reStructuredText'

import version
__version__ = version.__version__

import pyneb as pn
from orb.core import Tools
import os
import numpy as np
from scipy import ndimage
    
def get_cHbeta(ha, hb, tem=1e4, den=1e3, ext_law='CCM 89'):
    """Return the extinction coefficient c(Hbeta) from Halpha / Hbeta
    flux.

    :param ha: Flux of Halpha (can be an arbitrary unit as long as it
      is the same used for hb)

    :param hb: Flux of Hbeta (can be an arbitrary unit as long as it
      is the same used for ha)

    :param tem: (Optional) Temperature in K (default 1e4).

    :param den: (Optional) Density in cm-3 (default 1e3).

    :param ext_law: (Optional) Extinction law recongnized by Pyneb
      (default 'CCM 89').
    """
    rc = pn.RedCorr(law=ext_law)    
    th_ratio = get_Balmer_ratio(tem, den, nu=3)
    print 'Theroretical ratio: ', th_ratio
    obs_ratio = ha / hb
    rc.setCorr(obs_ratio/th_ratio, 6563., 4861.)
    return rc.cHbeta


def get_Balmer_ratio(tem, den, nu=3, try_fast=True,
                     table_name='balmer_table.fits'):
    """Return the Balmer ratio of H_nu over Hbeta (nu=4). Return
    Halpha/Hbeta ratio by default (if nu is 3).

    :param tem: Temperature in K
    
    :param dem: density in cm-3
    
    :param nu: (Optional) Transition number, must be >= 3. nu=3 is
      Halpha, nu=4 is Hbeta ... (default 3)

    :param try_fast: (Optional) If True, use a table to get the ratio
      instead of computing it. If the value is not in the table or if
      the table does not exist the computation use PyNeb function. The
      table can be created with :py:meth:`pnutils.create_balmer_table`.

    :param table_path: (Optional) Name of the table file. Must be in
      ORB data folder.
    """
    if try_fast:
        table_path = Tools(no_log=True)._get_orb_data_file_path(table_name)
        if os.path.exists(table_path):
            table, hdr = Tools(no_log=True).read_fits(table_path,
                                                      return_header=True)
            if (tem >= hdr['TEM_MIN'] and tem <= hdr['TEM_MAX']
                and den >= hdr['DEN_MIN'] and den <= hdr['DEN_MAX']):
                tem_pix = (tem - hdr['TEM_MIN'])/hdr['TEM_STEP'] 
                den_pix = (den - hdr['DEN_MIN'])/hdr['DEN_STEP'] 

                test = np.empty((2,1), dtype=float)
                test[:,0] = [tem_pix, den_pix]
                result = ndimage.map_coordinates(table, test)
                return ndimage.map_coordinates(table, test)[0]
    
    H1 = pn.RecAtom('H', 1)
    
    return (H1.getEmissivity(tem=tem, den=den, lev_i=nu, lev_j=2)
            / H1.getEmissivity(tem=tem, den=den, lev_i=4, lev_j=2))


def create_balmer_table(tems, dens, table_path='balmer_table.fits'):
    """Create a table for fast balmer ratio computation.

    This table is used directly by :py:meth:`pnutils.get_Balmer_ratio`.
    """
    
    nb_tem = tems.shape[0]
    nb_den = dens.shape[0]
    table = np.empty((nb_tem, nb_den), dtype=float)
    
    for i in range(nb_tem):
        for j in range(nb_den):
            table[i,j] = get_Balmer_ratio(tems[i], dens[j],
                                          try_fast=False)
            
    hdr = list()
    hdr.append(('TEM_MIN', np.min(tems), 'Min Temperature [K]'))
    hdr.append(('DEN_MIN', np.min(dens), 'Min Density [cm-3]'))
    hdr.append(('TEM_MAX', np.max(tems), 'Max Temperature [K]'))
    hdr.append(('DEN_MAX', np.max(dens), 'Max Density [cm-3]'))
    hdr.append(('TEM_STEP', tems[1] - tems[0], 'Max Temperature [K]'))
    hdr.append(('DEN_STEP', dens[1] - dens[0], 'Max Density [cm-3]'))
    
    Tools().write_fits('balmer.fits', table, fits_header=hdr,
                       overwrite=True)
