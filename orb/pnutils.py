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

    
def get_cHbeta(ha, hb, tem=1e4, den=1e3, ext_law='CCM 89'):
    rc = pn.RedCorr(law=ext_law)    
    th_ratio = get_Balmer_ratio(tem, den, nu=3)
    print 'Theroretical ratio: ', th_ratio
    obs_ratio = ha / hb
    rc.setCorr(obs_ratio/th_ratio, 6563., 4861.)
    return rc.cHbeta


def get_Balmer_ratio(tem, den, nu=3):
    H1 = pn.RecAtom('H', 1)
    return (H1.getEmissivity(tem=tem, den=den, lev_i=nu, lev_j=2)
            / H1.getEmissivity(tem=tem, den=den, lev_i=4, lev_j=2))
