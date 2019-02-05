#!/usr/bin/python2.7
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: constants.py

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


import math, os

##
## This file contains some useful constants
##


FWHM_COEFF = abs(2.*math.sqrt(2. * math.log(2.)))
"""Coefficient used to convert the width of a gaussian function to its FWHM (line_fwhm = line_width * FWHM)"""

FWHM_SINC_COEFF = 1.20671
"""Coefficien used to determine sinc fwhm"""

LIGHT_VEL_KMS = 299792.458
"""Velocity of the light in the vacuum in km.s-1"""

LIGHT_VEL_AAS = 2.99792458e18
"""Velocity of the light in the vacuum in A.s-1"""


FITS_CARD_MAX_STR_LENGTH = 18
"""Max length of a FITS string card"""

#SESAME_URL = "http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/SNVA?"
SESAME_URL = "http://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oxp/SNVA?"
"""Sesame URL """

VIZIER_URL_CA = "http://vizier.hia.nrc.ca/viz-bin/"
"""Vizier URL in Canada """

VIZIER_URL = "http://webviz.u-strasbg.fr/viz-bin/"
"""Vizier URL in Canada """

PLANCK = 6.6260755e-27
"""Planck constant in erg.s """

K_BOLTZMANN = 1.38064852e-16
"""Boltzmann constant in erg/K"""

ATOMIC_MASS = 1.66053886e-24
"""1 Atomic mass in g"""
