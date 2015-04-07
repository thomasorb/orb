#!/usr/bin/python2.7
# *-* coding: utf-8 *-*
# author : Thomas Martin (thomas.martin.1@ulaval.ca)
# File: globals.py

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


import math, os
"""
This file contains only some useful constants
"""


# Coefficient to convert the width of a gaussian function to its FWHM
# (line_fwhm = line_width * FWHM)
FWHM_COEFF = abs(2.*math.sqrt(2. * math.log(2.)))

# Velocity of the light in the vacuum in km.s-1
LIGHT_VEL_KMS = 299792.458 

# Max length of a FITS string card
FITS_CARD_MAX_STR_LENGTH = 18

# Sesame URL
SESAME_URL = "http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/NSV?"
