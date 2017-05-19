#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fit.py

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
import orb.data as od
import orb.constants
import gvar

def vel2sigma(vel, lines, axis_step):
    """Convert a velocity in km/s to a broadening in channels.
    :param vel: velocity in km/s
    :param lines: line position in the unit of axis_step
    :param axis_step: axis step size
    """
    sigma = lines * vel / orb.constants.LIGHT_VEL_KMS
    sigma /= axis_step # convert sigma cm-1->pix
    return gvar.fabs(sigma)


def sigma2vel(sigma, lines, axis_step):
    """Convert a broadening in channels to a velocity in km/s
    :param sigma: broadening in channels
    :param lines: line position in the unit of axis_step
    :param axis_step: axis step size
    """
    vel = sigma * axis_step # convert sigma pix->cm-1
    # convert sigma cm-1-> km/s
    vel = orb.constants.LIGHT_VEL_KMS * vel / lines
    return gvar.fabs(vel)
