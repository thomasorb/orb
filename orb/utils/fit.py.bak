#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fit.py

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
import math
import orb.constants
import gvar
import orb.utils.validate
import copy

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


def gvardict2pickdict(gvardict):
    """Convert a dictionary containing gvars into a nice pickable
    dictionary with couples of _mean / _sdev keys.

    Use the pickdict2gvardict to rerturn to the original dictionary.
    """
    if not isinstance(gvardict, dict):
        raise TypeError('gvardict must be a dict instance')
    idict = dict()    
    for ikey in gvardict:
        try:
            _mean = gvar.mean(gvardict[ikey])
            _sdev = gvar.sdev(gvardict[ikey])
            if not np.all(_sdev == 0.):
                idict[ikey + '_mean'] = _mean
                idict[ikey + '_sdev'] = _sdev
            else: raise TypeError
        except TypeError:
            idict[ikey] = gvardict[ikey]
        except AttributeError:
            idict[ikey] = gvardict[ikey]
    return idict

def pickdict2gvardict(pickdict):
    """Invert gvardict2pickdict."""
    if not isinstance(pickdict, dict):
        raise TypeError('pickdict must be a dict instance')
    pickdict = copy.copy(pickdict)
    idict = dict()
    _all_keys = pickdict.keys()
    for ikey in _all_keys:
        if ikey in pickdict:
            if '_mean' in ikey:
                prefix = ikey[:-len('_mean')]
                if prefix + '_sdev' in pickdict:
                    idict[prefix] = gvar.gvar(pickdict[ikey], pickdict[prefix + '_sdev'])
                    del pickdict[ikey]
                    del pickdict[prefix + '_sdev']
                else:
                    idict[ikey] = pickdict[ikey]
            else:
                idict[ikey] = pickdict[ikey]
    return idict

def paramslist2pick(paramslist):
    orb.utils.validate.is_iterable(paramslist, object_name='paramslist')
    pick = list()
    for idict in paramslist:
        pick.append(gvardict2pickdict(idict))
    return pick

def pick2paramslist(picklist):
    orb.utils.validate.is_iterable(picklist, object_name='picklist')
    paramslist = list()
    for idict in picklist:
        paramslist.append(pickdict2gvardict(idict))
    return paramslist

