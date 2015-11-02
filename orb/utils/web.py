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

import urllib2
from xml.dom import minidom
import orb.constants

def query_sesame(object_name, verbose=True):
    """Query the SESAME Database to get RA/DEC given the name of an
    object.
    
    :param object_name: Name of the object
    
    :returns: [RA, DEC]
    """
    if verbose:
        print "asking data from CDS server for : " + object_name
    words = object_name.split()
    object_name = words[0]
    if (len(words) > 1):
        for iword in range(1, len(words)):
            object_name += "+" + words[iword]
    
    url = orb.constants.SESAME_URL + '{}'.format(object_name)
    xml_result = urllib2.urlopen(url).read()
    dom = minidom.parseString(xml_result)
    node_list = dom.getElementsByTagName("jpos")
    if (node_list.length > 0):
        object_position = node_list[0].childNodes[0].data
    else:
        return []
    [ra, dec] = object_position.split()
    if verbose:
        print "{} RA: {}".format(object_name, ra)
        print "{} DEC: {}".format(object_name, dec)
    ra = ra.split(":")
    dec = dec.split(":")
    return ra, dec
