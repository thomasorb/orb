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
import urllib2
from xml.dom import minidom
import StringIO
import numpy as np
import socket
import orb.constants
import warnings

def query_sesame(object_name, verbose=True, degree=False, pm=False):
    """Query the SESAME Database to get RA/DEC given the name of an
    object.
    
    :param object_name: Name of the object

    :param verbose: (Optional) If True print messages (default True)

    :param degree: (Optional) If True return RA DEC in degrees
      (default False)

    :param pm: (Optional) If True proper motion is also returned
      (default False)
    
    :returns: [RA, DEC]
    """
    if verbose:
        logging.info("asking data from CDS server for : " + object_name)

    keys = list()
    for ikey in object_name.split():
        ikey = '%2B'.join(ikey.split('+'))
        keys.append(ikey)
    object_name = '+'.join(keys)
    url = orb.constants.SESAME_URL + '{}'.format(object_name)
    xml_result = urllib2.urlopen(url).read()
    dom = minidom.parseString(xml_result)
    # get position
    if not degree:
        node_list = dom.getElementsByTagName("jpos")
        if (node_list.length > 0):
            object_position = node_list[0].childNodes[0].data
        else:
            return []
        [ra, dec] = object_position.split()
    else:
        node_list = dom.getElementsByTagName("jradeg")
        if (node_list.length > 0):
            ra = float(node_list[0].childNodes[0].data)
        else:
            return []
        node_list = dom.getElementsByTagName("jdedeg")
        if (node_list.length > 0):
            dec = float(node_list[0].childNodes[0].data)
        else:
            return []

    # get proper motion
    node_list = dom.getElementsByTagName("pm")
    if (node_list.length > 0):
        pm_ra = float(
            node_list[0].getElementsByTagName('pmRA')[0].childNodes[0].data)
        pm_dec = float(
            node_list[0].getElementsByTagName('pmDE')[0].childNodes[0].data)
    else:
        pm_ra = None
        pm_dec = None
        
    
    if verbose:
        logging.info("RA: {}".format(ra))
        logging.info("DEC: {}".format(dec))
        if pm:
            logging.info("PM RA: {}".format(pm_ra))
            logging.info("PM DEC: {}".format(pm_dec))
        

    if not degree:
        ra = ra.split(":")
        dec = dec.split(":")

    if not pm:
        return ra, dec
    else:
        return ra, dec, pm_ra, pm_dec


def query_vizier(radius, target_ra, target_dec,
                 catalog='gaia', max_stars=100, return_all_columns=False):
    """Return a list of star coordinates around an object in a
    given radius based on a query to VizieR Services
    (http://vizier.u-strasbg.fr/viz-bin/VizieR)

    Note that the idea of this method has been picked from an IDL
    function: QUERYVIZIER
    (http://idlastro.gsfc.nasa.gov/ftp/pro/sockets/queryvizier.pro)

    :param radius: Radius around the target in arc-minutes.

    :param target_ra: Target RA in degrees

    :param target_dec: Target DEC in degrees

    :param max_stars: (Optional) Maximum number of rows to retrieve
      (default 100)

    :param catalog: (Optional) can be 'usno' - Version B1 of the US
      Naval Observatory catalog (2003), 'gaia' - GAIA DR1, or '2mass'
      - 2MASS (default Gaia)

    :param return_all_columns: (Optional) If True, return all
      columns. Else only ra, dec and Mag are returned (default False).
    """
    MAX_RETRY = 5
    if catalog == 'usno':
        catalog_id = 'USNO-B1.0'
        out = '_RAJ2000,_DEJ2000,e_RAJ2000,e_DEJ2000,R2mag'
        sort = '-R2mag'
    elif catalog == 'gaia':
        catalog_id = 'I/337/gaia'
        out = 'RA_ICRS,DE_ICRS,e_RA_ICRS,e_DE_ICRS,<Gmag>'
        sort='-<Gmag>'
    elif catalog == '2mass':
        catalog_id = 'II/246/out'
        out = 'RAJ2000,DEJ2000,errMaj,errMin,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag'
        sort='-Jmag'

    else: raise Exception("Bad catalog name. Can be 'usno', 'gaia' or '2mass'")

    params_number = len(out.split(','))

    logging.info("Sending query to VizieR server (catalog: {})".format(catalog))
    logging.info("Looking for stars at RA: %f DEC: %f"%(target_ra, target_dec))

    URL = (orb.constants.VIZIER_URL + "asu-tsv/?-source=" + catalog
           + "&-c.ra=%f"%target_ra + '&-c.dec=%f'%target_dec
           + "&-c.rm=%d"%int(radius)
           + '&-out.max=unlimited&-out.meta=-huD'
           + '&-out={}&-sort={}'.format(out, sort))

    logging.info('Vizier URL: {}'.format(URL))

    retry = 0
    while retry <= MAX_RETRY:
        try:
            query_result = urllib2.urlopen(URL, timeout=5)
            break

        except urllib2.URLError:
            retry += 1
            warnings.warn(
                'Vizier timeout, retrying ... {}/{}'.format(
                    retry, MAX_RETRY))
        except socket.timeout:
            retry += 1
            warnings.warn(
                'Vizier timeout, retrying ... {}/{}'.format(
                    retry, MAX_RETRY))

    if retry > MAX_RETRY:
        raise Exception(
            'Vizier server unreachable, try again later')

    query_result = query_result.read()
    output = StringIO.StringIO(query_result)
    star_list = list()
    for iline in output:
        if iline[0] != '#' and iline[0] != '-' and len(iline) > 3:
            iline = iline.split()
            if len(iline) == params_number:
                if ((float(iline[2])/1000.) < 0.5
                    and (float(iline[3])/1000.) < 0.5):
                    if not return_all_columns:
                        star_list.append((float(iline[0]),
                                          float(iline[1]),
                                          float(iline[4])))
                    else:
                        star_list.append(
                            list((float(iline[0]),
                                  float(iline[1]),
                                  float(iline[4])))
                            + list(np.array(iline[5:], dtype=float)))
                        

    # sorting list to get the brightest stars first
    star_list = np.array(sorted(star_list, key=lambda istar: istar[2]))
    
    logging.info("%d stars recorded in the given field"%len(star_list))
    logging.info("Magnitude min: {}, max:{}".format(
        np.min(star_list[:,2]), np.max(star_list[:,2])))
    return star_list[:max_stars,:]
