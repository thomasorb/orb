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

import logging
import urllib.request, urllib.error, urllib.parse
from xml.dom import minidom
import io
import numpy as np
import socket
import orb.constants
import warnings
import astroquery.vizier
import astropy.coordinates
import astropy.units
from astroquery.gaia import Gaia

def query_sesame(object_name, verbose=True, degree=False, pm=False):
    """Query the SESAME Database to get RA/DEC given the name of an
    object.
    
    :param object_name: Name of the object

    :param verbose: (Optional) If True print messages (default True)

    :param degree: (Optional) If True return RA DEC in degrees
      (default False)

    :param pm: (Optional) If True proper motion is also returned
      (default False) (mu_ra*cos(dec), mu_dec)
    
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
    xml_result = urllib.request.urlopen(url).read()
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


class Catalog(object):
    def __init__(self, name, out, sort):
        self.name = name
        self.out = out
        self.sort = sort


def query_vizier(radius, target_ra, target_dec,
                 catalog='gaia', max_stars=100, return_all_columns=False,
                 as_pandas=False):
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

    :param as_pandas: (Optional) If True, results are returned as a
      pandas.DataFrame instance. Else a numpy.ndarray instance is
      returned (default False).
    """    
    catalogs = {
        'usno': Catalog('USNO-B1.0',
                        '_RAJ2000,_DEJ2000,e_RAJ2000,e_DEJ2000,R2mag',
                        '-R2mag'),
    
        'gaia1': Catalog('I/337/gaia',
                        'RA_ICRS,DE_ICRS,e_RA_ICRS,e_DE_ICRS,<Gmag>,pmRA,pmDE,e_pmRA,e_pmDE,Epoch',
                        '-<Gmag>'),
        
        'gaia2': Catalog('I/345/gaia2',
                         'RA_ICRS,DE_ICRS,e_RA_ICRS,e_DE_ICRS,Gmag,pmRA,pmDE,e_pmRA,e_pmDE,Epoch',
                         '-Gmag'),

        'gaia3': Catalog('I/350/gaiaedr3',
                         'RA_ICRS,DE_ICRS,e_RA_ICRS,e_DE_ICRS,Gmag,pmRA,pmDE,e_pmRA,e_pmDE,Epoch',
                         '-Gmag'),

        
        '2mass': Catalog('II/246/out',
                         'RAJ2000,DEJ2000,errMaj,errMin,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag',
                         '-Jmag'),
        
        'pan-starrs': Catalog('II/349/ps1',
                              'RAJ2000,DEJ2000,e_RAJ2000,e_DEJ2000,gmag,e_gmag,rmag,e_rmag,imag,e_imag,zmag,e_zmag,ymag,e_ymag',
                         '-rmag')}

    # shortcuts
    catalogs['gaia'] = catalogs['gaia3']

    if catalog not in catalogs:
        raise Exception("Bad catalog name. Can be {}".format(list(catalogs.keys())))
    else:
        cat = catalogs[catalog]

    logging.info("Sending query to VizieR server (catalog: {})".format(catalog))
    logging.info("Looking for stars at RA: %f DEC: %f"%(target_ra, target_dec))

    coords = astropy.coordinates.SkyCoord(ra=target_ra, dec=target_dec,
                                          unit=(astropy.units.deg, astropy.units.deg),
                                          frame='icrs')
    if not return_all_columns:
        vizier = astroquery.vizier.Vizier(columns=cat.out.split(','))
    else:
        vizier = astroquery.vizier.Vizier()
    vizier.ROW_LIMIT = -1
    result = vizier.query_region(coords, radius=radius/60.*astropy.units.deg, catalog=cat.name)
    
    if len(result) == 0: raise Exception('Query returned nothing at {} with a radius of {} arcmin'.format(coords, radius))
    else: result = result[0]

    sorting_key = cat.sort
    reverse_sort = True
    if sorting_key[0] == '-':
        sorting_key = sorting_key[1:]
        reverse_sort = False

    result = result.to_pandas()
    
    result = result.sort_values(by=[sorting_key])
    if reverse_sort:
        result = result[::-1]

    result = result.dropna()
    result = result[:max_stars]
    result = result.reset_index(drop=True)
    
    logging.info("%d stars recorded in the given field"%len(result))
    logging.info("Magnitude min: {}, max:{}".format(
        np.min(result[sorting_key].values), np.max(result[sorting_key].values)))

    if not as_pandas:
        return result.values
    else:
        result['ra'] = result[cat.out.split(',')[0]]
        result['dec'] = result[cat.out.split(',')[1]]
    
        return result

        
def query_gaia(radius, target_ra, target_dec,
               max_stars=100, as_pandas=False):
    """Return a list of star coordinates around an object in a
    given radius based on a query to the last gaia catalog

    :param radius: Radius around the target in arc-minutes.

    :param target_ra: Target RA in degrees

    :param target_dec: Target DEC in degrees

    :param max_stars: (Optional) Maximum number of rows to retrieve
      (default 100)

    :param return_all_columns: (Optional) If True, return all
      columns. Else only ra, dec and Mag are returned (default False).

    :param as_pandas: (Optional) If True, results are returned as a
      pandas.DataFrame instance. Else a numpy.ndarray instance is
      returned (default False).
    """    
    logging.info("Looking for stars at RA: %f DEC: %f"%(target_ra, target_dec))

    coords = astropy.coordinates.SkyCoord(ra=target_ra, dec=target_dec,
                                          unit=(astropy.units.deg, astropy.units.deg),
                                          frame='icrs')
    Gaia.ROW_LIMIT = -1
    
    radius = astropy.units.Quantity(radius/60, astropy.units.deg)

    result = Gaia.query_object_async(coordinate=coords, width=2*radius, height=2*radius,
                                     columns=['ra', 'dec', 'ra_error', 'dec_error',
                                              'phot_g_mean_mag', 'pmra', 'pmdec', 'pmra_error',
                                              'pmdec_error', 'ref_epoch'])
    
    if len(result) == 0: raise Exception('Query returned nothing at {} with a radius of {} arcmin'.format(coords, radius))
    
    result = result.to_pandas()
    result = result.drop('dist', axis=1)

    sorting_key = 'phot_g_mean_mag'
    result = result.sort_values(by=[sorting_key])

    result = result.dropna()
    result = result[:max_stars]
    result = result.reset_index(drop=True)
    
    logging.info("%d stars recorded in the given field"%len(result))
    logging.info("Magnitude min: {}, max:{}".format(
        np.min(result[sorting_key].values), np.max(result[sorting_key].values)))

    if not as_pandas:
        return result.values

    # for backward compatibility
    result['Epoch'] = result['ref_epoch']
    result['pmRA'] = result['pmra']
    result['pmDE'] = result['pmdec']
    result['e_pmRA'] = result['pmra_error']
    result['e_pmDE'] = result['pmdec_error']
    result['e_RA_ICRS'] = result['ra_error']
    result['e_DE_ICRS'] = result['dec_error']
    result['Gmag'] = result['phot_g_mean_mag']
    
    
    return result

