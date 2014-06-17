#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: pnutils.py

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
