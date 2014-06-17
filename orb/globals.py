import math, os
"""
Some global constants
"""


global FWHM_COEFF # Coefficient to convert the width of a gaussian function
            # to its FWHM (line_fwhm = line_width * FWHM)
FWHM_COEFF = abs(2.*math.sqrt(2. * math.log(2.)))


global LIGHT_VEL_KMS # Velocity of the light in the vacuum
                     # in km.s-1
LIGHT_VEL_KMS = 299792.458 
