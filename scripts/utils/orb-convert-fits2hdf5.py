#!/usr/bin/env python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca> 
# File: orb-convert

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


from orb.core import OutHDFQuadCube, Tools, ProgressBar
import orb.utils.io
import sys
import os
import math


#### Warning: This script is a utilitary script. Do not use it if you
#### don't know what you are doing

to = Tools(config_file_name='config.sitelle.orb')
QUAD_NB = 9

inpath = str(sys.argv[1])
calibpath = str(sys.argv[2])
if not os.path.exists(inpath):
    raise Exception('Input file {} does not exist'.format(inpath))
if not os.path.exists(calibpath):
    raise Exception('Calibration map file {} does not exist'.format(calibpath))

outpath = os.path.splitext(os.path.split(inpath)[1])[0] + '.hdf5'
data, hdr = orb.utils.io.read_fits(inpath, return_header=True)
outcube = OutHDFQuadCube(outpath, data.shape, QUAD_NB, reset=True,
                         config_file_name='config.sitelle.orb')
outcube.append_calibration_laser_map(orb.utils.io.read_fits(calibpath))
progress = ProgressBar(QUAD_NB)

for iquad in range(QUAD_NB):
    progress.update(iquad, 'writing quadrant {}'.format(iquad))
    
    x_min, x_max, y_min, y_max = outcube._get_quadrant_dims(
        iquad, outcube.shape[0], outcube.shape[1],
        int(math.sqrt(float(QUAD_NB))))
    outcube.write_quad(iquad, data= data[x_min:x_max, y_min:y_max, :])
    
progress.end()
outcube.append_header(hdr)
outcube.close()
                    
