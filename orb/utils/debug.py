#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: debug.py

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

import os
import subprocess

def get_open_fds():
    """Return the number of open file descriptors

    .. warning:: Only works on UNIX-like OS

    .. note:: This is a useful debugging function that has been taken from: http://stackoverflow.com/questions/2023608/check-what-files-are-open-in-python
    """
    import resource
    pid = os.getpid()
    procs = subprocess.check_output(
        ["lsof", '-w', '-Ff', "-p", str( pid )])
    return len(filter( 
        lambda s: s and s[0] == 'f' and s[1:].isdigit(),
        procs.split( '\n' ))), resource.getrlimit(resource.RLIMIT_NOFILE)
