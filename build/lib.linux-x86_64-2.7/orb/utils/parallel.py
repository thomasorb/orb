#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: parallel.py

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

import os
import pp

def init_pp_server(ncpus=0, silent=False):
    """Initialize a server for parallel processing.

    :param ncpus: (Optional) Number of cpus to use. 0 means use all
      available cpus (default 0)
    
    :param silent: (Optional) If silent no message is printed
      (Default False).

    .. note:: Please refer to http://www.parallelpython.com/ for
      sources and information on Parallel Python software
    """
    ppservers = ()

    if ncpus == 0:
        job_server = pp.Server(ppservers=ppservers)
    else:
        job_server = pp.Server(ncpus, ppservers=ppservers)

    ncpus = job_server.get_ncpus()
    if not silent:
        print "Init of the parallel processing server with %d threads"%ncpus
    return job_server, ncpus

def close_pp_server(js):
    """
    Destroy the parallel python job server to avoid too much
    opened files.
    
    :param js: job server.

    .. note:: Please refer to http://www.parallelpython.com/ for
        sources and information on Parallel Python software.
    """
    # First shut down the normal way
    js.destroy()
    # access job server methods for shutting down cleanly
    js._Server__exiting = True
    js._Server__queue_lock.acquire()
    js._Server__queue = []
    js._Server__queue_lock.release()
    for worker in js._Server__workers:
        worker.t.exiting = True
        try:
            # add worker close()
            worker.t.close()
            os.kill(worker.pid, 0)
            os.waitpid(worker.pid, os.WNOHANG)
        except OSError:
            # PID does not exist
            pass
