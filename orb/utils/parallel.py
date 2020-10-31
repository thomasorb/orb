#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: parallel.py

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
import os
import getpass
import multiprocessing
import dill
import warnings
import traceback

# see https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
def run_dill_encoded(payload):
    #with warnings.catch_warnings():
    #    warnings.simplefilter("ignore")
    fun, args = dill.loads(payload)
    try:
        return fun(*args)
    except:
        print('%s: %s' % (fun, traceback.format_exc()))

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

class JobServer(object):

    def __init__(self, ncpus, timeout=1000):

        self.ncpus = int(ncpus)
        self.timeout = int(timeout)
        
        if self.ncpus == 0:
            self.ncpus = multiprocessing.cpu_count()

        self.pool = multiprocessing.get_context('spawn').Pool(processes=self.ncpus, maxtasksperchild=1)

    def submit(self, func, args=(), modules=()):
        
        if not isinstance(args, tuple):
            raise TypeError('args must be a tuple')

        if not isinstance(modules, tuple):
            raise TypeError('modules must be a tuple')

        job = apply_async(self.pool, func, args)
        
        return Job(job, self.timeout)

    def __del__(self):
        try:
            self.pool.close()
        except:
            logging.debug('exception occured during pool.close: ', traceback.format_exc()) 
        try:
            self.pool.join()
        except RuntimeError: pass
        except:
            logging.info('exception occured during pool.join: ', traceback.format_exc())
        logging.info('parallel processing closed')
        try:
            del self.pool
        except: pass
            
class Job(object):
    
    def __init__(self, job, timeout):
        self.job = job
        self.timeout = int(timeout)

    def __call__(self):
        try:
            return self.job.get(timeout=self.timeout)
        except multiprocessing.TimeoutError:
            logging.info('worker timeout: ', traceback.format_exc())
        except:
            logging.info('exception occured during worker execution: ', traceback.format_exc())
    

class RayJob(object):

    def __init__(self, job):
        self.job = job
    
    def __call__(self):
        return ray.get(self.job)

class RayJobServer(object):

    def submit(self, func, args=(), modules=()):

        def wrapped_f(func, args, modules):
            for imod in modules:
                exec((imod), locals())
            return func(*args)

        if not isinstance(args, tuple):
            raise TypeError('args must be a tuple')

        if not isinstance(modules, tuple):
            raise TypeError('modules must be a tuple')

        parsed_modules = list()
        for imod in modules:
            if not isinstance(imod, str):
                raise TypeError('each module must be a string')
            if not 'import' in imod:
                imod = 'import ' + imod
            parsed_modules.append(imod)
            
        return RayJob(ray.remote(wrapped_f).remote(func, args, parsed_modules))


def get_ncpus(ncpus):
    """Return ncpus considering the target ncpus and the hard limits setting"""
    
    WL_PATH = '/etc/orb-kernels-wl' # user white list
    NCPUS_PATH = '/etc/orb-kernels-ncpus' # ncpus limit

    # check if a hard configuration exists
    max_cpus = None
    if os.path.exists(NCPUS_PATH):
        with open(NCPUS_PATH, 'r') as f:
            for line in f:
                try:
                    max_cpus = int(line)
                except: pass

    in_wl = False
    if os.path.exists(WL_PATH):
        with open(WL_PATH, 'r') as f:
            for line in f:
                if getpass.getuser() in line:
                    in_wl = True
                    break

    if ncpus == 0 and not in_wl and max_cpus is not None:
        ncpus = max_cpus
        logging.debug('max cpus limited to {} because of machine hard limit configuration'.format(max_cpus))
    return ncpus

    
    
def init_pp_server(ncpus=0, silent=False, use_ray=False, timeout=1000):
    """Initialize a server for parallel processing.

    :param ncpus: (Optional) Number of cpus to use. 0 means use all
      available cpus (default 0)
    
    :param silent: (Optional) If silent no message is printed
      (Default False).

    .. note:: Please refer to http://www.parallelpython.com/ for
      sources and information on Parallel Python software
    """
    ncpus = get_ncpus(ncpus)

    if not use_ray:
        job_server = JobServer(ncpus, timeout=timeout)
        ncpus = job_server.ncpus
                
    else:
        ray.shutdown()
        ray.init(num_cpus=int(ncpus), configure_logging=False, object_store_memory=int(3e9))
        job_server = RayJobServer()
        ncpus = int(ray.available_resources()['CPU'])
    
    if not silent:
        logging.info("Init of the parallel processing server with %d threads"%ncpus)
    else:
        logging.debug("Init of the parallel processing server with %d threads"%ncpus)

    return job_server, ncpus

    

def close_pp_server(js, silent=False):
    """
    Destroy the parallel python job server to avoid too much
    opened files.
    
    :param js: job server.

    .. note:: Please refer to http://www.parallelpython.com/ for
        sources and information on Parallel Python software.
    """
    if not silent:
        logging.info("Closing parallel processing server")
    else:
        logging.debug("Closing parallel processing server")
    if isinstance(js, RayJobServer):
        ray.shutdown()
    else:
        del js

def get_stats_str(js):
    """Return job server statistics as a string"""
    _stats = js.get_stats()['local']
    return 'ncpus: {},  njobs: {}, rworker: {}, time: {}'.format(
        _stats.ncpus, _stats.njobs, _stats.rworker, _stats.time)
    
        
def timed_process(func, timeout, args=list()):
    """Run a timed process which terminates after timeout seconds if it
    does not return before.

    :param func: Timed func which will be terminated after timeout
      seconds. must be func(*args, returned_dict). The
      results of the function must be put in returned_dict.

    :param timeout: Timeout in s.

    :param args: arguments of the function

    :return: returned_dict

    .. note:: from https://stackoverflow.com/questions/492519/timeout-on-a-function-call

    """
    if not isinstance(args, list): raise TypeError('args must be a list')

    returned_dict = multiprocessing.Manager().dict()
    args.append(returned_dict)
    
    p = multiprocessing.Process(
        target=func,
        args=args)
    
    p.start()
    p.join(timeout)
    if p.is_alive():
        logging.debug("process reached timeout")

        # Terminate
        p.terminate()
        p.join()
        
    return returned_dict
                
                
