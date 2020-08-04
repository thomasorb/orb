
#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: validate.py

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


import numpy as np
import warnings
import orb.utils.err
import logging

def is_ndarray(obj, raise_exception=True, object_name='object'):
    """Check if object is a numpy.ndarray
    :param obj: object to validate
    :param raise_exception: If True raise an exception else raise a warning.
    """
    err_msg = '{} is not a numpy.ndarray'.format(object_name)
    if not isinstance(obj, np.ndarray):
        if raise_exception:
            raise orb.utils.err.ValidationError(err_msg)
        else:
            logging.warn(err_msg)
            return False
    return True

def has_dtype(obj, dtype, raise_exception=True, object_name='object'):
    """Check if object is a numpy.ndarray of the correct type
    :param obj: object to validate
    :param dtype: array dtype
    :param raise_exception: If True raise an exception else raise a warning.
    """
    def err_msg(obj_type):
        return '{} has type {} but should have type {}'.format(
            object_name, obj_type, dtype)

    is_ndarray(obj)
    if obj.dtype != dtype:
        if raise_exception:
            raise orb.utils.err.ValidationError(err_msg(obj.dtype))
        else:
            logging.warn(err_msg(obj.dtype))
            return False
    return True

def is_xdarray(obj, ndim, raise_exception=True, object_name='object'):
    """Check if object is a numpy.ndarray with the correct number of
    dimensions

    :param obj: object to validate
    :param ndim: number of dimensions
    :param raise_exception: If True raise an exception else raise a warning.
    """
    ndim = int(ndim)
    if ndim < 0: raise orb.utils.err.ValidationError('ndim must be positive')
    
    err_msg = '{} is not a {}d array'.format(object_name, ndim)

    if not is_ndarray(obj, raise_exception=raise_exception, object_name=object_name):
        return False

    if obj.ndim == ndim:
        return True
    if raise_exception:
        raise orb.utils.err.ValidationError(err_msg)
    else:
        logging.warn(err_msg)
        return False
    
def is_1darray(obj, raise_exception=True, object_name='object'):
    """Check if object is a 1d numpy.ndarray
    :param obj: object to validate
    :param raise_exception: If True raise an exception else raise a warning.
    """
    return is_xdarray(obj, 1, raise_exception=raise_exception, object_name=object_name)

def is_2darray(obj, raise_exception=True, object_name='object'):
    """Check if object is a 2d numpy.ndarray
    :param obj: object to validate
    :param raise_exception: If True raise an exception else raise a warning.
    """
    return is_xdarray(obj, 2, raise_exception=raise_exception, object_name=object_name)

def is_3darray(obj, raise_exception=True, object_name='object'):
    """Check if object is a 3d numpy.ndarray
    :param obj: object to validate
    :param raise_exception: If True raise an exception else raise a warning.
    """
    return is_xdarray(obj, 3, raise_exception=raise_exception, object_name=object_name)

def have_same_shape(objs, raise_exception=True, object_name='arrays'):
    """Check if all numpy.ndarrays have the same shape
    :param obj: list of objects to validate
    :param raise_exception: If True raise an exception else raise a warning.
    """
    err_msg = 'all {} do not have the same shape'.format(object_name)
    if isinstance(objs, list) or isinstance(objs, tuple):
        shape = None
        for iobj in objs:
            if is_ndarray(iobj, raise_exception=raise_exception):
                if shape is None:
                    shape = iobj.shape
                else:
                    if iobj.shape != shape:
                        if raise_exception:
                            raise orb.utils.err.ValidationError(err_msg)
                        else:
                            logging.warn(err_msg)
                            return False
            else: break
    else:
        raise orb.utils.err.ValidationError('objs must be a list')

    return True


def index(a, a_min, a_max, clip=True):
    """Return a valid index (clipped between a_min and a_max - 1) or raise
    an exception.

    :param a: index. Can be a list or an array of indexes.

    :param a_min: Min index.

    :param a_max: Max index (max possible index will be considered as
      a_max -1)

    :param clip: (Optional) If True return an index inside the
      boundaries, else: raise an exception (default True).
    """
    a = np.squeeze(np.array(a).astype(int))
    if a.size == 1: a = int(a)
    if not isinstance(a_min, int) or not isinstance(a_max, int):
        raise orb.utils.err.ValidationError('min and max boundaries must be integers')
    if np.any(a < a_min) or np.any(a >= a_max):
        if not clip: raise orb.utils.err.ValidationError('index is off boundaries [{}, {}['.format(a_min, a_max))
        else: return np.clip(a, a_min, a_max)
    else: return a

def is_iterable(obj, raise_exception=True, object_name='object'):
    """check if object is a tuple or a list or a 1darray

    :param obj: Object to check
    :param raise_exception: If True raise an exception else raise a warning.
    """
    err_msg = '{} must be a tuple or a list'.format(object_name)
    if not isinstance(obj, list) and not isinstance(obj, tuple):
        if not is_1darray(obj, raise_exception=False, object_name=object_name):            
            if raise_exception: raise orb.utils.err.ValidationError(err_msg)
            else:
                logging.warn(err_msg)
                return False
    return True

def has_len(obj, length, raise_exception=True, object_name='object'):
    """Check if object is 1d and if its length is correct
    :param obj: Object to check
    :param raise_exception: If True raise an exception else raise a warning.
    :param length: length of the object
    """
    err_msg = '{} must have length: {}'.format(object_name, length)
    length = int(length)
    if length < 0: raise orb.utils.err.ValidationError('length must be a positive int')

    ok = False
    try:
        is_iterable(obj, raise_exception=True)
    except orb.utils.err.ValidationError:
        try:
            is_1darray(obj, raise_exception=True)
        except orb.utils.err.ValidationError:
            raise orb.utils.err.ValidationError(
                '{} is not a tuple, a list or a 1d array'.format(object_name))
    
    if np.array(obj).size != length:
        if raise_exception:
            raise orb.utils.err.ValidationError(err_msg)
        else:
            logging.warn(err_msg)
            return False
    return True
