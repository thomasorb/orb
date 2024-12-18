#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fit.py

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

"""Fit module of ORB.

Defines the general Fitting classes and the fitting models. 

Best accessed through fit_lines_in_*() functions (defined at the end
of the file)
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"       


import numpy as np
import warnings
import copy
import scipy.optimize
import scipy.interpolate
import time
import logging

import gvar

import orb.utils.fft
import orb.constants
import orb.utils.spectrum
import orb.utils.fit
import orb.utils.stats
import orb.utils.validate
import orb.utils.err
import orb.cutils

from orb.core import Lines, FilterFile, Axis
import orb.core

class FitVector(object):
    """
    General fitting class for a 1d array of data based on
    Levenberg-Marquardt least square fit algorithm.

    Accept any combination of models (based on :py:class:`fit.Model`)

    .. note:: This class is a wrapper around
      :py:meth:`scipy.optimize.leastsq`. Most of its purpose consists
      in passing an array of purely free parameters to
      :py:meth:`scipy.optimize.leastsq` and creating the objective
      function from free and fixed parameters by combining the
      different models. 
    """

    models = None
    models_operation = None
    models_operations = ['add', 'mult']
    vector = None
    priors_list = None
    priors_keys_list = None

    max_fev = 5000
    fit_tol = None
    
    def __init__(self, vector, models, params, 
                 fit_tol=1e-8, signal_range=None, 
                 max_iter=None, docomplex=False, nogvar=False,
                 vector_err=None, force_positive_flux=True):
        """Init class.

        :param vector: Vector to fit
        
        :param models: list of models to combine. Must be a list of
          couples (models, model_operation). A model must be a class
          which derives from :py:class:`orb.fit.Model`. Model
          operation mist be 'add' if the model has to be added to the
          others or 'mult' if it has to be multiplied with the
          others. Models are combined in the order of the
          list. e.g. [(Model1, 'add'), (Model2, 'mult'), ...].

        :param params: list of parameters dictionaries for each
          model. Needed parameters are defined in each model.

        :param fit_tol: (Optional) Fit tolerance (default 1e-10)

        :param signal_range: (Optional) couple (min, max) defining the
          range of values considered in the fitting process.

        :param docomplex: (Optional) If data is complex and docomplex
          is True, tries a complex fit. If False, always fit the real
          part only.

        :param nogvar: (Optional) No gvar are returned.

        :param force_positive_flux: (Optional) Force fluxes to be
          positive (default True)
        """
        self.nogvar = bool(nogvar)
        
        if not isinstance(models, tuple) and not isinstance(models, list) :
            raise ValueError('models must be a tuple of (model, model_operation).')

        if not isinstance(params, tuple) and not isinstance(params, list):
            raise ValueError('params must be a tuple of parameter dictionaries.')

        if len(models) != len(params):
            raise Exception('there must be exactly one parameter dictionary by model')

        if max_iter is not None:
            max_iter = int(np.clip(max_iter, 0, 1e6))
            logging.debug('max iteration changed to {}'.format(max_iter))
        else:
            max_iter = 10000
        self.max_iter = max_iter

        if np.any(np.iscomplex(vector)):
            if docomplex:
                self.vector_imag = np.copy(vector.imag)
            else:
                self.vector_imag = None
        else:
            self.vector_imag = None
        vector = vector.real

        self.vector = np.copy(vector)
        if vector_err is not None:
            assert vector_err.size == vector.size, 'error vector must have same size as vector'
            self.sigma = np.copy(vector_err)
        else:
            self.sigma = np.ones_like(self.vector)
        if np.all(self.sigma == 0.): self.sigma.fill(1.)
        self.fit_tol = fit_tol
        self.normalization_coeff = np.nanmax(self.vector) - np.nanmedian(self.vector)
        self.vector /= self.normalization_coeff
        if self.vector_imag is not None:
            self.vector_imag /= self.normalization_coeff
        self.sigma /= self.normalization_coeff

        self.retry_count = 0
        self.models = list()
        self.models_operation = list()
        self.priors_list = list()
        self.priors_keys_list = list()
        self.bounds_list = list()
        params = orb.utils.fit.pick2paramslist(params)
        
        for i in range(len(models)):
            # init each model
            self.models.append(models[i][0](params[i]))
            if models[i][1] in self.models_operations:
                self.models_operation.append(models[i][1])
            else: raise Exception('Model operation must be in {}'.format(
                self.models_operations))
            # guess nan values for each model
            self.models[-1].make_guess(self.vector)
            
            self.priors_list.append(self.models[-1].get_priors())
            if isinstance(self.models[-1], Cm1LinesModel):
                self.bounds_list.append(self.models[-1].get_p_bounds(
                    force_positive_flux=force_positive_flux))
            else:
                self.bounds_list.append(self.models[-1].get_p_bounds())
            self.priors_keys_list.append(list(self.priors_list[-1].keys()))
        self.all_keys_index = None

        if signal_range is not None:
            if (np.nanmin(signal_range) >= 0 and
                np.nanmax(signal_range) < self.vector.shape[0]):
                self.signal_range = [int(np.min(signal_range)),
                                     int(np.max(signal_range))]
            
            else: raise Exception('Bad signal range: {}'.format(signal_range))
        else: self.signal_range = None    

    def _all_p_list2dict(self, p_list):
        """Concatenate a list of free parameters into a dictionary
        ready to be passed to the fit function.

        :param p_list: List of free parameters. This list is a list of
          tuple of parameters. Each tuple defining the free parameters
          for each model, i.e. : ([free_params_model1],
          [free_params_model2], ...)

        .. seealso:: :py:meth:`fit.FitVector._all_p_dict2list`
        """
        all_p = dict()
        for p in p_list:
            # check if two parameters have the same key
            for _k in p:
                if _k in list(all_p.keys()):
                    raise Exception('Two parameters are sharing the same key: {}'.format(_k))
            all_p.update(p)
        return all_p

    def _all_p_dict2list(self, p_vect):
        """Transform a dictionnary of free parameters as returned by
        the fit function into a list of free parameters (see
        :py:meth:`fit.FitVector._all_p_list2dict`)

        :param p_vect: dictionnary of free parameters.

        .. seealso:: :py:meth:`fit.FitVector._all_p_list2dict`
        """
        if isinstance(p_vect, tuple):
            p_vect = p_vect[0]

        p_list = list()
        last_index = 0
        for keys_list in self.priors_keys_list:
            ip_list = dict()
            for key in keys_list:
                # remove log(prior), sqrt(prior) from the list
                if 'log' in key:
                    ip_list[key[4:-1]] = p_vect[key[4:-1]]
                elif 'sqrt' in key:
                    ip_list[key[5:-1]] = p_vect[key[5:-1]]
                elif 'erfinv' in key:
                    ip_list[key[7:-1]] = p_vect[key[7:-1]]
                else:
                    ip_list[key] = p_vect[key]
            p_list.append(ip_list)
        return p_list

    def _all_p_dict2arr(self, p_dict):
        """Return a 1d array of free parameters from a dictionary of
        free parameters.

        :param p_dict: Free parameters dict
        """
        # check keys and create all_keys_index
        all_keys_index = dict()
        index = 0
        for keys_list in self.priors_keys_list:
            for key in keys_list:
                all_keys_index[key] = index
                index += 1
        if all_keys_index.keys() != p_dict.keys(): raise Exception(
            'Badly formatted input dict')
        
        self.all_keys_index = all_keys_index # save keys--index dict

        p_arr = np.empty(len(self.all_keys_index), dtype=float)
        for key in self.all_keys_index:
            p_arr[self.all_keys_index[key]] = p_dict[key]
            
        return p_arr

    def _all_p_arr2dict(self, p_arr, keep_gvar=False):
        """Return a dict of free parameters from a 1d array of
        free parameters.

        :param p_arr: Free parameters array

        :param keep_gvar: (Optional) If True, gvar values of the
          parameters are kept. Else they are converted to float
          (default False).
        """
        if self.all_keys_index is None:
            raise Exception('self._all_p_dict2arr() must be called first')
        if np.size(p_arr) != len(self.all_keys_index):
            raise Exception('Badly formatted input array of free parameters')

        p_dict = dict()
        for key in self.all_keys_index:
            ival = p_arr[self.all_keys_index[key]]
            if not keep_gvar: ival = float(gvar.mean(ival))
            p_dict[key] = ival

        return p_dict
        
    def get_model(self, all_p_free, return_models=False, x=None):
        """Return the combined model of the vector given a set of free
        parameters.

        This function is typically called to compute the objective
        function. It can also be called to obtain the final model
        based on fitted parameters.

        :param all_p_free: Vector of free parameters.

        :param return_models: (Optional) If True return also
          individual models (default False)

        :param x: (Optional) array of data points on which model is
          computed instead of a typical np.arange(step_nb) (default
          None).
        """
        if isinstance(all_p_free, tuple):
            all_p_free = all_p_free[0]

        # check nans
        for ikey in all_p_free:
            if np.isnan(gvar.sdev(all_p_free[ikey])):
                logging.debug('nan in passed parameters: {}'.format(all_p_free))
    
        step_nb = self.vector.shape[0]
        if x is None:
            x = np.arange(step_nb, dtype=float)
       
        if return_models:
            models = dict()
        else:
            models = list()
        all_p_list = self._all_p_dict2list(all_p_free)

        if self.vector_imag is not None: return_complex = True
        else: return_complex = False
        
        # all multiplicative models must be multiplied together before
        # beging applied to the the additive models using the
        # dedicated class option (because multiplication by a filter
        # function cannot be applied straighforward to a line model)
        mult_model = np.ones_like(self.vector, dtype=float)
        for i in range(len(self.models)):
            if self.models_operation[i] == 'mult':
                model_list = self.models[i].get_model(
                    x, all_p_list[i], return_models=return_models, return_complex=return_complex)
                
                if return_models:
                    model_to_append, models_to_append = model_list
                    models[self.models[i].__class__.__name__] = models_to_append
                else:
                    model_to_append = model_list
                mult_model *= model_to_append
            if self.models_operation[i] not in ['mult', 'add']:
                raise Exception('Bad model operation. Model operation must be in {}'.format(self.models_operations))

        model = None
        if np.all(mult_model == 1): mult_model = None
        
        for i in range(len(self.models)):
            if self.models_operation[i] == 'add':
                model_list = self.models[i].get_model(
                    x, all_p_list[i], return_models=return_models, multf=mult_model, return_complex=return_complex)
                if return_models:
                    model_to_append, models_to_append = model_list
                    models[self.models[i].__class__.__name__] = models_to_append
                else:
                    model_to_append = model_list
                        
                if model is None:
                    model = model_to_append
                else:
                    model += model_to_append
                
    
        if np.any(np.isnan(gvar.mean(model))):
            logging.debug('Nan in model')

        if return_models:
            return model, models
        else:
            return model
    
    def _get_model_onrange(self, x, *all_p_free):
        """Return the part of the model contained in the signal
        range.

        .. note:: This function has been defined only to be used with
          scipy.optimize.curve_fit.

        :param x: x vector on which model is computed

        :param *all_p_free: Vector of free parameters.
        """
        all_p_free = self._all_p_arr2dict(all_p_free)
        
        out = self.get_model(all_p_free, x=x)
        
        if self.vector_imag is None:
            return out[np.min(self.signal_range):np.max(self.signal_range)]
        else:
            out = orb.utils.vector.float2complex(out)
            return orb.utils.vector.complex2float((
                out[0][np.min(self.signal_range):np.max(self.signal_range)],
                out[1][np.min(self.signal_range):np.max(self.signal_range)]))
            

    def _get_vector_onrange(self):
        """Return the part of the vector contained in the signal
        range.

        .. note:: This function has been defined only to be used with
          scipy.optimize.curve_fit.
        """
        out = self.vector[
            np.min(self.signal_range):np.max(self.signal_range)]
        if self.vector_imag is None:
            return out
        else:
            out_imag = self.vector_imag[
                np.min(self.signal_range):np.max(self.signal_range)]
            return orb.utils.vector.complex2float((out, out_imag))
        

    def _get_sigma_onrange(self):
        """Return the part of the uncertainty on the vector contained
        in the signal range.

        .. note:: This function has been defined only to be used with
          scipy.optimize.curve_fit.
        """
        return self.sigma[
            np.min(self.signal_range):np.max(self.signal_range)]



    def fit(self, compute_mcmc_error=False):
        """Fit data vector.

        This is the central function of the class.

        :param compute_mcmc_error: (Optional) Compute Markov chain
          Monte-Carlo error on the fit parameters (Uncertainty
          estimates might be slighly better constrained but computing
          time can be orders of magnitude longer) (default False).
        """
        all_args = dict(locals()) # used in case fit is retried (must stay
                                  # at the very beginning of the
                                  # function ;)

        MCMC_RANDOM_COEFF = 1e-2
        
        start_time = time.time()
        priors_dict = self._all_p_list2dict(self.priors_list)
        bounds_dict = self._all_p_list2dict(self.bounds_list)
                        
        priors_arr = self._all_p_dict2arr(priors_dict)

        upper_bounds_arr = np.empty(len(self.all_keys_index), dtype=float)
        lower_bounds_arr = np.empty(len(self.all_keys_index), dtype=float)
        for key in self.all_keys_index:
            lower_bounds_arr[self.all_keys_index[key]] = bounds_dict[key][0]
            upper_bounds_arr[self.all_keys_index[key]] = bounds_dict[key][1]
        bounds = (lower_bounds_arr, upper_bounds_arr)
        try:
            fit_results = scipy.optimize.curve_fit(
                self._get_model_onrange,
                np.arange(self.vector.shape[0]),
                self._get_vector_onrange(),
                #sigma=self._get_sigma_onrange(),
                bounds=bounds,
                p0=priors_arr,
                method='trf',
                full_output=True,
                maxfev=self.max_iter)
        except RuntimeError as e:
            logging.debug('RuntimeError during fit: {}'.format(e))
            fit_results = list(['Runtime error during fit: {}'.format(e), 0])

        fit = type('fit', (), {})

        if 0 < fit_results[-1] < 5:
            fit.stopping_criterion = fit_results[-1]
            fit.error = None

            # compute uncertainties
            cov_x = fit_results[1]
            if np.all(np.isfinite(cov_x)):
                p_err = np.sqrt(np.diag(cov_x))
            else:
                p_err = np.empty_like(fit_results[0])
                p_err.fill(np.nan)

            fit.p = self._all_p_arr2dict(gvar.gvar(fit_results[0], p_err),
                                         keep_gvar=True)


            last_diff = fit_results[2]['fvec']

            fitted_vector = self._get_model_onrange(
                np.arange(self.vector.shape[0], dtype=float),
                *fit_results[0])
            vector = self._get_vector_onrange()

            residual = (vector - fitted_vector)

            if self.vector_imag is None:
                res_ratio = residual/self._get_sigma_onrange()
            else:
                res_ratio = residual
            res_ratio[np.isinf(res_ratio)] = np.nan
            fit.chi2 = np.nansum(res_ratio**2)
            fit.dof = self._get_vector_onrange().shape[0] - np.size(fit_results[0])
            fit.logGBF = np.nan
            fit.fitter_results = fit_results[2]

        else:
            logging.debug('bad classic fit ({}): {}'.format(fit_results[-1], fit_results[-2]))
            fit.stopping_criterion = 0
            fit.error = True

            
        ### fit results formatting ###
        if fit.error is None:
            fit_p = fit.p
            
            # correct for normalization coeff
            for key in list(fit_p.keys()):
                if 'amp' in key or 'cont_p' in key:
                    fit_p[key] = fit_p[key] * self.normalization_coeff
                    
            if fit.stopping_criterion == 0:
                logging.debug('Dit not converge: stopping criterion == 0')
                logging.warning('Did not converge')
                return []

            returned_data = dict()
            returned_data['iter_nb'] = fit.fitter_results['nfev']

            ## get fit model
            _model, _models = self.get_model(
                fit_p,
                return_models=True)
            if self.vector_imag is not None:
                _model = orb.utils.vector.float2complex(_model)
                for ikey in _models:
                    if isinstance(_models[ikey], list):
                        _new_model = list()
                        for imod in _models[ikey]:
                            _new_model.append(orb.utils.vector.float2complex(imod))
                        _models[ikey] = _new_model
                    else:
                        _models[ikey] = orb.utils.vector.float2complex(_models[ikey])
                            
            (returned_data['fitted_vector_gvar'],
             returned_data['fitted_models_gvar']) = _model, _models
            
            returned_data['fitted_vector'] = gvar.mean(returned_data['fitted_vector_gvar'])
            returned_data['fitted_models'] = dict()
            for imod in returned_data['fitted_models_gvar']:
                returned_data['fitted_models'][imod] = gvar.mean(
                    returned_data['fitted_models_gvar'][imod])

            ## return fitted parameters of each models
            full_p_list = list()
            full_p_list_err = list()
            full_p_list_gvar = list()
            
            
            p_fit_list = self._all_p_dict2list(fit_p)
            for i in range(len(self.models)):
                # recompute p_val from new p_free
                self.models[i].set_p_free(p_fit_list[i])
                _ipval = self.models[i].get_p_val()
                full_p_list.append(gvar.mean(_ipval))
                full_p_list_err.append(gvar.sdev(_ipval))
                full_p_list_gvar.append(_ipval)
                
            returned_data['fit_params'] = full_p_list
            returned_data['fit_params_err'] = full_p_list_err
            if not self.nogvar:
                returned_data['fit_params_gvar'] = full_p_list_gvar

            
            ## compute error on parameters
            # compute reduced chi square
            returned_data['rchi2'] = fit.chi2 / fit.dof
            returned_data['rchi2_err'] = np.sqrt(2./self._get_vector_onrange().shape[0])
            if not self.nogvar:
                returned_data['rchi2_gvar'] = gvar.gvar(returned_data['rchi2'],
                                                        returned_data['rchi2_err'])
            
            
            returned_data['chi2'] = fit.chi2
            returned_data['residual'] = residual * self.normalization_coeff
            # kolmogorov smirnov test: if p_value < 0.05 residual is not normal
            returned_data['ks_pvalue'] = scipy.stats.kstest(
                residual / np.std(orb.utils.stats.sigmacut(residual)), 'norm')[1]


            # compute MCMC uncertainty estimates
            if compute_mcmc_error:
                sigma = np.nanstd(last_diff)
                returned_data['fit_params_err_mcmc'] = self._compute_mcmc_error(
                    fit[0], cov_diag, sigma)

            returned_data['logGBF'] = fit.logGBF
            returned_data['fit_time'] = time.time() - start_time
            returned_data['signal_range'] = self.signal_range
            returned_data['nparams'] = priors_arr.size # number of free parameters
            # Bayesian information criterion
            returned_data['BIC'] = orb.utils.fit.BIC(returned_data['residual'], returned_data['nparams'])
            
        else:
            logging.debug('bad fit')
            return []


        if self.nogvar:
            del returned_data['fitted_vector_gvar']
            del returned_data['fitted_models_gvar']

        return returned_data
        
class Model(object):
    """
    Template class for fit models. This class cannot be used directly.

    The main purpose of a Model class is to output a model given a set
    of parameters.

    Methods that must be implemented by real classes:

      * :py:meth:`fit.Model.parse_dict`
      * :py:meth:`fit.Model.check_input`
      * :py:meth:`fit.Model.make_guess`
      * :py:meth:`fit.Model.get_model`


    .. note:: A model is computed from a given set of parameters
      stored in :py:attr:`fit.Model.p_val`. From this set some
      parameters are **free**, some are **fixed** and some are
      **covarying**, i.e. the value of a subset of parameters can be
      computed from 1 free parameter.

      Taking the definition of the parameters (free, fixed or
      covarying, stored in :py:attr:`fit.Model.p_def`) into account,
      the reduced free parameter set is stored in
      :py:attr:`fit.Model.p_free`, the reduced set of fixed parameters
      is stored in :py:attr:`fit.Model.p_fixed`, the set of covarying
      parameters is stored in :py:attr:`fit.Model.p_cov` and when the
      model needs to be computed, the full set of model parameters
      (:py:attr:`fit.Model.p_val`) is computed again from set.p_free`,
      :py:attr:`fit.Model.p_fixed`, :py:attr:`fit.Model.p_cov` and
      :py:attr:`fit.Model.p_def`.

      A group of covarying parameters is defined by the same label. If
      :py:attr:`fit.Model.p_def` is::

        ['free', 'fixed', '1', '2', 'free', 'fixed', '2', '1', '2', 'free']

      It means that we have 3 free parameters, 2 fixed parameters and
      2 groups of covarying parameters. The first group contains 2
      parameters and the second group contains 3 parameters. In this
      case the real number of free parameters will be 3 + 2 (one free
      parameter for each group of covarying parameters) = 5 and the
      real number of fixed parameters will be 2 + 5 (one fixed
      parameter for each covarying parameters) = 7.
      

      A Model class works this way :

      1. Init: the dictionary defining the parameters (free, fixed,
         covarying) and their values is parsed with
         :py:meth:`fit.Model.parse_dict`: :py:attr:`fit.Model.p_def`,
         :py:attr:`fit.Model.p_val` and :py:attr:`fit.Model.p_cov` are
         created. Then :py:meth:`fit.Model.val2free` is called to
         create :py:attr:`fit.Model.p_free` and
         :py:attr:`fit.Model.p_fixed`.
      
      2. the set of free parameters can then be changed with
         :py:meth:`fit.Model.set_p_free` before calling
         :py:meth:`fit.Model.get_model`. the updated values of
         :py:attr:`fit.Model.p_val` are computed before the model is created via
         :py:meth:`fit.Model.free2val`. A new set of free parameters
         can also be passed to :py:meth:`fit.Model.get_model`.
      

    """
    accepted_keys = ()
    """Accepted keys of the input dictionary (see
    :py:attr:`fit.Model.p_dict`)"""

    p_free = None
    """Up to date value of the free parameters. Its size is always
    less or equal to the size of the full set of parameters used
    directly to compute the model. It reflects the real number of
    fitted parameters. For each group of covarying parameters one free
    parameter is added. """

    p_bounds = None
    """tuples of bounds (lower, upper) on the free parameters"""

    p_fixed = None
    """Array of fixed parameters. Each covarying parameter is stored
    as fixed. And one free parameter is added for each group of
    covarying parameters."""
    
    p_dict = None
    """Input dictionary defining the parameters. Contains the initial
    values of the parameters"""
    
    p_def = None
    """Definition of the full set of parameters (fixed, free or
    covarying). This array as the same shape as :py:attr:`fit.Model.p_val`"""
    
    p_val = None
    """Up to date values of the full set of parameters used by the model
    (initial guess before fit, fitted value after fit). This array as
    the same shape as :py:attr:`fit.Model.p_def`. It does not reflect the real number
    of fitted parameters."""

    p_cov = None
    """dict that stores the groups of covarying parameters by label
    and their associated value and covarying operation (a pointer to a
    function), i.e.::

      {['cov_label_1': (value1, cov_operation1)],
       ['cov_label_2': (value2, cov_operation2)],
       ...}
     """

    def __init__(self, p_dict):
        """ Initialize model

        :param p_dict: Input dictionary defining the parameters of the
          model.
        
        parameters definition can be : 'free', 'fixed' or covarying
        (in this case any string label can be used to define a group
        of covarying parameters)

        During init p_dict is parsed with
        :py:meth:`fit.Model.parse_dict`: :py:attr:`fit.Model.p_def`,
        :py:attr:`fit.Model.p_val` and :py:attr:`fit.Model.p_cov` are
        created. Then :py:meth:`fit.Model.val2free` is called to create
        :py:attr:`fit.Model.p_free` and :py:attr:`fit.Model.p_fixed`.
        """
        # parse input dict
        if isinstance(p_dict, dict):
            self.p_dict = dict(p_dict)
            # create a copy of the dict which keys will be popped
            # during init. If there are still keys at the end of the
            # init an error will be raised.
            self.unused_keys = dict(self.p_dict)

            for key in list(self.p_dict.keys()):
                if key not in self.accepted_keys:
                    raise Exception('Input dictionary contains unknown key: {}'.format(key))
        
            self.parse_dict()
        else: raise ValueError('p must be a dict')

        
        # check input parameters
        self.check_input()
        
        # create free and fixed vectors
        self.val2free()
        if len(self.unused_keys) != 0:
            raise orb.utils.err.FitInitError('Some input keys where not used during fit init: {}'.format(list(self.unused_keys.keys())))

    def parse_dict(self):
        """Parse input dictionary to create :py:attr:`fit.Model.p_def`, :py:attr:`fit.Model.p_val` and
        :py:attr:`fit.Model.p_cov`"""
        raise NotImplementedError()

    def check_input(self):
        """Check input parameters"""
        raise NotImplementedError()

    def make_guess(self, v):
        """If a parameter value at init is a NaN this value is guessed.

        :param v: Data vector from which the guess is made.
        """
        raise NotImplementedError()

    def get_model(self, x, return_models=False, return_complex=False):
        """Compute a model M(x, p) for all passed x positions. p are
        the parameter values stored in :py:attr:`fit.Model.p_val`

        :param x: Positions where the model M(x, p) is computed.

        :param return_models: (Optional) If True return also
          individual models (default False)

        :param return_complex: (Optional) If True return a complex
          model (default False).
        """
        raise NotImplementedError()

    def get_p_free(self):
        """Return the vector of free parameters :py:attr:`fit.Model.p_free`"""
        return copy.copy(self.p_free)

    def get_priors(self):
        """Return priors
        """
        priors = dict(self.get_p_free())
        return priors

    def get_p_bounds(self):
        bounds = dict(self.get_p_free())
        for ikey in bounds:
            bounds[ikey] = (-np.inf, np.inf)
        return bounds
        
    def set_p_free(self, p_free):
        """Set the vector of free parameters :py:attr:`fit.Model.p_free`

        :param p_free: New vector of free parameters
        """
        if self.p_free.keys() == p_free.keys():
            self.p_free = dict(p_free)
            self.free2val()
        else: raise Exception('bad format of passed free parameters')

    def get_p_val(self):
        """Return :py:attr:`fit.Model.p_val` """
        return copy.copy(self.p_val)

    def set_p_val(self, p_val):
        """Set :py:attr:`fit.Model.p_val`

        .. warning:: This method is used to bypass all the initialized
          parameters and reuse an already initialized model with
          another full set of parameters. Note that you might want to
          call :py:meth:`fit.Model.get_model` directly after this
          method because any call to :py:meth:`fit.Model.set_p_free`
          or :py:meth:`fit.Model.free2val` will recompute
          :py:attr:`fit.Model.p_val` from the init values and the
          actual :py:attr:`fit.Model.p_free`.
        
        :param p_val: New full set of parameters.
        """
        if list(p_val.keys()) == list(self.p_val.keys()):
            self.p_val = copy.copy(p_val)
            self.val2free()
        else: raise Exception('bad format of passed val parameters')
        
    def val2free(self):
        """Recompute the set of free parameters
        :py:attr:`fit.Model.p_free` with the updated values of
        :py:attr:`fit.Model.p_val`"""
        if self.p_val is None or self.p_def is None or self.p_cov is None:
            raise Exception('class has not been well initialized: p_val, p_def and p_cov must be defined')
        self.p_free = dict()
        self.p_fixed = dict()
        passed_cov = list()
        for idef in self.p_def:
            if self.p_def[idef] == 'free':
                self.p_free[idef] = self.p_val[idef]
            elif self.p_def[idef] == 'fixed':
                self.p_fixed[idef] = self.p_val[idef]
            else:
                if self.p_def[idef] not in passed_cov:
                    self.p_free[self.p_def[idef]]= self.p_cov[self.p_def[idef]][0]
                self.p_fixed[idef] = self.p_val[idef]
                passed_cov += list([self.p_def[idef]])

        # remove sdev from p_fixed
        for idef in self.p_fixed:
            if self.p_fixed[idef] is not None:
                self.p_fixed[idef] = gvar.mean(self.p_fixed[idef]) 
        
    def free2val(self):
        """Read the array of parameters definition
        :py:attr:`fit.Model.p_def` and update the parameter values
        based on the new set of free parameters
        :py:attr:`fit.Model.p_free`.
        """
        if self.p_free is None or self.p_fixed is None or self.p_def is None or self.p_cov is None:
            raise Exception('class has not been well initialized, p_free, p_fixed, p_def and p_cov must be defined')

        passed_cov = dict()

        self.p_val = dict()

        for idef in self.p_def:
            if self.p_def[idef] == 'free':
                self.p_val[idef] = self.p_free[idef]
            elif self.p_def[idef] == 'fixed':
                self.p_val[idef] = self.p_fixed[idef]
            else: # covarying parameter
                if self.p_def[idef] not in passed_cov:
                    # if not already taken into account
                    passed_cov[self.p_def[idef]] = self.p_free[self.p_def[idef]]
                # covarying operation
                self.p_val[idef] = self.p_cov[self.p_def[idef]][1](
                    self.p_fixed[idef], passed_cov[self.p_def[idef]])
                
class FilterModel(Model):
    """
    Simple model of filter based on a real filter shape. The only
    possible free parameter is a wavelength/wavenumber shift.

    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'filter_function':,
       'shift_guess':,
       'shift_def':}

    :keyword filter_function: Transmission of the filter over the
      fitted spectral range (axis must be exactly the same).
      
    :keyword shift_guess: Guess on the filter shift in pixels.
    
    :keyword shift_def: Definition of the shift parameter, can be
      'free' or 'fixed'

    .. note:: This model must be multiplied with the other and used
      last.
    """

    accepted_keys = ('filter_function',
                     'shift_guess',
                     'shift_def')
    """Accepted keys of the input dictionary (see
    :py:attr:`fit.Model.p_dict`)"""

    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        if 'filter_function' in self.p_dict:
            self.filter_function = self.p_dict['filter_function']
            self.filter_function[np.isnan(self.filter_function)] = 0
            self.filter_axis = np.arange(self.filter_function.shape[0])
            self.filter_function = scipy.interpolate.UnivariateSpline(
                self.filter_axis, self.filter_function,
                k=1, s=0, ext=1)
            self.unused_keys.pop('filter_function')
        else: 
            raise Exception('filter_function must be given')

        if 'shift_guess' in self.p_dict:
            shift_guess = self.p_dict['shift_guess']
            self.unused_keys.pop('shift_guess')
        else: shift_guess = 0.
        self.p_val = {'filter_shift': shift_guess}
        
        if 'shift_def' in self.p_dict:
            shift_def = self.p_dict['shift_def']
            self.unused_keys.pop('shift_def')
        else:
            shift_def = 'free'
        self.p_def = {'filter_shift': shift_def}
        
        self.p_cov = dict()

    def check_input(self):
        pass

    def make_guess(self, v):
        pass

    def get_model(self, x, p_free=None, return_models=False, return_complex=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).
          
        :param return_models: (Optional) If True return also
          individual models (default False)

        :param return_complex: (Optional) If True return a complex
          model (default False).
        """
        if p_free is not None:
            self.set_p_free(p_free)
            
        self.free2val()
        if len(self.p_free) == 0:
            mod = copy.copy(self.filter_function(self.filter_axis))
        else:
            mod = copy.copy(self.filter_function(
                self.filter_axis
                + gvar.mean(self.p_free['filter_shift'])))

        if return_complex:
            mod = orb.utils.vector.complex2float((mod, np.zeros_like(mod)))

        if return_models:
            return mod, (mod)
        else:
            return mod


class ContinuumModel(Model):
    """
    Polynomial continuum model.

    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'poly_order':
       'poly_guess':}

    :keyword poly_order: Order of the polynomial to fit (be careful
      with high order polynomials).

    :keyword poly_guess: Initial guess on the coefficient values :
      must be a tuple of length poly_order + 1.

    .. note:: This model must be added to the others.
    """

    accepted_keys = ('poly_def',
                     'poly_order',
                     'poly_guess')
    """Accepted keys of the input dictionary (see
    :py:attr:`fit.Model.p_dict`)"""

    def _get_ikey(self, ip):
        """Return key corresponding to a coefficient of order ip

        :param ip: order of the coefficient
        """
        return 'cont_p{}'.format(int(ip))

    def _get_order_from_key(self, key):
        """Return the line nb of a given key

        :param key: Key to get line number from.
        """
        return int(key[6:])
        
    def get_p_val_as_array(self, p_val=None):
        if p_val is None:
            if p_val.keys() == self.p_val.keys():
                p_val = dict(self.p_val)
            else: raise Exception('Badly formatted p_val')

        ans = np.empty(self.poly_order + 1)

        for ipar in p_val:
            i = self._get_order_from_key(ipar)
            ans[i] = p_val[ipar]
            
        return ans
    
    
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        if 'poly_order' in self.p_dict:
            self.poly_order = int(self.p_dict['poly_order'])
            self.unused_keys.pop('poly_order')
        else: self.poly_order = 0

        self.p_val = dict()
        self.p_def = dict()    
        self.p_cov = dict() # no covarying parameters

        for ip in range(self.poly_order + 1):
            self.p_val[self._get_ikey(ip)] = None
            self.p_def[self._get_ikey(ip)] = 'free'

        if 'poly_def' in self.p_dict:
            if self.p_dict['poly_def'] is not None:
                if np.size(self.p_dict['poly_def']) == self.poly_order + 1:
                    for ip in range(self.poly_order + 1):
                        self.p_def[self._get_ikey(ip)] = self.p_dict['poly_def'][ip]
                else: raise Exception('poly_def must be an array of size equal to poly_order + 1')
            self.unused_keys.pop('poly_def')
            
        if 'poly_guess' in self.p_dict:
            if self.p_dict['poly_guess'] is not None:
                if np.size(self.p_dict['poly_guess']) == self.poly_order + 1:
                    for ip in range(self.poly_order + 1):
                        self.p_val[self._get_ikey(ip)] = self.p_dict['poly_guess'][ip]
                else: raise Exception('poly_guess must be an array of size equal to poly_order + 1')
            self.unused_keys.pop('poly_guess')

    def check_input(self):
        pass

    def make_guess(self, v):
        for key in list(self.p_val.keys()):
            if self.p_val[key] is None:
                order = self._get_order_from_key(key)
                self.p_val[key] = np.nanmedian(v)**(1./(order+1))
        self.val2free()

    def get_model(self, x, p_free=None, return_models=False, multf=None, return_complex=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).

        :param multf: 1d vector with the same length as x vector which
          represent the function by which the model must be multiplied.
          
        :param return_models: (Optional) If True return also
          individual models (default False)

        :param return_complex: (Optional) If True return a complex
          model (default False).
        """
        if p_free is not None:
            self.set_p_free(p_free)
            
        self.free2val()
        coeffs = [self.p_val[self._get_ikey(ip)] for ip in range(self.poly_order + 1)]
        mod = np.polyval(coeffs, x)
        
        if multf is not None:
            if isinstance(multf[0], gvar.GVar):
                multfsp_mean = scipy.interpolate.UnivariateSpline(
                    x, gvar.mean(multf), k=1, s=0, ext=2)
                multfsp_sdev = scipy.interpolate.UnivariateSpline(
                    x, gvar.sdev(multf), k=1, s=0, ext=2)
                mod *= gvar.gvar(multfsp_mean(x), multfsp_sdev(x))
            else:
                multfsp = scipy.interpolate.UnivariateSpline(
                    x, multf, k=1, s=0, ext=2)
                mod *= multfsp(x)

        if np.any(np.isnan(gvar.mean(mod))):
            logging.debug('Nan in model')

        if return_complex:
            mod = orb.utils.vector.complex2float((mod, np.zeros_like(mod)))

        if return_models:
            return mod, (mod)
        else:
            return mod
        
class LinesModel(Model):
    """
    Emission/absorption lines model with a channel unity in pixels.

    .. note:: This class is best seen as a basic class implemented
      with more physical unities by :py:class:`fit.Cm1LinesModel` or
      :py:class:`fit.NmLinesModel`.

    .. note:: Each line is built on 3 (or more) parameters : amplitude,
      FWHM, position and sigma/alpha (the 4th and 5th parameters are used only for some models -- see
      below for details on the different models).

      Some lines can have one or more covarying parameters: FWHM can
      be the same for all the lines (this is True if lines are not
      resolved), lines issued from the same ion can have the same
      speed (e.g. [NII] doublet, [SII] doublet, [OIII] doublet), and
      some fixed transition ratios between lines can also be set
      (e.g. [NII]6584/[NII]6548 can be set to 2.89, when [NII]6548 is
      likely to be really noisy).
    
    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'line_nb':,
       'fmodel':,
       'amp_def':,
       'pos_def':,
       'fwhm_def':,
       'sigma_def':, # only for sincgauss fmodel
       'alpha_def':, # only for sincphased fmode
       'amp_cov':,
       'pos_cov':,
       'fwhm_cov':,
       'sigma_cov':, # only for sincgauss fmodel
       'alpha_cov':, # only for sincphased fmodel
       'amp_guess':,
       'pos_guess':,
       'fwhm_guess':,
       'sigma_guess':, # only for sincgauss fmodel
       'alpha_guess':} #  only for sincphased fmodel

    :keyword line_nb: Number of lines.

    :keyword fmodel: Line shape, can be 'gaussian', 'sinc', 'sinc2' or
      'sincgauss'.

    :keyword amp_def: Definition of the amplitude parameter, can be
      'free', 'fixed' or set to a label that defines its covarying
      group.

    :keyword pos_def: Definition of the position parameter in pixels,
      can be 'free', 'fixed' or set to a label that defines its
      covarying group.

    :keyword fwhm_def: Definition of the FWHM parameter in pixels, can
      be 'free', 'fixed' or set to a label that defines its covarying
      group.

    :keyword sigma_def: Definition of the sigma parameter in pixels,
      can be 'free', 'fixed' or set to a label that defines its
      covarying group.

    :keyword amp_cov: Guess on the covariant value of the amplitude
      (best set to 0 in general). There must be as many values as
      covarying amplitude groups or only one value if it is the same
      for all groups.

    :keyword pos_cov: Guess on the covariant value of the velocity (in
      pixels). There must be as many values as covarying amplitude
      groups or only one value if it is the same
      for all groups.

    :keyword fwhm_cov: Guess on the covariant value of the FWHM
      (best set to 0 in general). There must be as many values as
      covarying amplitude groups or only one value if it is the same
      for all groups.

    :keyword sigma_cov: Guess on the covariant value of sigma (best
       set to 0 in general). There must be as many values as covarying
       amplitude groups or only one value if it is the same for all
       groups.

    :keyword amp_guess: Initial guess on the amplitude value of the
       lines. Best set to a NaN in general (it can be automatically
       guessed with good robusteness). But if lines have a covarying
       amplitude the initial guess fixes their ratio.

    :keyword pos_guess: Initial guess on the position of the lines:
       the PRECISE rest frame position must be given here, especially
       if lines share a covarying position, because their relative
       position will be fixed.

    :keyword fwhm_guess: Initial guess on the FWHM of the lines. This
      guess must be the MOST PRECISE possible (to a few 10%), it is by
      far the most unstable parameter especially for sinc lines.
     
    :keyword sigma_guess: Initial guess on the value of sigma. Best
      set to 0. in general


    Example: A red spectrum containing [NII]6548, Halpha, [NII]6584,
    [SII]6716 and [SII]6731, with a mean velocity of 1500 km/s (which
    translates in a pixel shift of 5.5), with a fixed amplitude ratio
    netween [NII] lines, the same speed for lines issued from the same
    ions and a shared FWHM between everybody but Halpha would be
    defined this way::
    
      {'line_nb' : 5,
       'amp_def' : ('1', 'free', '1', 'free', 'free'),
       'pos_def' : ('1', '2', '1', '3', '3'), 
       'fwhm_def': ('1', '2', '1', '1', '1'),
       'amp_cov': 0.,
       'pos_cov': 5.5,
       'fwhm_cov': 0.,
       'amp_guess': (1., np.nan, 2.89, np.nan, np.nan), # here the amplitude ratio between covarying [NII] lines is fixed.
       'pos_guess': (40,60,80,120,140), # positions are given in pixel and are purely arbitrary in this example
       'fwhm_guess': 2.43}

    .. note::
      Line shapes (fmodel keyword):
      
      * **gaussian**: A classical gaussian line shape. See :py:meth:`cutils.gaussian1d`.
      
      * **sinc**: A pure sinc line shape, True interferometric line shape
        if lines are strongly unresolved and if the interferometer has
        no assymetry (generally good on SITELLE/SpIOMM low res cubes
        --i.e. less than 500 steps-- if the line SNR is not too high
        --i.e. < 50--). See :py:meth:`cutils.sinc1d`.
      
      * **sinc2**: sinc2 = sqrt(sinc**2.). Can be used for spectra not
        corrected in phase (where the absolute value of the complex
        spectrum is taken).
      
      * **sincgauss**: Convolution of a Gaussian (of width **sigma**) and
        a sinc (FWHM). This line shape has a 4th parameter:
        sigma. This is much closer to the true line shape, but it
        takes much more time to compute because of the generally very
        small value of sigma. This can be used to fit resolved lines,
        like e.g. Halpha in absorption or active nucleus with broader
        emission. See :py:meth:`cutils.sincgauss1d`.
      
    """

    accepted_keys = ('line_nb',
                     'fmodel',
                     'ratio',
                     'amp_def',
                     'pos_def',
                     'fwhm_def',
                     'sigma_def', # only for sincgauss fmodel
                     'alpha_def', # only for sincphased fmode
                     'amp_cov',
                     'pos_cov',
                     'fwhm_cov',
                     'sigma_cov', # only for sincgauss fmodel
                     'alpha_cov', # only for sincphased fmodel
                     'amp_guess',
                     'pos_guess',
                     'fwhm_guess',
                     'sigma_guess', # only for sincgauss fmodel
                     'alpha_guess')
    """Accepted keys of the input dictionary (see
    :py:attr:`fit.Model.p_dict`)"""

    p_array = None
    """equivalent of :py:attr:`fit.Model.p_val` but presented as an
    array with each row corresponding to a line which is easier to
    handle."""


    param_keys = ['amp', 'pos', 'fwhm', 'sigma', 'alpha']
    """Parameter keys"""

    log_param_keys = ['fwhm', 'sigma']
    """Parameter keys that have a lognormal distribution"""

    same_param_keys = ['fwhm', 'sigma']
    """Parameter keys which must be the same if covarying"""

    def _get_ikey(self, key, iline):
        """Return key for a given line nb in
        :py:attr:`fit.Model.p_val` and :py:attr:`fit.Model.p_def`
        dictionnaries

        :param key: may be 'amp', 'pos', 'fwhm', 'sigma', 'alpha'
        :param iline: line number.
        """
        if key in self.param_keys:
            if 0 <= iline < self._get_line_nb():
                return '{}{}'.format(key, iline)
            else: raise Exception('Invalid line index, must be >=0 and < {}'.format(self._get_line_nb()))
        else: raise Exception('Invalid paramter key. Must be in {}'.format(self.param_keys))

    def _get_iline_from_key(self, key):
        """Return the line nb of a given key

        :param key: Key to get line number from.
        """
        for _k in self.param_keys:
            if _k in key:
                return int(key[len(_k):])
        raise Exception('Invalid format for key')

    def get_p_val_as_array(self, p_val=None):
        if p_val is None:
            if p_val.keys() == self.p_val.keys():
                p_val = dict(self.p_val)
            else: raise Exception('Badly formatted p_val')

        ans = np.empty((self._get_line_nb(),
                       len(p_val) // self._get_line_nb()))

        for ipar in p_val:
            iline = self._get_iline_from_key(ipar)
            if 'amp' in ipar:
                ans[iline, 0] = p_val[ipar]
            if 'pos' in ipar:
                ans[iline, 1] = p_val[ipar]
            if 'fwhm' in ipar:
                ans[iline, 2] = p_val[ipar]
            if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                if 'sigma' in ipar:
                    ans[iline, 3] = p_val[ipar]    
            if self._get_fmodel() in ['sincphased']:
                if 'alpha' in ipar:
                    ans[iline, 3] = p_val[ipar]
            if self._get_fmodel() in ['sincgaussphased']:
                if 'alpha' in ipar:
                    ans[iline, 4] = p_val[ipar]
            
        return ans    

    def get_priors(self):
        """Return priors. Replace gaussian distribution by lognormal
        distribution for some parameters.
        """
        priors = dict(self.get_p_free())
        for key in priors:
            priors[key] = gvar.mean(priors[key])
        return priors    

    def get_p_bounds(self, force_positive_flux=True):
        bounds = dict(self.get_p_free())
        for ikey in bounds:
            if 'amp' in ikey:
                if force_positive_flux:
                    bounds[ikey] = (0, np.inf)
                else:
                    bounds[ikey] = (-np.inf, np.inf)
            elif 'fwhm' in ikey:
                bounds[ikey] = (0, np.inf)
            elif 'sigma' in ikey:
                bounds[ikey] = (0, np.inf)
            else:
                bounds[ikey] = (-np.inf, np.inf)
        return bounds

    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        
        def parse_param(key, cov_operation):

            key_guess = key + '_guess'
            key_def = key + '_def'
            key_cov = key + '_cov'

            ## parse guess
            p_guess = dict()
            if key_guess in self.p_dict:
                orb.utils.validate.has_len(self.p_dict[key_guess], line_nb,
                                       object_name=key_guess)

                self.p_dict[key_guess] = np.atleast_1d(self.p_dict[key_guess])
                for iline in range(line_nb):
                    p_guess[self._get_ikey(key, iline)] = self.p_dict[key_guess][iline]
            else:
                for iline in range(line_nb):
                    p_guess[self._get_ikey(key, iline)] = None
                                                
            ## parse cov  
            if key_cov in self.p_dict:
                orb.utils.validate.is_iterable(self.p_dict[key_cov], object_name=key_cov)
                p_cov = np.atleast_1d(self.p_dict[key_cov])
            else:
                p_cov = None

            ## parse def
            p_cov_dict = dict()
            p_def = dict()
            if key_def in self.p_dict: # gives the definition of the parameter
                orb.utils.validate.has_len(self.p_dict[key_def], line_nb,
                                       object_name=key_def)

                for iline in range(line_nb):
                    p_def[self._get_ikey(key, iline)] = self.p_dict[key_def][iline]
                    
                # manage cov values
                for iline in range(line_nb):
                    if p_def[self._get_ikey(key, iline)] not in ['free', 'fixed']:
                        cov_symbol = str(key_def + str(p_def[self._get_ikey(key, iline)]))
                        # create singular symbol
                        p_def[self._get_ikey(key, iline)] = cov_symbol
                        
                        # fill cov dict
                        if cov_symbol not in p_cov_dict:
                            if p_cov is None:
                                # todo: set it to a covarying value
                                # which depends on the parameter
                                # operation (amp=1, others=0)
                                cov_value = 0. 
                            else:
                                if np.size(p_cov) > 0:
                                    cov_value = p_cov[0]
                                else:
                                    raise TypeError("not enough covarying parameters: {} must have the same size as the number of covarying parameters".format(key_cov))
                                p_cov = p_cov[1:] # used cov values are dumped
                    
                            p_cov_dict[cov_symbol] = (
                                np.squeeze(cov_value), cov_operation)
                            
                    elif p_cov is not None:
                        if np.size(p_cov) != 0: raise ValueError("{}_cov must not be set ({}) if {}_def is set to 'free' or 'fixed'".format(key, self.p_dict[key_cov], key))
                    logging.debug('Covarying symbols for {}: {}'.format(key, list(p_cov_dict.keys())))
                if p_cov is not None:
                    if np.size(p_cov) > 0: raise TypeError("too much covarying parameters ({}): {} ({}) must have the same size as the number of covarying parameters ({})".format(np.size(p_cov) + len(p_cov_dict), key_cov, self.p_dict[key_cov], len(p_cov_dict)))
            else:
                for iline in range(line_nb):
                    p_def[self._get_ikey(key, iline)] = 'free'

            self.p_def.update(p_def)
            self.p_val.update(p_guess)
            self.p_cov.update(p_cov_dict)
            if key_guess in self.p_dict: self.unused_keys.pop(key_guess)
            if key_def in self.p_dict: self.unused_keys.pop(key_def)
            if key_cov in self.p_dict: self.unused_keys.pop(key_cov)
            

        line_nb = self._get_line_nb()
        self.p_def = dict()
        self.p_val = dict()
        self.p_cov = dict()
        parse_param('amp', self._get_amp_cov_operation())
        parse_param('pos', self._get_pos_cov_operation())
        parse_param('fwhm', self._get_fwhm_cov_operation())
        if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
            parse_param('sigma', self._get_sigma_cov_operation())
        if self._get_fmodel() in ['sincphased', 'sincgaussphased']:
            parse_param('alpha', self._get_alpha_cov_operation())    

        # check cov values and def for log parameters sigma and fwhm
        for key_cov in self.p_cov:
            for key in self.same_param_keys:
                if key in key_cov:
                    keys_def = [key_def for key_def in self.p_def if self.p_def[key_def] == key_cov]
                    vals = [self.p_val[ikey_def] for ikey_def in keys_def]
                    vals = np.array(vals)
                    if np.any(vals - vals[0] != 0.): raise Exception('{} parameter must be the same for all the lines of the same covarying group'.format(key))
                    for ikey_def in keys_def:
                        self.p_val[ikey_def] = 0.
                    self.p_cov[key_cov] = (self.p_cov[key_cov][0] + vals[0], self.p_cov[key_cov][1])
                
        
    def _get_amp_cov_operation(self):
        """Return covarying amplitude operation"""
        return lambda x, y: x * y

    def _get_fwhm_cov_operation(self):
        """Return covarying FWHM operation"""
        return lambda x, y: x + y

    def _get_pos_cov_operation(self):
        """Return covarying position operation"""
        return lambda x, y: x + y

    def _get_sigma_cov_operation(self):
        """Return covarying sigma operation"""
        return lambda x, y: x + y

    def _get_alpha_cov_operation(self):
        """Return covarying alpha operation"""
        return lambda x, y: x + y


    def check_input(self):
        """Check input parameters"""
        for key in list(self.p_val.keys()):
            if self.p_val[key] is None:
                if 'pos' in key:
                    logging.warning('No initial position given')
                if 'fwhm' in key:
                    logging.warning('No initial fwhm given')
                if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                    if 'sigma' in key:
                        logging.warning('No initial sigma given')
                if self._get_fmodel() in ['sincphased', 'sincgaussphased']:
                    if 'alpha' in key:
                        logging.warning('No initial alpha given')
        ratio = self._get_ratio()
        if ratio is None:
            raise ValueError('ratio must be set to something else than None (e.g. 0.25)')
        if not 0 < ratio < 1:
            raise ValueError('ratio must be strictly between 0 and 1')



    def make_guess(self, v):
        """If a parameter value at init is a NaN this value is guessed.

        :param v: Data vector from which the guess is made.
        """
        FWHM_INIT = 2. * orb.constants.FWHM_SINC_COEFF
        FWHM_COEFF = 6.

        self._p_val2array()

        for key in list(self.p_array.keys()):
            if self.p_array[key] is None or np.isnan(self.p_array[key]): 
                if 'pos' in key:
                    raise Exception('initial guess on lines position must be given, no automatic lines detection is implemented at this level.')
                
                if 'fwhm' in key:
                    self.p_array[key] = FWHM_INIT
                    
                if 'sigma' in key:
                    self.p_array[key] = 1e-8
                    
                if 'alpha' in key:
                    self.p_array[key] = 1e-8

        # amp must be checked after the others
        for key in list(self.p_array.keys()):
            if self.p_array[key] is None:
                if 'amp' in key:
                    self.p_array[key] = np.nanmax(gvar.mean(v))

        self._p_array2val()
        self.val2free()
        

    def _get_line_nb(self):
        """Return the number of lines"""
        if 'line_nb' in self.p_dict:
            if 'line_nb' in self.unused_keys: self.unused_keys.pop('line_nb')
            return self.p_dict['line_nb']
        else:
            raise ValueError("'line_nb' must be set")

    def _get_ratio(self):
        """Return the ratio value (shortest side of the interferogram over longest side)"""
        if 'ratio' in self.p_dict:
            if 'ratio' in self.unused_keys: self.unused_keys.pop('ratio')
            return self.p_dict['ratio']
        else:
            raise ValueError("'ratio' must be set")

    def _get_fmodel(self):
        """Return the line model"""
        if 'fmodel' in self.p_dict:
            if 'fmodel' in self.unused_keys: self.unused_keys.pop('fmodel')
            return self.p_dict['fmodel']
        else:
            raise ValueError("'fmodel' must be set")
        
    def _p_val2array(self):
        self.p_array = dict(self.p_val)

    def _p_array2val(self, p_array=None):
        """Transform :py:attr:`fit.LinesModel.p_array` to :py:attr:`fit.Model.p_val`."""
        if p_array is None:
            self.p_val = dict(self.p_array)
        else:
            self.p_val = dict(p_array)


    def get_model(self, x, p_free=None, return_models=False, multf=None, return_complex=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).
          
        :param multf: 1d vector with the same length as x vector which
          represent the function by which the model must be multiplied.
          
        :param return_models: (Optional) If True return also
          individual models (default False)

        :param return_complex: (Optional) If True return a complex model.
        """
        if p_free is not None:
            self.set_p_free(p_free)

        self.free2val()
        self._p_val2array()
        line_nb = self._get_line_nb()
        fmodel = self._get_fmodel()
        ratio = self._get_ratio()
        mod = None
        models = list()
        
        if multf is not None:
            if isinstance(multf[0], gvar.GVar):
                multfsp_mean = scipy.interpolate.UnivariateSpline(
                    x, gvar.mean(multf), k=1, s=0, ext=2)
                multfsp_sdev = scipy.interpolate.UnivariateSpline(
                    x, gvar.sdev(multf), k=1, s=0, ext=2)
            else:
                multfsp = scipy.interpolate.UnivariateSpline(
                    x, multf, k=1, s=0, ext=2)

        for iline in range(line_nb):
            if multf is not None:
                try:
                    mult_amp = gvar.gvar(
                        multfsp_mean(gvar.mean(self.p_array[self._get_ikey('pos', iline)])),
                        multfsp_sdev(gvar.mean(self.p_array[self._get_ikey('pos', iline)])))
                except UnboundLocalError:
                    mult_amp = multfsp(gvar.mean(self.p_array[self._get_ikey('pos', iline)]))
            else:
                mult_amp = 1.

            if np.any(np.isnan(gvar.mean(mult_amp))): raise Exception('Nan in mult_amp')

            
            if fmodel == 'sinc':
                if return_complex:
                    model_function = orb.utils.spectrum.sinc1d_complex
                else:
                    model_function = orb.utils.spectrum.sinc1d
                    
                line_mod = model_function(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)])
                
            elif fmodel == 'mertz':
                line_mod = orb.utils.spectrum.mertz1d(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)],
                    ratio)
                if not return_complex:    
                    line_mod = line_mod[0] # return only the real part
                
            elif fmodel == 'sincgauss':
                if return_complex:
                    model_function = orb.utils.spectrum.sincgauss1d_complex
                else:
                    model_function = orb.utils.spectrum.sincgauss1d

                line_mod = model_function(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)],
                    self.p_array[self._get_ikey('sigma', iline)])

            elif fmodel == 'sincphased':
                if return_complex:
                    raise NotImplementedError('sincphased model not implemeted for complex vector')
                line_mod = orb.utils.spectrum.sinc1d_phased(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)],
                    self.p_array[self._get_ikey('alpha', iline)])

            elif fmodel == 'sincgaussphased':
                if return_complex:
                    raise NotImplementedError('sincgaussphased model not implemeted for complex vector')
                line_mod = orb.utils.spectrum.sincgauss1d_phased(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)],
                    self.p_array[self._get_ikey('sigma', iline)],
                    self.p_array[self._get_ikey('alpha', iline)])

            elif fmodel == 'sinc2':
                raise NotImplementedError()
                ## line_mod =  np.sqrt(utils.spectrum.sinc1d(
                ##     x, 0., p_array[iline, 0],
                ##     p_array[iline, 1], p_array[iline, 2])**2.)
                
            elif fmodel == 'gaussian':
                line_mod = orb.utils.spectrum.gaussian1d(
                    x, 0.,
                    self.p_array[self._get_ikey('amp', iline)],
                    self.p_array[self._get_ikey('pos', iline)],
                    self.p_array[self._get_ikey('fwhm', iline)])
            else:
                raise ValueError("fmodel must be set to 'sinc', 'gaussian', 'sincgauss', 'sincphased', 'sincgaussphased' or 'sinc2'")


            if return_complex:
                line_mod = orb.utils.vector.complex2float(line_mod)
                
            line_mod *= mult_amp
            
            if mod is None:
                mod = np.copy(line_mod)
            else:
                mod += np.copy(line_mod)

            models.append(line_mod)

        if return_models:
            return mod, models
        else:
            return mod

class Cm1LinesModel(LinesModel):
    """Emission/absorption lines model with a channel unity in cm-1.

    Reimplements :py:class:`fit.LinesModel` to use more physical units
    : channels are translated to cm-1 and velocity to km/s in input
    and output.

    .. seealso:: For more information please refer to
      :py:class:`fit.LinesModel`
    
    """
    accepted_keys = list(LinesModel.accepted_keys) + list((
        'step_nb',
        'step',
        'order',
        'nm_laser',
        'nm_laser_obs'))
    """Accepted keys of the input dictionary (see
    :py:attr:`fit.Model.p_dict`)"""

    def _w2pix(self, w):
        """Translate wavenumber to pixels"""        
        return orb.utils.spectrum.fast_w2pix(w, self.axis_min, self.axis_step)


    def _pix2w(self, pix):
        """Translate pixel to wavenumber"""
        return orb.utils.spectrum.fast_pix2w(pix, self.axis_min, self.axis_step)

    def _get_pos_cov_operation(self):
        """Return covarying position operation for an input velocity in km/s"""
        return lambda lines, vel: lines * gvar.sqrt((1. - vel / orb.constants.LIGHT_VEL_KMS)
                                                    / (1. + vel / orb.constants.LIGHT_VEL_KMS))
         
    def _p_val2array(self):
        """Transform :py:attr:`fit.Model.p_val` to :py:attr:`fit.LinesModel.p_array`"""
        # get lines pos / fwhm
        # convert pos cm-1->pix
        # convert fwhm cm-1->pix
        lines_cm1 = list()
        fwhm_cm1 = list()
        sigma_kms = list()
        for iline in range(self._get_line_nb()):
            lines_cm1.append(self.p_val[self._get_ikey('pos', iline)])
            fwhm_cm1.append(self.p_val[self._get_ikey('fwhm', iline)])
            if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                sigma_kms.append(self.p_val[self._get_ikey('sigma', iline)])
        lines_pix = self._w2pix(np.array(lines_cm1))
        fwhm_pix = np.array(fwhm_cm1) / self.axis_step
        if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
            sigma_pix = orb.utils.fit.vel2sigma(
                np.array(gvar.mean(sigma_kms), dtype=float), lines_cm1, self.axis_step)
                        
        self.p_array = dict(self.p_val)
        for iline in range(self._get_line_nb()):
            self.p_array[self._get_ikey('pos', iline)] = lines_pix[iline]
            self.p_array[self._get_ikey('fwhm', iline)] = fwhm_pix[iline]
            if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                # convert sigma km/s->pix
                self.p_array[self._get_ikey('sigma', iline)] = sigma_pix[iline]

        return self.p_array

    def _p_array2val(self, p_array=None):
        """Transform :py:attr:`fit.LinesModel.p_array` to :py:attr:`fit.Model.p_val`."""
        if p_array is None:
            p_array = dict(self.p_array)
            
        # get lines cm1 / fwhm
        # convert pos pix->cm-1
        # convert fwhm pix->cm-1
        lines_pix = list()
        fwhm_pix = list()
        sigma_pix = list()
        for iline in range(self._get_line_nb()):
            lines_pix.append(p_array[self._get_ikey('pos', iline)])
            fwhm_pix.append(p_array[self._get_ikey('fwhm', iline)])
            if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                sigma_pix.append(p_array[self._get_ikey('sigma', iline)])
            
        lines_cm1 = self._pix2w(np.array(lines_pix))
        fwhm_cm1 = np.array(fwhm_pix) * self.axis_step
        if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
            sigma_kms = orb.utils.fit.sigma2vel(
                np.array(sigma_pix), gvar.mean(lines_cm1), self.axis_step)

        self.p_val = dict(p_array)
        for iline in range(self._get_line_nb()):
            self.p_val[self._get_ikey('pos', iline)] = lines_cm1[iline]
            self.p_val[self._get_ikey('fwhm', iline)] = fwhm_cm1[iline]
            if self._get_fmodel() in ['sincgauss', 'sincgaussphased']:
                self.p_val[self._get_ikey('sigma', iline)] = sigma_kms[iline]

        return self.p_val
      
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        LinesModel.parse_dict(self)

        if 'step_nb' not in self.p_dict:
            raise orb.utils.err.FitInitError('step_nb keyword must be set' )
        self.step_nb = float(self.p_dict['step_nb'])
        self.unused_keys.pop('step_nb')
        
        if 'step' not in self.p_dict:
            raise orb.utils.err.FitInitError('step keyword must be set' )
        self.step = float(self.p_dict['step'])
        self.unused_keys.pop('step')
        
        if 'order' not in self.p_dict:
            raise orb.utils.err.FitInitError('order keyword must be set' )
        self.order = int(self.p_dict['order'])
        self.unused_keys.pop('order')
        
        if 'nm_laser' not in self.p_dict:
            raise orb.utils.err.FitInitError('nm_laser keyword must be set' )
        self.nm_laser = float(self.p_dict['nm_laser'])
        self.unused_keys.pop('nm_laser')
    
        if 'nm_laser_obs' not in self.p_dict:
            raise orb.utils.err.FitInitError('nm_laser_obs keyword must be set' )
        self.nm_laser_obs = float(self.p_dict['nm_laser_obs'])
        self.unused_keys.pop('nm_laser_obs')
        
        self.correction_coeff = self.nm_laser_obs / self.nm_laser

        self.axis_min = orb.cutils.get_cm1_axis_min(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)
    
        self.axis_step = orb.cutils.get_cm1_axis_step(
            self.step_nb, self.step, corr=self.correction_coeff)


class NmLinesModel(Cm1LinesModel):
    """Emission/absorption lines model with a channel unity in nm.

    Reimplements :py:class:`fit.Cm1LinesModel` to use nm instead of
    cm-1. Channels are translated to cm-1 and velocity to km/s in
    input and output.

    .. seealso:: For more information please refer to
      :py:class:`fit.Cm1LinesModel` and :py:class:`fit.LinesModel`
    
    """
    def _get_pos_cov_operation(self):
        """Return covarying position operation for an input velocity in km/s"""
        return lambda lines, vel: lines * np.sqrt((1. + vel / orb.constants.LIGHT_VEL_KMS)
                                                  / (1. - vel / orb.constants.LIGHT_VEL_KMS))

    
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        raise Exception('Not re-implemented')
        Cm1LinesModel.parse_dict(self)
        
        self.axis_min = orb.cutils.get_nm_axis_min(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)
        self.axis_step = orb.cutils.get_nm_axis_step(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)

################################################
#### CLASS Params ##############################
################################################
class Params(orb.core.Params):
    """Manage a set of parameters as a special dictionary which
    elements can be accessed like attributes.
    """
    # parameters cannot be modified when accessed by attribute
    def __setattr__(self, key, value):
        raise Exception('Parameter is read-only')

    

#############################################
#### CLASS InputParams ######################
#############################################
class InputParams(object):

    # simulate the use of this class as a dict converted class
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, value): return setattr(self, key, value)    
       
    def __init__(self, step_nb):
        self.params = list()
        self.models = list()
        
        self.base_params = Params()
        self.base_params['step_nb'] = int(step_nb)

        self.allparams = Params()
        self.allparams.update(self.base_params)
        
        self.axis_min = 0
        self.axis_step = 1
        self.axis_max = self.base_params.step_nb
        self.set_signal_range(self.axis_min, self.axis_max)
        
    def append_model(self, model, operation, params):
        if self.has_model(model):
            raise orb.utils.err.FitInputError('{} already added'.format(model))
        self.models.append([model, operation])
        self.params.append(params)
        self.check_signal_range()
        self.allparams.update(params)

    def set_signal_range(self, rmin, rmax):    
        if (not (self.axis_min <= rmin < rmax)
            or not (rmin < rmax <= self.axis_max)):
            raise orb.utils.err.FitInputError('Check rmin and rmax values. Must be between {} and {}'.format(self.axis_min, self.axis_max))
        
        signal_range_pix = orb.utils.spectrum.fast_w2pix(
            np.array([rmin, rmax], dtype=float),
            self.axis_min, self.axis_step)
        minx = max(1, int(np.min(signal_range_pix)))
        maxx = min(self.base_params.step_nb - 1,
                   int(np.ceil(np.max(signal_range_pix))))
        self.signal_range = (minx, maxx)
        self.check_signal_range()

    def has_model(self, model):
        for imod in self.models:
            if model == imod[0]: return True
        return False

    def check_signal_range(self):
        pass

    def clean_kwargs(self, kwargs, params):
        # remove used params from kwargs
        for key in list(params.keys()):
            if key in kwargs:
                kwargs.pop(key)
        return kwargs

    def convert(self):
        """Convert class to a pickable dict object
        """
        raw = dict()
        raw['models'] = list(self.models)
        
        _params = list()
        # here we convert all gvars convertible arrays or values to a
        # _mean / _sdev couple to avoid pickling gvars. These couples
        # must then be merged again as gvars in FitVector.__init__()
        raw['params'] = orb.utils.fit.paramslist2pick(self.params)
        
        raw['signal_range'] = list(self.signal_range)
        raw['base_params'] = dict(self.base_params)
        
        raw['allparams'] = dict()
        for _iparams in raw['params']:
            raw['allparams'].update(_iparams)
        raw['baseclass'] = self.__class__.__name__
        return raw
   
    def add_continuum_model(self, **kwargs):
        params = Params()
        params['poly_order'] = 0
        params['poly_guess'] = None

        params.update(kwargs)
        
        # remove bad keys in case
        for key in list(params.keys()):
            if key not in ContinuumModel.accepted_keys:
                del params[key]

        self.append_model(ContinuumModel, 'add', params)

        return self.clean_kwargs(kwargs, params)

    def _check_lines_params(self, kwargs, fwhm_guess, lines):
        # check user defined params (kwargs)
        params = Params()
        params.update(kwargs)

        if not 'fmodel' in params:
            raise orb.utils.err.FitInputError('fmodel must be set')

        # check single valued params
        for iparam in params:
            if '_def' in iparam or '_guess' in iparam or '_cov' in iparam:
                ival = np.atleast_1d(params[iparam])
                if ival.size == 1:
                    if not '_cov' in iparam:
                        params[iparam] = list(ival) * np.size(lines)
                    else:
                        params[iparam] = list(ival)
                    logging.debug('changed single-valued parameter {}: {}'.format(
                        iparam, params[iparam]))
                
        # check sigma value        
        if params.fmodel in ['sincgauss', 'sincgaussphased']:
            if 'fwhm_def' in params:
                if np.any(np.array(params.fwhm_def, dtype=str) != 'fixed'):
                    logging.warning('fmodel is a sincgauss and FWHM is not fixed')
            else:
                params['fwhm_def'] = ['fixed'] * np.size(lines)
                
            if 'sigma_def' in params:
                sigma_cov_vel = self._get_sigma_cov_vel(fwhm_guess, lines)

                params['sigma_def'] = np.array(params.sigma_def, dtype=str)
                orb.utils.validate.has_len(params.sigma_def, np.size(lines), object_name='sigma_def')
                orb.utils.validate.has_len(sigma_cov_vel, np.size(lines), object_name='sigma_cov_vel')
                
                if 'sigma_guess' in params:
                    orb.utils.validate.has_len(params.sigma_guess, np.size(lines), object_name='sigma_guess')
                    
                    # sigma cov vel is adjusted to the initial guess + apodization
                    sqroots = list()
                    for i in range(np.size(lines)):
                        isqroot = np.sqrt(sigma_cov_vel[i]**2. + params.sigma_guess[i]**2.)
                    
                        sqroots.append(isqroot)
                    sigma_cov_vel = np.array(sqroots)

                    _sigma_guess = sigma_cov_vel
                else: _sigma_guess = np.zeros_like(lines)
                        
                if 'sigma_cov' not in params:
                    _sigma_cov = list()
                    _allcov = list()

                for ipar in range(len(params.sigma_def)):
                    if params.sigma_def[ipar] not in ['free', 'fixed']:
                        _sigma_guess[ipar] = 0. # must be set to 0 if covarying
                        if 'sigma_cov' not in params:
                            if params.sigma_def[ipar] not in _allcov:
                                _allcov.append(params.sigma_def[ipar])
                                _sigma_cov.append(sigma_cov_vel[ipar])

                params['sigma_guess'] = list(_sigma_guess)
                if 'sigma_cov' not in params and len(_sigma_cov) > 0:
                    params['sigma_cov'] = list(_sigma_cov)

        if 'line_nb' in params:
            logging.warning('line_nb reset by user')
            del params.line_nb # this parameter cannot be changed

        if 'pos_guess' in params:
            raise orb.utils.err.FitInputError("Line position must be defined with the 'lines' parameter")

        return params
    
    def add_lines_model(self, lines, fwhm_guess, **kwargs):
        
        lines = np.array(lines)

        line_nb = np.size(lines)

        default_params = {
            'fmodel':'sinc',
            'line_nb':line_nb,
            'amp_def':['free'] * line_nb,
            'fwhm_def':['fixed'] * line_nb,
            'fwhm_guess':[fwhm_guess] * line_nb,
            'pos_def':['free'] * line_nb,
            'pos_guess':lines}

        # check and update default params with user kwargs
        params = self._check_lines_params(kwargs, fwhm_guess, lines)
        
        if 'fwhm_guess' in params:
            raise orb.utils.err.FitInputError('This parameter must be defined with the non-keyword parameter fwhm_guess')
        
        default_params.update(params)
        all_params = Params()
        all_params.update(self.base_params)
        all_params.update(default_params)

        # remove bad keys in case
        for key in list(all_params.keys()):
            if key not in LinesModel.accepted_keys:
                logging.debug('key removed: {}'.format(key))
                del all_params[key]

        self.append_model(LinesModel, 'add', all_params)
        
        # continuum model is automatically added
        kwargs = self.add_continuum_model(**kwargs)

        return self.clean_kwargs(kwargs, all_params)


################################################
#### CLASS Cm1InputParams ######################
################################################
class Cm1InputParams(InputParams):
    """Manage the input parameters for :py:class:`orb.fit.FitVector`
    and :py:meth:`orb.fit.fit_lines_in_spectrum`.
    """
    def __init__(self, step, order, step_nb, nm_laser,
                 theta_proj, theta_orig, apodization, zpd_index,
                 filter_name):

        """

        .. note:: A distinction is made between the incident angle of
          projection and the real incident angle because the incident
          angle of projection is use to define the projection axis and
          thus the channel of a given wavenumber while the original
          angle is used to define the theoretical fwhm (which is not
          modified during the projection).
        """
        
        self.params = list()
        self.models = list()

        self.base_params = Params()
        self.base_params['step_nb'] = int(step_nb)
        self.base_params['step'] = float(step)
        self.base_params['order'] = int(order)
        self.base_params['nm_laser'] = float(nm_laser)
        self.base_params['apodization'] = float(apodization)
        self.base_params['theta_proj'] = float(theta_proj)
        self.base_params['theta_orig'] = float(theta_orig)
        self.base_params['zpd_index'] = int(zpd_index)        

        self.base_params['axis_corr_proj'] = orb.utils.spectrum.theta2corr(theta_proj)
        self.base_params['axis_corr_orig'] = orb.utils.spectrum.theta2corr(theta_orig)

        self.base_params['nm_laser_obs'] = (self.base_params.nm_laser
                                            * self.base_params.axis_corr_proj)
        
        self.base_params['filter_name'] = str(filter_name)

        self.allparams = Params()
        self.allparams.update(self.base_params)
        
        self.axis_min = orb.cutils.get_cm1_axis_min(self.base_params.step_nb,
                                                self.base_params.step,
                                                self.base_params.order,
                                                corr=self.base_params.axis_corr_proj)
        self.axis_step = orb.cutils.get_cm1_axis_step(self.base_params.step_nb,
                                                   self.base_params.step,
                                                   corr=self.base_params.axis_corr_proj)
        self.axis_max = self.axis_min + (self.base_params.step_nb - 1) * self.axis_step

        self.axis = np.arange(self.base_params.step_nb) * self.axis_step + self.axis_min
        
        self.set_signal_range(self.axis_min, self.axis_max)
        
        self.filterfile = FilterFile(self.base_params.filter_name)

    def _get_sigma_cov_vel(self, fwhm_guess_cm1, lines_cm1):
        if self.base_params.apodization == 1.:
            sigma_cov_vel = [0] * np.size(lines_cm1) # km/s
        else:
            sigma_cov_vel = orb.utils.fit.sigma2vel(
                orb.utils.fft.apod2sigma(self.base_params.apodization,
                                     fwhm_guess_cm1.mean) / self.axis_step,
                lines_cm1, self.axis_step)
        return np.atleast_1d(sigma_cov_vel).astype(float)

        
    def add_lines_model(self, lines, **kwargs):

        # guess lines
        lines_cm1 = list()
        for iline in lines:
            if isinstance(iline, str):
                iline = Lines().get_line_cm1(iline)
            lines_cm1.append(iline)

        lines_cm1 = np.array(lines_cm1)

        # guess fwhm
        fwhm_guess_cm1 = orb.utils.spectrum.compute_line_fwhm(
            self.base_params.step_nb - self.base_params.zpd_index,
            self.base_params.step,
            self.base_params.order,
            apod_coeff=self.base_params.apodization,
            corr=self.base_params.axis_corr_orig,
            wavenumber=True)

        # guess sigma from apodization
        sigma_cov_vel = self._get_sigma_cov_vel(fwhm_guess_cm1, lines_cm1)

        sigma_guess = np.copy(sigma_cov_vel)

        # guess ratio
        ratio = (float(self.base_params.zpd_index)
                 / float(self.base_params.step_nb - self.base_params.zpd_index))

        
        line_nb = np.size(lines)
        default_params = {
            'line_nb':line_nb,
            'amp_def':['free'] * line_nb,
            'ratio':ratio,
            #'amp_cov':1., # never put a gvar here or the amplitude sdev is forced to a given value
            'fwhm_def':['fixed'] * line_nb,
            'fwhm_guess':[fwhm_guess_cm1] * line_nb,
            'pos_def':['free'] * line_nb,
            'pos_guess':lines_cm1,
            'fmodel':'sinc'}

        params = self._check_lines_params(kwargs, fwhm_guess_cm1, lines_cm1)
        
        default_params.update(params)

        all_params = Params()
        all_params.update(self.base_params)
        all_params.update(default_params)

        # remove bad keys in case
        for key in list(all_params.keys()):
            if key not in Cm1LinesModel.accepted_keys:
                logging.debug('key removed: {}'.format(key))
                del all_params[key]

        self.append_model(Cm1LinesModel, 'add', all_params)

        # continuum model is automatically added
        kwargs = self.add_continuum_model(**kwargs)

        return self.clean_kwargs(kwargs, all_params)

    def add_filter_model(self, **kwargs):

        if self.base_params.filter_name is None:
            raise orb.utils.err.FitInputError('filter_name is None')

        filter_function = self.filterfile.project(Axis(self.axis)).data
        

        default_params = Params()
        default_params['filter_function'] = filter_function
        default_params['shift_def'] = 'free'

        # load user params
        params = Params()
        params.update(kwargs)

        if 'filter_function' in params:
            raise orb.utils.err.FitInputError('filter function must be defined via the filter file path at the init of the class')

        default_params.update(params)

        all_params = Params()
        all_params.update(self.base_params)
        all_params.update(default_params)

        # remove bad keys in case
        for key in list(all_params.keys()):
            if key not in FilterModel.accepted_keys:
                del all_params[key]

        self.append_model(FilterModel, 'mult', all_params)

        self.clean_kwargs(kwargs, all_params)
        
    def check_signal_range(self):
        if self.has_model(FilterModel):
            filter_bandpass = self.filterfile.get_filter_bandpass_cm1()
            if (min(self.signal_range_cm1) > min(filter_bandpass)
                or max(self.signal_range_cm1) < max(filter_bandpass)):
                logging.warning('Filter model might be badly constrained with such a signal range')


    def set_signal_range(self, rmin, rmax):
        InputParams.set_signal_range(self, rmin, rmax)
        self.signal_range_cm1 = (rmin, rmax)


    def convert(self):
        raw = InputParams.convert(self)
        raw['signal_range_cm1'] = self.signal_range_cm1
        raw['axis_min'] = self.axis_min
        raw['axis_max'] = self.axis_max
        raw['axis_step'] = self.axis_step        
        raw['baseclass'] = self.__class__.__name__
        return raw


################################################
#### CLASS OutputParams ########################
################################################

class OutputParams(Params):

    def translate(self, inputparams, fitvector, nogvar=False):
        if isinstance(inputparams, InputParams):
            inputparams = inputparams.convert()
            
        inputparams['params'] = orb.utils.fit.pick2paramslist(inputparams['params'])
        
        all_inputparams = Params()

        for iparams in inputparams['params']:
            all_inputparams.update(iparams)
        all_inputparams.update(inputparams['base_params'])


        if isinstance(inputparams, Cm1InputParams) or inputparams['baseclass'] == 'Cm1InputParams':
            wavenumber = True
        elif isinstance(inputparams, InputParams)  or inputparams['baseclass'] == 'InputParams':
            wavenumber = None
        else:
            raise NotImplementedError()

        if not isinstance(fitvector, FitVector):
            raise Exception('fitvector must be an instance of FitVector')

        if 'fit_params_err_mcmc' in self:
            fit_params_err_key = 'fit_params_err_mcmc'
        else:
            fit_params_err_key = 'fit_params_err'

        ## create a formated version of the parameters:
        ## [N_LINES, (H, A, DX, FWHM, SIGMA, ALPHA)]

        line_nb = np.size(all_inputparams['pos_guess'])
        line_params = fitvector.models[0].get_p_val_as_array(self['fit_params'][0])

        if all_inputparams['fmodel'] in ['sincgauss', 'sincphased', 'sincgaussphased']:
            line_params[:,3] = np.abs(line_params[:,3])
        else:
            nan_col = np.empty(line_nb, dtype=float)
            nan_col.fill(np.nan)
            line_params = np.append(line_params.T, nan_col)
            line_params = line_params.reshape(
                line_params.shape[0]//line_nb, line_nb).T

        # evaluate continuum level at each position
        cont_params = self['fit_params'][1]

        if wavenumber is None:
            pos_pix = line_params[:,1]
        else:
            pos_pix = orb.utils.spectrum.fast_w2pix(
                line_params[:,1],
                inputparams['axis_min'],
                inputparams['axis_step'])

        cont_model = fitvector.models[1]
        cont_model.set_p_val(self['fit_params'][1])
        cont_level = cont_model.get_model(pos_pix)
        all_params = np.append(cont_level, line_params.T)
        line_params = all_params.reshape(
            (all_params.shape[0]//line_nb, line_nb)).T


        # compute vel err
        line_params_err = fitvector.models[0].get_p_val_as_array(
            self[fit_params_err_key][0])

        if all_inputparams.fmodel not in ['sincgauss', 'sincgaussphased', 'sincphased']:
            line_params_err = np.append(line_params_err.T, nan_col)
            line_params_err = line_params_err.reshape(
                line_params_err.shape[0]//line_nb, line_nb).T

        # evaluate error on continuum level at each position
        cont_params_err = self[fit_params_err_key][1]

        cont_params_err_max = dict()
        cont_params_err_min = dict()
        for key in cont_params:
            cont_params_err_max[key] = cont_params[key] + cont_params_err[key] / 2.
            cont_params_err_min[key] = cont_params[key] - cont_params_err[key] / 2.
        cont_model.set_p_val(cont_params_err_max)
        cont_level_max = cont_model.get_model(pos_pix)
        cont_model.set_p_val(cont_params_err_min)
        cont_level_min = cont_model.get_model(pos_pix)
        cont_level_err = gvar.fabs(cont_level_max - cont_level_min)
        all_params_err = np.append(cont_level_err, line_params_err.T)
        line_params_err = all_params_err.reshape(
            (all_params_err.shape[0]//line_nb, line_nb)).T


        # set 0 sigma to nan
        if all_inputparams.fmodel in ['sincgauss', 'sincgaussphased']:
            line_params[:,4][line_params[:,4] == 0.] = np.nan
            if fit_params_err_key in self:
                line_params_err[:,4][line_params_err[:,4] == 0.] = np.nan


        ## compute errors
        line_params = gvar.gvar(gvar.mean(line_params),
                                gvar.mean(line_params_err))

        if wavenumber is not None:
            # compute velocity
            pos_wave = line_params[:,2]
            velocity = orb.utils.spectrum.compute_radial_velocity(
                pos_wave, gvar.mean(all_inputparams.pos_guess),
                wavenumber=wavenumber)
            if not nogvar:
                self['velocity_gvar'] = velocity
            self['velocity'] = gvar.mean(velocity)
            self['velocity_err'] = gvar.sdev(velocity)

            # compute broadening
            sigma_total_kms = line_params[:,4]
            sigma_apod_kms = orb.utils.fit.sigma2vel(
                orb.utils.fft.apod2sigma(
                    all_inputparams.apodization, line_params[:,3]) / inputparams['axis_step'],
                pos_wave, inputparams['axis_step'])

            broadening = (gvar.fabs(sigma_total_kms**2
                                    - sigma_apod_kms**2))**0.5

            if not nogvar:
                self['broadening_gvar'] = broadening
            self['broadening'] = gvar.mean(broadening)
            self['broadening_err'] = gvar.sdev(broadening)

            # compute fwhm in Angstroms to get flux
            # If calibrated, amplitude unit must be in erg/cm2/s/A, then
            # fwhm/width units must be in Angströms
            if wavenumber:
                fwhm = orb.utils.spectrum.fwhm_cm12nm(
                    line_params[:,3], line_params[:,2]) * 10.
            else:
                fwhm = line_params[:,3] * 10.

            # compute sigma in Angstroms to get flux
            sigma = orb.utils.spectrum.fwhm_cm12nm(
                orb.utils.fit.vel2sigma(
                    line_params[:,4], line_params[:,2],
                    inputparams['axis_step']) * inputparams['axis_step'],
                line_params[:,2]) * 10.
        else:
            fwhm = line_params[:,3]
            sigma = line_params[:,4]


        ## compute flux
        if all_inputparams.fmodel in ['sincgauss', 'sincgaussphased']:
            flux = orb.utils.spectrum.sincgauss1d_flux(
                line_params[:,1], fwhm, sigma)
        elif all_inputparams.fmodel == 'gaussian':
            flux = orb.utils.spectrum.gaussian1d_flux(
                line_params[:,1],fwhm)
        elif all_inputparams.fmodel == 'sinc':
            flux = orb.utils.spectrum.sinc1d_flux(
                line_params[:,1], fwhm)
        elif all_inputparams.fmodel == 'sincphased':
            flux = orb.utils.spectrum.sinc1d_flux(
                line_params[:,1], fwhm)

        else:
            flux = None

        if flux is not None:
            if not nogvar:
                self['flux_gvar'] = flux
            self['flux'] = gvar.mean(flux)
            self['flux_err'] = gvar.sdev(flux)

        # compute SNR
        self['snr'] = gvar.mean(line_params[:,1]) / gvar.sdev(line_params[:,1])

        # store lines-params
        if not nogvar:
            self['lines_params_gvar'] = line_params
        self['lines_params'] = gvar.mean(line_params)
        self['lines_params_err'] = np.abs(gvar.sdev(line_params))
        self.update(all_inputparams)

        if nogvar:
            for ikey in self:
                if isinstance(self[ikey], np.ndarray):
                    if self[ikey].dtype == np.object:
                        self[ikey] = gvar.mean(self[ikey])
        return self
        
    def convert(self):
        """Convert class to a raw pickable format
        """
        raw = dict()
        for ipar in list(self.keys()):
            raw[ipar] = self[ipar]
        raw = orb.utils.fit.gvardict2pickdict(raw)
        return raw


    def get_axis(self):
        """Returns the axis of the fitted spectrum
        """
        return orb.core.Axis(orb.utils.spectrum.create_cm1_axis(
            self.step_nb, self.step, self.order, self.axis_corr_proj))
        
    def get_params(self):
        """Return basic parameters of the fitted spectrum
        """
        params = dict()
        keys = ('step', 'order', 'filter_name')
        for ikey in keys:
            params[ikey] = self[ikey]
        return params
    
    def get_spectrum(self):
        """Return fitted spectrum"""
        spectrum = orb.core.Cm1Vector1d(
            self.fitted_vector, axis=self.get_axis(), params=self.get_params())
        return spectrum
    
    def get_residual(self):
        """Return fit residual"""
        res = np.zeros_like(self.get_spectrum().data)
        res[np.min(self.signal_range):np.max(self.signal_range)] = self.residual
        spectrum = orb.core.Cm1Vector1d(res, axis=self.get_axis(), params=self.get_params())
        return spectrum

    def get_components(self):
        """Return the multiple components fitted (lines sharing the same velocity)"""
        
        ncomps = len(np.unique((self.velocity*1e5).astype(int))/1e5)
        linesnb = self.lines_params.shape[0]//ncomps
        cont = self.fitted_models['ContinuumModel']
        comps = list()
        for icomp in range(ncomps):
            axis, iflux = orb.fit.create_cm1_lines_model_raw(
                self.lines_params[icomp*linesnb:(icomp+1)*linesnb,2], 
                self.lines_params[icomp*linesnb:(icomp+1)*linesnb,1], 
                self.step, self.order, self.step_nb, self.axis_corr_proj, 
                self.zpd_index)
            ispec = orb.core.Cm1Vector1d(
                iflux + cont, axis=axis, params=self.get_params())
        
            comps.append(ispec)
        return comps
    
    def plot(self, *args, **kwargs):
        """Plot fitted spectrum. Convenient wrapper around pyplot.plot() function""" 
        self.get_spectrum().plot(*args, **kwargs)
        
    def plot_residual(self, *args, **kwargs):
        """Plot residual. Convenient wrapper around pyplot.plot() function""" 
        self.get_residual().plot(*args, **kwargs)
        
    def __repr__(self):
        """Called by repr() and print() to display the most important results of the fit."""
        lines_nm = 1e7/gvar.mean(self['pos_guess'])
        lines = list()
        for iline in lines_nm:
            line_name = orb.core.Lines().get_line_name(iline)
            if line_name is None:
                line_name = str(iline)
            lines.append(line_name)
        info = '=== Fit results ===\n'
        info += 'lines: {}, fmodel: {}\n'.format(lines, self['fmodel'])
        info += 'iterations: {}, fit time: {:.2e} s\n'.format(self['iter_nb'], self['fit_time'])
        info += 'number of free parameters: {}, BIC: {:.5e}, chi2: {:.2e}\n'.format(self['nparams'], self['BIC'], self['chi2'])
        
        info += 'Velocity (km/s): {} \n'.format(gvar.gvar(self['velocity'], self['velocity_err']))
        info += 'Flux: {}\n'.format(gvar.gvar(self['flux'], self['flux_err']))
        info += 'Broadening (km/s): {}\n'.format(gvar.gvar(self['broadening'], self['broadening_err']))
        info += 'SNR (km/s): {}\n'.format(self['snr'])
            
        return info
    
    __str__ = __repr__
    
        

################################################
#### Functions #################################
################################################


def _fit_lines_in_spectrum(spectrum, ip, fit_tol=1e-10,
                           compute_mcmc_error=False, max_iter=None, nogvar=False,
                           vector_err=None, force_positive_flux=True,
                           **kwargs):
    """raw function for spectrum fitting. Need the InputParams
    class to be defined before call.

    :param spectrum: The spectrum to fit (1d vector).

    :param ip: InputParams instance (can be created with
      fit._prepare_input_params())

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-10).

    :param compute_mcmc_error: (Optional) If True, uncertainty
      estimates are computed from a Markov chain Monte-Carlo
      algorithm. If the estimates can be better constrained, the
      fitting time is orders of magnitude longer (default False).

    :param max_iter: (Optional) Maximum number of iterations (default None)

    :param nogvar: (Optional) No gvar are returned. 

    :param force_positive_flux: (Optional) Force line fluxes and
      amplitudes to be positive.
        
    :param kwargs: (Optional) Model parameters that must be changed in
      the InputParams instance.

    """
    if isinstance(ip, InputParams):
        rawip = ip.convert()
    else: rawip = ip

    rawip['params'] = orb.utils.fit.pick2paramslist(rawip['params'])

    for iparams in rawip['params']:
        for key in iparams:
            if key in kwargs:
                if kwargs[key] is not None:
                    iparams[key] = kwargs[key]
                    logging.debug('last minute changed parameter {}: {}'.format(key, iparams[key]))

    logging.debug('fwhm guess: {}'.format(
        rawip['params'][0]['fwhm_guess']))

    rawip['params'] = orb.utils.fit.paramslist2pick(rawip['params'])
    fv = FitVector(spectrum,
                   rawip['models'], rawip['params'],
                   signal_range=rawip['signal_range'],
                   fit_tol=fit_tol,
                   max_iter=max_iter,
                   nogvar=nogvar,
                   vector_err=vector_err,
                   force_positive_flux=force_positive_flux)

    fit = fv.fit(compute_mcmc_error=compute_mcmc_error)


    if fit != []:
        fit = OutputParams(fit)
        return fit.translate(ip, fv, nogvar=nogvar)
    
    else: return []

def _prepare_input_params(step_nb, lines, step, order, nm_laser,
                          theta_proj, zpd_index, wavenumber=True,
                          filter_name=None,
                          theta_orig=None,
                          apodization=1., 
                          **kwargs):    
    """Prepare input parameters

    :param step_nb: Number of steps of the spectrum
    
    :param lines: Positions of the lines in nm/cm-1

    :param step: Step size in nm

    :param order: Folding order

    :param nm_laser: Calibration laser wavelength in nm.

    :param theta_proj: Projected incident angle of the spectrum in
      degrees. If the spectrum is not calibrated in wavenumber this
      angle is the incident angle of the spectrum. If the spectrum is
      wavenumber calibrated then theta_orig must be set to the real
      incident angle of the spectrum.

    :param zpd_index: Index of the ZPD in the interferogram.
    
    :param apodization: (Optional) Apodization level. Permit to separate the
      broadening due to the apodization and the real line broadening
      (see 'broadening' output parameter, default 1.).

    :param filter_name: (Optional) Filter file path (default
      None).

   :param theta_orig: (Optional) Real incident angle (in degrees) of
     the spectrum. Must be set if the spectrum has been calibrated
     i.e. prjetected on a new wavenumber axis (If None, the spectrum
     is considered to be uncalibrated and the original theta is set
     equal to the projected theta theta_proj) (default None).

    :param kwargs: (Optional) Fitting parameters of
      :py:class:`orb.fit.Cm1LinesInput` or
      :py:class:`orb.fit.FitVector`.
    """
    if wavenumber:
        inputparams = Cm1InputParams
    else:
        raise NotImplementedError()

    if theta_orig is None: theta_orig = theta_proj

    logging.debug("theta_orig {}, theta_proj: {}".format(theta_orig, theta_proj))
    ip = inputparams(step, order, step_nb,
                     nm_laser, theta_proj, theta_orig, apodization,
                     zpd_index, filter_name)

    kwargs = ip.add_lines_model(lines, **kwargs)

    if filter_name is not None:
        ip.add_filter_model(**kwargs)
        
    if 'signal_range' in kwargs:
        if kwargs['signal_range'] is not None:
            ip.set_signal_range(min(kwargs['signal_range']),
                                max(kwargs['signal_range']))
        kwargs.pop('signal_range')
    if len(kwargs) > 0:
        raise ValueError('some kwargs are unknown: {}. Please remove them.'.format(list(kwargs.keys())))
    return ip

def fit_lines_in_spectrum(spectrum, lines, step, order, nm_laser,
                          theta, zpd_index, wavenumber=True,
                          filter_name=None,
                          apodization=1.,
                          fit_tol=1e-10,
                          velocity_range=None,
                          compute_mcmc_error=False,
                          max_iter=None,
                          force_positive_flux=True,
                          **kwargs):
    
    """Fit lines in spectrum

    .. warning:: If spectrum is in wavenumber (option wavenumber set
      to True) input and output unit will be in cm-1. If spectrum is
      in wavelength (option wavenumber set to False) input and output
      unit will be in nm.

    :param spectrum: Spectrum to fit

    :param lines: Positions of the lines in nm/cm-1

    :param step: Step size in nm

    :param order: Folding order

    :param nm_laser: Calibration laser wavelength in nm.

    :param theta_proj: Projected incident angle of the spectrum in
      degrees.
      
    :param zpd_index: Index of the ZPD in the interferogram.
    
    :param apodization: (Optional) Apodization level. Permit to separate the
      broadening due to the apodization and the real line broadening
      (see 'broadening' output parameter, default 1.).

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-10).

    :param filter_name: (Optional) Filter file path (default
      None).

    :param velocity_range: (Optional) Range of velocity to check
      around the shift_guess value. If not None, a brute force
      algorithm is used to find the best velocity value. If more than
      one shift_guess is given (e.g. if lines are have different
      velocities, the mean velocity will be used as an initial
      velocity guess). The quality of this guess depends strongly on
      the spectrum noise. Try avoid using it with low a SNR spectrum.

    :param compute_mcmc_error: (Optional) If True, uncertainty
      estimates are computed from a Markov chain Monte-Carlo
      algorithm. If the estimates can be better constrained, the
      fitting time is orders of magnitude longer (default False).

    :param max_iter: (Optional) Maximum number of iterations (default None)

    :param kwargs: (Optional) Fitting parameters of
      :py:class:`orb.fit.Cm1LinesInput` or
      :py:class:`orb.fit.FitVector`.

    :return: a dictionary containing:

      * all the fit parameters [key: 'fit_params']
    
      * lines parameters [key: 'lines_params'] Lines parameters are
        given as an array of shape (lines_nb, 5). The order of the 5
        parameters for each lines is [height at the center of the
        line, amplitude, position, fwhm, sigma]. Position and FWHM are given
        in nm/cm-1 depending on the input unit (i.e. nm if wavenumber
        is False and cm-1 if wavenumber is True)
      
      * lines parameters errors [key: 'lines_params_err']

      * velocity [key: 'velocity'] Velocity of the lines in km/s

      * velocity error [key: 'velocity_err'] Error on the velocity of
        the lines in km/s

      * residual [key: 'residual']
      
      * chi-square [key: 'chi2']

      * reduced chi-square [key: 'rchi2']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont_params']

      * fitted spectrum [key: 'fitted_vector']

      * log(Gaussian Bayes Factor) [key: 'logGBF']
   
    """
    all_args = dict(locals()) # used in case fit is retried (must stay
                              # at the very beginning of the function
                              # ;)

    if velocity_range is not None:
        raise NotImplementedError()

                              
    ip = _prepare_input_params(spectrum.shape[0], lines, step, order, nm_laser,
                               theta, zpd_index, wavenumber=wavenumber,
                               filter_name=filter_name,
                               apodization=apodization,
                               **kwargs)

    fit = _fit_lines_in_spectrum(spectrum, ip,
                                 fit_tol=fit_tol,
                                 compute_mcmc_error=compute_mcmc_error,
                                 max_iter=max_iter,
                                 force_positive_flux=force_positive_flux)


    if fit != []:
        return fit
    
    elif ip.allparams.fmodel in ['sincgauss', 'sincgaussphased']:
        logging.warning('bad fit, fmodel replaced by a normal sinc')
        all_args['fmodel'] = 'sinc'
        return fit_lines_in_spectrum(**all_args)
        
    return []
    


def fit_lines_in_vector(vector, lines, fwhm_guess, fit_tol=1e-10,
    compute_mcmc_error=False, max_iter=None, **kwargs):
    
    """Fit lines in a vector

    Use this function only if little is known about the vector. A
    vector resulting from an interferogram FFT is assumed :
    i.e. regular axis, symmetrical line shape.

    .. warning:: All position units are in channels

    :param vector: vector to fit

    :param lines: Positions of the lines in channels

    :param fwhm_guess: Initial guess on the lines FWHM (in channels).

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-10).

    :param compute_mcmc_error: (Optional) If True, uncertainty
      estimates are computed from a Markov chain Monte-Carlo
      algorithm. If the estimates can be better constrained, the
      fitting time is orders of magnitude longer (default False).

    :param max_iter: (Optional) Maximum number of iterations (default None)


    :param kwargs: (Optional) Fitting parameters of
      :py:class:`orb.fit.LinesInput` or
      :py:class:`orb.fit.FitVector`.


    :return: a dictionary containing:

      * all the fit parameters [key: 'fit_params']
    
      * lines parameters [key: 'lines_params'] Lines parameters are
        given as an array of shape (lines_nb, 5). The order of the 5
        parameters for each lines is [height at the center of the
        line, amplitude, position, fwhm, sigma]. Postion and FWHM are given
        in channels.
      
      * lines parameters errors [key: 'lines_params_err']

      * residual [key: 'residual']
      
      * chi-square [key: 'chi2']

      * reduced chi-square [key: 'rchi2']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont_params']

      * fitted spectrum [key: 'fitted_vector']

      * log(Gaussian Bayes Factor) [key: 'logGBF']
   
    """
    ip = InputParams(vector.shape[0])

    ip.add_lines_model(lines, fwhm_guess, **kwargs)

    if 'signal_range' in kwargs:
        if kwargs['signal_range'] is not None:
            ip.set_signal_range(min(kwargs['signal_range']),
                                max(kwargs['signal_range']))
    
    fv = FitVector(vector,
                   ip.models, ip.params,
                   signal_range=ip.signal_range,
                   fit_tol=fit_tol,
                   max_iter=max_iter)
    
    fit = fv.fit(compute_mcmc_error=compute_mcmc_error)

    if fit != []:
        fit = OutputParams(fit)
        return fit.translate(ip.convert(), fv)

    return []


def create_cm1_lines_model_raw(lines_cm1, amp, step, order, step_nb, corr,
                               zpd_index, vel=0, sigma=0, alpha=0, fmodel='sinc'):
    """Return a simple emission-line spectrum model in cm-1 from raw
    parameters. For more physical parameters use
    create_cm1_lines_model().

    :param lines_cm1: lines in cm-1
    
    :param amp: Amplitude (must have the same size as lines)
    
    :param step: Step size

    :param order: Folding order

    :param step_nb: Total numer of steps

    :param corr: calibration coeff.

    :param zpd_index: ZPD index.
    
    :param vel: (Optional) Global velocity shift applied to all the
      lines (in km/s, default 0.)
    
    :param sigma: (Optional) Line broadening (in km/s, default 0.)

    :param alpha: (Optional) Phase coefficient of the lines (default
      0.)

    :param fmodel: (Optional) Lines model. Can be 'gaussian', 'sinc',
      'sincgauss', 'sincphased', 'sincgaussphased' (default sincgauss).

    """
    NM_LASER = 543.5 # can be anything
    
    fwhm_guess = orb.utils.spectrum.compute_line_fwhm(
        step_nb - zpd_index, step, order, corr, wavenumber=True)

    def get_defguess(param):
        if np.size(param) == 1:
            return np.arange(np.size(lines_cm1)).astype(str), list([param]) * np.size(lines_cm1)
        elif np.size(param) == np.size(lines_cm1):
            return np.arange(np.size(lines_cm1)).astype(str), param
        else: raise Exception('param size must be 1 or {} but is {}'.format(
                np.size(lines_cm1), np.size(param)))
        
    pos_def, pos_cov = get_defguess(vel)
    model_params = {
        'step_nb':step_nb,
        'step':step,
        'order':order,
        'nm_laser':NM_LASER,
        'nm_laser_obs':NM_LASER * corr,
        'line_nb':np.size(lines_cm1),
        'fwhm_def':['1'] * np.size(lines_cm1),
        'fwhm_guess':[fwhm_guess] * np.size(lines_cm1),
        'pos_guess':lines_cm1,
        'pos_cov':pos_cov,
        'pos_def':pos_def,
        'fmodel':fmodel,
        'amp_def':['free'] * np.size(lines_cm1),
        'ratio':zpd_index / (step_nb - zpd_index)
    }
    
    if fmodel in ['sincgauss', 'sincgaussphased']:
        sigma_def, sigma_cov = get_defguess(sigma)    
        sigma_params = {
            'sigma_def':sigma_def,
            'sigma_guess':np.zeros(np.size(lines_cm1)),
            'sigma_cov':sigma_cov}
        
        model_params.update(sigma_params)

    if fmodel in ['sincgaussphased',]:
        alpha_params = {
            'alpha_def':['1'] * np.size(lines_cm1),
            'alpha_guess':[alpha] * np.size(lines_cm1),
            'alpha_cov':[0.]}
        
        model_params.update(alpha_params)
    
        
    lines_model = Cm1LinesModel(model_params)
    
    p_free = dict(lines_model.p_free)
    for iline in range(np.size(lines_cm1)):
        p_free[lines_model._get_ikey('amp', iline)] = amp[iline]
        
    lines_model.set_p_free(p_free)
    spectrum = lines_model.get_model(np.arange(step_nb))

    spectrum = spectrum
    cm1_axis = orb.utils.spectrum.create_cm1_axis(
        spectrum.size, step, order, corr=corr)
    return cm1_axis, spectrum


def create_cm1_lines_model(lines_cm1, amp, step, order, resolution,
                           theta, vel=0., sigma=0., alpha=0.,
                           fmodel='sincgauss', ratio=0.25):
    """Return a simple emission-line spectrum model in cm-1

    :param lines: lines in cm-1
    
    :param amp: Amplitude (must have the same size as lines)
    
    :param step: Step size

    :param order: Folding order

    :param resolution: Resolution of the spectrum

    :param theta: Incident angle
    
    :param vel: (Optional) Global velocity shift applied to all the
      lines (in km/s, default 0.)
    
    :param sigma: (Optional) Line broadening (in km/s, default 0.)

    :param alpha: (Optional) Phase coefficient of the lines (default
      0.)

    :param fmodel: (Optional) Lines model. Can be 'gaussian', 'sinc',
      'sincgauss', 'sincphased', 'sincgaussphased' (default sincgauss).
    """

    if np.size(amp) != np.size(lines_cm1):
        raise Exception('The number of lines and the length of the amplitude vector must be the same')

    nm_laser = 543.5 # can be anything
    nm_laser_obs = nm_laser / np.cos(np.deg2rad(theta))
    
    step_nb = orb.utils.spectrum.compute_step_nb(resolution, step, order)
    
    total_step_nb = step_nb * (1. + ratio)
    zpd_index = total_step_nb - step_nb
    
    return create_cm1_lines_model_raw(
        lines_cm1, amp, step, order, total_step_nb, orb.utils.spectrum.theta2corr(theta),
        zpd_index, vel=vel, sigma=sigma, alpha=alpha, fmodel=fmodel)

def create_lines_model(lines, amp, fwhm, step_nb, line_shift=0.,
                       sigma=0., alpha=0., fmodel='sincgauss'):
    """Return a simple emission-line spectrum model with no physical units.

    :param lines: lines channels.
    
    :param amp: Amplitude (must have the same size as lines).
    
    :param fwhm: lines FWHM (in channels).
    
    :param step_nb: Number of steps of the spectrum.
    
    :param line_shift: (Optional) Global shift applied to all the
      lines (in channels, default 0.)
    
    :param sigma: (Optional) Sigma of the lines (in channels, default
      0.)

    :param alpha: (Optional) Phase coefficient of the lines (default
      0.)

    :param fmodel: (Optional) Lines model. Can be 'gaussian', 'sinc',
      'sincgauss', 'sincphased', 'sincgaussphased' (default sincgauss).
    """
    if np.size(amp) != np.size(lines):
        raise Exception('The number of lines and the length of the amplitude vector must be the same')

    lines_model = LinesModel(    
        {'line_nb':np.size(lines),
         'amp_def':'free',
         'fwhm_def':'1',
         'pos_guess':lines,
         'pos_cov':line_shift,
         'pos_def':'1',
         'fmodel':fmodel,
         'fwhm_guess':fwhm,
         'sigma_def':'1',
         'sigma_guess':sigma,
         'sigma_cov':0., # never more than 0.
         'alpha_def':'1',
         'alpha_guess':alpha,
         'alpha_cov':0.}) # never more than 0.
    p_free = dict(lines_model.p_free)
    for iline in range(np.size(lines)):
        p_free[lines_model._get_ikey('amp', iline)] = amp[iline]
    lines_model.set_p_free(p_free)
    spectrum = lines_model.get_model(np.arange(step_nb))
    #model, models = lines_model.get_model(np.arange(step_nb), return_models=True)
    return spectrum

