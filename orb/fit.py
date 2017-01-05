#!/usr/bin/python
# *-* coding: utf-8 *-*
# Author: Thomas Martin <thomas.martin.1@ulaval.ca>
# File: fit.py

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

"""Fit module of ORB.

Defines the general Fitting classes and the fitting models. 

Best accessed through fit_lines_in_*() functions (defined at the end
of the file)
"""

__author__ = "Thomas Martin"
__licence__ = "Thomas Martin (thomas.martin.1@ulaval.ca)"       


import numpy as np
import warnings
import math
import copy
import scipy.optimize
import scipy.interpolate
import time

import emcee

import utils.fft
import orb.data as od
import cutils
import constants
import utils.spectrum
import utils.fit


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
    p_init_list = None
    p_init_size_list = None

    max_fev = 5000
    fit_tol = None
    
    def __init__(self, vector, models, params, fit_tol=1e-10,
                 signal_range=None):
        """
        Init class.

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
        """
        
        if not isinstance(models, tuple):
            raise ValueError('models must be a tuple of (model, model_operation).')

        if not isinstance(params, tuple):
            raise ValueError('params must be a tuple of parameter dictionaries.')

        if len(models) != len(params):
            raise Exception('there must be exactly one parameter dictionary by model')

        self.retry_count = 0
        self.models = list()
        self.models_operation = list()
        self.p_init_list = list()
        self.p_init_size_list = list()
        for i in range(len(models)):
            # init each model
            self.models.append(models[i][0](params[i]))
            if models[i][1] in self.models_operations:
                self.models_operation.append(models[i][1])
            else: raise Exception('Model operation must be in {}'.format(
                self.models_operations))
            # guess nan values for each model
            self.models[-1].make_guess(vector)
            self.p_init_list.append(self.models[-1].get_p_free())
            self.p_init_size_list.append(self.p_init_list[-1].shape[0])

        self.vector = copy.copy(vector)
        self.fit_tol = fit_tol

         
        if signal_range is not None:
            if (np.nanmin(signal_range) >= 0 and
                np.nanmax(signal_range) < vector.shape[0]):
                self.signal_range = [int(np.min(signal_range)),
                                     int(np.max(signal_range))]
            
            else: raise Exception('Bad signal range: {}'.format(signal_range))
            

    def _all_p_list2vect(self, p_list):
        """Concatenate a list of free parameters into a 1d array ready
        to be passed to :py:meth:`scipy.optimize.leastsq`

        :param p_list: List of free parameters. This list is a list of
          tuple of parameters. Each tuple defining the free parameters
          for each model, i.e. : ([free_params_model1],
          [free_params_model2], ...)

        .. seealso:: :py:meth:`fit.FitVector._all_p_vect2list`
        """
        
        return np.concatenate([p for p in p_list])

    def _all_p_vect2list(self, p_vect):
        """Transform a 1d array of free parameters as returned by
        :py:meth:`scipy.optimize.leastsq` into a more comprehensive
        list of free parameters (see
        :py:meth:`fit.FitVector._all_p_list2vect`)

        :param p_vect: 1d array of free parameters.

        .. seealso:: :py:meth:`fit.FitVector._all_p_list2vect`
        """
        p_list = list()
        last_index = 0
        for i in range(len(self.p_init_size_list)):
            new_index = self.p_init_size_list[i] + last_index
            p_list.append(p_vect[last_index:new_index])
            last_index = int(new_index)
        return p_list

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
        step_nb = self.vector.shape[0]
        if x is None:
            x = np.arange(step_nb, dtype=float)
       
        model = None
        models = list()
        all_p_list = self._all_p_vect2list(all_p_free)
        for i in range(len(self.models)):
            model_list = self.models[i].get_model(
                x, all_p_list[i], return_models=return_models)
            if return_models:
                model_to_append, models_to_append = model_list
                models.append(models_to_append)
                
            else:
                model_to_append = model_list
            if self.models_operation[i] == 'add':
                if model is None:
                    model = model_to_append
                else:
                    model += model_to_append
            elif self.models_operation[i] == 'mult':
                if model is None:
                    model = model_to_append
                else:
                    model *= model_to_append
            else: raise Exception('Bad model operation. Model operation must be in {}'.format(self.models_operations))
            
        if return_models:
            return model, models
        else:
            return model
    
    def get_objective_function(self, all_p_free):
        """Return the objective function.

        Called by :py:meth:`scipy.optimize.leastsq`. This function
        computes the model with :py:meth:`fit.FitVector.get_model`
        based on a set of free parameters choosen by the fitting
        algorithm and substract the model to the fitted vector of real
        data.

        :param all_p_free: Vector of free parameters.
        """
        return (self.vector - self.get_model(all_p_free))[
            np.min(self.signal_range):np.max(self.signal_range)]

    def _get_model_onrange(self, x, *all_p_free):
        """Return the part of the model contained in the signal
        range.

        .. note:: This function has been defined only to be used with
          scipy.optimize.curve_fit.

        :param x: x vector on which model is computed

        :param *all_p_free: Vector of free parameters.
        """
        return self.get_model(all_p_free, x=x)[
            np.min(self.signal_range):np.max(self.signal_range)]

    def _get_vector_onrange(self):
        """Return the part of the vector contained in the signal
        range.

        .. note:: This function has been defined only to be used with
          scipy.optimize.curve_fit.
        """
        return self.vector[
            np.min(self.signal_range):np.max(self.signal_range)]

    def get_jacobian(self, all_p_free):
        """Return the Jacobian of the objective function.

        :param all_p_free: Vector of free parameters.    
        """
        raise Exception('Not implemented')

    def get_lnlikelihood(self, all_p_free, sigma):
        inv_sigma2 = 1.0 / (sigma**2.)
        return -0.5 * np.nansum(
            self.get_objective_function(all_p_free)**2. * inv_sigma2
            - np.log(inv_sigma2))

    def get_lnprior(self, all_p_free):
        return 0.

    def get_lnposterior_probability(self, all_p_free, sigma):
        lp = self.get_lnprior(all_p_free)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.get_lnlikelihood(all_p_free, sigma)        

    def _compute_mcmc_error(self, p, p_err, sigma):
        """Return Markov chain Monte Carlo error on the fit parameters.

        :param p: Fitted parameters
        :param p_err: Fitted uncertainty
        :param sigma: Noise on the spectrum
        
        .. warning:: This function has not been extensively tested.
        """     
        PERCENTILE = 16 # corresponds to a 1-sigma uncertainty
        NWALKERS_COEFF = 2 # number of walkers
        MCMC_RUN_NB = 500 # number of walker steps
        MCMC_RUN_THRESHOLD = 100 # Burn-in samples threshold
                
        # walkers definition which starts in a tiny gaussian ball around
        # the maximum likelihood found with a classic optimization
        ndim, nwalkers = np.size(p), np.size(p) * NWALKERS_COEFF
        pos = [p + p_err * np.random.randn(ndim) for i in range(nwalkers)]
        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                            self.get_lnposterior_probability,
                                            args=(sigma,))
            sampler.run_mcmc(pos, MCMC_RUN_NB)
            samples = sampler.chain[:, MCMC_RUN_THRESHOLD:, :].reshape((-1, ndim))
        
            ## import corner
            ## fig = corner.corner(samples, truths=p)
            ## fig.savefig("triangle.svg")
            ## fig.savefig("triangle.pdf")

            err_mcmc = np.array(
                zip(*np.percentile(samples, [PERCENTILE, 100-PERCENTILE],
                                   axis=0)))
            err_min_mcmc = err_mcmc[:,0] - p
            err_max_mcmc = err_mcmc[:,1] - p
            p_mcmc_err_list = self._all_p_vect2list(np.nanmean(
                (np.abs(err_min_mcmc),
                 np.abs(err_max_mcmc)), axis=0))

        except Exception, e:
            warnings.warn('An error has occured during MCMC uncertainty computation : {}'.format(e))
            err_mcmc = np.empty_like(p)
            err_mcmc.fill(np.nan)
            p_mcmc_err_list = self._all_p_vect2list(err_mcmc)
            
                        
        full_p_mcmc_err_list = list()
        for i in range(len(self.models)):
            # recompute p_val error from p_free error
            full_p_mcmc_err_list.append(self.models[i].get_p_val_err(
                p_mcmc_err_list[i]))

        
        ## import pylab as pl
        ## for i in range(ndim):
        ##     pl.hist(samples[:,i], bins=100)
        ##     pl.show()

        ## for iparam in range(ndim):
        ##     print samples[:, iparam]
        
        ## ## print sampler.chain.shape
        ## ## [pl.plot(sampler.chain[i,:,0]) for i in range(sampler.chain.shape[0])]
        ## ## pl.show()
        
        return np.abs(full_p_mcmc_err_list)
               

    def fit(self, use_jacobian=False, compute_mcmc_error=False, no_error=False):
        """Fit data vector.

        This is the central function of the class.

        :param use_jacobian: (Optional) if True the Jacobian is used during least
          square computation: less iteration, but possibly longer. Be
          careful when using it because its robustness (and
          usefullness) have not been demonstrated (default False).

        :param compute_mcmc_error: (Optional) Compute Markov chain
          Monte-Carlo error on the fit parameters (Uncertainty
          estimates might be slighly better constrained but computing
          time can be orders of magnitude longer) (default False).

        :param no_error: (Optional) If True, uncertainties are not
          computed (default False).
        """
        all_args = dict(locals()) # used in case fit is retried (must stay
                                  # at the very beginning of the
                                  # function ;)

        RETRY_MAX_NFEV_BYPARAM = 30
        RETRY_MAX_COUNTS_BYPARAM = 2
        RETRY_RANDOM_COEFF = 5e-2
        MCMC_RANDOM_COEFF = 1e-2
        
        start_time = time.time()
        p_init_vect = self._all_p_list2vect(self.p_init_list)

        if use_jacobian:
            Dfun = self.get_jacobian
        else:
            Dfun = None

        try:
            fit = scipy.optimize.curve_fit(
                self._get_model_onrange,
                np.arange(self.vector.shape[0]),
                self._get_vector_onrange(),
                p0=p_init_vect,
                sigma=None,
                method='lm',
                jac=Dfun,
                maxfev=self.max_fev,
                full_output=True,
                xtol=self.fit_tol)
        except RuntimeError:
            fit = [5]
            
        if fit[-1] <= 4:
            if fit[2]['nfev'] >= self.max_fev:
                return [] # reject maxfev bounded fit
            
            returned_data = dict()
            returned_data['iter-nb'] = fit[2]['nfev']

            ## get fit model
            (returned_data['fitted-vector'],
             returned_data['fitted-models']) = self.get_model(
                fit[0],
                return_models=True)
            
            ## return fitted parameters of each models
            full_p_list = list()
            p_fit_list = self._all_p_vect2list(fit[0])
            for i in range(len(self.models)):
                # recompute p_val from new p_free
                self.models[i].set_p_free(p_fit_list[i]) 
                full_p_list.append(self.models[i].get_p_val())
                
            returned_data['fit-params'] = full_p_list
            
            ## compute error on parameters
            # compute reduced chi square
            last_diff = fit[2]['fvec']
            chisq = np.sum(last_diff**2.)
            red_chisq = chisq / (np.size(self.vector) - np.size(fit[0]))
            returned_data['reduced-chi-square'] = red_chisq
            returned_data['chi-square'] = chisq
            returned_data['residual'] = last_diff

            if not no_error:
                # compute least square fit errors
                cov_x = fit[1]
                if np.all(np.isfinite(cov_x)):
                    self.retry_count = 0
                    cov_diag = np.sqrt(np.diag(cov_x))
                    p_fit_err_list = self._all_p_vect2list(cov_diag)

                    full_p_err_list = list()
                    for i in range(len(self.models)):
                        # recompute p_val error from p_free error
                        full_p_err_list.append(self.models[i].get_p_val_err(
                            p_fit_err_list[i]))

                    returned_data['fit-params-err'] = np.abs(full_p_err_list)

                    # compute MCMC uncertainty estimates
                    if compute_mcmc_error:
                        sigma = np.nanstd(last_diff)
                        returned_data['fit-params-err-mcmc'] = self._compute_mcmc_error(
                            fit[0], cov_diag, sigma)


                # if no covariance matrix can be obtained, maybe fit is
                # just too good and has converged too fast on one or more
                # parameter.
                elif fit[2]['nfev'] < RETRY_MAX_NFEV_BYPARAM * np.size(fit[0]):
                    if self.retry_count < RETRY_MAX_COUNTS_BYPARAM * np.size(fit[0]):
                        # first retry the fit with initial value set to a
                        # small random variation around the fitted values.
                        self.retry_count += 1
                        self.p_init_list = self._all_p_vect2list(
                            fit[0]
                            + np.random.randn(np.size(fit[0]))
                            * RETRY_RANDOM_COEFF * fit[0])
                        del all_args['self']
                        return self.fit(**all_args)
                    else:
                        # compute uncertainty via a Markov chain Monte
                        # Carlo algorithm
                        sigma = np.nanstd(last_diff)
                        returned_data['fit-params-err-mcmc'] = (
                            self._compute_mcmc_error(
                                fit[0], fit[0] * MCMC_RANDOM_COEFF, sigma))
                        returned_data['fit-params-err'] = returned_data['fit-params-err-mcmc']


                returned_data['fit-time'] = time.time() - start_time
            
        else:
            return []

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
    p_free = None
    """Up to date value of the free parameters. Its size is always
    less or equal to the size of the full set of parameters used
    directly to compute the model. It reflects the real number of
    fitted parameters. For each group of covarying parameters one free
    parameter is added. """

    p_fixed = None
    """Array of fixed parameters. Each covarying parameter is stored
    as fixed. And one free parameter is added for each group of
    covarying parameters."""
    
    p_dict = None
    """Input dictionary defining the parameters. Contains the initial
    values of the parameters"""
    
    p_def = None
    """Definition the full set of parameters (fixed, free or
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
            self.parse_dict()
        else: raise ValueError('p must be a dict')

        # check input parameters
        self.check_input()
        
        # create free and fixed vectors
        self.val2free()
        

    def parse_dict(self):
        """Parse input dictionary to create :py:attr:`fit.Model.p_def`, :py:attr:`fit.Model.p_val` and
        :py:attr:`fit.Model.p_cov`"""
        raise Exception('Not implemented')

    def check_input(self):
        """Check input parameters"""
        raise Exception('Not implemented')

    def make_guess(self, v):
        """If a parameter value at init is a NaN this value is guessed.

        :param v: Data vector from which the guess is made.
        """
        raise Exception('Not implemented')

    def get_model(self, x, return_models=False):
        """Compute a model M(x, p) for all passed x positions. p are
        the parameter values stored in :py:attr:`fit.Model.p_val`

        :param x: Positions where the model M(x, p) is computed.

        :param return_models: (Optional) If True return also
          individual models (default False)

        """
        raise Exception('Not implemented')

    def get_p_free(self):
        """Return the vector of free parameters :py:attr:`fit.Model.p_free`"""
        return copy.copy(self.p_free)

    def set_p_free(self, p_free):
        """Set the vector of free parameters :py:attr:`fit.Model.p_free`

        :param p_free: New vector of free parameters
        """
        if self.p_free.shape == p_free.shape:
            self.p_free = copy.copy(p_free)
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
        if p_val.shape == self.p_val.shape:
            self.p_val = copy.copy(p_val)
            self.val2free()
        else: raise Exception('bad format of passed val parameters')

    def get_p_val_err(self, p_err):
        """Return the uncertainty of a full set of parameters given
        the uncertainty on the free parameters.

        :param p_err: Uncertainty on the free parameters.
        """
        # copy real p_free and p_fixed values
        old_p_fixed = copy.copy(self.p_fixed)
        old_p_free = copy.copy(self.p_free)

        # set p_fixed to 0 and replace p_free with p_err
        self.p_fixed.fill(0.)
        self.set_p_free(p_err)

        # p_val_err is computed from the fake p_fixed, p_err set
        p_val_err = self.get_p_val()

        # class is reset to its original values
        self.p_fixed = copy.copy(old_p_fixed)
        self.set_p_free(old_p_free)
        
        return p_val_err
        
    def val2free(self):
        """Recompute the set of free parameters
        :py:attr:`fit.Model.p_free` with the updated values of
        :py:attr:`fit.Model.p_val`"""
        
        if self.p_val is None or self.p_def is None or self.p_cov is None:
            raise Exception('class has not been well initialized: p_val, p_def and p_cov must be defined')

        self.p_free = list()
        self.p_fixed = list()
        passed_cov = list()
        for i in range(np.size(self.p_def)):
            if self.p_def[i] == 'free':
                self.p_free += list([self.p_val[i]])
            elif self.p_def[i] == 'fixed':
                self.p_fixed += list([self.p_val[i]])
            else:
                if self.p_def[i] not in passed_cov:
                    self.p_free += list([self.p_cov[self.p_def[i]][0]])
                self.p_fixed += list([self.p_val[i]])
                passed_cov += list([self.p_def[i]])
        self.p_free = np.array(self.p_free, dtype=float)
        self.p_fixed = np.array(self.p_fixed, dtype=float)
        
    def free2val(self):
        """Read the array of parameters definition
        :py:attr:`fit.Model.p_def` and update the parameter values
        based on the new set of free parameters
        :py:attr:`fit.Model.p_free`.
        """
        if self.p_free is None or self.p_fixed is None or self.p_def is None or self.p_cov is None:
            raise Exception('class has not been well initialized, p_free, p_fixed, p_def and p_cov must be defined')
        passed_cov = dict()
        free_index = 0
        fixed_index = 0
        if np.size(self.p_free) > 0:
            self.p_val = np.zeros(np.array(self.p_val).shape)
        else:
            self.p_val = np.zeros(np.array(self.p_val).shape)
            
        for i in range(np.size(self.p_def)):
            if self.p_def[i] == 'free':
                self.p_val[i] = self.p_free[free_index]
                free_index += 1
            elif self.p_def[i] == 'fixed':
                self.p_val[i] = self.p_fixed[fixed_index]
                fixed_index += 1
            else: # covarying parameter
                if self.p_def[i] not in passed_cov: # if not already
                                                    # taken into
                                                    # account
                    passed_cov[self.p_def[i]] = self.p_free[free_index]
                    free_index += 1

                # covarying operation
                self.p_val[i] = (
                    self.p_cov[self.p_def[i]][1](
                        self.p_fixed[fixed_index], passed_cov[self.p_def[i]]))
                fixed_index += 1

                
class FilterModel(Model):
    """
    Simple model of filter based on a real filter shape. The only
    possible free parameter is a wavelength/wavenumber shift.

    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'filter-function':,
       'shift-guess':,
       'shift-def':}

    :keyword filter-function: Transmission of the filter over the
      fitted spectral range (axis must be exactly the same).
      
    :keyword shift-guess: Guess on the filter shift in pixels.
    
    :keyword shift-def: Definition of the shift parameter, can be
      'free' or 'fixed'

    .. note:: This model must be multiplied with the other and used
      last.
    """
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        if 'filter-function' in self.p_dict:
            self.filter_function = self.p_dict['filter-function']
            self.filter_function[np.isnan(self.filter_function)] = 0
            self.filter_axis = np.arange(self.filter_function.shape[0])
            self.filter_function = scipy.interpolate.UnivariateSpline(
                self.filter_axis, self.filter_function,
                k=1, s=0, ext=1)
        else: 
            raise Exception('filter-function must be given')
        if 'shift-guess' in self.p_dict:
            self.p_val = np.array(self.p_dict['shift-guess'], dtype=float)
        else: self.p_val = np.zeros(1, dtype=float)
        
        if 'shift-def' in self.p_dict:
            self.p_def = (self.p_dict['shift-def'],)
        else:
            self.p_def = ('free',)
        self.p_cov = dict()

    def check_input(self):
        pass

    def make_guess(self, v):
        pass

    def get_model(self, x, p_free=None, return_models=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).
          
        :param return_models: (Optional) If True return also
          individual models (default False)
        """
        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = copy.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        if np.size(self.p_free) == 0:
            mod = copy.copy(self.filter_function(self.filter_axis))
        else:
            mod = copy.copy(self.filter_function(self.filter_axis + self.p_free[0]))
        if return_models:
            return mod, (mod)
        else:
            return mod


class ContinuumModel(Model):
    """
    Polynomial continuum model.

    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'poly-order':
       'poly-guess':}

    :keyword poly-order: Order of the polynomial to fit (be careful
      with high order polynomials).

    :keyword poly-guess: Initial guess on the coefficient values :
      must be a tuple of length poly-order + 1.

    .. note:: This model must be added to the others.
    """
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        if 'poly-order' in self.p_dict:
            self.poly_order = int(self.p_dict['poly-order'])
        else: self.poly_order = 0

        if 'poly-guess' in self.p_dict:
            if self.p_dict['poly-guess'] is not None:
                if np.size(self.p_dict['poly-guess']) == self.poly_order + 1:
                    self.p_val = copy.copy(self.p_dict['poly-guess'])
                else: raise Exception('poly-guess must be an array of size equal to poly-order + 1')
            else:
                self.p_val = np.zeros(self.poly_order + 1, dtype=float)
        else:
            self.p_val = np.zeros(self.poly_order + 1, dtype=float)
            
        self.p_def = list()
        for i in range(self.poly_order + 1):
            self.p_def.append('free')
        self.p_cov = dict() # no covarying parameters

    def check_input(self):
        pass

    def make_guess(self, v):
        pass

    def get_model(self, x, p_free=None, return_models=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).
          
        :param return_models: (Optional) If True return also
          individual models (default False)
        """
        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = copy.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        mod = np.polyval(list(self.get_p_val()), x)
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

    .. note:: Each line is built on 3 (or 4) parameters : amplitude,
      FWHM, position and sigma (only for a the sincgauss model -- see
      below for details on the different models).

      Some lines can have one or more covarying parameters: FWHM can
      be the same for all the lines (this is True if lines are not
      resolved), lines issued from the same ion can have the same
      speed (e.g. [NII] doublet, [SII] doublet, [OIII] doublet), and
      some fixed transition ratios between lines can also be set
      (e.g. [NII]6584/[NII]6548 can be set to 2.89, when [NII]6548 is
      likely to be really noisy).
    
    Input dictionary :py:attr:`fit.Model.p_dict`::

      {'line-nb':,
       'fmodel':,
       'amp-def':,
       'pos-def':,
       'fwhm-def':,
       'sigma-dev':, # only for sincgauss fmodel
       'amp-cov':,
       'pos-cov':,
       'fwhm-cov':,
       'sigma-cov':, # only for sincgauss fmodel
       'amp-guess':,
       'pos-guess':,
       'fwhm-guess':,
       'sigma-guess':} # only for sincgauss fmodel

    :keyword line-nb: Number of lines.

    :keyword fmodel: Line shape, can be 'gaussian', 'sinc', 'sinc2' or
      'sincgauss'.

    :keyword amp-def: Definition of the amplitude parameter, can be
      'free', 'fixed' or set to a label that defines its covarying
      group.

    :keyword pos-def: Definition of the position parameter in pixels,
      can be 'free', 'fixed' or set to a label that defines its
      covarying group.

    :keyword fwhm-def: Definition of the FWHM parameter in pixels, can
      be 'free', 'fixed' or set to a label that defines its covarying
      group.

    :keyword sigma-def: Definition of the sigma parameter in pixels,
      can be 'free', 'fixed' or set to a label that defines its
      covarying group.

    :keyword amp-cov: Guess on the covariant value of the amplitude
      (best set to 0 in general). There must be as many values as
      covarying amplitude groups or only one value if it is the same
      for all groups.

    :keyword pos-cov: Guess on the covariant value of the velocity (in
      pixels). There must be as many values as covarying amplitude
      groups or only one value if it is the same
      for all groups.

    :keyword fwhm-cov: Guess on the covariant value of the FWHM
      (best set to 0 in general). There must be as many values as
      covarying amplitude groups or only one value if it is the same
      for all groups.

    :keyword sigma-cov: Guess on the covariant value of sigma (best
       set to 0 in general). There must be as many values as covarying
       amplitude groups or only one value if it is the same for all
       groups.

    :keyword amp-guess: Initial guess on the amplitude value of the
       lines. Best set to a NaN in general (it can be automatically
       guessed with good robusteness). But if lines have a covarying
       amplitude the initial guess fixes their ratio.

    :keyword pos-guess: Initial guess on the position of the lines:
       the PRECISE rest frame position must be given here, especially
       if lines share a covarying position, because their relative
       position will be fixed.

    :keyword fwhm-guess: Initial guess on the FWHM of the lines. This
      guess must be the MOST PRECISE possible (to a few 10%), it is by
      far the most unstable parameter especially for sinc lines.
     
    :keyword sigma-guess: Initial guess on the value of sigma. Best
      set to 0. in general


    Example: A red spectrum containing [NII]6548, Halpha, [NII]6584,
    [SII]6716 and [SII]6731, with a mean velocity of 1500 km/s (which
    translates in a pixel shift of 5.5), with a fixed amplitude ratio
    netween [NII] lines, the same speed for lines issued from the same
    ions and a shared FWHM between everybody but Halpha would be
    defined this way::
    
      {'line-nb' : 5,
       'amp-def' : ('1', 'free', '1', 'free', 'free'),
       'pos-def' : ('1', '2', '1', '3', '3'), 
       'fwhm-def': ('1', '2', '1', '1', '1'),
       'amp-cov': 0.,
       'pos-cov': 5.5,
       'fwhm-cov': 0.,
       'amp-guess': (1., np.nan, 2.89, np.nan, np.nan), # here the amplitude ratio between covarying [NII] lines is fixed.
       'pos-guess': (40,60,80,120,140), # positions are given in pixel and are purely arbitrary in this example
       'fwhm-guess': 2.43}

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
    p_array = None
    """equivalent of :py:attr:`fit.Model.p_val` but presented as an
    array with each row corresponding to a line which is easier to
    handle."""

    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        def parse_param(key, cov_operation):
            
            key_guess = key + '-guess'
            key_def = key + '-def'
            key_cov = key + '-cov'
            ## parse guess
            if key_guess in self.p_dict:
                p_guess = np.array(self.p_dict[key_guess])
                
                if np.size(p_guess) != line_nb:
                    if np.size(p_guess) == 1:
                        value = float(p_guess)
                        p_guess = np.empty(line_nb, dtype=float)
                        p_guess.fill(value)
                    else:
                        raise Exception("{} must have the same size as the number of lines or it must be a float".format(key_guess))
                    
            else:
                p_guess = np.empty(line_nb, dtype=float)
                p_guess.fill(np.nan)

            if np.size(p_guess) == 1:
                p_guess = [p_guess]

            ## parse cov  
            if key_cov in self.p_dict:
                p_cov = np.array(self.p_dict[key_cov], dtype=float)
            else:
                p_cov = ()
                
            ## parse def
            p_cov_dict = dict()
            if key_def in self.p_dict: # gives the definition of the parameter
                p_def = np.array(self.p_dict[key_def], dtype=object)
                if np.size(p_def) != line_nb:
                    if np.size(p_def) == 1:
                        value = str(p_def)
                        p_def = np.empty(line_nb, dtype=object)
                        p_def.fill(value)
                    else:
                        raise Exception("{} must have the same size as the number of lines or it must be a float".format(key_def))

                p_def = p_def.astype('|S10')

                if np.size(p_def) == 1:
                    p_def = [p_def]
                                    
                # manage cov values
                cov_index = 0
                for i in range(line_nb):
                    if p_def[i] != 'free' and p_def[i] != 'fixed':
                        # create singular symbol
                        cov_symbol = str(key_def + str(p_def[i]))
                        p_def[i] = cov_symbol
                        
                        # fill cov dict
                        if cov_symbol not in p_cov_dict:
                            if np.size(p_cov) == 0:
                                cov_value = 0
                            elif np.size(p_cov) == 1:
                                cov_value = p_cov
                            else:
                                if cov_index < np.size(p_cov):
                                    cov_value = p_cov[cov_index]
                                else:
                                    raise Exception("{} must have the same size as the number of covarying parameters or it must be a float".format(key_cov))
                                cov_index += 1
                                
                            p_cov_dict[cov_symbol] = (
                                cov_value, cov_operation)
            else:
                p_def = np.empty(line_nb, dtype='|S10')
                p_def.fill('free')
                

            self.p_def += list(p_def)
            self.p_val += list(p_guess)
            self.p_cov.update(p_cov_dict)

        line_nb = self._get_line_nb()
        
        self.p_def = list()
        self.p_val = list()
        self.p_cov = dict()
        parse_param('amp', self._get_amp_cov_operation())
        parse_param('pos', self._get_pos_cov_operation())
        parse_param('fwhm', self._get_fwhm_cov_operation())
        if self._get_fmodel() == 'sincgauss':
            parse_param('sigma', self._get_sigma_cov_operation())
        self.p_def = np.array(self.p_def, dtype=object)
        self.p_val = np.array(self.p_val, dtype=float)
        
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


    def check_input(self):
        """Check input parameters"""
        FWHM_INIT = 2. * constants.FWHM_SINC_COEFF

        p_array = self._p_val2array()
        
        # check pos
        if np.any(np.isnan(p_array[:,1])):
            warnings.warn('No initial position given')

        # check fwhm
        if np.any(np.isnan(p_array[:,2])):
            warnings.warn('No initial fwhm given')
            
        # check amp
        if np.any(np.isnan(p_array[:,0])):
            #warnings.warn('No initial amplitude given')
            pass

        # check sigma
        if self._get_fmodel() == 'sincgauss':
            if np.any(np.isnan(p_array[:,3])):
                warnings.warn('No initial sigma given')

    def make_guess(self, v):
        """If a parameter value at init is a NaN this value is guessed.

        :param v: Data vector from which the guess is made.
        """
        FWHM_INIT = 2. * constants.FWHM_SINC_COEFF
        FWHM_COEFF = 3.
        
        p_array = self._p_val2array()
        
        # check pos
        if np.any(np.isnan(p_array[:,1])):
            raise Exception('initial guess on lines position must be given, no automatic lines detection is implemented at this level.')

        # check fwhm
        if np.any(np.isnan(p_array[:,2])):
            p_array[np.isnan(p_array[:,2]), 2] = FWHM_INIT
            
        # check amp
        if np.any(np.isnan(p_array[:,0])):
            for iline in range(self._get_line_nb()):
                if np.isnan(p_array[iline,0]):
                    pos = p_array[iline,1]
                    fwhm = p_array[iline,2]
                    pos_min = pos - fwhm * FWHM_COEFF
                    pos_max = pos + fwhm * FWHM_COEFF + 1
                    if pos_min < 0: pos_min = 0
                    if pos_max >= np.size(v): pos_max = np.size(v) - 1
                    if pos_min < pos_max:
                        val_max = np.nanmax(v[
                            int(pos_min):int(pos_max)])
                    else: val_max = 0.
                    p_array[iline,0] = val_max
 

        # check sigma
        if self._get_fmodel() == 'sincgauss':
            if np.any(np.isnan(p_array[:,3])):
                p_array[np.isnan(p_array[:,3]), 3] = 1e-3

        self.p_val = self._p_array2val(p_array)
        self.val2free()

    def _p_val2array_raw(self, p_val):
        """Transform a vector like :py:attr:`fit.Model.p_val` to an
        array of lines parameters like :py:attr:`fit.LinesModel.p_array`
        """
        line_nb = self._get_line_nb()
        return copy.copy(p_val.reshape(
            (p_val.shape[0]/line_nb, line_nb)).T)

    def _p_array2val_raw(self, p_array):
        """Transform a vector like :py:attr:`fit.LinesModel.p_array`
        to an array of lines parameters like
        :py:attr:`fit.Model.p_val`.
        """
        return copy.copy(p_array.T.flatten())
        
    def _p_val2array(self):
        """Transform :py:attr:`fit.Model.p_val` to :py:attr:`fit.LinesModel.p_array`"""
        return self._p_val2array_raw(self.p_val)

    def _p_array2val(self, p_array):
        """Transform :py:attr:`fit.LinesModel.p_array` to :py:attr:`fit.Model.p_val`."""
        return self._p_array2val_raw(p_array)

    def _get_line_nb(self):
        """Return the number of lines"""
        if 'line-nb' in self.p_dict:
            return self.p_dict['line-nb']
        else:
            raise Exception("'line-nb' must be set")

    def _get_fmodel(self):
        """Return the line model"""
        if 'fmodel' in self.p_dict:
            return self.p_dict['fmodel']
        else:
            return 'gaussian'

    def get_model(self, x, p_free=None, return_models=False):
        """Return model M(x, p).

        :param x: Positions where the model M(x, p) is computed.

        :param p_free: (Optional) New values of the free parameters
          (default None).
          
        :param return_models: (Optional) If True return also
          individual models (default False)
        """
        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = copy.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        
        line_nb = self._get_line_nb()
        fmodel = self._get_fmodel()
        p_array = self._p_val2array()
        
        mod = None
        models = list()
        for iline in range(line_nb):
            
            if fmodel == 'sinc':
                line_mod = utils.spectrum.sinc1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2])
                
            elif fmodel == 'sincgauss':
                line_mod = utils.spectrum.sincgauss1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2],
                    p_array[iline, 3])
            elif fmodel == 'sinc2':
                line_mod =  np.sqrt(utils.spectrum.sinc1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2])**2.)
                
            elif fmodel == 'gaussian':
                line_mod = utils.spectrum.gaussian1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2])
            
            else:
                raise ValueError("fmodel must be set to 'sinc', 'gaussian', 'sincgauss' or 'sinc2'")
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
    def _w2pix(self, w):
        """Translate wavenumber to pixels"""
        return utils.spectrum.fast_w2pix(w, self.axis_min, self.axis_step)


    def _pix2w(self, pix):
        """Translate pixel to wavenumber"""
        return utils.spectrum.fast_pix2w(pix, self.axis_min, self.axis_step)

    def _get_pos_cov_operation(self):
        """Return covarying position operation for an input velocity in km/s"""
        return lambda lines, vel: lines * np.sqrt((1. - vel / constants.LIGHT_VEL_KMS)
                                                  / (1. + vel / constants.LIGHT_VEL_KMS))
         
    def _p_val2array(self):
        """Transform :py:attr:`fit.Model.p_val` to :py:attr:`fit.LinesModel.p_array`"""
        p_array = LinesModel._p_val2array(self)
        lines_cm1 = copy.copy(p_array[:,1])
        p_array[:,1] = self._w2pix(lines_cm1) # convert pos cm-1->pix
        p_array[:,2] /= self.axis_step # convert fwhm cm-1->pix
        if self._get_fmodel() == 'sincgauss':
            # convert sigma km/s->pix
            p_array[:,3] = utils.fit.vel2sigma(
                p_array[:,3], lines_cm1, self.axis_step)
        return p_array

    def _p_array2val(self, p_array):
        """Transform :py:attr:`fit.LinesModel.p_array` to :py:attr:`fit.Model.p_val`."""
        p_array[:,1] = self._pix2w(p_array[:,1]) # convert pos pix->cm-1
        p_array[:,2] *= self.axis_step # convert fwhm pix->cm-1
        if self._get_fmodel() == 'sincgauss':
            # convert sigma pix-> km/s
            p_array[:,3] = utils.fit.sigma2vel(
                p_array[:,3], p_array[:,1], self.axis_step)
        return LinesModel._p_array2val(self, p_array)
      

    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        LinesModel.parse_dict(self)

        if 'step-nb' not in self.p_dict:
            raise Exception('step-nb keyword must be set' )
        self.step_nb = float(self.p_dict['step-nb'])
        
        if 'step' not in self.p_dict:
            raise Exception('step keyword must be set' )
        self.step = float(self.p_dict['step'])
        
        if 'order' not in self.p_dict:
            raise Exception('order keyword must be set' )
        self.order = int(self.p_dict['order'])
        
        if 'nm-laser' not in self.p_dict:
            raise Exception('nm-laser keyword must be set' )
        self.nm_laser = float(self.p_dict['nm-laser'])
        
        if 'nm-laser-obs' not in self.p_dict:
            raise Exception('nm-laser-obs keyword must be set' )
        self.nm_laser_obs = float(self.p_dict['nm-laser-obs'])
        
        self.correction_coeff = self.nm_laser_obs / self.nm_laser
        
        self.axis_min = cutils.get_cm1_axis_min(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)
        self.axis_step = cutils.get_cm1_axis_step(
            self.step_nb, self.step, corr=self.correction_coeff)

    def get_p_val_err(self, p_err):
        """Return the uncertainty of a full set of parameters given
        the uncertainty on the free parameters.

        :param p_err: Uncertainty on the free parameters.
        """
        # copy real p_free and p_fixed values
        old_p_fixed = copy.copy(self.p_fixed)
        old_p_free = copy.copy(self.p_free)

        # set p_fixed to 0 and replace p_free with p_err
        p_array_raw = self._p_val2array_raw(self.get_p_val())
        lines_orig = copy.copy(p_array_raw[:,1])
        p_array_raw.fill(0.)
        p_array_raw[:,1] = lines_orig
        self.p_val = self._p_array2val_raw(p_array_raw)
        self.val2free() # set p_fixed from new p_val
        self.set_p_free(p_err) # set p_free to p_err
        p_array_err_raw = self._p_val2array_raw(self.get_p_val())
        p_array_err_raw[:,1] -= lines_orig
        p_err = self._p_array2val_raw(p_array_err_raw)
       
        # reset class to its original values
        self.p_fixed = copy.copy(old_p_fixed)
        self.set_p_free(old_p_free)
        
        return p_err


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
        return lambda lines, vel: lines * np.sqrt((1. + vel / constants.LIGHT_VEL_KMS)
                                                  / (1. - vel / constants.LIGHT_VEL_KMS))

    
    def parse_dict(self):
        """Parse input dictionary :py:attr:`fit.Model.p_dict`"""
        Cm1LinesModel.parse_dict(self)
        
        self.axis_min = cutils.get_nm_axis_min(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)
        self.axis_step = cutils.get_nm_axis_step(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)


def fit_lines_in_spectrum(spectrum, lines, step, order, nm_laser,
    nm_laser_obs, wavenumber=True, fwhm_guess=3.5, cont_guess=None,
    shift_guess=0., sigma_guess=0., fix_fwhm=False, cov_fwhm=True, cov_pos=True,
    fix_pos=False, cov_sigma=True, fit_tol=1e-10, poly_order=0,
    fmodel='gaussian', signal_range=None, filter_file_path=None,
    fix_filter=False, apodization=1., velocity_range=None,
    compute_mcmc_error=False, no_error=False):
    
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

    :param nm_laser_obs: Observed wavelength of the calibration laser.

    :param fwhm_guess: (Optional) Initial guess on the lines FWHM
      in nm/cm-1 (default 3.5).

    :param cont_guess: (Optional) Initial guess on the continuum
      (default None). Must be a tuple of poly_order + 1 values ordered
      with the highest orders first.

    :param shift_guess: (Optional) Initial guess on the global shift
      of the lines in km/s (default 0.).

    :param sigma_guess: (Optional) Initial guess on the line
      broadening in km/s (apodization broadening must not be taken
      into account) (default 0.).

    :param fix_fwhm: (Optional) If True, FWHM value is fixed to the
      initial guess (default False).

    :param cov_fwhm: (Optional) If True FWHM is considered to be the
      same for all lines and become a covarying parameter (default
      True).

    :param cov_pos: (Optional) If True the estimated relative
      positions of the lines (the lines parameter) are considered to
      be exact and only need to be shifted. Positions are thus
      covarying. Very useful but the initial estimation of the line
      relative positions must be very precise. This parameter can also
      be a tuple of the same length as the number of lines to
      distinguish the covarying lines. Covarying lines must share the
      same number. e.g. on 5 lines, [NII]6548, Halpha, [NII]6584,
      [SII]6717, [SII]6731, if each ion has a different velocity
      cov_pos can be : [0,1,0,2,2]. (default True).

    :param fix_pos: (Optional) Fix lines position (default False).

    :param cov_sigma: (Optional) If True the estimated enlargment of
      the lines (the lines parameter) is considered to the same for
      all the lines. Covarying lines must share the same
      number. e.g. on 5 lines, [NII]6548, Halpha, [NII]6584,
      [SII]6717, [SII]6731, if each ion has a different velocity
      dispersion cov_sigma can be : [0,1,0,2,2]. (default True).

    :param apodization: (Optional) Apodization level. Permit to separate the
      broadening due to the apodization and the real line broadening
      (see 'broadening' output parameter, default 1.).

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-10).

    :param poly_order: (Optional) Order of the polynomial used to fit
      continuum. Use high orders carefully (default 0).

    :param fmodel: (Optional) Fitting model. Can be 'gaussian', 
      'sinc', 'sincgauss' or 'sinc2' (default 'gaussian').

    :param signal_range: (Optional) A tuple (x_min, x_max) in nm/cm-1
      giving the lowest and highest wavelength/wavenumber containing
      signal.

    :param fix_filter: (Optional) If True filter position is fixed
      (default False).

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

    :param no_error: (Optional) If True, uncertainties are not
      computed (default False).

    :return: a dictionary containing:

      * all the fit parameters [key: 'fit-params']
    
      * lines parameters [key: 'lines-params'] Lines parameters are
        given as an array of shape (lines_nb, 4). The order of the 4
        parameters for each lines is [height at the center of the
        line, amplitude, position, fwhm]. Position and FWHM are given
        in nm/cm-1 depending on the input unit (i.e. nm if wavenumber
        is False and cm-1 if wavenumber is True)
      
      * lines parameters errors [key: 'lines-params-err']

      * velocity [key: 'velocity'] Velocity of the lines in km/s

      * velocity error [key: 'velocity-err'] Error on the velocity of
        the lines in km/s

      * residual [key: 'residual']
      
      * chi-square [key: 'chi-square']

      * reduced chi-square [key: 'reduced-chi-square']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont-params']

      * fitted spectrum [key: 'fitted-vector']
   
    """
    all_args = dict(locals()) # used in case fit is retried (must stay
                              # at the very beginning of the function
                              # ;)                     
    SIGMA_COV_VEL = 1e-2 # covariant sigma in km/s, must be > 0.
    
    correction_coeff = float(nm_laser_obs) / nm_laser

    if wavenumber:
        axis_min = cutils.get_cm1_axis_min(spectrum.shape[0], step, order,
                                           corr=correction_coeff)
        axis_step = cutils.get_cm1_axis_step(spectrum.shape[0], step, 
                                             corr=correction_coeff)
        linesmodel = Cm1LinesModel
  
    else:
        axis_min = cutils.get_nm_axis_min(spectrum.shape[0], step, order,
                                          corr=correction_coeff)
        axis_step = cutils.get_nm_axis_step(spectrum.shape[0], step, order,
                                            corr=correction_coeff)
        linesmodel = NmLinesModel

    if signal_range is not None:
        signal_range_pix = utils.spectrum.fast_w2pix(
            np.array(signal_range, dtype=float), axis_min, axis_step)
        minx = max(1, int(np.min(signal_range_pix)))
        maxx = min(spectrum.shape[0] - 1,
                   int(math.ceil(np.max(signal_range_pix))))
        
    else:
        signal_range_pix = [10, np.size(spectrum)-10]

    # brute force over the velocity range to find the best lines position
    if velocity_range is not None:
        # velocity step of one channel
        bf_step_kms = axis_step / lines[0] * constants.LIGHT_VEL_KMS
        bf_flux = list()
        mean_shift_guess = np.nanmean(shift_guess)
        bf_range = np.arange(mean_shift_guess - velocity_range,
                             mean_shift_guess + velocity_range,
                             bf_step_kms)
        if np.size(bf_range) > 1:
            # get the total flux in the lines channels for each velocity
            for ivel in bf_range:
                ilines_wav = utils.spectrum.line_shift(
                    ivel, lines, wavenumber=wavenumber)
                ilines_pix = utils.spectrum.fast_w2pix(
                    np.array(lines + ilines_wav, dtype=float),
                    axis_min, axis_step)
                bf_flux.append(np.nansum(spectrum[np.array(
                    np.round(ilines_pix), dtype=int)]))
            shift_guess = bf_range[np.nanargmax(bf_flux)]

    fwhm_def, pos_def, sigma_def = _translate_fit_inputs(
        fix_fwhm, cov_fwhm, fix_pos, cov_pos, cov_sigma)
        
    if apodization == 1.:
        sigma_cov_vel = SIGMA_COV_VEL # km/s
    else:
        sigma_cov_vel = utils.fit.sigma2vel(
            utils.fft.apod2sigma(apodization, fwhm_guess) / axis_step,
            lines, axis_step)
    sigma_cov_vel = np.sqrt(sigma_cov_vel**2. + sigma_guess**2.)
        #if sigma_cov_vel == 0.: sigma_cov_vel = SIGMA_COV_VEL
        
    ## import pylab as pl
    ## if wavenumber:
    ##     axis = utils.spectrum.create_cm1_axis(spectrum.shape[0], step, order, corr=nm_laser_obs/nm_laser)
    ## else:
    ##     axis = utils.spectrum.create_nm_axis(spectrum.shape[0], step, order, corr=nm_laser_obs/nm_laser)
    
    ## pl.plot(axis, spectrum)
    ## searched_lines = lines + utils.spectrum.line_shift(shift_guess, lines, wavenumber=wavenumber)
    ## [pl.axvline(x=iline) for iline in searched_lines]
    ## #[pl.axvline(x=iline, ls=':') for iline in signal_range]
    ## pl.show()
    ## quit()

    if filter_file_path is not None:
        filter_function = utils.filters.get_filter_function(
            filter_file_path, step, order, spectrum.shape[0],
            wavenumber=wavenumber,
            silent=True)[0]
        if wavenumber:
            filter_axis = create_cm1_axis(
                spectrum.shape[0], step, order, corr=1.)
            filter_axis_calib = create_cm1_axis(spectrum.shape[0], step, order,
                                                corr=correction_coeff)
        else:
            filter_axis = create_nm_axis(
                spectrum.shape[0], step, order, corr=1.)
            filter_axis_calib = create_nm_axis(spectrum.shape[0], step, order,
                                               corr=correction_coeff)
            
        filter_function = utils.vector.interpolate_axis(
            filter_function, filter_axis_calib, 1, old_axis=filter_axis)
        if fix_filter:
            filter_def = 'fixed'
        else:
            filter_def = 'free'
            
    else:
        filter_function = np.ones(spectrum.shape[0], dtype=float)
        filter_def = 'fixed'

    fs = FitVector(spectrum,
                   ((linesmodel, 'add'),
                    (ContinuumModel, 'add'),
                    (FilterModel, 'mult')),
                   ({'step-nb':spectrum.shape[0],
                     'step':step,
                     'order':order,
                     'nm-laser':nm_laser,
                     'nm-laser-obs':nm_laser_obs,
                     'line-nb':np.size(lines),
                     'amp-def':'free',
                     'fwhm-def':fwhm_def,
                     'pos-guess':lines,
                     'pos-cov':shift_guess,
                     'pos-def':pos_def,
                     'fmodel':fmodel,
                     'fwhm-guess':fwhm_guess,
                     'sigma-def':sigma_def,
                     'sigma-guess':0.,
                     'sigma-cov':sigma_cov_vel},
                    {'poly-order':poly_order,
                     'poly-guess':cont_guess},
                    {'filter-function':filter_function,
                     'shift-def':filter_def}),
                   fit_tol=fit_tol,
                   signal_range=signal_range_pix)
    
    fit = fs.fit(compute_mcmc_error=compute_mcmc_error,
                 no_error=no_error)
    
    if fit != []:
        
        return _translate_fit_results(
            fit, fs, lines, fmodel,
            compute_mcmc_error, wavenumber,
            axis_min=axis_min, axis_step=axis_step,
            apodization=apodization)
    
    else:
        if fmodel == 'sincgauss':
            all_args['fmodel'] = 'sinc'
            return fit_lines_in_spectrum(**all_args)
        
        return []
    


def fit_lines_in_vector(vector, lines, fwhm_guess=3.5,
    cont_guess=None, shift_guess=0., fix_fwhm=False, cov_fwhm=True,
    cov_pos=True, fix_pos=False, cov_sigma=True, fit_tol=1e-10, poly_order=0,
    fmodel='gaussian', signal_range=None, filter_file_path=None,
    fix_filter=False, compute_mcmc_error=False, no_error=False):
    
    """Fit lines in a vector

    Use this function only if little is known about the vector. A
    vector resulting from an interferogram FFT is assumed :
    i.e. regular axis, symmetrical line shape.

    .. warning:: All position units are in channels

    :param vector: vector to fit

    :param lines: Positions of the lines in channels

    :param fwhm_guess: (Optional) Initial guess on the lines FWHM
      in nm/cm-1 (default 3.5).

    :param cont_guess: (Optional) Initial guess on the continuum
      (default None). Must be a tuple of poly_order + 1 values ordered
      with the highest orders first.

    :param shift_guess: (Optional) Initial guess on the global shift
      of the lines in channels (default 0.).

    :param fix_fwhm: (Optional) If True, FWHM value is fixed to the
      initial guess (default False).

    :param cov_fwhm: (Optional) If True FWHM is considered to be the
      same for all lines and become a covarying parameter (default
      True).

    :param cov_pos: (Optional) If True the estimated relative
      positions of the lines (the lines parameter) are considered to
      be exact and only need to be shifted. Positions are thus
      covarying. Very useful but the initial estimation of the line
      relative positions must be very precise. This parameter can also
      be a tuple of the same length as the number of lines to
      distinguish the covarying lines. Covarying lines must share the
      same number. e.g. on 5 lines, [NII]6548, Halpha, [NII]6584,
      [SII]6717, [SII]6731, if each ion has a different velocity
      cov_pos can be : [0,1,0,2,2]. (default False).

    :param cov_sigma: (Optional) If True the estimated enlargment of
      the lines (the lines parameter) is considered to the same for
      all the lines. Covarying lines must share the same
      number. e.g. on 5 lines, [NII]6548, Halpha, [NII]6584,
      [SII]6717, [SII]6731, if each ion has a different velocity
      dispersion cov_sigma can be : [0,1,0,2,2]. (default True).

    :param fix_pos: (Optional) If True line position is fixed (default
      False).

    :param fit_tol: (Optional) Tolerance on the fit value (default
      1e-10).

    :param poly_order: (Optional) Order of the polynomial used to fit
      continuum. Use high orders carefully (default 0).

    :param fmodel: (Optional) Fitting model. Can be 'gaussian', 
      'sinc', 'sincgauss' or 'sinc2' (default 'gaussian').

    :param signal_range: (Optional) A tuple (x_min, x_max) in channels
      giving the lowest and highest wavelength/wavenumber containing
      signal.

    :param compute_mcmc_error: (Optional) If True, uncertainty
      estimates are computed from a Markov chain Monte-Carlo
      algorithm. If the estimates can be better constrained, the
      fitting time is orders of magnitude longer (default False).

    :param no_error: (Optional) If True, uncertainties are not
      computed (default False).


    :return: a dictionary containing:

      * all the fit parameters [key: 'fit-params']
    
      * lines parameters [key: 'lines-params'] Lines parameters are
        given as an array of shape (lines_nb, 4). The order of the 4
        parameters for each lines is [height at the center of the
        line, amplitude, position, fwhm]. Postion and FWHM are given
        in channels.
      
      * lines parameters errors [key: 'lines-params-err']

      * residual [key: 'residual']
      
      * chi-square [key: 'chi-square']

      * reduced chi-square [key: 'reduced-chi-square']

      * SNR [key: 'snr']

      * continuum parameters [key: 'cont-params']

      * fitted spectrum [key: 'fitted-vector']
   
    """
    if signal_range is not None:
        minx = max(1, int(np.min(signal_range)))
        maxx = min(vector.shape[0] - 1, int(math.ceil(np.max(signal_range))))
    else:
        minx = 0 ; maxx = vector.shape[0] - 1

    fwhm_def, pos_def, sigma_def = _translate_fit_inputs(
        fix_fwhm, cov_fwhm, fix_pos, cov_pos, cov_sigma)

    fs = FitVector(vector,
                   ((LinesModel, 'add'),
                    (ContinuumModel, 'add')),
                   ({'line-nb':np.size(lines),
                     'amp-def':'free',
                     'fwhm-def':fwhm_def,
                     'pos-guess':lines,
                     'pos-cov':shift_guess,
                     'pos-def':pos_def,
                     'fmodel':fmodel,
                     'fwhm-guess':fwhm_guess,
                     'sigma-def':sigma_def,
                     'sigma-guess':0.,
                     'sigma-cov':cov_sigma},
                    {'poly-order':poly_order,
                     'poly-guess':cont_guess}),
                   fit_tol=fit_tol,
                   signal_range=[minx, maxx])
    
    fit = fs.fit(compute_mcmc_error=compute_mcmc_error,
                 no_error=no_error)
    if fit != []:
        return _translate_fit_results(fit, fs, lines, fmodel,
                                      compute_mcmc_error,
                                      None)
    
    else:
        return []
    


def _translate_fit_inputs(fix_fwhm, cov_fwhm,
                          fix_pos, cov_pos,
                          cov_sigma):
    """Translate inputs of the fitting routines.

    Parameters have the same definition as in
    :py:meth:`fit_lines_in_vector` and
    :py:meth:`fit_lines_in_spectrum`.

    :return: fwhm_def, pos_def, sigma_def
    """


    if fix_fwhm: fwhm_def = 'fixed'
    elif cov_fwhm: fwhm_def = '1'
    else: fwhm_def = 'free'

    if not fix_pos:
        if np.size(cov_pos) > 1:
            pos_def = cov_pos
        else:
            if not cov_pos: pos_def = 'free'
            else: pos_def = '1'
    else: pos_def = 'fixed'

    if np.size(cov_sigma) > 1:
        sigma_def = cov_sigma
    else:
        if not cov_sigma: sigma_def = 'free'
        else: sigma_def = '1'

    return fwhm_def, pos_def, sigma_def




def _translate_fit_results(fit_results, fs, lines, fmodel,
                           compute_mcmc_error, wavenumber,
                           axis_min=None, axis_step=None,
                           apodization=None):
    """Translate raw fit results in readable results.

    :param fit_results: Output of FitVector.fit()

    :param wavenumber: True if unit is in cm-1, False if unit is in
      nm, None if unit is in channels.
    """
    units = [None, True, False]        

    if wavenumber not in units: raise Exception(
        'wavenumber must be in {}'.format(units))

    if wavenumber is not None:
        if axis_min is None or axis_step is None or apodization is None:
            raise Exception('optional parameters must all be set if wavenumber is not None')

    if compute_mcmc_error:
        fit_params_err_key = 'fit-params-err-mcmc'
    else:
        fit_params_err_key = 'fit-params-err'

    ## create a formated version of the parameters:
    ## [N_LINES, (H, A, DX, FWHM, SIGMA)]

    line_params = fit_results['fit-params'][0]
    line_nb = np.size(lines)
    line_params = line_params.reshape(
        (line_params.shape[0]/line_nb, line_nb)).T

    if fmodel == 'sincgauss':
        line_params[:,3] = np.abs(line_params[:,3])
    else:
        nan_col = np.empty(line_nb, dtype=float)
        nan_col.fill(np.nan)
        line_params = np.append(line_params.T, nan_col)
        line_params = line_params.reshape(
            line_params.shape[0]/line_nb, line_nb).T

    # evaluate continuum level at each position
    cont_params = fit_results['fit-params'][1]
    if wavenumber is None:
        pos_pix = line_params[:,1]
    else:
        pos_pix = utils.spectrum.fast_w2pix(
            line_params[:,1], axis_min, axis_step)

    cont_model = fs.models[1]
    cont_model.set_p_val(cont_params)
    cont_level = cont_model.get_model(pos_pix)
    all_params = np.append(cont_level, line_params.T)
    line_params = all_params.reshape(
        (all_params.shape[0]/line_nb, line_nb)).T


    if fit_params_err_key in fit_results:
        # compute vel err
        line_params_err = fit_results[fit_params_err_key][0]
        line_params_err = line_params_err.reshape(
            (line_params_err.shape[0]/line_nb, line_nb)).T

        if fmodel != 'sincgauss':
            line_params_err = np.append(line_params_err.T, nan_col)
            line_params_err = line_params_err.reshape(
                line_params_err.shape[0]/line_nb, line_nb).T

        # evaluate error on continuum level at each position
        cont_params_err = fit_results[fit_params_err_key][1]
        cont_model.set_p_val(cont_params + cont_params_err / 2.)
        cont_level_max = cont_model.get_model(pos_pix)
        cont_model.set_p_val(cont_params - cont_params_err / 2.)
        cont_level_min = cont_model.get_model(pos_pix)
        cont_level_err = np.abs(cont_level_max - cont_level_min)
        all_params_err = np.append(cont_level_err, line_params_err.T)
        line_params_err = all_params_err.reshape(
            (all_params_err.shape[0]/line_nb, line_nb)).T

    else:
        line_params_err = None

    # set 0 sigma to nan
    if fmodel == 'sincgauss':
        line_params[:,4][line_params[:,4] == 0.] = np.nan
        if fit_params_err_key in fit_results:
            line_params_err[:,4][line_params_err[:,4] == 0.] = np.nan


    ## compute errors
    line_params = od.array(line_params, line_params_err)

    if wavenumber is not None:
        # compute velocity
        pos_wave = line_params[:,2]

        velocity = utils.spectrum.compute_radial_velocity(
            pos_wave.astype(np.longdouble),
            od.array(lines, dtype=np.longdouble),
            wavenumber=wavenumber)

        fit_results['velocity'] = velocity.dat
        
        if line_params_err is not None:
            fit_results['velocity-err'] = np.abs(velocity.err)
        
        # compute broadening
        sigma_total_kms = line_params[:,4]
                                   
        sigma_apod_kms = od.array(utils.fit.sigma2vel(
            utils.fft.apod2sigma(
                apodization, line_params[:,3]) / axis_step,
            pos_wave, axis_step))
        broadening = od.sqrt(od.abs(sigma_total_kms**2
                                    - sigma_apod_kms**2))
       
        fit_results['broadening'] = broadening.dat
        if line_params_err is not None:
            fit_results['broadening-err'] = np.abs(broadening.err)

        # compute flux
        # If calibrated, amplitude unit must be in erg/cm2/s/A, then
        # fwhm/width units must be in Angströms
        if wavenumber:
            fwhm = utils.spectrum.fwhm_cm12nm(
                line_params[:,3], line_params[:,2]) * 10.
        else:
            fwhm = line_params[:,3] * 10.

        sigma = utils.spectrum.fwhm_cm12nm(
            utils.fit.vel2sigma(
                line_params[:,4], line_params[:,2],
                axis_step) * axis_step,
            line_params[:,2]) * 10.
    else:
        fwhm = line_params[:,3]
        sigma = line_params[:,4]


    ## compute flux
    if fmodel == 'sincgauss':
        flux = utils.spectrum.sincgauss1d_flux(
            line_params[:,1], fwhm, sigma)
    elif fmodel == 'gaussian':
        flux = utils.spectrum.gaussian1d_flux(
            line_params[:,1],fwhm)
    elif fmodel == 'sinc':
        flux = utils.spectrum.sinc1d_flux(
            line_params[:,1], fwhm)
    else:
        flux = None

    if flux is not None:
        fit_results['flux'] = flux.dat
        if line_params_err is not None:
            fit_results['flux-err'] = np.abs(flux.err)

    # compute SNR
    if line_params_err is not None:
        fit_results['snr'] = line_params.dat[:,1] / line_params.err[:,1]

    # store lines-params
    fit_results['lines-params'] = line_params.dat
    if line_params_err is not None:
        fit_results['lines-params-err'] = np.abs(line_params.err)

    return fit_results




def create_cm1_lines_model(lines_cm1, amp, step, order, resolution,
                           theta, vel=0., sigma=0.):
    """Return a simple emission-line spectrum model in cm-1

    :param lines: lines in cm-1
    
    :param amp: Amplitude (must have the same size as lines)
    
    :param step: Step size

    :param order: Folding order

    :param resolution: Resolution of the spectrum

    :param theta: Incident angle

    :param step_nb: Number of steps of the spectrum.
    
    :param vel: (Optional) Global velocity shift applied to all the
      lines (in km/s, default 0.)
    
    :param sigma: (Optional) Line broadening (in km/s, default 0.)
    """


    if np.size(amp) != np.size(lines_cm1):
        raise Exception('The number of lines and the length of the amplitude vector must be the same')

    nm_laser = 543.5 # can be anything
    nm_laser_obs = nm_laser / np.cos(np.deg2rad(theta))
    
    step_nb = utils.spectrum.compute_step_nb(resolution, step, order)
    fwhm_guess = utils.spectrum.compute_line_fwhm(
        step_nb, step, order, nm_laser_obs / nm_laser, wavenumber=True)

    lines_model = Cm1LinesModel(    
        {'step-nb':step_nb,
         'step':step,
         'order':order,
         'nm-laser':nm_laser,
         'nm-laser-obs':nm_laser_obs,
         'line-nb':np.size(lines_cm1),
         'amp-def':'free',
         'fwhm-def':'1',
         'pos-guess':lines_cm1,
         'pos-cov':vel,
         'pos-def':'1',
         'fmodel':'sincgauss',
         'fwhm-guess':fwhm_guess,
         'sigma-def':'1',
         'sigma-guess':sigma,
         'sigma-cov':0.})
    p_free = np.copy(lines_model.p_free)
    p_free[:np.size(lines_cm1)] = amp
    lines_model.set_p_free(p_free)
    spectrum = lines_model.get_model(np.arange(step_nb))
    #model, models = lines_model.get_model(np.arange(step_nb), return_models=True)
    return spectrum

def create_lines_model(lines, amp, fwhm, step_nb, line_shift=0., sigma=0.):
    """Return a simple emission-line spectrum model with no physical units.

    :param lines: lines channels
    
    :param amp: Amplitude (must have the same size as lines)
    
    :param fwhm: lines FWHM (in channels)
    
    :param step_nb: Number of steps of the spectrum.
    
    :param line_shift: (Optional) Global shift applied to all the
      lines (in channels, default 0.)
    
    :param sigma: (Optional) Sigma of the lines (in channels, default
      0.)
    """

    if np.size(amp) != np.size(lines):
        raise Exception('The number of lines and the length of the amplitude vector must be the same')

    lines_model = LinesModel(    
        {'line-nb':np.size(lines),
         'amp-def':'free',
         'fwhm-def':'1',
         'pos-guess':lines,
         'pos-cov':line_shift,
         'pos-def':'1',
         'fmodel':'sincgauss',
         'fwhm-guess':fwhm,
         'sigma-def':'1',
         'sigma-guess':sigma,
         'sigma-cov':0.})
    p_free = np.copy(lines_model.p_free)
    p_free[:np.size(lines)] = amp
    lines_model.set_p_free(p_free)
    spectrum = lines_model.get_model(np.arange(step_nb))
    #model, models = lines_model.get_model(np.arange(step_nb), return_models=True)
    return spectrum




def check_fit_cm1(lines_cm1, amp, step, order, resolution, theta,
                  snr, sigma=0, vel=0):
    """Create a model and fit it.

    This is a good way to check the quality and the internal coherency
    of the fitting routine.

    :param lines_cm1: Lines rest wavenumber in cm-1

    :param amp: Amplitude of the lines

    :param step: Step size in nm

    :param oder: Folding order

    :param resolution: Resolution

    :param theta: Incident angle

    :param snr: SNR of the strongest line

    :param sigma: (Optional) Line broadening in km/s (default 0.)

    :param vel: (Optional) Velocity in km/s (default 0.)
    """
    
    spectrum = create_cm1_lines_model(lines_cm1, amp, step,
                                      order, resolution, theta,
                                      sigma=sigma, vel=vel)

    # add noise
    if snr > 0.:
        spectrum += np.random.standard_normal(
            spectrum.shape[0]) * np.nanmax(amp)/snr

    cos_theta =  np.cos(np.deg2rad(theta))
    step_nb = spectrum.shape[0]
    fwhm_guess = utils.spectrum.compute_line_fwhm(
        step_nb, step, order, 1 / cos_theta, wavenumber=True)

    fit = fit_lines_in_spectrum(
        spectrum, lines_cm1, step, order, 1., 1./cos_theta,
        wavenumber=True, 
        fmodel='sincgauss', shift_guess=vel, cov_pos=True,
        fwhm_guess=fwhm_guess, fix_fwhm=True, fix_pos=False,
        cov_sigma=True, compute_mcmc_error=False, no_error=True)

    return fit, spectrum


def check_fit(lines, amp, fwhm, step_nb, snr, line_shift=0, sigma=0):
    """Create a model and fit it.

    This is a good way to check the quality and the internal coherency
    of the fitting routine.

    :param lines: lines channels
    
    :param amp: Amplitude (must have the same size as lines)
    
    :param fwhm: lines FWHM (in channels)
    
    :param step_nb: Number of steps of the spectrum.

    :param snr: SNR of the strongest line
    
    :param line_shift: (Optional) Global shift applied to all the
      lines (in channels, default 0.)
    
    :param sigma: (Optional) Sigma of the lines (in channels, default
      0.)
    """
    
    spectrum = create_lines_model(lines, amp, fwhm, step_nb,
                                  line_shift=line_shift, sigma=sigma)

    # add noise
    if snr > 0.:
        spectrum += np.random.standard_normal(
            spectrum.shape[0]) * np.nanmax(amp)/snr


    fit = fit_lines_in_vector(
        spectrum, lines, 
        fmodel='sincgauss', shift_guess=line_shift, cov_pos=True,
        fwhm_guess=fwhm, fix_fwhm=True, fix_pos=False,
        cov_sigma=True, compute_mcmc_error=False, no_error=True)

    return fit, spectrum
