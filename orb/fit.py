import numpy as np
import warnings

import cutils
import constants
import scipy.optimize
import scipy.interpolate
import time
import utils.spectrum

class FitVector(object):
    """
    General fitting class for a 1d array of data based on
    Levenberg-Marquardt least square fit algorithm.
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

        self.vector = np.copy(vector)
        self.fit_tol = fit_tol

        if signal_range is not None:
            if (np.nanmin(signal_range) > 0 and
                np.nanmax(signal_range) < vector.shape[0]):
                self.signal_range = [int(np.min(signal_range)),
                                     int(np.max(signal_range))]
            
        else: raise Exception('Bad signal range')
            

    def _all_p_list2vect(self, p_list):
        return np.concatenate([p for p in p_list])

    def _all_p_vect2list(self, p_vect):
        p_list = list()
        last_index = 0
        for i in range(len(self.p_init_size_list)):
            new_index = self.p_init_size_list[i] + last_index
            p_list.append(p_vect[last_index:new_index])
            last_index = int(new_index)
        return p_list

    def get_model(self, all_p_free):
        """Return the combined model of the vector given a set of free parameters"""
        step_nb = self.vector.shape[0]
        model = np.zeros(step_nb)
        all_p_list = self._all_p_vect2list(all_p_free)
        for i in range(len(self.models)):
            model_to_append = self.models[i].get_model(
                np.arange(step_nb, dtype=float),
                all_p_list[i])
            if self.models_operation[i] == 'add':
                model += model_to_append
            elif self.models_operation[i] == 'mult':
                model *= model_to_append
            else: raise Exception('Bad model operation. Model operation must be in {}'.format(self.models_operations))
                    
        return model
    
    def get_objective_function(self, all_p_free):
        return (self.vector - self.get_model(all_p_free))[
            np.min(self.signal_range):np.max(self.signal_range)]


    def fit(self):
        start_time = time.time()
        p_init_vect = self._all_p_list2vect(self.p_init_list)
            

        fit = scipy.optimize.leastsq(self.get_objective_function,
                                     p_init_vect,
                                     maxfev=self.max_fev, full_output=True,
                                     xtol=self.fit_tol)

        if fit[-1] <= 4:
            if fit[2]['nfev'] >= self.max_fev:
                return [] # reject maxfev bounded fit
            
            returned_data = dict()
            returned_data['iter-nb'] = fit[2]['nfev']

            ## get fit model
            returned_data['fitted-vector'] = self.get_model(fit[0])
            
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
            returned_data['residual'] = last_diff #* noise_value
            
            # compute least square fit errors
            cov_x = fit[1]
            if cov_x is not None:
                cov_x *= returned_data['reduced-chi-square']
                cov_diag = np.sqrt(np.abs(
                    np.array([cov_x[i,i] for i in range(cov_x.shape[0])])))
                p_fit_err_list = self._all_p_vect2list(cov_diag)
                full_p_err_list = list()
                for i in range(len(self.models)):
                    # recompute p_val error from p_free error
                    full_p_err_list.append(self.models[i].get_p_val_err(
                        p_fit_err_list[i]))
                
                returned_data['fit-params-err'] = full_p_err_list
                
            returned_data['fit-time'] = time.time() - start_time
        else:
            return []

        return returned_data
        


class Model(object):
    """Template class for fit models.

    Method that must be implemented by real classes:

    * parse_dict()
    * check_input()
    * make_guess()
    * get_model()
    
    """
    p_free = None # vector style free parameters
    p_fixed = None # vector style fixed parameters
    
    p_dict = None # dict style parameters
    p_def = None # definition of all the parameters
    p_val = None # values of the parameters (initial guess before fit,
                 # fitted value after fit)
    p_cov = None # dict giving values of the covariing parameters

    def __init__(self, p):

        # parse input dict
        if isinstance(p, dict):
            self.p_dict = dict(p)
            self.parse_dict()
        else: raise ValueError('p must be a dict')

        # check input parameters
        self.check_input()
        
        # create free and fixed vectors
        self.val2free()
        

    def parse_dict(self):
        raise Exception('Not implemented')

    def check_input(self):
        raise Exception('Not implemented')

    def make_guess(self, v):
        raise Exception('Not implemented')

    def get_model(self, x):
        raise Exception('Not implemented')

    def get_p_free(self):
        return np.copy(self.p_free)

    def set_p_free(self, p_free):
        if self.p_free.shape == p_free.shape:
            self.p_free = np.copy(p_free)
            self.free2val()
        else: raise Exception('bad format of passed free parameters')

    def get_p_val(self):
        return np.copy(self.p_val)

    def set_p_val(self, p_val):
        if p_val.shape == self.p_val.shape:
            self.p_val = np.copy(p_val)
            self.val2free()
        else: raise Exception('bad format of passed val parameters')

    def get_p_val_err(self, p_err):
        # copy real p_free and p_fixed values
        old_p_fixed = np.copy(self.p_fixed)
        old_p_free = np.copy(self.p_free)

        # set p_fixed to 0 and replace p_free with p_err
        self.p_fixed.fill(0.)
        self.set_p_free(p_err)

        # p_val_err is computed from the fake p_fixed, p_err set
        p_val_err = self.get_p_val()

        # class is reset to its original values
        self.p_fixed = np.copy(old_p_fixed)
        self.set_p_free(old_p_free)
        
        return p_val_err
        
    def val2free(self):
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
        if self.p_free is None or self.p_fixed is None or self.p_def is None or self.p_cov is None:
            raise Exception('class has not been well initialized, p_free, p_fixed, p_def and p_cov must be defined')
        passed_cov = dict()
        free_index = 0
        fixed_index = 0
        for i in range(np.size(self.p_def)):
            if self.p_def[i] == 'free':
                self.p_val[i] = self.p_free[free_index]
                free_index += 1
            elif self.p_def[i] == 'fixed':
                self.p_val[i] = self.p_fixed[fixed_index]
                fixed_index += 1
            else:
                if self.p_def[i] not in passed_cov:
                    passed_cov[self.p_def[i]] = self.p_free[free_index]
                    free_index += 1

                self.p_val[i] = (
                    self.p_cov[self.p_def[i]][1](
                        self.p_fixed[fixed_index], passed_cov[self.p_def[i]]))
                fixed_index += 1
class FilterModel(Model):
    """

    Dict-style parameters:

    {'filter-function':
     'shift-guess':
     'shift-def':}

    shift_guess must be given in pixels 
     
    """
    def parse_dict(self):
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

    def get_model(self, x, p_free=None):
        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = np.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        
        mod = np.copy(self.filter_function(self.filter_axis + self.p_free[0]))
        return mod                


class ContinuumModel(Model):
    """

    Dict-style parameters:

    {'poly-order':
     'cont-guess':}
     
    """
    def parse_dict(self):
        if 'poly-order' in self.p_dict:
            self.poly_order = int(self.p_dict['poly-order'])
        else: self.poly_order = 0

        if 'cont-guess' in self.p_dict:
            if self.p_dict['cont-guess'] is not None:
                if np.size(self.p_dict['cont-guess']) == self.poly_order + 1:
                    self.p_val = np.copy(self.p_dict['cont-guess'])
                else: raise Exception('cont-guess must be an array of size equal to poly-order + 1')
            else:
                self.p_val = np.zeros(self.poly_order + 1, dtype=float)
        else:
            self.p_val = np.zeros(self.poly_order + 1, dtype=float)
            
        self.p_def = list()
        for i in range(self.poly_order + 1):
            self.p_def.append('free')
        self.p_cov = dict()

    def check_input(self):
        pass

    def make_guess(self, v):
        pass

    def get_model(self, x, p_free=None):
        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = np.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        mod = np.polyval(self.get_p_val(), x)
        return mod
        
        
class LinesModel(Model):
    """

    Dict-style parameters:

    {'line-nb':
     'amp-def':
     'pos-def':
     'fwhm-def':
     'sigma-dev': # only for sincgauss fmodel
     'amp-cov':
     'pos-cov':
     'fwhm-cov':
     'sigma-cov': # only for sincgauss fmodel
     'amp-guess':
     'pos-guess':
     'fwhm-guess':
     'sigma-guess': # only for sincgauss fmodel}
     
    """
    def parse_dict(self):
        
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
                # manage cov values
                cov_index = 0
                for i in range(line_nb): 
                    if p_def[i] != 'free' and p_def[i] != 'fixed':
                        # create singular symbol
                        cov_symbol = str(key_def + p_def[i])
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
                                    raise Exception("{} must have the same size as the number of covariant parameters or it must be a float".format(key_cov))
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
        return lambda x, y: x + y

    def _get_fwhm_cov_operation(self):
        return lambda x, y: x + y

    def _get_pos_cov_operation(self):
        return lambda x, y: x + y

    def _get_sigma_cov_operation(self):
        return lambda x, y: x + y


    def check_input(self):

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
                    val_max = np.nanmax(v[
                        pos - fwhm * FWHM_COEFF
                        :pos + fwhm * FWHM_COEFF + 1])
                    p_array[iline,0] = val_max

        # check sigma
        if self._get_fmodel() == 'sincgauss':
            if np.any(np.isnan(p_array[:,3])):
                p_array[np.isnan(p_array[:,3]), 3] = 0.

        self.p_val = self._p_array2val(p_array)
        self.val2free()

    def _p_val2array_raw(self, p_val):
        line_nb = self._get_line_nb()
        return np.copy(p_val.reshape(
            p_val.shape[0]/line_nb, line_nb).T)

    def _p_array2val_raw(self, p_array):
        return np.copy(p_array.T.flatten())
        
    def _p_val2array(self):
        return self._p_val2array_raw(self.p_val)

    def _p_array2val(self, p_array):
        return self._p_array2val_raw(p_array)

    def _get_line_nb(self):
        if 'line-nb' in self.p_dict:
            return self.p_dict['line-nb']
        else:
            raise Exception("'line-nb' must be set")

    def _get_fmodel(self):
        if 'fmodel' in self.p_dict:
            return self.p_dict['fmodel']
        else:
            return 'gaussian'

    def get_model(self, x, p_free=None):

        if p_free is not None:
            if np.size(p_free) == np.size(self.p_free):
                self.p_free = np.copy(p_free)
            else:
                raise Exception('p_free has not the right shape it must be: {}'.format(self.p_free.shape))
            
        self.free2val()
        
        line_nb = self._get_line_nb()
        fmodel = self._get_fmodel()
        p_array = self._p_val2array()

        mod = np.zeros_like(x, dtype=float)
    
        for iline in range(line_nb):
            
            if fmodel == 'sinc':
                mod += cutils.sinc1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2])
                
            elif fmodel == 'lorentzian':
                mod += cutils.lorentzian1d(
                    x, 0., p_array[iline, 0], p_array[iline, 1],
                    p_array[iline, 2])

            elif fmodel == 'sincgauss':
                mod += cutils.sincgauss1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2], p_array[iline, 3])
                
            ## elif fmodel == 'sinc2':
            ##     mod += np.sqrt(cutils.sincgauss1d(
            ##         x, 0., p_array[iline, 0], p_array[iline, 1],
            ##         p_array[iline, 2], cov_p[1])**2.)
                
            elif fmodel == 'gaussian':
                mod += cutils.gaussian1d(
                    x, 0., p_array[iline, 0],
                    p_array[iline, 1], p_array[iline, 2])
            else:
                raise ValueError("fmodel must be set to 'sinc', 'gaussian' or 'sinc2'")
        return mod
        


class Cm1LinesModel(LinesModel):
    
    def _w2pix(self, w):
        return utils.spectrum.fast_w2pix(w, self.axis_min, self.axis_step)

    def _pix2w(self, pix):
        return utils.spectrum.fast_pix2w(pix, self.axis_min, self.axis_step)

    def _get_pos_cov_operation(self):
        return lambda lines, vel: lines - lines * vel / constants.LIGHT_VEL_KMS

    def _p_val2array(self):
        p_array = LinesModel._p_val2array(self)
        p_array[:,1] = self._w2pix(p_array[:,1]) # convert pos cm-1->pix
        p_array[:,2] /= self.axis_step # convert fwhm cm-1->pix
        return p_array

    def _p_array2val(self, p_array):
        p_array[:,1] = self._pix2w(p_array[:,1]) # convert pos pix->cm-1
        p_array[:,2] *= self.axis_step # convert fwhm pix->cm-1
        return LinesModel._p_array2val(self, p_array)
      

    def parse_dict(self):
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
        
        self.axis_min = utils.spectrum.get_cm1_axis_min(
            self.step_nb, self.step, self.order,
            corr=self.correction_coeff)
        self.axis_step = utils.spectrum.get_cm1_axis_step(
            self.step_nb, self.step,
            corr=self.correction_coeff)

    def get_p_val_err(self, p_err):
        # copy real p_free and p_fixed values
        old_p_fixed = np.copy(self.p_fixed)
        old_p_free = np.copy(self.p_free)

        # set p_fixed to 0 and replace p_free with p_err
        p_array_raw = self._p_val2array_raw(self.get_p_val())
        lines_orig = np.copy(p_array_raw[:,1])
        p_array_raw.fill(0.)
        p_array_raw[:,1] = lines_orig
        self.p_val = self._p_array2val_raw(p_array_raw)
        self.val2free() # set p_fixed from new p_val
        self.set_p_free(p_err) # set p_free to p_err
        p_array_err_raw = self._p_val2array_raw(self.get_p_val())
        p_array_err_raw[:,1] -= lines_orig
        p_err = self._p_array2val_raw(p_array_err_raw)
       
        # reset class to its original values
        self.p_fixed = np.copy(old_p_fixed)
        self.set_p_free(old_p_free)
        
        return p_err

        
