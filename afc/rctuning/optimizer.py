# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced-order RC tuning algorithm.
"""

# pylint: disable=too-many-arguments, bare-except, unused-variable
# pylint: disable=too-many-positional-arguments, too-many-locals
# pylint: disable=redefined-outer-name, unnecessary-list-index-lookup
# pylint: disable=too-many-instance-attributes, too-many-nested-blocks
# pylint: disable=too-many-branches, dangerous-default-value
# pylint: disable=consider-using-dict-items

import os
import numpy as np
import pandas as pd
from doper.utility import standard_report
from doper.wrapper import DOPER
from .optModel import model
from .utility import convert_rc_parameter
from .defaultConfig import get_rc_parameter, INPUTS_DESCRIPTION

try:
    from pyDOE import lhs
    PYDOE_LOADED = True
except:
    # print('WARNING: PyDOE not installed.')
    PYDOE_LOADED = False

ROOT = os.path.dirname(os.path.realpath(__file__))

def optimize_param(inputs, parameter, printing=True, model=model):
    '''run actual optmizaiton using doper.'''

    # check if ipopt file exists
    # if not os.path.exists('ipopt.opt'):
    #     print('WARNING: The "ipopt.opt" file does not exist.')

    # do optmization
    rctuner = DOPER(model=model,
                    parameter=parameter,
                    solver_path=parameter['wrapper']['solver_path'],
                    output_list=parameter['wrapper']['output_list'])
    if printing:
        print(f'** Objective is Mean Squared Error; model is {parameter["type"]} **')
    # solver_options = {'print_level': int(5), # default = 5
    #                 #   'print_options_documentation': 'no',
    #                   'max_cpu_time': int(300),
    #                 #   'mumps_mem_percent':int(10000),
    #                  }
    res = rctuner.do_optimization(inputs,
                                  tee=printing,
                                  options=parameter['wrapper']['solver_options'])

    duration, objective, df, model, result, termination, parameter = res

    if printing:
        print(standard_report(res, only_solver=True))
        print('Objective:', model.objective(), '\n')

    return res

def do_tuning(inputs, parameter, printing=True, lhs_samples=5, seed=1,
              rc_tuning_model=model, rc_converter=convert_rc_parameter):
    '''handler for rc tuning.'''

    inputs = inputs.copy(deep=True)
    i = 0
    res_lhs = pd.DataFrame()

    # run optimzation
    res = optimize_param(inputs, parameter, printing, rc_tuning_model)
    duration, objective, df, model, result, termination, parameter = res

    # store outputs
    for k,v in rc_converter(model, print_new=False).items():
        res_lhs.loc[i, k] = v
    res_lhs.loc[i, 'duration'] = duration
    res_lhs.loc[i, 'objective'] = objective
    res_lhs.loc[i, 'termination'] = str(termination)
    i += 1

    # run multiple optimizations with lhs
    if lhs_samples:

        if not PYDOE_LOADED:
            print('ERROR: pyDOE not loaded; no lhs executed.')
        if lhs_samples > 1 and PYDOE_LOADED:
            pars = [k for k,v in parameter.items() if isinstance(v, dict) \
                    and not k in ['objective', 'wrapper']]
            np.random.seed(seed)
            lhs_param = lhs(len(pars), lhs_samples-1)

            for l in lhs_param:
                try:
                    for p, _ in enumerate(pars):
                        ns = (parameter[pars[p]]['ub'] - parameter[pars[p]]['lb']) * l[p] \
                            + parameter[pars[p]]['lb']
                        parameter[pars[p]]['init'] = ns
                    res = optimize_param(inputs, parameter, printing, rc_tuning_model)
                    duration, objective, df, model, result, termination, parameter = res
                    for k,v in rc_converter(model, print_new=False).items():
                        res_lhs.loc[i, k] = v
                except:
                    duration = -1
                    objective = 1e6
                    termination = 'error'

                res_lhs.loc[i, 'duration'] = duration
                res_lhs.loc[i, 'objective'] = objective
                res_lhs.loc[i, 'termination'] = str(termination)
                i += 1
            # Re-run best result
            ix = res_lhs['objective'][res_lhs['termination']=='optimal'].idxmin()
            for p, _ in enumerate(pars):
                parameter[pars[p]]['init'] = res_lhs.loc[ix, pars[p]]
            res = optimize_param(inputs, parameter, printing, rc_tuning_model)
            duration, objective, df, model, result, termination, parameter = res

    new_param = rc_converter(model, print_new=printing)
    return duration, objective, df, new_param, inputs, res_lhs, termination

class RcTuning:
    '''rc tuning class'''

    def __init__(self, horizon=2*24*60*60, rctype='R4C2', mode='sepSlab', has_windows=True,
                 lhs_samples=False, fix_c=False, dampen_param=False, first_free=True,
                 first_dampen=False, printing=False, model=model,
                 rc_converter=convert_rc_parameter, resample=None,
                 min_std_troom=None, min_std_tslab=None, rc_parameter_init={},
                 use_internal_params=True):
        '''
        horizon (int): Horizon of the tuning, in seconds.
        rctype (str): RC model type.
        mode (str): Mode of tuning.
        lhs_samples (int): Number of samples for lhs sampling.
        fix_c (bool): Flag to fix the capacitive parameters.
        dampen_param (float): Dampening of parameter estimation.
        first_free (bool): First estimation default bouds.
        first_dampen (bool): Dampening of first estimation.
        printing (bool): Flag to print output.
        model (fun): RC tuning model.
        rc_converter (fun): Converter for new RC parameters.
        resample (str): resample input data.
        min_std_troom (float): Mimimum variation of room temperature to do tuning, in K.
        min_std_tslab (float): Mimimum variation of slab temperature to do tuning, in K.
        rc_parameter_init (dict): Initial RC parameters.
        use_internal_params (bool): Use internal RC parameter tracking.
        '''
        self.horizon = horizon
        self.rctype = rctype
        self.mode = mode
        self.has_windows = has_windows
        self.lhs_samples = lhs_samples
        self.fix_c = fix_c
        self.dampen_param = dampen_param
        self.first_free = first_free
        self.first_dampen = first_dampen
        self.printing = printing
        self.model = model
        self.rc_converter = rc_converter
        self.resample = resample
        self.min_std_troom = min_std_troom
        self.min_std_tslab = min_std_tslab
        self.use_internal_params = use_internal_params

        # internal variables
        self.parameters = {}
        self.execution_count = 0
        self.df = pd.DataFrame()
        self.new_param = rc_parameter_init
        self.res_lhs = pd.DataFrame()
        self.optimal = None
        self.inputs = pd.DataFrame()
        self.tuning_parameter = {}
        self.res = None

    def clean_and_select_inputs(self, inputs):
        '''clean and select input data.'''

        # check columns
        for k, v in INPUTS_DESCRIPTION.items():
            if k not in inputs.columns:
                raise ValueError(f'Missing "{k}" for "{v}" in inputs.')
        for c in inputs:
            if c not in INPUTS_DESCRIPTION:
                print(f'WARNING: Unused column "{c}" supplied in inputs.')

        # resample
        if self.resample:
            inputs = inputs.resample(self.resample).mean()

        # select inputs for horizon
        inputs = inputs.loc[inputs.index[-1]-pd.DateOffset(seconds=self.horizon):].copy()
        inputs_0 = inputs.iloc[:1].copy()
        inputs = inputs.iloc[1:]

        return inputs, inputs_0

    def do_tuning(self, inputs, rc_parameter_ext={}, rc_parameter_prev={}):
        '''class wrapper for tuning
        
        Inputs:
        inputs (pd.DataFrame): Input data for rc tuning.
        rc_parameter_ext (dict): External RC parameter, default None.
        rc_parameter_prev (dict): Previous RC parameter, default None.
        '''

        # clean and store inputs
        self.inputs, inputs_0 = self.clean_and_select_inputs(inputs)

        # get default RC parameter
        rc_parameter = get_rc_parameter(self.inputs,
                                        inputs_0,
                                        self.rctype,
                                        self.mode,
                                        self.has_windows)

        # update default RC parameter
        if rc_parameter_ext:
            rc_parameter.update(rc_parameter_ext)

        # update with internal RC parameters
        if self.use_internal_params:
            self.new_param.update(rc_parameter_prev)
            rc_parameter_prev = self.new_param

        # determine if enough variation for tuning
        run_tuning = True
        if self.min_std_troom and not (self.first_free and self.execution_count == 0):
            run_tuning = self.inputs['zone_troom'].std() > self.min_std_troom
        if self.min_std_tslab and not (self.first_free and self.execution_count == 0):
            run_tuning = self.inputs['zone_tslab'].std() > self.min_std_tslab

        # run tuning
        if run_tuning:

            # make rc bounds
            # check if first run free
            if not (self.first_free and self.execution_count == 0):
                # first dampen
                if self.first_dampen and self.execution_count == 0:
                    dampen = float(self.first_dampen)
                # other dampen
                elif self.dampen_param and self.execution_count > 0:
                    dampen = float(self.dampen_param)
                else:
                    dampen = None
                # set rc bounds
                for k in rc_parameter:
                    if k.startswith('R') or k.startswith('C'):
                        if k in rc_parameter_prev.keys():
                            rc_parameter[k]['init'] = rc_parameter_prev[k]
                            if dampen:
                                rc_parameter[k]['lb'] = max(0,
                                    rc_parameter[k]['init'] - rc_parameter[k]['init'] * dampen)
                                rc_parameter[k]['ub'] = max(0,
                                    rc_parameter[k]['init'] + rc_parameter[k]['init'] * dampen)
                        # else:
                        #     print(f'WARNING: Skipping, as {k} is not in rc_parameter_prev.')
                # fix c
                if self.fix_c:
                    for k in rc_parameter:
                        if k.startswith('C'):
                            rc_parameter[k]['init'] = rc_parameter_prev[k]
                            rc_parameter[k]['lb'] = rc_parameter_prev[k]
                            rc_parameter[k]['ub'] = rc_parameter_prev[k]
            # save for book keeping
            self.tuning_parameter = rc_parameter.copy()

            # do tuning
            self.res = do_tuning(self.inputs,
                                 rc_parameter,
                                 lhs_samples=self.lhs_samples,
                                 printing=self.printing,
                                 rc_tuning_model=self.model,
                                 rc_converter=self.rc_converter)
            duration, objective, self.df, new_param, inputs, res_lhs, termination = self.res

            # store
            self.parameters[self.inputs.index[0]] = new_param
            self.new_param = new_param
            self.optimal = str(termination) == 'optimal'
            self.res_lhs = res_lhs
            self.execution_count += 1
        else:
            duration = -1
            self.df = pd.DataFrame()
            self.new_param = rc_parameter_prev
            self.res_lhs = pd.DataFrame()
            self.optimal = False
            objective = -1

        return {'optimal': self.optimal,
                'duration': duration,
                'objective': objective,
                'new_param': self.new_param,
                'df': self.df,
                'res_lhs': self.res_lhs}
