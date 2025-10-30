
# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced-order RC tuning model.
"""

# pylint: disable=too-many-arguments, invalid-name, too-many-locals
# pylint: disable=too-many-statements, redefined-outer-name, use-a-generator
# pylint: disable=possibly-used-before-assignment

import pandas as pd
from doper.utility import pandas_to_dict
from pyomo.environ import ConcreteModel, Set, Param, Var, Constraint #, Binary
from pyomo.environ import Objective, minimize #, Piecewise

from .. import rcModel

def model(inputs, parameter):
    '''rc tuning model for doper'''
    inputs = inputs.copy(deep=True)
    if isinstance(inputs.index[0], pd.Timestamp):
        inputs.index = inputs.index.astype(int)/1e9 # Convert datetime to UNIX

    model = ConcreteModel()

    # Sets
    index = list(inputs.index.values)
    model.ts = Set(initialize=index, ordered=True, doc='timesteps')
    timestep = index[1] - index[0]
    model.timestep = timestep
    timestep_scale = 3600 / float(timestep)
    model.timestep_scale = timestep_scale
    evaluation_ts = list(model.ts)[1:-1] # Timestep for accounting (cutoff last timestep)
    model.temps = Set(initialize=range(len(parameter['temps_initial'])), doc='zone temperatures')

    # for doper
    model.sum_energy_cost = Var(bounds=(0,0))
    model.sum_demand_cost = Var(bounds=(0,0))
    model.sum_export_revenue = Var(bounds=(0,0))
    model.sum_regulation_revenue = Var(bounds=(0,0))

    # Parameter (inputs)
    model.outside_temperature = Param(model.ts,
                                      initialize=pandas_to_dict(inputs['outside_temperature']), \
                                      doc='outside temperature [C]')
    # model.wind_speed = Param(model.ts, initialize=pandas_to_dict(inputs['wind_speed']), \
    #                          doc='outdoor wind speed [m/s]')
#     model.how1 = Param(model.ts, initialize=pandas_to_dict(inputs['zone_how1']), \
#                        doc='heat transfer coefficient outdoor [W/K]')
    model.zone_qi = Param(model.ts, initialize=pandas_to_dict(inputs['zone_qi']), \
                          doc='convective heat gain [W]')
    model.zone_troom = Param(model.ts, initialize=pandas_to_dict(inputs['zone_troom']), \
                             doc='room temperature [C]')
    # model.zone_Qw2i = Param(model.ts, initialize=pandas_to_dict(inputs['zone_Qw2i']), \
    #                          doc='Qw2i heat flow [W]')

    # Variables (parameter to tune)
    param = parameter
    model.rc_parameter = []
    # one mass (air)
    model.Rw2i = Var(initialize=param['Rw2i']['init'],
                     bounds=(param['Rw2i']['lb'], param['Rw2i']['ub']))
    model.rc_parameter.append('Rw2i')
    model.Ci = Var(initialize=param['Ci']['init'],
                   bounds=(param['Ci']['lb'], param['Ci']['ub']))
    model.rc_parameter.append('Ci')
    # two mass (slab)
    if parameter['type'] in ['R2C2', 'R4C2', 'R5C2', 'R5C3', 'R6C3']:
        model.zone_qs = Param(model.ts, initialize=pandas_to_dict(inputs['zone_qs']), \
                            doc='radiative heat gain [W]')
        model.zone_tslab = Param(model.ts, initialize=pandas_to_dict(inputs['zone_tslab']), \
                                doc='slab temperature [C]')
        model.Ris = Var(initialize=param['Ris']['init'],
                        bounds=(param['Ris']['lb'], param['Ris']['ub']))
        model.rc_parameter.append('Ris')
        model.Cs = Var(initialize=param['Cs']['init'],
                    bounds=(param['Cs']['lb'], param['Cs']['ub']))
        model.rc_parameter.append('Cs')
    # three mass (wall)
    if parameter['type'] in ['R5C3', 'R6C3']:
        model.zone_qw = Param(model.ts, initialize=pandas_to_dict(inputs['zone_qw']), \
                              doc='radiative heat gain on walls [W]')
        model.zone_twall = Param(model.ts, initialize=pandas_to_dict(inputs['zone_twall']), \
                                 doc='wall temperature [C]')
        model.Riw = Var(initialize=param['Riw']['init'],
                        bounds=(param['Riw']['lb'], param['Riw']['ub']))
        model.rc_parameter.append('Riw')
        model.Cw = Var(initialize=param['Cw']['init'],
                       bounds=(param['Cw']['lb'], param['Cw']['ub']))
        model.rc_parameter.append('Cw')
    # window system
    if parameter['type'] in ['R4C2', 'R5C2', 'R5C3', 'R6C3']:
        model.zone_abs1 = Param(model.ts, initialize=pandas_to_dict(inputs['zone_abs1']), \
                                doc='window absorption outer layer [W]')
        model.zone_abs2 = Param(model.ts, initialize=pandas_to_dict(inputs['zone_abs2']), \
                                doc='window absorption inner layer [W]')
        model.Row1 = Var(initialize=param['Row1']['init'],
                        bounds=(param['Row1']['lb'], param['Row1']['ub']))
        model.rc_parameter.append('Row1')
        model.Rw1w2 = Var(initialize=param['Rw1w2']['init'],
                        bounds=(param['Rw1w2']['lb'], param['Rw1w2']['ub']))
        model.rc_parameter.append('Rw1w2')
    # exterior walls
    if parameter['type'] in ['R5C2', 'R6C3']:
        model.Roi = Var(initialize=param['Roi']['init'],
                        bounds=(param['Roi']['lb'], param['Roi']['ub']))
        model.rc_parameter.append('Roi')
    # model.Qw2i = Var(model.ts)

    # Windspeed model
    # model.how1 = Var(model.ts, doc='heat transfer coefficient outdoor')
    # def windspeed_model(model, ts):
    #     return model.how1[ts] == 1 / (parameter['window_area'] * \
    #                                   (parameter['convection_window_offset'] \
    #                                    + parameter['convection_window_scale'] \
    #                                    * model.wind_speed[ts]))
    # model.constraint_windspeed_model = Constraint(model.ts,
    #                                               rule=windspeed_model,
    #                                               doc='calc of heat transfer from wind')
    # if parameter['type'] in ['R5C3', 'R6C3']:
    #     def Qw2i_calc(model, ts):
    #         '''calculate window to interior heat flow'''
    #         Qw1_ext = model.zone_abs1[ts]
    #         Qw2_ext = model.zone_abs2[ts]
    #         Ti = model.zone_troom[ts]
    #         To = model.outside_temperature[ts]
    #         Row1 = model.Row1 #* model.how1[ts]
    #         Rw1w2 = model.Rw1w2
    #         Rw2i = model.Rw2i
    #         a = -Qw1_ext*Row1 - Qw2_ext*Row1 - Qw2_ext*Rw1w2 + Ti - To
    #         b = Row1 + Rw1w2 + Rw2i
    #         return model.Qw2i[ts] == a / b
    #     model.constraint_Qw2i_calc= Constraint(model.ts,
    #                                            rule=Qw2i_calc,
    #                                            doc='calc of heat transfer from window')

    # RC model
    model.zone_temp = Var(model.ts, model.temps, doc='temperature in zone')
    def zone_temp(model, ts, temps):
        '''calculate zone temperature'''
        if ts == model.ts.at(1):
            #if parameter['type'] in ['R5C3', 'R6C3']:
            #    print('*** Windspeed in Model ***')
            return model.zone_temp[ts, temps] == parameter['temps_initial'][temps]

        # inputs
        Ti_p = model.zone_temp[ts-timestep, 0]
        To = model.outside_temperature[ts]
        Qi_ext = model.zone_qi[ts]

        # two mass
        if 'C2' in parameter['type']:
            Ts_p = model.zone_temp[ts-timestep, 1]
            Qs_ext = model.zone_qs[ts]
        # three mass
        elif 'C3' in parameter['type']:
            Ts_p = model.zone_temp[ts-timestep, 1]
            Tw_p = model.zone_temp[ts-timestep, 2]
            Qs_ext = model.zone_qs[ts]
            Qw_ext = model.zone_qw[ts]
        # window system
        if any([parameter['type'].startswith(r) for r in ['R4', 'R5', 'R6']]):
            Qw1_ext = model.zone_abs1[ts]
            Qw2_ext = model.zone_abs2[ts]

        # rc model selection
        param = {}
        param['timestep'] = timestep
        for p in model.rc_parameter:
            param[p] = getattr(model, p)

        if parameter['type'] == 'R1C1':
            res_temps = rcModel.R1C1(1, Ti_p, To, Qi_ext, param)
        elif parameter['type'] == 'R2C2':
            res_temps = rcModel.R2C2(1, Ti_p, Ts_p, To, Qi_ext, Qs_ext, param)
        elif parameter['type'] == 'R4C2':
            res_temps = rcModel.R4C2(1, Ti_p, Ts_p, To, Qw1_ext, Qw2_ext,
                                        Qi_ext, Qs_ext, param)
        elif parameter['type'] == 'R5C2':
            # disable wall C and R
            Tw_p = Ts_p
            Qw_ext = 0
            param['Riw'] = 1e6
            param['Cw'] = 0
            res_temps = rcModel.R6C3(1, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
                                        Qi_ext, Qs_ext, Qw_ext, param)[:2]
        elif parameter['type'] in ['R5C3', 'R6C3']:
            # include wind
            # param['Row1'] = param['Row1'] * model.how1[ts]
            # param['Row1'] = model.how1[ts]
            if parameter['type'] == 'R6C3':
                # param['Roi'] = model.Roi
                # include wind
                # param['Roi'] = param['Roi'] * model.how1[ts]
                res_temps = rcModel.R6C3(1, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
                                            Qi_ext, Qs_ext, Qw_ext, param)
            res_temps = rcModel.R5C3(1, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
                                     Qi_ext, Qs_ext, Qw_ext, param)
        else:
            raise ValueError(f'RC model type {parameter["type"]} not supported.')
        return model.zone_temp[ts, temps] == res_temps[temps]
    model.constraint_zone_temp = Constraint(model.ts,
                                            model.temps,
                                            rule=zone_temp,
                                            doc='calculaiton of temperature')

    # Objective
    model.diff_troom = Var(model.ts, doc='difference room temperature [C]')
    model.mse_troom = Var(doc='mean squared room error [C]')

    def diff_troom(model, ts):
        '''calculate room air difference'''
        return model.diff_troom[ts] == model.zone_temp[ts, 0] - model.zone_troom[ts]
    model.constraint_diff_troom = Constraint(model.ts,
                                             rule=diff_troom,
                                             doc='difference room calculation')
    def mse_troom(model):
        '''calculate mse for troom'''
        return model.mse_troom == sum(model.diff_troom[t] ** 2 for t in evaluation_ts) / len(inputs)
    model.constraint_mse_troom = Constraint(rule=mse_troom,
                                            doc='mean squared error room temperature')

    model.diff_tslab = Var(model.ts, doc='difference slab temperature [C]')
    model.mse_tslab = Var(doc='mean squared slab error [C]')
    if 'weight_tslab' in parameter['objective'].keys():
        def diff_tslab(model, ts):
            '''calculate slab difference'''
            return model.diff_tslab[ts] == model.zone_temp[ts, 1] - model.zone_tslab[ts]
        model.constraint_diff_tslab = Constraint(model.ts,
                                                 rule=diff_tslab,
                                                 doc='difference slab calculation')

        def mse_tslab(model):
            '''calculate mse for tslab'''
            return model.mse_tslab == \
                sum(model.diff_tslab[t] ** 2 for t in evaluation_ts) / len(inputs)
        model.constraint_mse_tslab = Constraint(rule=mse_tslab,
                                                doc='mean squared error slab temperature')

    model.diff_twall = Var(model.ts, doc='difference wall temperature [C]')
    model.mse_twall = Var(doc='mean squared wall error [C]')
    if 'weight_twall' in parameter['objective'].keys():
        def diff_twall(model, ts):
            '''calculate wall difference'''
            return model.diff_twall[ts] == model.zone_temp[ts, 2] - model.zone_twall[ts]
        model.constraint_diff_twall = Constraint(model.ts,
                                                 rule=diff_twall,
                                                 doc='difference wall calculation')

        def mse_twall(model):
            '''calculate mse for twall'''
            return model.mse_twall == \
                sum(model.diff_twall[t] ** 2 for t in evaluation_ts) / len(inputs)
        model.constraint_mse_twall = Constraint(rule=mse_twall,
                                                doc='mean squared error wall temperature')

    # model.diff_Qw2i = Var(model.ts, doc='difference Qw2i [W]')
    # model.mse_Qw2i = Var(doc='mean squared Qw2i error [C]')
    # if parameter['type'] in ['R5C3', 'R6C3']:
    #     def diff_Qw2i(model, ts):
    #         return model.diff_Qw2i[ts] == model.Qw2i[ts] - model.zone_Qw2i[ts]
    #     model.constraint_diff_Qw2i = Constraint(model.ts,
    # rule=diff_Qw2i, doc='difference Qw2i calculation')

    #     def mse_Qw2i(model):
    #         return model.mse_Qw2i == \
    # sum(model.diff_Qw2i[t] ** 2 for t in evaluation_ts) / len(inputs)
    #     model.constraint_mse_Qw2i = Constraint(rule=mse_Qw2i,
    # doc='mean squared error Qw2i temperature')

    def objective_function(model):
        '''objective function'''
        par = parameter['objective']
        obj = model.mse_troom * par['weight_troom']
        if 'weight_tslab' in par.keys():
            obj += model.mse_tslab * par['weight_tslab']
        if 'weight_twall' in par.keys():
            obj += model.mse_twall * par['weight_twall']
            print('Adding TWall to objective')
        if 'weight_Qw2i' in par.keys():
            obj += model.mse_Qw2i * par['weight_Qw2i']
            print('Adding Qw2i to objective')
        return obj
    model.objective = Objective(rule=objective_function,
                                sense=minimize,
                                doc='objective function')
    return model
