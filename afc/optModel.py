# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Optimization model.
"""

# pylint: disable=bare-except, too-many-locals, invalid-name, too-many-statements
# pylint: disable=pointless-string-statement

import os
import sys
import itertools
import pandas as pd
from pyomo.environ import Objective, minimize
from pyomo.environ import Set, Param, Var, Constraint, Binary

from doper import pandas_to_dict
from doper.models.basemodel import base_model
# from doper.models.battery import add_battery

try:
    from .rcModel import R2C2, R4C2, R5C3, R6C3
except:
    sys.path.append(os.path.join('..', 'afc'))
    from rcModel import R2C2, R4C2, R5C3, R6C3

def control_model(inputs, parameter):
    """Control model for the AFC."""

    model = base_model(inputs, parameter)
    # model = add_battery(model, inputs, parameter)

    inputs = inputs.copy(deep=True)
    if isinstance(inputs.index[0], type(pd.to_datetime(0))):
        inputs.index = inputs.index.astype('int64').astype(int)/1e9 # Convert datetime to UNIX

    model.fzones = Set(initialize=parameter['facade']['windows'], doc='window zones')
    model.fstates = Set(initialize=parameter['facade']['states'], doc='facade states')
    model.fstate_bin = Var(model.ts, model.fzones, model.fstates, domain=Binary,
                           doc='facade binary')
    # Thermal Comfort
    model.temps = Set(initialize=range(len(parameter['zone']['temps_name'])),
                      doc='zone temperatures')
    model.zone_temp = Var(model.ts, model.temps, doc='temperature in zone')
    model.zone_heat = Var(model.ts, model.temps, bounds=(0, None), doc='heating in zone')
    model.zone_cool = Var(model.ts, model.temps, bounds=(0, None), doc='cooling in zone')
    model.zone_temp_max = Param(model.ts, model.temps,
        initialize=pandas_to_dict(inputs[[f'temp_{b}_max' for b in \
                                          parameter['zone']['temps_name']]], columns=model.temps),
                                doc='maximal temperatures')
    model.zone_temp_min = Param(model.ts, model.temps,
        initialize=pandas_to_dict(inputs[[f'temp_{b}_min' for b in \
                                          parameter['zone']['temps_name']]], columns=model.temps),
                                doc='minimal temperatures')

    # Facade loopup table
    fmap = list(itertools.product(model.fzones, model.fstates))
    model.facade_wpi = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'wpi_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                             doc='facade wpi lookup table')
    if parameter['zone']['param']['type'] == 'R2C2':
        model.facade_shg = Param(model.ts, model.fzones, model.fstates,
            initialize=pandas_to_dict(inputs[[f'shg_{z}_{s}' for z, s in fmap]], \
                                      columns=fmap), \
                                 doc='facade shg lookup table')
    model.facade_vil = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'ev_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                             doc='facade vil lookup table')
    model.facade_abs1 = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'abs1_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                              doc='facade abs1 lookup table')
    model.facade_abs2 = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'abs2_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                              doc='facade abs2 lookup table')
    model.facade_tsol = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'tsol_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                              doc='facade tsol lookup table')
    model.facade_iflr = Param(model.ts, model.fzones, model.fstates,
        initialize=pandas_to_dict(inputs[[f'iflr_{z}_{s}' for z, s in fmap]], \
                                  columns=fmap), \
                              doc='facade iflr lookup table')
    model.wpi_constraint_min = Param(model.ts, initialize=pandas_to_dict(inputs['wpi_min']), \
                                     doc='constraint for wpi lookup table')
    model.glare_constraint_max = Param(model.ts, initialize=pandas_to_dict(inputs['glare_max']), \
                                       doc='constraint for glare lookup table')

    # Facade model
    model.zone_wpi = Var(model.ts, doc='wpi in zone')
    model.zone_shg = Var(model.ts, doc='shg in zone')
    model.zone_vil = Var(model.ts, doc='vil in zone')
    model.zone_abs1 = Var(model.ts, doc='abs1 in zone')
    model.zone_abs2 = Var(model.ts, doc='abs2 in zone')
    model.zone_tsol = Var(model.ts, doc='tsol in zone')
    model.zone_iflr = Var(model.ts, doc='iflr in zone')
    model.fstate = Var(model.ts, model.fzones, doc='facade state')
    model.der_fstate = Var(model.ts, model.fzones, doc='facade state')
    # Precompute
    inputs['how1'] = 1 / (parameter['facade']['window_area'] * \
                          (parameter['facade']['convection_window_offset'] \
                           + parameter['facade']['convection_window_scale'] \
                           * inputs['wind_speed']))
    model.how1 = Param(model.ts, initialize=pandas_to_dict(inputs['how1']), \
                       doc='window heat transfer rate [K/W]')

    def fstate_unique(model, ts, fzone):
        return 1 == sum(model.fstate_bin[ts, fzone, s] for s in model.fstates)
    model.constraint_fstate_unique = Constraint(model.ts, model.fzones,
        rule=fstate_unique, doc='calculation of wpi')

    def zone_wpi(model, ts):
        return model.zone_wpi[ts] == \
            sum(model.facade_wpi[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_wpi = Constraint(model.ts, rule=zone_wpi, doc='calculation of wpi')

    if parameter['zone']['param']['type'] == 'R2C2':
        def zone_shg(model, ts):
            return model.zone_shg[ts] == \
                sum(model.facade_shg[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
        model.constraint_zone_shg = Constraint(model.ts, rule=zone_shg, doc='calculation of shg')

    def zone_vil(model, ts):
        return model.zone_vil[ts] == \
            sum(model.facade_vil[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_vil = Constraint(model.ts, rule=zone_vil, doc='calculation of vil')

    def zone_abs1(model, ts):
        return model.zone_abs1[ts] == \
            sum(model.facade_abs1[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_abs1 = Constraint(model.ts, rule=zone_abs1, doc='calculation of abs1')

    def zone_abs2(model, ts):
        return model.zone_abs2[ts] == \
            sum(model.facade_abs2[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_abs2 = Constraint(model.ts, rule=zone_abs2, doc='calculation of abs2')

    def zone_tsol(model, ts):
        return model.zone_tsol[ts] == \
            sum(model.facade_tsol[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_tsol = Constraint(model.ts, rule=zone_tsol, doc='calculation of tsol')

    def zone_iflr(model, ts):
        return model.zone_iflr[ts] == \
            sum(model.facade_iflr[ts, z, s] * model.fstate_bin[ts, z, s] for z, s in fmap)
    model.constraint_zone_iflr= Constraint(model.ts, rule=zone_iflr, doc='calculation of iflr')

    def fstate(model, ts, zone):
        return model.fstate[ts, zone] == \
            sum(s * model.fstate_bin[ts, zone, s] for s in model.fstates)
    model.constraint_fstate = Constraint(model.ts, model.fzones, rule=fstate,
                                         doc='calculation of fstate')

    def der_fstate(model, ts, zone):
        if ts == model.ts.at(1):
            return model.der_fstate[ts, zone] == 0
        return model.der_fstate[ts, zone] == \
            model.fstate[ts, zone] \
            - model.fstate[ts-model.timestep[ts], zone]
    model.constraint_der_fstate = Constraint(model.ts, model.fzones, rule=der_fstate,
                                             doc='calculation of der_fstate')

    # Visual Comfort
    model.zone_wpi_total = Var(model.ts, doc='total wpi in zone')
    model.zone_wpi_ext = Var(model.ts, bounds=(0, parameter['zone']['lighting_capacity']),
                             doc='artificial lighting lx')
    model.zone_lights_bin = Var(model.ts, domain=Binary, doc='lights binary')
    model.zone_glare = Var(model.ts, doc='glare in zone')
    model.zone_glare_penalty = Var(model.ts, bounds=(0, None), doc='glare penalty in zone')
    model.sum_glare_penalty = Var(bounds=(0, None), doc='sum of glare penalty in zone')
    model.zone_view_penalty = Var(model.ts, bounds=(0, None), doc='view penalty in zone')
    model.sum_view_penalty = Var(bounds=(0, None), doc='sum of view penalty in zone')
    model.zone_actuation = Var(model.ts, doc='actuation in zone')
    model.zone_actuation_pos = Var(model.ts, model.fzones, bounds=(0, None),
                                   doc='positive actuation in zone')
    model.zone_actuation_neg = Var(model.ts, model.fzones, bounds=(0, None),
                                   doc='negativ actuation in zone')
    model.zone_abs_actuation = Var(model.ts, doc='absolute actuation in zone')
    model.sum_zone_actuation = Var(bounds=(0, None), doc='sum of actuation in zone')

    def zone_wpi_total(model, ts):
        return model.zone_wpi_total[ts] == model.zone_wpi[ts] + model.zone_wpi_ext[ts]
    model.constraint_zone_wpi_total = Constraint(model.ts, rule=zone_wpi_total,
                                                 doc='total wpi calculation')

    def ctrl_lights_1(model, ts):
        return model.zone_wpi_ext[ts] <= model.zone_lights_bin[ts] * 1e3
    model.constraint_ctrl_lights_1 = Constraint(model.ts, rule=ctrl_lights_1,
                                                doc='Avoid heat with lights')

    def ctrl_lights_2(model, ts):
        return model.zone_wpi[ts] - model.wpi_constraint_min[ts] <= \
            (1 - model.zone_lights_bin[ts]) * 1e5
    model.constraint_ctrl_lights_2 = Constraint(model.ts, rule=ctrl_lights_2,
                                                doc='Avoid heat with lights')

    def ctrl_lights_3(model, ts):
        return model.zone_wpi_ext[ts] - model.wpi_constraint_min[ts] - model.zone_wpi[ts] <= \
            (1 - model.zone_lights_bin[ts]) * 1e5
    model.constraint_ctrl_lights_3 = Constraint(model.ts, rule=ctrl_lights_3,
                                                doc='Avoid heat with lights')

    def zone_wpi_min(model, ts):
        return model.zone_wpi_total[ts] >= model.wpi_constraint_min[ts]
    model.constraint_zone_wpi_min = Constraint(model.ts, rule=zone_wpi_min, doc='limit wpi')

    def zone_glare(model, ts):
        return model.zone_glare[ts] == model.zone_vil[ts] * 6.22e-5 + 0.184
    model.constraint_zone_glare = Constraint(model.ts, rule=zone_glare, doc='calculation of glare')

    def zone_limit_glare(model, ts):
        return model.zone_glare[ts] <= model.glare_constraint_max[ts]
    model.constraint_zone_limit_glare = Constraint(model.ts, rule=zone_limit_glare,
                                                   doc='limit glare')

    def zone_glare_penalty(model, ts):
        return model.zone_glare_penalty[ts] >= (model.zone_glare[ts] \
                                                - (model.glare_constraint_max[ts] \
                                                   - parameter['zone']['glare_diff'])) \
                                               * parameter['zone']['glare_scale']
    model.constraint_zone_glare_penalty = Constraint(model.ts, rule=zone_glare_penalty,
                                                     doc='calculation of glare penalty')

    def sum_glare_penalty(model):
        return model.sum_glare_penalty == \
            sum(model.zone_glare_penalty[t] for t in model.accounting_ts)
    model.constraint_sum_glare_penalty = Constraint(rule=sum_glare_penalty,
                                                    doc='calcualtion of sum glare penalty')

    def zone_view_penalty(model, ts, fzone):
        return model.zone_view_penalty[ts] >= (- model.fstate[ts, fzone] \
                                               + max(parameter['facade']['states'])) \
                                              * parameter['zone']['view_scale']
    model.constraint_zone_view_penalty = Constraint(model.ts, model.fzones, rule=zone_view_penalty,
                                                    doc='calculation of view penalty')

    def sum_view_penalty(model):
        return model.sum_view_penalty == \
            sum(model.zone_view_penalty[t] for t in model.accounting_ts)
    model.constraint_sum_view_penalty = Constraint(rule=sum_view_penalty,
                                                   doc='calcualtion of sum view penalty')

    def zone_actuation(model, ts):
        return model.zone_actuation[ts] == sum(model.der_fstate[ts, z] for z in model.fzones)
    model.constraint_zone_actuation = Constraint(model.ts, rule=zone_actuation,
                                                 doc='calcualtion of actuation')

    def zone_actuation_pos(model, ts, fzone):
        return model.zone_actuation_pos[ts, fzone] >= model.der_fstate[ts, fzone]
    model.constraint_zone_actuation_pos = Constraint(model.ts, model.fzones,
                                                     rule=zone_actuation_pos,
                                                     doc='calcualtion of pos actuation')

    def zone_actuation_neg(model, ts, fzone):
        return model.zone_actuation_neg[ts, fzone] >= -1 * model.der_fstate[ts, fzone]
    model.constraint_zone_actuation_neg = Constraint(model.ts, model.fzones,
                                                     rule=zone_actuation_neg,
                                                     doc='calcualtion of neg actuation')

    def zone_abs_actuation(model, ts):
        return model.zone_abs_actuation[ts] == \
            sum(model.zone_actuation_pos[ts, z] \
                + model.zone_actuation_neg[ts, z] for z in model.fzones)
    model.constraint_zone_abs_actuation = Constraint(model.ts, rule=zone_abs_actuation,
                                                     doc='calcualtion of abs actuation')

    def zone_sum_actuation(model):
        return model.sum_zone_actuation == \
            sum(model.zone_abs_actuation[t] for t in model.accounting_ts)
    model.constraint_zone_sum_actuation = Constraint(rule=zone_sum_actuation,
                                                     doc='calcualtion of sum actuation')

    ix_night = inputs.index[inputs[[c for c in inputs.columns \
                                    if ('wpi_' in c or 'tsol_' in c) \
                                    and not 'min' in c]].sum(axis=1) == 0]
    bin_clear = [0]*(len(model.fstates)-1)+[1]
    bin_fixed = [True]*len(model.fstates)
    for ts in ix_night:
        for fzone in model.fzones:
            for s in model.fstates:
                model.fstate_bin[ts, fzone, s] = bin_clear[s]
                model.fstate_bin[ts, fzone, s].fixed = bin_fixed[s]

    # Electricity
    model.p = Var(model.ts, doc='total electric power')
    model.p_lights = Var(model.ts, doc='electric power of lights')
    model.p_heating = Var(model.ts, doc='electric power of heating')
    model.p_cooling = Var(model.ts, doc='electric power of cooling')
    model.p_equipment = Param(model.ts, initialize=pandas_to_dict(inputs['equipment']),
                             doc='electric equipment load in room')
    model.zone_plugload = Param(model.ts, initialize=pandas_to_dict(inputs['plug_load']),
                                doc='thermal plugloads in room')
    model.zone_occload = Param(model.ts, initialize=pandas_to_dict(inputs['occupant_load']),
                               doc='thermal occupant load in room')
    model.occupancy_light = Param(model.ts, initialize=pandas_to_dict(inputs['occupancy_light']),
                                  doc='occupancy light in room')

    def zone_p(model, ts):
        return model.p[ts] == model.p_lights[ts] \
                              + model.p_heating[ts] \
                              + model.p_cooling[ts] \
                              + model.p_equipment[ts]
    model.constraint_zone_p = Constraint(model.ts, rule=zone_p, doc='electric power of zone')

    def zone_p_lights(model, ts):
        return model.p_lights[ts] == model.zone_wpi_ext[ts] \
                                     * parameter['zone']['lighting_efficiency'] \
                                     * model.occupancy_light[ts]
    model.constraint_zone_p_lights = Constraint(model.ts, rule=zone_p_lights,
                                                doc='electric power of lights')

    def zone_p_heat(model, ts):
        return model.p_heating[ts] == sum(model.zone_heat[ts, temps] for temps in model.temps) \
                                      * parameter['zone']['heating_efficiency']
    model.constraint_zone_p_heat = Constraint(model.ts, rule=zone_p_heat,
                                              doc='electric power of heating')

    def zone_p_cool(model, ts):
        return model.p_cooling[ts] == sum(model.zone_cool[ts, temps] for temps in model.temps) \
                                      * parameter['zone']['cooling_efficiency']
    model.constraint_zone_p_cool = Constraint(model.ts, rule=zone_p_cool,
                                              doc='electric power of cooling')

    # Integration in model (at node 0 only)
    def zone_total_load(model, ts, node):
        return model.building_load_dynamic[ts, node] == model.p[ts] / 1e3 # W -> kW
#         if node == model.nodes.at(parameter['zone']['zone_id']):
#             return model.building_load_dynamic[ts, node] == (model.p[ts]) / 1e3
#         else:
#             return model.building_load_dynamic[ts, node] == 0
    model.constraint_zone_total_load = Constraint(model.ts, model.nodes, rule=zone_total_load,
                                                  doc='total electric power to link models')

    def zone_total_load_end(model, ts, node):
        if ts == model.ts.at(len(model.ts)):
            return model.building_load_dynamic[ts, node] == \
                model.building_load_dynamic[ts-model.timestep[ts], node]
        return model.building_load_dynamic[ts, node] >= -9e6
    model.constraint_zone_total_load_end = Constraint(model.ts, model.nodes,
                                                      rule=zone_total_load_end,
                                                      doc='fix last timestep')

    # Thermal Comfort
    model.zone_qi = Var(model.ts, doc='convective gains in zone')
    model.zone_qw = Var(model.ts, doc='gains in wall (radiative) in zone')
    model.zone_qs = Var(model.ts, doc='gains in slab (radiative) in zone')

    def zone_qi(model, ts):
        param = parameter['zone']['param']
        Qi_ext = model.p_lights[ts] * (1 - parameter['zone']['lighting_split']) \
                 + model.zone_occload[ts] * (1 - parameter['zone']['occupancy_split']) \
                 + model.zone_plugload[ts] * (1 - parameter['zone']['plugload_split']) \
                 + model.zone_heat[ts, 0] - model.zone_cool[ts, 0]
        if param['type'] == 'R2C2':
            Qi_ext += model.zone_shg[ts] * (1 - parameter['zone']['shg_split'])
        else:
            Qi_ext += model.zone_tsol[ts] * (1 - parameter['zone']['tsol_split'])
        return model.zone_qi[ts] == Qi_ext
    model.constraint_zone_qi = Constraint(model.ts, rule=zone_qi, doc='calculaiton convective')

    def zone_qw(model, ts):
        param = parameter['zone']['param']
        Qw_ext = model.p_lights[ts] * parameter['zone']['lighting_split'] \
                 + model.zone_occload[ts] * parameter['zone']['occupancy_split'] \
                 + model.zone_plugload[ts] * parameter['zone']['plugload_split']
        if param['type'] == 'R2C2':
            Qw_ext += model.zone_shg[ts] * parameter['zone']['shg_split']
        else:
            Qw_ext += model.zone_tsol[ts] * parameter['zone']['tsol_split']
        return model.zone_qw[ts] == Qw_ext
    model.constraint_zone_qw = Constraint(model.ts, rule=zone_qw,
                                          doc='calculaiton radiative gains wall')

    def zone_qs(model, ts):
        return model.zone_qs[ts] == model.zone_iflr[ts]
    model.constraint_zone_qs = Constraint(model.ts, rule=zone_qs,
                                          doc='calculaiton radiative gains slab')

    def zone_temp(model, ts, temps):
        if ts == model.ts.at(1):
            return model.zone_temp[ts, temps] == parameter['zone']['temps_initial'][temps]

        Ti_p = model.zone_temp[ts-model.timestep[ts], 0]
        Ts_p = model.zone_temp[ts-model.timestep[ts], 1]
        To = model.outside_temperature[ts-model.timestep[ts]]
        Qi_ext = model.zone_qi[ts-model.timestep[ts]]
        Qw_ext = model.zone_qw[ts-model.timestep[ts]]
        param = parameter['zone']['param'].copy()
        param['timestep'] = model.timestep[ts]
        if param['type'] == 'R2C2':
            res_temps = R2C2(1, Ti_p, Ts_p, To, Qi_ext, Qw_ext, param)
        elif param['type'] in ['R4C2', 'R5C3', 'R6C3']:
            Qw1_ext = model.zone_abs1[ts-model.timestep[ts]]
            Qw2_ext = model.zone_abs2[ts-model.timestep[ts]]
            Qs_ext = model.zone_qs[ts-model.timestep[ts]]
            if param['type'] == 'R4C2':
                Qi_ext = Qi_ext + Qw_ext - Qs_ext
                Qw_ext = Qs_ext
                res_temps = R4C2(1, Ti_p, Ts_p, To, Qw1_ext, Qw2_ext, Qi_ext, Qw_ext, param)
            elif param['type'] == 'R5C3':
                Tw_p = model.zone_temp[ts-model.timestep[ts], 2]
                Qw_ext = Qw_ext - Qs_ext
                res_temps = R5C3(1, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
                                 Qi_ext, Qs_ext, Qw_ext, param)
            elif param['type'] == 'R6C3':
                Tw_p = model.zone_temp[ts-model.timestep[ts], 2]
                Qw_ext = Qw_ext - Qs_ext
                param['Row1'] = param['Row1'] * model.how1[ts]
                res_temps = R6C3(1, Ti_p, Ts_p, Tw_p, To, Qw1_ext, Qw2_ext,
                                 Qi_ext, Qs_ext, Qw_ext, param)
        else:
            raise ValueError(f'RC model type {param["type"]} not supported.')
        return model.zone_temp[ts, temps] == res_temps[temps]
    model.constraint_zone_temp = Constraint(model.ts, model.temps, rule=zone_temp,
                                            doc='calculaiton of temperature')

    def zone_heating_limit(model, ts, temps):
        return model.zone_heat[ts, temps] <= parameter['zone']['heat_max'][temps]
    model.constraint_zone_heating_limit = Constraint(model.ts, model.temps,
                                                     rule=zone_heating_limit,
                                                     doc='limit of heating')

    def zone_cooling_limit(model, ts, temps):
        return model.zone_cool[ts, temps] <= parameter['zone']['cool_max'][temps]
    model.constraint_zone_cooling_limit = Constraint(model.ts, model.temps,
                                                     rule=zone_cooling_limit,
                                                     doc='limit of cooling')

    def zone_max_temperature(model, ts, temps):
        if ts == model.ts.at(-1):
            return model.zone_temp[ts, temps] <= 1e6
        return model.zone_temp[ts, temps] <= model.zone_temp_max[ts-model.timestep[ts], temps]
    model.constraint_zone_max_temperature = Constraint(model.ts, model.temps,
                                                       rule=zone_max_temperature,
                                                       doc='max temperature')

    def zone_min_temperature(model, ts, temps):
        if ts == model.ts.at(-1):
            return model.zone_temp[ts, temps] <= 1e6
        return model.zone_temp[ts, temps] >= model.zone_temp_min[ts-model.timestep[ts], temps]
    model.constraint_zone_min_temperature = Constraint(model.ts, model.temps,
                                                       rule=zone_min_temperature,
                                                       doc='min temperature')

    if 'weight_degradation' in parameter['objective']:
        print('WARNING: No "degradation" in objective function.')
    def objective_function(model):
        return model.sum_energy_cost * parameter['objective']['weight_energy'] \
               + model.sum_demand_cost * parameter['objective']['weight_demand'] \
               + model.sum_export_revenue * parameter['objective']['weight_export'] \
               + model.sum_zone_actuation * parameter['objective']['weight_actuation'] \
               + model.sum_glare_penalty * parameter['objective']['weight_glare'] \
               + model.sum_view_penalty * parameter['objective']['weight_view'] \
               + model.co2_total * parameter['objective']['weight_ghg']
               #+ model.sum_regulation_revenue * parameter['objective']['weight_regulation']
    model.objective = Objective(rule=objective_function, sense=minimize, doc='objective function')
    return model

def afc_output_list():
    """DOPER output list for the AFC."""

    ctrlOutputs = []
    #ctrlOutputs.append({'name': '', 'data', '', 'df_label': ''})
    ctrlOutputs.append({'data': 'zone_wpi', 'df_label': 'Window Illuminance [lx]'})
    ctrlOutputs.append({'data': 'zone_wpi_ext', 'df_label': 'Artificial Illuminance [lx]'})
    ctrlOutputs.append({'data': 'p_lights', 'df_label': 'Power Lights [W]'})
    ctrlOutputs.append({'data': 'zone_shg', 'df_label': 'Solar Heat Gain [W]'})
    ctrlOutputs.append({'data': 'zone_glare', 'df_label': 'Glare [-]'})
    ctrlOutputs.append({'data': 'outside_temperature', 'df_label': 'Outside Air Temperature [C]'})
    ctrlOutputs.append({'data': 'p_heating', 'df_label': 'Power Heating [W]'})
    ctrlOutputs.append({'data': 'p_cooling', 'df_label': 'Power Cooling [W]'})
    ctrlOutputs.append({'data': 'zone_actuation', 'df_label': 'Actuation [-]'})
    ctrlOutputs.append({'data': 'zone_abs_actuation', 'df_label': 'Actuation Abs [-]'})
    ctrlOutputs.append({'data': 'glare_constraint_max', 'df_label': 'Glare Max [-]'})
    ctrlOutputs.append({'data': 'zone_glare_penalty', 'df_label': 'Glare Penalty [-]'})
    ctrlOutputs.append({'data': 'zone_view_penalty', 'df_label': 'View Penalty [-]'})
    ctrlOutputs.append({'data': 'wpi_constraint_min',
                        'df_label': 'Work Plane Illuminance Min [lx]'})
    ctrlOutputs.append({'data': 'zone_occload', 'df_label': 'Power Occupancy [W]'})
    ctrlOutputs.append({'data': 'zone_plugload', 'df_label': 'Power Plugs [W]'})
    ctrlOutputs.append({'data': 'zone_wpi_total', 'df_label': 'Work Plane Illuminance [lx]'})
    ctrlOutputs.append({'data': 'zone_abs1', 'df_label': 'Window Absorption 1 [W]'})
    ctrlOutputs.append({'data': 'zone_abs2', 'df_label': 'Window Absorption 2 [W]'})
    ctrlOutputs.append({'data': 'zone_tsol', 'df_label': 'Window Transmitted Solar [W]'})
    ctrlOutputs.append({'data': 'zone_qi', 'df_label': 'Convective Internal Gains [W]'})
    ctrlOutputs.append({'data': 'zone_qs', 'df_label': 'Radiative Slab Gains [W]'})
    ctrlOutputs.append({'data': 'zone_qw', 'df_label': 'Radiative Wall Gains [W]'})
    ctrlOutputs.append({'data': 'p_equipment', 'df_label': 'Power Equipment [W]'})
    ctrlOutputs.append({'data': 'fstate', 'df_label': 'Facade State %s',
                        'index': 'fzones'}) # bm, mid, up
    ctrlOutputs.append({'data': 'zone_temp', 'df_label': 'Temperature %s [C]',
                        'index': 'temps'}) # room, slab
    ctrlOutputs.append({'data': 'zone_temp_min', 'df_label': 'Temperature %s Min [C]',
                        'index': 'temps'}) # room, slab
    ctrlOutputs.append({'data': 'zone_temp_max', 'df_label': 'Temperature %s Max [C]',
                        'index': 'temps'}) # room, slab
    ctrlOutputs.append({'data': 'co2_profile_total', 'df_label': 'GHG Emissions [kg]'})

    for d in ctrlOutputs:
        d['name'] = d['data']
    return ctrlOutputs

"""
def convert_facades_model(model, parameter, old_names=False):

    columns = ['Window Illuminance [lx]','Artificial Illuminance [lx]','Power Lights [W]',
               'Solar Heat Gain [W]', 'Glare [-]','Room Temperature [C]',
               'Slab Temperature [C]','Outside Air Temperature [C]',
               'Power Heating [W]','Power Cooling [W]',
               'Room Temperature Max [C]','Slab Temperature Max [C]',
               'Room Temperature Min [C]','Slab Temperature Min [C]',
               'Actuation [-]','Actuation Abs [-]',
               'Glare Max [-]','Glare Penalty [-]','View Penalty [-]',
               'Work Plane Illuminance Min [lx]','Power Plugs [W]','Power Occupancy [W]',
               'Work Plane Illuminance [lx]',
               'Window Absorption 1 [W]','Window Absorption 2 [W]','Window Transmitted Solar [W]'
               'Convective Internal Gains [W]','Radiative Slab Gains [W]',
               'Radiative Wall Gains [W]', 'Power Equipment [W]']
    if model.fzones[-1] == 2 and old_names:
        columns += ['Tint Bottom [-]','Tint Middle [-]','Tint Top [-]']
    else:
        for z in model.fzones:
            columns += [f'Facade State {z}']
    if parameter['zone']['param']['type'] in ['R5C3', 'R6C3']:
        columns += ['Wall Temperature [C]','Wall Temperature Max [C]','Wall Temperature Min [C]']

    df = {}
    for t in model.ts:
        df[t] = [model.zone_wpi[t].value, model.zone_wpi_ext[t].value, model.p_lights[t].value,
                 model.zone_shg[t].value,
                 model.zone_glare[t].value, model.zone_temp[t,0].value, model.zone_temp[t,1].value,
                 model.outside_temperature[t], model.p_heating[t].value, model.p_cooling[t].value,
                 model.zone_temp_max[t,0], model.zone_temp_max[t,1], model.zone_temp_min[t,0],
                 model.zone_temp_min[t,1], model.zone_actuation[t].value,
                 model.zone_abs_actuation[t].value, model.glare_constraint_max[t],
                 model.zone_glare_penalty[t].value, model.zone_view_penalty[t].value,
                 model.wpi_constraint_min[t], model.zone_occload[t],
                 model.zone_plugload[t], model.zone_wpi_total[t].value, model.zone_abs1[t].value,
                 model.zone_abs2[t].value, model.zone_tsol[t].value, model.zone_qi[t].value,
                 model.zone_qs[t].value, model.zone_qw[t].value, model.p_equipment[t]]
        for z in model.fzones:
            df[t] += [model.fstate[t,z].value]
        if parameter['zone']['param']['type'] in ['R5C3', 'R6C3']:
            df[t] += [model.zone_temp[t,2].value,
                      model.zone_temp_max[t,2],
                      model.zone_temp_min[t,2]]

    df = pd.DataFrame(df).transpose()
    df.columns = columns
    df.index = pd.to_datetime(df.index, unit='s')

    return df
    
def pyomo_to_pandas(model, parameter):
    '''
        Utility to translate optimization output to a dataframe.

        Input
        -----
            model (pyomo.environ.ConcreteModel): The optimized model to be translated.
            parameter (dict): Configuration dictionary for the optimization.

        Returns
        -------
            df (pandas.DataFrame): A dataframe including the results of the optimization.
    '''
    df = convert_base_model(model, parameter)
    df = pd.concat([df, convert_battery(model, parameter)], axis=1)
    df = pd.concat([df, convert_facades_model(model, parameter)], axis=1)
    return df

def plot_streams(axs, temp, plot_times=False, title=None, ylabel=None, legend=False, loc=1):
    '''
        Utility to simplify plotting of subplots.

        Input
        -----
            axs (matplotlib.axes._subplots.AxesSubplot): The axis to be plotted.
            temp (pandas.Series): The stream to be plotted.
            plot_times (bool): Flag if time separation should be plotted. (default=True)
    '''
    axs.plot(temp)
    axs.legend(temp.columns, loc=2)
    if plot_times:
        idx0 = temp.index[temp.index.minute==0]
        axs.plot([idx0[idx0.hour==8],idx0[idx0.hour==8]],[temp.values.min(),temp.values.max()], color='orange', linestyle=':')
        axs.plot([idx0[idx0.hour==12],idx0[idx0.hour==12]],[temp.values.min(),temp.values.max()], color='red', linestyle=':')
        axs.plot([idx0[idx0.hour==18],idx0[idx0.hour==18]],[temp.values.min(),temp.values.max()], color='red', linestyle=':')
        axs.plot([idx0[idx0.hour==22],idx0[idx0.hour==22]],[temp.values.min(),temp.values.max()], color='orange', linestyle=':')
        if temp.values.min() < 0 and temp.values.max() > 0:
            axs.plot([idx0[0],idx0[-1]],[0,0], color='black', linestyle=':')
    if title: axs.set_title(title)
    if ylabel: axs.set_ylabel(ylabel)
    if legend: axs.legend(legend, loc=loc)

def plot_standard2(df, plot=True, plot_times=True, tight=True):
    '''
        A standard plotting template to present results.

        Input
        -----
            df (pandas.DataFrame): The resulting dataframe with the optimization result.
            plot (bool): Flag to plot or return the figure. (default=True)
            plot_times (bool): Flag if time separation should be plotted. (default=True)
            tight (bool): Flag to use tight_layout. (default=True)
            
        Returns
        -------
            None if plot == True.
            else:
                fig (matplotlib figure): Figure of the plot.
                axs (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot): Axis of the plot.
    '''
    fig, axs = plt.subplots(4,1, figsize=(12, 4*3), sharex=True, sharey=False, gridspec_kw = {'width_ratios':[1]})
    axs = axs.ravel()
    plot_streams(axs[0], df[['Import Power [kW]','Export Power [kW]']], plot_times=plot_times)
    plot_streams(axs[1], df[['Battery Power [kW]','Load Power [kW]','PV Power [kW]','Internal Power [kW]']], plot_times=plot_times)
    plot_streams(axs[2], df[['Tariff Energy [$/kWh]']], plot_times=plot_times)
    plot_streams(axs[3], df[['Battery SOC [-]']], plot_times=plot_times)
    #plot_streams(axs[4], df[['Reg Up [kW]','Reg Dn [kW]']], plot_times=plot_times)
    #plot_streams(axs[5], df[['Tariff Reg Up [$/kWh]','Tariff Reg Dn [$/kWh]']], plot_times=plot_times)
    if plot:
        if tight:
            plt.tight_layout()
        plt.show()
    else: return fig, axs

def plot_battery1(df, model, plot=True, tight=True):
    '''
        A standard plotting template to present results.

        Input
        -----
            df (pandas.DataFrame): The resulting dataframe with the optimization result.
            plot (bool): Flag to plot or return the figure. (default=True)
            plot_times (bool): Flag if time separation should be plotted. (default=True)
            tight (bool): Flag to use tight_layout. (default=True)
            
        Returns
        -------
            None if plot == True.
            else:
                fig (matplotlib figure): Figure of the plot.
                axs (numpy.ndarray of matplotlib.axes._subplots.AxesSubplot): Axis of the plot.
    '''
    fig, axs = plt.subplots(5,1, figsize=(12, 5*3), sharex=True, sharey=False, gridspec_kw = {'width_ratios':[1]})
    axs = axs.ravel()
    plot_streams(axs[0], df[['Battery Power [kW]','Load Power [kW]','PV Power [kW]']], \
                 title='Overview', ylabel='Power [kW]\n(<0:supply; >0:demand)', \
                 legend=['Battery','Load','PV'])
    plot_streams(axs[1], df[['Battery Power [kW]']+['Battery {!s} Power [kW]'.format(b) for b in model.batteries]], \
                 title='Battery Utilization', ylabel='Power [kW]\n(<0:discharge; >0:charge)', \
                 legend=['Total']+['Battery {!s}'.format(b) for b in model.batteries])
    plot_streams(axs[2], df[['Battery SOC [-]']+['Battery {!s} SOC [-]'.format(b) for b in model.batteries]]*100, \
                 title='Battery State of Charge', ylabel='SOC [%]', \
                 legend=['Total']+['Battery {!s}'.format(b) for b in model.batteries])
    plot_streams(axs[3], df[['Battery Avilable [-]']+['Battery {!s} Available [-]'.format(b) for b in model.batteries]], \
             title='Battery Availability', ylabel='Availability [-]\n(0:False; 1:True)', \
             legend=['Total']+['Battery {!s}'.format(b) for b in model.batteries])
    plot_streams(axs[4], df[['Battery External [kW]']+['Battery {!s} External [kW]'.format(b) for b in model.batteries]], \
             title='Battery External Demand', ylabel='Power [kW]', \
             legend=['Total']+['Battery {!s}'.format(b) for b in model.batteries])
    #plot_streams(axs[5], df[['Temperature [C]']+['Battery {!s} Temperature [C]'.format(b) for b in model.batteries]], \
    #         title='Battery Temperature', ylabel='Temperature [C]', \
    #         legend=['Outside']+['Battery {!s}'.format(b) for b in model.batteries])
    if plot:
        if tight:
            plt.tight_layout()
        plt.show()
    else: return fig, axs

'''
if __name__ == '__main__':
    solver_dir = 'solvers'
    parameter = dafault_parameter_laafb_report1()
    data = example_inputs_laafb_report1(parameter)
    parameter['objective']['weight_energy'] = 30
    del parameter['objective']['weight_degradation']

    smartDER = DOPER(model=control_model,
                     parameter=parameter,
                     solver_path=get_solver('cbc', solver_dir=solver_dir),
                     pyomo_to_pandas=pyomo_to_pandas)
    res = smartDER.do_optimization(data, tee=False)
    duration, objective, df, model, result, termination, parameter = res
    print(standard_report(res))
'''
"""
