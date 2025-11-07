# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Reduced-order RC tuning configuration.
"""

# pylint: disable=too-many-arguments, invalid-name, too-many-positional-arguments
# pylint: disable=too-many-branches, too-many-statements, use-a-generator

# description of inputs
INPUTS_DESCRIPTION = {'outside_temperature': 'Outside Air Temperature [C]',
                      'wind_speed': 'Outside Wind Speed [m/s]',
                      'zone_qi': 'Zonal Convective Heat Gain [W]',
                      'zone_qw': 'Zonal Radiative Heat Gain on Walls [W]',
                      'zone_qs': 'Zonal Radiative Heat Gain on Slab [W]',
                      'zone_absf': 'Zonal Extwall Absorption [W]',
                      'zone_abs1': 'Zonal Window Absorption at Outer Glass Layer [W]',
                      'zone_abs2': 'Zonal Window Absorption at Inner Glass Layer [W]',
                      'zone_troom': 'Zonal Room Temperature [C]',
                      'zone_twall': 'Zonal Wall Temperature [C]',
                      'zone_tslab': 'Zonal Slab Temperature [C]',
                      'zone_textw': 'Zonal Extwall Temperature [C]',
                      }

TEMPERATURES = ['zone_troom', 'zone_twall', 'zone_tslab', 'zone_textw']

TUNING_MODES = {'R1C1': ['allAir', 'noSol'],
                'R2C2': ['allAir', 'noSol', 'allSplit'],
                'R4C2': ['allAir', 'allSplit'],
                'R5C2': ['allAir', 'allSplit'],
                'R5C3': ['onlyRs', 'allSplit'],
                'R6C3': ['onlyRs', 'allSplit'],
                'R7C4': ['allSplit'],
                }

def default_parameter(rctype='R4C2', fixed_air_valume=None):
    '''default optmization parameter'''

    parameter = {}

    # wrapper
    parameter['solver_path'] = 'ipopt'
    parameter['solver_options'] = {'max_cpu_time': 5*60}
    parameter['output_list'] = make_output_list(rc_type=rctype)

    # rc parameter
    parameter['objective'] = {'weight_troom': 1}
    parameter['temps_initial'] = []

    parameter['type'] = rctype
    parameter['Rw2i'] = {'init': 1e-4, 'lb': 1e-7, 'ub': 1e-2}
    parameter['Ci'] = {'init': 1e6, 'lb': 1e5, 'ub': 1e7}
    # two mass (wall)
    if rctype in ['R2C2', 'R4C2', 'R5C2', 'R5C3', 'R6C3', 'R7C4']:
        parameter['Riw'] = {'init': 1e-2, 'lb': 1e-4, 'ub': 1e-1}
        parameter['Cw'] = {'init': 1e8, 'lb': 1e7, 'ub': 1e10}
    # three mass (slab)
    if rctype in ['R5C3', 'R6C3', 'R7C4']:
        parameter['Ris'] = {'init': 1e3, 'lb': 1e3, 'ub': 1e3}
        parameter['Cs'] = {'init': 5e5, 'lb': 1e5, 'ub': 5e6}
    # four mass (ext wall)
    if rctype in ['R7C4']:
        parameter['Rof'] = {'init': 1e-1, 'lb': 1e-3, 'ub': 1e1}
        parameter['Rfi'] = {'init': 1e-1, 'lb': 1e-3, 'ub': 1e1}
        parameter['Cf'] = {'init': 5e5, 'lb': 1e4, 'ub': 5e6}
    # window system
    if rctype in ['R4C2', 'R5C2', 'R5C3', 'R6C3', 'R7C4']:
        parameter['Row1'] = {'init': 1e-3, 'lb': 1e-4, 'ub': 1e-2}
        parameter['Rw1w2'] = {'init': 1e-2, 'lb': 1e-5, 'ub': 1e-1}
    # exterior walls
    if rctype in ['R5C2', 'R6C3']:
        parameter['Roi'] = {'init': 1e-1, 'lb': 1e-3, 'ub': 1e2}

    if fixed_air_valume:
        vol = float(fixed_air_valume) # m3
        d = 1.1644 # kg/m3 @ 303 K
        cp = 1.005 # kJ/kg.K @ 300 K
        c = vol * d * cp * 1e3 # J/K
        parameter['Ci'] = {'init': c, 'lb': round(c-1e4,-4), 'ub': round(c+1e4,-4)}

    return parameter

def sum_inputs(inputs, cols):
    '''sum and zero out multiple inputs'''
    for k, v in cols.items():
        inputs[k] = inputs[[k]+v].sum(axis=1)
        for c in v:
            inputs[c] = 0

def get_rc_parameter(inputs, inputs_0, rctype='R4C2', mode='sepSlab', has_windows=True):
    '''default rc parameters
        rctype: see TUNING_MODES.keys()
        mode: see TUNING_MODES
        has_windows (bool): zone has windows
    '''

    # get default rc parameter
    parameter = default_parameter(rctype=rctype)

    # initial temperature
    parameter['temps_initial'] = [float(inputs_0['zone_troom'].values[0])]
    if 'C2' in rctype:
        parameter['temps_initial'].append(float(inputs_0['zone_tslab'].values[0]))
    elif 'C3' in rctype:
        parameter['temps_initial'].append(float(inputs_0['zone_tslab'].values[0]))
        parameter['temps_initial'].append(float(inputs_0['zone_twall'].values[0]))
    elif 'C4' in rctype:
        parameter['temps_initial'].append(float(inputs_0['zone_tslab'].values[0]))
        parameter['temps_initial'].append(float(inputs_0['zone_twall'].values[0]))
        parameter['temps_initial'].append(float(inputs_0['zone_textw'].values[0]))

    # 1 mass sytem (no window)
    if rctype in ['R1C1']:
        if mode == 'allAir':
            # all internal and inner solar gains in air
            parameter['Ci'] = {'init': 1e7, 'lb': 1e6, 'ub': 1e10}
            sum_inputs(inputs, {'zone_qi': ['zone_abs2']})
        elif mode == 'noSol':
            # all internal and inner solar gains in air
            parameter['Ci'] = {'init': 1e7, 'lb': 1e6, 'ub': 1e10}
        else:
            raise ValueError(f'Mode {mode} not spooprted for {rctype}.')
    # 2 mass sytem (no window)
    elif rctype in ['R2C2']:
        if mode == 'allAir':
            # all internal and inner solar gains in air
            parameter['Ci'] = {'init': 1e7, 'lb': 1e6, 'ub': 1e9}
            sum_inputs(inputs, {'zone_qi': ['zone_qw', 'zone_abs2']})
        elif mode == 'noSol':
            # all internal and inner solar gains in air
            parameter['Ci'] = {'init': 1e7, 'lb': 1e6, 'ub': 1e10}
            sum_inputs(inputs, {'zone_qi': ['zone_qw']})
        # elif mode == 'sepWall':
        #     # inernal split but solar in air
        #     sum_inputs(inputs, {'zone_qi': ['zone_abs2']})
        # elif mode == 'sepAir':
        #     # inernal split but solar to wall
        #     sum_inputs(inputs, {'zone_qw': ['zone_abs2']})
        elif mode == 'allSplit':
            pass
        else:
            raise ValueError(f'Mode {mode} not spooprted for {rctype}.')
    # 2 mass sytem (with window)
    elif rctype in ['R4C2', 'R5C2']:
        if mode == 'allAir':
            # all internal gains in air
            parameter['Ci'] = {'init': 1e7, 'lb': 1e6, 'ub': 1e9}
            sum_inputs(inputs, {'zone_qi': ['zone_qw']})
        elif mode == 'allSplit':
            pass
        else:
            raise ValueError(f'Mode {mode} not spooprted for {rctype}.')
    # 3 mass sytem (with window)
    elif rctype in ['R5C3', 'R6C3']:
        if mode == 'onlyRs':
            # only r value for walls no C (qs = 0)
            parameter['Ris'] = {'init': 1, 'lb': 1e-3, 'ub': 1e3}
            parameter['Cs'] = {'init': 0, 'lb': 0, 'ub': 0}
            inputs['zone_qs'] = 0
        elif mode == 'allSplit':
            pass
        else:
            raise ValueError(f'Mode {mode} not spooprted for {rctype}.')
    # 4 mass sytem (with window)
    elif rctype in ['R7C4']:
        if mode == 'allSplit':
            pass
        else:
            raise ValueError(f'Mode {mode} not spooprted for {rctype}.')
    else:
        raise ValueError(f'Type {rctype} not supported.')

    # no windows (high R for all window)
    if not has_windows:
        if 'Row1' in parameter:
            parameter['Row1'] = {'init': 1e-7, 'lb': 1e-7, 'ub': 1e-7}
        if 'Rw1w2' in parameter:
            parameter['Rw1w2'] = {'init': 1e-7, 'lb': 1e-7, 'ub': 1e-7}
        if 'Rw2i' in parameter:
            parameter['Rw2i'] = {'init': 1e7, 'lb': 1e7, 'ub': 1e7}

    return parameter

def make_output_list(rc_type='R4C2'):
    '''make doper output list'''

    column_map = {
        'Error Room Temperature [C]': 'diff_troom',
        'Error Slab Temperature [C]': 'diff_tslab',
        'Outside Air Temperature [C]': 'outside_temperature',
        'Convective Internal Gains [W]': 'zone_qi',
        'Measured Room Temperature [C]': 'zone_troom',
        }
    # two mass
    if 'C2' in rc_type:
        column_map['Measured Wall Temperature [C]'] = 'zone_twall'
        column_map['Radiative Internal Gains [W]'] = 'zone_qw'
    # three mass
    elif 'C3' in rc_type:
        column_map['Measured Wall Temperature [C]'] = 'zone_twall'
        column_map['Measured Slab Temperature [C]'] = 'zone_tslab'
        column_map['Radiative Internal Gains [W]'] = 'zone_qw'
        column_map['Radiative Slab Gains [W]'] = 'zone_qs'
    # four mass
    elif 'C3' in rc_type:
        column_map['Measured Wall Temperature [C]'] = 'zone_twall'
        column_map['Measured Slab Temperature [C]'] = 'zone_tslab'
        column_map['Measured Extw Temperature [C]'] = 'zone_textw'
        column_map['Radiative Internal Gains [W]'] = 'zone_qw'
        column_map['Radiative Slab Gains [W]'] = 'zone_qs'
        # column_map['Radiative Extw Gains [W]'] = 'zone_qf'
    # add window system
    if any([rc_type.startswith(r) for r in ['R4', 'R5', 'R6', 'R7']]):
        column_map['Window Absorption 1 [W]'] = 'zone_abs1'
        column_map['Window Absorption 2 [W]'] = 'zone_abs2'
    # add exterior wall
    if any([rc_type.startswith(r) for r in ['R7']]):
        column_map['Extwall Absorption 1 [W]'] = 'zone_absf'

    output_list = []
    for k, v in column_map.items():
        output_list.append({'data': v, 'df_label': k})

    output_list.append({'data': 'zone_temp',
                        'index': 'temps',
                        'df_label': 'Temperature %s [C]'})

    return output_list
