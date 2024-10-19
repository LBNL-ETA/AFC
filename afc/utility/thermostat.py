# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Thermostat control module.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

def afc_to_hvac_setpoint(ctrl_outputs, tdead=0.5):
    """Utility to convert from AFC to thermostat setpoints."""

    # no control by default
    hvac_mode = 'X'
    new_cool_set = float(ctrl_outputs['cool_set'])
    new_heat_set = float(ctrl_outputs['heat_set'])

    if ctrl_outputs['hvac_control'] and ctrl_outputs['feasible']:

        # float by default
        hvac_mode = 'F'

        # get new setpoints if not occupied (make sure within bounds)
        if not ctrl_outputs['occupied']:

            # cooling
            if ctrl_outputs['power_cool'] > 1:
                hvac_mode = 'C'
                new_cool_set = min(ctrl_outputs['cool_set'],
                                   ctrl_outputs['t_room'])
                # make sure cool > heat
                new_cool_set = max(new_cool_set, new_heat_set + tdead)

            # heating
            elif ctrl_outputs['power_heat'] > 1:
                hvac_mode = 'H'
                new_heat_set = max(ctrl_outputs['heat_set'],
                                   ctrl_outputs['t_room'])
                # make sure heat < cool
                new_heat_set = min(new_heat_set, new_cool_set - tdead)

    return {'feasible': ctrl_outputs['feasible'],
            'mode': hvac_mode,
            'csp': new_cool_set,
            'hsp': new_heat_set}

def compute_thermostat_setpoints(df, cool_set, heat_set, feasible, control_hvac, occupied):
    """wrapper to compute thermostat setpoints"""

    if feasible:
        t_room = df['Temperature 0 [C]'].values[1]
        power_cool = df['Power Cooling [W]'].iloc[0]
        power_heat = df['Power Heating [W]'].iloc[0]
    else:
        t_room = 0
        power_cool = 0
        power_heat = 0

    ctrl_outputs = {
        'hvac_control': control_hvac,
        'feasible': feasible,
        't_room': t_room,
        'cool_set': cool_set,
        'heat_set': heat_set,
        'occupied': occupied,
        'power_cool': power_cool,
        'power_heat': power_heat,
    }

    return afc_to_hvac_setpoint(ctrl_outputs, tdead=0.5)
