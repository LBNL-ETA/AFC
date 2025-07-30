# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Default facade mappings.
"""

import itertools

# Conversion maps
shade_map_0x6 = {
    '[0, 0, 0]': [0,0,0,0,0,0], # 100 % dn
    '[1, 0, 0]': [1,0,0,0,0,0], # 80 %
    '[2, 0, 0]': [1,1,0,0,0,0], # 60 %
    '[2, 1, 0]': [1,1,1,0,0,0], # 40 %
    '[2, 2, 0]': [1,1,1,1,0,0], # 33 %
    '[2, 2, 1]': [1,1,1,1,1,0], # 13 %
    '[2, 2, 2]': [1,1,1,1,1,1]} # 0 % dn
height_shade_map_0x6 = [100, 80, 60, 40, 33, 13, 0]

shade_map_0x4 = {
    '[0, 0, 0]': [0,0,0,0,0], # 100 % dn
    '[1, 0, 0]': [1,0,0,0,0], # 80 %
    '[2, 0, 0]': [1,1,0,0,0], # 60 %
    '[2, 1, 0]': [1,1,1,0,0], # 40 %
    '[2, 2, 0]': [1,1,1,1,0], # 33 %
    '[2, 2, 2]': [1,1,1,1,1]} # 0 % dn
height_shade_map_0x4 = [100, 80, 60, 40, 33, 0]

blinds_map = {
    '[1, 1, 1]': [12], # clear
    '[1, 0, 0]': [1],  # -15
    '[2, 0, 0]': [2],  # -30
    '[3, 0, 0]': [3],  # -45
    '[4, 0, 0]': [4],  # -60
    '[5, 0, 0]': [5],  # -75
    '[1, 0, 1]': [6],  # 0
    '[0, 0, 1]': [7],  # 15
    '[0, 0, 2]': [8],  # 30
    '[0, 0, 3]': [9],  # 45
    '[0, 0, 4]': [10], # 60
    '[0, 0, 5]': [11], # 75
    '[0, 0, 0]': [0]}  # 90
#height_blinds_map = [0, 30, 60, 100, 100, 100, 0, 30, 60, 100, 100, 100, 100]
height_blinds_map = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]

def generate_dependent_combinations_table(variables, values):
    """make progressive table for shades."""
    if not values:
        return []

    table = []

    def generate_row(index, current_row):
        """process row"""
        if index == variables:
            table.append(current_row[:])
            return
        for value in values:
            if not current_row or value >= current_row[-1]:
                current_row.append(value)
                generate_row(index + 1, current_row)
                current_row.pop()

    generate_row(0, [])
    return table

def make_ctrl_map(mode, states, window_subdivs, window_zones):
    """make the control map."""
    # states per window
    states_per_window = dict(enumerate(states))

    # window actuation map
    window_act_map = {}
    # for window in range(window_zones if mode == 'ec' else 1):
    for window in range(window_subdivs):
        window_act_map[window] = list(states_per_window.values())

    # window control map
    window_ctrl_map = {}
    if mode == 'ec':
        windows_per_zone = int(window_subdivs / window_zones)
        logic_map = list(itertools.product(*([states] * window_zones)))
        ctrl_msp = [list(itertools.chain.from_iterable([[ss]*windows_per_zone for ss in l]))
            for l in logic_map]
    elif mode == 'shade':
        ctrl_msp = generate_dependent_combinations_table(window_subdivs, states_per_window.keys())
        ctrl_msp = [[states_per_window[v] for v in l] for l in ctrl_msp[::-1]]
    elif mode in ['blinds', 'tc']:
        windows_per_zone = window_subdivs
        logic_map = list(itertools.product(*([states] * 1)))
        ctrl_msp = [list(itertools.chain.from_iterable([[ss]*windows_per_zone for ss in l]))
            for l in logic_map]
    else:
        raise ValueError(f'Glazing system not implemented: {mode}')

    window_ctrl_map = dict(enumerate(ctrl_msp))
    return window_ctrl_map
