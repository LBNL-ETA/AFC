# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Default configuration paths for Radiance.
"""

#pylint: disable=too-many-arguments, too-many-positional-arguments

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_config(mode, wwr, name_sys=None, abs_path=True,
               root_cfg=None, root_sys=None):
    """Function to generate configuration."""

    if not root_cfg:
        root_cfg = os.path.join(ROOT, '..', 'resources', 'radiance')

    if not root_sys:
        root_sys = os.path.join(ROOT, '..', 'resources', 'radiance')

    if not name_sys:
        name_sys = mode

    filestruct = {}
    if mode in ['shade', 'dshade', 'blinds', 'ec']:
        cfg_mode = mode.replace('dshade', 'shade')
        config_path = os.path.join(root_cfg, f'room{wwr}WWR_{cfg_mode}.cfg')
        #filestruct['resources'] = os.path.join(root, 'bsdf', mode)
        filestruct["glazing_systems"] = os.path.join(root_sys, "glazing_systems", name_sys)
        filestruct["matrices"] = os.path.join(root_sys, "matrices", name_sys, str(wwr))
    else:
        print(f"ERROR: Mode {mode} not supported.")
    if abs_path:
        config_path = os.path.abspath(config_path)
        #filestruct['resources'] = os.path.abspath(filestruct['resources'])
        filestruct["glazing_systems"] = os.path.abspath(filestruct["glazing_systems"])
        filestruct["matrices"] = os.path.abspath(filestruct["matrices"])
    return filestruct, config_path
