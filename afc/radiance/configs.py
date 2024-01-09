# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Default configuration paths for Radiance.
"""

# pylint: disable=redefined-outer-name

import os

root = os.path.dirname(os.path.abspath(__file__))

def get_config(mode, wwr, abs_path=True, root=os.path.join(root, '..',
                                                           'resources', 'radiance')):
    """Function to generate configuration."""

    filestruct = {}
    if mode == 'shade':
        config_path = os.path.join(root, f'room{wwr}WWR_shade.cfg')
        filestruct['resources'] = os.path.join(root, 'BSDF_shade')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'dshade':
        mode = 'shade'
        config_path = os.path.join(root, f'room{wwr}WWR_shade.cfg')
        filestruct['resources'] = os.path.join(root, 'BSDF_dshade')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'blinds':
        config_path = os.path.join(root, f'room{wwr}WWR_blinds.cfg')
        filestruct['resources'] = os.path.join(root, f'BSDF_blinds{wwr}')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'ec':
        config_path = os.path.join(root, f'room{wwr}WWR_ec.cfg')
        filestruct['resources'] = os.path.join(root, 'BSDFs')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    else:
        print(f'ERROR: Mode {mode} not supported.')
    if abs_path:
        config_path = os.path.abspath(config_path)
        filestruct['resources'] = os.path.abspath(filestruct['resources'])
        filestruct['matrices'] = os.path.abspath(filestruct['matrices'])
    return filestruct, config_path
