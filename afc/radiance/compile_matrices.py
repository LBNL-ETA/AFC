# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Compilation of matrices for Radiance.
"""

import os
from .configs import get_config
from .forecast import Forecast

root = os.path.dirname(os.path.abspath(__file__))

for wwr in [0.6, 0.4]:
    for mode in ['shade', 'dshade', 'blinds', 'ec']:
        print(f'===> Compiling Matrices for {wwr} WWR for "{mode}".')

        filestruct, config_path = get_config(mode, wwr)
        emulator = Forecast(config_path, regenerate=True, filestruct=filestruct)
