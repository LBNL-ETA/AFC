import os
import sys
from configs import get_config
from forecast import Forecast

root = os.path.dirname(os.path.abspath(__file__))

for wwr in [0.6, 0.4]:
    for mode in ['shade', 'dshade', 'blinds', 'ec']:
        print('===> Compiling Matrices for {} WWR for "{}".'.format(wwr, mode))
        
        filestruct, config_path = get_config(mode, wwr)
        emulator = Forecast(config_path, regenerate=True, filestruct=filestruct)

