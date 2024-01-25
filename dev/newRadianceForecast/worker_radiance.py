import os
import sys
import json
import traceback
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

root = os.path.dirname(os.path.realpath(__file__))

# AFC
#root_controller = os.path.join(root, '..', '..', 'afc', 'radiance')
#sys.path.append(root_controller)
from afc.radiance import forecast as forecast
from afc.radiance import configs as mpc_config

# Emulator
root_emulator = os.path.join(root, '..', '..', '..', '..', 'PrivateRepos',
                             'DynamicFacades', 'emulator', 'emulator')
sys.path.append(root_emulator)
try:
    from radiance import radiance_emulator as radiance
    from radiance import configs as emu_config
    from radiance.radiance_emulator import shade_map_0x6, shade_map_0x4, blinds_map
    USE_EMU = True
except Exception as e:
    print('WANRNING: Did not load emulator')
    print(e)
    USE_EMU = False
    from afc.radiance.maps import shade_map_0x6, shade_map_0x4, blinds_map

def worker(inputs):
    
    wf, mode, wwr, ctrl_mode, control = inputs
    
    facade_type = mode
    if mode == 'dshade':
        facade_type = 'shade'

    data = pd.DataFrame()
    try:
        # Controller
        filestruct_mpc, config_path_mpc = mpc_config.get_config(mode, str(wwr))
        forecaster = forecast.Forecast(config_path_mpc, regenerate=False, facade_type=facade_type,
                                       filestruct=filestruct_mpc, wpi_loc="23back",
                                       _test_difference=True)
        data = forecaster.compute2(wf[['dni', 'dhi']])
        data = data.rename(columns={c:'ctrl_{}'.format(c) for c in data.columns})
        
        # Emulator
        if USE_EMU:
            filestruct_emu, config_path_emu = emu_config.get_config(mode, str(wwr))
            emulator = radiance.Radiance(config_path_emu, regenerate=False, facade_type=facade_type,
                                         filestruct=filestruct_emu, _test_difference=True)
            wf_emu = wf.rename(columns={'dni': 'weaHDirNor', 'dhi': 'weaHDifHor'})
            for ix in wf_emu.index:
                radiance_outputs = emulator.compute(wf_emu.loc[ix:ix].copy(deep=True), control, ix)
                for i, o in enumerate(radiance_outputs):
                    if type(o) == type(list()):
                        data.loc[ix, 'emu_{}'.format(i)] = o[0]
                        data.loc[ix, 'emu_{}-all'.format(i)] = str(o[1])
                    else:
                        data.loc[ix, 'emu_{}'.format(i)] = o
                        
            # Post Process
            data['emu_wpi'] = data['emu_0']
            data['emu_glare'] = data['emu_1'].astype(float)
            data['emu_abs1'] = data['emu_7']
            data['emu_abs2'] = data['emu_8']
            data['emu_tsol'] = data[['emu_2','emu_3','emu_4','emu_5','emu_6']].sum(axis=1)
            data['emu_iflr'] = data['emu_6']
        
        if mode in ['shade', 'dshade']:
            if wwr == 0.4:
                control = shade_map_0x4[str(control)]
            elif wwr == 0.6:
                control = shade_map_0x6[str(control)]
            else:
                print('ERROR: WWR of {} not defined.'.format(wwr))
        elif mode == 'blinds':
            control = blinds_map[str(control)]
        post = ['_{}_{}'.format(i, t) for i, t in enumerate(control)]
        data['ctrl_wpi'] = data[['ctrl_wpi{}'.format(p) for p in post]].sum(axis=1)
        data['ctrl_ev'] = data[['ctrl_ev{}'.format(p) for p in post]].sum(axis=1)
        data['ctrl_glare'] = data['ctrl_ev'] * 6.22e-5 + 0.184
        data['ctrl_glare_lim'] = data['ctrl_glare'].mask(data['ctrl_glare']>1, 1)
        data['ctrl_abs1'] = data[['ctrl_abs1{}'.format(p) for p in post]].sum(axis=1)
        data['ctrl_abs2'] = data[['ctrl_abs2{}'.format(p) for p in post]].sum(axis=1)
        data['ctrl_tsol'] = data[['ctrl_tsol{}'.format(p) for p in post]].sum(axis=1)
        data['ctrl_iflr'] = data[['ctrl_iflr{}'.format(p) for p in post]].sum(axis=1)
        
    except Exception as e:
        print(inputs, control)
        print(traceback.format_exc())
        print(e)
                                 
        
    return {'{}-{}-{}'.format(mode, wwr, ctrl_mode): data}