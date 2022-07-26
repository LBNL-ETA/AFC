"""
AFC example1 test module.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import datetime as dtm

root = os.path.dirname(os.path.abspath(__file__))
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from afc.ctrlWrapper import Controller
from afc.utility.weather import read_tmy3
from afc.radiance.configs import get_config
from afc.defaultConfig import default_parameter
from afc.utility.plotting import plot_standard1

def test1():
    """
    This is test1 to test the AFC functionality.
    """
    
    # read weather (forecast) data
    weather_path = os.path.join(os.path.dirname(root), 'dev', 'resources', 'weather', 
        'USA_CA_San.Francisco.Intl.AP.724940_TMY3.csv')
    weather, info = read_tmy3(weather_path)
    weather = weather.resample('5T').interpolate()
    st = dtm.datetime(dtm.datetime.now().year, 7, 1)
    wf = weather.loc[st:st+pd.DateOffset(hours=24),]
    df = wf[['DryBulb','DNI','DHI','Wspd']].copy()
    df = df[df.index.date == df.index[0].date()]
    
    # Initialize controller
    ctrl = Controller()
    
    # Get all variables
    print('All Input variables:', ctrl.get_model_variables())
    
    # Provide some wf input
    #st = weather.index[int(len(weather.index)/2)].date()

    
    print('OVERWRITE')
    df['DryBulb'] -= 40
    
    # Provide some inputs
    df['wpi_min'] = 500
    df['glare_max'] = 0.4
    df['generation_pv'] = 0
    df['load_demand'] = 0
    df['temp_room_max'] = 24
    df['temp_room_min'] = 20
    df['temp_slab_max'] = 23
    df['temp_slab_min'] = 21
    df['temp_wall_max'] = 40
    df['temp_wall_min'] = 10
    df['plug_load'] = 10 # W
    df['occupant_load'] = 15 # W
    df['equipment'] = 10 # W
    df.index = (df.index.astype(np.int64) / 10 ** 6).astype(str)
    #inputs = inputs[testcontroller.dfinputs].to_dict()
    
    # Inputs object
    inputs = {}
    inputs['radiance'] = {'regenerate': False, 'wwr': 0.4, 'wpi_loc': '23back'}
    inputs['radiance']['location'] = {'latitude': 37.7, 'longitude': 122.2, 'view_orient': 0,
                                      'timezone': 120, 'orient': 0, 'elevation': 100}
    inputs['df_input'] = df.to_dict()
    inputs['wf_all'] = df[['DNI','DHI']].to_dict()
    inputs['facade_initial'] = [3, 3, 3]
    inputs['temps_initial'] = [22, 22, 22]
    inputs['parameter'] = default_parameter()
    #inputs['parameter']['wrapper']['resample_variable_ts'] = False
    inputs['parameter']['wrapper']['precompute_radiance'] = True
    filestruct, rad_config = get_config('ec', str(inputs['radiance']['wwr']), root=os.path.join(os.path.dirname(root), 'dev', 'resources', 'radiance'))
    inputs['paths'] = {'rad_config': rad_config}
    inputs['paths']['rad_bsdf'] = filestruct['resources']
    inputs['paths']['rad_mtx'] = filestruct['matrices']
    
    # Query controller
    ctrl.do_step(inputs=inputs) # Initialize
    print('Log-message:\n', ctrl.do_step(inputs=inputs))
    print('Duration:\n', ctrl.get_output(keys=['rad_duration','varts_duration','optall_duration','glare_duration',
                                                       'opt_duration','outputs_duration','duration']))
    print('Optimization:\n', ctrl.get_output(keys=['opt_objective','opt_duration','opt_termination','duration']))
    df = pd.DataFrame(ctrl.get_output(keys=['df_output'])['df_output'])
    df.index = pd.to_datetime(df.index, unit='ms')
    
    
    # check
    res = ctrl.get_output(keys=['opt_objective','opt_duration','opt_termination','duration'])
    assert int(res['opt_objective']*1e1)/1e1 == 25.9
    assert res['opt_duration'] < 1
    assert res['opt_termination'] == 'optimal'
    assert res['duration'] < 2
