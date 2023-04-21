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

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    from afc.ctrlWrapper import Controller, make_inputs
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

    # Make inputs
    parameter = default_parameter(precompute_radiance=False)
    inputs = make_inputs(parameter, df)

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
    assert int(res['opt_objective']*1e1)/1e1 == 19.3
    assert res['opt_duration'] < 5
    assert res['opt_termination'] == 'optimal'
    assert res['duration'] < 60*5
