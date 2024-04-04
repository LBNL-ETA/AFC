"""
AFC example1 test module.
"""

import io
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
    weather, info = read_tmy3(weather_path, coerce_year=2023)
    weather = weather.resample('5min').interpolate()
    st = dtm.datetime(2023, 7, 1)
    wf = weather.loc[st:st+pd.DateOffset(hours=24),]
    df = wf[['temp_air','dni','dhi','wind_speed']].copy()
    df = df[df.index.date == df.index[0].date()]

    # Initialize controller
    ctrl = Controller()

    # Make inputs
    parameter = default_parameter()
    inputs = make_inputs(parameter, df)

    # Query controller
    ctrl.do_step(inputs=inputs)
    df = pd.read_json(io.StringIO(ctrl.get_output(keys=['output-data'])['output-data']))

    # check
    res = ctrl.get_output(keys=['opt-stats', 'duration'])
    assert 19.5 < res['opt-stats']['objective'] < 20.5
    assert res['opt-stats']['duration'] < 5
    assert res['opt-stats']['termination'] == 'optimal'
    assert res['duration']['all'] < 60*5
