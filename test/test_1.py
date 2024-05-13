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
    from afc.utility.weather import example_weather_forecast
    from afc.radiance.configs import get_config
    from afc.defaultConfig import default_parameter
    from afc.utility.plotting import plot_standard1

def test1():
    """
    This is test1 to test the AFC functionality.
    """

    # read weather (forecast) data
    wf = example_weather_forecast(date='2023-07-01')
    df = wf[wf.index.date == wf.index[0].date()]

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
    assert 20.0 < res['opt-stats']['objective'] < 20.5
    assert res['opt-stats']['duration'] < 1
    assert res['opt-stats']['termination'] == 'optimal'
    assert res['duration']['all'] < 60*5
