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

def compare_dataframes(df1, df2, tolerance=5):
    """
    Compare two pandas DataFrames to check if they are within a specified percentage of tolerance.

    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    tolerance (float): The percentage of tolerance.

    Returns:
    bool: True if the DataFrames are within the specified tolerance, False otherwise.
    """
    if df1.shape != df2.shape:
        return False

    for col in df1.columns:
        if col not in df2.columns:
            return False

        for idx in df1.index:
            val1 = df1.at[idx, col]
            val2 = df2.at[idx, col]

            if np.isnan(val1) and np.isnan(val2):
                continue

            if np.isnan(val1) or np.isnan(val2):
                return False

            if not np.isclose(val1, val2, rtol=tolerance / 100):
                return False

    return True

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
    print(ctrl.do_step(inputs=inputs))
    df_res = pd.read_json(io.StringIO(ctrl.get_output(keys=['output-data'])['output-data']))

    # check overall
    res = ctrl.get_output(keys=['opt-stats', 'duration'])
    assert 19.5 < res['opt-stats']['objective'] < 20.5
    assert res['opt-stats']['duration'] < 1
    assert res['opt-stats']['termination'] == 'optimal'
    assert res['duration']['all'] < 60*10

    # Compute frads/Radiance
    rad = ctrl.forecaster.compute2(df[['dni','dhi']])
    #rad.to_csv('frads_base.csv')

    # check radiance
    base = pd.read_csv(os.path.join(root, 'frads_base.csv'), index_col=0)
    if not compare_dataframes(base, rad, tolerance=5):
        rad.to_csv(os.path.join('frads.csv'))
    assert compare_dataframes(base, rad, tolerance=5)
    
def test2():
    """
    This is test2 to test the AFC functionality for different RC models.
    """

    # read weather (forecast) data
    wf = example_weather_forecast(date='2023-07-01')
    df = wf[wf.index.date == wf.index[0].date()]

    cases = []
    cases.append([{'type': 'R1C1', 'Rw2i': 0.1, 'Ci': 150e3}, ['room']])
    cases.append([{'type': 'R2C2', 'Rw2i': 0.1, 'Ci': 150e3, 'Ris': 0.01, 'Cs': 500e3},
                  ['room', 'slab']])

    for case in cases:
        print(case)
        
        # Initialize controller
        ctrl = Controller()

        # Make inputs
        parameter = default_parameter()
        parameter['zone']['param'] = case[0]
        parameter['zone']['temps_name'] = case[1]
        inputs = make_inputs(parameter, df)

        # Query controller
        print(ctrl.do_step(inputs=inputs))
        df_res = pd.read_json(io.StringIO(ctrl.get_output(keys=['output-data'])['output-data']))

        # check overall
        res = ctrl.get_output(keys=['opt-stats', 'duration'])
        assert res['opt-stats']['termination'] == 'optimal'

