"""
AFC example1 test module.
"""

import os
import sys
import time
import json
import pprint
import warnings
import numpy as np
import pandas as pd
import datetime as dtm

root = os.path.dirname(os.path.abspath(__file__))
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Improts from the AFC package
from afc.ctrlWrapper import Controller, make_inputs
from afc.utility.weather import read_tmy3
from afc.radiance.configs import get_config
from afc.defaultConfig import default_parameter
from afc.utility.plotting import plot_standard1

def example1():
    """
    This is an example of the AFC functionality.
    """
    
    print('Running AFC Example1...')
    print('Configuration: three zone electrochromic window')

    # read weather (forecast) data
    # this would normally come from a weather forecast module
    weather_path = os.path.join(os.path.dirname(root), 'dev', 'resources', 'weather', 
        'USA_CA_San.Francisco.Intl.AP.724940_TMY3.csv')
    weather, info = read_tmy3(weather_path, coerce_year=2023)
    #weather = weather.resample('5T').interpolate()
    st = dtm.datetime(2023, 7, 1)
    wf = weather.loc[st:st+pd.DateOffset(hours=24),]
    df = wf[['temp_air','dni','dhi','wind_speed']].copy()
    df = df[df.index.date == df.index[0].date()]

    # Initialize controller
    ctrl = Controller()

    # Make inputs
    parameter = default_parameter(precompute_radiance=False)
    inputs = make_inputs(parameter, df)

    # Query controller
    #log = ctrl.do_step(inputs=inputs) # first run to initialize
    log = ctrl.do_step(inputs=inputs) # run controller
    
    # Print results
    print(f'\nLog-message:\n{log}')
    #print('Duration:')
    #pprint.pprint(ctrl.get_output(keys=['rad_duration','varts_duration','optall_duration','glare_duration',
    #                                    'opt_duration','outputs_duration','duration']))
    #print('Optimization:')
    #pprint.pprint(ctrl.get_output(keys=['opt_objective','opt_duration','opt_termination','duration']))
    df = pd.DataFrame(ctrl.get_output(keys=['df_output'])['df_output'])
    df.index = pd.to_datetime(pd.to_numeric(df.index), unit='ms')
    print('Facade actuation during the day (when DNI > 0).')
    print('Facade 0 = bottom zone, Facade 1 = middle zone, Facade 2 = top zone')
    print('State 0.0 = fully tinted, State 1.0 and 2.0 = intermediate tint, state 3.0 = clear (double low-e)\n')
    print(df[['Facade State 0', 'Facade State 1', 'Facade State 2']][df['dni'] > 0])


if __name__ == '__main__':
    example1()