"""
AFC example1 test module.
"""

import io
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
from afc.utility.weather import example_weather_forecast
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
    wf = example_weather_forecast(date='2023-07-01')
    df = wf[wf.index.date == wf.index[0].date()]

    # Initialize controller
    ctrl = Controller()

    # Make inputs
    parameter = default_parameter()
    inputs = make_inputs(parameter, df)

    # Query controller
    log = ctrl.do_step(inputs=inputs) # run controller
    
    # Print results
    print(f'\nLog-message:\n{log}')
    #print('Duration:')
    #pprint.pprint(ctrl.get_output(keys=['rad_duration','varts_duration','optall_duration','glare_duration',
    #                                    'opt_duration','outputs_duration','duration']))
    #print('Optimization:')
    #pprint.pprint(ctrl.get_output(keys=['opt_objective','opt_duration','opt_termination','duration']))
    df = pd.read_json(io.StringIO(ctrl.get_output(keys=['output-data'])['output-data']))
    print('Facade actuation during the day (when DNI > 0).')
    print('Facade 0 = bottom zone, Facade 1 = middle zone, Facade 2 = top zone')
    print('State 0.0 = fully tinted, State 1.0 and 2.0 = intermediate tint, state 3.0 = clear (double low-e)\n')
    print(df[['Facade State 0', 'Facade State 1', 'Facade State 2']][df['dni'] > 0])


if __name__ == '__main__':
    example1()