# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Weather handling module.
"""

# pylint: disable=bare-except, invalid-name, import-outside-toplevel

import os
import warnings
import datetime as dt
import pandas as pd
import pvlib

warnings.filterwarnings('ignore', message='The forecast module algorithms' + \
    ' and features are highly experimental. ')

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    print('WARNING: Using local directory as root.')
    root = os.getcwd()

def read_tmy3(filename=None, coerce_year=dt.datetime.now().year, convert_numeric=True):
    """Reads TMY data file."""
    weather, info = pvlib.iotools.read_tmy3(filename=filename,
                                            coerce_year=coerce_year,
                                            map_variables=True)
    weather.index = weather.index - pd.DateOffset(hours=1)
    weather.index = weather.index.tz_localize(None)
    weather.index.name = None

    # make sure numeric outputs
    if convert_numeric:
        for c in weather.columns:
            weather[c] = pd.to_numeric(weather[c], errors='coerce')

    return weather, info

def read_mos(filename, coerce_year=dt.datetime.now().year):
    """Reads .mos data file."""
    skiprows = 0
    header = []
    metadata = []
    with open(filename, 'r', encoding='utf8') as f:
        line = f.readline()
        while line[0] == '#':
            if line.startswith('#LOCATION'):
                metadata = line.split(',')[1:]
            if line.startswith('#COMMENTS'):
                line = '#' + line
            if line.startswith('#C'):
                header.append(line[len(line.split(' ')[0])+1:-1])
            skiprows += 1
            line = f.readline()
            if line.startswith('double'):
                line = '#' + line
    weather = pd.read_csv(filename, skiprows=skiprows, header=None,
                          names=header, sep='\t', index_col=False)
    column_map = {}
    column_map['Time in seconds. Beginning of a year is 0s.'] = 'time'
    column_map['index'] = 'time'
    # Custom
    #column_map['Dry bulb temperature in Celsius at indicated time'] = 'DryBulb'
    #column_map['Global horizontal radiation in Wh/m2'] = 'GHI'
    #column_map['Direct normal radiation in Wh/m2'] = 'DNI'
    #column_map['Diffuse horizontal radiation in Wh/m2'] = 'DHI'
    # Modelica
    column_map['Dry bulb temperature in Celsius at indicated time'] = 'weaTDryBul'
    column_map['Dew point temperature in Celsius at indicated time'] = 'weaTDewPoi'
    column_map['Relative humidity in percent at indicated time'] = 'weaRelHum'
    column_map['Atmospheric station pressure in Pa at indicated time'] = 'weaPAtm'
    column_map['Wind speed in m/s at indicated time'] = 'weaWinSpe'
    column_map['Wind direction at indicated time. N=0, E=90, S=180, W=270'] = 'weaWinDir'
    column_map['Ceiling height in m'] = 'weaCelHei'
    column_map['Horizontal infrared radiation intensity in Wh/m2'] = 'weaHHorIR'
    column_map['Total sky cover at indicated time'] = 'weaNTot'
    column_map['Opaque sky cover at indicated time'] = 'weaNOpa'
    column_map['Global horizontal radiation in Wh/m2'] = 'weaHGloHor'
    column_map['Direct normal radiation in Wh/m2'] = 'weaHDirNor'
    column_map['Diffuse horizontal radiation in Wh/m2'] = 'weaHDifHor'
    weather = weather.rename(columns=column_map)
    start_time = pd.to_datetime(f'01-01-{coerce_year}')
    weather.index = [start_time + \
        pd.DateOffset(seconds=t) for t in weather['time']]
    return weather, metadata

def get_forecast(st=dt.datetime.now(), tz=-8,
                 loc=None, model=None):
    """Gathers latest weather forecast from NOAA."""
    if not loc:
        loc = {'latitude':37.617, 'longitude':-122.4}
    if not model:
        print('ERROR: No model supplied.')
    tz = f'Etc/GMT+{int(tz*-1)}'
    tz = None
    print('WARNING: "tz" fixed to None!')
    st = pd.Timestamp(st, tz=tz).replace(second=0,
                                         microsecond=0)
    et = st + pd.Timedelta(days=2)
    data = model.get_processed_data(loc['latitude'],
                                    loc['longitude'], st, et)
    data.index = data.index.tz_localize(None)
    return data, model

if __name__ == '__main__':
    import time
    root_repo = os.path.dirname(root)

    print('\nReading TMY3 file...')
    wea, loc0 = read_tmy3(os.path.join(
        root_repo, '..', 'dev', 'resources', 'weather',
        'USA_CA_San.Francisco.Intl.AP.724940_TMY3.csv'))
    st0 = wea.index[int(len(wea.index)/2)].date()
    print(wea.loc[st0:st0+pd.DateOffset(hours=23),
                      ['ghi','dhi','dni']])

    print('\nGetting weather forecast...')
    forecast, model0 = get_forecast(tz=loc0['TZ'],
        loc=loc0, model=None)
    st1 = time.time()
    forecast, model0 = get_forecast(tz=loc0['TZ'],
        loc=loc0, model=model0)
    print(f'Duration for forecast: {round(time.time()-st1,1)}s')
    print(forecast[['ghi','dhi','dni']].round(0))



def example_weather_forecast(date=None, horizon=24):
    if not date:
        # select today's date
        start_time = dt.datetime.now().date() 
    else:
        start_time = pd.to_datetime(date)
    
    # read weather (forecast) data
    weather_path = os.path.join(os.path.dirname(root), 'resources', 'weather',
        'USA_CA_San.Francisco.Intl.AP.724940_TMY3.csv')
    weather, info = read_tmy3(weather_path, coerce_year=start_time.year)
    weather = weather.resample('5min').interpolate()
    
    # output data
    wf = weather.loc[start_time:start_time+pd.DateOffset(hours=horizon),]
    wf = wf[['temp_air','dni','dhi','wind_speed']+['ghi']].copy()
    return wf