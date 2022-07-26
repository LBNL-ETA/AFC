import warnings
warnings.filterwarnings('ignore', message='The forecast module algorithms' + \
    ' and features are highly experimental. ')

import os
import pvlib
import datetime as dt
import pandas as pd

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    print('WARNING: Using local directory as root.')
    root = os.getcwd()

def read_tmy3(filename=None, coerce_year=dt.datetime.now().year):
    weather, info = pvlib.iotools.read_tmy3(filename=filename,
                                   coerce_year=coerce_year)
    weather.index = weather.index - pd.DateOffset(hours=1)
    weather.index = weather.index.tz_localize(None)
    weather.index.name = None
    return weather, info

def read_mos(filename, coerce_year=dt.datetime.now().year):
    skiprows = 0
    header = []
    metadata = []
    with open(filename, 'r') as f:
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
    start_time = pd.to_datetime('01-01-{}'.format(coerce_year))
    weather.index = [start_time + \
        pd.DateOffset(seconds=t) for t in weather['time']]
    return weather, metadata

def get_forecast(st=dt.datetime.now(), tz=-8,
                 loc={'latitude':37.617, 'longitude':-122.4},
                 model=None):
    if not model:
        from pvlib.forecast import HRRR
        model = HRRR()
    tz = "Etc/GMT+{}".format(int(tz*-1))
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
    weather, info = read_tmy3(os.path.join(
        root_repo, 'resources', 'weather', 
        'USA_CA_San.Francisco.Intl.AP.724940_TMY3.csv'))
    st = weather.index[int(len(weather.index)/2)].date()
    print(weather.loc[st:st+pd.DateOffset(hours=23), 
                      ['GHI','DHI','DNI']])
    
    print('\nGetting weather forecast...')
    forecast, model = get_forecast(tz=info['TZ'],
        loc=info, model=None)
    st = time.time()
    forecast, model = get_forecast(tz=info['TZ'],
        loc=info, model=model)
    print('Duration for forecast: {}s'.format(round(time.time()-st,1)))
    print(forecast[['ghi','dhi','dni']].round(0))
