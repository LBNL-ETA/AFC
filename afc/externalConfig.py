# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Configuration parser.
"""

# pylint: disable=invalid-name, bare-except, too-many-statements
# pylint: disable=too-many-locals

import os
import json
import warnings

from afc.defaultConfig import default_parameter, ft_to_m
from afc.radiance.configs import get_config

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = os.getcwd()

def read_json_config(config_path, json_conly=False):
    """Utility function to parse a json configuration file."""

    # Import and read the new configuration from the json file
    with open(config_path, encoding='utf8') as f:
        config = json.load(f)
    if json_conly:
        return config
    return config_from_dict(config)

DEFAULT_JSON_PATH = os.path.join(root, 'resources', 'config', 'example_config.json')
DEFAULT_DICT = read_json_config(DEFAULT_JSON_PATH, json_conly=True)

def config_from_dict(config):
    """Utility function to make configuration from a dictionary."""

    for k in read_json_config(DEFAULT_JSON_PATH, json_conly=True):
        if k not in config:
            warnings.warn(f'The configuration of {k} is missing, using default.')

    # Update heating and lighting efficiency dependung on the system type
    if config['system_heating']=='el':
        heating_efficiency = config['system_heating_eff']
    else:
        heating_efficiency = 1.0

    if config['system_light']=='FLU':
        lighting_efficiency = 0.5
    elif config['system_light']=='LED':
        lighting_efficiency = 0.25
    else:
        raise ValueError(f'Ligthing system {config["system_light"]} not defined.')

    # Update occupant glare prefernces (100%=>0.4; 80%=>0.3; 120%=>0.5)
    glare_max = config['occupant_glare'] * 0.004

    # Update occupant wpi prefernces (80%=>250lx 100%=>350lx 120%=>450lx.)
    wpi_min = 250 + max(0, (config['occupant_brightness'] - 80) * 5)

    # Upload default_parameter with arguments from json
    parameter = default_parameter(tariff_name=config['tariff_name'],
                                  facade_type=config['system_type'],
                                  room_height=config['room_height'],
                                  room_width=config['room_width'],
                                  room_depth=config['room_depth'],
                                  window_height=config['window_height'],
                                  window_sill=config['window_sill'],
                                  window_width=config['room_width'],
                                  system_cooling_eff=1/config['system_cooling_eff'],
                                  location_latitude=config['location_latitude'],
                                  location_longitude=config['location_longitude'],
                                  location_orientation=int(config['location_orientation']),
                                  view_orient=int(config['occupant_1_direction']),
                                  system_heating_eff=heating_efficiency,
                                  lighting_efficiency=lighting_efficiency,
                                  number_occupants=config['occupant_number'],
                                  schedule=None,
                                  wpi_min=wpi_min,
                                  glare_max=glare_max,
                                  instance_id=config['system_id'],
                                  debug=config['debug'],
                                  #timezone=timezone,
                                  #elevation=elevation,
                                 )

    # Update windows position and dimensions
    for wz in parameter['facade']['windows']:
        # all windows have same width
        window_width = ft_to_m(config['window_width'])
        # all windows have same height
        window_height = ft_to_m(config['window_height'] / len(parameter['facade']['windows']))
        # need to make sure windows are centered
        x_origin = ft_to_m((config['room_width'] - config['window_width']) / 2)
        # new window starts at sill + X*windows
        y_origin = ft_to_m(config['window_sill']) + wz *  window_height
        window = f'{x_origin} {y_origin} {window_width} {window_height}'
        parameter['radiance']['dimensions'][f'window{wz+1}'] = window

    # Update radiance paths
    root_rad = os.path.split(parameter['radiance']['paths']['rad_config'])[0]
    filestruct, rad_config = get_config(parameter['facade']['type'],
                                        str(0.6), # need to be fixed to www=0.6
                                        root=root_rad)
    parameter['radiance']['paths']['rad_config'] = rad_config
    parameter['radiance']['paths']['rad_bsdf'] = filestruct['resources']
    parameter['radiance']['paths']['rad_mtx'] = filestruct['matrices']

    # Update building parameters based on its age
    if config['building_age'] == 'new_constr':
        parameter['zone']['param']['Ci'] = 492790.1315
        parameter['zone']['param']['Cs'] = 3765860.312
        parameter['zone']['param']['Ris'] = 0.02364937286
        parameter['zone']['param']['Row1'] = 0.003045420646
        parameter['zone']['param']['Rw1w2'] = 0.1442566037
        parameter['zone']['param']['Rw2i'] = 0.0002577364781
    elif config['building_age'] == 'post-1980':
        parameter['zone']['param']['Ci'] = 492790.1315
        parameter['zone']['param']['Cs'] = 3765860.312
        parameter['zone']['param']['Ris'] = 0.02364937286
        parameter['zone']['param']['Row1'] = 0.003045420646
        parameter['zone']['param']['Rw1w2'] = 0.1442566037
        parameter['zone']['param']['Rw2i'] = 0.0002577364781
    elif config['building_age'] == 'pre-1980':
        parameter['zone']['param']['Ci'] = 492790.1315
        parameter['zone']['param']['Cs'] = 3765860.312
        parameter['zone']['param']['Ris'] = 0.02364937286
        parameter['zone']['param']['Row1'] = 0.003045420646
        parameter['zone']['param']['Rw1w2'] = 0.1442566037
        parameter['zone']['param']['Rw2i'] = 0.0002577364781
    else:
        raise ValueError(f'Building age {config["building_age"]} not defined.')

    # Update parameter wrapper
    parameter['wrapper']['log_dir'] = './logs'

    return parameter