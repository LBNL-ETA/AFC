#!/usr/bin/env python3
# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
User Interface.
"""

# pylint: disable=fixme

import argparse
import json
import datetime as dtm
from pathlib import Path

from flask import Flask, redirect, render_template_string, request, url_for, send_from_directory
from afc.defaultConfig import FT_TO_M
from afc.externalConfig import (DEFAULT_JSON_PATH, TARIFF_MAP,
                                coerce_config_types, validate_config)
from afc.utility.location import read_zipcodes

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / 'static'
AFC_CFG_PATH = ROOT / 'user_config.json'
ZIP_CSV_PATH = ROOT / 'zip_locations.csv'
AFC_SYSTEMS_PATH = (
    Path(__file__).resolve().parents[1] / 'resources' / 'radiance' / 'afc_systems.json'
)

# occupant distance levels as fractions of room depth (location 1..5)
_LOC_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0] # (loc-1)/4 for loc=1..5

# dimension fields submitted in display units, converted to metres on save
_DIM_FIELDS = ['room_width', 'room_height', 'room_depth',
               'window_width', 'window_height', 'window_sill']

# interface_ keys that map to a plain form field name (and vice versa)
_INTERFACE_FORM_MAP = {'interface_system_zones': 'system_zones'}

# keys handled explicitly in _build_config_initial, skipped in generic loop
_SKIP_KEYS = {'location_state', 'location_latitude', 'location_longitude',
              'location_elevation', 'interface_unit_preference',
              'occupant_1_distance', 'debug'}

app = Flask(__name__, static_folder=str(STATIC_DIR))

def _distance_to_location(dist_m, depth_m):
    """Map a stored distance (m) back to the nearest 1-5 location category."""
    if depth_m <= 0:
        return 2
    frac = dist_m / depth_m
    nearest = min(range(5), key=lambda i: abs(_LOC_FRACTIONS[i] - frac))
    return nearest + 1 # 1-based

def get_current_config():
    """Read and return the current config dict and a flag indicating if it is new (default)."""
    if AFC_CFG_PATH.exists():
        return json.loads(AFC_CFG_PATH.read_text(encoding='utf-8')), False
    return json.loads(Path(DEFAULT_JSON_PATH).read_text(encoding='utf-8')), True

def put_current_config(cfg):
    """Validate, back up the existing config, and write the new one."""
    validate_config(cfg)
    if AFC_CFG_PATH.exists():
        postfix = dtm.datetime.now().strftime('%Y%m%dT%H%M%S')
        backup = AFC_CFG_PATH.with_name(AFC_CFG_PATH.stem + f'_backup{postfix}.json')
        AFC_CFG_PATH.rename(backup)
    AFC_CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding='utf-8')

def del_current_config():
    """Back up and remove the current config, reverting to defaults on next load."""
    if AFC_CFG_PATH.exists():
        postfix = dtm.datetime.now().strftime('%Y%m%dT%H%M%S')
        backup = AFC_CFG_PATH.with_name(AFC_CFG_PATH.stem + f'_backup{postfix}.json')
        AFC_CFG_PATH.rename(backup)

def render_page(filename, **ctx):
    """Read an HTML file from static/ and render it with Jinja2."""
    html = (STATIC_DIR / filename).read_text(encoding='utf-8')
    return render_template_string(html, **ctx)

def _build_config_initial(temp):
    """Build the JS snippet that pre-fills the configuration form from stored config."""
    if 'location_state' not in temp:
        return ''
    m_to_ft = 1 / FT_TO_M
    unit_pref = temp['interface_unit_preference']
    # restore state/city dropdowns
    js = (
        f"document.getElementsByName('location_state')[0].value='{temp['location_state']}';"
        f"update_city();\n"
    )
    # restore unit selector
    js += f"var _u=document.getElementById('unit_select');if(_u)_u.value='{unit_pref}';\n"
    # restore occupant location from stored distance
    loc1 = _distance_to_location(float(temp['occupant_1_distance']), float(temp['room_depth']))
    js += (
        f"var _l=document.getElementsByName('occupant_1_location');"
        f"if(_l.length)_l[0].value='{loc1}';\n"
    )
    # restore all other form fields generically
    for k, v in temp.items():
        if k in _SKIP_KEYS:
            continue
        if k.startswith('interface_') and k not in _INTERFACE_FORM_MAP:
            continue
        if k.startswith('interface_preference_'):
            continue
        form_name = _INTERFACE_FORM_MAP[k] if k in _INTERFACE_FORM_MAP else k
        if k in _DIM_FIELDS:
            v = round(float(v) * m_to_ft, 1) if unit_pref == 'ft' else round(float(v), 1)
        js += f"var _e=document.getElementsByName('{form_name}');if(_e.length)_e[0].value='{v}';\n"
    # restore elevation in display unit
    elev_disp = round(float(temp['location_elevation']) * m_to_ft, 1) \
        if unit_pref == 'ft' else round(float(temp['location_elevation']), 2)
    js += (
        f"var _ev=document.getElementById('location_elevation_disp');"
        f"if(_ev){{_ev.value='{elev_disp}';_ev.dataset.unit='{unit_pref}';}}\n"
    )
    # restore debug checkbox
    debug_val = str(temp['debug']).lower()
    js += f"var _d=document.getElementsByName('debug');if(_d.length)_d[0].checked=({debug_val});\n"
    return js


def _process_config_post():
    """Parse and convert the config form POST into a config dict."""
    inputs = request.form.to_dict()
    locs = app.config['DICT_LOCS']
    # resolve lat/lon from state+city lookup
    inputs['location_latitude'] = round(
        locs['latitude'][inputs['location_state'], inputs['location_city']], 4
    )
    inputs['location_longitude'] = round(
        locs['longitude'][inputs['location_state'], inputs['location_city']], 4
    )
    inputs['interface_status'] = 'Updated Configuration.'
    unit = inputs.pop('unit')
    inputs['interface_unit_preference'] = unit
    scale = FT_TO_M if unit == 'ft' else 1.0
    # convert dimension fields from display unit to metres
    for f in _DIM_FIELDS:
        if f in inputs:
            inputs[f] = round(float(inputs[f]) * scale, 1)
    # elevation submitted separately in display unit
    inputs['location_elevation'] = round(float(inputs.pop('location_elevation_disp')) * scale, 1)
    inputs['debug'] = 'debug' in inputs and inputs['debug'] == 'true'
    # convert occupant location index to distance in metres
    loc1 = int(inputs.pop('occupant_1_location'))
    depth_m = float(inputs['room_depth'])
    inputs['occupant_1_distance'] = round(_LOC_FRACTIONS[loc1 - 1] * depth_m, 1)
    # rename plain form keys to interface_ prefixed keys
    for prefixed, plain in _INTERFACE_FORM_MAP.items():
        if plain in inputs:
            inputs[prefixed] = int(inputs.pop(plain))
    coerce_config_types(inputs)
    # TODO: occupants 2 and 3 - may be added in future
    inputs['occupant_number'] = 1
    return inputs

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve files from the static directory."""
    return send_from_directory(str(STATIC_DIR), filename)

@app.route('/', methods=['GET', 'POST'])
def preference():
    """Render and handle the occupant preferences page."""
    if request.method == 'GET':
        temp, _ = get_current_config()
        return render_page('preferences.html', **temp)
    inputs = request.form.to_dict()
    # cast preference_permanent from string to bool
    if 'interface_preference_permanent' in inputs:
        inputs['interface_preference_permanent'] = inputs['interface_preference_permanent'] == '1'
    temp, _ = get_current_config()
    temp.update(inputs)
    coerce_config_types(temp)
    put_current_config(temp)
    return render_page('print_dictionary.html', sorted_dictionary=sorted(inputs.items()))

@app.route('/config/', methods=['GET', 'POST'])
def config():
    """Render and handle the system configuration page."""
    if request.method == 'GET':
        temp, _ = get_current_config()
        states = app.config['DICT_STATE']
        afc_systems = app.config['AFC_SYSTEMS']
        return render_page(
            'configuration.html',
            dict_state=sorted(states.keys()),
            dict_state_all=json.dumps(states),
            set_initial=_build_config_initial(temp),
            tariff_map=TARIFF_MAP,
            afc_systems=afc_systems,
        )
    inputs = _process_config_post()
    temp, _ = get_current_config()
    temp.update(inputs)
    put_current_config(temp)
    return render_page('print_dictionary.html',
                       sorted_dictionary=sorted(temp.items()),
                       show_reset=False)

@app.route('/debug/', methods=['GET', 'POST'])
def debug():
    """Render the debug page and handle config reset."""
    if request.method == 'GET':
        temp, _ = get_current_config()
        return render_page('print_dictionary.html',
                           sorted_dictionary=sorted(temp.items()),
                           show_reset=True)
    if 'reset_config' in request.form and request.form['reset_config'] == '1':
        del_current_config()
        return render_page(
            'print_dictionary.html',
            sorted_dictionary={'Message:': 'The configuration was successfully reset.'}.items(),
        )
    return redirect(url_for('debug'))

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='AFC User Interface.')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to listen on (default: 8000)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config-dir', metavar='PATH',
                       help='Folder to store user_config.json (default: script directory)')
    group.add_argument('--config-file', metavar='PATH',
                       help='Path to an existing JSON config file to use as the active config')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    if args.config_file:
        cfg_path = Path(args.config_file).resolve()
        if not cfg_path.is_file():
            raise SystemExit(f'ERROR: --config-file does not exist: {cfg_path}')
        AFC_CFG_PATH = cfg_path
    elif args.config_dir:
        cfg_dir = Path(args.config_dir).resolve()
        if not cfg_dir.is_dir():
            raise SystemExit(f'ERROR: --config-dir does not exist: {cfg_dir}')
        AFC_CFG_PATH = cfg_dir / 'user_config.json'
    dict_state, dict_locs = read_zipcodes(ZIP_CSV_PATH)
    app.config['DICT_STATE'] = dict_state
    app.config['DICT_LOCS'] = dict_locs
    afc_systems_raw = json.loads(AFC_SYSTEMS_PATH.read_text(encoding='utf-8'))
    app.config['AFC_SYSTEMS'] = sorted(afc_systems_raw.keys())
    print(f'AFC User Interface running at http://{args.host}:{args.port}/')
    print(f'Config file: {AFC_CFG_PATH}')
    app.run(host=args.host, port=args.port, debug=False)
