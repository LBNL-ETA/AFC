# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Make Glazing Systems module.
"""

# pylint: disable=wrong-import-position, unbalanced-tuple-unpacking, too-many-locals
# pylint: disable=unused-variable

import os

# stop multi-thread from frads (numpy)
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import copy
import time
import json
import datetime as dtm
import multiprocessing as mp
import pandas as pd
import numpy as np

import frads as fr
from frads.window import OpeningDefinitions
import pywincalc as pwc

from .frads_library import create_glazing_system

ROOT = os.path.dirname(os.path.abspath(__file__))

def make_sage_systems(sage_config):
    """Create glazing systems from a SAGE configuration."""
    glazing_systems = {}

    for name, config in sage_config.items():
        for tint, tint_glass in config['outer_layers'].items():
            dict_layers = [isinstance(v, dict) for v in config['layers']]
            if not np.any(dict_layers):
                # one EC layer
                system = {'layers': [tint_glass] + config['layers'],
                          'flip_layers': config['flip_layers'],
                          'gap': config['gap'],
                          'shading': config['shading'],
                          'blinds': config['blinds']}
                glazing_systems[name.format(tvis=tint)] = copy.deepcopy(system)
            else:
                # two EC layers
                ec_layer_ix = dict_layers.index(True)
                additional_ec_layer = config['layers'][ec_layer_ix]
                for tint2, tint_glass2 in additional_ec_layer.items():
                    layers = copy.deepcopy(config['layers'])
                    layers[ec_layer_ix] = tint_glass2
                    system = {'layers': [tint_glass] + layers,
                              'flip_layers': config['flip_layers'],
                              'gap': config['gap'],
                              'shading': config['shading'],
                              'blinds': config['blinds']}
                    ttotal = ((tint/1e2) * (tint2/1e2)) * 1e2
                    tvis = f'{tint:02d}t{tint2:02d}={ttotal:.2f}'
                    tvis = f'{ttotal:.2f}'
                    glazing_systems[name.format(tvis=tvis)] = copy.deepcopy(system)

    return glazing_systems


def load_glazing_systems(systems_path):
    """Load data-driven glazing systems from a JSON file."""
    # load data-driven systems (glass only)
    with open(systems_path, encoding='utf-8') as f:
        other_systems = json.loads(f.read())
        for k in other_systems:
            other_systems[k]['gap']['gap'] = json.loads(other_systems[k]['gap']['gap'])
    return other_systems


def add_shading_systems(glazing_systems, shading_products, insides=None):
    """Add shading systems to existing glazing systems."""
    if not insides:
        insides = [True, False]
    all_glazings = [k for k in glazing_systems if not \
                    k.startswith('thc') and not k.startswith('sec')]
    for glazing in all_glazings:
        for shading_product in shading_products.items():
            slat_angles = [0]
            if shading_product[1]['type'] == 'blinds':
                slat_angles = shading_product[1]['slat_angles']
            for slat_angle in slat_angles:
                for inside in insides:
                    loc = 'i' if inside else 'o'
                    k = f'{glazing}_{loc}_{shading_product[0]}'
                    if shading_product[1]['type'] == 'blinds':
                        k += f'_{slat_angle:+}'
                    glazing_systems[k] = copy.deepcopy(glazing_systems[glazing])
                    glazing_systems[k]['shading'] = copy.deepcopy(shading_product[1])
                    glazing_systems[k]['shading']['inside'] = inside
                    glazing_systems[k]['shading']['slat_angle'] = slat_angle
                    glazing_systems[k]['blinds'] = shading_product[1]['type'] == 'blinds'


def add_info(glazing_systems, igsdb):
    """Add product information to glazing systems."""
    log_keys = ['product_id',
                'name',
                'manufacturer_name',
                'product_name',
                'nfrc_id',
                'short_description',
                'type',
                'subtype']
    for name, gs in glazing_systems.items():
        glazing_systems[name]['layer_infos'] = []
        for layer in gs['layers']:
            if not layer.startswith('products/igsdb_product'):
                continue
            pid = int(layer.split('_')[2].replace('.json', ''))
            record = [v for k, v in igsdb.items() if v['product_id'] == pid][0]
            r = {}
            for k in log_keys:
                r[k] = record[k]
            glazing_systems[name]['layer_infos'].append(copy.deepcopy(r))
    return glazing_systems


def flip_values(data, prefix_pairs=None):
    """Swap values between prefixed keys in a dictionary."""
    if not prefix_pairs:
        prefix_pairs = [('rb', 'rf'), ('tb', 'tf')]
    # Create a copy of the data to avoid modifying the original dictionary
    flipped_data = data.copy()

    # Iterate over each pair of prefixes
    for prefix1, prefix2 in prefix_pairs:
        # Find all keys that start with the first prefix
        keys1 = [key.replace(prefix1, '') for key in data if key.startswith(prefix1)]
        # Find all keys that start with the second prefix
        keys2 = [key.replace(prefix2, '') for key in data if key.startswith(prefix2)]

        # Determine the minimum number of keys to swap
        common_keys = list(set(keys1) & set(keys2))

        # Swap the values between the two sets of keys up to the minimum length
        for k in common_keys:
            key1 = f'{prefix1}{k}'
            key2 = f'{prefix2}{k}'
            flipped_data[key1], flipped_data[key2] = flipped_data[key2], flipped_data[key1]

    return flipped_data


def flip_record(record):
    """Flip the orientation of a glazing system record."""
    record = copy.deepcopy(record)

    # coated_side
    if record['coated_side'] == 'Back':
        record['coated_side'] = 'Front'
    elif record['coated_side'] == 'Front':
        record['coated_side'] = 'Back'

    # integrated_results_summary
    tt = record['integrated_results_summary'][0]
    tt = flip_values(tt, prefix_pairs=[('rb', 'rf'), ('tb', 'tf')])
    record['integrated_results_summary'] = [tt]

    # measured_data (Note: not flipping 'tir_back'/'tir_front')
    tt = record['measured_data']
    tt['emissivity_back'], tt['emissivity_front'] = \
        tt['emissivity_front'], tt['emissivity_back']
    record['measured_data'] = tt

    # spectral_data
    tt = pd.DataFrame(record['spectral_data']['spectral_data'])
    tt = tt.rename(columns={'Rb': 'Rf', 'Rf': 'Rb'})
    record['spectral_data']['spectral_data'] = tt.to_dict(orient='records')

    return record


def load_flip_store_record(product_path):
    """Load a record, flip it, and store the flipped version."""
    # load
    with open(product_path, encoding='utf-8') as f:
        record = json.loads(f.read())

    # flip
    record_flipped = flip_record(record)

    # store
    flipped_path = product_path.replace('.json', '_flipped.json')
    if not os.path.exists(flipped_path):
        with open(flipped_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(record_flipped))


def make_glazing_system(config):
    """Create a single glazing system from the given configuration dictionary."""
    name = config['name']
    gs = config['gs']
    save = config['save'] if 'save' in config else True
    res_dir = config['res_dir'] if 'res_dir' in config else ROOT
    igsdb = config['igsdb'] if 'igsdb' in config else {}
    debug = config['debug'] if 'debug' in config else False

    # make product jsons if not existent
    for layer in gs['layers']:
        if not os.path.exists(layer) and igsdb:
            # write record if igsbd exists
            with open(layer, 'w', encoding='utf-8') as f:
                pid = int(layer.split('_')[2].replace('.json', ''))
                record = [v for k, v in igsdb.items() if v['product_id'] == pid][0]
                f.write(json.dumps(record))

    # flip layers (flip record file)
    for flip_layer in gs['flip_layers']:
        product_path = gs['layers'][flip_layer]
        load_flip_store_record(product_path)
        gs['layers'][flip_layer] = \
            gs['layers'][flip_layer].replace('.json', '_flipped.json')

    # make config
    config = {
        'name': name,
        'nproc': mp.cpu_count() - 1,  # only used for blinds
        'mbsdf': True,  # melanopic bsdf
    }

    # make frads gap if defined
    if gs['gap']:
        gaps = gs['gap']['gap']
        if not isinstance(gaps, list):
            # single gap
            gaps = [gaps]
        config['gaps'] = [fr.Gap([fr.Gas(k, v) for k, v in gs['gap']['mix'].items()],
                                 gap) for gap in gaps]

    # make frads layers
    config['layer_inputs'] = [fr.LayerInput(layer) for layer in gs['layers']]
    if gs['shading']:
        openings = OpeningDefinitions(
            top_m=0,  # per Robert
            bottom_m=0,  # per Robert
            left_m=3e-3,  # per Robert (3mm)
            right_m=3e-3,  # per Robert (3mm)
            front_multiplier=0.05,  # per Robert (5%)
        )
        slat_angle_deg = gs['shading']['slat_angle'] if gs['blinds'] else 0
        layer = fr.LayerInput(gs["shading"]["layer"],
                              slat_angle_deg=slat_angle_deg,
                              openings=openings)
        gap = fr.Gap([fr.Gas('air', 1.0)], 0.0381)  # per Robert (1.5")
        if gs['shading']['inside']:
            # inside
            config['layer_inputs'].append(layer)
            config['gaps'].append(gap)
        else:
            # outside
            config['layer_inputs'] = [layer] + config['layer_inputs']
            config['gaps'] = [gap] + config['gaps']

    # call frads
    # glazing_system = fr.create_glazing_system(**config)
    # call modified frads
    glazing_system, glzsys, solres, visres = create_glazing_system(**config)

    # save
    if save:
        glazing_system.save(os.path.join(res_dir, f'{name}.json'))

    # stats
    gs['system_results'] = {}
    # Regeneration takes way too long; skipping blinds
    # if not (gs['blinds'] or gs['shading']):

    # smaller sphere for shading
    # if gs['shading']:
    #     bsdf_hemisphere = pwc.BSDFHemisphere.create(pwc.BSDFBasisType.SMALL)
    #     _, glzsys, solres, visres = create_glazing_system(bsdf_hemisphere=bsdf_hemisphere,
    #                                                       **config)
    try:
        gs['system_results']['shgc'] = glzsys.shgc()
        # gs['system_results']['u'] = glzsys.u()
        layer_temps = glzsys.layer_temperatures(pwc.TarcogSystemType.SHGC)
        for il, layer_temp in enumerate(layer_temps):
            gs['system_results'][f't_{il}'] = layer_temp - 273.15
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'ERROR: {name} {e}')
    gs['system_results']['vlt'] = visres.system_results.back.transmittance.direct_hemispherical
    # if not (gs['blinds'] or gs['shading']):
    #     colres = glzsys.color()
    #     colres_rgb = colres.system_results.back.transmittance.direct_hemispherical.rgb
    #     gs['system_results']['col_R'] = colres_rgb.R
    #     gs['system_results']['col_G'] = colres_rgb.G
    #     gs['system_results']['col_B'] = colres_rgb.B

    if debug:
        return glzsys, glazing_system
    return {name: gs.copy()}


def make_all_systems(glazing_systems, prefix='systems', res_dir=ROOT, cpus=None,
                     igsdb_paths=None):
    """Create all glazing systems and optionally store results."""
    if not igsdb_paths:
        igsdb_paths = {}

    # read igsdb(s)
    igsdb = {}
    # load igsdb databses
    for db_name, igsdb_path in igsdb_paths.items():
        with open(igsdb_path, encoding='utf-8') as f:
            igsdb.update(json.loads(f.read()))

    # make new resfolder
    dtm_fmt = '%Y%m%dT%H%M%S'
    res_dir = os.path.join(res_dir, f'{prefix}_{dtm.datetime.now().strftime(dtm_fmt)}')
    os.mkdir(res_dir)

    # make jobs
    jobs = []
    for name, gs in glazing_systems.items():
        jobs.append({'name': name, 'gs': gs, 'res_dir': res_dir, 'igsdb': igsdb})

    # make glazing systems
    if cpus:
        with mp.Pool(processes=cpus, maxtasksperchild=1) as pool:
            res = pool.map(make_glazing_system, jobs, chunksize=1)
            # combine dicts
            glazing_systems = {k: v for l in res for k, v in l.items()}
    else:
        for job in jobs:
            st = time.time()
            print(f'making {job["name"]} ...')
            glazing_systems.update(make_glazing_system(job))
            print(time.time() - st)

    # save overview
    with open(os.path.join(res_dir, 'window_systems.json'), 'w', encoding='utf-8') as f:
        if igsdb:
            glazing_systems = add_info(glazing_systems, igsdb)
        f.write(json.dumps(glazing_systems))

    return res_dir, glazing_systems
