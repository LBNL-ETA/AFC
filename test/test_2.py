"""
AFC example2 test module.
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

from afc.resources.radiance.make_glazing_systems import make_sage_systems, make_all_systems

ROOT = os.path.dirname(os.path.abspath(__file__))

def test1():
    """
    This is test1 to test the AFC functionality.
    """

    # make sage systems
    products_dir = os.path.join(ROOT, '..', 'afc', 'resources', 'radiance')
    sage_outer_layers = {6: os.path.join(products_dir, 
                                         'products/igsdb_product_7407.json')}
    sage_config = {}
    sage_config['sec02_{tvis:04.1f}'] = {
        'outer_layers': sage_outer_layers,
        'layers': [os.path.join(products_dir,
                                'products/igsdb_product_5054.json')], # 6mm low-e
        'flip_layers': [1],
        'gap': {'mix': {'air': 0.1, 'argon': 0.9},                  # 90% argon
                'gap': 0.0102},                                     # 10.2mm gap
        'shading': None,
        'blinds': None
        }
    glazing_systems = make_sage_systems(sage_config)

    # make systems
    make_all_systems(glazing_systems)
