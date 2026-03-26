# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Export AFC Systems module.
"""

import os
import json
import shutil

from afc.resources.radiance.make_glazing_systems import (
    make_sage_systems,
    load_glazing_systems,
    add_shading_systems,
    make_all_systems,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

# sage
sage_outer_layers = {
    0.4: "products/igsdb_product_7405_vt0.4.json",  # 0.4% tvis
    # 1:  'products/igsdb_product_7405.json',    # 1% tvis
    6: "products/igsdb_product_7407.json",  # 6% tvis
    # 13: 'products/igsdb_product_7403.json',    # 13% tvis
    18: "products/igsdb_product_7404.json",  # 18% tvis
    60: "products/igsdb_product_7406.json",  # 60% tvis
}
sage_config = {}
# double-pane
sage_config["sec02_{tvis:04.1f}"] = {
    "outer_layers": sage_outer_layers,
    "layers": ["products/igsdb_product_5054.json"],  # 6mm low-e (loE 180)
    "flip_layers": [1],
    "gap": {"mix": {"air": 0.1, "argon": 0.9}, "gap": 0.0102},  # 10.2mm gap
    "shading": None,
    "blinds": None,
}
# triple pane
sage_config["sec03_{tvis:04.1f}"] = {
    "outer_layers": sage_outer_layers,
    "layers": [
        "products/igsdb_product_364.json",  # 6mm clear
        "products/igsdb_product_5054.json",
    ],  # 6mm low-e (loE 180)
    "flip_layers": [2],  # flip low-e to be on outsdie
    "gap": {"mix": {"air": 0.1, "argon": 0.9}, "gap": [0.0122, 0.0122]},  # 2*12.2mm gap
    "shading": None,
    "blinds": None,
}

# shading products
shading_products = {}
shading_products["sh2%d"] = {
    "layer": "products_shading/SilverScreen 2% - Dark Gray.xml",
    "type": "shading",
    "note": "Rollease Acmeda Gray Shade with silver outside.",
}
shading_products["sh2%b"] = {
    "layer": "products_shading/SilverScreen 2% - White.xml",
    "type": "shading",
    "note": "Rollease Acmeda White Shade with silver outside.",
}
shading_products["sh5%d"] = {
    "layer": "products_shading/E Screen 5% Charcoal-Charcoal.xml",
    "type": "shading",
    "note": "Mermet Dark Shade.",
}
shading_products["sh5%b"] = {
    "layer": "products_shading/E Screen 5% White-White.xml",
    "type": "shading",
    "note": "Mermet White Shade.",
}
# Edited slat thickness, slat width, and slat spacing proportionally to be 2in in slat width.
# shading_products['bl00'] = {'layer': 'products_shading/igsdb_product_20800_edited.json',
#                             'type': 'blinds',
#                             'slat_angles': [0, 15, 30, 45, 60, 75, 90, -15, -30, -45, -60, -75],
#                             'note': 'Edited 2in in slat width.'}


def make_systems(glazing_path, res_dir=ROOT):
    """Create and export glazing systems, returning the result directory."""
    # make sage systems
    glazing_systems = make_sage_systems(sage_config)

    # load pure glazing systems
    glass_systems = load_glazing_systems(glazing_path)
    glass_systems = {
        "dgl00_clr": glass_systems["dgl00_clr"]
    }  # only select one double-pane base window
    glazing_systems.update(glass_systems)

    # add shading (no thc-thermochromic, no sec-sage); only inside
    add_shading_systems(glazing_systems, shading_products, insides=[True])

    print(len(glazing_systems))

    result_dir, glazing_systems = make_all_systems(
        glazing_systems, prefix="systems", res_dir=res_dir, cpus=None
    )
    return result_dir


def group_glazing_systems(results_dir):
    """Organize exported glazing system JSON files into per system directories."""
    sys_dir = os.path.join(results_dir, "..", "glazing_systems")
    os.makedirs(sys_dir, exist_ok=True)
    # load systems
    with open(os.path.join(results_dir, "window_systems.json"), encoding="utf-8") as f:
        systems = json.loads(f.read())
    # ecs
    for sys in [sys for sys in systems if sys.startswith("sec")]:
        sys_name = f"ec_{sys.split('_')[0]}"
        os.makedirs(os.path.join(sys_dir, sys_name), exist_ok=True)
        shutil.copy(
            os.path.join(results_dir, f"{sys}.json"),
            os.path.join(sys_dir, sys_name, f"{sys}.json"),
        )
    # shade
    for sys in [sys for sys in systems if "_sh" in sys]:
        ss = sys.split("_")
        sys_name = f"shade_{ss[3]}_{ss[2]}"
        os.makedirs(os.path.join(sys_dir, sys_name), exist_ok=True)
        shutil.copy(
            os.path.join(results_dir, f"{sys}.json"),
            os.path.join(sys_dir, sys_name, f"{sys}.json"),
        )
        base_sys = "_".join(sys.split("_")[:2])
        shutil.copy(
            os.path.join(results_dir, f"{base_sys}.json"),
            os.path.join(sys_dir, sys_name, f"{base_sys}.json"),
        )


if __name__ == "__main__":
    PATH_GS = "glazing_systems_20250417T160159.json"  # US only (glass)
    # make the systems
    temp_dir = make_systems(PATH_GS)
    # move to directories
    group_glazing_systems(temp_dir)
