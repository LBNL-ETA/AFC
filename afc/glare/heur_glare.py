# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Glare handler module.
"""

# pylint: disable=wrong-import-position, dangerous-default-value, redefined-outer-name
# pylint: disable=too-many-locals, pointless-string-statement, too-many-branches
# pylint: disable=too-many-instance-attributes

import os
import sys
import numpy as np
import pandas as pd
from pvlib import solarposition, irradiance

root = os.path.dirname(os.path.realpath(__file__))

sys.path.append(root)
from view_angle import make_view_config, view_config_from_rad#, sun_in_view

heur_view_config = {'top_start_alt':20, 'mid_start_alt':10, 'bot_start_alt':0,
                    'top_end_alt':55, 'mid_end_alt':45, 'bot_end_alt':35,
                    'start_lx':18e3, 'end_lx': 11e3, 'azi_fov':65}

DEFAULT_CONFIG = {'latitude': 37, 'longitude': 122, 'timezone': 120, 'orient': 0,
                  'view_orient': 's', 'view_dist': 1.22, 'view_height': 1.22}

DEFUALT_THR_60WWR = {'0': 4250, '90': 3750, '-90': 3750, '180': 3750}
DEFUALT_THR_40WWR = {'0': 3000, '90': 6000, '-90': 6000, '180': 6000}

class MultiZone:
    """Heuristic Glare and Daylight Controller"""
    def __init__(self, config=DEFAULT_CONFIG,
                 tvis=[0.01, 0.06, 0.18, 0.6], threshold_maps=DEFUALT_THR_60WWR,
                 view_config=None):
        self.config = config
        self.tvis = tvis
        self.threshold_maps = threshold_maps
        if not view_config:
            #if 'rad_cfg' in config.keys():
            #    pass
            #else:
            print('INFO: Not using Radiance config to generate view angles.')
            self.view_config = make_view_config(wwr=0.6,
                                                view_dist=config['view_dist'],
                                                view_height=config['view_height'])
        else:
            self.view_config = view_config

        self.lat = self.config['latitude']
        self.lon = self.config['longitude']
        self.tz = self.config['timezone']

        # determine control altitudes
        self.alts = pd.DataFrame()
        for k in sorted(self.view_config.keys()):
            if 'start_alt' in k:
                self.alts.loc[k.split('_')[0], 'start'] = self.view_config[k]
            if 'end_alt' in k:
                self.alts.loc[k.split('_')[0], 'end'] = self.view_config[k]

        self.n_zones = len(self.alts)
        self.gmodes = [False for _ in range(self.n_zones)]

        # make thresholds
        self.make_thresholds()

        self.ctrl = None
        self.results = None

    def make_thresholds(self):
        """make the glare thresholds"""
        self.thresholds = []

        threshold_map = self.threshold_maps[str(self.config['orient'])]
        for z in range(self.n_zones):
            if isinstance(threshold_map, dict):
                # threshold vaires per zone
                self.thresholds.append(threshold_map[z])
            else:
                # threshold static for all zones
                self.thresholds.append(threshold_map)

    def get_solar(self, weather_data):
        """get the solar data"""
        dni = weather_data['weaHDirNor']
        dhi = weather_data['weaHDifHor']
        if dni + dhi > 1:
            # solar position
            tz = int(self.tz / 15)
            weather_data.name = weather_data.name.tz_localize(f'Etc/GMT{tz:+d}')
            pos = solarposition.get_solarposition(weather_data.name,
                                                  self.lat,
                                                  -1*self.lon,
                                                  altitude=self.config['elevation'])
            alt = pos['elevation']
            azi = pos['azimuth']
            zen = pos['zenith']

            '''
            # Old (original) implementation
            if self.config['orient'] <= 205:
                azi_shift = (azi-180).apply(lambda x: x+90 if x > -90 else x+450)
            else:
                azi_shift = azi-180+450
            '''
            azi_shift = azi-180+90 # scale 0 to 180

            # incident illuminance
            face_tilt = 90
            face_azimuth = self.config['orient'] + 180
            ghi = dni * np.cos(np.radians(zen)) + dhi
            dni_extra = irradiance.get_extra_radiation(weather_data.name,
                                                       epoch_year=weather_data.name.year)
            total_irrad = irradiance.get_total_irradiance(face_tilt,
                                                          face_azimuth,
                                                          zen,
                                                          azi,
                                                          dni, ghi, dhi,
                                                          dni_extra=dni_extra, #1361
                                                          model='perez')
            inci = total_irrad['poa_global'] * 110
            return alt.values[0], azi_shift.values[0], inci.values[0], azi.values[0]
        return 0, 0, 0, 0

    def do_step(self, ts, ctrl_input={}):
        """compute"""
        # Parse inputs
        weather_data = ctrl_input['weather_data'].loc[ts]
        dni = weather_data['weaHDirNor']
        dhi = weather_data['weaHDifHor']
        alt, azi_shift, inci, azi = self.get_solar(weather_data)
        if dni + dhi > 0:
            if inci != 0:
                self.glare_mode_many(alt, azi_shift, inci)
                self.ctrl = self.daylight_mode(self.gmodes, inci, self.thresholds)
            else:
                inci = -1
                self.ctrl = [len(self.tvis)-1] * self.n_zones # bright
        else:
            inci = -1
            self.ctrl = [len(self.tvis)-1] * self.n_zones # bright

        self.results = {'alt': alt,
                        'azi_shift': azi_shift,
                        'inci': inci,
                        'azi': azi}
        for i, gmode in enumerate(self.gmodes):
            self.results[f'gmode_{i}'] = gmode

        return self.ctrl, self.results

    def daylight_mode(self, gmodes, inci, thresholds):
        """daylight mode no glare present"""
        ctrl = []
        for gmode, threshold in zip(gmodes, thresholds):
            if not gmode:
                vis_states = [t for t in self.tvis if t < (threshold/inci)]
                if len(vis_states) > 0:
                    ctrl.append(self.tvis.index(vis_states[-1])) # brightest mode which passes thr
                else:
                    ctrl.append(0) # no state satisfies; default to darkest state
            else:
                ctrl.append(0) # dark
        return ctrl

    def glare_mode_many(self, alt, azi, inci):
        """glare mode to avoid glare"""
        if self.config['view_orient'] == 's': #0: # S
            gm_start_azi = 90 - self.view_config['azi_fov'] + self.config['orient']
            gm_end_azi = 90 + self.view_config['azi_fov'] + self.config['orient']
        elif self.config['view_orient'] == 'w': #90: # w
            gm_start_azi = 90 + self.config['orient']
            gm_end_azi = 90 + self.view_config['azi_fov'] + self.config['orient']
        elif self.config['view_orient'] == 'e': #270: # E
            gm_start_azi = 90 - self.view_config['azi_fov'] + self.config['orient']
            gm_end_azi = 90 + self.config['orient']
        elif self.config['view_orient'] == 'n': #180: # N
            gm_start_azi = 0
            gm_end_azi = 0
        else:
            raise ValueError('The selected view_orient of' \
                + f'{self.config["view_orient"]} is not valid.' \
                + 'Please use one of s: south, w: west, e: east, n: north')
        start_lx = self.view_config['start_lx']
        end_lx = self.view_config['end_lx']
        gmodes = [False] * len(self.alts)

        if gm_start_azi < azi < gm_end_azi:
            if inci >= start_lx:
                for ix, zone in enumerate(self.alts.index):
                    if self.alts.loc[zone, 'start'] < alt < self.alts.loc[zone, 'end']:
                        gmodes[ix] = True
            elif end_lx <= inci <= start_lx:
                for ix, zone in enumerate(self.alts.index):
                    if self.alts.loc[zone, 'start'] < alt < self.alts.loc[zone, 'end'] \
                        and self.gmodes[ix]:
                        gmodes[ix] = True

        for ix, zone in enumerate(self.alts.index):
            self.gmodes[ix] = gmodes[ix]
        return self.gmodes

if __name__ == '__main__':

    import datetime as dt
    dt = dt.datetime(2019, 2, 1, 10, 1)
    ctrl_input = {'thermal_emulator_inputs': None,
        'weather_data': pd.DataFrame({'weaHDirNor': [100], 'weaHDifHor': [140]})}
    config = DEFAULT_CONFIG

    #print('Heur Logic:', heur_view_config)
    #print(make_view_config(wwr=0.4, view_dist=1.52, view_height=1.2))
    new_view_config = view_config_from_rad('resources/radiance/room0.4WWR_ec.cfg')
    print(new_view_config)

    view_config = new_view_config
    #view_config = heur_view_config
    #view_config = None
    controller = MultiZone(config=config, view_config=view_config)
    print(controller.do_step(dt, ctrl_input=ctrl_input))
