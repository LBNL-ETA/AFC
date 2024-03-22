import os
import sys
import pandas as pd
import tempfile as tf
import subprocess as sp

root = os.path.dirname(os.path.realpath(__file__))

sys.path.append(root)
from view_angle import make_view_config, sun_in_view, view_config_from_rad

sage_view_config = {'top_start_alt':20, 'mid_start_alt':10, 'bot_start_alt':0,
                    'top_end_alt':55, 'mid_end_alt':45, 'bot_end_alt':35,
                    'start_lx':18e3, 'end_lx': 11e3, 'azi_fov':65}

class Sage3zone(object):
    def __init__(self, config={'latitude':37, 'longitude': 122, 'timezone': 120, 'orient': 0, 'view_dist': 1.52, 'view_height': 1.2},
                 tvis = [0.01, 0.06, 0.18, 0.6], wwr=0.6,
                 view_config=None):
        self.config = config
        self.tvis = tvis
        self.wwr = wwr
        if not view_config:
            #if 'rad_cfg' in config.keys():
            #    pass
            #else:
            print('INFO: Not using Radiance config to generate view angles.')
            self.view_config = make_view_config(wwr=wwr, view_dist=config['view_dist'],
                                                view_height=config['view_height'])
        else:
            self.view_config = view_config
        self.results = {}
        self.lat = self.config['latitude']
        self.lon = self.config['longitude']
        self.tz = self.config['timezone']
        self.bot_gmode = False
        self.mid_gmode = False
        self.top_gmode = False
        # For controller
        self.alts = pd.DataFrame()
        for k in sorted(self.view_config.keys()):
            if 'start_alt' in k:
                self.alts.loc[k.split('_')[0], 'start'] = self.view_config[k]
            if 'end_alt' in k:
                self.alts.loc[k.split('_')[0], 'end'] = self.view_config[k]
        self.gmodes = [False] * len(self.alts)
        
    def get_solar(self, ts, weather_forecast):
        dni = weather_forecast['weaHDirNor'].values[0]
        dhi = weather_forecast['weaHDifHor'].values[0]
        if dni + dhi > 1:
            mo = ts.month
            da = ts.day
            hrs = ts.hour + ts.minute/60
            cmd = f'gendaylit {mo} {da} {hrs} -a {self.lat} -o {self.lon} -m {self.tz} -W {dni} {dhi}'
            cmd += f" | xform -rz {self.config['orient']}"
            res = sp.run(cmd, shell=True, check=True, stderr=sp.PIPE, stdout=sp.PIPE)
            if res.stderr:
                if not 'Warning: sun altitude below zero at time step' in res.stderr.decode():
                    print('WARNING: rpict:', res.stderr.decode())
            res = res.stdout.decode()
            line2 = res.splitlines()[3].split()
            alt, azi = map(float, line2[-2:])
            with tf.TemporaryDirectory() as td:
                grndpath = os.path.join(td, 'ground.rad')
                with open(grndpath, 'w') as wtr:
                    wtr.write('void plastic asphalt 0 0 5 .2 .2 .2 0 0\n')
                    wtr.write('asphalt polygon grnd 0 0 12 -5 -5 0 -5 5 0 5 5 0 5 -5 0\n')
                    wtr.write('asphalt polygon wall 0 0 12 -5 0 0 5 0 0 5 0 3 -5 0 3\n')
                skypath = os.path.join(td, 'sky.rad')
                with open(skypath, 'w') as wtr:
                    wtr.write(res)
                    wtr.write('skyfunc glow skyglow 0 0 4 1 1 1 0\n')
                    wtr.write('skyglow source sky 0 0 4 0 0 1 180\n')
                    wtr.write('skyglow source ground 0 0 4 0 0 -1 180\n')
                octpath = os.path.join(td, 'sky.oct')
                sp.run(f'oconv {grndpath} {skypath} > {octpath}', shell=True, check=True)
                cmd = f"echo 0 0 3 0 -1 0 | rtrace -w -h -I+ -ab 1 -ad 262144 {octpath} "
                cmd += "| rcalc -e '$1=$1*47.4+$2*119.9+$3*11.6'"
                res = sp.run(cmd, shell=True, check=True, stdout=sp.PIPE).stdout.decode()
            inci = float(res)
        
            # CHECK ME
            if self.config['orient'] <= 205:
                azi_shift = azi+90 if azi > -90 else azi+450
            else:
                azi_shift = azi+450
            
            return alt, azi_shift, inci, azi
        else:
            return 0, 0, 0, 0
    

    def do_step(self, ts, ctrl_input={}):
        # Parse inputs
        weather_forecast = ctrl_input['weather_forecast']
        dni = weather_forecast['weaHDirNor'].values[0]
        dhi = weather_forecast['weaHDifHor'].values[0]
        if dni + dhi > 0:
            alt, azi_shift, inci, azi = self.get_solar(ts, weather_forecast)            
            if inci != 0:
                self.glare_mode(alt, azi_shift, inci)
                ctrl = self.daylight_mode_c(self.bot_gmode, self.mid_gmode, self.top_gmode, inci)
                #ctrl = self.daylight_mode_b(mo, self.bot_gmode, self.mid_gmode, self.top_gmode, inci)
            else:
                inci = -1
                ctrl = [len(self.tvis)-1] * 3
        else:
            inci = -1
            ctrl = [len(self.tvis)-1] * 3

        self.results['inci'] = inci
        self.results['bot_gmode'] = self.bot_gmode
        self.results['mid_gmode'] = self.mid_gmode
        self.results['top_gmode'] = self.top_gmode
        return ctrl, ctrl_input['thermal_emulator_inputs'], {}

    def daylight_mode_b(self, month, bot_gmode, mid_gmode, top_gmode, inci):
        month = int(month)
        if month in (4, 10):
            top_threshold = 4000
            mid_threshold = 8000
            bot_threshold = 12000
        elif month in (1, 2, 3, 11, 12):
            top_threshold = 4000
            mid_threshold = 4000
            bot_threshold = 8000
        else:
            top_threshold = 6000
            mid_threshold = 6000
            bot_threshold = 6000
        if not bot_gmode:
            bot_vis = [t for t in self.tvis if t < (bot_threshold/inci)][-1]
            bot_ctrl = self.tvis.index(bot_vis)
        else:
            bot_ctrl = 0
        if not mid_gmode:
            mid_vis = [t for t in self.tvis if t < (mid_threshold/inci)][-1]
            mid_ctrl = self.tvis.index(mid_vis)
        else:
            mid_ctrl = 0
        if not top_gmode:
            top_vis = [t for t in self.tvis if t < (top_threshold/inci)][-1]
            top_ctrl = self.tvis.index(top_vis)
        else:
            top_ctrl = 0
        return [bot_ctrl, mid_ctrl, top_ctrl]

    def daylight_mode_c(self, bot_gmode, mid_gmode, top_gmode, inci):
        # top_threshold = 4500 if self.config['orient'] == 0 else 4000 #6000
        # mid_threshold = 4500 if self.config['orient'] == 0 else 4000 #6000
        # bot_threshold = 4500 if self.config['orient'] == 0 else 4000 #6000
        if self.wwr == 0.6:
            if self.config['orient'] == 0:
                # south
                top_threshold = 4250
                mid_threshold = 4250
                bot_threshold = 4250
            else:
                # north, east, west
                top_threshold = 3750
                mid_threshold = 3750
                bot_threshold = 3750
        elif self.wwr == 0.4:
            if self.config['orient'] == 0:
                # south
                top_threshold = 6000/2
                mid_threshold = 6000/2
                bot_threshold = 6000/2
            else:
                # north, east, west
                top_threshold = 6000
                mid_threshold = 6000
                bot_threshold = 6000

        if not bot_gmode:
            bot_vis = [t for t in self.tvis if t < (bot_threshold/inci)][-1]
            bot_ctrl = self.tvis.index(bot_vis)
        else:
            bot_ctrl = 0
        if not mid_gmode:
            mid_vis = [t for t in self.tvis if t < (mid_threshold/inci)][-1]
            mid_ctrl = self.tvis.index(mid_vis)
        else:
            mid_ctrl = 0
        if not top_gmode:
            top_vis = [t for t in self.tvis if t < (top_threshold/inci)][-1]
            top_ctrl = self.tvis.index(top_vis)
        else:
            top_ctrl = 0
        return [bot_ctrl, mid_ctrl, top_ctrl]


    def glare_mode(self, alt, azi, inci):

        gm_start_azi = 90 - self.view_config['azi_fov'] + self.config['orient']
        gm_end_azi = 90 + self.view_config['azi_fov'] + self.config['orient']
        top_start_alt = self.view_config['top_start_alt']
        mid_start_alt = self.view_config['mid_start_alt']
        bot_start_alt = self.view_config['bot_start_alt']
        top_end_alt = self.view_config['top_end_alt']
        mid_end_alt = self.view_config['mid_end_alt']
        bot_end_alt = self.view_config['bot_end_alt']
        start_lx = self.view_config['start_lx']
        end_lx = self.view_config['end_lx']
        bot_gmode = False
        mid_gmode = False
        top_gmode = False
        if gm_start_azi < azi < gm_end_azi:
            if inci >= start_lx:
                if bot_start_alt < alt < bot_end_alt:
                    bot_gmode = True
                if mid_start_alt < alt < mid_end_alt:
                    mid_gmode = True
                if top_start_alt < alt < top_end_alt:
                    top_gmode = True
            elif end_lx <= inci <= start_lx:
                if (bot_start_alt < alt < bot_end_alt) and self.bot_gmode:
                    bot_gmode = True
                if (mid_start_alt < alt < mid_end_alt) and self.mid_gmode:
                    mid_gmode = True
                if (top_start_alt < alt < top_end_alt) and self.top_gmode:
                    top_gmode = True
        self.bot_gmode, self.mid_gmode, self.top_gmode = bot_gmode, mid_gmode, top_gmode
        return bot_gmode, mid_gmode, top_gmode
        
    def glare_mode_many(self, alt, azi, inci):

        if self.config['view_orient'] == 0: # S
            gm_start_azi = 90 - self.view_config['azi_fov'] + self.config['orient']
            gm_end_azi = 90 + self.view_config['azi_fov'] + self.config['orient']
        elif self.config['view_orient'] == 90: # w
            gm_start_azi = 90 + self.config['orient']
            gm_end_azi = 90 + self.view_config['azi_fov'] + self.config['orient']        
        elif self.config['view_orient'] == 270: # E
            gm_start_azi = 90 - self.view_config['azi_fov'] + self.config['orient']
            gm_end_azi = 90 + self.config['orient']        
        elif self.config['view_orient'] == 180: # N
            gm_start_azi = 0
            gm_end_azi = 0
        else:
            raise ValueError(f'The selected view_orient of {self.config["view_orient"]} is not valid.' \
                + 'Please use one of 0: south, 90: west, 270: east, 180: north')
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
                    if self.alts.loc[zone, 'start'] < alt < self.alts.loc[zone, 'end'] and self.gmodes[ix]:
                        gmodes[ix] = True
                        
        for ix, zone in enumerate(self.alts.index):
            self.gmodes[ix] = gmodes[ix]
        return self.gmodes


if __name__ == '__main__':
    import pandas as pd
    import datetime as dt
    dt = dt.datetime(2019, 2, 1, 10, 1)
    dni = 100
    dhi = 140
    ctrl_input = {'thermal_emulator_inputs': None,
        'weather_forecast': pd.DataFrame({'weaHDirNor': [dni], 'weaHDifHor': [dhi]})}
    config = {'latitude':37, 'longitude': 122, 'timezone': 120, 'orient': 0, 'view_dist': 1.52, 'view_height': 1.2}
    
    #print('SAGE Logic:', sage_view_config)
    #print(make_view_config(wwr=0.4, view_dist=1.52, view_height=1.2))
    cfg_path = '/home/Christoph/Documents/PrivateRepos/DynamicFacades/emulator/resources/radiance/room0.4WWR_ec.cfg'
    new_view_config = view_config_from_rad(cfg_path)
    print(new_view_config)
    
    view_config = new_view_config
    #view_config = sage_view_config
    #view_config = None
    sage_controller = Sage3zone(config=config, view_config=view_config, wwr=0.4)
    print(sage_controller.do_step(dt, ctrl_input=ctrl_input))

