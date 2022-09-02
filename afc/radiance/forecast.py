"""
Radiance forecasting module
T.Wang
"""

import multiprocessing as mp
import os
import sys
import re
import subprocess as sp
import tempfile as tf
from configparser import ConfigParser
from frads import radmtx, makesky, radutil, room
import numpy as np
import copy
import pandas as pd
import shutil
import matplotlib.pyplot as plt

from .maps import shade_map_0x6, shade_map_0x4


root = os.path.dirname(os.path.realpath(__file__))

mult = np.linalg.multi_dot

def mtx_parser(path):
    with open(path) as rdr:
        raw = rdr.read()
    return radutil.mtx2nparray(raw)


def smx_parser(path):
    with open(path) as rdr:
        raw = rdr.read().strip().split('\n\n')
    header = raw[0]
    nrow, ncol, ncomp = radutil.header_parser(header)
    data = [i.splitlines() for i in raw[1:]]
    rdata = np.array([[i.split()[::ncomp][0] for i in row] for row in data],
                     dtype=float)
    gdata = np.array([[i.split()[1::ncomp][0] for i in row] for row in data],
                     dtype=float)
    bdata = np.array([[i.split()[2::ncomp][0] for i in row] for row in data],
                     dtype=float)
    if ncol == 1:
        assert np.size(bdata, 1) == nrow
        assert np.size(bdata, 0) == ncol
        rdata = rdata.flatten()
        gdata = gdata.flatten()
        bdata = bdata.flatten()
    else:
        assert np.size(bdata, 0) == nrow
        assert np.size(bdata, 1) == ncol
    return rdata, gdata, bdata


def checkout(cmd):
    return sp.check_output(cmd, shell=True)


class Forecast(object):
    def __init__(self, cfg_path, regenerate=None, facade_type='ec', wpi_plot=False, wpi_loc='occupant',
                 location=None, filestruct=None, _test_difference=False, dimensions=None):
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.parse_config(cfg_path)
        if regenerate:
            self.remake_matrices = True
        if filestruct:
            self.filestruct = filestruct
            self.bsdfd = self.filestruct['resources']
            self.matricesd = self.filestruct['matrices']
        if location:
            self.lat = location['latitude']
            self.lon = location['longitude']
            self.timezone = location['timezone']
            self.elevation = location['elevation']
            self.orient = location['orient']
        if dimensions:
            self.dims['width'] = dimensions['width']
            self.dims['depth'] = dimensions['depth']
            self.dims['height'] = dimensions['height']
            self.dims['window1'] = dimensions['window1']
            self.dims['window2'] = dimensions['window2']
            self.dims['window3'] = dimensions['window3']
        self.facade_type = facade_type
        self.wpi_plot = wpi_plot
        self.wpi_loc = wpi_loc
        self._test_difference = _test_difference
        self.get_paths()
        self.make_room()
        self.make_matrices()
        self.load_mtx()
        self.klems_coeff = radutil.angle_basis_coeff("Klems Full")
        self.wwr = float(cfg_path.split('room')[1].split('WWR')[0])
        if 'shade' in self.facade_type and float(self.wwr) == 0.6:
            self.shade_map = shade_map_0x6
        elif 'shade' in self.facade_type and float(self.wwr) == 0.4:
            self.shade_map = shade_map_0x4
        self.new_map = {}
        self.init = True

    def parse_config(self, cfg_path):
        _config = ConfigParser()
        if _config.read(cfg_path) == []:
            raise ValueError('The location of the "config" file is wrong. ' + \
                'Check location at {}.'.format(cfg_path))
        cfg = _config._sections
        site_info = cfg['Site']
        self.lat = site_info['latitude']
        self.lon = site_info['longitude']
        self.timezone = site_info['timezone']
        self.elevation = site_info['elevation']
        self.orient = site_info['orientation']
        self.dims = cfg['Dimensions']
        self.filestruct = cfg['FileStructure']
        self.matricesd = os.path.join(self.root, self.filestruct['matrices'])
        self.bsdfd = os.path.join(self.root, self.filestruct['resources'],
                                  'BSDFs')
        simctrl = cfg['SimulationControl']
        self.parallel = True if simctrl['parallel'] == 'True' else False
        self.vmx_opt = simctrl['vmx']
        self.vsmx_opt = simctrl['vsmx']
        self.dmx_opt = simctrl['dmx']
        self.remake_matrices = True if simctrl[
            're-make_matrices'] == 'True' else False
        self.view = cfg['View']['view1']
        self.grid_height = float(cfg['Grid']['height'])
        self.grid_spacing = float(cfg['Grid']['spacing'])

    def get_paths(self):
        """Where things are?"""
        btdfd_vis = os.path.join(self.bsdfd, 'vis')
        btdfd_shgc = os.path.join(self.bsdfd, 'shgc')
        btdfd_sol = os.path.join(self.bsdfd, 'sol')
        self.btdfs_vis = sorted([
            os.path.join(self.root, btdfd_vis, i)
            for i in os.listdir(btdfd_vis) if i.endswith('.mtx')
        ])
        self.btdfs_shgc = sorted([
            os.path.join(self.root, btdfd_shgc, i)
            for i in os.listdir(btdfd_shgc) if i.endswith('.mtx')
        ])
        self.btdfs_tsol = sorted([
            os.path.join(self.root, btdfd_sol, i)
            for i in os.listdir(btdfd_sol) if '_Tsol' in i
        ])
        self.btdfs_ftsol = sorted([
            os.path.join(self.root, btdfd_sol, i)
            for i in os.listdir(btdfd_sol) if 'fTsol' in i
        ])
        self.abs1 = sorted([
            os.path.join(self.root, btdfd_sol, i)
            for i in os.listdir(btdfd_sol) if 'Abs1' in i
        ])
        self.abs2 = sorted([
            os.path.join(self.root, btdfd_sol, i)
            for i in os.listdir(btdfd_sol) if 'Abs2' in i
        ])

    def skycmd_win(self, cmd):
        cmd = cmd.replace('echo', '(echo')
        cmd = cmd.replace('"place', 'place')
        cmd = cmd.replace('\\n', '&echo ')
        cmd = cmd.replace('" | gend', ') | gend')
        return cmd

    def make_room(self):
        """Make a side-lit shoebox room."""
        self.theroom = room.make_room(self.dims)
        self.mlib = radutil.material_lib()
        self.sensor_grid = radutil.gen_grid(
            self.theroom.floor.flip(), self.grid_height, self.grid_spacing)
        self.nsensor = len(self.sensor_grid)
        self.sensor_grid.append(self.view.split())

    def make_matrices(self):
        self.vmxs = {}
        self.dmxs = {}
        self.clngmxs = {}
        self.flrmxs = {}
        self.wwmxs = {}
        self.ewmxs = {}
        self.nwmxs = {}
        env = self.mlib + self.theroom.srf_prims
        srfmx = [self.clngmxs, self.flrmxs, self.wwmxs, self.ewmxs, self.nwmxs]
        roomsrf = copy.deepcopy(self.theroom.srf_prims)
        roomsrf = [radutil.parse_polygon(i) for i in roomsrf]
        for srf in roomsrf[1:5]:
            srf['polygon'] = srf['polygon'].flip()
            srf['real_args'] = srf['polygon'].to_real()
        srfsndr = [radmtx.Sender.as_surface(prim_list=[i], basis='u') for i in roomsrf[:5]]
        for wname in self.theroom.swall.windows:
            ovmx = os.path.join(self.matricesd,
                                'vmx{}'.format(wname))
            odmx = os.path.join(self.matricesd,
                                'dmx{}'.format(wname))
            self.vmxs[wname] = ovmx
            self.dmxs[wname] = odmx
            for idx in range(5): # five interior surfaces
                _name = roomsrf[idx]['identifier']
                _out = os.path.join(self.matricesd, f"vsmx{_name}{wname}")
                srfmx[idx][wname] = _out
            if self.remake_matrices:
                wndw = self.theroom.wndw_prims[wname]
                wndw_rcvr = radmtx.Receiver.as_surface(
                    prim_list=[wndw], basis='kf', out=None)
                vmx_res = radmtx.rfluxmtx(
                    sender=radmtx.Sender.as_pts(pts_list=self.sensor_grid),
                    receiver=wndw_rcvr,
                    env=env,
                    opt=self.vmx_opt)
                with open(ovmx, 'wb') as wtr:
                    wtr.write(vmx_res)
                for idx in range(5): # five interior surfaces
                    _name = roomsrf[idx]['identifier']
                    _sender = srfsndr[idx]
                    _out = os.path.join(self.matricesd, f"vsmx{_name}{wname}")
                    _svmx_res = radmtx.rfluxmtx(
                        sender=_sender,
                        receiver=wndw_rcvr,
                        env=self.mlib,
                        opt=self.vsmx_opt,)
                    with open(_out, 'wb') as wtr:
                        wtr.write(_svmx_res)
                dmx_res = radmtx.rfluxmtx(
                    sender=radmtx.Sender.as_surface(prim_list=[wndw], basis='kf'),
                    receiver=radmtx.Receiver.as_sky(basis='r4'),
                    env=env,
                    opt=self.dmx_opt)
                with open(odmx, 'wb') as wtr:
                    wtr.write(dmx_res)

    def load_mtx(self):
        self._vmxs = [mtx_parser(self.vmxs[vmx]) for vmx in self.vmxs]
        self._dmxs = [mtx_parser(self.dmxs[dmx]) for dmx in self.dmxs]
        self._tvis = [mtx_parser(i) for i in self.btdfs_vis]
        self._tsol = [mtx_parser(i) for i in self.btdfs_tsol]
        self._ftsol = [mtx_parser(i) for i in self.btdfs_ftsol]
        self._abs1 = [mtx_parser(i) for i in self.abs1]
        self._abs2 = [mtx_parser(i) for i in self.abs2]
        self._shg = [mtx_parser(i) for i in self.btdfs_shgc]
        self._clngmxs = [mtx_parser(self.clngmxs[i]) for i in self.clngmxs]
        self._flrmxs = [mtx_parser(self.flrmxs[i]) for i in self.flrmxs]
        self._wwmxs = [mtx_parser(self.wwmxs[i]) for i in self.wwmxs]
        self._ewmxs = [mtx_parser(self.ewmxs[i]) for i in self.ewmxs]
        self._nwmxs = [mtx_parser(self.nwmxs[i]) for i in self.nwmxs]
        
    def wpi_map(self, data, center_w, center_d, fov):
        height = float(self.dims['height'])
        a = fov / 2
        view = np.tan(np.deg2rad(a)) * (height - self.grid_height)
        map_center = data.copy(deep=True)
        map_center.index = map_center.index - center_w
        map_center.columns = map_center.columns - center_d
        map_center = map_center.apply(lambda x: np.sqrt(x.index**2 + x.name**2))
        map_center[map_center > view] = -1
        map_center[(map_center <= view) & (map_center >= 0)] = 0
        map_center = map_center + 1
        map_center = map_center.replace(0, np.nan)
        return map_center.values
        
    def process_wpi(self, wpi_grid):
    
        if self.init:
            width = float(self.dims['width'])
            depth = float(self.dims['depth'])

            spacing = self.grid_spacing
            n_w = int(width / spacing)
            n_d = int(depth / spacing)
            dist_w = round((width - (n_w - 1) * spacing) / 2, 4)
            dist_d = round((depth - (n_d - 1) * spacing) / 2, 4)

            data = pd.DataFrame(wpi_grid.reshape(n_w, n_d))
            self.wpi_data = data.copy(deep=True)
            self.wpi_data.index = pd.Series([dist_w] + [spacing for _ in data.index[1:]]).cumsum().values
            self.wpi_data.columns = pd.Series([dist_d] + [spacing for _ in data.columns[1:]]).cumsum().values
            
            # Center
            center_w = round(width / 2, 4)
            center_d = round(depth / 2, 4)
            fov = 60 # deg
            self.map_center = self.wpi_map(self.wpi_data, center_w, center_d, fov)

            # Occupant
            occ_w = round(width / 2, 4)
            #occ_d = 1.525 # Occupant
            occ_d = round(depth / 3 * 2, 4)
            fov = 60 # deg
            self.map_23 = self.wpi_map(self.wpi_data, occ_w, occ_d, fov)
            
            self.n_w = n_w
            self.n_d = n_d
        
        data = pd.DataFrame(wpi_grid.reshape(self.n_w, self.n_d))
        data_center = data * self.map_center
        wpi_23 = data * self.map_23
        
        if self.wpi_plot:
            fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=False, sharey=False)
            axs[0].set_title('All WPI sensors (avg={})'.format(round(data.mean().mean(),1)))
            pos = axs[0].imshow(data.values, cmap='hot')
            axs[0].set_xticks(np.arange(0, self.n_d, 1))
            axs[0].set_xticklabels([round(self.wpi_data.columns[int(i)],1) for i in axs[0].get_xticks()])
            axs[0].set_yticks(np.arange(0, self.n_w, 1))
            axs[0].set_yticklabels([round(self.wpi_data.index[int(i)],1) for i in axs[0].get_yticks()])
            fig.colorbar(pos, ax=axs[0])
            
            axs[1].set_title('Cone view 60 deg room centered (avg={})'.format(round(data_center.mean().mean(),1)))
            pos = axs[1].imshow(data_center.values, cmap='hot')
            axs[1].set_xticks(np.arange(0, self.n_d, 1))
            axs[1].set_xticklabels([round(self.wpi_data.columns[int(i)],1) for i in axs[1].get_xticks()])
            axs[1].set_yticks(np.arange(0, self.n_w, 1))
            axs[1].set_yticklabels([round(self.wpi_data.index[int(i)],1) for i in axs[1].get_yticks()])
            fig.colorbar(pos, ax=axs[1])
            
            axs[2].set_title('Cone view 60 deg occupant centered (avg={})'.format(round(wpi_23.mean().mean(),1)))
            pos = axs[2].imshow(wpi_23.values, cmap='hot')
            axs[2].set_xticks(np.arange(0, self.n_d, 1))
            axs[2].set_xticklabels([round(self.wpi_data.columns[int(i)],1) for i in axs[2].get_xticks()])
            axs[2].set_yticks(np.arange(0, self.n_w, 1))
            axs[2].set_yticklabels([round(self.wpi_data.index[int(i)],1) for i in axs[2].get_yticks()])
            fig.colorbar(pos, ax=axs[2])
            
            fig.tight_layout()
            fig.savefig('wpi-plot_{}.jpeg'.format(self.date_time.strftime('%Y%m%dT%H%M')))
            

        
        wpi_all = data.mean().mean()
        wpi_center = data_center.mean().mean()
        wpi_23 = wpi_23.mean().mean()
        
        if self.wpi_loc == '23back':
            wpi = wpi_23
        elif self.wpi_loc == 'center':
            wpi = wpi_center
        elif self.wpi_loc == 'all':
            wpi = wpi_all
        else:
            print('ERROR: wpi_loc "{}" not available.'.format(self.wpi_loc))
            
        return wpi, wpi_all, wpi_center, wpi_23

    def compute(self, weather_df):
        """Compute the resulting illuminance and solar gain for all shade states.
        Arguments:
            weather_df: datetime indexed pandas dataframe with DNI and DHI data
        Return:
            wpi_df, shg_df, ev_df: Datetime indexed dataframes for each window zone and
            window state. wz0-3: bottom-to-top; ws0-n: sorted window state file name from
            small to large.
        """
        weather_df = weather_df.copy()
        col = weather_df.columns
        weather_df.loc[:, 'month'] = weather_df.index.month
        weather_df.loc[:, 'day'] = weather_df.index.day
        weather_df.loc[:,'hours'] = weather_df.index.hour + \
            weather_df.index.minute / 60
        _col = ['month', 'day', 'hours']
        _col.extend(col)
        _df = weather_df[_col]
        _df = _df.round({'hours': 3})
        sky_data = [' '.join(i.split()[2:]) for i in _df.to_string().splitlines()[1:]]
        ntime = len(sky_data)
        vissky = makesky.gendaymtx(sky_data, self.lat,
                                   self.lon, self.timezone,
                                   self.elevation, rotate=self.orient)[0]
        solsky = makesky.gendaymtx(sky_data, self.lat,
                                   self.lon, self.timezone,
                                   self.elevation, solar=True,
                                   rotate=self.orient)[0]
        viscvt = 'rmtxop -fa -c 47.4 119.9 11.6'
        solcvt = 'rmtxop -fa -c .265 .67 .065'
        shgccvt = 'rmtxop -fa -c 1 0 0'
        wrapsky = '"' if os.name == 'nt' else "\'"
        cmds = []
        for wname in self.theroom.swall.windows:
            wndw_zone = self.theroom.swall.windows[wname]
            for idx_t, btdf_vis in enumerate(self.btdfs_vis):
                vis_cmd = "rmtxop {} {} {} {}!{}{} | {} - | getinfo - ".format(
                    self.vmxs[wname], btdf_vis, self.dmxs[wname],
                    wrapsky, vissky, wrapsky, viscvt)
                shg_cmd = "rmtxop {} {} {}!{}{} | {} - | rmtxop -fa -s {} - | getinfo -".format(
                    self.btdfs_shgc[idx_t], self.dmxs[wname], wrapsky, solsky,
                    wrapsky, shgccvt, wndw_zone.area())
                if os.name == 'nt':
                    vis_cmd = self.skycmd_win(vis_cmd)
                    shg_cmd = self.skycmd_win(shg_cmd)
                cmds.append(vis_cmd)
                cmds.append(shg_cmd)
        if self.parallel:
            process = mp.Pool(mp.cpu_count())
            raw = [i.decode().split() for i in process.map(checkout, cmds)]
        else:
            #raw = [checkout(cmd).decode().split() for cmd in cmds]

            raw = []
            for cmd in cmds:
                st = time.time()
                raw.append(checkout(cmd).decode().split())
                et = time.time()
                print(cmd[:10], round(et - st, 2))

        col = [
            '{}_{}'.format(zone, state)
            for zone in range(len(self.theroom.swall.windows))
            for state in range(len(self.btdfs_vis))
        ]
        vis_df = pd.DataFrame(raw[::2], dtype=float).transpose()
        wpi_df = vis_df.iloc[:-ntime]
        awpi_df = pd.DataFrame(
            [wpi_df.iloc[n::ntime].mean() for n in range(ntime)],
            index=_df.index)
        awpi_df.columns = ['wpi_' + c for c in col]
        ev_df = vis_df.iloc[-ntime:].set_index(_df.index)
        ev_df.columns = ['vil_' + c for c in col]
        shg_df = pd.DataFrame(raw[1::2],
                              dtype=float).transpose().set_index(_df.index)
        shg_df.columns = ['shg_' + c for c in col]
        return pd.concat([awpi_df, shg_df, ev_df], axis=1)

    def compute2(self, weather_df, weather_cutoff=75):
        """Compute the resulting illuminance and solar gain for all shade states.
        Arguments:
            weather_df: datetime indexed pandas dataframe with DNI and DHI data
            weather_cutoff: cutoff for weather forecast, W/m2
        Return:
            output_df: Datetime indexed dataframes for each window zone and
            window state. wz0-3: bottom-to-top; ws0-n: sorted window state file name from
            small to large.
        """
        weather_df = weather_df.copy()

        # Filter weather
        wather_filter = (weather_df['DNI'] <= weather_cutoff) \
            & (weather_df['DHI'] <= weather_cutoff)
        weather_df['DNI'] = weather_df['DNI'].mask( \
            wather_filter, 0)
        weather_df['DHI'] = weather_df['DHI'].mask( \
            wather_filter, 0)

        col = weather_df.columns
        weather_df.loc[:, 'month'] = weather_df.index.month
        weather_df.loc[:, 'day'] = weather_df.index.day
        weather_df.loc[:,'hours'] = weather_df.index.hour + \
            weather_df.index.minute / 60
        _col = ['month', 'day', 'hours']
        _col.extend(col)
        _df = weather_df[_col]
        _df = _df.round({'hours': 3})
        sky_data = [
            ' '.join(i.split()[2:]) for i in _df.to_string().splitlines()[1:]
        ]
        #_sky_data = f'gendaylit {_df.index.month} {_df.index.day} {hours} -a {self.lat} -o {self.lon} '
        #_sky_data += f'-m {self.timezone} -W {int(DNI)} {int(DHI)} '
        #vissky = _sky_data + f'-O 0 | xform -rz {self.orient}'
        #solsky = _sky_data + f'-O 1 | xform -rz {self.orient}'
        vissky = makesky.gendaymtx(sky_data, self.lat, self.lon, self.timezone,
                                   self.elevation, rotate=self.orient)[0]
        solsky = makesky.gendaymtx(sky_data, self.lat, self.lon, self.timezone,
                                   self.elevation, solar=True, rotate=self.orient)[0]
        tmpd = tf.mkdtemp(prefix='sky')
        tmpwea = os.path.join(tmpd, 'wea')
        tmpvis = os.path.join(tmpd, 'vis')
        tmpsol = os.path.join(tmpd, 'sol')
        weavis, genvis = vissky.split('| ')
        _, gensol = solsky.split('| ')
        with open(tmpwea, 'w') as wtr:
            wtr.write(weavis[6:].replace('\\n', '\n')[:-2])
        vis_cmd = genvis.split() + [tmpwea]
        sol_cmd = gensol.split() + [tmpwea]
        vissky_result = sp.run(vis_cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        with open(tmpvis, 'wb') as wtr:
            wtr.write(vissky_result.stdout)
        solsky_result = sp.run(sol_cmd, stderr=sp.PIPE, stdout=sp.PIPE)
        with open(tmpsol, 'wb') as wtr:
            wtr.write(solsky_result.stdout)
        #sp.run('{} | genskyvec -m 4 > {}'.format(vissky, tmpvis), shell=True)
        #sp.run('{} | genskyvec -m 4 > {}'.format(solsky, tmpsol), shell=True)
        _vis = smx_parser(tmpvis)
        _sol = smx_parser(tmpsol)
        output_df = pd.DataFrame(index=_df.index)
        flr_area = self.theroom.floor.area()
        for idx in range(len(self.theroom.swall.windows)):
            _wndw_area = self.theroom.swall.windows['window{}'.format(
                idx + 1)].area()
            for idx_t, _ in enumerate(self.btdfs_vis):
                wpi_col = 'wpi_{}_{}'.format(idx, idx_t)
                ev_col = 'ev_{}_{}'.format(idx, idx_t)
                shg_col = 'shg_{}_{}'.format(idx, idx_t)
                tsol_col = 'tsol_{}_{}'.format(idx, idx_t)
                abs1_col = 'abs1_{}_{}'.format(idx, idx_t)
                abs2_col = 'abs2_{}_{}'.format(idx, idx_t)
                iflr_col = 'iflr_{}_{}'.format(idx, idx_t)
                visres = radutil.mtxmult([
                    self._vmxs[idx], self._tvis[idx_t], self._dmxs[idx], _vis]) * 179
                    
                # Process WPI
                wpi = []
                for i, t in enumerate(np.transpose(visres[:150])):
                    self.date_time = weather_df.index[i]
                    wpi_temp = self.process_wpi(t)
                    wpi.append(wpi_temp[0])
                #output_df[wpi_col] = np.mean(visres[:150], axis=0)
                output_df[wpi_col] = wpi
                output_df[ev_col] = visres[-1]
                output_df[shg_col] = radutil.mtxmult(
                    [self._shg[idx_t], self._dmxs[idx], _sol])[0] * _wndw_area
                abs1 = radutil.mtxmult(
                    [self._abs1[idx_t], self._dmxs[idx], _sol])[0] * _wndw_area
                abs2 = radutil.mtxmult(
                    [self._abs2[idx_t], self._dmxs[idx], _sol])[0] * _wndw_area
                output_df[tsol_col] = (radutil.mtxmult([self._ftsol[idx_t], self._dmxs[idx], _sol]).transpose()*self.klems_coeff).sum(axis=1) * _wndw_area
                output_df[iflr_col] = radutil.mtxmult(
                    [self._flrmxs[idx], self._ftsol[idx_t], self._dmxs[idx], _sol])[0] * np.pi * flr_area
                output_df[abs1_col] = abs1
                output_df[abs2_col] = abs2
                

                
                
                
        shutil.rmtree(tmpd)
        
        # Process if shade
        if 'shade' in self.facade_type and not self._test_difference:
            if not self.new_map:
                cols = output_df.columns
                for prefix in np.unique([c.split('_')[0] for c in cols]):
                    for i, v in enumerate(list(self.shade_map.values())[:-1]):
                        self.new_map[f'{prefix}_0_{i}'] = [f'{prefix}_{ii}_{vv}' for ii, vv in enumerate(v)]
                    i += 1
                    self.new_map[f'{prefix}_0_{i}'] = [f'{prefix}_{ii}_1' for ii, vv in enumerate(v)]
            temp = output_df.copy(deep=True)
            for k, v in self.new_map.items():
                output_df[k] = temp[v].sum(axis=1)
            output_df = output_df[self.new_map.keys()]
            
        self.init = False
        
        return output_df


if __name__ == "__main__":
    import sys
    import time
    from configs import get_config
    root = os.path.dirname(os.path.abspath(__file__))


    '''
    config_path = os.path.join(root, '..', '..', 'resources', 'radiance','room0.6WWR_blinds.cfg')

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    filestruct = {}
    filestruct['resources'] = os.path.join(os.path.dirname(config_path), 'BSDF_blinds0.6')
    filestruct['matrices'] = os.path.join(os.path.dirname(config_path), 'matrices', 'blinds0.6')
    '''

    wwr = 0.6 # [0.4, 0.6]
    mode = 'dshade' # ['shade', 'blinds', 'ec']
    facade_type = 'shade' # ['shade', 'blinds', 'ec']
    filestruct, config_path = get_config(mode, wwr)
    forecaster = Forecast(config_path, facade_type=facade_type, regenerate=False, filestruct=filestruct, wpi_plot=True, wpi_loc='23back')
    '''
    for i in range(24*12-2):
        df = df.append(df)
    '''
    for i in range(3):
        #df = pd.DataFrame([[800, 80], [750, 100], [100, 10]],
        #                       index=pd.DatetimeIndex([
        #                           pd.datetime(2019, 12, 21, 12, 0),
        #                           pd.datetime(2019, 12, 21, 12, 10),
        #                           pd.datetime(2019, 12, 21, 12, 20),
        #                       ]))
        df = pd.DataFrame(index=pd.date_range('2020-01-01 00:00',
                                              '2020-01-02 00:00',
                                              freq='5T'))
        #df = pd.DataFrame(index=pd.date_range('2020-01-01 00:00',
        #                                      '2020-01-01 00:05',
        #                                      freq='5T'))
        np.random.seed(1)
        df['DNI'] = np.random.uniform(0, 1000, len(df))
        np.random.seed(1)
        df['DHI'] = np.random.uniform(0, 250, len(df))
        
        fake_df = pd.DataFrame([[185.5, 42], [165.3, 38.3]],
                       index=pd.DatetimeIndex([pd.datetime(2017, 9, 1, 18, 15),
                                               pd.datetime(2017, 9, 1, 18, 20)]),
                       columns=['DNI','DHI'])
        
        
        st = time.time()
        #res = forecaster.compute(df)
        res = forecaster.compute2(fake_df)
        print(time.time() - st)
    #print('Forecast of 24h x 5min in {} s'.format(round(time.time() - st, 1)))
    #print(res.columns)
    #for k,v in forecaster.new_map.items():
    #    print(k, v)
    print(res)
    print(res['wpi_0_6'])
