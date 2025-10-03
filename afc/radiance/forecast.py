# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

"""
Advanced Fenestration Controller
Radiance forecasting module.
"""

# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-arguments
# pylint: disable=redefined-outer-name, invalid-name, too-many-statements
# pylint: disable=consider-using-dict-items, protected-access, pointless-string-statement
# pylint: disable=import-outside-toplevel, too-many-positional-arguments, too-many-branches
# pylint: disable=dangerous-default-value, consider-using-generator, unused-variable
# pylint: disable=wrong-import-position

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1' # this is faster in most situations (np.linalg.multi_dot)

import time
import json
#import multiprocessing as mp
from configparser import ConfigParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import frads as fr
from frads import room, matrix, methods, geom
from frads.methods import MatrixConfig
import pyradiance as pr

from afc.radiance.maps import make_ctrl_map
from afc.radiance.constants import KFOMG, OMEGAS
from afc.radiance.utility import create_trapezoid_mask

root = os.path.dirname(os.path.realpath(__file__))

# versions
with open(os.path.join(root, "..", "__init__.py"), "r", encoding="utf8") as f:
    AFC_VERSION = json.loads(f.read().split("__version__ = ")[1])
FRADS_VERSION = fr.__version__


class Forecast:
    """Radiance forecaster class for the AFC."""

    def __init__(
        self,
        cfg_path,
        regenerate=None,
        facade_type="ec",
        window_ctrl_map={},
        wpi_plot=False,
        wpi_loc="23back",
        wpi_all=False,
        wpi_config={'grid_height': 0.76, 'grid_spacing': 0.3},
        view_config={'view_dist': 1.22, 'view_orient': 's'},
        location=None,
        filestruct=None,
        _test_difference=False,
        dimensions=None,
        render=False,
        reflectances={'floor': 0.2, 'walls': 0.5, 'ceiling': 0.7},
        n_cpus=-1
    ):
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.parse_config(cfg_path)
        if regenerate:
            self.remake_matrices = True
        if filestruct:
            self.filestruct = filestruct
            self.glzsysd = self.filestruct["glazing_systems"]
            self.matricesd = self.filestruct["matrices"]
        if location:
            self.lat = location["latitude"]
            self.lon = location["longitude"]
            self.timezone = location["timezone"]
            self.elevation = location["elevation"]
            self.orient = location["orient"]
        if dimensions:
            # delete windows
            for k in [k for k in self.dims if k.startswith('window')]:
                del self.dims[k]
            # add new config
            self.dims.update(dimensions)
        self.facade_type = facade_type
        self.window_ctrl_map = window_ctrl_map
        self.wpi_plot = wpi_plot
        self.wpi_loc = wpi_loc
        self.wpi_all = wpi_all
        self.wpi_config = wpi_config
        self.view_config = view_config
        self._test_difference = _test_difference
        self.render = render
        self.reflectances = reflectances
        self.n_cpus = n_cpus

        # define cpus:
        if self.n_cpus > 0:
            os.environ['OPENBLAS_NUM_THREADS'] = str(int(self.n_cpus))

        # make wpi grid config
        self.grid_height = float(self.wpi_config['grid_height'])
        self.grid_spacing = float(self.wpi_config['grid_spacing'])

        # make view
        self.make_view()

        # init
        self.glazing_system_paths = sorted(
            [
                os.path.join(self.root, self.glzsysd, i)
                for i in os.listdir(self.glzsysd)
                if i.endswith(".json")
            ]
        )
        if facade_type == 'blinds': # reverse order for blinds
            self.glazing_system_paths = self.glazing_system_paths[::-1]
        self.facade_states = {k:os.path.split(v)[-1].split('.json')[0]
            for k, v in enumerate(self.glazing_system_paths)}
        self.facade_states_inv = {v:k for k,v in self.facade_states.items()}
        self.glazing_systems = [
            fr.window.load_glazing_system(i) for i in self.glazing_system_paths
        ]

        # make window_ctrl_map
        self.window_zones = len([k for k in self.dims if k.startswith('window')])
        self.logical_windows = self.window_zones if facade_type == 'ec' else 1
        if not self.window_ctrl_map:
            self.window_ctrl_map = make_ctrl_map(facade_type,
                                                 self.facade_states.values(),
                                                 self.window_zones,
                                                 self.logical_windows)
        self.window_ctrl_map_n = {k:[self.facade_states_inv[vv] for vv in v]
            for k,v in self.window_ctrl_map.items()}
        self.logical_states = list(range(len(self.window_ctrl_map)))
        if facade_type == 'ec':
            self.logical_window_states = list(range(len(self.facade_states)))
        else:
            self.logical_window_states = self.logical_states

        self.make_room()
        self.get_workflow()
        self.make_matrices()
        self.get_matrices()

        #self.wwr = float(cfg_path.split("room")[1].split("WWR")[0])
        #if "shade" in self.facade_type and float(self.wwr) == 0.6:
        #    self.shade_map = shade_map_0x6
        #elif "shade" in self.facade_type and float(self.wwr) == 0.4:
        #    self.shade_map = shade_map_0x4
        self.new_map = {}
        self.init = True

        # Initialize variables
        self.wpi_data = None
        self.map_center = None
        self.map_23 = None
        self.map_trap = None
        self.n_w = None
        self.n_d = None
        self.date_time = None

    def parse_config(self, cfg_path):
        """Function to parse configuration file."""

        config = ConfigParser()
        if not config.read(cfg_path):
            raise ValueError(
                'The location of the "config" file is wrong. '
                + f"Check location at {cfg_path}."
            )
        cfg = config._sections
        site_info = cfg["Site"]
        self.lat = site_info["latitude"]
        self.lon = site_info["longitude"]
        self.timezone = site_info["timezone"]
        self.elevation = site_info["elevation"]
        self.orient = site_info["orientation"]
        self.dims = cfg["Dimensions"]
        self.filestruct = cfg["FileStructure"]
        self.matricesd = os.path.join(self.root, self.filestruct["matrices"])
        self.bsdfd = os.path.join(self.root, self.filestruct["resources"], "BSDFs")
        simctrl = cfg["SimulationControl"]
        self.parallel = simctrl["parallel"] == "True"
        self.vmx_opt = simctrl["vmx"]
        self.vsmx_opt = simctrl["vsmx"]
        self.dmx_opt = simctrl["dmx"]
        self.remake_matrices = simctrl["re-make_matrices"] == "True"
        self.view = cfg["View"]["view1"]
        self.grid_height = float(cfg["Grid"]["height"])
        self.grid_spacing = float(cfg["Grid"]["spacing"])

    def make_view(self, height=1.22):
        '''Funciton to make a view'''

        # sky will be rotated
        direction_map = {'s': '0 -1',
                         'n': '0 1',
                         'w': '-1 0',
                         'e': '1 0'}
        center = float(self.dims['width']) / 2
        view_dist = self.view_config['view_dist']
        view_orient = self.view_config['view_orient']
        self.view = f'{center} {view_dist} {height} {direction_map[view_orient]} 0'

    def make_room(self):
        """Function to make a side-lit shoebox room."""

        windows = [
            [float(n) for n in v.split()] for k, v in self.dims.items() if "window" in k
        ]
        self.theroom = room.create_south_facing_room(
            width=float(self.dims["width"]),
            depth=float(self.dims["depth"]),
            floor_floor=float(self.dims["height"]) * 1.3,
            floor_ceiling=float(self.dims["height"]),
            swall_thickness=float(self.dims["facade_thickness"]),
            wpd=windows,
        )

        # reflectances
        self.theroom.materials[0].fargs = [self.reflectances['floor']]*3 + [0, 0] # floor
        self.theroom.materials[1].fargs = [self.reflectances['walls']]*3 + [0, 0] # walls
        self.theroom.materials[2].fargs = [self.reflectances['ceiling']]*3 + [0, 0] # ceiling

        self.sensor_grid = geom.gen_grid(
            self.theroom.floor.base, self.grid_height, self.grid_spacing
        )

    def get_workflow(self):
        """Function to create workflow."""
        _view = self.view.split()
        model = self.theroom.model_dump()
        model["sensors"] = {
            "wpi": {"data": self.sensor_grid},
            #"view": {"data": [_view]},
        }

        # add view
        view = pr.create_default_view()
        view.vp = tuple([float(x) for x in _view[0:3]])
        view.vdir = tuple([float(x) for x in _view[3:6]])
        view.type = "a"
        view.horiz = 180
        view.vert = 180
        model["views"] = {"view": {"view": view}}

        # complete settings
        settings = {
            "method": 3,
            "latitude": self.lat,
            "longitude": self.lon,
            "time_zone": self.timezone,
            "site_elevation": self.elevation,
            "save_matrices": True,
            "output_directory": self.matricesd,
            #"sensor_window_matrix": self.vmx_opt.split(),
            #"daylight_matrix": self.dmx_opt.split(),
            #"surface_window_matrix": self.vsmx_opt.split(),
            "daylight_matrix": ['-ab', '2', '-c', '5000'],
            "sensor_window_matrix": ['-ab', '5', '-ad', '8192', '-lw', '5e-5'],
            "surface_window_matrix": ['-ab', '5', '-ad', '8192', '-lw', '5e-5', '-c', '10000'],
            "sky_basis": "r4",
            "name": f"afc_version={AFC_VERSION}, frads_version={FRADS_VERSION}",
        }
        self.workflow_config = methods.WorkflowConfig.from_dict(
            {"model": model, "settings": settings}
        )
        self.workflow = methods.ThreePhaseMethod(self.workflow_config)

    def make_matrices(self):
        """Function to make matrices."""
        if self.remake_matrices:
            self.workflow.mfile.unlink(missing_ok=True)
        self.workflow.generate_matrices(view_matrices=False)

    def integrate_matrix(self, matrix):
        """Convert from BSDF to transmission matrix"""
        ncol = len(matrix[0])
        # nrow = len(matrix)
        if ncol == 145:
            solid_angles = OMEGAS["kf"]
        elif ncol == 73:
            solid_angles = OMEGAS["kh"]
        elif ncol == 41:
            solid_angles = OMEGAS["kq"]
        else:
            raise KeyError("Unknown bsdf basis")
        assert len(solid_angles) == ncol
        # assert len(solid_angles) == nrow
        return np.array(
            [
                [
                    [ele * omg, ele * omg, ele * omg]
                    for ele, omg in zip(row, solid_angles)
                ]
                for row in matrix
            ]
        )

    def convert_mtx_to_frads(self, mtx, integrate=True):
        """Convert plain mtx array to frads matrix object"""
        if integrate:
            return MatrixConfig(matrix_data=self.integrate_matrix(mtx))
        return MatrixConfig(matrix_data=mtx)

    def get_matrices(self):
        """load the matrices"""
        if self.render:
            for gs in self.glazing_systems:
                visible_name = f"{gs.name}_visible"
                solar_name = f"{gs.name}_solar"
                abs1_name = f"{gs.name}_abs1"
                abs2_name = f"{gs.name}_abs2"
                visible_mtx = self.convert_mtx_to_frads(gs.visible_front_transmittance)
                self.workflow.config.model.materials.matrices[visible_name] = visible_mtx
                solar_mtx = self.convert_mtx_to_frads(gs.solar_front_transmittance)
                self.workflow.config.model.materials.matrices[solar_name] = solar_mtx
                abs1_array = np.array(gs.solar_front_absorptance[0]).reshape(1, -1)
                abs1_mtx = self.convert_mtx_to_frads(abs1_array)
                self.workflow.config.model.materials.matrices[abs1_name] = abs1_mtx
                abs2_array = np.array(gs.solar_front_absorptance[1]).reshape(1, -1)
                abs2_mtx = self.convert_mtx_to_frads(abs2_array)
                self.workflow.config.model.materials.matrices[abs2_name] = abs2_mtx
                for window in self.workflow.config.model.windows.values():
                    window_polygon = geom.parse_polygon(pr.parse_primitive(window.bytes)[0])
                    _geom = fr.window.get_proxy_geometry(window_polygon, gs)
                    window.proxy_geometry[visible_name] = b"\n".join(_geom)

        self._tvis = [
            self.integrate_matrix(gs.visible_front_transmittance)
            for gs in self.glazing_systems
        ]
        self._tsol = [
            self.integrate_matrix(gs.solar_front_transmittance)
            for gs in self.glazing_systems
        ]
        self._abs1 = [
            self.integrate_matrix(
                np.array(gs.solar_front_absorptance[0]).reshape(1, -1)
            )
            for gs in self.glazing_systems
        ]
        self._abs2 = [
            self.integrate_matrix(
                np.array(gs.solar_front_absorptance[1]).reshape(1, -1)
            )
            for gs in self.glazing_systems
        ]

    def wpi_map(self, data, center_w, center_d, fov):
        """Function to make wpi map."""

        height = float(self.dims["height"])
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

    def wpi_map_trapezoid(self, data, slope):
        """Function to make wpi map."""

        map_center = data.copy(deep=True)
        y = map_center.columns
        x = map_center.index
        mask = create_trapezoid_mask(x, y,
                                     p1=(x[0], y[0]), # fixed
                                     p2=(x[-1], y[0]), # fixed
                                     p3=(x[-1]-slope+x[0], y[-1]), # compensate for offsets
                                     p4=(slope, y[-1])) # absolute
        map_center.loc[:, :] = np.transpose(mask)
        map_center = map_center.replace(0, np.nan)
        return map_center.values

    def process_wpi(self, wpi_grid, index=0):
        """Function to process wpi."""

        if self.init and index == 0:
            width = float(self.dims["width"])
            depth = float(self.dims["depth"])

            spacing = self.grid_spacing
            n_w = int(width / spacing)
            n_d = int(depth / spacing)
            dist_w = round((width - (n_w - 1) * spacing) / 2, 4)
            dist_d = round((depth - (n_d - 1) * spacing) / 2, 4)

            data = pd.DataFrame(wpi_grid.reshape(n_w, n_d))
            self.wpi_data = data.copy(deep=True)
            self.wpi_data.index = (
                pd.Series([dist_w] + [spacing for _ in data.index[1:]]).cumsum().values
            )
            self.wpi_data.columns = (
                pd.Series([dist_d] + [spacing for _ in data.columns[1:]])
                .cumsum()
                .values
            )

            # room center
            center_w = round(width / 2, 4)
            center_d = round(depth / 2, 4)
            fov = 60  # deg
            self.map_center = self.wpi_map(self.wpi_data, center_w, center_d, fov)

            # 2/3 back
            occ_w = round(width / 2, 4)
            # occ_d = 1.525 # Occupant
            occ_d = round(depth / 3 * 2, 4)
            fov = 60  # deg
            self.map_23 = self.wpi_map(self.wpi_data, occ_w, occ_d, fov)

            # trapezoid in back (m on y-axis)
            slope = float(self.wpi_loc.split('trap')[1]) if self.wpi_loc.startswith('trap') else 1
            self.map_trap = self.wpi_map_trapezoid(self.wpi_data, slope)

            self.n_w = n_w
            self.n_d = n_d

        data = pd.DataFrame(wpi_grid.reshape(self.n_w, self.n_d))
        data_center = data * self.map_center
        wpi_23 = data * self.map_23
        wpi_trap = data * self.map_trap

        if self.wpi_plot and index == 0:

            def plot_wpi(axs, data, title=''):
                axs.set_title(title)
                pos = axs.imshow(data, cmap="hot")
                axs.set_xticks(np.arange(0, self.n_d, 1))
                axs.set_xticklabels(
                    [round(self.wpi_data.columns[int(i)], 1) for i in axs.get_xticks()]
                )
                axs.set_yticks(np.arange(0, self.n_w, 1))
                axs.set_yticklabels(
                    [round(self.wpi_data.index[int(i)], 1) for i in axs.get_yticks()]
                )
                fig.colorbar(pos, ax=axs)

            fig, axs = plt.subplots(4, 1, figsize=(6, 12), sharex=False, sharey=False)

            # all avg
            tt_avg = round(data.mean().mean(), 1)
            plot_wpi(axs[0], data.values,
                     title=f"All WPI sensors (avg={tt_avg})")

            # cone 60 deg room centered
            tt_avg = round(data_center.mean().mean(), 1)
            plot_wpi(axs[1], data_center.values,
                     title=f"Cone view 60 deg room centered (avg={tt_avg})")

            # cone 60 deg 2/3 back centered
            tt_avg = round(wpi_23.mean().mean(), 1)
            plot_wpi(axs[2], wpi_23.values,
                     title=f"Cone view 60 deg 2/3 back centered (avg={tt_avg})")

            # trapezoid centered
            tt_avg = round(wpi_trap.mean().mean(), 1)
            plot_wpi(axs[3], wpi_trap.values,
                     title=f"Trapezoid centered (avg={tt_avg})")

            fig.tight_layout()
            tt_dt = self.date_time.strftime("%Y%m%dT%H%M")
            fig.savefig(f"wpi-plot_{tt_dt}.jpeg")
            plt.close(fig)

        wpi_avg = data.mean().mean()
        wpi_center = data_center.mean().mean()
        wpi_23 = wpi_23.mean().mean()
        wpi_trap = wpi_trap.mean().mean()

        if self.wpi_loc == "23back":
            wpi = wpi_23
        elif self.wpi_loc == "center":
            wpi = wpi_center
        elif self.wpi_loc == "avg":
            wpi = wpi_avg
        elif self.wpi_loc.startswith('trap'):
            wpi = wpi_trap
        else:
            print(f'ERROR: wpi_loc "{self.wpi_loc}" not available.')
            wpi = None

        return wpi, data, [wpi_avg, wpi_center, wpi_23, wpi_trap]

    '''
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
        weather_df.loc[:, "month"] = weather_df.index.month
        weather_df.loc[:, "day"] = weather_df.index.day
        weather_df.loc[:, "hours"] = (
            weather_df.index.hour + weather_df.index.minute / 60
        )
        _col = ["month", "day", "hours"]
        _col.extend(col)
        _df = weather_df[_col]
        _df = _df.round({"hours": 3})
        sky_data = [" ".join(i.split()[2:]) for i in _df.to_string().splitlines()[1:]]
        ntime = len(sky_data)
        vissky = makesky.gendaymtx(
            sky_data,
            self.lat,
            self.lon,
            self.timezone,
            self.elevation,
            rotate=self.orient,
        )[0]
        solsky = makesky.gendaymtx(
            sky_data,
            self.lat,
            self.lon,
            self.timezone,
            self.elevation,
            solar=True,
            rotate=self.orient,
        )[0]
        viscvt = "rmtxop -fa -c 47.4 119.9 11.6"
        # solcvt = 'rmtxop -fa -c .265 .67 .065'
        shgccvt = "rmtxop -fa -c 1 0 0"
        wrapsky = '"' if os.name == "nt" else "'"
        cmds = []
        for wname in self.theroom.swall.windows:
            wndw_zone = self.theroom.swall.windows[wname]
            for idx_t, btdf_vis in enumerate(self.btdfs_vis):
                vis_cmd = f"rmtxop {self.vmxs[wname]} {btdf_vis} {self.dmxs[wname]}"
                vis_cmd += f" {wrapsky}!{vissky}{wrapsky} | {viscvt} - | getinfo - "
                shg_cmd = f"rmtxop {self.btdfs_shgc[idx_t]} {self.dmxs[wname]}"
                shg_cmd += f" {wrapsky}!{solsky}{wrapsky} | {shgccvt} - |"
                shg_cmd += f" rmtxop -fa -s {wndw_zone.area()} - | getinfo -"
                if os.name == "nt":
                    vis_cmd = self.skycmd_win(vis_cmd)
                    shg_cmd = self.skycmd_win(shg_cmd)
                cmds.append(vis_cmd)
                cmds.append(shg_cmd)
        if self.parallel:
            with mp.Pool(mp.cpu_count()) as pool:
                raw = [i.decode().split() for i in pool.map(checkout, cmds)]
        else:
            # raw = [checkout(cmd).decode().split() for cmd in cmds]

            raw = []
            for cmd in cmds:
                st = time.time()
                raw.append(checkout(cmd).decode().split())
                et = time.time()
                print(cmd[:10], round(et - st, 2))

        col = [
            f"{zone}_{state}"
            for zone in range(len(self.theroom.swall.windows))
            for state in range(len(self.btdfs_vis))
        ]
        vis_df = pd.DataFrame(raw[::2], dtype=float).transpose()
        wpi_df = vis_df.iloc[:-ntime]
        awpi_df = pd.DataFrame(
            [wpi_df.iloc[n::ntime].mean() for n in range(ntime)], index=_df.index
        )
        awpi_df.columns = ["wpi_" + c for c in col]
        ev_df = vis_df.iloc[-ntime:].set_index(_df.index)
        ev_df.columns = ["vil_" + c for c in col]
        shg_df = pd.DataFrame(raw[1::2], dtype=float).transpose().set_index(_df.index)
        shg_df.columns = ["shg_" + c for c in col]
        return pd.concat([awpi_df, shg_df, ev_df], axis=1)
    '''

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
        wather_filter = (weather_df["dni"] <= weather_cutoff) & (
            weather_df["dhi"] <= weather_cutoff
        )
        weather_df["dni"] = weather_df["dni"].mask(wather_filter, 0)
        weather_df["dhi"] = weather_df["dhi"].mask(wather_filter, 0)

        col = weather_df.columns
        weather_df.loc[:, "month"] = weather_df.index.month
        weather_df.loc[:, "day"] = weather_df.index.day
        weather_df.loc[:, "hours"] = (
            weather_df.index.hour + weather_df.index.minute / 60
        )
        _col = ["month", "day", "hours"]
        _col.extend(col)
        _df = weather_df[_col]
        _df = _df.round({"hours": 3})

        _vis = self.workflow.get_sky_matrix(
            time=list(weather_df.index.tolist()),
            dni=weather_df["dni"].tolist(),
            dhi=weather_df["dhi"].tolist(),
            solar_spectrum=False,
            orient=self.orient
        )
        _sol = self.workflow.get_sky_matrix(
            time=list(weather_df.index.tolist()),
            dni=weather_df["dni"].tolist(),
            dhi=weather_df["dhi"].tolist(),
            solar_spectrum=True,
            orient=self.orient
        )
        output_df = pd.DataFrame(index=_df.index)
        flr_area = np.linalg.norm(self.theroom.floor.base.area)

        if self.render:
            for i, ix in enumerate(weather_df.index):
                if 10 <= ix.hour <= 18:
                    wdw_state = list(self.workflow.config.model.materials.matrices.keys())[0]
                    fc_state = {}
                    for w in self.workflow.config.model.windows.keys():
                        fc_state[w] = wdw_state
                    edgps, ev = self.workflow.calculate_edgps(
                        'view',
                        fc_state,
                        ix,
                        dni=weather_df.loc[ix, 'dni'],
                        dhi=weather_df.loc[ix, 'dhi'],
                        ambient_bounce=1,
                        save_hdr=f'ctrl_{ix}.hdr',
                    )
                    pr.ra_tiff(pr.pcond(f'ctrl_{ix}.hdr', human=True), out=f'ctrl_{ix}.tif')
        for idx, window in enumerate(self.theroom.swall.windows):
            _wndw_area = np.linalg.norm(window.polygon.area)
            _window_name = window.primitive.identifier
            for idx_t, _ in enumerate(self.glazing_systems):
                wpi_col = f"wpi_{idx}_{idx_t}"
                wpi_all_col = f"wpi-all_{idx}_{idx_t}"
                ev_col = f"ev_{idx}_{idx_t}"
                tsol_col = f"tsol_{idx}_{idx_t}"
                abs1_col = f"abs1_{idx}_{idx_t}"
                abs2_col = f"abs2_{idx}_{idx_t}"
                iflr_col = f"iflr_{idx}_{idx_t}"
                visres = matrix.matrix_multiply_rgb(
                    self.workflow.sensor_window_matrices["wpi"].array[idx],
                    self._tvis[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _vis,
                    weights=[47.4, 119.9, 11.6],
                )

                # process ev
                evres = matrix.matrix_multiply_rgb(
                    self.workflow.sensor_window_matrices["view"].array[idx],
                    self._tvis[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _vis,
                    weights=[47.4, 119.9, 11.6],
                )
                output_df[ev_col] = evres[0]

                # process wpi (iterate through horizon)
                wpi = []
                wpi_all = []
                for i, t in enumerate(np.transpose(visres)):
                    self.date_time = weather_df.index[i]
                    wpi_temp = self.process_wpi(t, index=idx+idx_t+i)
                    wpi.append(wpi_temp[0])
                    if self.wpi_all:
                        wpi_all.append(wpi_temp[1].to_json())
                output_df[wpi_col] = wpi
                if self.wpi_all:
                    output_df[wpi_all_col] = wpi_all

                # process solar
                abs1 = matrix.matrix_multiply_rgb(
                    self._abs1[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _sol,
                    weights=[0.3, 0.4, 0.3],
                )[0]
                abs2 = matrix.matrix_multiply_rgb(
                    self._abs2[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _sol,
                    weights=[0.3, 0.4, 0.3],
                )[0]
                tsol = matrix.matrix_multiply_rgb(
                    self._tsol[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _sol,
                    weights=[0.3, 0.4, 0.3],
                )
                iflr = matrix.matrix_multiply_rgb(
                    self.workflow.surface_window_matrices["floor"].array[idx],
                    self._tsol[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _sol,
                    weights=[0.3, 0.4, 0.3],
                )[0]
                output_df[tsol_col] = (tsol.T * KFOMG).sum(axis=1) * _wndw_area
                output_df[iflr_col] = iflr * np.pi * flr_area
                output_df[abs1_col] = abs1 * _wndw_area
                output_df[abs2_col] = abs2 * _wndw_area

        # Process if shade
        if self.facade_type in ['shade', 'blinds'] and not self._test_difference:
            if not self.new_map:
                cols = output_df.columns
                for prefix in np.unique([c.split("_")[0] for c in cols]):
                    for i, v in enumerate(list(self.window_ctrl_map_n.values())):
                        self.new_map[f"{prefix}_0_{i}"] = [
                            f"{prefix}_{ii}_{vv}" for ii, vv in enumerate(v)
                        ]
            temp = output_df.copy(deep=True)
            for k, v in self.new_map.items():
                output_df[k] = temp[v].sum(axis=1)
            output_df = output_df[self.new_map.keys()]

        self.init = False

        return output_df


def test(
    wwr=0.4,  # [0.4, 0.6]
    mode="dshade",  # ['dshade', 'shade', 'blinds', 'ec']
    facade_type="shade",  # ['shade', 'blinds', 'ec']
    single_step=True,
):
    """test function for radiance"""

    from afc.radiance.configs import get_config

    # root = os.path.dirname(os.path.abspath(__file__))

    # configuration
    print("Running example for:", wwr, mode, facade_type)
    filestruct, config_path = get_config(mode, wwr)
    # config_path = \
    #    '/usr/local/lib/python3.10/dist-packages/afc/resources/radiance/room0.4WWR_ec.cfg'

    print(filestruct, config_path)

    forecaster = Forecast(
        config_path,
        facade_type=facade_type,
        regenerate=False,
        filestruct=filestruct,
        wpi_plot=False,
        wpi_all=True,
        #wpi_loc="trap0.5",
        wpi_loc="23back",
        render=False,
        n_cpus=-1,
    )

    for _ in range(3):

        # multi-step
        df = pd.DataFrame(
            index=pd.date_range("2020-01-01 00:00", "2020-01-05 00:00", freq="5min")
        )
        np.random.seed(1)
        df["dni"] = np.random.uniform(0, 1000, len(df))
        np.random.seed(1)
        df["dhi"] = np.random.uniform(0, 250, len(df))

        # single-step
        fake_df = pd.DataFrame(
            [[185.5, 42], [165.3, 38.3]],
            index=pd.DatetimeIndex(
                [pd.to_datetime("2017-09-01 18:15"), pd.to_datetime("2017-09-01 18:20")]
            ),
            columns=["dni", "dhi"],
        )

        st = time.time()
        # res = forecaster.compute(df)
        tt = fake_df if single_step else df
        res = forecaster.compute2(tt)
        print(time.time() - st)
    print(f"Forecast of 24h x 5min in {round(time.time() - st, 2)} s")
    print(res.columns)
    # for k,v in forecaster.new_map.items():
    #    print(k, v)
    #print([c for c in res.columns if c.startswith('wpi_')])
    print(res)
    # print(res["wpi_0_5"])
    #print(res['wpi-all_0_0'])
    return res


if __name__ == "__main__":

    import warnings

    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    #res = test(wwr=0.4, mode="shade", facade_type="shade")
    res = test(wwr=0.4, mode="blinds", facade_type="blinds", single_step=False)
    #res = test(wwr=0.4, mode="ec", facade_type="ec")
    #res.to_csv("radiance-forecast_new.csv")
