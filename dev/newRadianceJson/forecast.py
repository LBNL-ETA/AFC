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
# pylint: disable=import-outside-toplevel, too-many-positional-arguments

import os
import time
import json

# import multiprocessing as mp
from configparser import ConfigParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import frads as fr
from frads import room, matrix, methods, geom, window

from maps import shade_map_0x6, shade_map_0x4
from frads_constants import KFOMG, OMEGAS

root = os.path.dirname(os.path.realpath(__file__))

mult = np.linalg.multi_dot

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
        wpi_plot=False,
        wpi_loc="23back",
        view_config={'view_dist': 1.22, 'view_orient': 's'},
        location=None,
        filestruct=None,
        _test_difference=False,
        dimensions=None,
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
        self.wpi_plot = wpi_plot
        self.wpi_loc = wpi_loc
        self.view_config = view_config
        self._test_difference = _test_difference
        
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
        self.glazing_systems = [
            fr.window.load_glazing_system(i) for i in self.glazing_system_paths
        ]
        self.make_room()
        self.get_workflow()
        self.make_matrices()
        self.get_matrices()
        self.wwr = float(cfg_path.split("room")[1].split("WWR")[0])
        if "shade" in self.facade_type and float(self.wwr) == 0.6:
            self.shade_map = shade_map_0x6
        elif "shade" in self.facade_type and float(self.wwr) == 0.4:
            self.shade_map = shade_map_0x4
        self.new_map = {}
        self.init = True

        # Initialize variables
        self.wpi_data = None
        self.map_center = None
        self.map_23 = None
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
        self.sensor_grid = geom.gen_grid(
            self.theroom.floor.base, self.grid_height, self.grid_spacing
        )

    def get_workflow(self):
        """Function to create workflow."""
        _view = self.view.split()
        model = self.theroom.model_dump()
        model["sensors"] = {
            "wpi": {"data": self.sensor_grid},
            "view": {"data": [_view]},
        }
        model["views"] = {}
        settings = {
            "method": 3,
            "latitude": self.lat,
            "longitude": self.lon,
            "time_zone": self.timezone,
            "site_elevation": self.elevation,
            "save_matrices": True,
            "matrix_dir": self.matricesd,
            "sensor_window_matrix": self.vmx_opt.split(),
            "daylight_matrix": self.dmx_opt.split(),
            "surface_window_matrix": self.vsmx_opt.split(),
            "sky_basis": "r4",
            "name": f"afc_version={AFC_VERSION}, frads_version={FRADS_VERSION}",
        }
        workflow_config = methods.WorkflowConfig.from_dict(
            {"model": model, "settings": settings}
        )
        self.workflow = methods.ThreePhaseMethod(workflow_config)

    def make_matrices(self):
        """Function to make matrices."""
        if self.remake_matrices:
            self.workflow.mfile.unlink(missing_ok=True)
        self.workflow.generate_matrices(view_matrices=False)

    def intergrate_matrix(self, matrix):
        # Convert from BSDF to transmission matrix
        ncol = len(matrix[0])
        nrow = len(matrix)
        print(ncol, nrow)
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

    def get_matrices(self):
        self._tvis = [
            self.intergrate_matrix(gs.visible_back_transmittance)
            for gs in self.glazing_systems
        ]
        self._tsol = [
            self.intergrate_matrix(gs.solar_back_transmittance)
            for gs in self.glazing_systems
        ]
        self._abs1 = [
            self.intergrate_matrix(
                np.array(gs.solar_front_absorptance[0]).reshape(1, -1)
            )
            for gs in self.glazing_systems
        ]
        self._abs2 = [
            self.intergrate_matrix(
                np.array(gs.solar_front_absorptance[1]).reshape(1, -1)
            )
            for gs in self.glazing_systems
        ]
        print("done")

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

    def process_wpi(self, wpi_grid):
        """Function to process wpi."""

        if self.init:
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

            # Center
            center_w = round(width / 2, 4)
            center_d = round(depth / 2, 4)
            fov = 60  # deg
            self.map_center = self.wpi_map(self.wpi_data, center_w, center_d, fov)

            # Occupant
            occ_w = round(width / 2, 4)
            # occ_d = 1.525 # Occupant
            occ_d = round(depth / 3 * 2, 4)
            fov = 60  # deg
            self.map_23 = self.wpi_map(self.wpi_data, occ_w, occ_d, fov)

            self.n_w = n_w
            self.n_d = n_d

        data = pd.DataFrame(wpi_grid.reshape(self.n_w, self.n_d))
        data_center = data * self.map_center
        wpi_23 = data * self.map_23

        if self.wpi_plot:
            fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=False, sharey=False)
            tt_avg = round(data.mean().mean(), 1)
            axs[0].set_title(f"All WPI sensors (avg={tt_avg})")
            pos = axs[0].imshow(data.values, cmap="hot")
            axs[0].set_xticks(np.arange(0, self.n_d, 1))
            axs[0].set_xticklabels(
                [round(self.wpi_data.columns[int(i)], 1) for i in axs[0].get_xticks()]
            )
            axs[0].set_yticks(np.arange(0, self.n_w, 1))
            axs[0].set_yticklabels(
                [round(self.wpi_data.index[int(i)], 1) for i in axs[0].get_yticks()]
            )
            fig.colorbar(pos, ax=axs[0])

            tt_avg = round(data_center.mean().mean(), 1)
            axs[1].set_title(f"Cone view 60 deg room centered (avg={tt_avg})")
            pos = axs[1].imshow(data_center.values, cmap="hot")
            axs[1].set_xticks(np.arange(0, self.n_d, 1))
            axs[1].set_xticklabels(
                [round(self.wpi_data.columns[int(i)], 1) for i in axs[1].get_xticks()]
            )
            axs[1].set_yticks(np.arange(0, self.n_w, 1))
            axs[1].set_yticklabels(
                [round(self.wpi_data.index[int(i)], 1) for i in axs[1].get_yticks()]
            )
            fig.colorbar(pos, ax=axs[1])

            tt_avg = round(wpi_23.mean().mean(), 1)
            axs[2].set_title(f"Cone view 60 deg occupant centered (avg={tt_avg})")
            pos = axs[2].imshow(wpi_23.values, cmap="hot")
            axs[2].set_xticks(np.arange(0, self.n_d, 1))
            axs[2].set_xticklabels(
                [round(self.wpi_data.columns[int(i)], 1) for i in axs[2].get_xticks()]
            )
            axs[2].set_yticks(np.arange(0, self.n_w, 1))
            axs[2].set_yticklabels(
                [round(self.wpi_data.index[int(i)], 1) for i in axs[2].get_yticks()]
            )
            fig.colorbar(pos, ax=axs[2])

            fig.tight_layout()
            tt_dt = self.date_time.strftime("%Y%m%dT%H%M")
            fig.savefig(f"wpi-plot_{tt_dt}.jpeg")

        wpi_all = data.mean().mean()
        wpi_center = data_center.mean().mean()
        wpi_23 = wpi_23.mean().mean()

        if self.wpi_loc == "23back":
            wpi = wpi_23
        elif self.wpi_loc == "center":
            wpi = wpi_center
        elif self.wpi_loc == "all":
            wpi = wpi_all
        else:
            print(f'ERROR: wpi_loc "{self.wpi_loc}" not available.')
            wpi = None

        return wpi, wpi_all, wpi_center, wpi_23

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
        )
        _sol = self.workflow.get_sky_matrix(
            time=list(weather_df.index.tolist()),
            dni=weather_df["dni"].tolist(),
            dhi=weather_df["dhi"].tolist(),
            solar_spectrum=True,
        )
        output_df = pd.DataFrame(index=_df.index)
        flr_area = np.linalg.norm(self.theroom.floor.base.area)
        for idx, window in enumerate(self.theroom.swall.windows):
            _wndw_area = np.linalg.norm(window.polygon.area)
            _window_name = window.primitive.identifier
            for idx_t, _ in enumerate(self.glazing_systems):
                wpi_col = f"wpi_{idx}_{idx_t}"
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
                evres = matrix.matrix_multiply_rgb(
                    self.workflow.sensor_window_matrices["view"].array[idx],
                    self._tvis[idx_t],
                    self.workflow.daylight_matrices[_window_name].array,
                    _vis,
                    weights=[47.4, 119.9, 11.6],
                )

                # Process WPI
                wpi = []
                for i, t in enumerate(np.transpose(visres)):
                    self.date_time = weather_df.index[i]
                    wpi_temp = self.process_wpi(t)
                    wpi.append(wpi_temp[0])
                output_df[wpi_col] = wpi
                output_df[ev_col] = evres[0]
                # check against pre-mtx
                # abs1 size is wrong
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
        if "shade" in self.facade_type and not self._test_difference:
            if not self.new_map:
                cols = output_df.columns
                for prefix in np.unique([c.split("_")[0] for c in cols]):
                    for i, v in enumerate(list(self.shade_map.values())[:-1]):
                        self.new_map[f"{prefix}_0_{i}"] = [
                            f"{prefix}_{ii}_{vv}" for ii, vv in enumerate(v)
                        ]
                    i += 1
                    self.new_map[f"{prefix}_0_{i}"] = [
                        f"{prefix}_{ii}_1" for ii, vv in enumerate(v)
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
):
    """test funciton for radiance"""

    from configs import get_config

    # root = os.path.dirname(os.path.abspath(__file__))

    # configuration
    print("Running example for:", wwr, mode, facade_type)
    filestruct, config_path = get_config(mode, wwr)

    print(filestruct, config_path)

    forecaster = Forecast(
        config_path,
        facade_type=facade_type,
        regenerate=False,
        filestruct=filestruct,
        wpi_plot=False,
        wpi_loc="23back",
    )

    for _ in range(3):
        # df = pd.DataFrame([[800, 80], [750, 100], [100, 10]],
        #                       index=pd.DatetimeIndex([
        #                           pd.datetime(2019, 12, 21, 12, 0),
        #                           pd.datetime(2019, 12, 21, 12, 10),
        #                           pd.datetime(2019, 12, 21, 12, 20),
        #                       ]))
        df = pd.DataFrame(
            index=pd.date_range("2020-01-01 00:00", "2020-01-02 00:00", freq="5min")
        )
        np.random.seed(1)
        df["dni"] = np.random.uniform(0, 1000, len(df))
        np.random.seed(1)
        df["dhi"] = np.random.uniform(0, 250, len(df))

        fake_df = pd.DataFrame(
            [[185.5, 42], [165.3, 38.3]],
            index=pd.DatetimeIndex(
                [pd.to_datetime("2017-09-01 18:15"), pd.to_datetime("2017-09-01 18:20")]
            ),
            columns=["dni", "dhi"],
        )

        st = time.time()
        # res = forecaster.compute(df)
        res = forecaster.compute2(fake_df)
        print(time.time() - st)
    print(f"Forecast of 24h x 5min in {round(time.time() - st, 2)} s")
    # print(res.columns)
    # for k,v in forecaster.new_map.items():
    #    print(k, v)
    print(res)
    # print(res["wpi_0_5"])
    return res


if __name__ == "__main__":

    import warnings

    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    res = test(wwr=0.4, mode="ec", facade_type="ec")
    res.to_csv("radiance-forecast_new.csv")
