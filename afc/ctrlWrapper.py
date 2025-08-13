# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Main controller wrapper module.
"""

# pylint: disable=invalid-name, bare-except, import-outside-toplevel
# pylint: disable=too-many-locals, redefined-outer-name, too-many-instance-attributes
# pylint: disable=too-many-branches, too-many-statements, broad-exception-caught
# pylint: disable=unused-variable

import io
import os
import sys
import copy
import time
import json
import logging
import warnings
import traceback
import numpy as np
import pandas as pd

from fmlc.baseclasses import eFMU

try:
    root = os.path.dirname(os.path.abspath(__file__))
    from .utility.thermostat import compute_thermostat_setpoints
except:
    root = os.getcwd()
    from afc.utility.thermostat import compute_thermostat_setpoints

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

map_weather = {'dni':'weaHDirNor', 'dhi':'weaHDifHor'}

class Controller(eFMU):
    """Main controller wrapper class for the AFC."""

    def __init__(self):

        super().__init__()

        self.input = {
            'input-data': None,
            'wf-all': None,
            'facade-initial': None,
            'temps-initial': None,
            'parameter': None
        }
        self.output = {
            'duration': None,
            'opt-stats': None,
            'glaremode': None,
            'ctrl-facade': None,
            'ctrl-thermostat': None,
            'ctrl-troom': None,
            'output-data': None,
            'valid': None
        }
        self.init = True
        self.root = root

        # Placeholder
        self.doper = None
        self.get_solver = None
        self.standard_report = None
        self.resample_variable_ts = None
        self.forecaster = None
        self.forecaster_class = None
        self.compute_periods = None
        self.get_tariff = None
        self.make_view_config = None
        self.view_config_from_rad = None
        self.glare_handler_class = None
        self.msg = None
        self.tariff = None
        self.controller = None
        self.glare_handler = None
        self.glare_ctrl_cols = None
        self.data = None
        self.res = None
        self.parameter = None

    def init_functions(self):
        """Function to initialize controller."""

        rm_paths = [p for p in sys.path if 'Documents' in p]
        _ = [sys.path.remove(p) for p in rm_paths]

#         # Add to path
#         sys.path.append(self.root)
#         for path in self.input['paths'].values():
#             sys.path.append(path)

        # DOPER
        #sys.path.append(os.path.join(root, '..', '..', 'doper_private'))
        from doper import DOPER, get_solver, standard_report, resample_variable_ts, compute_periods
        from doper.data.tariff import get_tariff

        #from DOPER.wrapper import DOPER
        #from computetariff import compute_periods
        self.doper = DOPER
        self.get_solver = get_solver
        self.standard_report = standard_report
        self.resample_variable_ts = resample_variable_ts

        # Radiance modules
        from afc.radiance import forecast
        self.forecaster = forecast

        # Controller modules
        #from ComputeTariff import compute_periods
        self.compute_periods = compute_periods
        self.get_tariff = get_tariff

        # Glare logic
        #sys.path.append(os.path.join(self.input['paths']['emulator'], 'controller'))
        from afc.glare.view_angle import make_view_config, view_config_from_rad
        from afc.glare.heur_glare import MultiZone
        self.make_view_config = make_view_config
        self.view_config_from_rad = view_config_from_rad
        self.glare_handler_class = MultiZone

    def log_results(self):
        """Function to log results."""

        log_dir = self.input['parameter']['wrapper']['log_dir']
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_dir = os.path.join(log_dir, str(self.input['parameter']['wrapper']['instance_id']))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        fname = str(self.data.index[0]).replace(' ','').replace(':','')
        with open(os.path.join(log_dir, fname+'.txt'), 'w', encoding='utf8') as f:
            json.dump(self.controller.parameter, f)
        self.data.to_csv(os.path.join(log_dir, fname+'.csv'))

    def compute(self):
        """Main compute function."""

        try:
            st = time.time()
            self.msg = ''

            # Parse input dataframe
            inputs = pd.read_json(io.StringIO(self.input['input-data']))
            inputs.index = pd.to_datetime(inputs.index)

            # configuration
            self.parameter = self.input['parameter']

            # add demand profiles
            if self.parameter['wrapper']['compute_loads']:
                inputs = make_inputs(self.parameter, inputs, return_json=False)['input-data']

            # Setup controller
            if self.init:

                self.init_functions()

                # Radiance forecaster
                rad_paths = self.parameter['radiance']['paths']
                filestruct = {}
                filestruct['glazing_systems'] = rad_paths['rad_systems']
                filestruct['matrices'] = rad_paths['rad_mtx']
                window_ctrl_map = self.parameter['facade']['window_ctrl_map']
                reflectances = self.parameter['radiance']['reflectances']
                self.forecaster = \
                    self.forecaster.Forecast(rad_paths['rad_config'],
                                             regenerate=self.parameter['radiance']['regenerate'],
                                             location=self.parameter['radiance']['location'],
                                             facade_type=self.parameter['facade']['type'],
                                             window_ctrl_map=window_ctrl_map,
                                             wpi_loc=self.parameter['radiance']['wpi_loc'],
                                             view_config=self.parameter['radiance']['view'],
                                             filestruct=filestruct,
                                             dimensions=self.parameter['radiance']['dimensions'],
                                             wpi_all=self.parameter['radiance']['wpi_all'],
                                             wpi_config=self.parameter['radiance']['wpi_config'],
                                             reflectances=reflectances,
                                             n_cpus=self.parameter['radiance']['n_cpus'])
                # Update logical window system
                self.parameter['facade']['logical_windows'] = \
                    list(range(self.forecaster.logical_windows))
                self.parameter['facade']['logical_window_states'] = \
                    self.forecaster.logical_window_states
                # Store class
                if self.parameter['radiance']['store_class']:
                    self.forecaster_class = self.forecaster

                # Glare controller
                rad_config = {'Dimensions': self.parameter['radiance']['dimensions'],
                              'View': {'views': self.forecaster.view}}
                view_config = self.view_config_from_rad(rad_config, start_lx=20e3, end_lx=12.5e3,
                    combine_windows=not 'ec' in self.parameter['facade']['type'])
                glare_config = copy.deepcopy(self.parameter['radiance']['location'])
                glare_config.update(self.parameter['radiance']['view'])
                self.glare_handler = \
                    self.glare_handler_class(config=glare_config,
                                             view_config=view_config,
                                             tvis=self.parameter['facade']['tvis'])

                # Precompute (simulation)
                if self.parameter['wrapper']['precompute_radiance']:

                    # Weather forecast
                    wf_all = pd.DataFrame().from_dict(self.input['wf-all'])
                    wf_all.index = pd.to_datetime(wf_all.index, unit='ms')

                    # Radiance forecaster
                    temp = pd.DataFrame()
                    for d in sorted(np.unique(wf_all.index.date)):
                        if temp.empty:
                            tt = wf_all[['dni','dhi']][wf_all.index.date == d]
                            temp = self.forecaster.compute2(tt)
                        else:
                            temp = pd.concat([temp, self.forecaster.compute2( \
                                wf_all[['dni','dhi']][wf_all.index.date == d])])
                    self.forecaster = temp

                    # Glare handler
                    wf = wf_all[['dni','dhi']].copy(deep=True).rename(columns=map_weather)
                    for ix in wf.index:
                        glare_ctrl = self.glare_handler.do_step(ix, {'weather_data': wf})
                        for i, g in enumerate(self.glare_handler.gmodes):
                            wf.loc[ix, f'zone{i}_gmode'] = int(g)
                        self.glare_ctrl_cols = \
                            [f'glare_ctrl_{i}' for i in range(len(glare_ctrl[0]))]
                        wf.loc[ix, self.glare_ctrl_cols] = glare_ctrl[0]
                    self.glare_handler = wf

                # DOPER controller
                from afc.optModel import control_model#, pyomo_to_pandas
                if self.parameter['wrapper']['solver_dir']:
                    solver_path = \
                        self.get_solver(self.parameter['wrapper']['solver_name'],
                                        solver_dir=self.parameter['wrapper']['solver_dir'])
                else:
                    solver_path = self.get_solver(self.parameter['wrapper']['solver_name'])
                pyomo_logger = \
                    logging.WARNING if self.parameter['wrapper']['printing'] else \
                        logging.ERROR
                self.tariff = self.get_tariff(self.parameter['wrapper']['tariff_name'])
                output_list = self.parameter['wrapper']['output_list']
                self.controller = self.doper(model=control_model,
                                             parameter=self.parameter,
                                             solver_path=solver_path,
                                             pyomo_logger=pyomo_logger,
                                             output_list=output_list)

            # Compute radiance
            st1 = time.time()
            self.output['duration'] = {}

            if self.parameter['wrapper']['precompute_radiance']:
                data = self.forecaster.loc[inputs.index]
            else:
                data = self.forecaster.compute2(inputs[['dni','dhi']])
            # cutoff
            rad_cutoff = self.parameter['facade']['rad_cutoff']
            for k in rad_cutoff.keys():
                for c in data.columns:
                    if f'{k}_' in c:
                        data.loc[:,c] = data[c].mask(data[c] < rad_cutoff[k][0], 0)
                        data.loc[:,c] = data[c].mask(data[c] > rad_cutoff[k][1], rad_cutoff[k][1])
            self.output['duration']['radiance'] = time.time() - st1

            # Glare handler
            st1 = time.time()
            if self.parameter['wrapper']['precompute_radiance']:
                wf = self.glare_handler.loc[inputs.index]
            else:
                wf = inputs[['dni','dhi']].copy(deep=True).rename(columns=map_weather)
                for ix in wf.index:
                    glare_ctrl = self.glare_handler.do_step(ix, {'weather_data': wf})
                    for i, g in enumerate(self.glare_handler.gmodes):
                        wf.loc[ix, f'zone{i}_gmode'] = int(g)
                    self.glare_ctrl_cols = [f'glare_ctrl_{i}' for i in range(len(glare_ctrl[0]))]
                    wf.loc[ix, self.glare_ctrl_cols] = glare_ctrl[0]

            #flip_z = 'ec' in self.parameter['facade']['type']
            gmodes = []
            for nz in self.parameter['facade']['logical_windows']:
                wf_key = f'zone{nz}_gmode'
                gmodes.append(wf.loc[wf.index[0], wf_key])
                # non-dark states to clear
                for t in self.parameter['facade']['logical_window_states'][1:]:
                    #wf_key = f'zone{t if not flip_z else nz}_gmode'
                    #if not wf_key in wf.columns:
                    #    wf_key = f'zone{t-1 if not flip_z else nz-1}_gmode'
                    data[f'ev_{nz}_{t}'] = data[f'ev_{nz}_{t}'].mask( \
                        (wf[wf_key] > 0) & (data[f'ev_{nz}_{t}']>0), 2e4)
            self.output['duration']['glare'] = time.time() - st1

            # Compute other inputs
            data = pd.concat([data, inputs], axis=1)
            data['oat'] = inputs['temp_air']
            data['wind_speed'] = inputs['wind_speed']
            data['battery_0_avail'] = 0
            data['battery_0_demand'] = 0
            data['battery_reg'] = 0

            # Update SOCs
            if isinstance(self.input['temps-initial'], str):
                self.input['temps-initial'] = json.loads(self.input['temps-initial'])
            self.parameter['zone']['temps_initial'] = self.input['temps-initial']
            if isinstance(self.input['facade-initial'], str):
                self.input['facade-initial'] = json.loads(self.input['facade-initial'])
            self.parameter['facade']['fstate_initial'] = self.input['facade-initial']

            # Make sure temp_initial is feasible
            dead_band = 1e-2
            cool_set = data['temp_room_max'].values[0]
            heat_set = data['temp_room_min'].values[0]
            troom = self.parameter['zone']['temps_initial'][0]
            troom = max(heat_set+dead_band, min(cool_set-dead_band, troom))
            self.parameter['zone']['temps_initial'][0] = troom

            # Variable timestep
            st1 = time.time()
            if self.parameter['wrapper']['resample_variable_ts']:

                # remove string columns
                string_cols = data.columns[data.apply(lambda x: x.dtype == 'object').values]
                if len(string_cols) > 0:
                    self.msg += f'WARNING: Removing string columns: {string_cols}'
                    data = data.drop(columns=string_cols)

                # check columns
                cols = self.parameter['wrapper']['cols_fill']
                if not 'temp_room_max' in cols[0].lower():
                    print('ERROR: "temp_room_max" is not in first column of "cols_fill".')

                # ensure ramp when occupancy starts
                for ix in data.index[data[cols[0]].diff()<0]:
                    ix_st = max(data.index[0]+pd.DateOffset(minutes=5),
                                ix-pd.DateOffset(minutes=55))
                    data.loc[ix_st:ix-pd.DateOffset(minutes=5), cols] = np.nan
                resample_ts = (data.index[1] - data.index[0]).total_seconds()
                data = data.resample(f'{int(resample_ts)}s').interpolate()

                # ensure resampling when occupancy ends
                for ix in data.index[data[cols[0]].diff()>0]:
                    data.loc[ix, cols] = \
                        data.loc[data.index[max(0, data.index.get_loc(ix)-1)], cols].values

                # limit starting ramp
                t_init = self.parameter['zone']['temps_initial'][0]
                data[cols[0]] = \
                    np.min([[t_init+(i+1)*self.parameter['wrapper']['limit_slope'] \
                             for i in range(len(data))], data[cols[0]]], axis=0)
                data[cols[1]] = \
                    np.max([[t_init-(i+1)*self.parameter['wrapper']['limit_slope'] \
                             for i in range(len(data))], data[cols[1]]], axis=0)

                # resample
                data = self.resample_variable_ts(data, \
                    reduced_start=int(self.parameter['wrapper']['reduced_start']),
                    reduced_ts=int(self.parameter['wrapper']['reduced_ts']),
                    cols_fill=self.parameter['wrapper']['cols_fill'])
            self.output['duration']['varts'] = time.time() - st1

            # Compute and update tariff
            data, _ = self.compute_periods(data, self.tariff, self.parameter)

            # Check for nan
            if pd.isnull(data).any().any():
                print(data)
                data.to_csv('error_controller_inputs.csv')
                raise ValueError('NAN values in MPC input. Index:',
                                 data.index[pd.isnull(data).any().to_numpy().nonzero()[0]])

            # Run optimization
            st1 = time.time()
            self.data = data.round(self.parameter['wrapper']['inputs_cutoff'])

            # Store for debug
            # data.to_csv('inputs_{}.csv'.format(data.index[0]))
            # cfg = {}
            # cfg['parameter'] = self.parameter
            # cfg['options'] = self.parameter['options']
            # with open('cfg_{}.json'.format(data.index[0]), 'w') as f:
            #     f.write(json.dumps(cfg))

            printing = self.parameter['wrapper']['printing']
            self.res = \
                self.controller.do_optimization(self.data,
                                                parameter=self.parameter,
                                                options=self.parameter['solver_options'],
                                                tee=printing,
                                                print_error=printing)
            duration, objective, df, model, result, termination, parameter = self.res
            df = pd.concat([df, data], axis=1)
            self.output['duration']['optall'] = time.time() - st1

            # Write outputs
            st1 = time.time()
            self.output['opt-stats'] = {'duration': float(duration),
                                        'termination': str(termination),
                                        'objective': float(objective) if objective else None}
            self.output['valid'] = bool(objective)
            self.output['glaremode'] = list(gmodes)

            # Compute thermostat setpoints
            thermostat = compute_thermostat_setpoints(
                df, cool_set, heat_set, self.output['valid'], True, False
            )
            self.output['ctrl-thermostat'] = thermostat
            if self.output['valid']:
                self.output['ctrl-troom'] = float(df['Temperature 0 [C]'].values[1])
            else:
                self.output['ctrl-troom'] = None

            # Compute shade state
            if self.output['valid']:
                # apply mpc setpoint
                uShade = df[[f'Facade State {z}' for \
                    z in self.parameter['facade']['logical_windows']]].iloc[0].values
                self.output['ctrl-facade'] = [round(float(u),1) for u in uShade]
            elif self.parameter['wrapper']['use_fallback']:
                # use heuristic control
                ix = wf.index[0]
                uShade = [round(float(u),1) for u in wf.loc[ix, self.glare_ctrl_cols]]
                self.output['ctrl-facade'] = uShade
            else:
                # hold previous value
                # self.output['ctrl-facade'] = None
                pass

            df = df.astype(float).fillna(-1)
            self.output['output-data'] = df.to_json()
            self.output['duration']['outputs'] = time.time() - st1
            self.output['duration']['all'] = time.time() - st

            # Store if long optimization
            if self.output['duration']['all'] > self.parameter['wrapper']['log_overtime']:
                self.log_results()

            self.init = False
            self.msg += self.standard_report(self.res)

        except Exception as e:
            self.msg += f'\nERROR: {e}\n\n{traceback.format_exc()}'
            for k in self.output:
                self.output[k] = None
        return self.msg

def make_inputs(parameter, df, ext_df=pd.DataFrame(), return_json=True):
    """Utility function to make inputs."""

    df = df.copy(deep=True)

    # Time-variable schedule
    if parameter['occupant']['schedule']:
        print('ERROR: Not implemented!')
        raise NotImplementedError

    # Define default columns
    df_col_map = {
        'plug_load': parameter['occupant']['plug_load'], # W
        'occupant_load': parameter['occupant']['occupant_load'], # W
        'equipment': parameter['occupant']['equipment'], # W
        'occupancy_light': parameter['occupant']['occupancy_light'], # 0-unocc, 1-occ
        'wpi_min': parameter['occupant']['wpi_min'],
        'glare_max': parameter['occupant']['glare_max'],
        'temp_room_max': parameter['occupant']['temp_room_max'],
        'temp_room_min': parameter['occupant']['temp_room_min'],
        'wind_speed': 0,
        'generation_pv': 0,
        'load_demand': 0,
        'temp_slab_max': 1e3,
        'temp_slab_min': 0,
        'temp_wall_max': 1e3,
        'temp_wall_min': 0,
        'grid_co2_intensity': 0,
    }

    # Default inputs
    for k, v in df_col_map.items():
        if not k in df.columns:
            df[k] = v

    # Add External inputs (if any)
    for c in ext_df:
        df[c] = ext_df[c]

    # Map parameter and make Inputs object
    inputs = {}
    if return_json:
        inputs['input-data'] = df.to_json()
    else:
        inputs['input-data'] = df
    inputs['wf-all'] = None
    inputs['facade-initial'] = parameter['facade']['fstate_initial']
    inputs['temps-initial'] = parameter['zone']['temps_initial']
    inputs['parameter'] = parameter

    return inputs

if __name__ == '__main__':

    root = os.getcwd()

    from afc.utility.weather import example_weather_forecast
    # from afc.radiance.configs import get_config
    from afc.defaultConfig import default_parameter
    from afc.utility.plotting import plot_standard1

    # read weather (forecast) data
    wf = example_weather_forecast(date='2023-07-01')
    df = wf[wf.index.date == wf.index[0].date()]

    # Initialize controller
    ctrl = Controller()

    # Get all variables
    print('All Input variables:', ctrl.get_model_variables())

    parameter = default_parameter()
    inputs = make_inputs(parameter, df)

    # Query controller
    print('Log-message:\n', ctrl.do_step(inputs=inputs))
    print('Duration:\n', ctrl.get_output(keys=['duration']))
    print('Optimization:\n', ctrl.get_output(keys=['opt-stats']))
    df = pd.read_json(io.StringIO(ctrl.get_output(keys=['output-data'])['output-data']))

    try:
        plot_standard1(pd.concat([wf, df], axis=1).ffill().iloc[:-1])
    except:
        pass
