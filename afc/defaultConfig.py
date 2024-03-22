# Advanced Fenestration Controller (AFC) Copyright (c) 2023, The
# Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals
# from the U.S. Dept. of Energy). All rights reserved.

""""Advanced Fenestration Controller
Default configuration.
"""

# pylint: disable=too-many-arguments, bare-except, too-many-locals
# pylint: disable=invalid-name, dangerous-default-value

import os
import sys
from doper.models.basemodel import default_output_list
from doper.examples import default_parameter as default_parameter_doper

try:
    from .optModel import afc_output_list
    from .radiance.configs import get_config
except:
    sys.path.append(os.path.join('..', 'afc'))
    from optModel import afc_output_list
    from radiance.configs import get_config

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = os.getcwd()

def ft_to_m(k):
    """Convert feet to meter."""

    return k * 0.3048

def ft2_to_m2(k):
    """Convert square feet to square meter."""

    return k * (0.3048 ** 2)

def get_facade_config(parameter, facade_type='ec-71t', window_area=2.56*2.78):
    """Default configuration for facade."""

    parameter['facade'] = {}

    # cutoffs for computation
    parameter['facade']['rad_cutoff'] = {}
    parameter['facade']['rad_cutoff']['wpi'] = [5, 1e6] # lx
    parameter['facade']['rad_cutoff']['ev'] = [250, 1e6] # (0.2 - 0.184) / 6.22e-5 = 250 lx
    parameter['facade']['rad_cutoff']['shg'] = [0, 1e6] # W
    parameter['facade']['rad_cutoff']['abs1'] = [50, 1e6] # W
    parameter['facade']['rad_cutoff']['abs2'] = [5, 1e6] # W
    parameter['facade']['rad_cutoff']['tsol'] = [5, 1e6] # W
    parameter['facade']['rad_cutoff']['iflr'] = [5, 1e6] # W

    # other configuration
    parameter['facade']['convection_window_scale'] = 4 # From model; only for R6C3
    parameter['facade']['convection_window_offset'] = 4 # From model; only for R6C3

    # window area
    parameter['facade']['window_area'] = window_area # in m2

    # define facade
    if facade_type == 'ec-71t':
        parameter['facade']['type'] = 'ec'
        parameter['facade']['windows'] = [0, 1, 2]
        parameter['facade']['states'] = [0, 1, 2, 3]
        parameter['facade']['fstate_initial'] = [3, 3, 3] # Initial state of facade
    else:
        raise ValueError(f'The facade type "{facade_type}" is not available.')

    return parameter

def get_radiance_config(parameter, regenerate=False, wwr=0.4, latitude=37.7, longitude=122.2,
                        view_orient=0, timezone=120, orient=0, elevation=100, width=3.05,
                        depth=4.57, height=3.35,
                        windows=['.38 .22 2.29 .85', '.38 1.07 2.29 .85', '.38 1.98 2.29 .51']):
    """Default configuration for radiance.
    
    Default window parameters for 71T.
    Viewing from outside, facing the south wall, the lower left corner of the wall is (0, 0, 0)
    window1 = .38 .22 2.29 .85
        is 0.38m from the left edge, 0.22 meter from the ground, 2.29 in width, and 0.85 in height.
    Similarly, "view1 = 1.525 1 1.22 0 -1 0"
        is at 1.525 meter from the west wall(xposition), 1 meter from the south wall(y position),
        1.22m height(z position), facing in 0 -1 0 direction(south)
    """

    # setup Radiance
    parameter['radiance'] = {}
    parameter['radiance']['regenerate'] = regenerate
    parameter['radiance']['wwr'] = round(wwr, 1)
    parameter['radiance']['wpi_loc'] = '23back'
    # location
    parameter['radiance']['location'] = {}
    parameter['radiance']['location']['latitude'] = latitude
    if longitude < 0:
        print('WARNING: Longitude for Radiance must be positive for the western hemisphere.')
    parameter['radiance']['location']['longitude'] = longitude
    parameter['radiance']['location']['view_orient'] = view_orient
    parameter['radiance']['location']['timezone'] = timezone
    parameter['radiance']['location']['orient'] = orient
    parameter['radiance']['location']['elevation'] = elevation
    # dimensions
    parameter['radiance']['dimensions'] = {}
    parameter['radiance']['dimensions']['width'] = width
    parameter['radiance']['dimensions']['depth'] = depth
    parameter['radiance']['dimensions']['height'] = height
    if len(windows) == len(parameter['facade']['windows']):
        for wz, window in enumerate(windows):
            parameter['radiance']['dimensions'][f'window{wz+1}'] = window
    else:
        raise ValueError(f'Number of window zones {parameter["facade"]["windows"]} does'\
        'not match window dimensions {windows}.')

    # paths
    filestruct, rad_config = get_config(parameter['facade']['type'],
                                        str(parameter['radiance']['wwr']),
                                        root=os.path.join(root, 'resources', 'radiance'))
    parameter['radiance']['paths'] = {}
    parameter['radiance']['paths']['rad_config'] = rad_config
    parameter['radiance']['paths']['rad_bsdf'] = filestruct['resources']
    parameter['radiance']['paths']['rad_mtx'] = filestruct['matrices']

    return parameter

def get_zone_config(parameter, lighting_efficiency=0.24, system_cooling_eff=1/3.5,
                    system_heating_eff=0.95, zone_area=15, zone_type='single_office'):
    """Default configuration for thermal zone."""

    parameter['zone'] = {}

    # Setup
    parameter['zone']['lighting_efficiency'] = lighting_efficiency # W/lx
    parameter['zone']['heating_efficiency'] = system_heating_eff
    parameter['zone']['cooling_efficiency'] = system_cooling_eff
    parameter['zone']['heat_max'] = [200*zone_area, 200*zone_area, 200*zone_area] # 200 W_th/m2
    parameter['zone']['cool_max'] = [200*zone_area, 200*zone_area, 200*zone_area] # 200 W_th/m2

    # thermal model
    if zone_type == 'single_office':
#         parameter['zone']['param'] = {'type':'R4C2',
#                                       'Row1': 0.0037992496008808323,
#                                       'Rw1w2': 0.10706442491986229,
#                                       'Rw2i': 3.3602377759986217e-07,
#                                       'Ci': 211414.5114368095,
#                                       'Ris': 0.012804832879362456,
#                                       'Cs': 3268802.970556823}

        # updated on 2023/02/06 with e19 results summer
        parameter['zone']['param'] = {'type': 'R4C2',
                                      'Ci': 492790.131488945,
                                      'Cs': 3765860.3115474223,
                                      'Ris': 0.023649372856050448,
                                      'Row1': 0.0030454206460150783,
                                      'Rw1w2': 0.14425660371050014,
                                      'Rw2i': 0.0002577364781085182}
    else:
        raise ValueError(f'The zone type "{zone_type}" is not available.')

    # defaults
    parameter['zone']['zone_id'] = 1 # zone id for multizone
    parameter['zone']['lighting_capacity'] = 1000 # lx
    parameter['zone']['temps_name'] = ['room', 'slab'] #, 'wall']
    parameter['zone']['temps_initial'] = [22.5, 22.5, 22.5]

    parameter['zone']['lighting_split'] = 0.5 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['plugload_split'] = 0 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['occupancy_split'] = 1 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['tsol_split'] = 1 # All Tsol on surfaces (1=rad, 0=conv)
    parameter['zone']['shg_split'] = 0 # All SHG in air 1=rad, 0=conv

    # penalty definition
    parameter['zone']['glare_diff'] = 0.1 # Lower bound of glare penalty (glare_max - glare_diff)
    parameter['zone']['glare_scale'] = 10 # Scale of glare cost function (ATTENTION absolute value)
    parameter['zone']['view_scale'] = 0.0 # Scale of view cost function (ATTENTION absolute value)

    return parameter

def get_occupant_config(parameter, schedule=None, wpi_min=500, glare_max=0.4,
                        temp_room_max=24, temp_room_min=20,
                        plug_load=150, occupant_load=100, equipment=150,
                        zone_type='single_office', number_occupants=1):
    """Default configuration for occupant."""

    parameter['occupant'] = {}

    # constraints
    parameter['occupant']['schedule'] = schedule
    parameter['occupant']['wpi_min'] = wpi_min
    parameter['occupant']['glare_max'] = glare_max
    parameter['occupant']['temp_room_max'] = temp_room_max
    parameter['occupant']['temp_room_min'] = temp_room_min

    # loads
    if zone_type == 'single_office':
        parameter['occupant']['plug_load'] = plug_load * number_occupants # W
        parameter['occupant']['occupant_load'] = occupant_load * number_occupants # W
        parameter['occupant']['equipment'] = equipment * number_occupants # W
        parameter['occupant']['occupancy_light'] = 1 # 0-unoccupied, 1-occupied
    else:
        raise ValueError(f'The zone type "{zone_type}" is not available.')

    return parameter

def default_parameter(tariff_name='e19-2020', hvac_control=True,
                      facade_type='ec-71t', room_height=11,
                      room_width=10, room_depth=15, window_count=2,
                      window_height=8.0, window_sill=0.5, window_width=4.5,
                      lighting_efficiency=0.24, system_cooling_eff=1/3.5,
                      system_heating_eff=0.95, zone_area=15,
                      zone_type='single_office', weight_actuation=0,
                      weight_glare=0, precompute_radiance=False,
                      location_latitude=37.85, location_longitude=-122.24,
                      location_orientation=0, view_orient=0,
                      timezone=120, elevation=100, number_occupants=1,
                      schedule=None, wpi_min=500, glare_max=0.4, instance_id=0,
                      debug=False):
    """Function to load the default parameters for AFC."""

    window_area = ft2_to_m2((window_height-window_sill) * window_width * window_count)
    facade_area = ft2_to_m2(room_width * room_height)
    wwr = window_area / facade_area

    # initialize with defaults for optimization
    parameter = default_parameter_doper()
    #parameter = parameter_add_battery(parameter)

    # setup facade
    parameter = get_facade_config(parameter,
                                  facade_type = facade_type,
                                  window_area = window_area)

    # setup zone
    parameter = get_zone_config(parameter,
                                lighting_efficiency = lighting_efficiency,
                                system_cooling_eff = system_cooling_eff,
                                system_heating_eff = system_heating_eff,
                                zone_area = zone_area,
                                zone_type = zone_type)

    # setup Radiance
    if location_longitude > 0:
        print('WARNING: Longitude must be negative for the western hemisphere.')
    parameter = get_radiance_config(parameter,
                                    regenerate = False,
                                    wwr = wwr,
                                    latitude = location_latitude,
                                    longitude = -1*location_longitude,
                                    view_orient = view_orient,
                                    timezone = timezone,
                                    orient = location_orientation,
                                    elevation = elevation,
                                    width = ft_to_m(room_width),
                                    depth = ft_to_m(room_depth),
                                    height = ft_to_m(room_height),
                                    windows=[
                                        '.38 .22 2.29 .85',
                                        '.38 1.07 2.29 .85',
                                        '.38 1.98 2.29 .51'
                                    ]
                                )

    # setup occupant
    parameter = get_occupant_config(parameter,
                                    schedule = schedule,
                                    wpi_min = wpi_min,
                                    glare_max = glare_max,
                                    temp_room_max = 24,
                                    temp_room_min = 20,
                                    zone_type = zone_type,
                                    number_occupants = number_occupants)

    # Enable HVAC control
    parameter['system']['hvac_control'] = hvac_control

    # Defaults for DOPER
    parameter['objective'] = {}
    parameter['objective']['weight_energy'] = 22 # 30/7*5=21.5 Weight of tariff (energy) cost
    parameter['objective']['weight_demand'] = 1 # Weight of tariff (demand) cost in objective
    parameter['objective']['weight_export'] = 0 # Weight of revenue (export) in objective
    parameter['objective']['weight_actuation'] = weight_actuation # Weight of facade actuation
    parameter['objective']['weight_glare'] = weight_glare # Weight of glare penalty in objective
    parameter['objective']['weight_view'] = 0 # Weight of view penalty in objective
    parameter['solver_options'] = {} # Pyomo solver options
    parameter['solver_options']['seconds'] = int(60) # Maximal solver time, in seconds
    #parameter['solver_options']['maxIterations'] = int(1e6) # Maximal iterations
    parameter['solver_options']['loglevel'] = int(0) # Log level of solver
    #parameter['solver_options']['dualT'] = 1e-7
    #parameter['solver_options']['dualB'] = 1e-7
    parameter['site']['import_max'] = 1e9 # Disable import limit
    parameter['site']['export_max'] = 1e9 # Disable export limit

    # Defaults for wrapper
    parameter['wrapper'] = {}
    parameter['wrapper']['instance_id'] = instance_id # Unique instance id
    parameter['wrapper']['printing'] = debug # Console printing of solver
    parameter['wrapper']['log_overtime'] = 60-5 # Log inputs when long solving time, in seconds
    parameter['wrapper']['log_dir'] = './logs' # Directory to store logs
    parameter['wrapper']['inputs_cutoff'] = 6 # Cutoff at X digits to prevent numeric noise
    parameter['wrapper']['resample_variable_ts'] = True # Use variable timestep in model
    parameter['wrapper']['reduced_start'] = 1*60 # Time offset when variable ts starts, in minutes
    parameter['wrapper']['reduced_ts'] = 60 # Resampled timestep for reduced timestep, in minutes
    parameter['wrapper']['cols_fill'] = \
        ['temp_room_max', 'temp_room_min'] # columns to apply fill method (default is temp cols)
    parameter['wrapper']['limit_slope'] = 1 # C per 5 minute timestep
    parameter['wrapper']['precompute_radiance'] = precompute_radiance # Precompute rad for all data
    parameter['wrapper']['solver_name'] = 'cbc'
    parameter['wrapper']['solver_dir'] = None
    output_list = default_output_list(parameter) + afc_output_list()
    parameter['wrapper']['output_list'] = output_list
    parameter['wrapper']['tariff_name'] = tariff_name

    return parameter

if __name__ == '__main__':
    import pprint
    pprint.pprint(default_parameter())
