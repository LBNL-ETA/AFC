import sys

def default_parameter_dynfacade(path_to_doper=None):

    if path_to_doper: sys.path.append(path_to_doper)
    try:
        from doper.examples import default_parameter
    except:
        raise ImportError('Path to DOPER correct? {}'.format(path_to_doper))

    parameter = default_parameter()
    
    # Disable battery
    for k in parameter['battery'].keys():
        parameter['battery'][k] = [0]
    parameter['battery']['count'] = 1
    parameter['battery']['efficiency_charging'] = [1]
    parameter['battery']['efficiency_discharging'] = [1]
    
    # Change weights for 1 month (30 days)
    parameter['objective']['weight_energy'] = 22 # 30/7*5=21.5 # Weight of tariff (energy) cost in objective
    parameter['objective']['weight_demand'] = 1 # Weight of tariff (demand) cost in objective
    parameter['objective']['weight_export'] = 0 # Weight of revenue (export) in objective
    parameter['objective']['weight_regulation'] = 0 # Weight of revenue (regulation) in objective
    del parameter['objective']['weight_degradation']

    # New for MPC Dynamic Facades
    parameter['facade'] = {}
    parameter['facade']['type'] = 'ec'
    parameter['facade']['windows'] = [0, 1, 2]
    parameter['facade']['states'] = [0, 1, 2, 3]
    parameter['facade']['rad_cutoff'] = {}
    parameter['facade']['rad_cutoff']['wpi'] = [5, 1e6] # lx
    parameter['facade']['rad_cutoff']['ev'] = [250, 1e6] # (0.2 - 0.184) / 6.22e-5 = 250 lx
    parameter['facade']['rad_cutoff']['shg'] = [0, 1e6] # W
    parameter['facade']['rad_cutoff']['abs1'] = [50, 1e6] # W
    parameter['facade']['rad_cutoff']['abs2'] = [5, 1e6] # W
    parameter['facade']['rad_cutoff']['tsol'] = [5, 1e6] # W
    parameter['facade']['rad_cutoff']['iflr'] = [5, 1e6] # W  
    parameter['facade']['window_area'] = 2.56 * 2.782751 # Update with different WWR!
    parameter['facade']['convection_window_scale'] = 4 # From model
    parameter['facade']['convection_window_offset'] = 4 # From model
    parameter['zone'] = {}
    parameter['zone']['lighting_capacity'] = 1000 # lx
    parameter['zone']['lighting_efficiency'] = 0.24 # W/lx
    parameter['zone']['temps_name'] = ['room', 'slab'] #, 'wall']
    parameter['zone']['temps_initial'] = [22.5, 22.5, 22.5]
    #parameter['zone']['param'] = {'type':'R2C2','Roi':0.15,'Ci':0.5e5,'Ris':0.02,'Cs':1e6}
    #parameter['zone']['param'] = {'type':'R6C3', 'Row1': 1.099997353695485, 'Rw1w2': 0.06713534063954742, 'Rw2i': 0.01391462357138664, 'Ci': 30000.00456023709, 'Ris': 0.01720299150366536, 'Cs': 3121887.1399409827, 'Riw': 0.012087409467249222, 'Cw': 824698.3534759376, 'Roi': 0.28988292286870565}
    parameter['zone']['param'] = {'type':'R4C2', 'Row1': 0.0037992496008808323, 'Rw1w2': 0.10706442491986229, 'Rw2i': 3.3602377759986217e-07, 'Ci': 211414.5114368095, 'Ris': 0.012804832879362456, 'Cs': 3268802.970556823}
    parameter['zone']['heat_max'] = [1e3, 1e3, 1e3] # W_th
    parameter['zone']['cool_max'] = [1e3, 1e3, 1e3] # W_th
    parameter['zone']['heating_efficiency'] = 1
    parameter['zone']['cooling_efficiency'] = 1/3.5 # Match emulator
    parameter['zone']['lighting_split'] = 0.5 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['plugload_split'] = 0 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['occupancy_split'] = 1 # Match to emulator (1=rad, 0=conv)
    parameter['zone']['tsol_split'] = 1 # All Tsol on surfaces (1=rad, 0=conv)
    parameter['zone']['shg_split'] = 0 # All SHG in air 1=rad, 0=conv
    parameter['zone']['glare_diff'] = 0.1 # Lower bound of glare penalty (glare_max - glare_diff)
    parameter['zone']['glare_scale'] = 10 # Scale of glare cost function (ATTENTION absolute value)
    parameter['zone']['view_scale'] = 0.1 # Scale of view cost function (ATTENTION absolute value)
    parameter['zone']['fstate_initial'] = [3, 3, 3] # Initiali state of facade
    parameter['objective']['weight_actuation'] = 0.01 # Weight of facade actuation in objective
    parameter['objective']['weight_glare'] = 0 # Weight of glare penalty in objective
    parameter['objective']['weight_view'] = 0 # Weight of view penalty in objective
    parameter['options'] = {}
    parameter['options']['seconds'] = int(60) # Maximal solver time, in seconds
    #parameter['options']['maxIterations'] = int(1e6) # Maximal iterations
    parameter['options']['loglevel'] = int(0) # Log level of solver
    #parameter['options']['dualT'] = 1e-7
    #parameter['options']['dualB'] = 1e-7
    parameter['wrapper'] = {}
    parameter['wrapper']['printing'] = False # Console printing of solver
    parameter['wrapper']['log_overtime'] = 60-5 # Log inputs when long solving time, in seconds
    parameter['wrapper']['inputs_cutoff'] = 6 # Cutoff at X digits to prevent numeric noise
    parameter['wrapper']['resample_variable_ts'] = True # Use variable timestep in model
    parameter['wrapper']['reduced_start'] = 1*60 # Time offset when variable timestep starts, in minutes
    parameter['wrapper']['reduced_ts'] = 60 # Resampled timestep for reduced timestep, in minutes
    parameter['wrapper']['precompute_radiance'] = True # Precompute radiance for full period

    return parameter

if __name__ == '__main__':
    print(default_parameter_dynfacade())