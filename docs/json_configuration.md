## Defining JSON user inputs

The simplest way for a user to set up the AFC controller is through a dedicated JSON configuration file, an example can be found [here](https://github.com/LBNL-ETA/AFC/blob/master/afc/resources/config/example_config.json). The json file allows the user to adjust model parameters and preferences without any interaction with the Python code. Below is a list of parameters and their description.

```python
from afc.externalConfig import read_json_config, DEFAULT_JSON_PATH

# Path to json configuration
json_config_path = DEFAULT_JSON_PATH

# Create the inputs vector
parameter = read_json_config(json_config_path)
inputs = make_inputs(parameter, wf)
```

- `building_age`: [str] Indicator of building vintage. Used to update `parameter['zone']['param']`. Default value is 'new-constr'. Possible values are 'new-constr', 'post-1980' and 'pre-1980'.
- `debug`: [bool] Flag to enable console printing of the solver. Used to update `parameter['wrapper']['printing']`. Default value is 'False'.
- `interface_status`: [str] Status from the user interface. Default value is 'Updated Configuration'.
- `location_city`: [str] City location of the building. Default value is 'Berkeley'.
- `location_latitude`: [float] Latitude of the building, in degree. Used to update `parameter['radiance']['location']['latitude']`. Default value is 37.85.
- `location_longitude`: [float] Longitude of the building, in degree. Used to update `parameter['radiance']['location']['longitude']`. Default value is -122.24.
- `location_orientation`: [float] Orientation of the building, in degree. Note that 0 corresponds to South, 90 is West, -90 is East, and 180 is North. Used to update `parameter['radiance']['location']['orientation']`. Default value is 0.
- `location_state`: [str] State location of the building. Default value is 'CA'.
- `occupant_1_direction` [float] Orientation of the occupant inside the building, in degree relative to the building orientation. For a `location_orientation` of 90 deg (West), a `occupant_1_direction` of -90 corresponds to a view towards South. Used to update `parameter['radiance']['location']['view_orient']`. Default value is 0.
- `occupant_brightness` [float] Occupant preference for minimum work plane illuminance, in percent. 80% corresponds to 250 lux, 100% to 350 lux, and 120% to 450 lux. Used to map `parameter['occupant']['wpi_min']`. Default value is 100% (350 lux).
- `occupant_glare` [float] Occupant preference for maximum acceptable glare, in percent. 80% corresponds to dpg of 0.3, 100% to 0.4, and 120% to 0.5. Used to map `parameter['occupant']['glare_max']`. Default value is 100% (0.4). 
- `occupant_number` [int] Number of occupants in the room. Used as multiple to update unitary metrics such as the electric equipment load (`parameter['occupant']['equipment']`), thermal plug load (`parameter['occupant']['plug_load']`) and thermal occupant load (`parameter['occupant']['occupant_load']`). Default value is 1.
- `room_depth` [float] Depth of the building, in feet. Used to update `parameter['radiance']['dimensions']['depth']`. Default value is 15.
- `room_height` [float] Height of the building, in feet. Used to update `parameter['radiance']['dimensions']['height']`. Default value is 11.
- `room_width` [float] Width of the building, in feet. Used to update `parameter['radiance']['dimensions']['width']`. Default value is 10.
- `system_cooling` [str] Type of energy used by the cooling system. Default value is 'el' (for electric). No other technologies are implemented at this time.
- `system_cooling_eff` [float] The coefficient of performance of the cooling system. Used to update `parameter['zone']['cooling_efficiency']`. Default value is 3.5.
- `system_heating` [str] Type of energy used by the heating system. Default value is 'el' (for electric). No other technologies are implemented at this time.
- `system_heating_eff` [float] Electric efficiency of the heating system. If the heating system type is electric, then it is used to update `parameter['zone']['heating_efficiency']`. Otherwise, the latter parameter is set to 1. Default value is 0.95.
- `system_id` [str] Unique instance identifier. Used to update `parameter['wrapper']['instance_id']`. Default value is 'Test-AAA'.
- `system_light` [str] Type of the lighting system. Used to map `parameter['zone']['lighting_efficiency']`. Possible values are fluorescent ('FLU') which sets efficiency to 0.5 W/lux and 'LED' which sets it to 0.25 W/lux. Default value is 'LED'.
- `system_type` [str] Type of the facade. Used to update `parameter['facade']['type']`. Currently only supports the default (three-zone electrochromic window). Default value is 'ec-71t'.
- `tariff_name` [str] Name of the used tariff rate. Used to update `parameter['wrapper']['tariff_name']`. A full list of available tariffs can be found in the DOPER documentation [here](https://github.com/LBNL-ETA/DOPER/blob/master/doper/data/tariff.py). Default value is 'e19-2020'.
- `window_count` [int] Number of windows on the building facade. Used to multiply the window width. Used to update the coordinates in `parameter['radiance']['dimensions']['windows1']`, `parameter['radiance']['dimensions']['windows2']`, and `parameter['radiance']['dimensions']['windows3']`. Default value is 2.
- `window_height` [float] Height of a window, in feet. Used to update the coordinates in `parameter['radiance']['dimensions']['windows1']`, `parameter['radiance']['dimensions']['windows2']`, and `parameter['radiance']['dimensions']['windows3']`. Default value is 8.5.
- `window_sill` [float] Sill of a window, in feet. Used to update the coordinates in `parameter['radiance']['dimensions']['windows1']`, `parameter['radiance']['dimensions']['windows2']`, and `parameter['radiance']['dimensions']['windows3']`. Default value is 0.5.
- `window_width` [float] Width of a window, in feet. Used to update the coordinates in `parameter['radiance']['dimensions']['windows1']`, `parameter['radiance']['dimensions']['windows2']`, and `parameter['radiance']['dimensions']['windows3']`. Default value is 4.5.
