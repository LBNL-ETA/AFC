import math
from configparser import ConfigParser

def sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width, get_angle=False):
    """
    Assumption: view facing the window with window horizontally centered in view.
    dist: view distance to the window
    height: view height from floor
    window_bottom_height: distance of bottom of window to floor
    window_top_height: distance of top of window to floor
    window_width: window width

    Return:
        bot_alt: Bottom altitude when sun in view through the window
        top_alt: Top altitude when sun in view through the window
        left_azi: Left aziumth when sun in view through the window
        right_azi: Right aziumth when sun in view through the window
    """
    if view_height > window_top_height:
        print("WARNING: view_angle: Window below viewpoint.")
        bot_alt, top_alt, left_azi, right_azi, view_window_angle = 0, 0, 0, 0, 0
    else:
        bot_vw = abs(window_bottom_height - view_height)
        top_vw = abs(window_top_height - view_height)
        top_alt = max(0, math.degrees(math.atan(top_vw/view_dist)))
        if view_height > window_bottom_height:
            bot_alt = 0
        else:
            bot_alt = max(0, math.degrees(math.atan(bot_vw/view_dist)))
        view_window_angle = math.degrees(math.atan((window_width/2)/view_dist))
        left_azi = 90 - view_window_angle
        right_azi = 90 + view_window_angle
    if get_angle:
        return bot_alt, top_alt, left_azi, right_azi, view_window_angle
    else:
        return bot_alt, top_alt, left_azi, right_azi
        
def view_config_from_rad(cfg_path, start_lx=18e3, end_lx=11e3, fov_offset=2.5):
    # Load Radiance config
    _config = ConfigParser()
    if _config.read(cfg_path) == []:
        raise ValueError('The location of the "config" file is wrong. ' + \
            'Check location at {}.'.format(cfg_path))
    cfg = _config._sections
    
    # Parse parameters
    window_bottom_heights = []
    window_top_heights = []
    for k,v in cfg['Dimensions'].items():
        if 'window' in k:
            window_bottom_heights.append(float(v.split(' ')[1]))
            window_top_heights.append(round(window_bottom_heights[-1] + float(v.split(' ')[3]), 2))
            window_width = float(v.split(' ')[2])
    view_name = 'views' if 'views' in cfg['View'] else 'view1'
    view_dist = float(cfg['View'][view_name].split(' ')[1])
    view_height = float(cfg['View'][view_name].split(' ')[2])
    
    # Make config
    view_config = {}
    for i in range(len(window_bottom_heights)):
        bot_alt, top_alt, left_azi, right_azi, view_window_angle = sun_in_view(view_dist, view_height, window_bottom_heights[i], window_top_heights[i], window_width, get_angle=True)
        view_config[f'window{i}_start_alt'] = round(bot_alt, 2)
        view_config[f'window{i}_end_alt'] = round(top_alt, 2)
        view_config['azi_fov'] = round(view_window_angle + fov_offset, 2)
        
    view_config['start_lx'] = start_lx
    view_config['end_lx'] = end_lx
        
    return view_config
    
def make_view_config(view_dist, view_height, wwr, window_width=2.71, start_lx=18e3, end_lx=11e3, fov_offset=2.5):
    window_height_labels = ['bot', 'mid', 'top']
    # Those are for 71T emulator model
    if wwr == 0.6:
        window_top_heights = [0.99, 1.85, 2.72]
        window_bottom_heights = [0.13, 0.99, 1.86]
    elif wwr == 0.4:
        window_top_heights = [1.57, 2.14, 2.73]
        window_bottom_heights = [1.0, 1.57, 2.16]    
    
    view_config = {}
    for i in range(len(window_height_labels)):
        bot_alt, top_alt, left_azi, right_azi, view_window_angle = sun_in_view(view_dist, view_height, window_bottom_heights[i],
                                                                               window_top_heights[i], window_width, get_angle=True)
        view_config[f'{window_height_labels[i]}_start_alt'] = round(bot_alt, 2)
        view_config[f'{window_height_labels[i]}_end_alt'] = round(top_alt, 2)
        view_config['azi_fov'] = round(view_window_angle + fov_offset, 2)
        
    view_config['start_lx'] = start_lx
    view_config['end_lx'] = end_lx
        
    return view_config


if __name__ == "__main__":
    '''
    view_dist = 1.52
    view_height = 1.2
    print("View distance to window: ", view_dist)
    print("View height: ", view_height)
    window_width = 2.71
    #WWR60 top pane
    print("WWR60 top pane")
    window_bottom_height = 1.86
    window_top_height = 2.72
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    #WWR60 mid pane
    print("WWR60 mid pane")
    window_bottom_height = 0.99
    window_top_height = 1.85
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    #WWR60 bot pane
    print("WWR60 bot pane")
    window_bottom_height = .13
    window_top_height = .99
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    #WWR40 top pane
    print("WWR40 top pane")
    window_bottom_height = 2.16
    window_top_height = 2.73
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    #WWR40 mid pane
    print("WWR40 mid pane")
    window_bottom_height = 1.57
    window_top_height = 2.14
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    #WWR40 bot pane
    print("WWR40 bot pane")
    window_bottom_height = 1.0
    window_top_height = 1.57
    bot_alt, top_alt, left_azi, right_azi = sun_in_view(view_dist, view_height, window_bottom_height, window_top_height, window_width)
    print("Bottom altitude: ", bot_alt)
    print("Top altitude: ", top_alt)
    print("Left azimuth: ", left_azi)
    print("Right azimuth: ", right_azi)
    print("==============================")
    '''
    cfg_path = r'C:\Users\Christoph\Documents\PrivateRepos\DynamicFacades\emulator\resources\radiance\room0.4WWR_ec.cfg'
    view_config = view_config_from_rad(cfg_path)
    print(view_config)



