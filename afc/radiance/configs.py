import os

root = os.path.dirname(os.path.abspath(__file__))

def get_config(mode, wwr, abs_path=True, root=os.path.join(root, '..', '..', 'resources', 'radiance')):
    filestruct = {}
    if mode == 'shade':
        config_path = os.path.join(root, 'room{}WWR_shade.cfg'.format(wwr))
        filestruct['resources'] = os.path.join(root, 'BSDF_shade')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'dshade':
        mode = 'shade'
        config_path = os.path.join(root, 'room{}WWR_shade.cfg'.format(wwr))
        filestruct['resources'] = os.path.join(root, 'BSDF_dshade')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'blinds':
        config_path = os.path.join(root, 'room{}WWR_blinds.cfg'.format(wwr))
        filestruct['resources'] = os.path.join(root, 'BSDF_blinds{}'.format(wwr))
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    elif mode == 'ec':
        config_path = os.path.join(root, 'room{}WWR_ec.cfg'.format(wwr))
        filestruct['resources'] = os.path.join(root, 'BSDFs')
        filestruct['matrices'] = os.path.join(root, 'matrices', mode, str(wwr))
    else:
        print('ERROR: Mode {} not supported.'.format(mode))
    if abs_path:   
        config_path = os.path.abspath(config_path)
        filestruct['resources'] = os.path.abspath(filestruct['resources'])
        filestruct['matrices'] = os.path.abspath(filestruct['matrices'])
    return filestruct, config_path