import numpy as np
import torch
import gym
import json
import os
import os.path as osp


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_space_shape(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return 1
    else:
        raise Exception('unsupported type')

def check_task(tasks):
    ob_shape, ac_shape = None, None
    for task in tasks:
        task_ob_shape = get_space_shape(task.observation_space)
        task_ac_shape = get_space_shape(task.action_space)
        if ob_shape is None:
            ob_shape = task_ob_shape
            ac_shape = task_ac_shape
        else:
            assert ob_shape == task_ob_shape and ac_shape == task_ac_shape, 'shape mismatch between tasks'
    return ob_shape, ac_shape

def mkdir(path):
    os.makedirs(path, exist_ok=True)

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + '.' + o.__name__}
        elif callable(o):
            return {'$function': o.__module__ + '.' + o.__name__}
        return super().default(o)

def decode_func(dct):
    if '$class' in dct:
        module_name, class_name = dct['$class'].split('.')
        return getattr(__import__(module_name), class_name)
    elif '$function' in dct:
        module_name, func_name = dct['$function'].split('.')
        return getattr(__import__(module_name), func_name)
    return dct

def save_cfg(log_file, cfg):
    mkdir(osp.dirname(log_file))
    with open(log_file, 'w') as f:
        json.dump(cfg, f, cls=MyEncoder)

def load_cfg(log_file):
    with open(log_file, 'r') as f:
        cfg = json.load(f, object_hook=decode_func)
    return cfg

def save_cfgs(log_dir, cfgs):
    assert isinstance(cfgs, dict), 'cfgs should be like [{"cfg_type": cfg_data}]'
    mkdir(osp.join(log_dir, 'config'))
    for cfg_type, cfg_data in cfgs.items():
        cfg_log_path = osp.join(log_dir, 'config', cfg_type + '.json')
        save_cfg(cfg_log_path, cfg_data)

def load_cfgs(log_dir):
    cfg_dir = osp.join(log_dir, 'config')
    cfgs = {}
    cfg_files = [f for f in os.listdir(cfg_dir) if f.endswith('.json')]
    for cfg_file in cfg_files:
        cfg_path = osp.join(cfg_dir, cfg_file)
        cfg_type = cfg_file[:-5]
        cfgs[cfg_type] = load_cfg(cfg_path)
    return cfgs
