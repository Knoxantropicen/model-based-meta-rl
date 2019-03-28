import numpy as np
import torch
import gym


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_space_shape(space):
    if isinstance(space, gym.spaces.Box):
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
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

