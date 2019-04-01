import os.path as osp


ROOT_DIR = osp.dirname(osp.abspath(__file__))
LOCAL_LOG_DIR = osp.join(ROOT_DIR, 'data')

train_tasks = ['CartPoleTask-v0']
test_task = 'CartPoleTask-v0'

train_cfg = {
        # general
        'seed': 0,
        'iteration_num': 1e3,
        'task_sample_num': 100,
        'task_sample_frequency': 5,
        # adaption
        'adaption_update_num': 5,
        # learning rate
        'phi': 1e-2,
        'beta': 1e-3,
        'eta': 1e-3,
        # sample
        'M': 10,
        'K': 8,
        'rollout_len': 30, # >= M+K
        # dataset
        'dataset_size': 1e4,
        # model prediction
        'pred_std': 0.1,
        }

test_cfg = {
        'seed': 0,
        'iteration_num': 1e3,
        'render': True,
        }

net_cfg = {
        'hid_shape': 32,
        'hid_num': 2,
        'activation': 'tanh',
        }

controller_cfg = {
        'K': 10, # sample num
        'T': 5, # horizon
        }
