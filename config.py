import os.path as osp


ROOT_DIR = osp.dirname(osp.abspath(__file__))
LOCAL_LOG_DIR = osp.join(ROOT_DIR, 'data')

train_tasks = ['CartPoleTask-v0']
test_task = 'CartPoleTask-v0'

train_cfg = {
        # general
        'seed': 0,
        'iteration_num': 1e3,
        'task_sample_num': 32,
        'task_sample_frequency': 5,
        'eval_frequency': 5,
        'eval_sample_num': 5,
        # adaptation
        'adaptation_update_num': 5,
        # learning rate
        'phi': 1e-3,
        'beta': 1e-3,
        'eta': 3e-4,
        # sample
        'M': 8,
        'K': 8,
        'rollout_len': 20, # >= M+K
        'rollout_num': 8,
        # dataset
        'dataset_size': 1e3,
        # model prediction
        'pred_std': 0.1,
        # loss
        'loss_type': 'mse',
        'loss_scale': 100,
        }

test_cfg = {
        'seed': 0,
        'iteration_num': 1e3,
        'render': False,
        }

net_cfg = {
        'hid_shape': 32,
        'hid_num': 2,
        'activation': 'tanh',
        }

controller_cfg = {
        'K': 200, # sample num
        'T': 8, # horizon
        'noise_mu': 0.0,
        'noise_sigma': 10.0,
        'lamda': 1.0,
        }
