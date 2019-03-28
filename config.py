# TODO: add tasks
train_tasks = None
test_task = None

train_cfg = {
        # general
        'seed': 0,
        'iteration_num': 1e3,
        'task_sample_num': 10,
        'task_sample_frequency': 5,
        # learning rate
        'phi_initial': 1e-2,
        'beta': 1e-3,
        'eta': 1e-3,
        # sample
        'M': 10,
        'K': 8,
        'rollout_len': 30, # >= M+K
        # dataset
        'dataset_size': 1e4,
        }

test_cfg = {
        'iteration_num': 1e3,
        'render': True,
        }

net_cfg = {
        'hid_shape': 128,
        'hid_num': 2,
        'activation': 'tanh',
        }

controller_cfg = {
        'K': 10, # sample num
        'T': 5, # horizon
        }
