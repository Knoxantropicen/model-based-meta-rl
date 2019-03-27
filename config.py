train_cfg = {
        # general
        'seed': 0,
        'iteration_num': 1e3,
        'task_sample_num': 10,
        'task_sample_frequency': 5,
        # learning rate
        'phi_initial': 1e-2,
        'M': 10,
        'K': 8,
        'beta': 1e-3,
        'eta': 1e-3,
        # network
        'hid_shape': 128,
        'hid_num': 2,
        'activation': 'tanh',
        # dataset
        'dataset_size': 1e4,
        # controller
        'plan_horizon': 5,
        }

test_cfg = {
        'iteration_num': 1e3,
        'render': True,
        }

# TODO: add tasks
train_tasks = None
test_task = None
