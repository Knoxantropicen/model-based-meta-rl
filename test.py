from argparse import ArgumentParser
import gym

import task
from mbmrl import MBMRL
from controller import MPPI
from net import Net
from config import test_task, test_cfg, controller_cfg, net_cfg
from tools.utils import check_task, load_cfgs
from tools.logger import setup_logger, create_log_dir

cfgs = {
    'test_task': test_task,
    'test': test_cfg,
}

def main():
    global cfgs

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--snapshot-mode', type=str, default='last')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()

    train_log_dir = create_log_dir(first_time=False, exp_prefix=args.exp_name, seed=args.train_seed)
    cfgs.update(load_cfgs(train_log_dir))

    logger = setup_logger(first_time=False, exp_prefix=args.exp_name, seed=args.train_seed,
        cfgs=cfgs, snapshot_mode=args.snapshot_mode, snapshot_gap=None)

    test_task = gym.make(cfgs['test_task'])
    sample_train_task = gym.make(cfgs['train_tasks'][0])
    # check if test task has same dimensions of observation and action as training tasks
    ob_shape, ac_shape = check_task([test_task, sample_train_task])
    del sample_train_task

    model = Net(input_shape=ob_shape + ac_shape, output_shape=ob_shape, **cfgs['net'])
    controller = MPPI(num_threads=args.num_threads, **cfgs['controller'])

    algo = MBMRL(None, model, controller, logger, num_threads=1, **cfgs['train'])
    algo.test(test_task, load_iter=args.iter, render=args.render, debug=args.debug, **cfgs['test'])


if __name__ == '__main__':
    main()
