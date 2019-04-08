from argparse import ArgumentParser
import gym

import task
from mbmrl import MBMRL
from controller import MPPI
from net import Net
from config import train_tasks, train_cfg, controller_cfg, net_cfg
from tools.utils import check_task, load_cfgs
from tools.logger import setup_logger

cfgs = {
        'train_tasks': train_tasks,
        'train': train_cfg,
        'controller': controller_cfg,
        'net': net_cfg,
    }

def main():
    global cfgs

    parser = ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='default')
    parser.add_argument('--snapshot-mode', type=str, default='last')
    parser.add_argument('--snapshot-gap', type=int, default=1)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--iter', type=int, default=None)
    parser.add_argument('--num-threads', type=int, default=1)
    args = parser.parse_args()

    logger = setup_logger(first_time=not args.resume, exp_prefix=args.exp_name, seed=train_cfg['seed'],
        cfgs=cfgs, snapshot_mode=args.snapshot_mode, snapshot_gap=args.snapshot_gap)

    # load configs when resume training
    if args.resume:
        cfgs = load_cfgs(logger.get_log_dir())
    
    train_tasks = [gym.make(t) for t in cfgs['train_tasks']]

    # get shape of task space
    ob_shape, ac_shape = check_task(train_tasks)

    # build dynamics network
    model = Net(input_shape=ob_shape + ac_shape, output_shape=ob_shape, **cfgs['net'])

    # build controller for adaptation
    controller = MPPI(**cfgs['controller'])

    # train model-based meta RL
    algo = MBMRL(tasks=train_tasks, model=model, controller=controller, logger=logger, 
        num_threads=args.num_threads, **cfgs['train'])
    algo.train(resume=args.resume, load_iter=args.iter)


if __name__ == '__main__':
    main()
