from argparse import ArgumentParser

from mbmrl import MBMRL
from controller import MPPI
from net import Net
from config import train_tasks, train_cfg, controller_cfg, net_cfg
from utils import check_task, load_cfgs
from logger import setup_logger


cfgs = {
    'tasks': train_tasks,
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
    args = parser.parse_args()

    logger = setup_logger(first_time=not args.resume, exp_prefix=args.exp_name, seed=train_cfg['seed'],
        cfgs=cfgs, snapshot_mode=args.snapshot_mode, snapshot_gap=args.snapshot_gap)

    # load configs when resume training
    if args.resume:
        cfgs = load_cfgs(logger.get_log_dir())

    # get shape of task space
    ob_shape, ac_shape = check_task(cfgs['tasks'])

    # build dynamics network
    model = Net(input_shape=ob_shape + ac_shape, output_shape=ob_shape, **cfgs['net'])

    # build controller for adaptation
    controller = MPPI(**cfgs['controller'])

    # train model-based meta RL
    algo = MBMRL(train_tasks, model, controller, logger, **cfgs['train'])
    algo.train(resume=args.resume, load_iter=args.iter)


if __name__ == '_main__':
    main()
