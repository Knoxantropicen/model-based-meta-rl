from mbmrl import MBMRL
from controller import MPPI
from net import Net
from config import train_tasks, train_cfg, controller_cfg, net_cfg
from utils import check_task


def main():
    # get shape of task space
    ob_shape, ac_shape = check_task(train_tasks)

    # build dynamics network
    model = Net(input_shape=ob_shape + ac_shape, output_shape=ob_shape, **net_cfg)

    # build controller for adaptation
    controller = MPPI(**controller_cfg)
    
    # train model-based meta RL
    algo = MBMRL(train_tasks, model, controller, **train_cfg)
    algo.train()


if __name__ == '_main__':
    main()
