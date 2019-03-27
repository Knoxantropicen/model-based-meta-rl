from mlmrl import MBMRL
from config import train_cfg, train_tasks

def main():
    # TODO: prepare controller
    controller = None
    algo = MBMRL(train_tasks, controller, train_cfg)
    algo.train()

if __name__ == '_main__':
    main()
