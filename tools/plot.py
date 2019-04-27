from argparse import ArgumentParser
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from datetime import datetime
import dateutil.tz

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ROOT_DIR, LOCAL_LOG_DIR

LOG_MAP = {
    'loss': 'Model Loss',
    'reward': 'Reward',
    'iter': 'Iteration',
    'model_step': 'Total Model Steps',
    'task_step': 'Total Task Steps',
    'time': 'Total Time (s)',
}


def get_index_from_csv_head(desc, name):
    try:
        return desc.index(name)
    except:
        print('%s doesn\'t exist in progress csv!' % name)
        raise


def plot_rewards(progress_csvs, save_dir,
    value='loss', by='iter', do_fit=False, fit_only=False, fit_order=6, y_range=None, x_range=None):

    ax = plt.gca()

    exp_names = []
    for progress in progress_csvs:
        desc, data, exp_name, plot_name = progress['desc'], progress['data'], progress['exp_name'], progress['plot_name']
        exp_names.append(exp_name)

        x = data[:, desc.index(LOG_MAP[by])]
        if x_range is not None:
            left = np.argmax(np.array(x) >= x_range[0])
            right = np.argmin(np.array(x) <= x_range[1])
            if right == 0:
                right = len(x) - 1
            x = x[left:right]

        if value == 'reward':
            value_idxs = [get_index_from_csv_head(desc, desc_item) for desc_item in desc if desc_item.startswith(LOG_MAP[value])]
            ys = [data[:, value_idx] for value_idx in value_idxs]
        else:
            value_idx = get_index_from_csv_head(desc, LOG_MAP[value])
            y = data[:, value_idx]
            ys = [y]
        
        for y in ys:
            if x_range is not None:
                y = y[left:right]
            if y_range is not None:
                y = np.clip(y, y_range[0], y_range[1])

            color = next(ax._get_lines.prop_cycler)['color']
            if do_fit:
                if not fit_only:
                    plt.plot(x, y, color=color, alpha=0.5)
                fit = np.polyfit(x, y, fit_order)
                fit = np.polyval(fit, x)
                plt.plot(x, fit, lw=2, label=plot_name, color=color)
            else:
                plt.plot(x, y, label=plot_name)

    plt.ticklabel_format(style='sci', scilimits=(-1e3, 1e3), axis='x')
    plt.ticklabel_format(style='sci', scilimits=(-1e5, 1e5), axis='y')
    plt.legend(loc='lower right')
    plt.xlabel(LOG_MAP[by])
    plt.ylabel(LOG_MAP[value])
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    env_save_dir = os.path.join(save_dir, '_'.join(exp_names))
    os.makedirs(env_save_dir, exist_ok=True)
    plt.savefig(os.path.join(env_save_dir, '_'.join(exp_names) + '_' + value + '_by_' + by + '_' + 
        datetime.now(dateutil.tz.tzlocal()).strftime('%m%d%H%M') + '.png'))
    plt.close()


def main():

    parser = ArgumentParser(description='Plot')
    parser.add_argument('--exp-name', type=str, nargs='+')
    parser.add_argument('--plot-name', default=None, type=str, nargs='+')
    parser.add_argument('--value', type=str, default='loss')
    parser.add_argument('--fit', default=False, action='store_true')
    parser.add_argument('--fit-only', default=False, action='store_true')
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--by', default='iter')
    parser.add_argument('--y-range', default=None, type=float, nargs='+')
    parser.add_argument('--x-range', default=None, type=float, nargs='+')
    args = parser.parse_args()
    for val in [args.value, args.by]:
        assert val in LOG_MAP, '%s is invalid argument!' % val
    if args.x_range is not None:
        assert len(args.x_range) == 2, 'invalid range input!'
    if args.y_range is not None:
        assert len(args.y_range) == 2, 'invalid range input!'
    if args.plot_name is not None:
        assert len(args.plot_name) == len(args.exp_name), 'invalid plot name for all experiments!'
    else:
        args.plot_name = args.exp_name

    progress_csvs = []
    for exp_name, plot_name in zip(args.exp_name, args.plot_name):
        file_name = os.path.join(LOCAL_LOG_DIR, exp_name, 'progress.csv')
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            description = next(reader)
        data = np.genfromtxt(file_name, delimiter=',')[1:]
        progress_csvs.append({'desc': description, 'data': data, 'exp_name': exp_name, 'plot_name': plot_name})

    save_dir = os.path.join(ROOT_DIR, 'plot')
    plot_rewards(progress_csvs, save_dir,
        value=args.value, by=args.by, do_fit=args.fit, fit_only=args.fit_only, fit_order=args.order, y_range=args.y_range, x_range=args.x_range)


if __name__ == '__main__':
    main()
