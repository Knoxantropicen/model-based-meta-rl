import os
import os.path as osp
import sys
import json
import datetime
import dateutil.tz
import pickle
import csv

import config
from utils import mkdir, save_cfgs
from tabulate import tabulate

class Logger:
    '''
    '''
    def __init__(self):
        self._log_dir = None

        self._snapshot_dir = None
        self._snapshot_mode = 'last'
        self._snapshot_gap = 1

        self._text_outputs = []
        self._tabular_outputs = []
        self._text_fds = {}
        self._tabular_fds = {}
        self._tabular_header_written = set()

        self._prefixes = []
        self._prefix_str = ''
        self._tabular = []
        self._tabular_prefixes = []
        self._tabular_prefix_str = ''

    def _add_output(self, file_name, arr, fds, mode='a'):
        if file_name not in arr:
            mkdir(osp.dirname(file_name))
            arr.append(file_name)
            fds[file_name] = open(file_name, mode)
    
    def _remove_output(self, file_name, arr, fds):
        if file_name in arr:
            fds[file_name].close()
            del fds[file_name]
            arr.remove(file_name)

    def add_text_output(self, file_name):
        self._add_output(file_name, self._text_outputs, self._text_fds, mode='a')
    
    def remove_text_output(self, file_name):
        self._remove_output(file_name, self._text_outputs, self._text_fds)

    def add_tabular_output(self, file_name, first_time=False, relative_to_snapshot_dir=False):
        if first_time:
            if relative_to_snapshot_dir:
                file_name = osp.join(self._snapshot_dir, file_name)
            self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode='w')
        else:
            self._add_output(file_name, self._tabular_outputs, self._tabular_fds, mode='a')
            for tabular_fd in self._tabular_fds:
                self._tabular_header_written.add(tabular_fd)

    def remove_tabular_output(self, file_name, relative_to_snapshot_dir=False):
        if relative_to_snapshot_dir:
            file_name = osp.join(self._snapshot_dir, file_name)
        if self._tabular_fds[file_name] in self._tabular_header_written:
            self._tabular_header_written.remove(self._tabular_fds[file_name])
        self._remove_output(file_name, self._tabular_outputs, self._tabular_fds)

    def set_log_dir(self, log_dir):
        self._log_dir = log_dir

    def get_log_dir(self):
        return self._log_dir

    def set_snapshot_dir(self, dir_name):
        self._snapshot_dir = dir_name

    def get_snapshot_dir(self):
        return self._snapshot_dir

    def set_snapshot_mode(self, mode):
        self._snapshot_mode = mode

    def get_snapshot_mode(self):
        return self._snapshot_mode

    def set_snapshot_gap(self, gap):
        self._snapshot_gap = gap

    def get_snapshot_gap(self):
        return self._snapshot_gap

    def push_prefix(self, prefix):
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self):
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def log(self, s, with_prefix=True, with_timestamp=True):
        out = s
        if with_prefix:
            out = self._prefix_str + out
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)
        print(out)
        for fd in list(self._text_fds.values()):
            fd.write(out + '\n')
            fd.flush()
        sys.stdout.flush()

    def record_tabular(self, key, val):
        self._tabular.append(
            (self._tabular_prefix_str + str(key), str(val)))

    def dump_tabular(self, *args, **kwargs):
        wh = kwargs.pop("write_header", None)
        if len(self._tabular) > 0:
            for line in tabulate(self._tabular).split('\n'):
                self.log(line, *args, **kwargs)
            tabular_dict = dict(self._tabular)
            for tabular_fd in list(self._tabular_fds.values()):
                writer = csv.DictWriter(tabular_fd,
                                        fieldnames=list(
                                            tabular_dict.keys()))
                if wh or (
                        wh is None and tabular_fd not in self._tabular_header_written):
                    writer.writeheader()
                    self._tabular_header_written.add(tabular_fd)
                writer.writerow(tabular_dict)
                tabular_fd.flush()
            del self._tabular[:]

    def push_tabular_prefix(self, key):
        self._tabular_prefixes.append(key)
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def pop_tabular_prefix(self, ):
        del self._tabular_prefixes[-1]
        self._tabular_prefix_str = ''.join(self._tabular_prefixes)

    def save_extra_data(self, data, file_name='extra_data.pkl'):
        file_name = osp.join(self._snapshot_dir, file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_extra_data(self, file_name='extra_data.pkl'):
        file_name = osp.join(self._snapshot_dir, file_name)
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data

    def save_params(self, iter, params):
        if self._snapshot_dir:
            if self._snapshot_mode == 'all':
                file_name = osp.join(self._snapshot_dir, 'iter_%d.pkl' % iter)
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap":
                if iter % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'iter_%d.pkl' % iter)
                    pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == "gap_and_last":
                if iter % self._snapshot_gap == 0:
                    file_name = osp.join(self._snapshot_dir,
                                         'iter_%d.pkl' % iter)
                    pickle.dump(params, open(file_name, "wb"))
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                pickle.dump(params, open(file_name, "wb"))
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError

    def load_params(self, iter=None):
        if self._snapshot_dir:
            if self._snapshot_mode in ['all', 'gap'] or (self._snapshot_mode == 'gap_and_last' and iter is not None):
                assert iter is not None, 'must specify iteration when loading params'
                file_name = osp.join(self._snapshot_dir, 'iter_%d.pkl' % iter)
                with open(file_name, 'rb') as f:
                    return pickle.load(f)
            elif self._snapshot_mode in ['last', 'gap_and_last']:
                file_name = osp.join(self._snapshot_dir, 'params.pkl')
                with open(file_name, 'rb') as f:
                    return pickle.load(f)
            elif self._snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError


def create_exp_name(exp_prefix, seed=0):
    return "%s_s-%d" % (exp_prefix, seed)

def create_log_dir(first_time, exp_prefix, seed=0):
    exp_name = create_exp_name(exp_prefix, seed)
    log_dir = osp.join(config.LOCAL_LOG_DIR, exp_name)
    if osp.exists(log_dir) and first_time:
        print('warning: log directory already exists {}'.format(log_dir))
    mkdir(log_dir)
    return log_dir

def setup_logger(first_time=True, exp_prefix="default", seed=0,
        cfgs=None, text_log_file='debug.log', tabular_log_file='progress.csv', 
        snapshot_mode='last', snapshot_gap=1):
    '''
    '''
    logger = Logger()
    log_dir = create_log_dir(first_time, exp_prefix, seed)
    logger.set_log_dir(log_dir)

    if cfgs is not None and first_time:
        save_cfgs(log_dir, cfgs)

    text_log_path = osp.join(log_dir, text_log_file)
    tabular_log_path = osp.join(log_dir, tabular_log_file)
    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path, first_time=first_time)

    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)

    exp_name = log_dir.split('/')[-1]
    logger.push_prefix('[%s] ' % exp_name)

    return logger
