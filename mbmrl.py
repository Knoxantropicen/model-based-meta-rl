import numpy as np
import torch
from torch import nn, autograd
from torch.distributions.normal import Normal
import gym
from copy import deepcopy
from collections import deque, OrderedDict
import os
import os.path as osp
import gtimer as gt

from net import Net
from utils import set_seed


class MBMRL:
    ''' 
    Model-Based Meta-Reinforcement Learning
    Original Paper: Nagabandi et al., 2019, 'Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning'
    '''
    def __init__(self, tasks, model, controller, logger, seed, iteration_num, task_sample_num, task_sample_frequency,
            rollout_len, adaption_update_num, M, K, beta, eta, phi, dataset_size):
        set_seed(seed)
        self.logger = logger

        self.tasks = np.array(tasks)
        self.controller = controller
        self.dataset = deque(maxlen=dataset_size)

        self.iteration_num = iteration_num
        self.task_sample_num = task_sample_num
        self.task_sample_frequency = task_sample_frequency
        self.rollout_len = rollout_len
        self.adaption_update_num = adaption_update_num
        self.M = M
        self.K = K
        self.beta = beta
        self.eta = eta

        self.theta = model
        self.meta_optimizer = torch.optim.Adam(self.theta.parameters(), lr=self.beta)
        self.phi = torch.tensor(phi, requires_grad=True)
        self.lr_optimizer = torch.optim.Adam([self.phi], lr=self.eta)

        self.theta_loss = None
        self._epoch_start_time = None
        self._n_task_steps_total = 0
        self._n_model_steps_total = 0
        self._n_rollouts_total = 0

    def _get_params(self, iter):
        return {
            'iteration': iter,
            'theta': self.theta.state_dict(),
            'phi': self.phi,
            'loss': self.theta_loss,
            'n_task_steps': self._n_task_steps_total,
            'n_model_steps': self._n_model_steps_total,
            'n_rollouts': self._n_rollouts_total,
            }
    
    def _set_params(self, params):
        self.theta.load_state_dict(params['theta'])
        self.phi = params['phi']
        self.theta_loss = params['loss']
        self._n_task_steps_total = params['n_task_steps']
        self._n_model_steps_total = params['n_model_steps']
        self._n_rollouts_total = params['n_rollouts']
        return params['iter']
        
    def _get_extra_data(self):
        return {
            'dataset': self.dataset,
            }
    
    def _set_extra_data(self, extra_data):
        self.dataset = extra_data['dataset']

    def _compute_theta_loss(self, traj, new_theta=None):
        # traj: [[s1, a1, s2], [s2, a2, s3], ...]
        loss = 0
        for i in len(traj):
            state, action, next_state = traj[i]
            dyn_input = torch.cat((state, action), 0)
            delta_state = self.theta(dyn_input, new_params=new_theta)
            dyn_normal = Normal(state + delta_state)
            loss -= dyn_normal.log_prob(next_state)
            self._n_model_steps_total += 1
        return loss

    def _adaptation_update(self, traj):
        loss = self._compute_theta_loss(traj)
        d_theta = autograd.grad(loss, self.theta.parameters())

        new_theta_dict = {key: val.clone() for key, val in self.theta.state_dict().items()}
        new_theta_params = OrderedDict()
        for (key, val), d in zip(self.theta.named_parameters(), d_theta):
            new_theta_params[key] = val - self.phi * d
            new_theta_dict[key] = new_theta_params[key]

        def zero_grad(params):
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
        
        for _ in range(self.adaption_update_num):
            new_loss = self._compute_theta_loss(traj, new_theta_dict)
            zero_grad(new_theta_params.values())
            d_theta = autograd.grad(new_loss, new_theta_params.values(), create_graph=True)
            for (key, val), d in zip(self.theta.named_parameters(), d_theta):
                new_theta_params[key] = val - self.phi * d
                new_theta_dict[key] = new_theta_params[key]

        return new_theta_dict

    def _meta_update(self, meta_loss):
        self.meta_optimizer.zero_grad()
        self.lr_optimizer.zero_grad()
        meta_loss.backward(retain_graph=True)
        self.meta_optimizer.step()
        self.lr_optimizer.step()

    def _collect_traj(self, task):
        rollout = []
        state = task.reset()
        self.controller.set_task(task)
        t = 0
        while t < self.rollout_len:
            past_traj = rollout[-self.M:]
            new_theta_dict = self._adaptation_update(past_traj)
            action = self.controller.plan(self.theta, new_theta_dict, state)
            next_state, _, done, _ = task.step(action)
            rollout.append([torch.tensor(state), torch.tensor(action), torch.tensor(next_state)])
            t += 1
            self._n_task_steps_total += 1
            if done:
                if t < self.rollout_len:
                    rollout = []
                    state = task.reset()
                    t = 0
                    # TODO: what if trajectory is always shorter than M+K
                    print('collect trajectory failed, rollout not long enough')
        return rollout

    def _sample_traj(self):
        rollout = np.random.choice(self.dataset)
        start_idx = np.random.choice(len(rollout) + 1 - self.M - self.K)
        traj = rollout[start_idx: start_idx + self.M + self.K]
        return traj

    def _sample_task(self):
        # TODO: support task distribution
        task = np.random.choice(self.tasks)
        return task

    def _start_iteration(self, iter):
        self.logger.push_prefix('Iteration #%d | ' % iter)

    def _end_iteration(self):
        self._record_stats()
        self.logger.pop_prefix()

    def _record_stats(self):
        times_itrs = gt.get_times().stamps.itrs
        sample_time = times_itrs['sample'][-1]
        adaptation_time = times_itrs['adaptation'][-1]
        meta_time = times_itrs['meta'][-1]
        iter_time = sample_time + adaptation_time + meta_time
        total_time = gt.get_times().total

        self.logger.record_tabular('Model Loss', self.theta_loss)
        self.logger.record_tabular('Dataset Size', len(self.dataset))
        self.logger.record_tabular('Total Model Steps', self._n_model_steps_total)
        self.logger.record_tabular('Total Task Steps', self._n_task_steps_total)
        self.logger.record_tabular('Total Rollouts', self._n_rollouts_total)
        self.logger.record_tabular('Sample Time (s)', sample_time)
        self.logger.record_tabular('Adaptation Time (s)', adaptation_time)
        self.logger.record_tabular('Meta Time (s)', meta_time)
        self.logger.record_tabular('Iteration Time (s)', iter_time)
        self.logger.record_tabular('Total Time (s)', total_time)
        
        self.logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def train(self, resume=False, load_iter=None):
        gt.reset()
        gt.set_def_unique(False)
        start_iter = 0

        if resume:
            params = self.logger.load_params(load_iter)
            start_iter = self._set_params(params)
            self.theta.train()

            extra_data = self.logger.load_extra_data()
            self._set_extra_data(extra_data)

        for i in gt.timed_for(range(start_iter, self.iteration_num), save_itrs=True):
            self._start_iteration(i)

            gt.stamp('sample')
            if i % self.task_sample_frequency == 0:
                self.logger.log('Data Collection')
                task = self._sample_task()
                # collect trajectories on sampled tasks and aggregate to dataset
                # TODO: check if only collect one rollout on one task
                rollout = self._collect_traj(task)
                self._n_rollouts_total += 1
                self.dataset.append(rollout)
                np.random.shuffle(self.dataset)

            gt.stamp('adaptation')
            self.logger.log('Adaptation Update')
            new_losses = []
            for j in range(self.task_sample_num):
                # sample M+K steps from dataset
                traj = self._sample_traj()
                # do adaptation update, get new theta and gradients
                new_theta_dict = self._adaptation_update(traj[:self.M])
                # compute loss using new theta
                new_loss = self._compute_theta_loss(traj[self.M:], new_theta_dict)
                new_losses.append(new_loss)
            self.theta_loss = torch.mean(new_losses)

            # do meta update on theta and phi
            gt.stamp('meta')
            self.logger.log('Meta Update')
            self._meta_update(self.theta_loss)

            self._end_iteration()
            self.logger.save_params(i, self._get_params(i))
            self.logger.save_extra_data(self._get_extra_data())

    def test(self, task, iteration_num, render):
        rollout = []
        state = task.reset()
        self.controller.set_task(task)
        for i in range(iteration_num):
            past_traj = rollout[-self.M:]
            new_theta_dict = self._adaptation_update(past_traj)
            action = self.controller.plan(self.theta, state, new_theta_dict)
            next_state, _, done, _ = task.step(action)
            self.theta.load_state_dict(new_theta_dict)
            if render:
                # TODO: check render() arguments
                task.render()
            rollout.append([torch.tensor(state), torch.tensor(action), torch.tensor(next_state)])
            if done:
                rollout = []
                state = task.reset()

