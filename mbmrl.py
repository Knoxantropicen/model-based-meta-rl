import numpy as np
import torch
from torch import nn, autograd
from torch.distributions.normal import Normal
import gym
from copy import deepcopy
from collections import deque
import os

from net import Net
from utils import set_seed


class MBMRL:
    ''' 
    Model-Based Meta-Reinforcement Learning
    Original Paper: Nagabandi et al., 2019, Learning to Adapt in Dynamic, Real-World Environments through Meta-Reinforcement Learning
    '''
    def __init__(self, tasks, model, controller, seed, iteration_num, task_sample_num, task_sample_frequency,
            rollout_len, M, K, beta, eta, dataset_size, phi_initial):
        set_seed(seed)

        self.tasks = np.array(tasks)
        self.controller = controller
        self.dataset = deque(maxlen=dataset_size)

        self.theta = model
        self.phi = []
        for param in self.theta.parameters():
            self.phi.append(torch.full_like(param, phi_initial))

        self.iteration_num = iteration_num
        self.task_sample_num = task_sample_num
        self.task_sample_frequency = task_sample_frequency
        self.rollout_len = rollout_len
        self.M = M
        self.K = K
        self.beta = beta
        self.eta = eta

    def load_param(self):
        # TODO: support
        pass

    def save_param(self):
        # TODO: support
        pass

    def compute_theta_loss(self, theta, traj):
        # traj: [[s1, a1, s2], [s2, a2, s3], ...]
        loss = 0
        for i in len(traj):
            state, action, next_state = traj[i]
            dyn_input = torch.cat((state, action), 0)
            dyn_output = theta(dyn_input)
            dyn_normal = Normal(dyn_output)
            loss -= dyn_normal.log_prob(next_state)
        return loss

    def compute_d_theta(self, theta, loss):
        return autograd.grad(loss, theta.parameters())

    def compute_d2_theta(self, theta, loss):
        d_theta = autograd.grad(loss, theta.parameters(), create_graph=True)
        d2_theta = autograd.grad(d_theta, theta.parameters())
        return d2_theta, d_theta

    def adaptation_update(self, theta, traj, get_d2=False):
        loss = self.compute_theta_loss(traj, theta)
        if get_d2:
            d2_theta, d_theta = self.compute_d2_theta(theta, loss)
        else:
            d_theta = self.compute_d_theta(theta, loss)
        new_theta = deepcopy(theta)
        for i, (param, d_t) in enumerate(zip(new_theta.parameters(), d_theta)):
            param.data -= self.phi[i] * d_t / len(traj)
        if get_d2:
            return new_theta, d_theta, d2_theta
        else:
            return new_theta, d_theta

    def meta_update(self, new_losses, new_thetas, d_thetas, d2_thetas):
        d_theta_sum = 0
        d_phi_sum = 0
        for new_loss, new_theta, d_theta, d2_theta in zip(new_losses, new_thetas, d_thetas, d2_thetas):
            d_new_theta = self.compute_d_theta(new_theta, new_loss)
            for i, d2 in enumerate(d2_theta):
                d2 *= self.phi[i]
            d_theta_sum += d_new_theta * (1 - d2_theta)
            d_phi_sum += d_new_theta * (-d_theta)
        d_theta_sum = d_theta_sum * self.beta / self.task_sample_num
        d_phi_sum = d_phi_sum * self.eta / self.task_sample_num
        for i, (param, d_t, d_p) in enumerate(zip(self.theta.parammeters(), d_theta_sum, d_phi_sum)):
            param.data -= d_t
            self.phi[i] -= d_p

    def collect_traj(self, task):
        rollout = []
        state = task.reset()
        self.controller.set_task(task)
        t = 0
        while t < self.rollout_len:
            past_traj = rollout[-self.M:]
            new_theta, _ = self.adaptation_update(past_traj, self.theta)
            action = self.controller.plan(new_theta, state)
            next_state, reward, done, _ = task.step(action)
            rollout.append([torch.tensor(state), torch.tensor(action), torch.tensor(next_state)])
            t += 1
            if done:
                if t < self.rollout_len:
                    rollout = []
                    state = task.reset()
                    t = 0
                    # TODO: what if trajectory is always shorter than M+K
                    print('collect trajectory failed, rollout not long enough')
        return rollout

    def sample_traj(self):
        rollout = np.random.choice(self.dataset)
        start_idx = np.random.choice(len(rollout) + 1 - self.M - self.K)
        traj = rollout[start_idx: start_idx + self.M + self.K]
        return traj

    def sample_task(self):
        # TODO: support task distribution
        task = np.random.choice(self.tasks)
        return task

    def train(self):
        for i in range(self.iteration_num):
            if i % self.task_sample_frequency == 0:
                task = self.sample_task()
                # collect trajectories on sampled tasks and aggregate to dataset
                # TODO: check if only collect one rollout on one task
                rollout = self.collect_traj(task)
                self.dataset.append(rollout)
                np.random.shuffle(self.dataset)
            new_losses, new_thetas, d_thetas, d2_thetas = [], [], [], []
            for j in range(self.task_sample_num):
                # sample M+K steps from dataset
                traj = self.sample_traj()
                # do adaptation update, get new theta and gradients
                new_theta, d_theta, d2_theta = self.adaptation_update(traj[:self.M], self.theta, get_d2=True)
                # compute loss using new theta
                new_loss = self.compute_theta_loss(new_theta, traj[self.M:])
                new_losses.append(new_loss)
                new_thetas.append(new_theta)
                d_thetas.append(d_theta)
                d2_thetas.append(d2_theta)
            # do meta update on theta and phi
            self.meta_update(new_losses, new_thetas, d_thetas, d2_thetas)

    def test(self, task, iteration_num, render):
        rollout = []
        state = task.reset()
        self.controller.set_task(task)
        theta = deepcopy(self.theta)
        for i in range(iteration_num):
            past_traj = rollout[-self.M:]
            new_theta, _ = self.adaptation_update(past_traj, theta)
            action = self.controller.plan(new_theta, state)
            next_state, reward, done, _ = task.step(action)
            theta = new_theta
            if render:
                # TODO: check render() arguments
                task.render()
            rollout.append([torch.tensor(state), torch.tensor(action), torch.tensor(next_state)])
            if done:
                rollout = []
                state = task.reset()

