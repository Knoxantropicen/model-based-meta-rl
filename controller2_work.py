import numpy as np
import torch
from copy import deepcopy

from tools.utils import get_space_shape


class Controller:
    '''
    Basic class of custom controllers, several methods required
    '''
    def set_task(self, task):
        raise NotImplementedError

    def plan(self, dynamics, state):
        raise NotImplementedError


class MPPI(Controller):
    '''
    Model Predictive Path Integral Controller
    Original Paper: Williams et al., 2017, 'Information Theoretic MPC for Model-Based Reinforcement Learning'
    '''
    def __init__(self, T, K, U=None, noise_mu=0.0, noise_sigma=1.0, lamda=1.0):
        self.task = None

        self.T = T # number of timesteps
        self.K = K # number of samples
        self.U = U # initial control sequence

        # control hyper parameters
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.lamda = lamda

    def set_task(self, task):
        self.task = task
        if self.U is None:
            self.U = [self.task.action_space.sample()] * self.T
        self.U = np.float32(self.U)
        self.noise = self._sample_noise()
        self.task.reset()
        self.u_init = 0
        self.x_init = self.task.env.state

    def _sample_noise(self):
        action_dim = get_space_shape(self.task.action_space)
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, 
            size=(self.K, self.T, action_dim)).astype('f')

    def _compute_costs(self, dynamics, state_init, noise, new_dynamics_params=None):
        costs = [0] * self.K
        for k in range(self.K):
            self.task.env.state = self.x_init
            for t in range(self.T):
                action = self.U[t] + noise[k, t]
                state, reward, done, _ = self.task.step(*action)
                #if done:
                #    self.task_.reset()
                costs[k] += (-reward)
                #costs[k] += (-reward) + self.lamda * np.dot(self.U[t], noise[k, t, :]) / self.noise_sigma
        return costs

    def _compute_importance_weights(self, costs):
        beta = np.min(costs)
        weights = np.exp((-1 / self.lamda) * (costs - beta))
        weights /= np.sum(weights)
        return weights

    def plan(self, dynamics, state, new_dynamics_params=None):
        costs = self._compute_costs(dynamics, state, self.noise, new_dynamics_params)
        weights = self._compute_importance_weights(costs)
        self.U += [np.sum(weights * self.noise[:, t, 0]) for t in range(self.T)]
        action = self.U[0]
        
        self.task.env.state = self.x_init
        s, r, _, _ = self.task.step(action)
        print('cost', -r)
        self.task.render()

        self.U = np.roll(self.U, -1)
        self.U[-1] = self.u_init
        self.x_init = self.task.env.state

