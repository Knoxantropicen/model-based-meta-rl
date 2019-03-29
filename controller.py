import numpy as np
import torch

from utils import get_space_shape


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
            self.U = self.task.action_space.sample()

    def _sample_noise(self):
        action_dim = get_space_shape(self.task.action_space)
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, 
            size=(self.K, self.T, action_dim))

    def _compute_costs(self, dynamics, state_init, noise):
        costs = [0] * self.K
        for k in range(self.K):
            state = torch.tensor(state_init)
            for t in range(self.T):
                action = torch.tensor(self.U[t] + noise[k, t, :])
                state = dynamics(torch.cat((state, action), 0))
                costs[k] += self.task.get_cost(state) + self.lamda * np.dot(self.U[t], noise[k, t, :]) / self.noise_sigma
        return costs

    def _compute_importance_weights(self, costs):
        beta = np.min(costs)
        weights = np.exp((-1 / self.lamda) * (costs - beta))
        weights /= np.sum(weights)
        return weights

    def plan(self, dynamics, state):
        noise = self._sample_noise()
        costs = self._compute_costs(dynamics, state, noise)
        weights = self._compute_importance_weights(costs)
        action = self.U[0] + np.sum(weights * noise[:, 0, :])
        return action

