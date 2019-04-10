import numpy as np
import torch
import torch.multiprocessing as mp
from tools.utils import get_space_shape


class Controller:
    '''
    Basic class of custom controllers, several methods required
    '''
    def set_task(self, task):
        raise NotImplementedError

    def plan(self, dynamics, state):
        raise NotImplementedError


def _compute_costs_per_thread(pid, queue, K, T, U, state_init, noise, dynamics, new_dynamics_params, task):
    costs = [0] * K
    for k in range(K):
        state = torch.tensor(state_init)
        for t in range(T):
            action = torch.tensor(U[t] + noise[k, t, :])
            delta_state = dynamics(torch.cat((state, action), 0), new_dynamics_params)
            state += delta_state
            costs[k] += task.env.get_cost(state, action)
    if queue is None:
        return [pid, costs]
    else:
        queue.put([pid, costs])


class MPPI(Controller):
    '''
    Model Predictive Path Integral Controller
    Original Paper: Williams et al., 2017, 'Information Theoretic MPC for Model-Based Reinforcement Learning'
    '''
    def __init__(self, T, K, U=None, noise_mu=0.0, noise_sigma=1.0, lamda=1.0, num_threads=1):
        self.task = None

        self.T = T # number of timesteps
        self.K = K # number of samples
        self.U = U # initial control sequence

        # control hyper parameters
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.lamda = lamda

        self.num_threads = num_threads

    def set_task(self, task):
        self.task = task
        if self.U is None:
            self.U = [self.task.action_space.sample()] * self.T
        self.U = np.float32(self.U)

    def _sample_noise(self):
        action_dim = get_space_shape(self.task.action_space)
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, 
            size=(self.K, self.T, action_dim)).astype('f')

    def _compute_costs_serial(self, dynamics, state_init, noise, new_dynamics_params=None):
        costs = [0] * self.K
        for k in range(self.K):
            state = torch.tensor(state_init)
            for t in range(self.T):
                action = torch.tensor(self.U[t] + noise[k, t, :])
                delta_state = dynamics(torch.cat((state, action), 0), new_dynamics_params)
                state += delta_state
                costs[k] += self.task.env.get_cost(state, action)
        return costs

    def _compute_costs_parallel(self, dynamics, state_init, noise, new_dynamics_params=None):
        workers = []
        queue = mp.Queue()
        # distribute K to all threads
        Ks = np.full(self.num_threads, self.K // self.num_threads)
        Ks[:self.K % self.num_threads] += 1
        for pid, K in zip(range(self.num_threads), Ks):
            if K > 0:
                if new_dynamics_params is not None:
                    for key, val in new_dynamics_params.items():
                        new_dynamics_params[key] = val.detach()
                worker_args = (pid, queue, K, self.T, self.U, state_init, noise, dynamics, new_dynamics_params, self.task)
                workers.append(mp.Process(target=_compute_costs_per_thread, args=worker_args))
        for worker in workers:
            worker.start()

        all_costs = [None] * len(workers)
        for _ in workers:
            pid, costs = queue.get()
            all_costs[pid] = costs
        all_costs = np.concatenate(all_costs)
        return all_costs

    def _compute_costs(self, *args, **kwargs):
        if self.num_threads > 1:
            return self._compute_costs_parallel(*args, **kwargs)
        else:
            return self._compute_costs_serial(*args, **kwargs)
        
    def _compute_importance_weights(self, costs):
        beta = np.min(costs)
        weights = np.exp((-1 / self.lamda) * (costs - beta))
        weights /= np.sum(weights)
        return weights

    def plan(self, dynamics, state, new_dynamics_params=None):
        noise = self._sample_noise()
        costs = self._compute_costs(dynamics, state, noise, new_dynamics_params)
        weights = self._compute_importance_weights(costs)
        action = self.U[0] + np.sum(weights * noise[:, 0, :].T)
        return action

