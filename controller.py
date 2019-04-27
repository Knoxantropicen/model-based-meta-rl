import numpy as np
import torch
import torch.multiprocessing as mp
from tools.utils import get_space_shape, cuda
from copy import deepcopy

class Controller:
    '''
    Basic class of custom controllers, several methods required
    '''
    def set_task(self, task):
        raise NotImplementedError

    def plan(self, dynamics, state):
        raise NotImplementedError


def _compute_costs_per_thread(pid, queue, K, T, U, state_init, noise, dynamics, new_dynamics_params, task):
    costs = torch.zeros(K)
    state = torch.stack(([torch.tensor(state_init, dtype=torch.float32)] * K))
    for t in range(T):
        action = torch.stack(([torch.tensor(U[t] + noise[k, t, :]) for k in range(K)]))
        gpu_id = next(dynamics.parameters()).device
        delta_state = dynamics(cuda(torch.cat((state, action), -1), gpu_id), new_dynamics_params).detach()
        next_state = state + delta_state.cpu()
        cost, done = task.env.get_cost(state, action, next_state)
        state = next_state
        costs += cost
        state = done * task.env.get_reset_state(K) + (1.0 - done) * state
    costs = costs.numpy()
    if queue is None:
        return [pid, costs]
    else:
        queue.put([pid, costs])

def _compute_real_costs_per_thread(pid, queue, K, T, U, state_init, noise, task):
    costs = np.zeros(K)
    for k in range(K):
        task.reset()
        task.env.set_new_state(state_init)
        for t in range(T):
            action = U[t] + noise[k, t, :]
            _, reward, done, _ = task.step(action)
            costs[k] -= reward
            if done:
                task.reset()
    if queue is None:
        return [pid, costs]
    else:
        queue.put([pid, costs])


class MPPI(Controller):
    '''
    Model Predictive Path Integral Controller
    Original Paper: Williams et al., 2017, 'Information Theoretic MPC for Model-Based Reinforcement Learning'
    '''
    def __init__(self, T, K, lamda=1.0, num_threads=1):
        self.task = None

        self.T = int(T) # number of timesteps
        self.K = int(K) # number of samples
        self.U = None # control sequence
        self.u_init = None

        # control hyper parameters
        self.noise_mu = None
        self.noise_sigma = None
        self.lamda = lamda

        self.num_threads = num_threads

    def set_task(self, task):
        self.task = deepcopy(task)
        self.noise_mu, self.noise_sigma, self.u_init = self.task.env.get_control_params()
        self.U = np.tile(self.u_init, (self.T, 1))
        if self.u_init is None:
            self.u_init = self.task.action_space.sample()
            self.U = np.zeros((self.T, get_space_shape(self.task.action_space)))
            for t in range(self.T):
                self.U[t] = self.task.action_space.sample()
        self.U = np.float32(self.U)
        self.noise = self._sample_noise()

    def _sample_noise(self):
        action_dim = get_space_shape(self.task.action_space)
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, 
            size=(self.K, self.T, action_dim)).astype('f')

    def _compute_costs_serial(self, dynamics, state_init, new_dynamics_params=None):
        costs = torch.zeros(self.K)
        state = torch.stack(([torch.tensor(state_init, dtype=torch.float32)] * self.K))
        for t in range(self.T):
            action = torch.stack(([torch.tensor(self.U[t] + self.noise[k, t, :]) for k in range(self.K)]))
            gpu_id = next(dynamics.parameters()).device
            delta_state = dynamics(cuda(torch.cat((state, action), -1), gpu_id), new_dynamics_params).detach()
            next_state = state + delta_state.cpu()
            cost, done = self.task.env.get_cost(state, action, next_state)
            state = next_state
            costs += cost
            state = done * self.task.env.get_reset_state(self.K) + (1.0 - done) * state
        costs = costs.numpy()
        return costs

    def _compute_costs_parallel(self, dynamics, state_init, new_dynamics_params=None):
        workers = []
        queue = mp.Queue()
        # distribute K to all threads
        Ks = np.full(self.num_threads, self.K // self.num_threads, dtype=np.int)
        Ks[:self.K % self.num_threads] += 1
        for pid, K in zip(range(self.num_threads), Ks):
            if K > 0:
                if new_dynamics_params is not None:
                    for key, val in new_dynamics_params.items():
                        new_dynamics_params[key] = val.detach()
                worker_args = (pid, queue, K, self.T, self.U, state_init, self.noise, dynamics, new_dynamics_params, self.task)
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

    def _compute_real_costs_serial(self, state_init):
        costs = np.zeros(self.K)
        for k in range(self.K):
            self.task.reset()
            self.task.env.set_new_state(state_init)
            for t in range(self.T):
                action = self.U[t] + self.noise[k, t, :]
                _, reward, done, _ = self.task.step(action)
                costs[k] -= reward
                if done:
                    self.task.reset()
        self.task.env.set_new_state(state_init)
        return costs

    def _compute_real_costs_parallel(self, state_init):
        workers = []
        queue = mp.Queue()
        # distribute K to all threads
        Ks = np.full(self.num_threads, self.K // self.num_threads, dtype=np.int)
        Ks[:self.K % self.num_threads] += 1
        for pid, K in zip(range(self.num_threads), Ks):
            if K > 0:
                worker_args = (pid, queue, K, self.T, self.U, state_init, self.noise, self.task)
                workers.append(mp.Process(target=_compute_real_costs_per_thread, args=worker_args))
        for worker in workers:
            worker.start()

        all_costs = [None] * len(workers)
        for _ in workers:
            pid, costs = queue.get()
            all_costs[pid] = costs
        all_costs = np.concatenate(all_costs)
        return all_costs

    def _compute_real_costs(self, *args, **kwargs):
        if self.num_threads > 1:
            return self._compute_real_costs_parallel(*args, **kwargs)
        else:
            return self._compute_real_costs_serial(*args, **kwargs)
        
    def _compute_importance_weights(self, costs):
        beta = np.min(costs)
        weights = np.exp((-1 / self.lamda) * (costs - beta))
        weights /= np.sum(weights)
        return weights

    def plan(self, dynamics, state, new_dynamics_params=None, debug=False):
        if debug:
            costs = self._compute_real_costs(state)
        else:
            costs = self._compute_costs(dynamics, state, new_dynamics_params)
        weights = self._compute_importance_weights(costs)
        self.U += np.sum(weights * self.noise.T, axis=-1).T
        action = self.U[0]
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.u_init if self.u_init is not None else self.task.action_space.sample()
        return action

class MPC(MPPI):
    def plan(self, dynamics, state, new_dynamics_params=None, debug=False):
        if debug:
            costs = self._compute_real_costs(state)
        else:
            costs = self._compute_costs(dynamics, state, new_dynamics_params)
        best_k = np.argmin(costs)
        action = self.U[0] + self.noise[best_k, 0, :]
        return action
