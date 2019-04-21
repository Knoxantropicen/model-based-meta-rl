import numpy as np
import torch
import gym
import gym.spaces as gsp
from gym.envs.classic_control import CartPoleEnv, PendulumEnv
from gym.envs.mujoco import AntEnv

class Task(gym.Env):
    def get_cost(self, state, action, next_state):
        '''
        return vectorized cost [n] and done [n, 1] (torch.tensor, dtype=float)
        '''
        raise NotImplementedError

    def get_reset_state(self, n):
        '''
        return vectorized reset state [n, dim_state] (torch.tensor, dtype=float)
        '''
        raise NotImplementedError
    
    def reformat_action(self, action):
        if isinstance(self.action_space, gsp.Box):
            action = action.astype(self.action_space.dtype)
        elif isinstance(self.action_space, gsp.Discrete):
            action = np.clip(action.round(), 0, self.action_space.n - 1).astype(int)
        elif isinstance(self.action_space, gsp.MultiBinary):
            action = np.clip(action.round(), 0, 1).astype(self.action_space.dtype)
        elif isinstance(self.action_space, gsp.MultiDiscrete):
            action = np.clip(action.round(), 0, self.action_space.nvec - 1).astype(self.action_space.dtype)
        elif self.action_space is None:
            pass
        else:
            raise NotImplementedError
        return action
    
    def step(self, action, *args, **kwargs):
        action = self.reformat_action(action)
        next_state, reward, done, info = super().step(action, *args, **kwargs)
        return np.float32(next_state), reward, done, info

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        return np.float32(state)

# custom tasks
class CartPoleTask(Task, CartPoleEnv):
    def get_cost(self, state, action, next_state):
        x, theta = next_state[:, 0], next_state[:, 2]
        done = ((x < -self.x_threshold) \
                | (x > self.x_threshold) \
                | (theta < -self.theta_threshold_radians) \
                | (theta > self.theta_threshold_radians)).float()
        cost = done - 1.0
        done = done.unsqueeze(1)
        return cost, done

    def get_reset_state(self, n):
        return torch.FloatTensor(n, 4).uniform_(-0.05, 0.05)

    def step(self, action, *args, **kwargs):
        action = self.reformat_action(action)
        next_state, reward, done, info = super().step(action, *args, **kwargs)
        if done:
            reward = 0
        return np.float32(next_state), reward, done, info

class PendulumTask(Task, PendulumEnv):
    def get_cost(self, state, action, next_state):
        costh, sinth, thdot = next_state[:, 0], next_state[:, 1], next_state[:, 2]
        costh, sinth = torch.clamp(costh, -1, 1), torch.clamp(sinth, -1, 1)
        def get_from_cos_sin(cosx, sinx):
            xc, xs = torch.acos(cosx), torch.asin(sinx)
            return xc * 2 * (((xs > 0) | ((xs == 0) & (xc < 0.5 * np.pi))).float() - 0.5)
        th = get_from_cos_sin(costh, sinth)
        action = torch.clamp(action[:, 0], -self.max_torque, self.max_torque)
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)
        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * action ** 2
        done = torch.zeros_like(cost).unsqueeze(1)
        return cost, done

    def get_reset_state(self, n):
        theta = torch.FloatTensor(n, 1).uniform_(-np.pi, np.pi)
        thetadot = torch.FloatTensor(n, 1).uniform_(-1.0, 1.0)
        return torch.cat((torch.cos(theta), torch.sin(theta), thetadot), -1)

# mujoco
class AntTask(Task, AntEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def get_cost(self, state, action, next_state):
        xposbefore, xposafter = state[:, 0], next_state[:, 0]
        cfrc_ext = next_state[:, self.model.nq + self.model.nv:]
        forward_reward = ((xposafter - xposbefore) / self.dt)
        ctrl_cost = 0.5 * torch.pow(action, 2).sum(dim=1)
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        notdone = torch.isfinite(next_state).all(dim=1) & (next_state[:, 2] >= 0.2) & (next_state[:, 2] <= 1.0)
        cost, done = -reward, (~notdone).float().unsqueeze(1)
        return cost, done
        
    def get_reset_state(self, n):
        qpos = torch.tensor(self.init_qpos, dtype=torch.float) + torch.FloatTensor(n, self.model.nq).uniform_(-0.1, 0.1)
        qvel = torch.tensor(self.init_qvel, dtype=torch.float) + torch.randn(n, self.model.nv) * 0.1
        return torch.cat((qpos, qvel), -1)