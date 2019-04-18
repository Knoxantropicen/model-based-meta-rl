import numpy as np
import gym
import gym.spaces as gsp
from gym.envs.classic_control import CartPoleEnv, PendulumEnv

class Task(gym.Env):
    def get_cost(self, state, action):
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
    def get_cost(self, state, action):
        x, _, theta, _ = state
        done = x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        cost = 0 if done else -1
        return cost

    def step(self, action, *args, **kwargs):
        action = self.reformat_action(action)
        next_state, reward, done, info = super().step(action, *args, **kwargs)
        if done:
            reward = 0
        return np.float32(next_state), reward, done, info

class PendulumTask(Task, PendulumEnv):
    def get_cost(self, state, action):
        costh, sinth, thdot = state
        costh, sinth = np.clip(costh, -1, 1), np.clip(sinth, -1, 1)
        def get_from_cos_sin(cosx, sinx):
            xc, xs = np.arccos(cosx), np.arcsin(sinx)
            return xc if xs > 0 or (xs == 0 and xc < 0.5 * np.pi) else -xc
        th = get_from_cos_sin(costh, sinth)
        action = np.clip(action, -self.max_torque, self.max_torque)[0]
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)
        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * action ** 2
        return cost
