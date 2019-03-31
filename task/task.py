import numpy as np
import gym
import gym.spaces as gsp
from gym.envs.classic_control import CartPoleEnv

class Task(gym.Env):
    def get_cost(self, state):
        raise NotImplementedError
    
    def reformat_action(self, action):
        if isinstance(self.action_space, gsp.Box):
            action = action.astype(self.dtype)
        elif isinstance(self.action_space, gsp.Discrete):
            action = action.round().astype(int)
        elif isinstance(self.action_space, gsp.MultiBinary):
            action = np.clip(action.round(), 0, 1).astype(self.dtype)
        elif isinstance(self.action_space, gsp.MultiDiscrete):
            action = np.clip(action.round(), 0, self.nvec).astype(self.dtype)
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
    def get_cost(self, state):
        x, _, theta, _ = state
        done = x < -self.x_threshold \
                or x < self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        cost = 1 if done else 0
        return cost
