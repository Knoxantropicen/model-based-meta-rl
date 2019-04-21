import numpy as np
import gym
import gym.spaces as gsp
from gym.envs.classic_control import CartPoleEnv, PendulumEnv
from gym.envs.mujoco import AntEnv

class Task(gym.Env):
    def get_cost(self, state, action):
        '''
        return cost and done
        '''
        raise NotImplementedError

    def get_reset_state(self):
        '''
        return reset state
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
    def get_cost(self, state, action):
        x, _, theta, _ = state
        done = bool(x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians)
        cost = 0.0 if done else -1.0
        return cost, done

    def get_reset_state(self):
        return self.np_random.uniform(low=-0.05, high=0.05, size=(4,))

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
        done = False
        return cost, done

    def get_reset_state(self):
        high = np.array([np.pi, 1])
        state = self.np_random.uniform(low=-high, high=high)
        theta, thetadot = state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

# mujoco
class AntTask(Task, AntEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def get_cost(self, state, action):
        qpos, qvel = state[:self.model.nq], state[self.model.nq:self.model.nq + self.model.nv]
        self.set_state(qpos, qvel)
        xposbefore = self.get_body_com('torso')[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com('torso')[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        new_state = self.state_vector()
        notdone = np.isfinite(new_state).all() and new_state[2] >= 0.2 and new_state[2] <= 1.0
        cost, done = -reward, not notdone
        return cost, done
        
    def get_reset_state(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()