import gym

class Task(gym.Env):
    def get_cost(self, state):
        raise NotImplementedError

# TODO: define custom tasks
