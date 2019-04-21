from gym.envs.registration import register

# register custom tasks
register(
    id='CartPoleTask-v0',
    entry_point='task.task:CartPoleTask',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='PendulumTask-v0',
    entry_point='task.task:PendulumTask',
    max_episode_steps=200,
)

register(
    id='AntTask-v0',
    entry_point='task.task:AntTask',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
