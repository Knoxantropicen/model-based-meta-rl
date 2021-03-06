from gym.envs.registration import register

# register custom tasks
register(
    id='CartPoleTask-v0',
    entry_point='task.task:CartPoleTask',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='AntTask-v0',
    entry_point='task.task:AntTask',
    max_episode_steps=100,
    reward_threshold=6000.0,
)

register(
    id='HalfCheetahTask-v0',
    entry_point='task.task:HalfCheetahTask',
    max_episode_steps=100,
    reward_threshold=4800.0,
)

register(
    id='SwimmerTask-v0',
    entry_point='task.task:SwimmerTask',
    max_episode_steps=100,
    reward_threshold=360.0,
)