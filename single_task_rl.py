from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, PushButton
import numpy as np
from custom_env import CustomEnv
import argparse

rlbench_env_config = {
    'task': "CloseMicrowave",  #
    'static_env': False,  #
    'headless_env': False,  #
    'save_demos': True,  #
    'learn_reward_frequency': 100,  #
    'episodes': 10,  #
    'sequence_len': 150,  #
    'obs_type': "LowDimension"  # LowDimension WristCameraRGB
}

class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        arm = -1 + 2 * np.random.random_sample(self.action_size - 1)
        gripper = [-0.9]  # Always open
        return np.concatenate([arm, gripper], axis=-1)

env = CustomEnv(rlbench_env_config)

agent = Agent(env.env.action_size)

training_steps = 200
episode_length = 50
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs, task_obs, state_obs = env.reset()
    action = agent.act(state_obs)
    obs, task_obs, state_obs, reward, done = env.step(action)

print('Done')
#env.shutdown()
