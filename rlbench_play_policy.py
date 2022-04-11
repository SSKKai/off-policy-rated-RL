import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from reward_net import RewardNetwork
import glfw
from custom_env import CustomEnv

import hydra


class Play_Policy(object):
    def __init__(self, config):

        # Configurations
        self.sac_hyparams = config.sac_hyperparams
        self.reward_hyparams = config.reward_hyperparams
        self.rlb_env_config = config.rlb_env_config

        # Experiment setup
        self.episode_len = config.experiment.episode_len
        self.max_episodes = config.experiment.max_episodes

        # Environment
        self.env = CustomEnv(self.rlb_env_config)
        self.env.reset()
        self.state_dim = self.rlb_env_config.state_dim
        action_size = self.env.env.action_size
        action_high = np.ones(action_size, dtype=np.float32)
        action_low = np.ones(action_size, dtype=np.float32) * (-1)
        action_low[-1] = 0.0
        self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})

        torch.manual_seed(self.sac_hyparams.seed)
        np.random.seed(self.sac_hyparams.seed)

        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=self.sac_hyparams)

        # Memory
        self.agent_memory = ReplayMemory(self.sac_hyparams.replay_size, self.sac_hyparams.seed, self.reward_hyparams.state_only)

        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], self.episode_len, args=self.reward_hyparams)


        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        self.rank_count = 0

        self.reward_list = []
        self.reward_prime_list = []
        self.e_reward_list = []
        self.e_reward_prime_list = []
        self.episode_len_list = []

    def evaluate(self, i_episode, episode_len):
        print("----------------------------------------")
        for episode in range(i_episode):
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state == None] = 0.0
            state = state.astype(np.float32)

            episode_reward = 0
            episode_reward_prime = 0
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.select_action(state, evaluate=True)

                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action)  # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)
                # reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if episode_steps % episode_len == 0:
                    done = True
                self.reward_network.push_data(state, action, reward, done)
            print("e steps: {}, Reward: {}".format(episode_steps, round(episode_reward, 2)))
            # if episode > 1 and episode % 20 == 0:
            #     self.reward_network.rank()
        print("----------------------------------------")



if __name__ == '__main__':
    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name="CloseMicrowave")

    policy = Play_Policy(config)
    policy.agent.load_model("models/sac_actor_CloseMicrowave_rl_solved", "models/sac_critic_CloseMicrowave_rl_solved")

    policy.evaluate(100, 250)
