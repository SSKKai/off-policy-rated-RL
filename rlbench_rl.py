import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
from reward_net import RewardNetwork
import glfw
import hydra

from custom_env import CustomEnv


class OPRRL(object):
    def __init__(self, config):
        self.sac_hyparams = config.sac
        self.reward_hyparams = config.reward
        self.env_config = config.env

        # Experiment setup
        self.episode_len = config.experiment.episode_len
        self.max_episodes = config.experiment.max_episodes
        
        # Environment
        self.env = CustomEnv(self.env_config)
        self.env.reset()
        self.state_dim = self.env_config.state_dim #ReachTarget:8+3, PushButton:8+3, CloseMicrowave:8+11
        action_size = self.env.env.action_size
        action_high = np.ones(action_size, dtype=np.float32)
        action_low = np.ones(action_size, dtype=np.float32)*(-1)
        action_low[-1] = 0.0
        self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})
        
        torch.manual_seed(self.sac_hyparams.seed)
        np.random.seed(self.sac_hyparams.seed)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=self.sac_hyparams)
        
        # Memory
        self.agent_memory = ReplayMemory(self.sac_hyparams.replay_size, self.sac_hyparams.seed)
        
        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], self.episode_len, args=self.reward_hyparams)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        
        self.rank_count = 0

        self.start_steps_1 = 150000
        
        
    def evaluate(self, i_episode, episode_len):
        print("----------------------------------------")
        for _  in range(i_episode):
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state==None] = 0.0
            state = state.astype(np.float32)

            episode_reward = 0
            episode_reward_prime = 0
            done = False
            episode_steps = 0
            while not done:
                action = self.agent.select_action(state, evaluate=True)

                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action) # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)
                # reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if episode_steps % episode_len == 0:
                    done = True
            print("Reward: {}".format(round(episode_reward, 2)))
        print("----------------------------------------")

    def train(self):

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_steps = 0
            done = False
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state==None] = 0.0
            state = state.astype(np.float32)
        
            while not done:
                if self.sac_hyparams.start_steps > self.total_numsteps:
                    action = self.env.randn_action()  # Sample random action
                else:
                    action = self.agent.select_action(state)  # Sample action from policy
        
                if len(self.agent_memory) > self.sac_hyparams.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.sac_hyparams.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                        self.updates += 1
        
                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action) # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)

                episode_steps += 1
                self.total_numsteps += 1
                episode_reward += reward

                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.episode_len else float(not done)

                self.agent_memory.push(state, action, reward, next_state, mask) # push data to agent memory

                if episode_steps % self.episode_len == 0:
                    done = True
        
                state = next_state
        
            if i_episode > self.max_episodes:
                break

            print("E{}, t numsteps: {}, e steps: {}, reward: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2)))
        
            # if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
            #     self.evaluate(i_episode)
                
        # self.env.close()





if __name__ == '__main__':

    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name="CloseMicrowave")

    oprrl = OPRRL(config)
    
    
    oprrl.train()
           