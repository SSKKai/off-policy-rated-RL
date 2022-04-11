import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from utils import get_wandb_config, set_seeds
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from reward_net import RewardNetwork
import glfw
from custom_env import CustomEnv

import hydra
import wandb

class OPRRL(object):
    def __init__(self, config):

        # Configurations
        self.sac_hyparams = config.sac
        self.reward_hyparams = config.reward
        self.env_config = config.env

        # Experiment setup
        self.episode_len = config.experiment.episode_len
        self.max_episodes = config.experiment.max_episodes
        
        # Environment
        self.env = CustomEnv(self.env_config)
        self.env.reset()
        self.state_dim = self.env_config.state_dim
        action_size = self.env.env.action_size
        action_high = np.ones(action_size, dtype=np.float32)
        action_low = np.ones(action_size, dtype=np.float32)*(-1)
        action_low[-1] = 0.0
        self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})
        
        # torch.manual_seed(self.sac_hyparams.seed)
        # np.random.seed(self.sac_hyparams.seed)
        set_seeds(self.sac_hyparams.seed)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=self.sac_hyparams)
        
        # Memory
        self.agent_memory = ReplayMemory(self.sac_hyparams.replay_size, self.sac_hyparams.seed, self.reward_hyparams.state_only)
        
        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], self.episode_len, args=self.reward_hyparams)

        # wandb logger
        self.wandb_log = config.experiment.wandb_log
        if self.wandb_log:
            config_wandb = get_wandb_config(config)
            self.logger = wandb.init(config = config, project='oprrl_'+config.experiment.env_type+'_'+config.env.task)
            self.logger.config.update(config_wandb)
            # self.logger.watch(self.agent.policy)
            # self.logger.watch(self.reward_network.reward_network)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        self.rank_count = 0
        
        self.reward_list = []
        self.reward_prime_list = []
        self.e_reward_list = []
        self.e_reward_prime_list = []
        self.episode_len_list = []
        
        self.training_flag = 2
        self.s1_episode = 100
        
        
    def evaluate(self, i_episode, episode_len):
        print("----------------------------------------")
        for _ in range(i_episode):
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

    def train_alt(self):

        frequency_flag = 1
        reach_count = 0

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_reward_prime = 0
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
                        if self.wandb_log:
                            self.logger.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss})

                next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action) # Step
                next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
                next_state[next_state == None] = 0.0
                next_state = next_state.astype(np.float32)
                reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                
                episode_steps += 1
                self.total_numsteps += 1
                episode_reward += reward
                episode_reward_prime += reward_prime
        
                # Ignore the "done" signal if it comes from hitting the time horizon.
                # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
                mask = 1 if episode_steps == self.episode_len else float(not done)

                if episode_steps % self.episode_len == 0:
                    done = True
                
                
                self.agent_memory.push(state, action, reward_prime, next_state, mask)
                
                self.reward_network.push_data(state, action, reward, done) # push data to reward memory

                state = next_state


            
            # self.writer.add_scalar('e_reward/episode_reward_true', episode_reward, i_episode)
            # self.writer.add_scalar('e_reward_prime/episode_reward_prime', episode_reward_prime, i_episode)
            if self.wandb_log:
                self.logger.log({"e_reward": episode_reward, "e_reward_prime": episode_reward_prime, "episode_steps": episode_steps, "i_episode": i_episode})

            self.e_reward_list.append(episode_reward)
            self.e_reward_prime_list.append(episode_reward_prime)
            self.episode_len_list.append(episode_steps)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}, reward_prime: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2),round(episode_reward_prime, 2)))
            
            
            if frequency_flag == 1:
                learn_frequency = 10
                self.reward_network.num_to_rank = 10
                if episode_reward > -100: # -50
                    reach_count += 1
                if reach_count > 8:
                    frequency_flag = 2
            
            if frequency_flag == 2:
                learn_frequency = 100
                self.reward_network.num_to_rank = 20
            
            # learn reward
            if i_episode % learn_frequency == 0 and i_episode <= 5000:
                self.reward_network.rank()
                self.rank_count += 1
                print('rank successfully')
                
                if self.rank_count >= 5:
                    for i in range(8):  # 5
                        # self.reward_network.learn_reward()
                        loss, acc = self.reward_network.learn_reward_soft()
                else:
                    # self.reward_network.learn_reward()
                    loss, acc = self.reward_network.learn_reward_soft()

                if self.wandb_log:
                    self.logger.log({'reward_loss': loss, 'acc': acc, 'rank_count': self.rank_count})

                if self.training_flag == 2:
                    self.agent_memory.relabel_memory(self.reward_network)
            
            
            
            if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(self.sac_hyparams.eval_episodes, self.episode_len)
            
            if i_episode > self.max_episodes:
                break
        # self.env.close()


if __name__ == '__main__':

    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name="PushButton")

    oprrl = OPRRL(config)
    # oprrl.train_alt()
