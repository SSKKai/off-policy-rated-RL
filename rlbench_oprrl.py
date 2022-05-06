import argparse
import time
import gym
from collections import deque
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
        self.seeds = config.experiment.seed
        self.change_flag_reward = config.experiment.change_flag_reward

        # Environment
        self.env_type = config.experiment.env_type

        if self.env_type == "rlbench":
            self.env = CustomEnv(self.env_config)
            self.env.reset()
            self.state_dim = self.env_config.state_dim
            action_size = self.env.env.action_size
            action_high = np.ones(action_size, dtype=np.float32)
            action_low = np.ones(action_size, dtype=np.float32)*(-1)
            action_low[-1] = 0.0
            self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})

        elif self.env_type == "mujoco":
            if self.env_config.terminate_when_unhealthy is None:
                self.env = gym.make(self.env_config.task)
            else:
                self.env = gym.make(self.env_config.task, terminate_when_unhealthy=self.env_config.terminate_when_unhealthy)
            self.env._max_episode_steps = self.episode_len
            self.env.seed(self.seeds)
            self.env.action_space.seed(self.seeds)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space

        else:
            raise Exception('wrong environment type, available: rlbench/mujoco')
        
        # torch.manual_seed(self.sac_hyparams.seed)
        # np.random.seed(self.sac_hyparams.seed)
        set_seeds(self.seeds)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=self.sac_hyparams)
        
        # Memory
        self.agent_memory = ReplayMemory(self.sac_hyparams.replay_size, self.seeds, self.reward_hyparams.state_only)
        
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

    def env_reset(self):
        if self.env_type == "rlbench":
            obs, task_obs, state_obs = self.env.reset()
            state = np.concatenate([state_obs, task_obs], axis=-1)
            state[state == None] = 0.0
            state = state.astype(np.float32)

        elif self.env_type == "mujoco":
            state = self.env.reset()

        return state

    def env_step(self, action):
        if self.env_type == "rlbench":
            next_obs, next_task_obs, next_state_obs, reward, done = self.env.step(action)  # Step
            next_state = np.concatenate([next_state_obs, next_task_obs], axis=-1)
            next_state[next_state == None] = 0.0
            next_state = next_state.astype(np.float32)

        elif self.env_type == "mujoco":
            next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done


        
    def evaluate(self, i_episode=20, episode_len=250, evaluate_mode=True):
        print("----------------------------------------")
        for _ in range(i_episode):
            state = self.env_reset()

            episode_reward = 0
            episode_reward_prime = 0
            done = False
            episode_steps = 0
            while not done:
                if self.env_type == "mujoco":
                    if self.env_config.render:
                        self.env.render()
                action = self.agent.select_action(state, evaluate=evaluate_mode)
                next_state, reward, done = self.env_step(action)
                # reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                episode_reward += reward
                state = next_state
                episode_steps += 1
                if episode_steps % episode_len == 0:
                    done = True
            print("Reward: {}".format(round(episode_reward, 2)))

        # elif self.env_type == "mujoco":
        #     for _ in range(i_episode):
        #         state = self.env.reset()
        #         episode_reward = 0
        #         episode_reward_prime = 0
        #         done = False
        #         while not done:
        #             if self.sac_hyparams.render:
        #                 self.env.render()
        #             action = self.agent.select_action(state, evaluate=evaluate_mode)
        #             next_state, reward, done, _ = self.env.step(action)
        #             reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
        #             episode_reward += reward
        #             episode_reward_prime += reward_prime
        #             state = next_state
        #         print("Reward: {}".format(round(episode_reward, 2)))


        print("----------------------------------------")

    def train_alt(self):

        frequency_flag = 1
        reach_count = 0
        succ_de = deque(maxlen=50)

        for i_episode in itertools.count(1):
            episode_reward = 0
            episode_reward_prime = 0
            episode_steps = 0
            done = False
            state = self.env_reset()
        
            while not done:
                if self.sac_hyparams.start_steps > self.total_numsteps:
                    if self.env_type == "rlbench":
                        action = self.env.randn_action()  # Sample random action
                    elif self.env_type == "mujoco":
                        action = self.env.action_space.sample()
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

                next_state, reward, done = self.env_step(action)
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



            if self.wandb_log:
                if self.env_type == "rlbench":
                    if episode_steps < self.episode_len:
                        task_succ = True
                    else:
                        task_succ = False
                    succ_de.append(task_succ)
                    succ_rate = np.mean(succ_de)

                    self.logger.log({"e_reward": episode_reward, "e_reward_prime": episode_reward_prime, "episode_steps": episode_steps, "success_rate": succ_rate, "i_episode": i_episode})
                elif self.env_type == "mujoco":
                    self.logger.log({"e_reward": episode_reward, "e_reward_prime": episode_reward_prime, "episode_steps": episode_steps, "i_episode": i_episode})

            self.e_reward_list.append(episode_reward)
            self.e_reward_prime_list.append(episode_reward_prime)
            self.episode_len_list.append(episode_steps)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}, reward_prime: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2),round(episode_reward_prime, 2)))
            
            
            if frequency_flag == 1:
                learn_frequency = 10
                self.reward_network.num_to_rank = 5
                if episode_reward > self.change_flag_reward: # -50 -100 -150
                    reach_count += 1
                if reach_count > 8:
                    frequency_flag = 2
            
            if frequency_flag == 2:
                learn_frequency = 100
                self.reward_network.num_to_rank = 20
                #############
                # self.reward_network.sample_method = 'distance sample'
            
            # learn reward
            if i_episode % learn_frequency == 0 and i_episode <= 5000:
                self.reward_network.rank()
                self.rank_count += 1
                print('rank successfully')
                
                acc_ls = []
                if self.rank_count >= 5:
                    ############################
                    if frequency_flag == 1:
                    #########################

                        for i in range(8):  # 5
                            # self.reward_network.learn_reward()
                            acc = self.reward_network.learn_reward_soft()
                            acc_ls.append(acc)

                    #####################################################
                    if frequency_flag == 2:
                        for i in range(self.rank_count):  # 5
                            # self.reward_network.learn_reward()
                            acc = self.reward_network.learn_reward_soft()
                            acc_ls.append(acc)
                            if acc > 0.965:
                                break
                    ###################################################
                else:
                    # self.reward_network.learn_reward()
                    acc = self.reward_network.learn_reward_soft()
                    acc_ls.append(acc)

                acc = np.mean(acc_ls)

                if self.wandb_log:
                    self.logger.log({'acc': acc, 'rank_count': self.rank_count})

                if self.training_flag == 2:
                    self.agent_memory.relabel_memory(self.reward_network)
            
            
            
            if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(self.sac_hyparams.eval_episodes, self.episode_len)
            
            if i_episode > self.max_episodes:
                break
        # self.env.close()


if __name__ == '__main__':

    with hydra.initialize(config_path="config"):
        config = hydra.compose(config_name="Walker")

    oprrl = OPRRL(config)

    oprrl.train_alt()


