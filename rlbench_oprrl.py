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

class OPRRL(object):
    def __init__(self, sac_hyperparams, reward_hyperparams, rlb_env_config):
        self.sac_hyparams = sac_hyperparams
        self.reward_hyperparams = reward_hyperparams
        self.rlb_env_config = rlb_env_config
        
        # Environment
        self.env = CustomEnv(rlb_env_config)
        self.env.reset()
        self.state_dim = 8+3 #8+43
        action_size = self.env.env.action_size
        action_high = np.ones(action_size, dtype=np.float32)
        action_low = np.ones(action_size, dtype=np.float32)*(-1)
        action_low[-1] = 0.0
        self.action_dim = argparse.Namespace(**{'high': action_high, 'low': action_low, 'shape': (action_size,)})
        
        torch.manual_seed(sac_hyperparams.seed)
        np.random.seed(sac_hyperparams.seed)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=sac_hyperparams)
        
        #Tesnorboard
        # self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), sac_hyperparams.env_name,
        #                                                             sac_hyperparams.policy, "autotune" if sac_hyperparams.automatic_entropy_tuning else ""))
        
        # Memory
        self.agent_memory = ReplayMemory(sac_hyperparams.replay_size, sac_hyperparams.seed, reward_hyperparams.state_only)
        
        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], 250, args=reward_hyperparams)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        self.rank_count = 0
        
        self.reward_list = []
        self.reward_prime_list = []
        self.e_reward_list = []
        self.e_reward_prime_list = []
        
        self.training_flag = 2
        self.s1_episode = 100
        
        
    def evaluate(self, i_episode):
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
                if episode_steps % 250 == 0:
                    done = True
            print("Reward: {}".format(round(episode_reward, 2)))
        print("----------------------------------------")
        
        
    def train(self):
        #self.reward_network.load_reward_model(reward_path='reward_models/reward_model_Ant_v3_5313')
        #learn_frequency = self.reward_hyparams.learn_reward_frequency
        frequency_flag = 1

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
                mask = 1 if episode_steps == 250 else float(not done)
                
                
                # if self.training_flag == 1:
                #     self.agent_memory.push(state, action, reward, next_state, mask)
                if self.training_flag == 2:
                    self.agent_memory.push(state, action, reward_prime, next_state, mask)
                
                self.reward_network.push_data(state, action, reward, done) # push data to reward memory
                
                #self.reward_list.append(reward)
                #self.reward_prime_list.append(reward_prime)
                state = next_state

                if episode_steps % 250 == 0:
                    done = True
            
            # self.writer.add_scalar('e_reward/episode_reward_true', episode_reward, i_episode)
            # self.writer.add_scalar('e_reward_prime/episode_reward_prime', episode_reward_prime, i_episode)
            self.e_reward_list.append(episode_reward)
            self.e_reward_prime_list.append(episode_reward_prime)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}, reward_prime: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2),round(episode_reward_prime, 2)))
            
            # swith training flag
            if self.training_flag == 1:
                if episode_reward > 100:
                    self.training_flag = 2
                    #self.agent_memory.relabel_memory(self.reward_network)
                    print('swith to section 2')
                    self.agent.save_model(env_name='Ant_v3', suffix=str(int(episode_reward)))
                    self.reward_network.save_reward_model(env_name="Ant_v3", version=str(int(episode_reward)))
            
            if frequency_flag == 1:
                learn_frequency = 10
                self.reward_network.num_to_rank = 10
                # if self.training_flag == 2:
                #     frequency_flag = 3
                if episode_reward > 300:
                    frequency_flag = 3
            
            # if frequency_flag == 2:
            #     learn_frequency = 30
            #     self.reward_network.num_to_rank = 10
            #     if self.training_flag == 2:
            #         frequency_flag = 3
            
            if frequency_flag == 3:
                learn_frequency = 100
                self.reward_network.num_to_rank = 20
            
            self.frequency_flag = frequency_flag
            
            # learn reward
            if i_episode % learn_frequency == 0:
                self.reward_network.rank()
                self.rank_count += 1
                print('rank successfully')
                
                if self.rank_count >= 5:
                    for i in range(5):
                        self.reward_network.learn_reward()
                else:
                    self.reward_network.learn_reward()
                
                if self.training_flag == 2:
                    self.agent_memory.relabel_memory(self.reward_network)
            
            
            
            if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(i_episode)
            
            if i_episode > self.sac_hyparams.max_episodes:
                break
        self.env.close()



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
                mask = 1 if episode_steps == 250 else float(not done)

                if episode_steps % 250 == 0:
                    done = True
                
                
                self.agent_memory.push(state, action, reward_prime, next_state, mask)
                
                self.reward_network.push_data(state, action, reward, done) # push data to reward memory

                state = next_state


            
            # self.writer.add_scalar('e_reward/episode_reward_true', episode_reward, i_episode)
            # self.writer.add_scalar('e_reward_prime/episode_reward_prime', episode_reward_prime, i_episode)
            self.e_reward_list.append(episode_reward)
            self.e_reward_prime_list.append(episode_reward_prime)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}, reward_prime: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2),round(episode_reward_prime, 2)))
            
            
            if frequency_flag == 1:
                learn_frequency = 10
                self.reward_network.num_to_rank = 10
                if episode_reward > -50:
                    reach_count += 1
                if reach_count > 8:
                    frequency_flag = 2
            
            if frequency_flag == 2:
                learn_frequency = 100
                self.reward_network.num_to_rank = 20
            
            # learn reward
            if i_episode % learn_frequency == 0:
                self.reward_network.rank()
                self.rank_count += 1
                print('rank successfully')
                
                if self.rank_count >= 5:
                    for i in range(8): #5
                        self.reward_network.learn_reward()
                else:
                    self.reward_network.learn_reward()
                
                if self.training_flag == 2:
                    self.agent_memory.relabel_memory(self.reward_network)
            
            
            
            if i_episode % self.sac_hyparams.eval_per_episode == 0 and self.sac_hyparams.eval is True:
                self.evaluate(2)
            
            if i_episode > self.sac_hyparams.max_episodes:
                break
        self.env.close()


if __name__ == '__main__':
    sac_hyperparams = {
        'env_name': "Ant-v3",  #Mujoco Gym environment  HalfCheetah-v2  BipedalWalkerHardcore-v3 Ant-v3 Walker2d-v3
        'policy': "Gaussian",  #Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval': True,  #Evaluates a policy a policy every 10 episode (default: True)
        'eval_per_episode': 30,  #evaluate policy per episode
        'render': True,  #Render when evaluate policy
        'test_episodes': 1,
        'gamma': 0.99,  
        'tau': 0.005,  #target smoothing coefficient(τ) (default: 0.005)
        'lr': 0.0003,  
        'alpha': 0.2,  #Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
        'automatic_entropy_tuning': True,  #Automaically adjust α (default: False)
        'seed': 123456,  #random seed (default: 123456)
        'batch_size': 256,
        'max_steps': 50000,  #maximum number of steps (default: 1000000)
        'max_episodes': 6000,  #maximum number of episodes (default: 3000)
        'hidden_size': 256,
        'updates_per_step': 1,  #model updates per simulator step (default: 1)
        'start_steps': 10000,  #Steps sampling random actions (default: 10000)
        'target_update_interval': 1,  #Value target update per no. of updates per step (default: 1)
        'replay_size': 1000000,  #size of replay buffer (default: 10000000)
        'cuda': True
        }
    
    reward_hyperparams = {
        'sample_method': "random sample",  #Sample method for sampling a batch of trajectories to get ranked
        'rank_by_true_reward': True,  #rank the trajectory by true reward or by human
        'state_only': False,  #the reward net is r(s,a) or r(s)
        'hidden_dim': 256,  #hidden dim for reward network
        'learn_reward_frequency': 100,   #learn reward per N episodes
        'num_to_rank': 10,  #num to rank per reward update
        'traj_capacity': 200,  #trajectory capacity of reward buffer
        'lr': 0.001  
        }

    rlb_env_config = {
        'task': "ReachTarget",  #
        'static_env': False,  #
        'headless_env': False,  #
        'save_demos': True,  #
        'learn_reward_frequency': 100,  #
        'episodes': 10,  #
        'sequence_len': 150,  #
        'obs_type': "LowDimension"  # LowDimension WristCameraRGB
    }
    
    sac_hyperparams = argparse.Namespace(**sac_hyperparams)
    reward_hyperparams = argparse.Namespace(**reward_hyperparams)
    
    
    oprrl = OPRRL(sac_hyperparams, reward_hyperparams, rlb_env_config)
    oprrl.train_alt()
