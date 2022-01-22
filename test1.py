import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from reward_net import RewardNetwork
import glfw

class OPRRL(object):
    def __init__(self, sac_hyperparams, reward_hyperparams):
        self.sac_hyparams = sac_hyperparams
        self.reward_hyparams = reward_hyperparams
        
        # Environment
        # env = NormalizedActions(gym.make(hyperparameters.env_name))
        self.env = gym.make(sac_hyperparams.env_name, terminate_when_unhealthy = False)
        #self.env._max_episode_steps = 300
        self.env.seed(sac_hyperparams.seed)
        self.env.action_space.seed(sac_hyperparams.seed)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space
        
        torch.manual_seed(sac_hyperparams.seed)
        np.random.seed(sac_hyperparams.seed)
        
        # Agent
        self.agent = SAC(self.state_dim, self.action_dim, args=sac_hyperparams)
        
        #Tesnorboard
        self.writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), sac_hyperparams.env_name,
                                                                    sac_hyperparams.policy, "autotune" if sac_hyperparams.automatic_entropy_tuning else ""))
        
        # Memory
        self.agent_memory = ReplayMemory(sac_hyperparams.replay_size, sac_hyperparams.seed)
        
        # Reward Net
        self.reward_network = RewardNetwork(self.state_dim, self.action_dim.shape[0], self.env._max_episode_steps, args = reward_hyperparams)
        
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        
        self.rank_count = 0
        
        self.training_flag = 0
        self.s1_episode = 150
        
    def evaluate(self, i_episode):
        print("----------------------------------------")
        for _  in range(self.sac_hyparams.test_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_reward_prime = 0
            done = False
            while not done:
                if self.sac_hyparams.render:
                    self.env.render()
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.env.step(action)
                reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                episode_reward += reward
                episode_reward_prime += reward_prime
                state = next_state
            print("Reward: {}, reward_prime: {}".format(round(episode_reward, 2),round(episode_reward_prime, 2)))
        print("----------------------------------------")
        #glfw.terminate()
        
        
    def train(self):
        
        print('section 1 start')
        learn_frequency = self.reward_hyparams.learn_frequency_1
        
        for i_episode in itertools.count(1):
            
            # initial episode
            episode_reward = 0
            episode_reward_prime = 0
            episode_steps = 0
            done = False
            state = self.env.reset()
            
            #########################################################################################
            ##### episode loop #####
            #########################################################################################
            while not done:
            
                if self.training_flag == 0:
                
                    
                    if i_episode < self.s1_episode/2:
                        action = self.agent.select_action(state)  # Sample random action
                    else:
                        action = self.env.action_space.sample()  # Sample action from policy
            
                    next_state, reward, done, _ = self.env.step(action) # Step
                    reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                    episode_reward += reward
                    episode_reward_prime += reward_prime
                    episode_steps += 1
                    
                    self.reward_network.push_data(state, action, reward, done) # push data to reward memory
                    state = next_state
                        
                #######################################################################################
                
                elif self.training_flag == 1:
                
                    if self.total_numsteps < self.sac_hyparams.start_steps:
                        action = self.env.action_space.sample()  # Sample random action
                    else:
                        action = self.agent.select_action(state)  # Sample action from policy
            
                    if len(self.agent_memory) > self.sac_hyparams.batch_size:
                        # Number of updates per step in environment
                        for i in range(self.sac_hyparams.updates_per_step):
                            # Update parameters of all the networks
                            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                            self.updates += 1
        
                    next_state, reward, done, _ = self.env.step(action) # Step
                    reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                    
                    episode_steps += 1
                    self.total_numsteps += 1
                    episode_reward += reward
                    episode_reward_prime += reward_prime
                    
                    mask = 1 if episode_steps == self.env._max_episode_steps else float(not done)
    
                    self.agent_memory.push(state, action, reward, next_state, mask) # push data to agent memory
                    self.reward_network.push_data(state, action, reward, done) # push data to reward memory
                    state = next_state
                        
                #######################################################################################
                
                elif self.training_flag == 2:
                    
                    action = self.agent.select_action(state)  # Sample action from policy
            
                    # Number of updates per step in environment
                    for i in range(self.sac_hyparams.updates_per_step):
                        # Update parameters of all the networks
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
                        self.updates += 1
        
                    next_state, reward, done, _ = self.env.step(action) # Step
                    reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                    
                    episode_steps += 1
                    self.total_numsteps += 1
                    episode_reward += reward
                    episode_reward_prime += reward_prime
                    
                    sac_hyperparams
    
                    self.agent_memory.push(state, action, reward_prime, next_state, mask) # push data to agent memory
                    self.reward_network.push_data(state, action, reward, done) # push data to reward memory
                    state = next_state
                    
            #######################################################################################
            
            # print and log
            self.writer.add_scalar('reward/train', episode_reward, i_episode)
            print("E{}, t numsteps: {}, e steps: {}, reward: {}, reward_prime: {}".format(i_episode, self.total_numsteps, episode_steps,
                                                                                                            round(episode_reward, 2),round(episode_reward_prime, 2)))
            
            # swith training flag
            if self.training_flag == 0:
                if i_episode == self.s1_episode:
                    self.training_flag = 1
                    print('swith to section 2')
                    
            elif self.training_flag == 1:
                if episode_reward > 1000:
                    self.training_flag = 2
                    self.agent_memory.relabel_memory(self.reward_network)
                    print('swith to section 3')
                    self.agent.save_model(env_name='Ant_v3', suffix=str(int(episode_reward)))
            
            elif self.training_flag == 2:
                if episode_reward > 1500:
                    learn_frequency = self.reward_hyparams.learn_frequency_2
                    self.reward_network.num_to_rank = self.reward_hyparams.num_to_rank_2
                
            
            
            # learn reward
            if i_episode % learn_frequency == 0:
                self.reward_network.rank()
                self.rank_count += 1
                print('rank successfully')
                
                if self.rank_count >= 3:
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





if __name__ == '__main__':
    sac_hyperparams = {
        'env_name': "Ant-v3",  #Mujoco Gym environment  HalfCheetah-v2  BipedalWalkerHardcore-v3
        'policy': "Gaussian",  #Policy Type: Gaussian | Deterministic (default: Gaussian)
        'eval': True,  #Evaluates a policy a policy every 10 episode (default: True)
        'eval_per_episode': 50,  #evaluate policy per episode
        'render': False,  #Render when evaluate policy
        'test_episodes': 3,
        'gamma': 0.99,  
        'tau': 0.005,  #target smoothing coefficient(τ) (default: 0.005)
        'lr': 0.0003,#default=0.0003 / 0.00005
        'alpha': 0.2,  #Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
        'automatic_entropy_tuning': True,  #Automaically adjust α (default: False)
        'seed': 123456,  #random seed (default: 123456)
        'batch_size': 256,
        'max_steps': 50000,  #maximum number of steps (default: 1000000)
        'max_episodes': 6000,  #maximum number of episodes (default: 3000)
        'hidden_size': 256,
        'updates_per_step': 1,  #model updates per simulator step (default: 1)
        'start_steps': 10000,  #Steps sampling random actions (default: 10000 , 200000)
        'target_update_interval': 1,  #Value target update per no. of updates per step (default: 1)
        'replay_size': 1000000,  #size of replay buffer (default: 10000000)
        'cuda': True
        }
    
    reward_hyperparams = {
        'sample_method': "random sample",  #Sample method for sampling a batch of trajectories to get ranked
        'rank_by_true_reward': True,  #rank the trajectory by true reward or by human
        'state_only': False,  #the reward net is r(s,a) or r(s)
        'hidden_dim': 256,  #hidden dim for reward network
        'learn_frequency_1': 30,   #learn reward per N episodes
        'learn_frequency_2': 200,
        'num_to_rank': 10,  #num to rank per reward update
        'num_to_rank_2': 20,
        'traj_capacity': 200,  #trajectory capacity of reward buffer
        'lr': 0.0005  
        }
    
    
    sac_hyperparams = argparse.Namespace(**sac_hyperparams)
    reward_hyperparams = argparse.Namespace(**reward_hyperparams)
    
    
    oprrl = OPRRL(sac_hyperparams, reward_hyperparams)
    
    
    oprrl.train()
           
    


            # #########################################################################################
            # while not done:
            #     if self.total_numsteps < self.sac_hyparams.start_steps:
            #         action = self.env.action_space.sample()  # Sample random action
            #     else:
            #         action = self.agent.select_action(state)  # Sample action from policy
        
            #     if len(self.agent_memory) > self.sac_hyparams.batch_size:
            #         # Number of updates per step in environment
            #         for i in range(self.sac_hyparams.updates_per_step):
            #             # Update parameters of all the networks
            #             critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.agent_memory, self.sac_hyparams.batch_size, self.updates)
            #             self.updates += 1
    
            #     next_state, reward, done, _ = self.env.step(action) # Step
            #     reward_prime = self.reward_network.get_reward(state, action).detach().cpu().numpy()[0]
                
            #     episode_steps += 1
            #     self.total_numsteps += 1
            #     episode_reward += reward
            #     episode_reward_prime += reward_prime

            #     self.agent_memory.push(state, action, reward_prime, next_state, done) # push data to agent memory
            #     self.reward_network.push_data(state, action, reward, done) # push data to reward memory
            #     state = next_state
            # #######################################################################################


