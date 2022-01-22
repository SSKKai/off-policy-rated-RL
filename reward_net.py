import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta,Normal
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Reward_Net(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Reward_Net, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs):
        x = F.leaky_relu(self.linear1(inputs))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))
        x = torch.tanh(self.linear4(x))
        return x


class RewardNetwork(object):
    def __init__(self, state_dim, action_dim, episode_length, args):
        self.device = device
        
        self.state_only = args.state_only
        self.rank_by_true_reward = args.rank_by_true_reward
        self.new_trajectory = True
        self.new_reward_traj = True
        
        
        self.traj_capacity = args.traj_capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        
        self.num_to_rank = args.num_to_rank
        
        if self.state_only:
            num_inputs = self.state_dim
        else:
            num_inputs = self.state_dim+self.action_dim
        self.reward_network = Reward_Net(num_inputs, args.hidden_dim).to(device=self.device)
        
        self.sample_method = args.sample_method
        
        self.buffer = []
        self.true_reward_buffer = []
        self.ranked_trajs = []
        self.ranked_labels = []
        
        self.train_batch_size = 128
        self.epoch = 3
        self.optimizer = optim.Adam(self.reward_network.parameters(), lr=args.lr)
    
    def get_reward(self, state, action):
        inputs = np.concatenate([state, action], axis=-1)
        reward = self.reward_network(torch.from_numpy(inputs).float().to(self.device))
        return reward
    
    
    def push_data(self, state, action, reward, done):
        if self.state_only:
            inputs = state
        else:
            inputs = np.concatenate([state, action], axis=-1)
        
        if self.new_trajectory:
            self.buffer.append([])
            self.new_trajectory = False
        
        self.buffer[-1].append(inputs)
        if done:
            self.new_trajectory = True
            ################## temp ######################
            if len(self.buffer[-1]) < self.episode_length:
                del self.buffer[-1]
            ###############################################
            if len(self.buffer) > self.traj_capacity:
                self.buffer = self.buffer[1:]
        
        if self.rank_by_true_reward:
            self.add_true_reward (reward, done)
    
    def add_true_reward (self, reward, done):
        if self.new_reward_traj:
            self.true_reward_buffer.append([])
            self.new_reward_traj = False
        
        self.true_reward_buffer[-1].append(reward)
        if done:
            self.new_reward_traj = True
            ################## temp ######################
            if len(self.true_reward_buffer[-1]) < self.episode_length:
                del self.true_reward_buffer[-1]
            ###############################################
            if len(self.true_reward_buffer) > self.traj_capacity:
                self.true_reward_buffer = self.true_reward_buffer[1:]
    
    def rank(self):
        if self.sample_method == 'random sample':
            rank_batch,rank_index = self.random_sample()
        else:
            raise Exception('wrong sample method for reward learning')
        
        rank_label = self.get_rank(rank_batch, rank_index)
        
        self.ranked_trajs.extend(rank_batch)
        self.ranked_labels.extend(rank_label)
        
        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if (i not in rank_index)]
        if self.rank_by_true_reward:
            self.true_reward_buffer = [self.true_reward_buffer[i] for i in range(len(self.true_reward_buffer))
                                       if (i not in rank_index)]
        
        
    
    #random sample
    def random_sample(self):
        
        if len(self.buffer) < self.num_to_rank:
            num_to_rank = len(self.buffer)
        else: 
            num_to_rank = self.num_to_rank
        
        sample_index = random.sample(range(len(self.buffer)), num_to_rank)
        sample_batch = []
        
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])
        #batch = random.sample(self.buffer, self.num_to_rank)
        return sample_batch, sample_index
    
    
    def get_rank(self, rank_batch, rank_index):
        rank_label = []
        if self.rank_by_true_reward:
            for i in range(self.num_to_rank):
                total_reward = sum(self.true_reward_buffer[rank_index[i]])
                
                # if total_reward <= -1500:
                #     rank = 0
                # elif -1500 < total_reward <= 0:
                #     rank = 2*(total_reward+1500)/1500
                # elif total_reward > 0:
                #     rank = 2 + 8*total_reward/5000
                #     if rank > 10:
                #         rank = 10
                
                if total_reward/self.episode_length <= -1.5:
                    rank = 0
                elif -1.5 < total_reward/self.episode_length <= 0:
                    rank = 2*(total_reward/self.episode_length+1.5)/1.5
                elif total_reward > 0:
                    rank = 2 + 8*(total_reward/self.episode_length)/5
                    if rank > 10:
                        rank = 10
                
                rank_label.append(rank)
        
        return rank_label
    
    def make_training_batch(self):
        index_list = []
        if len(self.ranked_trajs) >= 20:
            batch_size = self.train_batch_size
        else:
            batch_size = 32
            
        while len(index_list) < batch_size:
            index = random.sample(range(len(self.ranked_trajs)),2)
            index.sort()
            if index not in index_list:
                index_list.extend([index])
        
        traj_list_1 = []
        traj_list_2 = []
        rank_list = []
        for idx in index_list:
            traj_list_1.extend([self.ranked_trajs[idx[0]]])
            traj_list_2.extend([self.ranked_trajs[idx[1]]])
            rank_list.extend([[self.ranked_labels[idx[0]], self.ranked_labels[idx[1]]]])
        
        
        #labels = [0 if rank[0] > rank[1] else 0.5 if rank[0]==rank[1] else 1 for rank in rank_list]
        labels = [0 if rank[0] > rank[1] else 1 for rank in rank_list]
        labels = np.array(labels)
        traj_list_1 = np.array(traj_list_1)
        traj_list_2 = np.array(traj_list_2)
        
        
        labels = torch.from_numpy(labels.flatten()).long().to(self.device)
        
        rewards_1 = self.reward_network(torch.from_numpy(traj_list_1).float().to(self.device))
        rewards_2 = self.reward_network(torch.from_numpy(traj_list_2).float().to(self.device))
        rewards_1 = rewards_1.sum(axis=1)
        rewards_2 = rewards_2.sum(axis=1)
        rewards = torch.cat([rewards_1, rewards_2], axis=-1)
        
        return rewards, labels
        

    
    def learn_reward(self):
        for epoch in range(self.epoch):
            self.optimizer.zero_grad()
            index_list = []
            if len(self.ranked_trajs) >= 20:
                batch_size = self.train_batch_size
            else:
                batch_size = 32
                
            while len(index_list) < batch_size:
                index = random.sample(range(len(self.ranked_trajs)),2)
                index.sort()
                if index not in index_list:
                    if abs(self.ranked_labels[index[0]]-self.ranked_labels[index[1]]) < 1 or random.random()>0.8:
                        index_list.extend([index])
            
            # while len(index_list) < batch_size:
            #     index = random.sample(range(len(self.ranked_trajs)),1)
            #     index_2 = random.sample(range(len(self.ranked_trajs)),1)
            #     while abs(self.ranked_labels[index[0]]-self.ranked_labels[index_2[0]])>1 and random.random()<0.8:
            #         index_2 = random.sample(range(len(self.ranked_trajs)),1)
            #     index.extend(index_2)
            #     index.sort()
            #     if index not in index_list:
            #         index_list.extend([index])
                
            
            traj_list_1 = []
            traj_list_2 = []
            rank_list = []
            for idx in index_list:
                traj_list_1.extend([self.ranked_trajs[idx[0]]])
                traj_list_2.extend([self.ranked_trajs[idx[1]]])
                rank_list.extend([[self.ranked_labels[idx[0]], self.ranked_labels[idx[1]]]])
            
            
            #labels = [0 if rank[0] > rank[1] else 0.5 if rank[0]==rank[1] else 1 for rank in rank_list]
            labels = [0 if rank[0] > rank[1] else 1 for rank in rank_list]
            labels = np.array(labels)
            traj_list_1 = np.array(traj_list_1)
            traj_list_2 = np.array(traj_list_2)
            
            
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            
            rewards_1 = self.reward_network(torch.from_numpy(traj_list_1).float().to(self.device))
            rewards_2 = self.reward_network(torch.from_numpy(traj_list_2).float().to(self.device))
            rewards_1 = rewards_1.sum(axis=1)
            rewards_2 = rewards_2.sum(axis=1)
            rewards = torch.cat([rewards_1, rewards_2], axis=-1)
            
            
            loss = nn.CrossEntropyLoss()(rewards, labels)
            loss.backward()
            self.optimizer.step()
        
        #soft_cross_entropy_loss()
        #return
    
    
    
    def save_reward_model(self, env_name, version, reward_path = None):
        if not os.path.exists('reward_models/'):
            os.makedirs('reward_models/')
        
        if reward_path is None:
            reward_path = "reward_models/reward_model_{}_{}".format(env_name, version)
        
        print('Saving reward network to {}'.format(reward_path))
        torch.save(self.reward_network.state_dict(), reward_path)
    
    def load_reward_model(self, reward_path):
        if reward_path is not None:
            print('Loading reward network to {}'.format(reward_path))
            self.reward_network.load_state_dict(torch.load(reward_path))
        else:
            print('fail to load reward network, please enter the reward path')
    
    
    def save_trajs(self, env_name, version, trajs_path=None, labels_path=None, num_to_sample = 20):
        if not os.path.exists('saved_trajs/'):
            os.makedirs('saved_trajs/')
        
        if trajs_path is None:
            trajs_path = "saved_trajs/trajs_{}_{}.npy".format(env_name, version)
        if labels_path is None:
            labels_path = "saved_trajs/labels_{}_{}.npy".format(env_name, version)
        
        index = random.sample(range(len(self.ranked_trajs)), num_to_sample)
        
        save_trajs = [self.ranked_trajs[idx] for idx in index]
        save_labels = [self.ranked_labels[idx] for idx in index]
        save_trajs = np.array(save_trajs)
        save_labels = np.array(save_labels)
        
        label_order = np.argsort(save_labels)
        save_labels.sort()
        save_trajs = save_trajs.take(label_order, axis=0)
        
        np.save(trajs_path, save_trajs)
        np.save(labels_path, save_labels)
    
    def test_reward_model(self, trajs_path, labels_path):
        
        trajs = np.load(trajs_path)
        labels = np.load(labels_path)
        
        rewards = self.reward_network(torch.from_numpy(trajs).float().to(self.device))
        rewards = rewards.sum(axis=1)
        rewards = rewards.detach().cpu().numpy()
        
        return rewards, labels
        
        










