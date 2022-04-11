import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.distributions import Beta,Normal
from sklearn.cluster import KMeans
import math
import random
import heapq


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
        # return x
        return (x-1)/2


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
        self.padding_mask_method = args.padding_mask_method
        self.label_type = args.label_type

        self.buffer = []
        self.true_reward_buffer = []
        self.ranked_trajs = []
        self.ranked_labels = []
        self.ranked_lens = []

        self.prio_alpha = args.prio_alpha
        self.sample_batch_prob = []

        self.best_trajs_num = args.best_trajs_num
        self.best_traj_index = []

        
        self.train_batch_size = 128
        self.epoch = 3
        self.optimizer = optim.Adam(self.reward_network.parameters(), lr=args.lr)


    
    def get_reward(self, state, action):
        if self.state_only:
            inputs = state
        else:
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
            if len(self.buffer) > self.traj_capacity:
                self.buffer = self.buffer[1:]
        
        if self.rank_by_true_reward:
            self.add_true_reward(reward, done)
    
    def add_true_reward (self, reward, done):
        if self.new_reward_traj:
            self.true_reward_buffer.append([])
            self.new_reward_traj = False
        
        self.true_reward_buffer[-1].append(reward)
        if done:
            self.new_reward_traj = True
            if len(self.true_reward_buffer) > self.traj_capacity:
                self.true_reward_buffer = self.true_reward_buffer[1:]

    
    def rank(self):
        if self.sample_method == 'random sample':
            rank_traj, rank_index = self.random_sample_buffer()
        elif self.sample_method == 'distance sample':
            rank_traj, rank_index = self.distance_sample_buffer()
        else:
            raise Exception('wrong sample method for reward learning')
        
        rank_label = self.get_rank(rank_traj, rank_index)

        rank_traj, rank_len = self.padding(rank_traj)
        
        self.ranked_trajs.extend(rank_traj)
        self.ranked_labels.extend(rank_label)
        self.ranked_lens.extend(rank_len)
        
        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if (i not in rank_index)]
        if self.rank_by_true_reward:
            self.true_reward_buffer = [self.true_reward_buffer[i] for i in range(len(self.true_reward_buffer))
                                       if (i not in rank_index)]

        self.update_sample_prob(self.prio_alpha)


    def random_sample_buffer(self):
        
        if len(self.buffer) < self.num_to_rank:
            num_to_rank = len(self.buffer)
        else: 
            num_to_rank = self.num_to_rank
        
        sample_index = random.sample(range(len(self.buffer)), num_to_rank)
        sample_batch = []
        
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])
        # batch = random.sample(self.buffer, self.num_to_rank)
        return sample_batch, sample_index

    def distance_sample_buffer(self):

        if len(self.buffer) < self.num_to_rank:
            num_to_rank = len(self.buffer)
        else:
            num_to_rank = self.num_to_rank

        padded_buffer, buffer_len_list = self.padding(self.buffer)
        buffer_mask = self.make_mask(buffer_len_list)[0]

        buffer_mask = buffer_mask.to(self.device)
        padded_buffer = np.array(padded_buffer)

        rewards = self.reward_network(torch.from_numpy(padded_buffer).float().to(self.device))
        rewards = torch.squeeze(rewards, axis=-1)
        rewards = rewards * buffer_mask
        rewards = rewards.sum(axis=1)

        rewards = rewards.detach().cpu().numpy().reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_to_rank, random_state=0).fit(rewards)
        centers = kmeans.cluster_centers_.squeeze(axis=-1)

        sample_index = []
        for center in centers:
            idx = (np.abs(rewards - center)).argmin()
            sample_index.append(idx)

        sample_batch = []
        for i in range(num_to_rank):
            sample_batch.append(self.buffer[sample_index[i]])

        return sample_batch, sample_index




    def padding(self, traj_list, method=None):

        if method is None:
            method = self.padding_mask_method

        batch_len = len(traj_list)
        traj_len_list = [len(traj) for traj in traj_list]
        pad_len_list = [self.episode_length - traj_len_list[i] for i in range(batch_len)]

        pad_list = []

        for i in range(batch_len):
            if "zeros" in method:
                traj_pad = [np.zeros(self.action_dim + self.state_dim) for _ in range(pad_len_list[i])]
            if "edge" in method:
                traj_pad = [traj_list[i][-1] for _ in range(pad_len_list[i])]
            if "last" in method:
                n = [int(s) for s in method.split() if s.isdigit()][0]
                pad_unit = traj_list[i][-n:]
                traj_pad = [pad_unit for _ in range(int(np.ceil(pad_len_list[i]/n)))]
                traj_pad = [sa for st in traj_pad for sa in st]
                if len(traj_pad) > pad_len_list[i]:
                    del traj_pad[0:len(traj_pad) - pad_len_list[i]]

            pad_list.append(traj_pad)

        padded_traj_list = [traj_list[i] + pad_list[i] for i in range(batch_len)]

        return padded_traj_list, traj_len_list


    def update_sample_prob(self, prio_alpha):
        priority = list(map(lambda x: x ** prio_alpha, self.ranked_labels))
        prio_sum = sum(priority)

        self.sample_batch_prob = [prio / prio_sum for prio in priority]

        if self.best_trajs_num > 0:
            self.best_traj_index = list(map(self.ranked_labels.index, heapq.nlargest(self.best_trajs_num, self.ranked_labels)))

    
    def get_rank(self, rank_batch, rank_index):
        rank_label = []
        if self.rank_by_true_reward:
            for idx in rank_index:
                total_reward = sum(self.true_reward_buffer[idx])

                if total_reward/self.episode_length <= -1.5:
                    rank = 0
                elif -1.5 < total_reward/self.episode_length <= 0:
                    rank = 2*(total_reward/self.episode_length+1.5)/1.5
                elif total_reward > 0:
                    rank = 2 + 8*(total_reward/self.episode_length)/5
                    if rank > 10:
                        rank = 10

                if total_reward/self.episode_length <= -3.2:  # -1.2 -2 -3
                    rank = 0
                else:
                    rank = ((total_reward/self.episode_length)/3.2 + 1)*10
                
                rank_label.append(rank)
        return rank_label


    def make_mask(self, traj_len_list):
        dim = np.array(traj_len_list).ndim

        if dim == 1:
            traj_lens = [traj_len_list]
        elif dim == 2:
            if "normal mask" in self.padding_mask_method:
                len1 = [len[0] for len in traj_len_list]
                len2 = [len[1] for len in traj_len_list]
            elif "shortest mask" in self.padding_mask_method:
                len1 = [min(len) for len in traj_len_list]
                len2 = len1
            elif "no mask" in self.padding_mask_method:
                len1 = [self.episode_length for _ in traj_len_list]
                len2 = len1

            acc_len1 = [len[0] for len in traj_len_list]
            acc_len2 = [len[1] for len in traj_len_list]

            traj_lens = [len1, len2, acc_len1, acc_len2]

        masks = []
        for lens in traj_lens:
            lens = torch.tensor(lens)
            mask = torch.arange(self.episode_length)[None, :] < lens[:, None]
            masks.append(mask)

        return masks


    def get_labels(self, rank_list):

        acc_labels = [0 if rank[0] > rank[1] else 1 for rank in rank_list]

        if self.label_type == "onehot":
            labels = [[1, 0] if rank[0] > rank[1] else [0, 1] for rank in rank_list]

        else:
            labels = []
            for rank in rank_list:
                if rank[0] == rank[1]:
                    label_1 = 0.5
                    label_2 = 0.5
                else:
                    if "adaptive" in self.label_type:
                        smoothing_alpha = 1 / ((2 + abs(rank[0] - rank[1])) ** 2)
                    elif "smoothing" in self.label_type:
                        try:
                            smoothing_alpha = float(self.label_type.split()[1])
                        except IndexError:
                            smoothing_alpha = 0.05
                    label_1 = (1 - smoothing_alpha) * (rank[0] > rank[1]) + smoothing_alpha / 2
                    label_2 = (1 - smoothing_alpha) * (rank[0] < rank[1]) + smoothing_alpha / 2

                labels.append([label_1, label_2])

        return labels, acc_labels

    def random_batch_index(self, batch_size):

        index_list = []

        while len(index_list) < batch_size:
            idx = random.sample(range(len(self.ranked_trajs)), 2)
            idx.sort()
            if idx not in index_list:
                if abs(self.ranked_labels[idx[0]] - self.ranked_labels[idx[1]]) < 1 or random.random() > 0:  # 0.8
                    index_list.extend([idx])

        return index_list

    def priority_batch_index(self, batch_size):

        index = list(range(len(self.sample_batch_prob)))

        index_list = []

        if len(self.best_traj_index) > 0:
            while len(index_list) < int(np.ceil(batch_size*0.05)):
                idx1 = random.sample(self.best_traj_index, 1)[0]
                idx2 = random.choices(index, weights=self.sample_batch_prob, k=1)[0]
                idx = [idx1, idx2]
                idx.sort()
                if idx not in index_list:
                    if abs(self.ranked_labels[idx[0]] - self.ranked_labels[idx[1]]) < 1 or random.random() > 0.8:  # 0.8
                        index_list.extend([idx])

        while len(index_list) < batch_size:
            idx = random.choices(index, weights=self.sample_batch_prob, k=2)
            idx.sort()
            if idx not in index_list:
                index_list.extend([idx])

        return index_list

    def make_batch(self):

        if len(self.ranked_trajs) >= 20:
            batch_size = self.train_batch_size
        else:
            batch_size = 2 * len(self.ranked_trajs)

        # make index
        index_list = self.priority_batch_index(batch_size)
        # index_list = self.random_batch_index(batch_size)


        # index_list = []
        # while len(index_list) < batch_size:
        #     index = random.sample(range(len(self.ranked_trajs)), 2)
        #     index.sort()
        #     if index not in index_list:
        #         if abs(self.ranked_labels[index[0]] - self.ranked_labels[index[1]]) < 1 or random.random() > 0:  # 0.8
        #             index_list.extend([index])

        traj_list_1 = []
        traj_list_2 = []
        rank_list = []
        len_list = []
        for idx in index_list:
            traj_list_1.extend([self.ranked_trajs[idx[0]]])
            traj_list_2.extend([self.ranked_trajs[idx[1]]])
            rank_list.extend([[self.ranked_labels[idx[0]], self.ranked_labels[idx[1]]]])
            len_list.extend([[self.ranked_lens[idx[0]], self.ranked_lens[idx[1]]]])

        return traj_list_1, traj_list_2, rank_list, len_list, index_list



    def learn_reward_soft(self):

        for epoch in range(self.epoch):
            self.optimizer.zero_grad()

            # make batch
            traj_list_1, traj_list_2, rank_list, len_list, index_list = self.make_batch()

            # make mask
            mask_1, mask_2, acc_mask_1, acc_mask_2 = self.make_mask(len_list)

            # make labels
            labels, acc_labels = self.get_labels(rank_list)


            # training batch to device
            traj_list_1 = np.array(traj_list_1)
            traj_list_2 = np.array(traj_list_2)
            traj_list_1 = torch.from_numpy(traj_list_1).float().to(self.device)
            traj_list_2 = torch.from_numpy(traj_list_2).float().to(self.device)
            mask_1 = mask_1.to(self.device)
            mask_2 = mask_2.to(self.device)
            acc_mask_1 = acc_mask_1.to(self.device)
            acc_mask_2 = acc_mask_2.to(self.device)
            labels = torch.tensor(labels).to(self.device)
            acc_labels = torch.tensor(acc_labels).to(self.device)


            # compute loss
            rewards_1 = self.reward_network(traj_list_1)
            rewards_2 = self.reward_network(traj_list_2)

            rewards_1 = torch.squeeze(rewards_1, axis=-1)
            rewards_2 = torch.squeeze(rewards_2, axis=-1)

            acc_rewards_1 = rewards_1 * acc_mask_1
            acc_rewards_2 = rewards_2 * acc_mask_2
            rewards_1 = rewards_1 * mask_1
            rewards_2 = rewards_2 * mask_2

            rewards_1 = rewards_1.sum(axis=1)
            rewards_2 = rewards_2.sum(axis=1)
            rewards = torch.stack((rewards_1, rewards_2),axis=1)

            log_probs = torch.nn.functional.log_softmax(rewards, dim=1)
            loss = -(labels * log_probs).sum() / rewards.shape[0]

            loss.backward()
            self.optimizer.step()

            # compute acc
            acc_rewards_1 = acc_rewards_1.sum(axis=1)
            acc_rewards_2 = acc_rewards_2.sum(axis=1)
            acc_rewards = torch.stack((acc_rewards_1, acc_rewards_2),axis=1)
            _, predicted_labels = torch.max(acc_rewards.data, 1)
            acc = (predicted_labels == acc_labels).sum().item()/len(predicted_labels)

            return loss, acc
    

    
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
        
        










