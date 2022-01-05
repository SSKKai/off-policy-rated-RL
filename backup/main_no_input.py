import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import glfw

def evaluate(agent, i_episode, num_test_episodes = 3, is_render = False):
    avg_reward = 0.
    for _  in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            if is_render:
                env.render()
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        avg_reward += episode_reward
    avg_reward /= num_test_episodes


    writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(num_test_episodes, round(avg_reward, 2)))
    print("----------------------------------------")
    #glfw.terminate()



hyperparameters = {
    'env_name': "Ant-v2",  #Mujoco Gym environment  HalfCheetah-v2  BipedalWalkerHardcore-v3
    'policy': "Gaussian",  #Policy Type: Gaussian | Deterministic (default: Gaussian)
    'eval': True,  #Evaluates a policy a policy every 10 episode (default: True)
    'eval_per_episode': 10,  #evaluate policy per episode
    'render': True,  #Render when evaluate policy
    'test_episodes': 3,
    'gamma': 0.99,  
    'tau': 0.005,  #target smoothing coefficient(τ) (default: 0.005)
    'lr': 0.0003,  
    'alpha': 0.2,  #Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)
    'automatic_entropy_tuning': True,  #Automaically adjust α (default: False)
    'seed': 123456,  #random seed (default: 123456)
    'batch_size': 256,
    'num_steps': 1000001,  #maximum number of steps (default: 1000000)
    'hidden_size': 256,
    'updates_per_step': 1,  #model updates per simulator step (default: 1)
    'start_steps': 10000,  #Steps sampling random actions (default: 10000)
    'target_update_interval': 1,  #Value target update per no. of updates per step (default: 1)
    'replay_size': 1000000,  #size of replay buffer (default: 10000000)
    'cuda': True
    }
hyperparameters = argparse.Namespace(**hyperparameters)


# Environment
# env = NormalizedActions(gym.make(hyperparameters.env_name))
env = gym.make(hyperparameters.env_name)
env.seed(hyperparameters.seed)
env.action_space.seed(hyperparameters.seed)

torch.manual_seed(hyperparameters.seed)
np.random.seed(hyperparameters.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args=hyperparameters)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), hyperparameters.env_name,
                                                             hyperparameters.policy, "autotune" if hyperparameters.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(hyperparameters.replay_size, hyperparameters.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if hyperparameters.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > hyperparameters.batch_size:
            # Number of updates per step in environment
            for i in range(hyperparameters.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, hyperparameters.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > hyperparameters.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % hyperparameters.eval_per_episode == 0 and hyperparameters.eval is True:
        evaluate(agent, i_episode, num_test_episodes = hyperparameters.test_episodes, is_render = hyperparameters.render)
        # avg_reward = 0.
        # episodes = hyperparameters.test_episodes
        # for _  in range(episodes):
        #     state = env.reset()
        #     episode_reward = 0
        #     done = False
        #     while not done:
        #         if hyperparameters.render:
        #             env.render()
        #         action = agent.select_action(state, evaluate=True)

        #         next_state, reward, done, _ = env.step(action)
        #         episode_reward += reward


        #         state = next_state
        #     avg_reward += episode_reward
        # avg_reward /= episodes


        # writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        # print("----------------------------------------")
        # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        # print("----------------------------------------")

env.close()