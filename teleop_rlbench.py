import os
import time
import torch
import numpy as np
from argparse import ArgumentParser
from human_feedback import correct_action
from utils import KeyboardObserver, TrajectoriesDataset, loop_sleep
from custom_env import CustomEnv

config = {
    'task': "CloseMicrowave",  #
    'static_env': False,  #
    'headless_env': False,  #
    'save_demos': True,  #
    'learn_reward_frequency': 100,  #
    'episodes': 1,  #
    'sequence_len': 150,  #
    'obs_type': "LowDimension"  # LowDimension WristCameraRGB
}


env = CustomEnv(config)
keyboard_obs = KeyboardObserver()
env.reset()
gripper_open = 0.9
time.sleep(5)
print("Go!")
episodes_count = 0
while episodes_count < config["episodes"]:
    start_time = time.time()
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_open])
    if keyboard_obs.has_joints_cor() or keyboard_obs.has_gripper_update():
        action = correct_action(keyboard_obs, action)
        gripper_open = action[-1]
    state, next_task_obs, next_state_obs, reward, done = env.step(action)
    ee_pose = np.array([getattr(state, 'gripper_pose')[:3]])
    target_pose = np.array([getattr(state, 'task_low_dim_state')])
    distance = np.sqrt((target_pose[0, 0] - ee_pose[0, 0]) ** 2 + (target_pose[0, 1] - ee_pose[0, 1]) ** 2 + (
                target_pose[0, 2] - ee_pose[0, 2]) ** 2)
    print(distance)
    if keyboard_obs.reset_button:
        env.reset()
        gripper_open = 0.9
        keyboard_obs.reset()
    elif done:
        env.reset()
        gripper_open = 0.9
        episodes_count += 1
        keyboard_obs.reset()
        done = False
    else:
        loop_sleep(start_time)


