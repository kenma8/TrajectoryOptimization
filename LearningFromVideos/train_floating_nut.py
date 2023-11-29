"""
This file trains an RL model that corrects a noisy trajectory using a floating nut.
It uses the PPO RL algorithm from Stable-Baselines3 and a custom robosuite environment.

The custom environment is a modified nut assembly environment that contains a FloatingNut
object rather than a SquareNut/RoundNut. The robosuite/robosuite/environments/manipulation/nut_assembly.py
file has been modified for this purpose. All other programs use the original nut_assembly.py file,
which can be found on the robosuite repo.
"""


import os
import json
import h5py
import numpy as np
import pdb
import imageio
import matplotlib.pyplot as plt
from time import sleep
import argparse
from typing import Callable

from robosuite import robosuite
from robosuite.robosuite.controllers import load_controller_config
from robosuite.robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat, euler2mat
from robosuite.robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

ACTION_LEN = 50

CAMERA_ID = 3

TRANSLATION_STDDEV = 0.004
ROTATION_STDDEV = np.pi / 80
NOISE_COUNT = 4

INITIAL_POSE = np.array([-0.11, 0.16, 0.84, 0, 0, 0])

KEY_POINTS = [np.array([-0.11, 0.16, 0.84, 0, 0, 0]), np.array([0.23, 0.1, 1.1, 0, 0, 0]), np.array([0.23, 0.1, 0.84, 0, 0, 0])]
POINTS_PER_KP = 16

RENDER = False

N_STEPS = POINTS_PER_KP * (len(KEY_POINTS) - 1)
BATCH_SIZE = 8
LEARNING_RATE = 0.005
DISCOUNT_FACTOR = 0

SEED = 1

RUN_NAME = 'floating_nut_varied_start_1.2.4.6'

def creat_env():
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = True
    env = robosuite.make(
        env_name='NutAssemblyFloating',
        robots='Panda',
        gripper_types="default",
        controller_configs=controller_config, 
        env_configuration="single-arm-opposed",
        has_renderer=RENDER,     
        has_offscreen_renderer=False,
        ignore_done=False,        
        control_freq=100,
        horizon=N_STEPS * 5,
        use_object_obs=True,
        use_camera_obs=False,
        hard_reset=False,
    )
    return env

def generate_trajectory(key_points):
    trajectory = []

    for i in range(len(key_points) - 1):
        for j in range(POINTS_PER_KP):
            trajectory.append(key_points[i] + (key_points[i + 1] - key_points[i]) * (j + 1) / POINTS_PER_KP)

    return trajectory

def generate_noise(points, run_name):
    noisy_points = []
    
    translation_samples = np.random.normal(0, TRANSLATION_STDDEV, (len(points) * NOISE_COUNT, 3))
    rotation_samples = np.random.normal(0, ROTATION_STDDEV, (len(points) * NOISE_COUNT, 3))
    
    for j in range(NOISE_COUNT):
        for i in range(len(points)):
            new_pos = points[i][:3] + translation_samples[i + j * len(points)]
            new_rot = points[i][3:] + rotation_samples[i + j * len(points)] 
            noisy_points.append(np.concatenate((new_pos, new_rot), axis=0))

    np.save('noisy_points/{}_points'.format(run_name), np.array(noisy_points))

    return noisy_points

def generate_initial_state(env):
    env.reset()

    for i in range(ACTION_LEN):
        obs, _, _, _ = env.step(INITIAL_POSE)

    init_state = env.sim.get_state()
    return init_state

def train(env, trajectory, noisy_points, run_name, init_state=None):

    keys = ['FloatingNut_pos', 'FloatingNut_quat']
    gym_env = GymWrapper(env, KEY_POINTS, keys=keys) #noisy_points, trajectory, init_state, KEY_POINTS[0], NOISE_COUNT
    
    monitor_env = Monitor(gym_env)

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 20000,
        "env_name": "FloatingNut"
    }

    run = wandb.init(
        project="LearningFromVideos",
        config=config,
        sync_tensorboard=True,
        save_code=True,
    )

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    #model = PPO.load('rl_models/{}/model(1).zip'.format(RUN_NAME), monitor_env, n_steps=N_STEPS, batch_size=BATCH_SIZE, learning_rate=linear_schedule(LEARNING_RATE), verbose=1, tensorboard_log=f"rl_runs/{RUN_NAME}.0", device='cuda') 
    model = PPO(config["policy_type"], monitor_env, n_steps=N_STEPS, batch_size=BATCH_SIZE, learning_rate=linear_schedule(LEARNING_RATE), gamma=DISCOUNT_FACTOR, verbose=1, tensorboard_log=f"rl_runs/{run_name}", device='cuda')

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"rl_models/{run_name}",
            verbose=2,
        )
    )
    
    
def view(env, trajectory, init_state=None):
    keys = ['FloatingNut_pos', 'FloatingNut_quat']
    #noisy_points = np.load('noisy_points/{}_points.npy'.format(RUN_NAME)) #generate_noise(trajectory) 
    gym_env = GymWrapper(env, KEY_POINTS, keys=keys) #noisy_points, trajectory, init_state, KEY_POINTS[0], NOISE_COUNT

    model = PPO.load('rl_models/{}/model'.format(RUN_NAME))

    obs = gym_env.reset()[0]
    gym_env.render()
    
    trajectory = generate_trajectory([obs[6:]] + KEY_POINTS[1:])

    noisy_x = []
    real_x = []
    model_x = []
    noisy_y = []
    real_y = []
    model_y = []
    noisy_z = []
    real_z = []
    model_z = []
    noisy_rx = []
    real_rx = []
    model_rx = []
    noisy_ry = []
    real_ry = []
    model_ry = []
    noisy_rz = []
    real_rz = []
    model_rz = []

    t = []

    for i in range(len(trajectory)):
        action, _states = model.predict(obs)
        noisy_x.append(obs[0])
        model_x.append(action[0] * 0.012 + obs[0])
        real_x.append(trajectory[i][0])
        noisy_y.append(obs[1])
        model_y.append(action[1] * 0.012 + obs[1])
        real_y.append(trajectory[i][1])
        noisy_z.append(obs[2])
        model_z.append(action[2] * 0.012 + obs[2])
        real_z.append(trajectory[i][2])
        noisy_rx.append(obs[3])
        model_rx.append(action[3] * 3*np.pi/80 + obs[3])
        real_rx.append(trajectory[i][3])
        noisy_ry.append(obs[4])
        model_ry.append(action[4] * 3*np.pi/80 + obs[4])
        real_ry.append(trajectory[i][4])
        noisy_rz.append(obs[5])
        model_rz.append(action[5] * 3*np.pi/80 + obs[5])
        real_rz.append(trajectory[i][5])
        obs = gym_env.step(action)[0]
        print('Flattened Obs', obs)
        t.append(i)

    obs = gym_env.reset()[0]
    for j in range(len(trajectory)):
        action, _states = model.predict(obs)
        obs = gym_env.step(np.zeros(6))[0]
        print('Flattened Obs', obs)

    plt.plot(t, noisy_x, color='red')
    plt.plot(t, real_x, color='green')
    plt.plot(t, model_x, color='blue')
    plt.show()

    plt.plot(t, noisy_y, color='red')
    plt.plot(t, real_y, color='green')
    plt.plot(t, model_y, color='blue')
    plt.show()

    plt.plot(t, noisy_z, color='red')
    plt.plot(t, real_z, color='green')
    plt.plot(t, model_z, color='blue')
    plt.show()

    plt.plot(t, noisy_rx, color='red')
    plt.plot(t, real_rx, color='green')
    plt.plot(t, model_rx, color='blue')
    plt.show()

    plt.plot(t, noisy_ry, color='red')
    plt.plot(t, real_ry, color='green')
    plt.plot(t, model_ry, color='blue')
    plt.show()

    plt.plot(t, noisy_rz, color='red')
    plt.plot(t, real_rz, color='green')
    plt.plot(t, model_rz, color='blue')
    plt.show()

def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed(SEED)

    if args.run_name is not None:
        run_name = args.run_name
    else:
        run_name = RUN_NAME

    env = creat_env()
    trajectory = generate_trajectory(KEY_POINTS)
    
    #init_state = generate_initial_state(env)
    if RENDER:
        view(env, trajectory) 
    else:
        noisy_points = generate_noise(trajectory, run_name)
        train(env, trajectory, noisy_points, run_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--run_name', type=str)
    args = parser.parse_args()
    main(args)
