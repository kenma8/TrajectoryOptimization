import os
import json
import h5py
import numpy as np
import pdb
import imageio
import matplotlib.pyplot as plt
from time import sleep

import robosuite
from robosuite.robosuite.utils.input_utils import *
from robosuite.robosuite.wrappers import GymWrapper

from stable_baselines3 import PPO

from train_rl import get_init_state

def get_dataset(dataset_path):
    """
    Given hdf5 dataset path, retrieve the file, list of demos, and environment metadata.
    """
    assert os.path.exists(dataset_path)
    f = h5py.File(dataset_path, "r")

    demos = list(f["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    env_meta = json.loads(f["data"].attrs["env_args"])

    return f, demos, env_meta

def make_env(env_meta):
    """
    Create a robosuite environment with an 'OSC_POSE' controller based on the environment metadata of the original demonstration.
    """

    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['kp'] = [75, 50, 50, 150, 150, 150, 150]
    env_meta['env_kwargs']['controller_configs']['damping_ratio'] = 3

    render_env = robosuite.robosuite.make(
        env_meta['env_name'],
        robots=env_meta['env_kwargs']['robots'],
        gripper_types="default", 
        controller_configs=env_meta['env_kwargs']['controller_configs'],
        env_configuration="single-arm-opposed",
        has_renderer=True,     
        has_offscreen_renderer=False, #True, 
        ignore_done=False,        
        control_freq=20,
        horizon=1000,
        use_object_obs=True,
        use_camera_obs=False
    
        #use_camera_obs=True,
        #camera_names='agentview',
        #camera_heights=1024,
        #camera_widths=1024,
    )
    return render_env

def main():
    pdb.set_trace()
    dataset_path = 'datasets/square/ph/low_dim.hdf5'
    f, demos, env_meta = get_dataset(dataset_path)

    init_state = get_init_state(f, demos, env_meta)

    keys = ['SquareNut_pos', 'SquareNut_quat']
    gym_env = GymWrapper(make_env(env_meta), init_state, keys=keys)

    model = PPO.load('rl_models/test_tensorboard')

    obs = gym_env.reset()[0]
    print(obs)
    gym_env.render()

    """
    for i in range(100):
        action = np.array([0.25, 0.1, 1.1, 0, np.pi, 0, 1])
        obs, _, _, _ = gym_env.step(action)
        gym_env.render()
    """
    while True:
        action, _states = model.predict(obs)
        obs = gym_env.step(action)[0]
        gym_env.render()
    
    

if __name__ == "__main__":
    main()
