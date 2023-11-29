import numpy as np
import pdb
import matplotlib.pyplot as plt
from time import sleep
import random

import robosuite
from robosuite.robosuite.utils.input_utils import *
from robosuite.robosuite.wrappers import GymWrapper
from robosuite.robosuite.utils.transform_utils import euler2mat, mat2quat, quat2axisangle, quat2mat
from test_grasps import get_dataset, get_cam_poses, make_env, get_grasps, calc_trajectory, move_to_grasp, compute_EtoO, camera_id, RENDER

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

RANDOM_SEED = 3

def generate_noise():
    
    random.seed(RANDOM_SEED)

    pos_noise = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.05, 0.05)])
    noisy_pos = np.array([0.24, 0.105, 1]) + pos_noise

    euler_noise = np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
    noisy_euler = np.array([0, 0, np.pi/2]) + euler_noise
    noisy_quat = mat2quat(euler2mat(noisy_euler))

    return noisy_pos, noisy_quat


def get_init_state(f, demos, env_meta):
    cam_poses = get_cam_poses()
    suite_env = make_env(env_meta)
    demo = 1
    grasps = get_grasps(demos[demo])
    cam_pose = cam_poses[demo]
    grasp_trajectory = calc_trajectory(cam_pose, grasps[33])
    eef_pose, nut_pose = move_to_grasp(f, demos[demo], suite_env, grasp_trajectory)
    T_EtoO = compute_EtoO(eef_pose, nut_pose)
    
    T_OtoN = np.eye(4)
    noisy_pos, noisy_quat = generate_noise()
    T_OtoN[:3, :3] = quat2mat(noisy_quat)
    T_OtoN[:3, 3] = noisy_pos
    T_EtoN = T_OtoN @ T_EtoO
    T_EtoN[:3, :3] = T_EtoN[:3, :3] @ euler2mat(np.array([0, 0, -np.pi/2]))
    action = np.concatenate((T_EtoN[:3, 3], quat2axisangle(mat2quat(T_EtoN[:3, :3])), [1]), axis=0)

    print("Moving to noisy pose")
    for i in range(100):
        suite_env.step(action)
        if RENDER:
            suite_env.viewer.set_camera(camera_id=camera_id)
            suite_env.viewer.render()

    init_state = suite_env.sim.get_state()
    print("Finished moving to noisy pose")
    return init_state

def train(env_meta, init_state):

    env_meta['env_kwargs']['controller_configs']['control_delta'] = False
    env_meta['env_kwargs']['controller_configs']['kp'] = [75, 50, 50, 150, 150, 150, 150]
    env_meta['env_kwargs']['controller_configs']['damping_ratio'] = 3

    print(env_meta['env_kwargs']['robots'])

    suite_env = suite.robosuite.make(
        env_meta['env_name'],
        robots=env_meta['env_kwargs']['robots'],
        gripper_types="default", 
        controller_configs=env_meta['env_kwargs']['controller_configs'],
        env_configuration="single-arm-opposed",
        has_renderer=RENDER,     
        has_offscreen_renderer=False, #True, 
        ignore_done=False,        
        control_freq=20,
        horizon=100,
        use_object_obs=True,
        use_camera_obs=False
    )


    keys = ['SquareNut_pos', 'SquareNut_quat']
    gym_env = GymWrapper(suite_env, init_state, keys=keys)

    monitor_env = Monitor(gym_env)

    run_name = "test_tensorboard"

    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 500000,
    }

    model = PPO(config["policy_type"], monitor_env, verbose=1, tensorboard_log=f"rl_runs/{run_name}")
    model.learn(
        total_timesteps=config["total_timesteps"],
    )

    model.save(f'rl_models/{run_name}')


def main():
    dataset_path = 'datasets/square/ph/low_dim.hdf5'
    f, demos, env_meta = get_dataset(dataset_path)

    init_state = get_init_state(f, demos, env_meta)
    train(env_meta, init_state)

if __name__ == "__main__":
    main()