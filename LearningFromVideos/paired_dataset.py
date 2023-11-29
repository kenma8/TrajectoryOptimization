"""
This file generates a paired dataset of object trajectories and demonstration videos
that can be used to train a video classifier model. The object trajectories are also used
to test grasps.
"""

import os
import h5py
import argparse
import random
import numpy as np
import pandas as pd
import json

import robosuite.robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import imageio

def get_trajectories(f, demos, env_meta):
    """
    Creates and environment from demo environment metadata and plays back each demo,
    tracking the object position and quaternion. These object poses are saved in pickle
    formate for use in other files. 

    Args:
        f (file): hdf5 file containing demos
        demos (list): list of demo names (strings) 
        env_meta (dict): dictionary containing environment metadata
    """

    env = robosuite.robosuite.make(
        env_meta['env_name'],
        robots=env_meta['env_kwargs']['robots'],
        gripper_types="default", 
        controller_configs=env_meta['env_kwargs']['controller_configs'],
        env_configuration="single-arm-opposed",
        has_renderer=False,     
        has_offscreen_renderer=True,         
        control_freq=20,
        use_object_obs=True,
        use_camera_obs=True,
        camera_names="agentview",   
        camera_depths=True,                    
    )

    demo_names = []
    timesteps = []
    object_locations = []
    object_quaternions = []

    for i in range(len(demos)):
        print("Playing back demo {}... (press ESC to quit)".format(demos[i]))

        ep = demos[i]
        demo_names.append(ep)

        states = f["data/{}/states".format(ep)][:]
        actions = f["data/{}/actions".format(ep)][:]

        env.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        
        ts = []
        loc = []
        quat = []

        for j in range(actions.shape[0]):
            print("step ", j)
            obs, _, _, _ = env.step(actions[j])
            ts.append(j)
            loc.append(obs['SquareNut_pos'])
            quat.append(obs['SquareNut_quat'])
            print(obs['SquareNut_pos'])
        
        timesteps.append(np.array(ts))
        object_locations.append(np.array(loc))
        object_quaternions.append(np.array(quat))

    data = {
        'demo_names': demo_names,
        'timesteps': timesteps,
        'object_locations': object_locations,
        'object_quaternions': object_quaternions
    }

    df = pd.DataFrame(data)
    print(df)

    df.to_pickle('rm_paired_dataset/demo_trajectories.pkl')


def create_videos(f, demos, env_meta):
    """
    Utilizes video writer to capture and save a video of each demo in a file.

    Args:
        f (file): hdf5 file containing demos
        demos (list): list of demo names (strings) 
        env_meta (dict): dictionary containing environment metadata
    """

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, 
        render=False,            
        render_offscreen=True,   
    )

    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    

    def playback_trajectory(demo_key):
        
        Simple helper function to playback the trajectory stored under the hdf5 group @demo_key and
        write frames rendered from the simulation to the active @video_writer.
        
        video_path = "rm_paired_dataset/videos/{}.mp4".format(demo_key)
        video_writer = imageio.get_writer(video_path, fps=20)

        init_state = f["data/{}/states".format(demo_key)][0]
        model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)
        
        env.reset_to(initial_state_dict)
        
        actions = f["data/{}/actions".format(demo_key)][:]
        for t in range(actions.shape[0]):
            env.step(actions[t])
            video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
            video_writer.append_data(video_img)
        
        video_writer.close()

    #for ep in demos:
    print("Playing back demo key: {}".format("demo_0"))
    playback_trajectory("demo_0")


if __name__ == "__main__":
    
    demo_path = "datasets/square/ph/"
    hdf5_path = os.path.join(demo_path, "demo_v141.hdf5")
    f = h5py.File(hdf5_path, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    env_meta = json.loads(f["data"].attrs["env_args"])

    get_trajectories(f, demos, env_meta)

    #create_videos(f, demos, env_meta)

    f.close()


    
