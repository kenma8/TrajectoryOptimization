import os
import json
import h5py
import numpy as np
import pdb
import imageio
import matplotlib.pyplot as plt
from time import sleep

from robosuite import robosuite
from robosuite.robosuite.controllers import load_controller_config
from robosuite.robosuite.utils.transform_utils import quat2axisangle, axisangle2quat

from matplotlib import pyplot as plt


ACTION_LEN = 10

INITIAL_POSE = [-0.11, 0.16, 0.84, 0, 0, 0]

camera_id = 3

def main():
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config['control_delta'] = True
    env = robosuite.make(
        env_name='NutAssemblyFloating',
        robots='Panda',
        gripper_types="default",
        controller_configs=controller_config, 
        env_configuration="single-arm-opposed",
        has_renderer=True,     
        has_offscreen_renderer=False,
        ignore_done=False,        
        control_freq=100,
        horizon=100000,
        use_object_obs=True,
        use_camera_obs=False,
        camera_names="agentview"
    )

    obs = env.reset()

    
    obj_x = []
    obj_y = []
    obj_z = []
    obj_rx = []
    obj_ry = []
    obj_rz = []

    exp_x = []
    exp_y = []
    exp_z = []
    exp_rx = []
    exp_ry = []
    exp_rz = []

    t = []

    for i in range(ACTION_LEN * 3):
        obs, _, _, _ = env.step(INITIAL_POSE)
        print(env.sim.data.body_xquat[env.obj_body_id['FloatingNut']][[1, 2, 3, 0]])
        env.viewer.set_camera(camera_id=camera_id)
        env.viewer.render()

    action = np.array([0.23, 0.1, 1, 0, 0, 0])
    print(quat2axisangle(axisangle2quat(action[3:])))
    action[3:] = quat2axisangle(axisangle2quat(action[3:]))
    for i in range(ACTION_LEN):
        obs, _, _, _ = env.step(action)
        print(env.sim.data.body_xquat[env.obj_body_id['FloatingNut']][[1, 2, 3, 0]])
        rot = quat2axisangle(obs['FloatingNut_quat'])
        obj_x.append(obs['FloatingNut_pos'][0])
        obj_y.append(obs['FloatingNut_pos'][1])
        obj_z.append(obs['FloatingNut_pos'][2])
        obj_rx.append(rot[0])
        obj_ry.append(rot[1])
        obj_rz.append(rot[2])
        exp_x.append(action[0])
        exp_y.append(action[1])
        exp_z.append(action[2])
        exp_rx.append(action[3])
        exp_ry.append(action[4])
        exp_rz.append(action[5])
        t.append(i)
        env.viewer.set_camera(camera_id=camera_id)
        env.viewer.render()
    
    print(axisangle2quat(action[3:]))
    print(obs['FloatingNut_quat'])
    
    
    action = np.array([0.23, 0.1, 0.84, 0, 0, 0])
    for i in range(ACTION_LEN):
        obs, _, _, _ = env.step(action)
        rot = quat2axisangle(obs['FloatingNut_quat'])
        obj_x.append(obs['FloatingNut_pos'][0])
        obj_y.append(obs['FloatingNut_pos'][1])
        obj_z.append(obs['FloatingNut_pos'][2])
        obj_rx.append(rot[0])
        obj_ry.append(rot[1])
        obj_rz.append(rot[2])
        exp_x.append(action[0])
        exp_y.append(action[1])
        exp_z.append(action[2])
        exp_rx.append(action[3])
        exp_ry.append(action[4])
        exp_rz.append(action[5])
        t.append(i + ACTION_LEN)
        env.viewer.set_camera(camera_id=camera_id)
        env.viewer.render()
    """
    action = np.array([0.23, 0.1, 0.84, 0, 0, 0])
    for i in range(ACTION_LEN):
        rot = quat2axisangle(obs['FloatingNut_quat'])
        obj_x.append(obs['FloatingNut_pos'][0])
        obj_y.append(obs['FloatingNut_pos'][1])
        obj_z.append(obs['FloatingNut_pos'][2])
        obj_rx.append(rot[0])
        obj_ry.append(rot[1])
        obj_rz.append(rot[2])
        exp_x.append(action[0])
        exp_y.append(action[1])
        exp_z.append(action[2])
        exp_rx.append(action[3])
        exp_ry.append(action[4])
        exp_rz.append(action[5])
        t.append(i + 2 * ACTION_LEN)
        obs, _, _, _ = env.step(action)
        env.viewer.set_camera(camera_id=camera_id)
        env.viewer.render()
    """
    
    plt.plot(t, obj_x, color='red')
    plt.plot(t, exp_x, color='blue')
    plt.show()
    plt.plot(t, obj_y, color='red')
    plt.plot(t, exp_y, color='blue')
    plt.show()
    plt.plot(t, obj_z, color='red')
    plt.plot(t, exp_z, color='blue')
    plt.show()
    plt.plot(t, obj_rx, color='red')
    plt.plot(t, exp_rx, color='blue')
    plt.show()
    plt.plot(t, obj_ry, color='red')
    plt.plot(t, exp_ry, color='blue')
    plt.show()
    plt.plot(t, obj_rz, color='red')
    plt.plot(t, exp_rz, color='blue')
    plt.show()
    

if __name__ == "__main__":
    main()
