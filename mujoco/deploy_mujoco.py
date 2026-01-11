import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    
    # Get H12 root directory (one level up from mujoco folder)
    h12_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{H12_ROOT}", h12_root)
        xml_path = config["xml_path"].replace("{H12_ROOT}", h12_root)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    
    # Observation history buffer: (history_length, obs_dim)
    history_length = 5
    # obs_dim = 3(omega) + 3(gravity) + 3(cmd) + 12(qpos) + 12(qvel) + 12(last_action) = 45
    obs_single = np.zeros(3 + 3 + 3 + num_actions + num_actions + num_actions)
    obs_history = np.zeros((history_length, obs_single.shape[0]))
    last_action = np.zeros(num_actions, dtype=np.float32)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # Construct observation vector matching environment config:
                # [base_ang_vel(3), projected_gravity(3), velocity_commands(3), 
                #  joint_pos_rel(12), joint_vel_rel(12), last_action(12)]
                obs_single[:3] = omega
                obs_single[3:6] = gravity_orientation
                obs_single[6:9] = cmd * cmd_scale
                obs_single[9:9 + num_actions] = qj
                obs_single[9 + num_actions:9 + 2 * num_actions] = dqj
                obs_single[9 + 2 * num_actions:9 + 3 * num_actions] = last_action
                
                # Shift history and add new observation
                obs_history = np.roll(obs_history, shift=1, axis=0)
                obs_history[0] = obs_single
                
                # Flatten history for policy input: (history_length * obs_dim,)
                obs_tensor = torch.from_numpy(obs_history.flatten()).unsqueeze(0).float()
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                last_action = action.copy()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)