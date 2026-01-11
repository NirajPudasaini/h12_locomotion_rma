import time

import mujoco.viewer
import mujoco
import numpy as np
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
        
        kps_arms = np.array(config["kps_arms"], dtype=np.float32)
        kds_arms = np.array(config["kds_arms"], dtype=np.float32)

        legs_motor_pos_lower_limit_list = np.array(config["legs_motor_pos_lower_limit_list"], dtype=np.float32)
        legs_motor_pos_upper_limit_list = np.array(config["legs_motor_pos_upper_limit_list"], dtype=np.float32)

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
    
    # Arm control (keep at 0 position)
    target_arm_pos = np.zeros(len(kps_arms), dtype=np.float32)

    counter = 0
    
    # Observation history buffer matching IsaacLab config
    history_length = 5
    obs_dim = 3 + 3 + 3 + 27 + 27 + 12  # base_ang_vel + projected_gravity + velocity_commands + joint_pos_rel(27) + joint_vel_rel(27) + last_action(12)
    obs_single = np.zeros(obs_dim, dtype=np.float32)
    obs_history = np.zeros((history_length, obs_dim), dtype=np.float32)
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
            
            # PD control for legs (first 12 DOF)
            tau_legs = pd_control(target_dof_pos, d.qpos[7:19], kps, np.zeros_like(kds), d.qvel[6:18], kds)
            
            # PD control for arms (DOF 13-27, indices 19:33 in qpos, 18:32 in qvel)
            tau_arms = pd_control(target_arm_pos, d.qpos[19:], np.zeros_like(kps_arms), np.zeros_like(kds_arms), d.qvel[18:], kds_arms)
            
            d.ctrl[:] = 0  # Reset all controls
            d.ctrl[:12] = tau_legs  # Apply leg control
            d.ctrl[12:] = tau_arms  # Apply arm control to remaining actuators
            
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Extract state from MuJoCo
                qj_all = d.qpos[7:]  # All 27 joint positions (or however many are in the model)
                dqj_all = d.qvel[6:]  # All 27 joint velocities
                qj_leg = d.qpos[7:19]  # First 12 joints for PD control
                quat = d.qpos[3:7]  # Floating base quaternion
                omega = d.qvel[3:6]  # Floating base angular velocity
                
                # Scale observations to match IsaacLab training (from h12_locomotion_env_cfg.py)
                omega_scaled = omega * 0.2  # base_ang_vel scale=0.2
                gravity_orientation = get_gravity_orientation(quat)  # projected_gravity (no scale)
                
                # For joint_pos_rel: compare all joints against default_angles, padding with zeros for extra joints
                default_all = np.concatenate([default_angles, np.zeros(len(qj_all) - len(default_angles))])
                qj_rel = (qj_all - default_all) * 1.0  # joint_pos_rel (all joints, no scale)
                dqj_scaled = dqj_all * 0.05  # joint_vel_rel scale=0.05
                cmd_scaled = cmd * cmd_scale  # velocity_commands (uses cmd_scale from config)
                
                # Build observation: [base_ang_vel(3), projected_gravity(3), velocity_commands(3),
                #                     joint_pos_rel(27), joint_vel_rel(27), last_action(12)]
                obs_single[0:3] = omega_scaled
                obs_single[3:6] = gravity_orientation
                obs_single[6:9] = cmd_scaled
                obs_single[9:9+len(qj_rel)] = qj_rel  # All joint positions
                obs_single[9+len(qj_rel):9+2*len(qj_rel)] = dqj_scaled  # All joint velocities
                obs_single[9+2*len(qj_rel):9+2*len(qj_rel)+12] = last_action  # 12 actions
                
                # Shift history and add new observation
                obs_history = np.roll(obs_history, shift=1, axis=0)
                obs_history[0] = obs_single
                
                # Flatten history for policy input
                obs_tensor = torch.from_numpy(obs_history.flatten()).unsqueeze(0).float()
                
                # Policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                last_action = action.copy()
                
                # Transform action to target positions
                target_dof_pos = action * action_scale + default_angles
                
                # Constrain target positions within joint limits
                target_dof_pos = np.clip(target_dof_pos, legs_motor_pos_lower_limit_list, legs_motor_pos_upper_limit_list)
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)