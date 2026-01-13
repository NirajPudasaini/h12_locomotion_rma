#!/usr/bin/env python3
"""
Environment inspection using the same setup as training.
Run with: python isaaclab/scripts/rsl_rl/inspect_env.py --task Template-H12-Locomotion-v0 --num_envs 2
"""

import argparse
from omegaconf import DictConfig

import torch
import numpy as np

from isaaclab.app import AppLauncher


def main(cfg: DictConfig):
    """Inspect environment data."""
    
    print("\n" + "=" * 120)
    print("H12 ENVIRONMENT DATA INSPECTION")
    print("=" * 120)
    
    from isaaclab_tasks.utils.index_manager import ReferencedEnv
    from isaaclab.envs import ManagerBasedRLEnv
    
    # Get task config
    task_cfg = ReferencedEnv(cfg.task).task_cfg
    task_cfg.scene.num_envs = cfg.num_envs
    
    print("\n[1] ENVIRONMENT CONFIGURATION")
    print("-" * 120)
    print(f"Task: {cfg.task}")
    print(f"Number of envs: {task_cfg.scene.num_envs}")
    print(f"Simulation dt: {task_cfg.sim.dt}")
    print(f"Decimation: {task_cfg.decimation}")
    print(f"Episode length: {task_cfg.episode_length_s}s")
    print(f"Control frequency: {1 / (task_cfg.sim.dt * task_cfg.decimation):.1f} Hz")
    
    # Create environment
    print("\n[2] CREATING ENVIRONMENT...")
    print("-" * 120)
    env = ManagerBasedRLEnv(cfg=task_cfg)
    print("✓ Environment created")
    print(f"  Device: {env.device}")
    print(f"  Num envs: {env.num_envs}")
    
    # Reset
    print("\n[3] RESETTING ENVIRONMENT...")
    print("-" * 120)
    obs_dict, info = env.reset()
    print("✓ Reset complete")
    
    # Observation breakdown
    print("\n[4] OBSERVATIONS")
    print("-" * 120)
    print(f"Observation groups: {list(obs_dict.keys())}\n")
    
    for group_name, obs in obs_dict.items():
        print(f"{'='*60}")
        print(f"Group: '{group_name}'")
        print(f"{'='*60}")
        print(f"Shape: {obs.shape}")
        print(f"Dtype: {obs.dtype}")
        print(f"Range: [{obs.min().item():.6f}, {obs.max().item():.6f}]")
        print(f"Mean: {obs.mean().item():.6f}, Std: {obs.std().item():.6f}")
        
        if group_name == "policy" and obs.shape[-1] == 375:
            print(f"\nBreakdown (5 timesteps × 75 dims/timestep):")
            
            single_ts_obs = obs[0, :75]
            
            idx = 0
            print(f"\n  [0:3]   base_ang_vel:       {single_ts_obs[idx:idx+3].cpu().numpy()}")
            idx += 3
            
            print(f"  [3:6]   projected_gravity:  {single_ts_obs[idx:idx+3].cpu().numpy()}")
            idx += 3
            
            print(f"  [6:9]   velocity_commands:  {single_ts_obs[idx:idx+3].cpu().numpy()}")
            idx += 3
            
            # Joint position (12 DOF for legs)
            jp = single_ts_obs[idx:idx+12]
            idx += 12
            print(f"  [9:21]  joint_pos_rel (12 DOF):")
            print(f"    Left:  {jp[0:6].cpu().numpy()}")
            print(f"    Right: {jp[6:12].cpu().numpy()}")
            
            # Joint velocity (12 DOF)
            jv = single_ts_obs[idx:idx+12]
            idx += 12
            print(f"  [21:33] joint_vel_rel (12 DOF):")
            print(f"    Left:  {jv[0:6].cpu().numpy()}")
            print(f"    Right: {jv[6:12].cpu().numpy()}")
            
            # Last action (12 DOF)
            la = single_ts_obs[idx:idx+12]
            idx += 12
            print(f"  [33:45] last_action (12 DOF):")
            print(f"    Left:  {la[0:6].cpu().numpy()}")
            print(f"    Right: {la[6:12].cpu().numpy()}")
            
            print(f"\n  Note: This shows first timestep only. Total history: 5 timesteps.")
    
    # Action space
    print("\n[5] ACTIONS")
    print("-" * 120)
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    sample_action = env.action_space.sample()
    print(f"\nSample action (first env):")
    print(f"  Shape: {sample_action.shape}")
    print(f"  Range: [{sample_action.min().item():.6f}, {sample_action.max().item():.6f}]")
    print(f"  Left leg (6 DOF):  {sample_action[0, :6].cpu().numpy()}")
    print(f"  Right leg (6 DOF): {sample_action[0, 6:12].cpu().numpy()}")
    
    # Step with zero action
    print("\n[6] STEPPING ENVIRONMENT (zero action)")
    print("-" * 120)
    
    action = torch.zeros((env.num_envs, 12), device=env.device)
    obs_dict, rewards, dones, truncs, info = env.step(action)
    
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards (env 0): {rewards[0].item():.6f}")
    if env.num_envs > 1:
        print(f"Rewards (env 1): {rewards[1].item():.6f}")
    
    # Reward breakdown
    if hasattr(env, 'reward_manager') and hasattr(env.reward_manager, '_term_names'):
        print(f"\nReward breakdown (env 0):")
        total_rew = 0.0
        for term_name in sorted(env.reward_manager._term_names):
            if term_name in env.reward_manager._term_dof_values:
                term_val = env.reward_manager._term_dof_values[term_name][0].item()
                term_cfg = env.reward_manager._term_cfg_dict[term_name]
                term_weight = term_cfg.weight
                contribution = term_val * term_weight
                total_rew += contribution
                
                if abs(contribution) > 1e-6 or term_weight != 0:
                    print(f"  {term_name:30s}: {term_val:10.6f} × {term_weight:8.2f} = {contribution:10.6f}")
        
        print(f"  {'-'*60}")
        print(f"  {'Total reward':30s}: {total_rew:10.6f}")
    
    # Robot state
    print("\n[7] ROBOT STATE")
    print("-" * 120)
    robot = env.scene["robot"]
    
    print(f"Root state shape: {robot.root_state_w.shape}")
    print(f"\nEnv 0 root state:")
    root_state = robot.root_state_w[0].cpu().numpy()
    print(f"  Position:        {root_state[0:3]}")
    print(f"  Quaternion:      {root_state[3:7]}")
    print(f"  Lin velocity:    {root_state[7:10]}")
    print(f"  Ang velocity:    {root_state[10:13]}")
    
    print(f"\nJoint positions (env 0):")
    jp_data = robot.joint_pos[0].cpu().numpy()
    print(f"  Shape: {jp_data.shape}")
    print(f"  Values:")
    for i, (name, pos) in enumerate(zip(robot.joint_names, jp_data)):
        print(f"    {i:2d} {name:30s}: {pos:8.4f}")
    
    print(f"\nJoint velocities (env 0):")
    jv_data = robot.joint_vel[0].cpu().numpy()
    for i, (name, vel) in enumerate(zip(robot.joint_names, jv_data)):
        print(f"    {i:2d} {name:30s}: {vel:8.4f}")
    
    # Commands
    print("\n[8] COMMANDS")
    print("-" * 120)
    if hasattr(env, 'command_manager') and hasattr(env.command_manager, '_term_names'):
        for cmd_name in env.command_manager._term_names:
            if cmd_name in env.command_manager._commands:
                cmd_data = env.command_manager._commands[cmd_name][0].cpu().numpy()
                print(f"{cmd_name}: {cmd_data}")
    
    # Scene info
    print("\n[9] SCENE ENTITIES")
    print("-" * 120)
    print(f"Scene entities: {list(env.scene.keys())}")
    
    # Contact forces
    if "contact_forces" in env.scene:
        contact_sensor = env.scene["contact_forces"]
        net_forces = contact_sensor.net_forces_w
        print(f"\nContact forces shape: {net_forces.shape}")
        print(f"Contact forces (env 0, first 4 bodies):")
        for i in range(min(4, net_forces.shape[1])):
            print(f"  Body {i}: {net_forces[0, i].cpu().numpy()}")
    
    env.close()
    
    print("\n" + "=" * 120)
    print("INSPECTION COMPLETE")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect H12 environment")
    parser = AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--task", type=str, default="Template-H12-Locomotion-v0", help="Task to inspect")
    parser.add_argument("--num_envs", type=int, default=2, help="Number of environments")
    
    args = parser.parse_args()
    
    app_launcher = AppLauncher(args)
    app_context = app_launcher.app_context
    
    with app_context:
        main(args)
    
    app_launcher.close()
