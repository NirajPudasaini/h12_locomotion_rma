#!/usr/bin/env python3
"""Script to print Isaac Lab H12 joint names and ordering."""

import argparse
import torch
from isaaclab.app import AppLauncher

# Add args
parser = argparse.ArgumentParser(description="Print H12 locomotion environment joint names")
parser.add_argument("--task", type=str, default="h12_locomotion_rma-v0")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after app is launched
import gymnasium as gym

# Create environment
try:
    env = gym.make(args.task, cfg_entry_point="h12_locomotion_rma.tasks", num_envs=1, headless=True)
except Exception as e:
    print(f"Error creating environment with task '{args.task}': {e}")
    print("\nTrying to manually create the environment...")
    
    # Manual import and creation
    from h12_locomotion_rma.tasks.manager_based.h12_locomotion_rma.h12_locomotion_env_12dof_stand import (
        H12LocomotionEnvCfg,
    )
    from isaaclab.envs import ManagerBasedRLEnv
    
    env_cfg = H12LocomotionEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

# Print joint information
print("\n" + "=" * 80)
print("H12 LOCOMOTION RMA - JOINT NAMES (robot.data.joint_names):")
print("=" * 80)

robot = env.scene["robot"]
joint_names = robot.data.joint_names
for i, name in enumerate(joint_names):
    print(f"Index {i:2d}: {name}")

print("\n" + "=" * 80)
print(f"Total joints: {len(joint_names)}")
print("=" * 80)

# Also print action joint names if they differ
print("\n" + "=" * 80)
print("ACTION CONFIGURATION:")
print("=" * 80)
print(f"Action scale: {env.action_manager.action.scale}")
print(f"Action shape: {env.action_manager.action.data.joint_pos.shape}")

simulation_app.close()
