#!/usr/bin/env python3
"""
Standalone test script for H12 locomotion environment.
Tests if the scene loads, robot initializes, and sensors work.
"""

import sys
import os

# Add isaaclab/h12_locomotion_rma to path
isaaclab_path = os.path.join(os.path.dirname(__file__), "..", "isaaclab", "h12_locomotion_rma")
sys.path.insert(0, isaaclab_path)

import torch
from isaaclab.app import AppLauncher

# Launch the simulator first
app_launcher = AppLauncher(headless=False)
sim = app_launcher.app

# Now import the environment after the app is initialized
from h12_locomotion_rma.tasks.manager_based.h12_locomotion_rma.h12_locomotion_rma_env_cfg import H12LocomotionRmaEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

def test_environment():
    """Test the H12 locomotion environment."""
    print("\n" + "="*60)
    print("Testing H12 Locomotion RMA Environment")
    print("="*60)
    
    # Create environment config
    env_cfg = H12LocomotionRmaEnvCfg()
    print(f"✓ Environment config created")
    print(f"  - Num envs: {env_cfg.scene.num_envs}")
    print(f"  - Env spacing: {env_cfg.scene.env_spacing}")
    
    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print(f"✓ Environment instantiated")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✓ Environment reset")
    print(f"  - Observation shape: {obs.shape}")
    
    # Test a few steps
    print("\nRunning 10 simulation steps...")
    for step in range(10):
        # Random actions
        actions = env.action_space.sample()
        obs, rewards, dones, truncated, info = env.step(actions)
        
        if step % 5 == 0:
            print(f"  Step {step}: obs_shape={obs.shape}, reward_shape={rewards.shape}")
    
    print("✓ All steps completed successfully")
    
    # Close environment
    env.close()
    print("✓ Environment closed")
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_environment()
