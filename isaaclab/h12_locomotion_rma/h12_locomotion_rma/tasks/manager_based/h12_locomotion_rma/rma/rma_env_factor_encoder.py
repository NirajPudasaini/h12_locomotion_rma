# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment Factor Encoder for RMA.

This module implements the environment factor encoder.

The encoder observes proprioceptive and other privileged information and
encodes it into a compact representation of environment factors (friction, 
payload, etc.) that the policy can use to adapt.
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EnvFactorEncoder(nn.Module):
    """Environment factor encoder network for RMA adaptation.
    
    This encoder takes in observations from the environment (e.g., proprioceptive data,
    contact forces, etc.) and produces a compact representation of environment factors.
    These factors can then be fed to the policy for rapid adaptation.
    
    Args:
        input_dim: Dimension of input observations
        encoding_dim: Dimension of the encoded environment factors
        hidden_dim: Dimension of hidden layers
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, encoding_dim),
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


def get_adaptation_obs(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Extract observations used for environment adaptation.
    
    This function extracts the observations that should be passed to the
    environment factor encoder. These typically include proprioceptive data
    and other state information that reveals environment properties.
    
    Args:
        env: The environment instance
        
    Returns:
        Tensor of shape [num_envs, obs_dim] containing adaptation observations
    """
    # This will be filled in with actual observation extraction logic
    # For now, we'll use the critic observations which include base_lin_vel
    # which provides privileged information about environment factors
    obs = env.observation_manager.get_observations()["critic"]
    return obs


def compute_adaptation_loss(
    encoded_factors: torch.Tensor,
    target_factors: torch.Tensor,
) -> torch.Tensor:
    """Compute the adaptation loss for the encoder.
    
    Args:
        encoded_factors: Encoded environment factors from encoder [batch_size, encoding_dim]
        target_factors: Ground truth environment factors [batch_size, encoding_dim]
        
    Returns:
        Scalar loss value
    """
    return torch.nn.functional.mse_loss(encoded_factors, target_factors)
