# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Adaptation Module for RMA.

This module implements the adaptation component as described in:
"RMA: Rapid Motor Adaptation for Legged Robots" (Kumar et al., 2021)

The adaptation module takes the base policy output and environment factors,
and produces adapted actions that are better suited to the current environment.
This enables rapid online adaptation without retraining the policy.
"""

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class AdaptationModule(nn.Module):
    """Adaptation module for rapid motor adaptation.
    
    This module takes the base policy action and environment factor encoding
    and produces adapted actions. It can be trained to learn how to modify
    the base policy's actions based on environment characteristics.
    
    Args:
        action_dim: Dimension of the action space
        encoding_dim: Dimension of environment factor encoding
        hidden_dim: Dimension of hidden layers
    """
    
    def __init__(self, action_dim: int, encoding_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.action_dim = action_dim
        self.encoding_dim = encoding_dim
        
        # Network that learns how to adapt base policy actions
        self.adaptation_network = nn.Sequential(
            nn.Linear(action_dim + encoding_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(
        self,
        base_action: torch.Tensor,
        env_factors: torch.Tensor,
    ) -> torch.Tensor:
        """Produce adapted actions.
        
        Args:
            base_action: Actions from base policy [batch_size, action_dim]
            env_factors: Encoded environment factors [batch_size, encoding_dim]
            
        Returns:
            Adapted actions [batch_size, action_dim]
        """
        # Concatenate base action with environment factors
        combined = torch.cat([base_action, env_factors], dim=-1)
        
        # Compute action adjustment
        action_delta = self.adaptation_network(combined)
        
        # Return base action + adaptation (residual learning)
        return base_action + action_delta


class RMAPolicyWrapper(nn.Module):
    """Wrapper that combines base policy with adaptation module.
    
    This wrapper takes the base policy and adaptation module and provides
    a unified interface for inference. During testing, it can be used to
    rapidly adapt to new environments.
    
    Args:
        base_policy: The base policy network
        adaptation_module: The adaptation module
        env_factor_encoder: The environment factor encoder
    """
    
    def __init__(
        self,
        base_policy: nn.Module,
        adaptation_module: AdaptationModule,
        env_factor_encoder: nn.Module,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.adaptation_module = adaptation_module
        self.env_factor_encoder = env_factor_encoder
        
    def forward(
        self,
        obs: torch.Tensor,
        adaptation_obs: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get adapted policy action.
        
        Args:
            obs: Policy observations [batch_size, obs_dim]
            adaptation_obs: Observations for adaptation (if None, use obs)
            
        Returns:
            Adapted actions [batch_size, action_dim]
        """
        # Get base policy action
        with torch.no_grad():
            base_action = self.base_policy(obs)
        
        # Get environment factors
        if adaptation_obs is None:
            adaptation_obs = obs
        env_factors = self.env_factor_encoder(adaptation_obs)
        
        # Adapt actions
        adapted_action = self.adaptation_module(base_action, env_factors)
        
        return adapted_action
