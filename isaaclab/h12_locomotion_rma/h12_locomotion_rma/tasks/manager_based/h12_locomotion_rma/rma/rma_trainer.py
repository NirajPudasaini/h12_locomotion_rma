# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RMA Training Module.

This module handles the training of the RMA components:
1. Environment factor encoder
2. Adaptation module
3. Base policy (standard RL)

Based on "RMA: Rapid Motor Adaptation for Legged Robots" (Kumar et al., 2021)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from h12_locomotion_rma.isaaclab.h12_locomotion_rma.h12_locomotion_rma.tasks.manager_based.h12_locomotion_rma.rma.rma_env_factor_encoder import EnvFactorEncoder
    from h12_locomotion_rma.isaaclab.h12_locomotion_rma.h12_locomotion_rma.tasks.manager_based.h12_locomotion_rma.rma.rma_adaptation_module import AdaptationModule


class RMATrainer:
    """Trainer for RMA components.
    
    This trainer manages the training of the environment factor encoder
    and adaptation module alongside the base policy training.
    """
    
    def __init__(
        self,
        env_factor_encoder: "EnvFactorEncoder",
        adaptation_module: "AdaptationModule",
        learning_rate: float = 1e-3,
        encoder_weight: float = 0.1,
        adaptation_weight: float = 0.1,
    ):
        """Initialize RMA trainer.
        
        Args:
            env_factor_encoder: Environment factor encoder network
            adaptation_module: Adaptation module network
            learning_rate: Learning rate for encoder and adaptation module
            encoder_weight: Weight for encoder loss in total loss
            adaptation_weight: Weight for adaptation loss in total loss
        """
        self.env_factor_encoder = env_factor_encoder
        self.adaptation_module = adaptation_module
        
        self.encoder_weight = encoder_weight
        self.adaptation_weight = adaptation_weight
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(
            env_factor_encoder.parameters(),
            lr=learning_rate,
        )
        self.adaptation_optimizer = optim.Adam(
            adaptation_module.parameters(),
            lr=learning_rate,
        )
        
    def compute_encoder_loss(
        self,
        adaptation_obs: torch.Tensor,
        target_factors: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for environment factor encoder.
        
        Args:
            adaptation_obs: Observations for adaptation
            target_factors: Ground truth environment factors
            
        Returns:
            Encoder loss
        """
        encoded_factors = self.env_factor_encoder(adaptation_obs)
        loss = torch.nn.functional.mse_loss(encoded_factors, target_factors)
        return loss
    
    def compute_adaptation_loss(
        self,
        base_action: torch.Tensor,
        env_factors: torch.Tensor,
        target_action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for adaptation module.
        
        The adaptation module should learn to modify base policy actions
        to better match optimal actions in different environments.
        
        Args:
            base_action: Actions from base policy
            env_factors: Encoded environment factors
            target_action: Optimal actions (from privileged training)
            
        Returns:
            Adaptation loss
        """
        adapted_action = self.adaptation_module(base_action, env_factors)
        loss = torch.nn.functional.mse_loss(adapted_action, target_action)
        return loss
    
    def update_encoder(
        self,
        adaptation_obs: torch.Tensor,
        target_factors: torch.Tensor,
    ) -> Dict[str, float]:
        """Update environment factor encoder.
        
        Args:
            adaptation_obs: Observations for adaptation
            target_factors: Ground truth environment factors
            
        Returns:
            Dictionary containing loss values
        """
        self.encoder_optimizer.zero_grad()
        loss = self.compute_encoder_loss(adaptation_obs, target_factors)
        loss.backward()
        self.encoder_optimizer.step()
        
        return {"encoder_loss": loss.item()}
    
    def update_adaptation(
        self,
        base_action: torch.Tensor,
        adaptation_obs: torch.Tensor,
        target_action: torch.Tensor,
    ) -> Dict[str, float]:
        """Update adaptation module.
        
        Args:
            base_action: Actions from base policy
            adaptation_obs: Observations for adaptation
            target_action: Optimal actions
            
        Returns:
            Dictionary containing loss values
        """
        self.adaptation_optimizer.zero_grad()
        
        # Encode environment factors
        env_factors = self.env_factor_encoder(adaptation_obs)
        
        # Compute adaptation loss
        loss = self.compute_adaptation_loss(
            base_action,
            env_factors.detach(),  # Stop gradient through encoder
            target_action,
        )
        
        loss.backward()
        self.adaptation_optimizer.step()
        
        return {"adaptation_loss": loss.item()}
    
    def update(
        self,
        adaptation_obs: torch.Tensor,
        target_factors: Optional[torch.Tensor] = None,
        base_action: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Unified update step for RMA components.
        
        Args:
            adaptation_obs: Observations for adaptation
            target_factors: Ground truth environment factors (optional)
            base_action: Base policy actions (optional)
            target_action: Optimal actions (optional)
            
        Returns:
            Dictionary containing all loss values
        """
        losses = {}
        
        # Update encoder if target factors provided
        if target_factors is not None:
            losses.update(self.update_encoder(adaptation_obs, target_factors))
        
        # Update adaptation module if targets provided
        if base_action is not None and target_action is not None:
            losses.update(
                self.update_adaptation(base_action, adaptation_obs, target_action)
            )
        
        return losses
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.encoder_optimizer.param_groups[0]["lr"]
    
    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate for all optimizers."""
        for optimizer in [self.encoder_optimizer, self.adaptation_optimizer]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
