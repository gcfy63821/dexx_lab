"""Train-time augmentation for point clouds.

Designed to be cheap (vectorized torch, no python loops). Outputs new tensors
— never modifies inputs in place — so the env can keep raw clouds for debug.
"""
from __future__ import annotations

import torch


class PointCloudAugmentation:
    def __init__(
        self,
        jitter_std: float = 0.005,
        dropout_ratio: float = 0.05,
        hand_noise_std: float = 0.002,
        force_noise_ratio: float = 0.2,
        force_dropout_prob: float = 0.10,
    ):
        self.jitter_std = float(jitter_std)
        self.dropout_ratio = float(dropout_ratio)
        self.hand_noise_std = float(hand_noise_std)
        self.force_noise_ratio = float(force_noise_ratio)
        self.force_dropout_prob = float(force_dropout_prob)

    def augment(
        self,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
        hand_pc: torch.Tensor,
        tactile_pc: torch.Tensor,
        tactile_force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns augmented copies of all five tensors."""
        scene_pc, scene_mask = self._aug_scene(scene_pc, scene_mask)
        hand_pc = self._aug_hand(hand_pc)
        tactile_force = self._aug_tactile_force(tactile_force)
        return scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force

    # ------------------------------------------------------------------
    def _aug_scene(
        self,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step Gaussian jitter (noise scales with z, RealSense model) +
        random per-point dropout that flips mask bits."""
        if self.jitter_std > 0:
            z = scene_pc[..., 2:3].abs()
            noise = (
                self.jitter_std
                * (1.0 + z)
                * torch.randn_like(scene_pc)
            )
            scene_pc = scene_pc + noise
        if self.dropout_ratio > 0:
            drop = torch.rand(scene_pc.shape[:2], device=scene_pc.device) < self.dropout_ratio
            scene_mask = scene_mask & (~drop)
            # zero out the points whose mask just dropped (for cleanliness;
            # the encoder uses the mask but defensive)
            scene_pc = scene_pc * scene_mask.unsqueeze(-1).float()
        return scene_pc, scene_mask

    def _aug_hand(self, hand_pc: torch.Tensor) -> torch.Tensor:
        if self.hand_noise_std <= 0:
            return hand_pc
        return hand_pc + self.hand_noise_std * torch.randn_like(hand_pc)

    def _aug_tactile_force(self, tactile_force: torch.Tensor) -> torch.Tensor:
        """Multiplicative noise on force magnitudes + occasional per-finger
        zero-out (sensor dropout simulation)."""
        if self.force_noise_ratio > 0:
            mult = 1.0 + self.force_noise_ratio * torch.randn_like(tactile_force)
            tactile_force = tactile_force * mult.clamp(min=0.0)
        if self.force_dropout_prob > 0:
            # Per (batch, finger) zero-out. Tactile points are 5 fingers × 5
            # taxels = 25 by default; we treat first dim as batch, second dim
            # as a flat index — apply per-point (cheap).
            drop = torch.rand_like(tactile_force[..., :1]) < self.force_dropout_prob
            tactile_force = torch.where(drop, torch.zeros_like(tactile_force), tactile_force)
        return tactile_force
