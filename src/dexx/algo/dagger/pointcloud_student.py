"""PointCloud student actor (proprio + 3 point-cloud sources → action).

Plugs the `PointCloudEncoder` (from `tasks/franka_sharpa/pointcloud/`) under
a simple MLP that concatenates proprio obs with the encoder feature.

`act_inference` mirrors the `ActorCriticAsymmetric.act_inference` API so the
DAgger / PPO drivers can use a unified interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from dexx.tasks.franka_sharpa.pointcloud import PointCloudEncoder


class PointCloudStudent(nn.Module):
    """Actor: (proprio, scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force)
    → action. Deterministic mean-only inference path (matches PPO student
    `act_inference` convention which clips a sampled action; here we return
    the MLP output directly)."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        pc_encoder_cfg: dict,
        hidden: tuple = (512, 256),
        action_clip: float = 1.0,
    ):
        super().__init__()
        self.pc_encoder = PointCloudEncoder(**pc_encoder_cfg)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.action_clip = float(action_clip)

        mlp_in = proprio_dim + pc_encoder_cfg["output_dim"]
        layers: list[nn.Module] = []
        prev = mlp_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ELU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.mlp = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def encode(
        self,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
        hand_pc: torch.Tensor,
        tactile_pc: torch.Tensor,
        tactile_force: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.pc_encoder(
            scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force, tactile_mask
        )

    def forward(
        self,
        proprio: torch.Tensor,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
        hand_pc: torch.Tensor,
        tactile_pc: torch.Tensor,
        tactile_force: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pc_feat = self.encode(
            scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force, tactile_mask
        )
        x = torch.cat([proprio, pc_feat], dim=-1)
        return self.mlp(x)

    @torch.no_grad()
    def act_inference(self, obs_dict: dict) -> torch.Tensor:
        """Convenience: obs_dict keys must include 'obs' (proprio) plus the
        five PC tensors. Returns clamped action."""
        action = self.forward(
            obs_dict["obs"],
            obs_dict["scene_pc"],
            obs_dict["scene_mask"],
            obs_dict["hand_pc"],
            obs_dict["tactile_pc"],
            obs_dict["tactile_force"],
            obs_dict.get("tactile_mask", None),
        )
        return torch.clamp(action, -self.action_clip, self.action_clip)
