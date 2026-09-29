"""PointNet backbone + 3 fusion strategies for the policy network.

The encoder lives in the *policy network*, not the env. Env exposes raw point
clouds (variable-content, fixed-size with padding mask); the actor consumes
them and produces a single feature vector concatenated to proprio obs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PointNetBackbone(nn.Module):
    """Per-point MLP → masked max-pool → projection.

    Use LayerNorm so it tolerates variable point counts + small batches at
    deploy. Masked points are pushed to `-inf` before the max-pool so they
    can't dominate features.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 64, hidden: tuple = (64, 128, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.per_point = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden[-1], output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """x: (B, N, in_dim). mask: (B, N) bool (True = real point). → (B, out_dim)."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, in_dim), got {tuple(x.shape)}")
        feats = self.per_point(x)  # (B, N, hidden[-1])
        if mask is not None:
            # set masked features to -inf so max-pool ignores them
            feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            # if a whole batch is masked, max returns -inf — clamp to 0 to keep training stable
            pooled = feats.max(dim=1).values
            pooled = torch.where(
                torch.isfinite(pooled), pooled, torch.zeros_like(pooled)
            )
        else:
            pooled = feats.max(dim=1).values
        return self.proj(pooled)


class PointCloudEncoder(nn.Module):
    """Encode (scene, hand, tactile) point clouds → single feature vector.

    Three strategies:
        "early_concat"     : single PointNet over concatenated [xyz,type,force]
        "separate_encode"  : per-source PointNets, concat + linear projection
        "scene_only"       : scene PointNet only (ablation baseline)
    """

    SUPPORTED = ("early_concat", "separate_encode", "scene_only")

    def __init__(
        self,
        fusion_strategy: str = "early_concat",
        n_scene: int = 1024,
        n_hand: int = 11,
        n_tactile: int = 25,
        tactile_feat_dim: int = 1,
        output_dim: int = 64,
        ablate_tactile_pc: bool = False,
        type_repr: str = "scalar",
    ):
        super().__init__()
        if fusion_strategy not in self.SUPPORTED:
            raise ValueError(
                f"fusion_strategy={fusion_strategy!r} not in {self.SUPPORTED}"
            )
        if type_repr not in ("scalar", "onehot"):
            raise ValueError(f"type_repr={type_repr!r} must be 'scalar' or 'onehot'")
        self.fusion_strategy = fusion_strategy
        self.n_scene = int(n_scene)
        self.n_hand = int(n_hand)
        self.n_tactile = int(n_tactile)
        self.tactile_feat_dim = int(tactile_feat_dim)
        self.output_dim = int(output_dim)
        self.type_repr = type_repr
        self.type_dim = 1 if type_repr == "scalar" else 3
        # Modality ablation: when True, drop the tactile point block from the
        # unified cloud (early_concat) / skip the tactile sub-net (separate).
        # Dimension-preserving — the maxpool + projection output dim is unchanged.
        self.ablate_tactile_pc = bool(ablate_tactile_pc)

        if fusion_strategy == "early_concat":
            # Channels: 3 xyz + type_dim (1 scalar or 3 one-hot) + tactile_feat_dim force.
            in_dim = 3 + self.type_dim + self.tactile_feat_dim
            self.backbone = PointNetBackbone(input_dim=in_dim, output_dim=output_dim)
        elif fusion_strategy == "separate_encode":
            self.scene_net = PointNetBackbone(input_dim=3, output_dim=64)
            self.hand_net = PointNetBackbone(input_dim=3, output_dim=32)
            self.tactile_net = PointNetBackbone(
                input_dim=3 + tactile_feat_dim, output_dim=32
            )
            self.projection = nn.Sequential(
                nn.Linear(64 + 32 + 32, output_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_strategy == "scene_only":
            self.backbone = PointNetBackbone(input_dim=3, output_dim=output_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        scene_pc: torch.Tensor,
        scene_mask: torch.Tensor,
        hand_pc: torch.Tensor,
        tactile_pc: torch.Tensor,
        tactile_force: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All inputs batched.
        Args:
            scene_pc:      (B, n_scene, 3)
            scene_mask:    (B, n_scene) bool
            hand_pc:       (B, n_hand, 3)
            tactile_pc:    (B, n_tactile, 3)
            tactile_force: (B, n_tactile, tactile_feat_dim)
            tactile_mask:  (B, n_tactile) bool — optional. True = real point,
                           False = gated/masked. Defaults to all True.
        Returns:
            (B, output_dim)
        """
        # Build / sanitize tactile_mask (None = all True for backward compat).
        if tactile_mask is None:
            tactile_mask = torch.ones(
                scene_pc.shape[0], self.n_tactile,
                dtype=torch.bool, device=scene_pc.device,
            )

        if self.fusion_strategy == "scene_only":
            return self.backbone(scene_pc, scene_mask)

        if self.fusion_strategy == "separate_encode":
            sf = self.scene_net(scene_pc, scene_mask)
            hf = self.hand_net(hand_pc)
            if self.ablate_tactile_pc:
                # Skip the tactile sub-net's contribution but keep the final
                # projection's input dim unchanged (feed zeros for tf).
                tf = torch.zeros(
                    sf.shape[0],
                    self.tactile_net.proj[0].out_features,
                    device=sf.device,
                    dtype=sf.dtype,
                )
            else:
                tactile_in = torch.cat([tactile_pc, tactile_force], dim=-1)
                tf = self.tactile_net(tactile_in, tactile_mask)
            return self.projection(torch.cat([sf, hf, tf], dim=-1))

        # early_concat: build a unified (B, N_total, 3 + type_dim + force_dim) tensor.
        B = scene_pc.shape[0]
        device = scene_pc.device
        dt = scene_pc.dtype
        tdim = self.type_dim
        fdim = self.tactile_feat_dim

        def _type_for(kind: str, n: int) -> torch.Tensor:
            """Build (B, n, tdim) type embedding for {scene, hand, tactile}."""
            if self.type_repr == "scalar":
                val = {"scene": 0.0, "hand": 1.0, "tactile": 2.0}[kind]
                return torch.full((B, n, 1), val, device=device, dtype=dt)
            # one-hot: scene=[1,0,0], hand=[0,1,0], tactile=[0,0,1]
            idx = {"scene": 0, "hand": 1, "tactile": 2}[kind]
            t = torch.zeros((B, n, 3), device=device, dtype=dt)
            t[..., idx] = 1.0
            return t

        # Scene: type=scene, force=zeros
        scene_feat = torch.cat(
            [scene_pc,
             _type_for("scene", self.n_scene),
             torch.zeros(B, self.n_scene, fdim, device=device, dtype=dt)],
            dim=-1,
        )

        # Hand: type=hand, force=zeros
        hand_feat = torch.cat(
            [hand_pc,
             _type_for("hand", self.n_hand),
             torch.zeros(B, self.n_hand, fdim, device=device, dtype=dt)],
            dim=-1,
        )

        if self.ablate_tactile_pc:
            # Drop the tactile block entirely: unified cloud = scene + hand.
            # Dimension-preserving — maxpool + projection output dim unchanged.
            all_pts = torch.cat([scene_feat, hand_feat], dim=1)
            true_extra = torch.ones(
                B, self.n_hand, device=device, dtype=torch.bool
            )
            full_mask = torch.cat([scene_mask, true_extra], dim=1)
            return self.backbone(all_pts, full_mask)

        # Tactile: type=tactile, force=value. Force shape (B, n_tactile, fdim)
        # is passed through as-is; the backbone input_dim accounts for fdim.
        if tactile_force.shape[-1] != fdim:
            # Fallback: if cfg fdim mismatches actual env output, fall back to
            # magnitude when actual > fdim, or pad with zeros otherwise. Should not
            # happen in practice — cfg + env should agree.
            if tactile_force.shape[-1] > fdim:
                t_force = tactile_force.norm(dim=-1, keepdim=True)
            else:
                t_force = torch.zeros(B, self.n_tactile, fdim, device=device, dtype=dt)
        else:
            t_force = tactile_force.to(dtype=dt)
        tactile_feat = torch.cat(
            [tactile_pc, _type_for("tactile", self.n_tactile), t_force], dim=-1
        )

        all_pts = torch.cat([scene_feat, hand_feat, tactile_feat], dim=1)
        # Build a full mask: scene_mask + all-True for hand + tactile_mask.
        hand_true = torch.ones(B, self.n_hand, device=device, dtype=torch.bool)
        full_mask = torch.cat([scene_mask, hand_true, tactile_mask], dim=1)
        return self.backbone(all_pts, full_mask)
