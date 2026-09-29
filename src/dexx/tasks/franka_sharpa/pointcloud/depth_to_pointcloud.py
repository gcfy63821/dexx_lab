"""GPU depth → fixed-size point cloud.

The franka-sharpa visual stack normalizes depth to [0,1] inside the raycaster
(`VisualRaycaster.render_depth()`). We accept either:
  - normalized depth (and de-normalize internally using `depth_min` / `depth_max`)
  - raw meters (`accepts_normalized=False`)

Camera frame convention follows `VisualRaycaster._build_ray_dirs_local`:
    ROS optical — x-right, y-down, z-forward (into scene).
"""
from __future__ import annotations

import torch


class DepthToPointCloud:
    """Convert batched depth images to a per-batch fixed-size point cloud.

    Inputs from `VisualRaycaster.render_depth()` are normalized to `[0, 1]`
    via `(d - depth_min) / (depth_max - depth_min)`. We invert that and
    treat normalized values clamped to the endpoints as invalid (1.0 ↔
    `depth_max` is far-clamp, 0.0 ↔ `depth_min` is near-clamp / drop fill).
    """

    def __init__(
        self,
        height: int,
        width: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_min: float = 0.1,
        depth_max: float = 2.0,
        max_points: int = 1024,
        device: torch.device | str = "cuda:0",
        accepts_normalized: bool = True,
    ):
        self.H = int(height)
        self.W = int(width)
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.max_points = int(max_points)
        self.device = torch.device(device)
        self.accepts_normalized = bool(accepts_normalized)

        # Precompute pixel grid in ROS-optical convention (x right, y down,
        # z forward). u increases right, v increases down.
        v = torch.arange(self.H, dtype=torch.float32, device=self.device)
        u = torch.arange(self.W, dtype=torch.float32, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing="ij")
        # cache normalized (u - cx) / fx and (v - cy) / fy so per-batch
        # conversion is just one multiply by z.
        self._u_norm = ((u_grid + 0.5) - self.cx) / self.fx  # (H, W)
        self._v_norm = ((v_grid + 0.5) - self.cy) / self.fy  # (H, W)

    # ------------------------------------------------------------------
    def back_project(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-project depth → camera-frame points (B, H*W, 3) + valid mask
        (B, H*W) without subsampling. Use this when you need to post-filter
        points by world-frame position (e.g. workspace bbox crop) before the
        final subsample."""
        if depth.ndim != 3:
            raise ValueError(f"depth must be (B, H, W), got {tuple(depth.shape)}")
        B = depth.shape[0]
        if depth.shape[1] != self.H or depth.shape[2] != self.W:
            raise ValueError(
                f"depth shape ({depth.shape[1]}, {depth.shape[2]}) != "
                f"({self.H}, {self.W})"
            )
        depth = depth.to(self.device, dtype=torch.float32)

        if self.accepts_normalized:
            z = depth * (self.depth_max - self.depth_min) + self.depth_min
        else:
            z = depth

        x = self._u_norm.unsqueeze(0) * z
        y = self._v_norm.unsqueeze(0) * z
        pts = torch.stack([x, y, z], dim=-1).reshape(B, -1, 3)

        z_flat = z.reshape(B, -1)
        eps = 1e-3
        valid = (z_flat > self.depth_min + eps) & (z_flat < self.depth_max - eps)
        return pts, valid

    def convert(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-project depth then subsample to `self.max_points`. Convenience
        wrapper for callers that don't need world-frame post-filtering."""
        pts, valid = self.back_project(depth)
        return self.subsample(pts, valid, self.max_points)

    @staticmethod
    def subsample(
        pts: torch.Tensor,
        valid: torch.Tensor,
        max_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch random subsample to `max_points`. Pads with zeros + False mask."""
        B, N, _ = pts.shape
        device = pts.device
        # `torch.multinomial` needs probability weights. Use validity as the
        # weight; for batches with zero valid points we'd hit a NaN — so fall
        # back to uniform weights and the mask will mark all as False anyway.
        counts = valid.sum(dim=1)             # (B,)
        any_valid = counts > 0                # (B,)

        # Build weight matrix: where valid, weight=1; else 0 (or 1 if no valid).
        weights = valid.float()
        weights[~any_valid] = 1.0             # avoid all-zero rows

        # If any batch has < max_points valid, allow replacement = True.
        # We do per-batch sampling instead of one fused multinomial because
        # max_points may exceed N for some batches if their valid count is low.
        n_valid_min = int(counts.min().item()) if B > 0 else 0
        replacement = n_valid_min < max_points

        idx = torch.multinomial(
            weights, max_points, replacement=replacement
        )  # (B, max_points)

        pts_sampled = torch.gather(
            pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        # Final mask: True iff the sampled index was actually valid.
        mask_sampled = torch.gather(valid, 1, idx)

        # Zero out points that ended up padded so downstream encoders don't
        # see arbitrary positions.
        pts_sampled = pts_sampled * mask_sampled.unsqueeze(-1).float()

        return pts_sampled, mask_sampled

    # ------------------------------------------------------------------
    def transform_to_world(
        self,
        points: torch.Tensor,
        cam_pos: torch.Tensor,
        cam_rot_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Transform points from camera frame → world frame.

        Args:
            points:         (B, N, 3) in camera ROS-optical frame
            cam_pos:        (3,) or (B, 3) world position of the camera
            cam_rot_matrix: (3, 3) or (B, 3, 3) camera→world rotation matrix
        Returns:
            (B, N, 3) in world frame.
        """
        if cam_rot_matrix.ndim == 2:
            cam_rot_matrix = cam_rot_matrix.unsqueeze(0).expand(points.shape[0], -1, -1)
        if cam_pos.ndim == 1:
            cam_pos = cam_pos.unsqueeze(0).expand(points.shape[0], -1)
        # points_w = R @ points_c + t   (treat points as column vectors).
        # In batched form: (B, N, 3) @ R^T → (B, N, 3)
        return torch.bmm(points, cam_rot_matrix.transpose(1, 2)) + cam_pos.unsqueeze(1)
