"""FrankaSharpaPointCloudEnv — point-cloud-flavored visual + force env.

Output of `_get_observations`:

    {
        "policy":         (N, obs_dim)              ← parent proprio + tactile flat obs
        "priv_info":      (N, priv_dim)             ← inherited
        "proprio_hist":   (N, hist_len, hist_dim)   ← inherited
        "scene_pc":       (N, n_scene, 3)           ← world or arm-base frame
        "scene_mask":     (N, n_scene) bool
        "hand_pc":        (N, n_hand, 3)
        "tactile_pc":     (N, n_tactile, 3)
        "tactile_force":  (N, n_tactile, tactile_feat_dim)
    }

Encoder lives in the *policy network*, not the env, so the env stays
agnostic to fusion strategy / output_dim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_mul

from .franka_sharpa_force_poseobs_env import FrankaSharpaForcePoseObsEnv
from .pointcloud import (
    DepthToPointCloud,
    PointCloudAugmentation,
)

if TYPE_CHECKING:
    from .franka_sharpa_pointcloud_env_cfg import FrankaSharpaPointCloudEnvCfg


# Per-fingertip surface offsets (5 points each, ~elastomer surface), in the
# elastomer's local frame. Front/back/left/right ±5mm + center.
_FINGERTIP_OFFSETS = torch.tensor(
    [
        [0.000,  0.000, 0.000],
        [0.005,  0.000, 0.000],
        [-0.005, 0.000, 0.000],
        [0.000,  0.005, 0.000],
        [0.000, -0.005, 0.000],
    ],
    dtype=torch.float32,
)  # (5, 3)


class FrankaSharpaPointCloudEnv(FrankaSharpaForcePoseObsEnv):
    cfg: "FrankaSharpaPointCloudEnvCfg"

    def __init__(
        self,
        cfg: "FrankaSharpaPointCloudEnvCfg",
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Visual raycaster for the scene depth ----
        from .visual_raycaster import VisualRaycaster
        self.visual_raycaster = VisualRaycaster(self)
        self.visual_raycaster.setup()

        # ---- depth → pointcloud helper ----
        self._depth2pc = DepthToPointCloud(
            height=int(cfg.camera_height),
            width=int(cfg.camera_width),
            fx=float(cfg.camera_fx),
            fy=float(cfg.camera_fy),
            cx=float(cfg.camera_cx),
            cy=float(cfg.camera_cy),
            depth_min=float(cfg.depth_min),
            depth_max=float(cfg.depth_max),
            max_points=int(cfg.pc_num_scene_points),
            device=self.device,
            accepts_normalized=True,
        )

        # ---- Augmentation ----
        self._pc_augmentation = PointCloudAugmentation(
            jitter_std=float(cfg.pc_jitter_std),
            dropout_ratio=float(cfg.pc_dropout_ratio),
            hand_noise_std=float(cfg.pc_hand_noise_std),
            force_noise_ratio=float(cfg.pc_force_noise_ratio),
            force_dropout_prob=float(cfg.pc_force_dropout_prob),
        )

        # ---- Hand body indices (11 by default, configurable) ----
        self._pc_hand_indices = self._build_hand_pc_indices()
        if len(self._pc_hand_indices) != int(cfg.pc_num_hand_points):
            # We want exactly pc_num_hand_points; truncate / pad as needed.
            # Truncate is enough; we never expect to pad above the body count.
            self._pc_hand_indices = self._pc_hand_indices[: int(cfg.pc_num_hand_points)]
            assert len(self._pc_hand_indices) == int(cfg.pc_num_hand_points), (
                f"Could not assemble {cfg.pc_num_hand_points} hand bodies; "
                f"got {len(self._pc_hand_indices)}."
            )

        # ---- Tactile: elastomer body ids + per-finger surface offset table ----
        # `self.elastomer_ids` (set in base env) is the list of 5 fingertip
        # elastomer body indices.
        self._pc_tactile_per_finger = int(cfg.pc_num_tactile_points) // len(self.elastomer_ids)
        if self._pc_tactile_per_finger * len(self.elastomer_ids) != int(cfg.pc_num_tactile_points):
            raise ValueError(
                f"pc_num_tactile_points={cfg.pc_num_tactile_points} must be divisible "
                f"by number of elastomers ({len(self.elastomer_ids)})"
            )
        if self._pc_tactile_per_finger != _FINGERTIP_OFFSETS.shape[0]:
            # If the user asks for ≠5 points per finger, we just resample / interpolate
            # the static offsets table to that count along axis 0. Cheap.
            base = _FINGERTIP_OFFSETS.shape[0]
            idx = torch.linspace(0, base - 1, self._pc_tactile_per_finger).long()
            self._tactile_offsets_local = _FINGERTIP_OFFSETS[idx].to(self.device)
        else:
            self._tactile_offsets_local = _FINGERTIP_OFFSETS.to(self.device)
        # cached shape (n_finger, n_per, 3)
        self._tactile_offsets_local = (
            self._tactile_offsets_local.unsqueeze(0)
            .expand(len(self.elastomer_ids), -1, -1)
            .contiguous()
        )

        # ---- Allocate output buffers ----
        nE = self.num_envs
        self.scene_pc = torch.zeros(
            nE, int(cfg.pc_num_scene_points), 3, device=self.device
        )
        self.scene_mask = torch.zeros(
            nE, int(cfg.pc_num_scene_points), dtype=torch.bool, device=self.device
        )
        self.hand_pc = torch.zeros(
            nE, int(cfg.pc_num_hand_points), 3, device=self.device
        )
        self.tactile_pc = torch.zeros(
            nE, int(cfg.pc_num_tactile_points), 3, device=self.device
        )
        self.tactile_force = torch.zeros(
            nE,
            int(cfg.pc_num_tactile_points),
            int(cfg.pc_tactile_feature_dim),
            device=self.device,
        )

        print(
            "[FrankaSharpaPointCloudEnv] PC dims: "
            f"scene={cfg.pc_num_scene_points} hand={cfg.pc_num_hand_points} "
            f"tactile={cfg.pc_num_tactile_points} | "
            f"fusion={cfg.pc_fusion_strategy} | "
            f"hand bodies: {[self.hand.body_names[i] for i in self._pc_hand_indices]}"
        )

    # ------------------------------------------------------------------
    # Body-index resolver
    # ------------------------------------------------------------------
    def _build_hand_pc_indices(self) -> list[int]:
        """Return up to `cfg.pc_num_hand_points` hand body indices.

        Priority:
          1. cfg.pc_hand_body_names (without side prefix) — if any is missing
             from hand_body_names, fall back to default.
          2. Default = evenly spaced subset of self.hand_body_indices
             so we get the wrist + first/middle/last of each finger.
        """
        cfg = self.cfg
        want_n = int(cfg.pc_num_hand_points)

        if cfg.pc_hand_body_names:
            side = self.hand_side
            full = [f"{side}_{n}" for n in cfg.pc_hand_body_names]
            if all(n in self.hand_body_names for n in full):
                return [self.hand.body_names.index(n) for n in full]
            print(
                f"[PointCloudEnv] cfg.pc_hand_body_names missing some bodies; "
                f"falling back to default ({full})"
            )

        # Default: pick `want_n` evenly spaced bodies out of hand_body_indices.
        if len(self.hand_body_indices) <= want_n:
            return list(self.hand_body_indices)
        step = max(1, len(self.hand_body_indices) // want_n)
        picks = [self.hand_body_indices[i * step] for i in range(want_n)]
        return picks

    # ------------------------------------------------------------------
    # Per-step pointcloud assembly
    # ------------------------------------------------------------------
    def _compute_scene_pc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Render depth → back-project ALL pixels → transform to env-local →
        workspace bbox crop → subsample to `pc_num_scene_points`.

        The bbox crop is what concentrates points on the manipulation region
        and away from table / curtain hits. Skipped if both workspace_min and
        workspace_max are None.
        """
        depth_norm = self.visual_raycaster.render_depth()       # (N, H, W) in [0, 1]
        pts_cam, valid_all = self._depth2pc.back_project(depth_norm)  # (N, H*W, 3), (N, H*W)

        # Camera world pose (replicating VisualRaycaster's logic).
        ray = self.visual_raycaster
        arm_pos_w = self.hand.data.root_pos_w
        arm_quat_w = self.hand.data.root_quat_w
        cam_pos_local = ray._cam_pos_local[None] + ray._cam_pos_jitter
        cam_quat_local = quat_mul(
            ray._cam_quat_local[None].expand(self.num_envs, 4).contiguous(),
            ray._cam_quat_jitter,
        )
        cam_pos_w = arm_pos_w + quat_apply(arm_quat_w, cam_pos_local)
        cam_quat_w = quat_mul(arm_quat_w, cam_quat_local)
        cam_rot_w = _quat_to_rotmat(cam_quat_w)

        if self.cfg.pc_in_world_frame:
            pts_w = self._depth2pc.transform_to_world(pts_cam, cam_pos_w, cam_rot_w)
            pts_local = pts_w - self.scene.env_origins.unsqueeze(1)
        else:
            pts_local = pts_cam

        # Workspace bbox crop (env-local frame, in meters). Drops table /
        # curtain hits before the random subsample so the 1024 final points
        # land on the manipulation region.
        ws_min = self.cfg.pc_workspace_min
        ws_max = self.cfg.pc_workspace_max
        if ws_min is not None and ws_max is not None:
            ws_min_t = torch.as_tensor(ws_min, device=self.device, dtype=pts_local.dtype)
            ws_max_t = torch.as_tensor(ws_max, device=self.device, dtype=pts_local.dtype)
            in_box = (
                (pts_local >= ws_min_t).all(dim=-1)
                & (pts_local <= ws_max_t).all(dim=-1)
            )
            valid_all = valid_all & in_box

        return self._depth2pc.subsample(
            pts_local, valid_all, self.cfg.pc_num_scene_points
        )

    def _compute_hand_pc(self) -> torch.Tensor:
        """11 hand body positions, env-local frame."""
        pos_w = self.hand.data.body_pos_w[:, self._pc_hand_indices]  # (N, 11, 3) world
        return pos_w - self.scene.env_origins.unsqueeze(1)

    def _compute_tactile_pc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 25, 3) per-finger surface points + (N, 25, F) per-point force.

        F = 1 when cfg.pc_tactile_use_vec3 is False (scalar magnitude).
        F = 3 when cfg.pc_tactile_use_vec3 is True (elastomer-local 3D vec).
        """
        # 5 elastomer link poses
        elastomer_pose = self.hand.data.body_link_state_w[:, self.elastomer_ids, :7]
        elastomer_pos = elastomer_pose[..., :3]                 # (N, 5, 3) world
        elastomer_quat = elastomer_pose[..., 3:7]               # (N, 5, 4) wxyz

        N = elastomer_pos.shape[0]
        F = len(self.elastomer_ids)
        P = self._tactile_offsets_local.shape[1]

        # Offsets in elastomer frame → world frame
        # (N, F, P, 3) by rotating offsets via per-finger quat then adding pos.
        off_local = self._tactile_offsets_local.unsqueeze(0).expand(N, -1, -1, -1)  # (N,F,P,3)
        # Flatten F*P for quat_apply, then unflatten.
        quat_rep = elastomer_quat.unsqueeze(2).expand(-1, -1, P, -1).reshape(-1, 4)
        off_flat = off_local.reshape(-1, 3)
        off_w = quat_apply(quat_rep, off_flat).reshape(N, F, P, 3)
        tac_pts_w = elastomer_pos.unsqueeze(2) + off_w
        tac_pts_local = tac_pts_w - self.scene.env_origins.unsqueeze(1).unsqueeze(2)
        tac_pts_local = tac_pts_local.reshape(N, F * P, 3)

        use_vec3 = bool(getattr(self.cfg, "pc_tactile_use_vec3", False))
        if not use_vec3:
            # Scalar magnitude path
            forces = self.last_contacts.unsqueeze(-1).expand(-1, -1, P)  # (N, F, P)
            forces = forces.reshape(N, F * P, 1)
            return tac_pts_local, forces

        # vec3 path: take world-frame 3D force from parent's last_contacts_vec_w
        # (set in _refresh_lab when present), rotate world→elastomer-local using
        # quat_apply_inverse with per-finger elastomer_quat.
        from isaaclab.utils.math import quat_apply_inverse
        if not hasattr(self, "last_contacts_vec_w"):
            # Parent didn't populate vec yet (e.g. before first step) — fall back to zeros.
            forces_vec_w = torch.zeros(N, F, 3, device=self.device)
        else:
            forces_vec_w = self.last_contacts_vec_w  # (N, F, 3) world
        # world → elastomer-local
        forces_vec_local = quat_apply_inverse(
            elastomer_quat.reshape(-1, 4), forces_vec_w.reshape(-1, 3)
        ).reshape(N, F, 3)
        # Replicate per-finger 3D vec to all P surface points of that finger.
        forces_vec_per_point = forces_vec_local.unsqueeze(2).expand(-1, -1, P, -1)  # (N, F, P, 3)
        forces_vec_per_point = forces_vec_per_point.reshape(N, F * P, 3)
        return tac_pts_local, forces_vec_per_point

    # ------------------------------------------------------------------
    # Obs hook
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()

        scene_pc, scene_mask = self._compute_scene_pc()
        hand_pc = self._compute_hand_pc()
        tactile_pc, tactile_force = self._compute_tactile_pc()

        if not getattr(self, "_is_deploy_env", False):
            scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force = (
                self._pc_augmentation.augment(
                    scene_pc, scene_mask, hand_pc, tactile_pc, tactile_force
                )
            )

        # --- Vision ablation: drop the camera-derived scene points ---
        # Done after augmentation so the aug pipeline still sees well-formed
        # inputs. Clearing scene_mask is what actually removes them from the
        # max-pool; zeroing the coordinates too keeps any downstream consumer
        # (debug viz, records) from reading stale geometry.
        if getattr(self.cfg, "pc_ablate_scene_pc", False):
            scene_pc = torch.zeros_like(scene_pc)
            scene_mask = torch.zeros_like(scene_mask)

        # --- Force scale normalization (sim-real consistent) ---
        # Applied BEFORE binary/ablate so threshold stays in raw N units.
        _scale = float(getattr(self.cfg, "pc_force_scale", 1.0))
        if _scale != 1.0:
            tactile_force = tactile_force / _scale

        # --- Modality ablation: force representation / zeroing ---
        _is_vec3 = bool(getattr(self.cfg, "pc_tactile_use_vec3", False))
        if getattr(self.cfg, "pc_force_repr", "scalar") == "binary":
            if _is_vec3:
                raise ValueError(
                    "pc_force_repr='binary' incompatible with pc_tactile_use_vec3=True; "
                    "binary semantics on vec3 is undefined. Pick one."
                )
            _thr = float(getattr(self.cfg, "contact_threshold", 0.2))
            tactile_force = (tactile_force > (_thr / _scale)).float()
        if getattr(self.cfg, "pc_ablate_tactile_force", False):
            tactile_force = torch.zeros_like(tactile_force)

        # --- Force-gated tactile PC: "no contact → no point" semantics ---
        # See pc_tactile_force_gate / pc_tactile_gate_mode in cfg.
        # `tactile_mask` is always present in obs_dict (all-True if gate==0 or
        # gate_mode=='zero', actual gating mask if gate_mode=='mask').
        _gate = float(getattr(self.cfg, "pc_tactile_force_gate", 0.0))
        _mode = str(getattr(self.cfg, "pc_tactile_gate_mode", "zero"))
        if _gate > 0.0:
            force_mag = tactile_force.abs().amax(dim=-1)                      # (N, n_tac)
            keep_bool = force_mag >= _gate                                    # (N, n_tac) bool
            if _mode == "zero":
                k = keep_bool.unsqueeze(-1).float()
                tactile_pc = tactile_pc * k
                tactile_force = tactile_force * k
                tactile_mask = torch.ones_like(keep_bool, dtype=torch.bool)
            elif _mode == "mask":
                # Leave xyz + force untouched; encoder will skip via mask.
                tactile_mask = keep_bool
            else:
                raise ValueError(
                    f"pc_tactile_gate_mode={_mode!r} must be 'zero' or 'mask'"
                )
        else:
            tactile_mask = torch.ones(
                tactile_force.shape[:2], dtype=torch.bool, device=tactile_force.device
            )

        # Mirror to env attrs (useful for debug / external access).
        self.scene_pc = scene_pc
        self.scene_mask = scene_mask
        self.hand_pc = hand_pc
        self.tactile_pc = tactile_pc
        self.tactile_force = tactile_force
        self.tactile_mask = tactile_mask

        obs_dict.update(
            {
                "scene_pc": scene_pc,
                "scene_mask": scene_mask,
                "hand_pc": hand_pc,
                "tactile_pc": tactile_pc,
                "tactile_force": tactile_force,
                "tactile_mask": tactile_mask,
            }
        )
        return obs_dict


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def _quat_to_rotmat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Batched wxyz quat → (B, 3, 3) rotation matrix."""
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError(f"expected (B, 4) wxyz, got {tuple(quat_wxyz.shape)}")
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]
    # Build rotation matrix.
    B = quat_wxyz.shape[0]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )
