"""PointCloud deploy env — real-robot sibling of `FrankaSharpaPointCloudEnv`.

Builds the same obs dict the sim PC env produces:

    {
      "policy":        (1, 550)          ← V3 proprio + tactile obs (the sim env's
                                           557-d layout minus the obj_pose tail,
                                           which deployable students drop)
      "priv_info":     (1, priv_dim)     ← from parent (zeroed in deploy)
      "proprio_hist":  (1, hist_len, hist_dim)  ← from parent
      "scene_pc":      (1, n_scene, 3)   ← depth → unproject → crop → subsample
      "scene_mask":    (1, n_scene) bool
      "hand_pc":       (1, n_hand, 3)    ← from pk FK on hand bodies (env-local frame)
      "tactile_pc":    (1, n_tactile, 3) ← from pk FK on elastomer links + 5 offsets each
      "tactile_force": (1, n_tactile, 1) ← Sharpa SDK F6 per-finger force replicated to 5 points
    }

Arm: Polymetis (V2). The parent V3 runs pk FK on the real Franka + Sharpa state
every step, exposing `self._fk_body_pos_minus_env`; the hand and tactile points
reuse it.

DEPTH path: deploy_pc.py hands the env a depth source (the ZMQ subscriber to
the camera host) via `set_depth_source`. Each `_get_observations` pulls the
newest frame, back-projects with the sim intrinsics, moves it into env-local
with the calibrated camera-in-armbase extrinsic, crops to `pc_workspace_min/max`
and subsamples to `pc_num_scene_points`.

TACTILE_PC path: 5 elastomer link positions from pk FK, each with the 5 offsets
of `_FINGERTIP_OFFSETS` (same as sim) → 25 surface points. Force magnitudes come
from Sharpa SDK F6 (5 per-finger normal force scalars) replicated 5×.
"""
from __future__ import annotations

import numpy as np
import torch

from .franka_sharpa_force_critic_horizon_deploy_env_v3 import (
    FrankaSharpaForceCriticHorizonDeployEnvV3,
)

# Reuse the sim env's depth-to-PC class — identical math.
from .pointcloud.depth_to_pointcloud import DepthToPointCloud


# Default tactile per-fingertip surface offsets (5 points each in elastomer
# local frame). Copied 1:1 from `franka_sharpa_pointcloud_env._FINGERTIP_OFFSETS`
# so deploy tactile_pc geometry matches training.
_FINGERTIP_OFFSETS_LOCAL = torch.tensor(
    [
        [0.000,  0.000, 0.000],
        [0.005,  0.000, 0.000],
        [-0.005, 0.000, 0.000],
        [0.000,  0.005, 0.000],
        [0.000, -0.005, 0.000],
    ],
    dtype=torch.float32,
)  # (5, 3)


# All sim2real constants come from dexx.deploy_config (single source of truth).
from dexx import deploy_config as _dcfg

# Sim camera intrinsics — REAL must match for sim2real alignment.
_SIM_INTRINSICS = _dcfg.SIM_INTRINSICS
_DEPTH_H, _DEPTH_W = _dcfg.DEPTH_H, _dcfg.DEPTH_W
_DEPTH_MIN_M, _DEPTH_MAX_M = 0.1, 1.5  # match sim depth crop

# arm_base in env-local frame (sim convention). Real PC computed in arm-base
# frame is shifted by this to land in env-local (where sim PC lives).
_ARM_BASE_POS_IN_ENV_LOCAL = _dcfg.arm_base_pos_np()


class FrankaSharpaPointCloudDeployEnv(FrankaSharpaForceCriticHorizonDeployEnvV3):
    """PointCloud deploy env (task `franka-sharpa-pointcloud-polymetis-deploy`).

    Inherits proprio + Polymetis arm + Sharpa SDK + pk-FK pipeline from V3.
    Adds: scene/hand/tactile PC tensors and `_get_observations()` extension.

    Knobs the cfg must provide (copy from sim `FrankaSharpaPointCloudEnvCfg`):
      pc_num_scene_points (int), pc_num_hand_points (int), pc_num_tactile_points (int)
      pc_hand_body_names (list[str], without side prefix)
      pc_workspace_min, pc_workspace_max (list[float], env-local frame)
      pc_in_world_frame (bool) — kept True by default; matches sim
      pc_force_repr ("scalar" | "binary"), pc_ablate_tactile_pc, pc_ablate_tactile_force
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.device
        # ---- PC dimensions
        self._pc_n_scene = int(getattr(cfg, "pc_num_scene_points", 1024))
        self._pc_n_hand = int(getattr(cfg, "pc_num_hand_points", 11))
        self._pc_n_tactile = int(getattr(cfg, "pc_num_tactile_points", 25))
        n_elastomers = 5  # fixed Sharpa H4 hand
        self._pc_tactile_per_finger = self._pc_n_tactile // n_elastomers
        assert self._pc_tactile_per_finger * n_elastomers == self._pc_n_tactile, (
            f"pc_num_tactile_points ({self._pc_n_tactile}) must be divisible by 5"
        )

        # ---- Tactile offsets (5 per elastomer, in elastomer-local frame)
        # If pc_tactile_per_finger != 5, slice / loop.
        offs = _FINGERTIP_OFFSETS_LOCAL[: self._pc_tactile_per_finger].to(device)
        # Expand to (F=5, P=offs_per_finger, 3) for vectorized rotation later.
        self._tactile_offsets_local = (
            offs.unsqueeze(0).expand(n_elastomers, -1, -1).clone()
        )  # (5, P, 3)

        # ---- Hand body subset (indices into self.hand_body_names — parent V3
        # already builds hand_body_names from the URDF; we just need the
        # subset matching cfg.pc_hand_body_names).
        side = getattr(cfg, "hand_side", "right")
        want = list(getattr(cfg, "pc_hand_body_names", None) or [])
        if not want:
            # Fallback: take parent's first n_hand bodies
            want = [n.removeprefix(f"{side}_") for n in self.hand_body_names[: self._pc_n_hand]]
        # Resolve to indices in parent's hand_body_names list (with side prefix)
        full_names = [f"{side}_{n}" for n in want]
        idx_in_parent = []
        for n in full_names:
            if n in self.hand_body_names:
                idx_in_parent.append(self.hand_body_names.index(n))
            else:
                self.get_logger().warn(
                    f"[PCDeploy] pc_hand_body name '{n}' not in parent's hand_body_names; "
                    f"will zero-fill. Available: {self.hand_body_names[:5]}..."
                )
                idx_in_parent.append(-1)
        self._pc_hand_indices_in_parent = idx_in_parent  # list[int], -1 = missing
        assert len(self._pc_hand_indices_in_parent) == self._pc_n_hand, (
            f"hand body resolution mismatch: got {len(self._pc_hand_indices_in_parent)} "
            f"want {self._pc_n_hand}"
        )

        # ---- Elastomer body names (for tactile_pc FK lookup)
        self._elastomer_body_names = [
            f"{side}_{finger}_elastomer"
            for finger in ("thumb", "index", "middle", "ring", "pinky")
        ]
        # Index into parent V3's pk FK output (hand_body_names list). pk FK
        # produces results for every name in hand_body_names; elastomers should
        # be in there since they're URDF links.
        self._elastomer_indices_in_parent = []
        for n in self._elastomer_body_names:
            if n in self.hand_body_names:
                self._elastomer_indices_in_parent.append(self.hand_body_names.index(n))
            else:
                self.get_logger().warn(
                    f"[PCDeploy] elastomer body '{n}' missing from hand_body_names "
                    f"— tactile_pc for this finger will be zeros."
                )
                self._elastomer_indices_in_parent.append(-1)

        # ---- Output buffers (batch=1 for deploy)
        self.scene_pc = torch.zeros((1, self._pc_n_scene, 3), device=device, dtype=torch.float32)
        self.scene_mask = torch.zeros((1, self._pc_n_scene), device=device, dtype=torch.bool)
        self.hand_pc = torch.zeros((1, self._pc_n_hand, 3), device=device, dtype=torch.float32)
        self.tactile_pc = torch.zeros((1, self._pc_n_tactile, 3), device=device, dtype=torch.float32)
        self.tactile_force = torch.zeros(
            (1, self._pc_n_tactile, 1), device=device, dtype=torch.float32
        )

        # ---- Depth → PC converter (sim intrinsics; same class as sim env)
        self._d2pc = DepthToPointCloud(
            height=_DEPTH_H, width=_DEPTH_W,
            fx=_SIM_INTRINSICS["fx"], fy=_SIM_INTRINSICS["fy"],
            cx=_SIM_INTRINSICS["cx"], cy=_SIM_INTRINSICS["cy"],
            depth_min=_DEPTH_MIN_M, depth_max=_DEPTH_MAX_M,
            max_points=self._pc_n_scene,
            device=device, accepts_normalized=False,
        )
        # Camera mount extrinsic: set by deploy_pc.py (set_camera_extrinsic) to
        # the one the student was trained with; the shipped file is a default.
        _T = _dcfg.default_camera_extrinsic(required=False)
        self._T_cam_in_armbase = None if _T is None else torch.from_numpy(_T).to(device)
        self._arm_offset = torch.from_numpy(_ARM_BASE_POS_IN_ENV_LOCAL).to(device)
        self._ws_min = torch.tensor(list(cfg.pc_workspace_min), dtype=torch.float32, device=device)
        self._ws_max = torch.tensor(list(cfg.pc_workspace_max), dtype=torch.float32, device=device)

        # ---- Depth source (set externally — see deploy_pc.py)
        self._depth_source = None  # callable: () → fp32 (H, W) torch on device, or None
        self._depth_age = None     # callable: () → seconds since the newest frame

        self.get_logger().info(
            f"[PCDeploy] init done. n_scene={self._pc_n_scene} n_hand={self._pc_n_hand} "
            f"n_tactile={self._pc_n_tactile} ({self._pc_tactile_per_finger}/finger)"
        )
        # Make tactile-mode explicit so users see at startup which feature dim
        # the env will emit (must match what the ckpt was trained with).
        _use_vec3 = bool(getattr(cfg, "pc_tactile_use_vec3", False))
        _force_repr = getattr(cfg, "pc_force_repr", "scalar")
        _feat_dim = 3 if _use_vec3 else 1
        self.get_logger().info(
            f"[PCDeploy] tactile force mode: "
            f"use_vec3={_use_vec3}, force_repr={_force_repr!r} → "
            f"tactile_feat_dim={_feat_dim} (out shape (1, {self._pc_n_tactile}, {_feat_dim}))"
        )
        self.get_logger().info(
            f"[PCDeploy] hand bodies: "
            f"{[self.hand_body_names[i] if i >= 0 else '<missing>' for i in self._pc_hand_indices_in_parent]}"
        )

    # ------------------------------------------------------------------
    # External hooks — call after env init from deploy_pc.py
    # ------------------------------------------------------------------
    def set_depth_source(self, fn, age_fn=None):
        """fn() returns a (H, W) fp32 torch tensor (meters) on self.device, or
        None if no frame yet; called once per `_get_observations`. age_fn()
        returns the seconds since the newest frame arrived: with it, a camera
        that stops publishing trips the stale-sensor e-stop."""
        self._depth_source = fn
        self._depth_age = age_fn

    def _sensor_ages(self) -> dict:
        ages = super()._sensor_ages()
        if self._depth_age is not None and not getattr(self.cfg, "pc_ablate_scene_pc", False):
            ages["depth"] = float(self._depth_age())
        return ages

    def set_camera_extrinsic(self, T_4x4: np.ndarray):
        """Update the cam-in-armbase 4x4 transform. Call once during deploy init
        with calibrated value."""
        self._T_cam_in_armbase = torch.from_numpy(np.asarray(T_4x4, dtype=np.float32)).to(self.device)
        self.get_logger().info(f"[PCDeploy] camera extrinsic updated:\n{T_4x4}")

    # ------------------------------------------------------------------
    # PC computation
    # ------------------------------------------------------------------
    def _compute_scene_pc_real(self) -> tuple[torch.Tensor, torch.Tensor]:
        """RealSense depth → unproject → crop → subsample. Returns (1, N, 3), (1, N) bool."""
        # Vision ablation, as in the training env: a student trained with the
        # scene cloud zeroed must not be handed live points.
        if getattr(self.cfg, "pc_ablate_scene_pc", False):
            return (torch.zeros_like(self.scene_pc),
                    torch.zeros_like(self.scene_mask))
        if self._depth_source is None:
            return self.scene_pc, self.scene_mask  # zeros, no-op
        if self._T_cam_in_armbase is None:
            raise RuntimeError("no camera extrinsic: call set_camera_extrinsic() first")
        depth = self._depth_source()  # (H, W) or None
        if depth is None:
            return self.scene_pc, self.scene_mask
        depth = depth.to(self.device, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        with torch.no_grad():
            pts_cam, valid_raw = self._d2pc.back_project(depth)  # (1, H*W, 3), (1, H*W) bool
            cam_pos = self._T_cam_in_armbase[:3, 3]
            cam_rot = self._T_cam_in_armbase[:3, :3]
            pts_arm = self._d2pc.transform_to_world(pts_cam, cam_pos, cam_rot)
            pts_local = pts_arm + self._arm_offset
            in_box = (
                (pts_local >= self._ws_min).all(dim=-1)
                & (pts_local <= self._ws_max).all(dim=-1)
            )
            valid = valid_raw & in_box
            pts_sampled, mask_sampled = self._d2pc.subsample(pts_local, valid, self._pc_n_scene)
        return pts_sampled, mask_sampled

    def _compute_hand_pc_real(self) -> torch.Tensor:
        """Pull hand-body env-local positions from parent V3's pk FK output."""
        # Parent V3 fills self._fk_body_pos_minus_env every step in
        # _compute_fk() — shape (1, n_full_hand_bodies, 3). It's indexed by
        # self.hand_body_names. Subset to our pc_hand_body_names.
        fk_pos = getattr(self, "_fk_body_pos_minus_env", None)
        if fk_pos is None or fk_pos.shape[1] < len(self.hand_body_names):
            # Not yet populated — return zeros.
            return self.hand_pc
        out = torch.zeros_like(self.hand_pc)  # (1, n_hand, 3)
        for k, idx in enumerate(self._pc_hand_indices_in_parent):
            if 0 <= idx < fk_pos.shape[1]:
                out[0, k] = fk_pos[0, idx]
        return out.to(self.device)

    def _compute_tactile_pc_real(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 25 tactile surface points + per-finger force.

        Tactile point positions: 5 elastomer body env-local positions (from pk FK)
        + 5 local offsets each = 25 points. Without elastomer quaternion FK we
        skip the per-finger rotation and use offset in elastomer-local axis-aligned
        approximation (acceptable for ±5mm offsets).

        Force: pulled from parent's smoothed contact force `self.last_contacts`
        (shape (1, 5)) which `_refresh_lab` updates from Sharpa SDK F6.
        """
        fk_pos = getattr(self, "_fk_body_pos_minus_env", None)
        out_pts = torch.zeros_like(self.tactile_pc)  # (1, 25, 3)
        if fk_pos is not None:
            for f, idx in enumerate(self._elastomer_indices_in_parent):
                if not (0 <= idx < fk_pos.shape[1]):
                    continue
                center = fk_pos[0, idx]  # (3,) env-local
                # apply per-finger offsets (axis-aligned approx, no quat rotation)
                P = self._pc_tactile_per_finger
                pts_f = center.unsqueeze(0) + self._tactile_offsets_local[f, :P]  # (P, 3)
                out_pts[0, f * P:(f + 1) * P] = pts_f

        # ---- Force ----
        # Branch on cfg.pc_tactile_use_vec3 — must match training so the
        # PointNet per-point input dim (= 3 xyz + type_dim + tactile_feat_dim)
        # is what the encoder weights expect:
        #   scalar / binary  → tactile_feat_dim=1, shape (1, 25, 1)
        #   vec3             → tactile_feat_dim=3, shape (1, 25, 3)
        # Vec3 source: parent V3's `last_contacts_vec3` (1, 5, 3) in
        # elastomer-local frame, populated from Sharpa F6[:3].
        #
        # Scaling: V3.get_tactile_info already multiplied `last_contacts` /
        # `last_contacts_vec3` by `cfg.force_scale` (default 2.0) for the
        # proprio-force obs path. Sim PC training does NOT apply that — it
        # reads raw N from contact sensors. To match training distribution
        # for the PointNet tactile input, we UNDO V3's force_scale here, THEN
        # apply `pc_force_scale` (the PC-specific normalization divisor,
        # mirrors the force-scale normalization in franka_sharpa_pointcloud_env.py).
        P = self._pc_tactile_per_finger
        use_vec3 = bool(getattr(self.cfg, "pc_tactile_use_vec3", False))
        pc_scale = float(getattr(self.cfg, "pc_force_scale", 1.0))
        v3_force_scale = float(getattr(self.cfg, "force_scale", 1.0))

        if use_vec3:
            v = getattr(self, "last_contacts_vec3", None)
            if v is None or v.shape != (1, 5, 3):
                v = torch.zeros((1, 5, 3), device=self.device)
            v = v.to(self.device)                                       # (1, 5, 3)
            # Replicate per-finger vec3 to all P points of that finger.
            v_rep = v.unsqueeze(2).expand(-1, -1, P, -1)                # (1, 5, P, 3)
            out_force = v_rep.reshape(1, 5 * P, 3)                      # (1, 25, 3)
            # Undo V3's force_scale so PC tactile distribution matches sim training.
            if v3_force_scale != 1.0:
                out_force = out_force / v3_force_scale
            if pc_scale != 1.0:
                out_force = out_force / pc_scale
            # `binary` repr is undefined for vec3 (same as sim env policy).
            # If user accidentally sets both, ignore binary and log once.
            if getattr(self.cfg, "pc_force_repr", "scalar") == "binary" and not getattr(self, "_warned_vec3_binary", False):
                self.get_logger().warn(
                    "[PCDeploy] pc_tactile_use_vec3=True + pc_force_repr='binary' "
                    "is undefined — using vec3 as-is."
                )
                self._warned_vec3_binary = True
        else:
            contacts = getattr(self, "last_contacts", None)
            if contacts is not None and contacts.shape[-1] >= 5:
                f_per_finger = contacts[0, :5].to(self.device)          # (5,)
            else:
                f_per_finger = torch.zeros(5, device=self.device)
            f_rep = f_per_finger.unsqueeze(-1).expand(-1, P).reshape(-1)  # (25,)
            out_force = f_rep.unsqueeze(0).unsqueeze(-1)                # (1, 25, 1)
            # Undo V3's force_scale → match sim training raw-N units.
            if v3_force_scale != 1.0:
                out_force = out_force / v3_force_scale
            if pc_scale != 1.0:
                out_force = out_force / pc_scale
            if getattr(self.cfg, "pc_force_repr", "scalar") == "binary":
                # Same quantity and formula as the sim env
                # (franka_sharpa_pointcloud_env.py): threshold = cfg.contact_threshold
                # in raw-N units; `out_force` is raw N / pc_force_scale, so the
                # threshold is divided by pc_scale to put it on the same scale.
                thr = float(getattr(self.cfg, "contact_threshold", 0.2))
                out_force = (out_force > (thr / max(pc_scale, 1e-6))).float()

        if getattr(self.cfg, "pc_ablate_tactile_force", False):
            out_force = torch.zeros_like(out_force)

        return out_pts, out_force

    # ------------------------------------------------------------------
    # Obs hook
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()  # parent fills "policy" + "priv_info" + "proprio_hist"

        scene_pc, scene_mask = self._compute_scene_pc_real()
        hand_pc = self._compute_hand_pc_real()
        tactile_pc, tactile_force = self._compute_tactile_pc_real()

        # Ablation semantics MUST match sim env (franka_sharpa_pointcloud_env.py:
        # pc_ablate_tactile_pc ONLY zeros points; tactile_force is gated by the
        # separate `pc_ablate_tactile_force` flag, which `_compute_tactile_pc_real`
        # already handled). Zeroing force here too would conflate the two flags
        # (e.g. PC tactile branch OFF + proprio-force ON).
        if getattr(self.cfg, "pc_ablate_tactile_pc", False):
            tactile_pc = torch.zeros_like(tactile_pc)

        # Cache for debug / external read
        self.scene_pc = scene_pc
        self.scene_mask = scene_mask
        self.hand_pc = hand_pc
        self.tactile_pc = tactile_pc
        self.tactile_force = tactile_force

        obs_dict["scene_pc"] = scene_pc
        obs_dict["scene_mask"] = scene_mask
        obs_dict["hand_pc"] = hand_pc
        obs_dict["tactile_pc"] = tactile_pc
        obs_dict["tactile_force"] = tactile_force

        # ----------------------------------------------------------------
        # DEBUG: every 30 steps, summarise scene_pc so we can verify the
        # RealSense → workspace-cropped point cloud actually contains the
        # target object. All coords are env-local frame (sim convention).
        # Object center is read from the demo trajectory's current frame.
        # ----------------------------------------------------------------
        self._pc_debug_count = int(getattr(self, '_pc_debug_count', 0)) + 1
        if self._pc_debug_count == 1 or self._pc_debug_count % 30 == 0:
            try:
                import numpy as _np
                m = scene_mask[0].detach().cpu().numpy().astype(bool)
                pts = scene_pc[0].detach().cpu().numpy()[m]    # (M, 3) valid points only
                n_valid = int(m.sum())
                # Expected object position (env-local frame) from demo trajectory
                obj_pos_env = None
                try:
                    cur_idx = int(self._get_demo_idx()[0].item())
                    obj_T = self.demo_data["obj_trajectory"][0, cur_idx]  # (4,4) env-local
                    obj_pos_env = obj_T[:3, 3].detach().cpu().numpy()
                except Exception:
                    pass
                if pts.shape[0] > 0:
                    pc_lo = pts.min(axis=0); pc_hi = pts.max(axis=0)
                    pc_mean = pts.mean(axis=0)
                else:
                    pc_lo = pc_hi = pc_mean = _np.array([float('nan')] * 3)
                msg = (f'[PC-DBG #{self._pc_debug_count}] '
                       f'valid={n_valid}/{scene_pc.shape[1]} '
                       f'xyz_min=[{pc_lo[0]:+.3f},{pc_lo[1]:+.3f},{pc_lo[2]:+.3f}] '
                       f'xyz_max=[{pc_hi[0]:+.3f},{pc_hi[1]:+.3f},{pc_hi[2]:+.3f}] '
                       f'xyz_mean=[{pc_mean[0]:+.3f},{pc_mean[1]:+.3f},{pc_mean[2]:+.3f}]')
                if obj_pos_env is not None and pts.shape[0] > 0:
                    # Count points within ±10cm of expected object center.
                    R = 0.10
                    d = pts - obj_pos_env[None]
                    in_obj = int((_np.abs(d) < R).all(axis=1).sum())
                    msg += (f' obj_pos_env=[{obj_pos_env[0]:+.3f},{obj_pos_env[1]:+.3f},{obj_pos_env[2]:+.3f}] '
                            f'pts_within_10cm={in_obj}')
                self.get_logger().info(msg)
            except Exception as e:
                self.get_logger().warn(f'[PC-DBG] failed: {e}')

        return obs_dict
