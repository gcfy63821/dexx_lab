# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
VisualRaycaster: BVH-based simulated depth camera for the point-cloud student.

Renders an `[N, H, W]` depth image with a `simple_raycaster.MultiMeshRaycaster`
pass over the scene meshes (arm, hand, table, object). Used by
`FrankaSharpaPointCloudEnv`, which back-projects the depth into the student's
point cloud. Supports robotool_batch data (`rt/...` indices).

Indexing layout in `MultiMeshRaycaster.meshes_wp` (must stay stable):

    [0 .. A-1]            arm links     (A ~= 8)
    [A .. A+H-1]          hand links    (H ~= 25, after VL/no-mesh filter)
    [A+H]                 table         (1)
    [A+H+1 .. A+H+U]      unique object meshes (U >= 1)

`mesh_indices` per env has shape `[N, A + H + 1 + 1]`: a fixed prefix of
arm+hand+table for every env, plus a final column with each env's object
mesh slot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

from dexx.tasks.hand_imitation.dataset.factory import ManipDataFactory
from dexx.tasks.franka_sharpa.franka_sharpa_env import rotmat_to_quat

if TYPE_CHECKING:
    from dexx.tasks.franka_sharpa.franka_sharpa_env import FrankaSharpaEnv


def _rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angle rotvec → (N, 4) wxyz quat. Safe for zero-length rv."""
    from isaaclab.utils.math import quat_from_angle_axis
    angle = torch.linalg.norm(rv, dim=-1)
    safe = angle.unsqueeze(-1).clamp_min(1e-8)
    axis = rv / safe
    return quat_from_angle_axis(angle, axis)


@dataclass
class _MeshLayout:
    """Bookkeeping for the stable mesh-index layout in the raycaster.

    Static prefix order: [arm | hand | table | curtains × n_curtain] then
    per-env objects. n_curtain may be 0 if curtains disabled.
    """

    n_arm: int = 0          # A
    n_hand: int = 0         # H
    table_idx: int = -1     # A + H
    n_curtain: int = 0      # C  (3 if enabled, 0 otherwise)
    curtain_start_idx: int = -1  # A + H + 1
    obj_start_idx: int = -1 # A + H + 1 + C
    n_obj_unique: int = 0   # U

    @property
    def n_static(self) -> int:
        """Meshes shared identically across envs (arm+hand+table+curtains)."""
        return self.n_arm + self.n_hand + 1 + self.n_curtain

    @property
    def n_total(self) -> int:
        """Total mesh slots in the raycaster."""
        return self.n_static + self.n_obj_unique

    @property
    def n_per_env(self) -> int:
        """Number of mesh slots queried per env (static prefix + 1 obj)."""
        return self.n_static + 1


class VisualRaycaster:
    """BVH-raycast depth camera attached to a FrankaSharpa env."""

    # USD subtree keywords (link prims have <link>/visuals/... and /collisions/...)
    _VISUAL_KEYS = ("/visuals", "/visual")
    _COLLISION_KEYS = ("/collisions", "/collision")

    # Body-name skip rules for filtering virtual / fingertip-only frames
    _SKIP_NAME_SUFFIX = ("_VL",)
    _SKIP_NAME_CONTAIN = ("fingertip",)  # frames, no geometry

    def __init__(self, env: "FrankaSharpaEnv"):
        self.env = env
        self.device: str = str(env.device)
        self.num_envs: int = int(env.num_envs)

        cfg = env.cfg
        # Depth image size
        self.height: int = int(getattr(cfg, "camera_height", 256))
        self.width: int = int(getattr(cfg, "camera_width", 256))
        self.n_rays: int = self.height * self.width

        # Depth normalization range (m)
        self.depth_min: float = float(getattr(cfg, "depth_min", 0.1))
        self.depth_max: float = float(getattr(cfg, "depth_max", 2.0))

        # Raycast hit clamps
        self.raycaster_min_dist: float = float(getattr(cfg, "raycaster_min_dist", 0.01))
        self.raycaster_max_dist: float = float(getattr(cfg, "raycaster_max_dist", 2.0))

        # Mesh decimation factors (0.0 disables; >0.0 == quadric_decimation strength)
        self.link_simplify_factor: float = float(
            getattr(cfg, "raycaster_link_simplify_factor", 0.0)
        )
        self.scene_simplify_factor: float = float(
            getattr(cfg, "raycaster_simplify_factor", 0.5)
        )

        # Camera intrinsics. Prefer explicit fx/fy/cx/cy on cfg (real-camera
        # calibration). Fall back to Isaac Sim's PinholeCameraCfg-style
        # focal_length / horizontal_aperture (square pixels + centered).
        self.focal_length: float = float(getattr(cfg, "camera_focal_length", 21.77))
        self.horizontal_aperture: float = float(
            getattr(cfg, "camera_horizontal_aperture", 36.0)
        )
        _fx = getattr(cfg, "camera_fx", None)
        _fy = getattr(cfg, "camera_fy", None)
        _cx = getattr(cfg, "camera_cx", None)
        _cy = getattr(cfg, "camera_cy", None)
        self.fx: float = (
            float(_fx) if _fx is not None
            else self.focal_length / self.horizontal_aperture * self.width
        )
        self.fy: float = float(_fy) if _fy is not None else self.fx
        self.cx: float = float(_cx) if _cx is not None else 0.5 * self.width
        self.cy: float = float(_cy) if _cy is not None else 0.5 * self.height
        import math
        _hfov_deg = math.degrees(2.0 * math.atan(0.5 * self.width / self.fx))
        _vfov_deg = math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))
        print(
            f"[VisualRaycaster] intrinsics: fx={self.fx:.2f}  fy={self.fy:.2f}  "
            f"cx={self.cx:.2f}  cy={self.cy:.2f}  W×H={self.width}×{self.height}\n"
            f"[VisualRaycaster] FoV: HFoV={_hfov_deg:.2f}°  VFoV={_vfov_deg:.2f}°  "
            f"(real D455 calib: HFoV≈79.4°, VFoV≈63.7°)"
        )

        # ----- handles populated in setup() -----
        self.raycaster = None
        self.layout: _MeshLayout = _MeshLayout()

        self._raycast_body_ids: Optional[torch.Tensor] = None
        self._table_pos_w_const: Optional[torch.Tensor] = None
        self._table_quat_w_const: Optional[torch.Tensor] = None
        self._curtain_pos_w_const: Optional[torch.Tensor] = None
        self._curtain_quat_w_const: Optional[torch.Tensor] = None
        self._mesh_indices: Optional[torch.Tensor] = None
        self._env_obj_mesh_idx: Optional[torch.Tensor] = None
        self._cam_pos_local: Optional[torch.Tensor] = None
        self._cam_quat_local: Optional[torch.Tensor] = None
        self._ray_dirs_local: Optional[torch.Tensor] = None

        # ---- Per-episode noise state (resampled when episode_length_buf == 0)
        # Camera extrinsic jitter (arm-base frame) + per-env depth bias (m).
        self._cam_pos_jitter: torch.Tensor = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32,
        )
        _q = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        _q[:, 0] = 1.0  # wxyz identity
        self._cam_quat_jitter: torch.Tensor = _q
        self._depth_bias_per_env: torch.Tensor = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32,
        )
        self._noise_warm_started: bool = False

        self._is_setup: bool = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Build the raycaster and cache per-env constants.

        Must be called after `env._setup_scene()` has spawned all prims.
        """
        if self._is_setup:
            return

        from simple_raycaster.raycaster import MultiMeshRaycaster
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        if stage is None:
            raise RuntimeError(
                "VisualRaycaster.setup(): USD stage is None. Did you call "
                "setup() before _setup_scene()?"
            )

        # 1) Robot link meshes (arm + hand) from env_0
        arm_meshes, arm_names, arm_body_ids = [], [], []
        hand_meshes, hand_names, hand_body_ids = [], [], []
        skipped_no_mesh: List[str] = []

        for body_idx, body_name in enumerate(self.env.hand.body_names):
            if any(body_name.endswith(sfx) for sfx in self._SKIP_NAME_SUFFIX):
                continue
            if any(kw in body_name for kw in self._SKIP_NAME_CONTAIN):
                continue
            prim_path = f"/World/envs/env_0/Robot/{body_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            mesh = self._extract_link_trimesh(prim)
            if mesh is None or len(mesh.vertices) == 0:
                skipped_no_mesh.append(body_name)
                continue
            if self.link_simplify_factor > 0.0:
                try:
                    mesh = mesh.simplify_quadric_decimation(self.link_simplify_factor)
                except Exception:  # pylint: disable=broad-except
                    pass
            if body_name.startswith("fr3_"):
                arm_meshes.append(mesh)
                arm_names.append(body_name)
                arm_body_ids.append(body_idx)
            else:
                hand_meshes.append(mesh)
                hand_names.append(body_name)
                hand_body_ids.append(body_idx)

        if len(arm_meshes) + len(hand_meshes) == 0:
            raise RuntimeError(
                "VisualRaycaster.setup(): no robot link meshes loaded. "
                "Check env_0 Robot prim path / mesh subtree predicate."
            )

        # 2) Table mesh (centered at origin; per-step pose feeds world placement)
        table_size = tuple(self.env.cfg.table_cfg.spawn.size)
        table_mesh = trimesh.creation.box(extents=table_size)

        # 2b) Curtain meshes (thin black walls forming a 3-sided backdrop).
        # Static, no collision; only show up in raycaster depth.
        curtain_meshes, curtain_local_poses = self._build_curtain_meshes()

        # 3) Unique object meshes (robotool_batch only)
        obj_meshes_unique, env_to_obj_local = self._load_unique_object_meshes()

        # 4) Construct the raycaster — order MUST match _MeshLayout below
        A = len(arm_meshes)
        H = len(hand_meshes)
        C = len(curtain_meshes)
        U = len(obj_meshes_unique)
        all_meshes = (
            arm_meshes + hand_meshes
            + [table_mesh]
            + curtain_meshes
            + obj_meshes_unique
        )
        self.raycaster = MultiMeshRaycaster(all_meshes, device=self.device)
        self.raycaster.initialize()

        self.layout = _MeshLayout(
            n_arm=A, n_hand=H, table_idx=A + H,
            n_curtain=C, curtain_start_idx=(A + H + 1) if C > 0 else -1,
            obj_start_idx=A + H + 1 + C, n_obj_unique=U,
        )

        # 5) Robot body-id lookup, in the same order as arm_meshes ++ hand_meshes
        self._raycast_body_ids = torch.tensor(
            arm_body_ids + hand_body_ids, dtype=torch.long, device=self.device
        )

        # 6) Table world-pose constants (env_origins[i] + cfg local pos)
        table_pos_local = torch.tensor(
            self.env.cfg.table_cfg.init_state.pos,
            device=self.device, dtype=torch.float32,
        )
        table_quat_local = torch.tensor(
            self.env.cfg.table_cfg.init_state.rot,
            device=self.device, dtype=torch.float32,
        )
        table_pos_w = self.env.scene.env_origins + table_pos_local[None]      # [N, 3]
        table_quat_w = table_quat_local[None].expand(self.num_envs, 4).contiguous()
        self._table_pos_w_const = table_pos_w.unsqueeze(1).contiguous()       # [N, 1, 3]
        self._table_quat_w_const = table_quat_w.unsqueeze(1).contiguous()     # [N, 1, 4]

        # 6b) Curtain world-pose constants (env_origins + curtain local pos)
        if C > 0:
            curtain_pos_local = torch.tensor(
                [p for p, _ in curtain_local_poses],
                device=self.device, dtype=torch.float32,
            )                                                                  # [C, 3]
            curtain_quat_local = torch.tensor(
                [q for _, q in curtain_local_poses],
                device=self.device, dtype=torch.float32,
            )                                                                  # [C, 4]
            curtain_pos_w = (
                self.env.scene.env_origins[:, None] + curtain_pos_local[None]
            )                                                                  # [N, C, 3]
            curtain_quat_w = curtain_quat_local[None].expand(
                self.num_envs, C, 4,
            ).contiguous()                                                     # [N, C, 4]
            self._curtain_pos_w_const = curtain_pos_w.contiguous()
            self._curtain_quat_w_const = curtain_quat_w.contiguous()
        else:
            self._curtain_pos_w_const = None
            self._curtain_quat_w_const = None

        # 7) mesh_indices: static prefix shared, last column per-env
        n_static = self.layout.n_static
        n_per_env = self.layout.n_per_env
        self._mesh_indices = torch.empty(
            (self.num_envs, n_per_env), dtype=torch.int64, device=self.device,
        )
        self._mesh_indices[:, :n_static] = torch.arange(
            n_static, device=self.device, dtype=torch.int64,
        )[None].expand(self.num_envs, -1)
        self._env_obj_mesh_idx = (env_to_obj_local + self.layout.obj_start_idx).to(
            device=self.device, dtype=torch.int64,
        )
        self._mesh_indices[:, -1] = self._env_obj_mesh_idx

        # 8) Camera extrinsics (cfg override, else the shipped calibration)
        self._cam_pos_local, self._cam_quat_local = self._build_camera_extrinsics()

        # 9) Pinhole ray dirs in camera local frame
        self._ray_dirs_local = self._build_ray_dirs_local()

        self._is_setup = True

        print(
            f"[VisualRaycaster] setup ok: "
            f"arm_links={A}, hand_links={H}, table=1, curtains={C}, unique_objects={U}, "
            f"total_meshes={A + H + 1 + C + U}, n_per_env={n_per_env}, "
            f"H={self.height}, W={self.width}, n_rays={self.n_rays}"
        )
        if skipped_no_mesh:
            print(
                f"[VisualRaycaster] skipped {len(skipped_no_mesh)} link(s) "
                f"with no mesh subtree (first 8): {skipped_no_mesh[:8]}"
            )

    # ------------------------------------------------------------------
    # Per-step render
    # ------------------------------------------------------------------
    def render_depth(self) -> torch.Tensor:
        """Cast rays once and return normalized `[N, H, W]` depth on device.

        Applies sim2real noise on training envs (unless cfg.enable_depth_noise=False
        or env._is_deploy_env=True):
          - Per-episode camera extrinsic jitter (pos + rot, resampled at reset)
          - Per-step Gaussian depth noise σ = a·z² + b·z (RealSense model)
          - Per-episode constant depth bias
          - Per-step per-pixel dropout (set to max range → far-clamp = 1.0)
        """
        if not self._is_setup:
            raise RuntimeError(
                "VisualRaycaster.render_depth called before setup(). "
                "Call setup() at the end of the env's __init__."
            )

        from isaaclab.utils.math import quat_apply, quat_mul

        N = self.num_envs
        R = self.n_rays
        H, W = self.height, self.width

        # ---- DR: resample per-episode noise for envs that just reset.
        # `episode_length_buf == 0` ↔ first render after reset.
        # at_reset_buf is cleared by compute_observations before this method.
        noise_on = (
            bool(getattr(self.env.cfg, "enable_depth_noise", True))
            and not bool(getattr(self.env, "_is_deploy_env", False))
        )
        if noise_on:
            just_reset = (
                self.env.episode_length_buf == 0
            ).nonzero(as_tuple=False).squeeze(-1)
            if just_reset.numel() > 0 or not self._noise_warm_started:
                ids = (
                    just_reset if self._noise_warm_started
                    else torch.arange(N, device=self.device)
                )
                self._resample_per_episode_noise(ids)
                self._noise_warm_started = True

        # (1) Mesh world poses — order matches _MeshLayout: arm | hand | table | curtains | obj
        robot_pos_w = self.env.hand.data.body_pos_w[:, self._raycast_body_ids]   # [N, A+H, 3]
        robot_quat_w = self.env.hand.data.body_quat_w[:, self._raycast_body_ids]  # [N, A+H, 4]

        obj_pos_w = self.env.object.data.root_pos_w.unsqueeze(1)    # [N, 1, 3]
        obj_quat_w = self.env.object.data.root_quat_w.unsqueeze(1)  # [N, 1, 4]

        pos_parts = [robot_pos_w, self._table_pos_w_const]
        quat_parts = [robot_quat_w, self._table_quat_w_const]
        if self._curtain_pos_w_const is not None:
            pos_parts.append(self._curtain_pos_w_const)
            quat_parts.append(self._curtain_quat_w_const)
        pos_parts.append(obj_pos_w)
        quat_parts.append(obj_quat_w)
        mesh_pos_w = torch.cat(pos_parts, dim=1)   # [N, n_per_env, 3]
        mesh_quat_w = torch.cat(quat_parts, dim=1)  # [N, n_per_env, 4]

        # (2) Camera world pose. Apply per-episode jitter in arm-base frame
        # BEFORE transforming to world. Pos jitter is additive in base frame;
        # rot jitter composes on the right (small rotation around camera's
        # current local orientation).
        arm_pos_w = self.env.hand.data.root_pos_w   # [N, 3]
        arm_quat_w = self.env.hand.data.root_quat_w  # [N, 4]

        cam_pos_local = self._cam_pos_local[None] + self._cam_pos_jitter           # [N, 3]
        cam_quat_local = quat_mul(
            self._cam_quat_local[None].expand(N, 4).contiguous(),
            self._cam_quat_jitter,
        )                                                                          # [N, 4]
        cam_pos_w = arm_pos_w + quat_apply(arm_quat_w, cam_pos_local)
        cam_quat_w = quat_mul(arm_quat_w, cam_quat_local)                          # [N, 4]

        # (3) Build ray starts/dirs in world frame
        ray_starts_w = cam_pos_w[:, None].expand(N, R, 3).contiguous()
        ray_dirs_w = quat_apply(
            cam_quat_w[:, None].expand(N, R, 4),
            self._ray_dirs_local.expand(N, R, 3),
        )

        # (4) Raycast against per-env mesh subset
        _, hit_dist = self.raycaster.raycast_fused(
            mesh_pos_w=mesh_pos_w,
            mesh_quat_w=mesh_quat_w,
            ray_starts_w=ray_starts_w,
            ray_dirs_w=ray_dirs_w,
            mesh_indices=self._mesh_indices,
            min_dist=self.raycaster_min_dist,
            max_dist=self.raycaster_max_dist,
        )

        # (5) Convert ray-direction euclidean distance → perpendicular z-depth
        # (`distance_to_image_plane`), matching real depth cameras.
        z_hat = self._ray_dirs_local[0, :, 2]  # [R]
        hit_dist = hit_dist * z_hat.unsqueeze(0)  # [N, R]

        # (5b) Per-step depth noise (Gaussian + bias + dropout) in METERS,
        # before clamp+normalize. Skipped on deploy / when noise disabled.
        depth_m = hit_dist.reshape(N, H, W)
        if noise_on:
            depth_m = self._apply_depth_noise(depth_m)

        # (6) Clamp + normalize to [0, 1]
        depth = torch.clamp(depth_m, self.depth_min, self.depth_max)
        depth = (depth - self.depth_min) / (self.depth_max - self.depth_min + 1e-8)
        return depth

    # ------------------------------------------------------------------
    # Noise helpers (sim2real DR)
    # ------------------------------------------------------------------
    def _resample_per_episode_noise(self, env_ids: torch.Tensor) -> None:
        """Sample new camera pose jitter + depth bias for the given env ids."""
        cfg = self.env.cfg
        n = env_ids.numel()
        if n == 0:
            return

        pos_std = float(getattr(cfg, "cam_pose_pos_noise", 0.0))
        if pos_std > 0:
            self._cam_pos_jitter[env_ids] = pos_std * torch.randn(
                (n, 3), device=self.device, dtype=torch.float32,
            )
        else:
            self._cam_pos_jitter[env_ids] = 0.0

        rot_std = float(getattr(cfg, "cam_pose_rot_noise", 0.0))
        if rot_std > 0:
            rv = rot_std * torch.randn((n, 3), device=self.device, dtype=torch.float32)
            self._cam_quat_jitter[env_ids] = _rotvec_to_quat(rv)
        else:
            self._cam_quat_jitter[env_ids] = 0.0
            self._cam_quat_jitter[env_ids, 0] = 1.0   # wxyz identity

        bias_std = float(getattr(cfg, "depth_const_bias_std", 0.0))
        if bias_std > 0:
            self._depth_bias_per_env[env_ids] = bias_std * torch.randn(
                (n,), device=self.device, dtype=torch.float32,
            )
        else:
            self._depth_bias_per_env[env_ids] = 0.0

    def _apply_depth_noise(self, depth_m: torch.Tensor) -> torch.Tensor:
        """Per-step Gaussian + constant bias + dropout. Operates in meters."""
        cfg = self.env.cfg
        a = float(getattr(cfg, "depth_noise_quadratic_coef", 0.0))
        b = float(getattr(cfg, "depth_noise_linear_coef", 0.0))
        if a > 0 or b > 0:
            sigma = a * depth_m * depth_m + b * depth_m
            depth_m = depth_m + sigma * torch.randn_like(depth_m)

        if torch.any(self._depth_bias_per_env != 0):
            depth_m = depth_m + self._depth_bias_per_env[:, None, None]

        drop_prob = float(getattr(cfg, "depth_dropout_prob", 0.0))
        if drop_prob > 0:
            drop_mask = torch.rand_like(depth_m) < drop_prob
            depth_m = torch.where(
                drop_mask,
                torch.full_like(depth_m, self.raycaster_max_dist),
                depth_m,
            )
        return depth_m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_link_trimesh(self, prim) -> Optional[trimesh.Trimesh]:
        """Walk subtree (instance-aware) and return combined link Trimesh.

        FR3 arm link USD layout has `visuals`/`collisions` as **instanceable
        Xforms**, with the actual `Mesh` prims inside USD prototypes (e.g.
        `/__Prototype_4/fr3_link0_visual/mesh`). The simple_raycaster walker
        does not recurse through instances on intermediate children, so we
        traverse with `Usd.TraverseInstanceProxies()` here and compute each
        mesh's transform in the link-local frame so that the BVH can place
        the mesh correctly per-step from `articulation.body_pos_w/body_quat_w`.

        Tries visuals → collisions → any (path-substring filtered) so we
        prefer the higher-fidelity visual mesh when available, falling back
        to collision geometry for links that ship only one of the two.
        """
        from pxr import Usd, UsdGeom

        time = Usd.TimeCode.Default()
        try:
            link_xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time)
            link_xform_inv = link_xform.GetInverse()
        except Exception:  # pylint: disable=broad-except
            return None

        def _collect(predicate) -> Optional[trimesh.Trimesh]:
            tris: List[trimesh.Trimesh] = []
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if p.GetTypeName() != "Mesh":
                    continue
                if predicate is not None and not predicate(p):
                    continue
                usd_mesh = UsdGeom.Mesh(p)
                verts = usd_mesh.GetPointsAttr().Get()
                face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
                face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
                if verts is None or face_indices is None:
                    continue
                verts = np.asarray(verts, dtype=np.float32)
                if face_counts is not None and any(int(c) != 3 for c in face_counts):
                    # Triangulate any non-tri polygon by fan-triangulation.
                    face_idx = np.asarray(face_indices, dtype=np.int64)
                    fc = np.asarray(face_counts, dtype=np.int64)
                    tris_indices = []
                    cursor = 0
                    for n in fc:
                        if n < 3:
                            cursor += int(n); continue
                        base = face_idx[cursor]
                        for k in range(1, int(n) - 1):
                            tris_indices.append(
                                [base, face_idx[cursor + k], face_idx[cursor + k + 1]]
                            )
                        cursor += int(n)
                    faces = np.asarray(tris_indices, dtype=np.int64)
                else:
                    faces = np.asarray(face_indices, dtype=np.int64).reshape(-1, 3)
                if faces.size == 0:
                    continue
                # Mesh world transform via instance-proxy is correct for both
                # uninstanced and prototype meshes.
                mesh_world = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(time)
                mesh_in_link = mesh_world * link_xform_inv  # column-major math
                m_np = np.array(mesh_in_link, dtype=np.float32).T  # USD col-major → row-major
                tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                tri.apply_transform(m_np)
                tris.append(tri)
            if not tris:
                return None
            combined: trimesh.Trimesh = trimesh.util.concatenate(tris)
            try:
                combined.merge_vertices()
            except Exception:  # pylint: disable=broad-except
                pass
            return combined

        def _path_pred(keys):
            def _pred(mesh_prim):
                p = str(mesh_prim.GetPath())
                return any(k in p for k in keys)
            return _pred

        for keys in (self._VISUAL_KEYS, self._COLLISION_KEYS, None):
            try:
                m = _collect(_path_pred(keys) if keys else None)
                if m is not None and len(m.vertices) > 0:
                    return m
            except Exception:  # pylint: disable=broad-except
                continue
        return None

    def _load_unique_object_meshes(
        self,
    ) -> Tuple[List[trimesh.Trimesh], torch.Tensor]:
        """Dedupe by (obj_mesh_path, obj_scale); apply scale to vertices."""
        obj_meshes_unique: List[trimesh.Trimesh] = []
        key_to_local_idx: Dict[Tuple[str, float], int] = {}

        env_to_obj_local = torch.empty(
            self.num_envs, dtype=torch.int64, device=self.device,
        )

        data_indices = self.env.data_indices
        if not data_indices:
            raise RuntimeError(
                "VisualRaycaster: env.data_indices is empty; cannot load object meshes."
            )

        for env_id in range(self.num_envs):
            data_idx = data_indices[env_id % len(data_indices)]
            ds_type = ManipDataFactory.dataset_type(data_idx)
            if "robotool" not in ds_type:
                raise NotImplementedError(
                    f"VisualRaycaster currently supports robotool_batch only; "
                    f"got data_idx={data_idx!r} (dataset_type={ds_type!r}). "
                    f"OakInk2 support is deferred to a follow-up."
                )
            dset = self.env.demo_dataset_dict[ds_type][data_idx]
            mesh_path = dset.get("obj_mesh_path", "")
            if not mesh_path or not os.path.exists(mesh_path):
                raise RuntimeError(
                    f"VisualRaycaster: obj_mesh_path missing for {data_idx!r} "
                    f"-> {mesh_path!r}"
                )
            scale = float(dset.get("obj_scale", 1.0))
            key = (mesh_path, scale)
            if key not in key_to_local_idx:
                m = trimesh.load(mesh_path, process=False)
                if isinstance(m, trimesh.Scene):
                    m = trimesh.util.concatenate(list(m.geometry.values()))
                if not isinstance(m, trimesh.Trimesh):
                    raise RuntimeError(
                        f"VisualRaycaster: failed to load Trimesh from {mesh_path}"
                    )
                if scale != 1.0:
                    m.apply_scale(scale)
                key_to_local_idx[key] = len(obj_meshes_unique)
                obj_meshes_unique.append(m)
            env_to_obj_local[env_id] = key_to_local_idx[key]

        return obj_meshes_unique, env_to_obj_local

    def _build_camera_extrinsics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the camera extrinsic and decode it into (pos[3], quat[4] WXYZ).

        Matrix = camera-in-arm-base transform (camera pose expressed in the FR3
        base frame), ROS optical convention (x-right, y-down, z-forward).

        Resolution order:
          cfg.camera_extrinsic_path   : path to a 4x4 .npy (highest priority)
          cfg.camera_extrinsic_matrix : a 4x4 array/list
          otherwise                   : the shipped calibration,
                                        dexx.deploy_config.default_camera_extrinsic()
                                        (calib/camera_align/current.npy)
        """
        cfg = self.env.cfg
        override = None
        ext_path = getattr(cfg, "camera_extrinsic_path", None)
        ext_mat = getattr(cfg, "camera_extrinsic_matrix", None)
        if ext_path is not None:
            override = np.load(ext_path).astype(np.float32)
            print(f"[raycaster] camera extrinsic overridden from {ext_path}:\n{override}")
        elif ext_mat is not None:
            override = np.asarray(ext_mat, dtype=np.float32)
            print("[raycaster] camera extrinsic overridden from cfg.camera_extrinsic_matrix")
        if override is not None:
            assert override.shape == (4, 4), (
                f"camera extrinsic must be 4x4, got {override.shape}")

        if override is not None:
            transform_matrix = override
        else:
            # No literal here. The shipped calibration is the fallback, and a
            # missing one is an error rather than a guess — see
            # dexx.deploy_config.default_camera_extrinsic.
            from dexx import deploy_config as _dcfg
            transform_matrix = _dcfg.default_camera_extrinsic()
            print("[raycaster] camera extrinsic: no override given, using the "
                  f"shipped {_dcfg.CAMERA_EXTRINSIC_DEFAULT_FILE}", flush=True)
        pos_np = transform_matrix[:3, 3]
        rot_np = transform_matrix[:3, :3]

        rot_t = torch.from_numpy(rot_np).float().unsqueeze(0)  # [1, 3, 3]
        quat_t = rotmat_to_quat(rot_t)[0].to(self.device)      # [4] WXYZ
        pos_t = torch.from_numpy(pos_np).float().to(self.device)  # [3]
        return pos_t, quat_t

    def _build_ray_dirs_local(self) -> torch.Tensor:
        """Pinhole rays in the ROS optical (x-right, y-down, z-fwd) frame.

        Uses self.fx, self.fy, self.cx, self.cy directly. Resolved in __init__
        from either explicit cfg overrides (real-camera calib) or Isaac Sim's
        focal_length / horizontal_aperture.
        """
        H, W = self.height, self.width
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy

        i_grid, j_grid = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        x = (j_grid + 0.5 - cx) / fx
        y = (i_grid + 0.5 - cy) / fy
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=-1)              # [H, W, 3]
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        return dirs.reshape(1, H * W, 3).contiguous()      # [1, R, 3]

    def _build_curtain_meshes(
        self,
    ) -> Tuple[List[trimesh.Trimesh], List[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]]:
        """Build 3 curtain wall meshes + their env-local poses.

        Layout: three thin boxes forming a U-shape around the robot, opening
        toward +x (camera side).

        Disable with `cfg.enable_curtains = False`. Tune via cfg fields:
            curtain_height, curtain_thickness,
            curtain_back_x, curtain_back_y_halflen,
            curtain_side_x_center, curtain_side_x_halflen, curtain_side_y.

        Returns (meshes, [(pos_xyz, quat_wxyz), ...]). Quats are identity.
        """
        cfg = self.env.cfg
        if not bool(getattr(cfg, "enable_curtains", True)):
            return [], []

        table_pos = self.env.cfg.table_cfg.init_state.pos
        table_size = self.env.cfg.table_cfg.spawn.size
        table_top_z = float(table_pos[2]) + 0.5 * float(table_size[2])

        height = float(getattr(cfg, "curtain_height", 1.2))
        thickness = float(getattr(cfg, "curtain_thickness", 0.04))
        back_x = float(getattr(cfg, "curtain_back_x", -0.75))
        side_y = float(getattr(cfg, "curtain_side_y", 0.55))
        side_x_center = float(getattr(cfg, "curtain_side_x_center", 0.20))
        side_x_halflen = float(getattr(cfg, "curtain_side_x_halflen", 1.30))
        back_y_halflen = float(getattr(cfg, "curtain_back_y_halflen", 0.85))

        center_z = table_top_z + 0.5 * height
        back_extents = (thickness, 2.0 * back_y_halflen, height)
        side_extents = (2.0 * side_x_halflen, thickness, height)

        meshes = [
            trimesh.creation.box(extents=back_extents),
            trimesh.creation.box(extents=side_extents),
            trimesh.creation.box(extents=side_extents),
        ]
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        local_poses = [
            ((back_x,          0.0,    center_z), identity_quat),   # back
            ((side_x_center, +side_y,  center_z), identity_quat),   # left
            ((side_x_center, -side_y,  center_z), identity_quat),   # right
        ]
        return meshes, local_poses
