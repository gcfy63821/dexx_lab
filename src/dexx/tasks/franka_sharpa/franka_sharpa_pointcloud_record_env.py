"""FrankaSharpaPointCloudRecordEnv — pointcloud env + an extra third-person
RGB+depth Camera sensor for offline video recording.

This env is functionally identical to `FrankaSharpaPointCloudEnv` (same obs,
same action space, same physics), with one addition: per env, a USD Camera
prim is spawned at a fixed third-person viewpoint of the workspace, and a
`isaaclab.sensors.Camera` wraps it so we can read `rgb` and
`distance_to_image_plane` data each step.

Use case: re-rolling out a trained PointCloud policy and recording an RGB +
depth + scene_pc video for inspection. Not meant for training (RGB rendering
is GPU-heavy; this env should be used with `--num_envs=1..16`,
`--enable_cameras`, and `--headless`).

The camera frames are NOT injected into the policy obs dict — they are kept
on the env and read via `get_record_camera_outputs()`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

from .franka_sharpa_pointcloud_env import FrankaSharpaPointCloudEnv

if TYPE_CHECKING:
    from .franka_sharpa_pointcloud_record_env_cfg import (
        FrankaSharpaPointCloudRecordEnvCfg,
    )


class FrankaSharpaPointCloudRecordEnv(FrankaSharpaPointCloudEnv):
    cfg: "FrankaSharpaPointCloudRecordEnvCfg"

    def _setup_scene(self):
        # 1) Build the parent's scene (robot, table, curtains, objects, raycaster…).
        super()._setup_scene()

        # 2) Add the recording camera. We replicate one camera per env via the
        #    `prim_path` regex `/World/envs/env_.*/RecordCam`, matching the
        #    way Isaac Lab clones sensors across all envs. The CameraCfg.offset
        #    is a placeholder — we override the world pose right after init in
        #    `_apply_record_camera_pose()` using `set_world_poses_from_view`.
        rec = self.cfg.record_camera
        camera_cfg = CameraCfg(
            prim_path=f"{self.scene.env_regex_ns}/RecordCam",
            update_period=0.0,
            height=int(rec.height),
            width=int(rec.width),
            data_types=list(rec.data_types),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(rec.focal_length),
                focus_distance=400.0,
                horizontal_aperture=float(rec.horizontal_aperture),
                clipping_range=(float(rec.clipping_near), float(rec.clipping_far)),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(rec.pos),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        self.record_camera = Camera(cfg=camera_cfg)
        self.scene.sensors["record_camera"] = self.record_camera

    def _apply_record_camera_pose(self):
        """Aim the record camera at the configured look-at target.

        Called once after env init (in the viz script) so we don't have to
        compute the quaternion manually. Uses `set_world_poses_from_view`
        which takes camera position + target. Positions are env-local — we
        offset by `scene.env_origins` so each env's camera looks at its own
        workspace center.
        """
        rec = self.cfg.record_camera
        env_origins = self.scene.env_origins  # (N, 3) world
        device = env_origins.device
        cam_pos = env_origins + torch.tensor(rec.pos, device=device, dtype=env_origins.dtype)
        cam_tgt = env_origins + torch.tensor(rec.target, device=device, dtype=env_origins.dtype)
        self.record_camera.set_world_poses_from_view(cam_pos, cam_tgt)

    # ------------------------------------------------------------------
    # Helpers used by the offline viz script.
    # ------------------------------------------------------------------
    def get_record_camera_outputs(self, env_id: int = 0) -> dict:
        """Return the latest captured camera frames for `env_id`.

        Keys depend on `cfg.record_camera.data_types`. Typically:
            "rgb"                       (H, W, 4) uint8
            "distance_to_image_plane"   (H, W)    float32 meters
        """
        out = {}
        for k, v in self.record_camera.data.output.items():
            out[k] = v[env_id].detach().cpu()
        return out

    def step_record_camera(self):
        """Force the camera to render the current scene. Call AFTER physics
        step but BEFORE reading outputs. Safe to call multiple times.
        """
        self.record_camera.update(dt=float(self.physics_dt))
