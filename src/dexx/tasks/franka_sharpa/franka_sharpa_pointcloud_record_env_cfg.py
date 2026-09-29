"""Config for `FrankaSharpaPointCloudRecordEnv`.

Inherits the pointcloud env cfg + adds a third-person Camera for video
recording.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from .franka_sharpa_pointcloud_env_cfg import FrankaSharpaPointCloudEnvCfg


@configclass
class RecordCameraCfg:
    """Third-person recording camera placement + intrinsics.

    Default pose: 0.8 m in front of arm base (env-local x=0.6), 0.5 m off to
    the side, 0.7 m above the table — looking back toward the manipulation
    region. Rotation `rot_wxyz` is in env-local frame and is set so the
    camera looks toward the workspace center.
    """
    height: int = 480
    width: int = 640
    data_types: tuple = ("rgb", "distance_to_image_plane")
    focal_length: float = 18.0
    horizontal_aperture: float = 24.0
    clipping_near: float = 0.05
    clipping_far: float = 5.0
    # Camera position + look-at target, both in env-local meters (env_origins
    # are at world z=0 so env-local ≈ world here). The env overrides the
    # default CameraCfg.offset with `camera.set_world_poses_from_view`
    # right after init, so we only need pos + target — no manual quaternion.
    #
    # Default = roughly the policy's depth-camera pose (a D455 eye-on-base
    # calibration; the live one is calib/camera_align/current.npy):
    #   cam_pos_in_arm_base = (1.230, -0.147, 0.637)
    #   arm_base_world      = (-0.1, 0, 0.415)  (dexx.deploy_config.ARM_BASE_POS)
    #   → cam_pos_world ≈ (1.13, -0.147, 1.05)
    # Look-at = manipulation region center on the table (z=table_top + 5mm).
    pos: tuple = (1.13, -0.147, 1.05)
    target: tuple = (0.30, 0.00, 0.45)


@configclass
class FrankaSharpaPointCloudRecordEnvCfg(FrankaSharpaPointCloudEnvCfg):
    record_camera: RecordCameraCfg = RecordCameraCfg()
