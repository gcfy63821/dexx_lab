"""dexx.deploy_config — SINGLE SOURCE OF TRUTH for sim2real / deploy constants.

The constants shared by the env cfgs, the deploy env, the depth subscribers and
the retarget scripts are collected here so you only edit ONE file when your table
height, camera, or NUC bridge changes.

This module imports NOTHING from `dexx` (leaf module) — safe to import anywhere
without circular-import risk. Values below MUST match what the policy was trained
with; changing them changes runtime geometry.

Edit here, then everything downstream picks it up:
  - env cfgs (arm_base_pos, camera intrinsics, PC workspace crop)
  - deploy env (arm-base offset, intrinsics, workspace fallback)
  - depth subscribers (sim intrinsics)
  - retarget scripts (arm_base_pos, table height)
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Table & arm-base geometry — ENV-LOCAL frame, meters
# ─────────────────────────────────────────────────────────────────────────────
# Table top surface z in env-local. The manipulated object is placed relative to
# THIS plane (table-relative), independent of the arm base.
TABLE_SURFACE_Z: float = 0.415

# fr3_link0 (arm base) height in env-local. If your real base sits at a
# different height above the table, change THIS. 0.415 = base mounted level
# with the table surface.
#
# This is not a free parameter. Demonstrations store joint angles, so moving the
# base moves the hand by the same amount against an object that has not moved --
# 17 mm is 13-50% of the hand's 3.3-13.2 cm grasp aperture. Every retarget, every
# teacher and every policy trained against another value has to be redone, and
# the camera extrinsic re-measured. Do not change it to make a number look right.
ARM_BASE_Z: float = 0.415
ARM_BASE_POS: tuple = (-0.1, 0.0, ARM_BASE_Z)
ARM_BASE_ROT: tuple = (1.0, 0.0, 0.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Camera intrinsics — 320x240 (the sim/deploy PC back-projection resolution)
# D455 640x480 depth decimated x2 -> 320x240, whose intrinsics match these to <1.5px.
# ─────────────────────────────────────────────────────────────────────────────
SIM_INTRINSICS: dict = {"fx": 193.33, "fy": 193.06, "cx": 160.08, "cy": 121.05}
DEPTH_H: int = 240
DEPTH_W: int = 320

# ─────────────────────────────────────────────────────────────────────────────
# PointCloud workspace crop — ENV-LOCAL frame, meters. Recorded in every student
# checkpoint (pc_env_meta) and restored at eval/deploy.
#   z_min 0.417: 2 mm above the table top. At the table plane the table surface
#                takes most of the 1024 points; a few mm higher and the object's
#                base is cut off.
#   z_max 0.70:  above it most of the points are the robot's own arm, whose pose
#                the policy already has from forward kinematics.
# (Checkpoints without a recorded crop assume 0.420 / 1.30.)
# ─────────────────────────────────────────────────────────────────────────────
PC_WORKSPACE_MIN: tuple = (0.00, -0.40, TABLE_SURFACE_Z + 0.002)
PC_WORKSPACE_MAX: tuple = (0.80,  0.25, 0.70)

# ─────────────────────────────────────────────────────────────────────────────
# Deploy comm — Polymetis joint bridge (NUC) + camera depth publisher (ZMQ)
# ─────────────────────────────────────────────────────────────────────────────
POLYMETIS_STATE_PORT: int = 5560   # bridge PUB (joint/ee state)
POLYMETIS_CMD_PORT: int = 5561     # bridge PULL (joint targets)
# Example camera-host depth publisher addr; override per-run with --depth_zmq_addr.
CAMERA_ZMQ_ADDR_EXAMPLE: str = "tcp://<CAM_HOST>:5562"


def arm_base_pos_np() -> np.ndarray:
    """ARM_BASE_POS as a float32 numpy array (for the deploy PC offset)."""
    return np.array(ARM_BASE_POS, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Camera extrinsic — a FILE, not a constant
# ─────────────────────────────────────────────────────────────────────────────
# The camera-in-armbase 4x4 is a property of how the camera is bolted down, so it
# is calibrated per mount and lives in `calib/camera_align/`. It is deliberately
# not a literal in this file (tutorial/06 shows how to inspect and compare them).
#
# Always pass --camera_extrinsic explicitly for a run you intend to reproduce;
# this default only covers forgetting it.
CAMERA_EXTRINSIC_DEFAULT_FILE: str = "calib/camera_align/current.npy"


def _repo_root() -> str:
    # src/dexx/deploy_config.py -> src/dexx -> src -> <repo>
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_camera_extrinsic(required: bool = True) -> "np.ndarray | None":
    """Load the shipped camera-in-armbase 4x4 (ROS optical convention).

    Raises by default rather than returning a guess: a silently wrong camera
    pose produces a policy that trains happily and reaches for the wrong place.
    """
    path = os.path.join(_repo_root(), CAMERA_EXTRINSIC_DEFAULT_FILE)
    if not os.path.exists(path):
        if not required:
            return None
        raise FileNotFoundError(
            f"no camera extrinsic at {path}. Calibrate one (tutorial/06) or pass "
            f"--camera_extrinsic <file.npy>. There is deliberately no hard-coded "
            f"fallback.")
    T = np.load(path).astype(np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected a 4x4 matrix, got {T.shape}")
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Sharpa Wave hand SDK — installed on the inference host, not pip-installable
# ─────────────────────────────────────────────────────────────────────────────
SHARPA_SDK_ENV: str = "SHARPA_SDK_PYTHON"   # -> .../SharpaWaveSDK/python


def import_sharpa_sdk():
    """Import the Sharpa Wave SDK (`sharpa` package) for real-hand access.

    Uses an importable `sharpa` if there is one, otherwise the directory in
    $SHARPA_SDK_PYTHON. The package locates its own native libraries."""
    sdk = os.environ.get(SHARPA_SDK_ENV)
    if sdk and sdk not in sys.path:
        sys.path.insert(0, sdk)
    try:
        import sharpa
    except ImportError as exc:
        raise ImportError(
            f"Sharpa Wave SDK not importable. Set {SHARPA_SDK_ENV}=/path/to/SharpaWaveSDK/python "
            f"(see docs/DEPLOY.md)."
        ) from exc
    return sharpa
