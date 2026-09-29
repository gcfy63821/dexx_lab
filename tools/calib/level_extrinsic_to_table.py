#!/usr/bin/env python3
"""Level an extrinsic's height and tilt against the table plane.

Step 2 of the recommended procedure (tutorial/06), run on the same capture
session as step 1 (`calibrate_extrinsic_live_icp.py`).

Each reference constrains different degrees of freedom. ICP on the arm pins
all six, but with a single-viewpoint cloud matched against closed meshes it can
slide by a few tenths of a degree between iterations. The table constrains only
three (height and two tilts) but constrains them exactly, because sim puts its
top perfectly flat at `deploy_config.TABLE_SURFACE_Z`. So: keep yaw and
in-plane translation from ICP, and re-fit the remaining three against the
table. That is how the shipped `current.npy` was made.

The robot is excluded from the table fit by forward kinematics on the arm
joint angles recorded with each depth frame, not by asking the operator to move
it away: an arm parked over the table drags the plane fit and reads as tilt.
The capture holds no hand joint angles, so the hand is posed at its zero
(nominal) configuration and `--clearance` absorbs the finger deviation.

Usage:
    python tools/calib/level_extrinsic_to_table.py \\
        --init calib/camera_align/extrinsic_YYYYMMDD_icp.npy \\
        --session logs/calib_live \\
        --out calib/camera_align/extrinsic_YYYYMMDD_tablelevel.npy

Only valid when the real table top is where sim puts it: a mat or cloth on the
table adds its thickness to the fitted height.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


def _load_icp_tool():
    """tools/calib/calibrate_extrinsic_live_icp.py (tools/ is not a package)."""
    import importlib.util
    name = "calibrate_extrinsic_live_icp"
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_icp = _load_icp_tool()
ARM_BASE_POS, TABLE_Z = _icp.ARM_BASE_POS, _icp.TABLE_Z
backproject, cam_to_env, check_out_path = _icp.backproject, _icp.cam_to_env, _icp.check_out_path
fit_plane, in_table_search_box, tilt_deg = _icp.fit_plane, _icp.in_table_search_box, _icp.tilt_deg
intrinsics_from_args, rot_angle_deg = _icp.intrinsics_from_args, _icp.rot_angle_deg
sim_robot_points, urdf_path = _icp.sim_robot_points, _icp.urdf_path

# Hand joints used for the exclusion mask (the capture records arm joints only).
NOMINAL_HAND_Q = np.zeros(22)

# Levelling pivot (env-local): the workspace centre on the table top, so the
# lever arm to the far side of the table does not turn a small rotation into a
# large translation.
DEFAULT_PIVOT = (0.4, -0.1, TABLE_Z)


def table_points(pts: np.ndarray, arm: np.ndarray, clear: float) -> np.ndarray:
    """Points in the table search box farther than `clear` from any robot sample."""
    cand = pts[in_table_search_box(pts)]
    if not len(cand) or not len(arm):
        return cand
    from scipy.spatial import cKDTree
    d, _ = cKDTree(arm).query(cand, k=1)
    return cand[d > clear]


def rotation_onto_z(n: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector n onto +z (Rodrigues)."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    axis = np.cross(n, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), float(n[2])
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def level_extrinsic(T: np.ndarray, tbl_env: np.ndarray, table_z: float = TABLE_Z,
                    pivot=DEFAULT_PIVOT):
    """Level camera-in-armbase T so the env-local table points `tbl_env`
    (computed with T) lie flat at `table_z`.

    Returns (T_new, info). Only height and the two tilts change; the correction
    is a rotation about the pivot with no yaw component plus a vertical shift.
    """
    T = np.asarray(T, dtype=np.float64)
    tbl_env = np.asarray(tbl_env, dtype=np.float64)
    piv = np.asarray(pivot, dtype=np.float64)

    n, dz = fit_plane(tbl_env, table_z)
    R = rotation_onto_z(n)
    lev = (tbl_env - piv) @ R.T + piv
    dz2 = float(np.median(lev[:, 2] - table_z))
    lev[:, 2] -= dz2
    n2, dz3 = fit_plane(lev, table_z)

    # Correction in env-local: p' = R (p - piv) + piv - dz2 e_z. The extrinsic
    # lives in the arm-base frame (env-local minus ARM_BASE_POS), so the
    # translation also picks up (R - I) @ ARM_BASE_POS.
    base = ARM_BASE_POS.astype(np.float64)
    t_corr = piv - R @ piv - np.array([0.0, 0.0, dz2])
    T_new = np.eye(4)
    T_new[:3, :3] = R @ T[:3, :3]
    T_new[:3, 3] = R @ (T[:3, 3] + base) + t_corr - base
    info = dict(tilt_before_deg=tilt_deg(n), height_before_mm=dz * 1000,
                tilt_after_deg=tilt_deg(n2), height_after_mm=dz3 * 1000,
                rot_deg=rot_angle_deg(R),
                shift_mm=float(np.linalg.norm(T_new[:3, 3] - T[:3, 3]) * 1000))
    return T_new, info


def main():
    ap = argparse.ArgumentParser(
        description="Re-fit height and tilt of an extrinsic against the table plane.")
    ap.add_argument("--init", required=True,
                    help="extrinsic to level, normally the output of calibrate_extrinsic_live_icp.py solve")
    ap.add_argument("--session", required=True,
                    help="capture directory from calibrate_extrinsic_live_icp.py capture")
    ap.add_argument("--out", required=True,
                    help="NEW .npy for the result, e.g. "
                         "calib/camera_align/extrinsic_YYYYMMDD_tablelevel.npy (current.npy is refused)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out")
    ap.add_argument("--clearance", type=float, default=0.06,
                    help="m from any arm or hand mesh point for a point to count as table")
    ap.add_argument("--side", choices=("right", "left"), default="right",
                    help="hand mounted on the arm (selects fr3_with_<side>_sharpa_wave.urdf)")
    ap.add_argument("--sim_pts", type=int, default=120000)
    ap.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                    default=None,
                    help="back-projection intrinsics (default: deploy_config.SIM_INTRINSICS); "
                         "use the same values as the ICP step")
    ap.add_argument("--pivot", type=float, nargs=3, default=list(DEFAULT_PIVOT),
                    help="env-local levelling pivot (default: workspace centre on the table)")
    args = ap.parse_args()

    out = check_out_path(args.out, args.overwrite)
    intr = intrinsics_from_args(args.intrinsics)
    poses = sorted(glob.glob(os.path.join(args.session, "pose_*.npz")))
    if not poses:
        sys.exit(f"no captures in {args.session}")
    T = np.load(args.init).astype(np.float64)

    print(f"init = {args.init}\n{len(poses)} poses, excluding points within "
          f"{args.clearance*100:.0f} cm of the arm and {args.side} hand\n")
    urdf = urdf_path(args.side)
    tbl = []
    for p in poses:
        z = np.load(p)
        pts = cam_to_env(backproject(z["depth"], intr), T)
        robot = sim_robot_points(z["joints"], args.sim_pts, hand_q=NOMINAL_HAND_Q,
                                 with_normals=False, urdf=urdf)
        t = table_points(pts, robot, args.clearance)
        print(f"  {os.path.basename(p)}: {len(t)} table points (from {len(pts)} total)")
        tbl.append(t)
    tbl = np.concatenate(tbl)
    if len(tbl) < 2000:
        sys.exit(f"only {len(tbl)} table points survived — cannot level reliably")

    T_new, info = level_extrinsic(T, tbl, TABLE_Z, args.pivot)
    print(f"\nbefore: {len(tbl)} pts | tilt {info['tilt_before_deg']:.3f} deg | "
          f"height {info['height_before_mm']:+.2f} mm")
    print(f"after : tilt {info['tilt_after_deg']:.3f} deg | "
          f"height {info['height_after_mm']:+.2f} mm")
    print(f"\ncorrection applied: {info['rot_deg']:.3f} deg, "
          f"camera moved {info['shift_mm']:.1f} mm")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, T_new.astype(np.float32))
    print(f"saved -> {args.out}")
    print(np.array2string(T_new, precision=6, suppress_small=True))
    print("next: compare against the previous file with "
          "tutorial/06_camera_calibration/inspect_extrinsic.py")


if __name__ == "__main__":
    main()
