#!/usr/bin/env python3
"""Turn an easy_handeye result into a camera-in-armbase 4x4 (initial extrinsic).

easy_handeye / easy_handeye2 (eye-on-base: camera fixed, marker on the arm)
estimates the camera pose in the robot base frame. This converts that result to
the `calib/camera_align/*.npy` convention the codebase uses: a 4x4 transform of
the *depth* optical frame (x right, y down, z forward) in `fr3_link0`.

It is an initial guess. Refine it with the recommended procedure in
tutorial/06 (multi-pose arm ICP, then level to the table): the marker-based
estimate is typically off by millimetres to a centimetre and a degree or so,
which the point-cloud crop and the policy notice.

Inputs accepted:
  * easy_handeye2 `.calib` (YAML with `parameters:` and `transform: translation /
    rotation`), as saved to ~/.ros2/easy_handeye2/calibrations/<name>.calib
  * easy_handeye (ROS1) `.yaml` (`eye_on_hand: false`, `transformation: {x, y, z,
    qx, qy, qz, qw}`), as saved to ~/.ros/easy_handeye/<name>_eye_on_base.yaml

Frames. easy_handeye reports `tracking_base_frame` in `robot_base_frame`:
  * robot_base_frame must be the arm base, `fr3_link0`, or a frame identical to
    it (`base` in franka_ros2; check with `ros2 run tf2_ros tf2_echo base fr3_link0`).
  * tracking_base_frame is usually the colour optical frame (the marker is
    detected in the RGB image), but the deploy back-projects the depth image.
    Pass the depth-to-colour extrinsic of your camera with --color_T_depth
    (the depth optical frame expressed in the colour optical frame) to convert;
    read it on the camera host with `--print_realsense_extrinsic`. Without it
    the colour frame is written as is and ICP has to absorb the offset
    (a few centimetres on a D455).
  * tracking_base_frame must be an optical frame (name containing "optical").
    A body frame such as `camera_link` is x-forward, not z-forward, so the
    result would be off by the body-to-optical rotation; it is refused unless
    --allow_non_optical is passed (then compose the body-to-optical rotation
    yourself before using the result).

Usage:
    python tools/calib/handeye_to_extrinsic.py --calib ~/.ros2/easy_handeye2/calibrations/fr3_d455.calib \\
        --color_T_depth <TX TY TZ QX QY QZ QW from --print_realsense_extrinsic> \\
        --out calib/camera_align/extrinsic_YYYYMMDD_handeye.npy

    # on the camera host (pyrealsense2 installed):
    python tools/calib/handeye_to_extrinsic.py --print_realsense_extrinsic
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ARM_BASE_FRAMES = {"fr3_link0", "base", "panda_link0"}


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_matrix(t, q_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(q_xyzw)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def load_handeye(path: str) -> tuple[np.ndarray, dict]:
    """Return (T_base_camera 4x4, metadata) from an easy_handeye(2) result file."""
    import yaml

    with open(path) as f:
        d = yaml.safe_load(f)
    if "transform" in d:  # easy_handeye2 .calib
        p = d.get("parameters", {})
        tr, rot = d["transform"]["translation"], d["transform"]["rotation"]
        T = pose_to_matrix([tr["x"], tr["y"], tr["z"]], [rot["x"], rot["y"], rot["z"], rot["w"]])
        meta = {"calibration_type": p.get("calibration_type"),
                "robot_base_frame": p.get("robot_base_frame"),
                "tracking_base_frame": p.get("tracking_base_frame")}
    elif "transformation" in d:  # easy_handeye (ROS1) .yaml
        tf = d["transformation"]
        T = pose_to_matrix([tf["x"], tf["y"], tf["z"]], [tf["qx"], tf["qy"], tf["qz"], tf["qw"]])
        meta = {"calibration_type": "eye_on_hand" if d.get("eye_on_hand") else "eye_on_base",
                "robot_base_frame": d.get("robot_base_frame"),
                "tracking_base_frame": d.get("tracking_base_frame")}
    else:
        raise ValueError(f"{path}: neither an easy_handeye2 .calib nor an easy_handeye .yaml")
    return T, meta


def convert(T_base_color: np.ndarray, color_T_depth: np.ndarray | None) -> np.ndarray:
    """Camera frame the deploy uses (depth optical) in the arm base."""
    return T_base_color if color_T_depth is None else T_base_color @ color_T_depth


def check_extrinsic(T: np.ndarray) -> None:
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or abs(np.linalg.det(R) - 1) > 1e-5:
        raise ValueError("rotation is not orthonormal with det +1")
    if not np.allclose(T[3], [0, 0, 0, 1]):
        raise ValueError("bottom row is not [0 0 0 1]")


def print_realsense_extrinsic() -> None:
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    try:
        depth = prof.get_stream(rs.stream.depth)
        color = prof.get_stream(rs.stream.color)
        ex = depth.get_extrinsics_to(color)  # maps a depth-frame point into the colour frame
        R = np.array(ex.rotation).reshape(3, 3).T  # librealsense stores it column-major
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, ex.translation
    finally:
        pipe.stop()
    # quaternion of R (w >= 0)
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w) if w > 1e-6 else 0.0
    y = (R[0, 2] - R[2, 0]) / (4 * w) if w > 1e-6 else 0.0
    z = (R[1, 0] - R[0, 1]) / (4 * w) if w > 1e-6 else 0.0
    t = T[:3, 3]
    print("depth optical frame in the colour optical frame (color_T_depth):")
    print(np.array2string(T, precision=6, suppress_small=True))
    print(f"--color_T_depth {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {x:.6f} {y:.6f} {z:.6f} {w:.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", help="easy_handeye2 .calib or easy_handeye .yaml result file")
    ap.add_argument("--color_T_depth", type=float, nargs=7, metavar=("TX", "TY", "TZ", "QX", "QY", "QZ", "QW"),
                    help="depth optical frame in the tracking (colour) optical frame; omit if the "
                         "calibration already tracked the depth frame")
    ap.add_argument("--out", help="output .npy (a new, dated file; never current.npy)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow_non_optical", action="store_true",
                    help="accept a tracking_base_frame whose name does not contain 'optical' "
                         "(e.g. camera_link, x-forward); the result is then NOT in the optical "
                         "convention the codebase expects")
    ap.add_argument("--print_realsense_extrinsic", action="store_true",
                    help="print the D4xx depth-to-colour extrinsic (needs pyrealsense2 and the camera)")
    args = ap.parse_args(argv)

    if args.print_realsense_extrinsic:
        print_realsense_extrinsic()
        return 0
    if not args.calib or not args.out:
        ap.error("--calib and --out are required")
    if os.path.basename(args.out) == "current.npy":
        ap.error("refusing to write current.npy: write a dated file and switch deliberately")
    if os.path.exists(args.out) and not args.overwrite:
        ap.error(f"{args.out} exists (pass --overwrite)")

    T_base_cam, meta = load_handeye(args.calib)
    if meta.get("calibration_type") not in (None, "eye_on_base"):
        ap.error(f"calibration_type is {meta['calibration_type']!r}; the camera must be fixed (eye_on_base)")
    base = meta.get("robot_base_frame")
    if base and base not in ARM_BASE_FRAMES:
        print(f"[handeye] WARNING: robot_base_frame {base!r} is not the arm base (fr3_link0); "
              f"compose with base->fr3_link0 first", file=sys.stderr)
    tracking = meta.get("tracking_base_frame") or "?"
    if tracking == "?":
        print("[handeye] WARNING: no tracking_base_frame in the result; assuming an optical frame "
              "(x right, y down, z forward)", file=sys.stderr)
    elif "optical" not in tracking and not args.allow_non_optical:
        ap.error(f"tracking_base_frame {tracking!r} is not an optical frame (x right, y down, "
                 f"z forward); calibrate against the camera's *_optical_frame, or pass "
                 f"--allow_non_optical and convert the result yourself")
    ctd = None if args.color_T_depth is None else pose_to_matrix(args.color_T_depth[:3], args.color_T_depth[3:])
    if ctd is None and "color" in tracking:
        print(f"[handeye] WARNING: tracking frame {tracking!r} is a colour frame and no --color_T_depth "
              f"was given; the result is the colour camera, ICP must absorb the offset", file=sys.stderr)

    T = convert(T_base_cam, ctd)
    check_extrinsic(T)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, T.astype(np.float32))
    view = T[:3, 2]
    print(f"[handeye] {args.calib}: {meta}")
    print(f"[handeye] camera at {np.round(T[:3, 3], 4)} m in fr3_link0, looking along {np.round(view, 3)}")
    print(f"[handeye] wrote {args.out}; refine it next (tutorial/06, recommended procedure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
