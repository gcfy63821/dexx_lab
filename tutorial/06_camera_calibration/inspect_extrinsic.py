"""Inspect and compare camera extrinsics, without a camera.

A 4x4 camera-in-armbase transform in the ROS optical convention (x-right, y-down,
z-forward). This script validates one, and — more usefully — reports how far two
of them are apart.

That comparison is the whole lesson: two calibrations of the same camera can
sit centimetres and degrees apart without anything raising a warning. A
difference like that is obvious the moment you print it and invisible otherwise.

    python tutorial/06_camera_calibration/inspect_extrinsic.py
    python tutorial/06_camera_calibration/inspect_extrinsic.py a.npy b.npy
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate(name: str, M: np.ndarray) -> bool:
    ok = True
    if M.shape != (4, 4):
        print(f"  [FAIL] {name}: shape {M.shape}, expected (4, 4)")
        return False
    R = M[:3, :3]
    orth = np.abs(R @ R.T - np.eye(3)).max()
    det = float(np.linalg.det(R))
    bottom_ok = np.allclose(M[3], [0, 0, 0, 1], atol=1e-6)
    if orth > 1e-4:
        print(f"  [FAIL] {name}: rotation not orthonormal (max |RR^T - I| = {orth:.2e})")
        ok = False
    if abs(det - 1.0) > 1e-4:
        print(f"  [FAIL] {name}: det(R) = {det:.6f}, expected +1 "
              f"({'left-handed / mirrored' if det < 0 else 'scaled'})")
        ok = False
    if not bottom_ok:
        print(f"  [FAIL] {name}: bottom row is {M[3]}, expected [0 0 0 1]")
        ok = False
    return ok


def describe(name: str, M: np.ndarray, table_z: float, base_z: float) -> None:
    t = M[:3, 3]
    # The stored transform is camera-in-armbase; env-local adds the base height.
    print(f"  {name}")
    print(f"    camera position, arm-base frame : [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m")
    print(f"    camera height above the table   : {100*(t[2] + base_z - table_z):+.1f} cm")
    # +z of the optical frame is the viewing direction.
    fwd = M[:3, 2]
    print(f"    viewing direction               : [{fwd[0]:+.3f} {fwd[1]:+.3f} {fwd[2]:+.3f}]")
    pitch = np.degrees(np.arcsin(-fwd[2] / max(np.linalg.norm(fwd), 1e-9)))
    print(f"    downward tilt                   : {pitch:+.1f} deg")


def compare(a_name: str, A: np.ndarray, b_name: str, B: np.ndarray) -> None:
    dt = B[:3, 3] - A[:3, 3]
    R = A[:3, :3].T @ B[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    mag = np.linalg.norm(dt) * 100
    flag = ""
    if mag > 2.0 or ang > 1.0:
        flag = "   <-- these are different calibrations, not noise"
    print(f"  {a_name:34s} -> {b_name:34s}  "
          f"|dt|={mag:6.1f} cm  dtheta={ang:5.2f} deg{flag}")


def main() -> int:
    from dexx import deploy_config as dc
    table_z, base_z = float(dc.TABLE_SURFACE_Z), float(dc.ARM_BASE_Z)

    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob(os.path.join(ROOT, "calib", "camera_align", "*.npy")))
    if not paths:
        print("no extrinsics found; pass paths explicitly")
        return 1

    mats: dict[str, np.ndarray] = {}
    print("=== validate")
    ok = True
    for p in paths:
        name = os.path.basename(p)
        M = np.load(p).astype(np.float64)
        good = validate(name, M)
        ok &= good
        if good:
            mats[name] = M
            print(f"  [ok] {name}")

    print("\n=== describe")
    for name, M in mats.items():
        describe(name, M, table_z, base_z)

    if len(mats) > 1:
        print("\n=== drift between the shipped calibrations")
        names = list(mats)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                compare(names[i], mats[names[i]], names[j], mats[names[j]])

    print("\nPass one of these to training, evaluation and deploy with")
    print("  --camera_extrinsic calib/camera_align/<file>.npy")
    print("Omitting it falls back to the shipped current.npy and says so; a")
    print("missing file is an error, never a guess.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
