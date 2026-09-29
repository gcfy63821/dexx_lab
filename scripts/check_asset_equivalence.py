"""Compare a merged URDF's link frames against a reference USD, without Isaac.

Swapping a robot asset silently shifts frames, and everything downstream — the
retargeted demos, the arm-base height, the camera extrinsics — is calibrated
against the old geometry. This is the cheap guard: it runs in ~0.2 s and needs
no simulator, so there is no excuse for skipping it after an asset change.

    python scripts/check_asset_equivalence.py \
        --reference_usd /path/to/old/Right_final.usda --side right

The reference USD is expected to store each link as a prim under the
articulation root, transformed to the HOME pose (all joints at zero) — which is
what the Isaac URDF importer produces. The URDF side is evaluated analytically
by chaining joint ``<origin>`` transforms with every joint at zero.

Exit code is non-zero on mismatch, so it can gate a build.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GENERATED = os.path.normpath(os.path.join(_HERE, "..", "assets", "generated"))

POS_TOL_MM = 0.1
ROT_TOL_DEG = 0.01


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def urdf_fk_home(path: str) -> dict[str, np.ndarray]:
    """World transform of every link with all joints at zero."""
    root = ET.parse(path).getroot()
    links = {l.get("name") for l in root.findall("link")}
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        T = np.eye(4)
        T[:3, :3] = rpy_to_mat(*rpy)
        T[:3, 3] = xyz
        children.setdefault(j.find("parent").get("link"), []).append((j.find("child").get("link"), T))

    base = (links - {c for v in children.values() for c, _ in v}).pop()
    out = {base: np.eye(4)}
    stack = [base]
    while stack:
        cur = stack.pop()
        for child, T in children.get(cur, []):
            out[child] = out[cur] @ T
            stack.append(child)
    return out


def usd_home(path: str) -> dict[str, np.ndarray]:
    """Authored transform of every link prim under the reference USD's root."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    root = stage.GetDefaultPrim()
    out = {}
    for prim in root.GetChildren():
        x = UsdGeom.Xformable(prim)
        if not x or not x.GetOrderedXformOps():
            continue
        # Gf matrices are row-vector convention; transpose to column-vector.
        out[prim.GetName()] = np.array(
            x.GetLocalTransformation(Usd.TimeCode.Default()), dtype=float
        ).T
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="URDF vs reference-USD frame equivalence.")
    ap.add_argument("--reference_usd", required=True)
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--urdf", default=None, help="Defaults to the generated merged URDF for --side.")
    ap.add_argument("--pos_tol_mm", type=float, default=POS_TOL_MM)
    ap.add_argument("--rot_tol_deg", type=float, default=ROT_TOL_DEG)
    a = ap.parse_args()

    urdf = a.urdf or os.path.join(_GENERATED, f"fr3_with_{a.side}_sharpa_wave.urdf")
    A, B = urdf_fk_home(urdf), usd_home(a.reference_usd)

    common = sorted(set(A) & set(B))
    missing = sorted(set(B) - set(A))
    print("=" * 78)
    print(f"urdf      : {urdf}")
    print(f"reference : {a.reference_usd}")
    print(f"urdf links={len(A)}  reference links={len(B)}  comparable={len(common)}")
    if missing:
        print(f"*** in REFERENCE but not URDF ({len(missing)}): {missing}")

    rows = []
    for n in common:
        dp = float(np.linalg.norm(A[n][:3, 3] - B[n][:3, 3])) * 1000.0
        R = A[n][:3, :3] @ B[n][:3, :3].T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        rows.append((n, dp, ang))
    rows.sort(key=lambda r: -r[1])

    dmax = max(r[1] for r in rows)
    amax = max(r[2] for r in rows)
    print(f"position  max={dmax:.4f} mm  mean={np.mean([r[1] for r in rows]):.4f} mm")
    print(f"rotation  max={amax:.4f} deg")
    print("-" * 78)
    for n, d, ang in rows[:10]:
        print(f"  {n:30s} {d:10.4f} mm {ang:10.4f} deg")

    ok = not missing and dmax < a.pos_tol_mm and amax < a.rot_tol_deg
    print("=" * 78)
    print("VERDICT:", "EQUIVALENT" if ok else "*** MISMATCH ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
