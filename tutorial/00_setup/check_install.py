"""Verify the installation without starting Isaac Sim.

Deliberately does not import isaaclab: the point is to fail in one second on a
missing asset rather than twenty minutes into a training run. Every check names
the lesson that explains what the missing thing is for.
"""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKS: list[tuple[str, str, str]] = [
    # (relative path, what it is, lesson)
    ("src/dexx/deploy_config.py", "the sim2real constants", "01"),
    ("data/retargeting/robotool_batch/mano2sharpa_rh", "retargeted demonstrations", "03"),
    ("data/robotool_batch", "source demonstrations + object meshes", "03"),
    ("checkpoints/teacher_poseobs.pth", "the pretrained state expert", "04"),
    ("checkpoints/student_lean_v6_L1.pth", "the deployed lean student", "05"),
    ("calib/camera_align/current.npy", "the live camera extrinsic", "06"),
    ("scripts/train_dagger_pc.py", "the distillation entry point", "05"),
    ("scripts/eval.py", "the evaluation entry point", "08"),
    ("deploy/deploy_pc.py", "the real-robot runtime", "09"),
]


def main() -> int:
    ok = True

    print("=== files")
    for rel, what, lesson in CHECKS:
        p = os.path.join(ROOT, rel)
        good = os.path.exists(p)
        ok &= good
        print(f"  [{'ok' if good else 'MISSING'}] {rel}")
        if not good:
            print(f"        {what} — see tutorial/{lesson}_*/")

    print("\n=== package")
    try:
        dexx = importlib.import_module("dexx")
        print(f"  [ok] import dexx  ({os.path.dirname(dexx.__file__)})")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] import dexx: {e}")
        print("        run `pip install -e .` from the repository root")
        return 1

    print("\n=== constants (lesson 01)")
    try:
        from dexx import deploy_config as dc
        table = dc.TABLE_SURFACE_Z
        base = dc.ARM_BASE_Z
        lo, hi = dc.PC_WORKSPACE_MIN, dc.PC_WORKSPACE_MAX
        print(f"  table top     z = {table}")
        print(f"  arm base      z = {base}   (base is {100*(base-table):+.1f} cm vs table)")
        print(f"  workspace crop  = z {lo[2]} .. {hi[2]}")
        # The crop floor must sit above the table, or the table itself becomes
        # "the object" in every point cloud.
        good = lo[2] >= table
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] crop floor is at or above the table top")
        if not good:
            print(f"        crop floor {lo[2]} < table {table}: the table would be "
                  f"sampled as scene geometry. See tutorial/01_frames_and_constants/")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] reading deploy_config: {e}")

    print("\n=== camera calibrations available (lesson 06)")
    cdir = os.path.join(ROOT, "calib", "camera_align")
    if os.path.isdir(cdir):
        for f in sorted(x for x in os.listdir(cdir) if x.endswith(".npy")):
            print(f"  {f}")
    else:
        ok = False
        print("  [MISSING] calib/camera_align/")

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
