"""Assert the frame invariants that hold regardless of where your robot is bolted.

Checks relationships, not values: re-mount the arm or raise the table and this
still passes. It fails when a constant has been re-coupled to another one, which
is the failure mode that does not announce itself.

Pure arithmetic — no Isaac Sim, runs in a second.
"""
from __future__ import annotations

import os
import sys

from dexx import deploy_config as dc

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        if detail:
            print(f"         {detail}")
        FAILURES.append(name)


def main() -> int:
    table = float(dc.TABLE_SURFACE_Z)
    base_z = float(dc.ARM_BASE_Z)
    base = tuple(float(x) for x in dc.ARM_BASE_POS)
    lo = tuple(float(x) for x in dc.PC_WORKSPACE_MIN)
    hi = tuple(float(x) for x in dc.PC_WORKSPACE_MAX)

    print(f"table={table}  arm_base_z={base_z}")
    print(f"workspace x {lo[0]}..{hi[0]}  y {lo[1]}..{hi[1]}  z {lo[2]}..{hi[2]}\n")

    print("=== the decoupling invariant")
    # The demonstration is placed relative to the TABLE, never the arm base.
    # If these are tied together, moving the robot drags the demo with it and the
    # grasp silently shifts.
    # The demo is lifted into the scene by the loader's mujoco2gym transform,
    # whose translation must be the table height in both the env and the retargeter.
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for rel in ("src/dexx/tasks/franka_sharpa/franka_sharpa_env.py", "scripts/retarget.py"):
        src = open(os.path.join(repo, rel)).read()
        check(f"demo placement in {rel} is anchored to TABLE_SURFACE_Z",
              "_dcfg.TABLE_SURFACE_Z" in src and "ARM_BASE_Z" not in src,
              "the mujoco2gym translation must read deploy_config.TABLE_SURFACE_Z; "
              "tying it to the arm base moves the demonstration when the arm is re-mounted")

    print("\n=== the crop box")
    check("crop floor is above the table top",
          lo[2] >= table,
          f"floor {lo[2]} <= table {table}: the table becomes scene geometry and "
          f"eats the point budget")
    check("crop floor is not more than 2 cm above the table",
          lo[2] - table <= 0.02,
          f"floor sits {100*(lo[2]-table):.1f} cm above the table; short objects "
          f"(3-5 cm tall) start getting clipped from below")
    check("crop box is non-degenerate on every axis",
          all(hi[i] > lo[i] for i in range(3)),
          f"{lo} .. {hi}")
    check("crop ceiling leaves at least 20 cm of headroom",
          hi[2] - table >= 0.20,
          f"only {100*(hi[2]-table):.0f} cm above the table; lifts get clipped")

    print("\n=== the arm base")
    rot = tuple(float(x) for x in dc.ARM_BASE_ROT)
    check("arm base rotation is identity (world->base is a pure translation)",
          abs(rot[0] - 1.0) < 1e-9 and max(abs(x) for x in rot[1:]) < 1e-9,
          f"ARM_BASE_ROT={rot}. Base-frame conversion assumes a translation "
          f"only; a rotated base needs every consumer revisited.")
    check("arm base sits at or above the table",
          base_z >= table - 1e-9,
          f"arm base {base_z} below table {table}")

    print("\n=== the depth camera")
    fx, fy = dc.SIM_INTRINSICS["fx"], dc.SIM_INTRINSICS["fy"]
    cx, cy = dc.SIM_INTRINSICS["cx"], dc.SIM_INTRINSICS["cy"]
    check("principal point lies inside the image",
          0 < cx < dc.DEPTH_W and 0 < cy < dc.DEPTH_H,
          f"({cx}, {cy}) vs {dc.DEPTH_W}x{dc.DEPTH_H}")
    check("principal point is near the image centre",
          abs(cx - dc.DEPTH_W / 2) < 0.1 * dc.DEPTH_W
          and abs(cy - dc.DEPTH_H / 2) < 0.1 * dc.DEPTH_H,
          f"({cx}, {cy}) vs centre ({dc.DEPTH_W/2}, {dc.DEPTH_H/2})")
    check("fx and fy agree to within 5%",
          abs(fx - fy) / max(fx, fy) < 0.05,
          f"fx={fx} fy={fy}: non-square pixels are possible but usually a sign "
          f"that colour intrinsics were pasted in place of depth ones")

    import math
    hfov = 2 * math.degrees(math.atan(dc.DEPTH_W / (2 * fx)))
    print(f"\n  horizontal FoV = {hfov:.1f} deg at {dc.DEPTH_W}x{dc.DEPTH_H}")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print("ALL FRAME INVARIANTS HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
