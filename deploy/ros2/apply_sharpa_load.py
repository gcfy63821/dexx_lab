#!/usr/bin/env python3
"""ROS2 backend, robot PC: push the Sharpa hand load to Franka FCI.

The arm controller compensates gravity with libfranka's load model, so the
hand's mass and centre of mass must be known to FCI. The reference setup set
them in Desk (an end-effector profile, auto-loaded at launch: m = 1.48 kg,
COM = (-0.00163, 0.0061, 0.04042) m in the flange frame). This script is the
alternative when Desk was not used: FCI rejects ``setLoad`` while a controller
runs, so it

    1. deactivates ``dexhand_joint_impedance_controller``
    2. calls ``/service_server/set_load`` (franka_msgs/srv/SetLoad)
    3. reactivates the controller

Run once after every ``ros2 launch franka_bringup dexhand_joint_impedance_controller.launch.py``,
in a shell with ROS2 Humble and the franka_ros2 workspace sourced:

    python deploy/ros2/apply_sharpa_load.py --profile Robot-Right.endeffector-profile.json
    python deploy/ros2/apply_sharpa_load.py --mass 1.48 --com -0.00163 0.0061 0.04042

Check the result:
    ros2 topic echo /franka_robot_state_broadcaster/robot_state --once | grep -A 6 inertia
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


DEFAULT_CONTROLLER = "dexhand_joint_impedance_controller"
DEFAULT_SET_LOAD_SERVICE = "/service_server/set_load"


def parse_profile(path: str) -> tuple[float, list[float], list[float]]:
    """Read mass + COM + inertia from a Desk-exported end-effector profile JSON.

    Expected JSON layout (matches Desk's export):
        {
          "inertial": {
            "mass": 1.48,
            "centerOfMass": {"x": ..., "y": ..., "z": ...},
            "inertia": {"x11": ..., "x12": ..., "x13": ...,
                        "x22": ..., "x23": ..., "x33": ...}
          },
          ...
        }
    Returns (mass, com[3], load_inertia[9_column_major]).
    """
    with open(path) as f:
        prof = json.load(f)
    inert = prof["inertial"]
    mass = float(inert["mass"])
    com = inert["centerOfMass"]
    com_xyz = [float(com["x"]), float(com["y"]), float(com["z"])]
    ix = inert["inertia"]
    x11, x12, x13 = float(ix["x11"]), float(ix["x12"]), float(ix["x13"])
    x22, x23 = float(ix["x22"]), float(ix["x23"])
    x33 = float(ix["x33"])
    # Build symmetric 3x3 → flatten column-major (libfranka convention).
    I = [
        [x11, x12, x13],
        [x12, x22, x23],
        [x13, x23, x33],
    ]
    load_inertia_col_major = [
        I[0][0], I[1][0], I[2][0],
        I[0][1], I[1][1], I[2][1],
        I[0][2], I[1][2], I[2][2],
    ]
    return mass, com_xyz, load_inertia_col_major


def _run(cmd: list[str], check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if check and rc != 0:
        print(f"  [exit {rc}]")
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=str, default=None,
                   help="Path to a Desk-exported endeffector-profile.json (mass, COM, inertia).")
    p.add_argument("--controller", type=str, default=DEFAULT_CONTROLLER)
    p.add_argument("--service", type=str, default=DEFAULT_SET_LOAD_SERVICE)
    p.add_argument("--mass", type=float, default=None,
                   help="Override mass (kg). If unset, read from --profile.")
    p.add_argument("--com", type=float, nargs=3, default=None,
                   help="Override COM in flange frame (x y z, in m).")
    p.add_argument("--no_reactivate", action="store_true",
                   help="Leave controller deactivated after set_load (for chained workflows).")
    args = p.parse_args()

    if args.mass is not None and args.com is not None:
        mass = float(args.mass)
        com = list(args.com)
        # Tiny default inertia if user only overrides mass/COM
        load_inertia = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
        print(f"[apply_load] using CLI overrides: mass={mass}, com={com}")
    else:
        if args.profile is None or not os.path.exists(args.profile):
            print(f"[apply_load] ERROR: pass --profile <file> (found: {args.profile}) "
                  f"or both --mass and --com")
            sys.exit(2)
        mass, com, load_inertia = parse_profile(args.profile)
        print(f"[apply_load] read profile from: {args.profile}")
        print(f"    mass = {mass} kg")
        print(f"    com  = {com}  (flange frame)")
        print(f"    load_inertia (col-major flat) = {load_inertia}")

    # 1) Deactivate controller
    rc = _run(["ros2", "control", "switch_controllers",
                "--deactivate", args.controller], check=False)
    if rc != 0:
        print("[apply_load] WARN: deactivate failed — controller may already be off.")
    time.sleep(1.0)

    # 2) set_load
    payload = (
        "{"
        f"mass: {mass}, "
        f"center_of_mass: [{com[0]}, {com[1]}, {com[2]}], "
        f"load_inertia: [{', '.join(str(v) for v in load_inertia)}]"
        "}"
    )
    print()
    rc = _run([
        "ros2", "service", "call", args.service,
        "franka_msgs/srv/SetLoad", payload,
    ])
    if rc != 0:
        print("[apply_load] ERROR: set_load service call failed.")
        if not args.no_reactivate:
            print("[apply_load] Trying to reactivate controller anyway...")
            _run(["ros2", "control", "switch_controllers",
                  "--activate", args.controller], check=False)
        sys.exit(3)

    time.sleep(1.0)

    # 3) Reactivate controller
    if not args.no_reactivate:
        rc = _run(["ros2", "control", "switch_controllers",
                    "--activate", args.controller], check=False)
        if rc != 0:
            print("[apply_load] WARN: reactivate failed.")
            sys.exit(4)

    print()
    print("=" * 60)
    print("Sharpa load applied to FCI.")
    print(f"  m_load = {mass} kg")
    print(f"  com    = {com}")
    print("Verify with:")
    print('  ros2 topic echo /franka_robot_state_broadcaster/robot_state '
          '--once | grep -A 6 "inertia_load"')
    print("=" * 60)


if __name__ == "__main__":
    main()
