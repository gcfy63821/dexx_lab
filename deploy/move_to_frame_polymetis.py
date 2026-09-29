#!/usr/bin/env python3
"""Move the real FR3 (via Polymetis on the NUC) to the SAME joint pose as one
frame of a robotool_batch retarget pkl — i.e. the exact pose rendered by
`tools/calib/gen_sim_frame_ply.py` for camera-extrinsic alignment.

The arm target is `opt_arm_joint_pos[frame]` (fr3_joint1..7, radians). Joint
angles are base-frame-invariant, so the arm-base placement in sim does NOT
change the commanded joint config — the real arm lands in the identical
configuration. Only the 7 arm joints move; the hand is left untouched.

Talks to the NUC via PolymetisArmClient (ZMQ -> polymetis_joint_bridge.py ->
RobotInterface joint impedance), identical transport to
`replay_motion_polymetis.py`.

Prereqs (NUC, polymetis-local env):
    conda activate polymetis-local && python deploy/polymetis_joint_bridge.py
    # and make sure NO other controller policy is running.

Usage (training PC, dexmanip env):
    python deploy/move_to_frame_polymetis.py \
        --ip <NUC_IP> \
        --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
        --frame 0 \
        --approach_time 6.0 --hold
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np
from dexx import deploy_config as _dcfg

sys.path.insert(0, ".")

ARM_JOINT_NAMES = ["fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
                   "fr3_joint5", "fr3_joint6", "fr3_joint7"]

# FR3 position limits (same as replay_motion_polymetis.py)
FR3_JOINT_POS_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
])


def load_target(pkl_path: str, frame: int) -> np.ndarray:
    d = pickle.load(open(pkl_path, "rb"))
    arm = np.asarray(d["opt_arm_joint_pos"], dtype=np.float64)
    n = arm.shape[0]
    if frame < 0:
        frame = n + frame
    if not (0 <= frame < n):
        raise ValueError(f"frame {frame} out of range [0, {n})")
    q = arm[frame]
    print(f"[move] pkl={pkl_path}")
    print(f"[move] frame={frame}/{n}  base={d.get('arm_base_pos')}")
    print(f"[move] target arm q = {q.round(4).tolist()}")
    return q


def check_limits(q: np.ndarray) -> list[str]:
    issues = []
    for j in range(7):
        lo, hi = FR3_JOINT_POS_LIMITS[j]
        if q[j] < lo + 0.02 or q[j] > hi - 0.02:
            issues.append(f"J{j+1}={q[j]:+.3f} outside [{lo:+.3f},{hi:+.3f}]")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True, help="NUC bridge IP (the Polymetis joint bridge host).")
    p.add_argument("--state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
    p.add_argument("--cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT)
    p.add_argument("--pkl", required=True, type=str)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--approach_time", type=float, default=6.0,
                   help="seconds to interpolate from current pose to target")
    p.add_argument("--ctrl_freq", type=float, default=30.0)
    p.add_argument("--max_start_delta", type=float, default=1.2,
                   help="abort if current pose is farther than this (rad, max joint) "
                        "from the target — sanity guard against a wild move")
    p.add_argument("--kq", type=str, default=None, help="JSON list of 7 Kq (else Polymetis default)")
    p.add_argument("--kqd", type=str, default=None, help="JSON list of 7 Kqd")
    p.add_argument("--hold", action="store_true",
                   help="keep holding the target (republish) until Ctrl+C — needed so "
                        "the arm stays put while you capture / align the camera")
    p.add_argument("--dry_run", action="store_true", help="print target + safety only, no motion")
    p.add_argument("--force_unsafe", action="store_true")
    args = p.parse_args()

    q_target = load_target(args.pkl, args.frame)

    issues = check_limits(q_target)
    if issues:
        print("[PREFLIGHT] target near/over FR3 limits:")
        for i in issues:
            print("   -", i)
        if not args.force_unsafe:
            print("[ABORT] refusing (pass --force_unsafe to override).")
            return
    else:
        print("[PREFLIGHT] target within FR3 joint limits.")

    if args.dry_run:
        print("[dry_run] not connecting / not moving.")
        return

    kq = json.loads(args.kq) if args.kq else None
    kqd = json.loads(args.kqd) if args.kqd else None

    from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import PolymetisArmClient
    client = PolymetisArmClient(
        ip_address=args.ip, state_port=args.state_port, cmd_port=args.cmd_port,
        kq=kq, kqd=kqd, start_impedance=False,
    )

    print("[move] waiting for bridge state ...")
    for _ in range(100):
        if client.arm_data_received:
            break
        time.sleep(0.1)
    if not client.arm_data_received:
        print("[ERR] no state from NUC bridge. Is polymetis_joint_bridge.py running?")
        client.shutdown()
        return

    start = client.arm_joint_positions.detach().cpu().numpy().astype(np.float64)
    print(f"[move] current pose = {start.round(4).tolist()}")

    start_delta = float(np.abs(start - q_target).max())
    print(f"[move] max joint delta to target = {start_delta:.3f} rad")
    if start_delta > args.max_start_delta and not args.force_unsafe:
        print(f"[ABORT] delta {start_delta:.3f} > max_start_delta {args.max_start_delta}. "
              f"Move the arm closer first, or raise --max_start_delta / --force_unsafe.")
        client.shutdown()
        return

    print("[move] engaging joint impedance ...")
    client.start_joint_impedance()
    time.sleep(1.0)

    ctrl_dt = 1.0 / args.ctrl_freq
    n_appr = max(int(args.approach_time * args.ctrl_freq), 1)
    print(f"[move] interpolating to target over {args.approach_time:.1f}s "
          f"({n_appr} steps @ {args.ctrl_freq:.0f}Hz) ...")
    for i in range(n_appr):
        a = (i + 1) / n_appr
        client.publish_arm_joint_pos(start * (1 - a) + q_target * a)
        time.sleep(ctrl_dt)

    # settle
    for _ in range(int(0.5 * args.ctrl_freq)):
        client.publish_arm_joint_pos(q_target)
        time.sleep(ctrl_dt)

    reached = client.arm_joint_positions.detach().cpu().numpy().astype(np.float64)
    err = np.abs(reached - q_target)
    print(f"[move] reached  = {reached.round(4).tolist()}")
    print(f"[move] joint err= {err.round(4).tolist()}  (max {err.max():.4f} rad = {np.degrees(err.max()):.2f} deg)")

    if args.hold:
        print("[move] HOLDING target (republishing @ %.0fHz). Capture / align now. Ctrl+C to release."
              % args.ctrl_freq)
        try:
            while True:
                client.publish_arm_joint_pos(q_target)
                time.sleep(ctrl_dt)
        except KeyboardInterrupt:
            print("\n[move] released hold.")
    else:
        print("[move] done (no --hold; controller keeps last target until another policy runs).")

    client.shutdown()


if __name__ == "__main__":
    main()
