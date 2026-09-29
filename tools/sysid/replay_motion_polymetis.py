#!/usr/bin/env python3
"""Replay a canonical motion CSV on the real Franka via POLYMETIS and record
target/actual — the Polymetis recorder. (The ROS2 equivalent is `replay_motion_ros2.py`.)

Uses PolymetisArmClient (ZMQ -> NUC polymetis_joint_bridge.py -> RobotInterface
joint impedance). Output pkl schema is IDENTICAL to replay_motion_sim.py, so
`analyze_motion.py` can diff it against `replay_motion_sim.py` directly.

Prereqs:
    # NUC (polymetis env): server on :50051 + bridge:
    conda activate polymetis-local && python deploy/polymetis_joint_bridge.py
    # and STOP any other controller on the NUC — Polymetis runs a single policy.

Usage (training PC, dexmanip env):
    python tools/sysid/replay_motion_polymetis.py \
        --ip <NUC_IP> \
        --motion tools/sysid/motions/chirp_sweep.csv \
        --output logs/system_id/polymetis/chirp_sweep_real.pkl
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from dexx import deploy_config as _dcfg

sys.path.insert(0, ".")

ARM_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

# FR3 safety limits
FR3_JOINT_POS_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
])
FR3_VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
FR3_ACC_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
SAFETY_MARGIN = 0.70


def preflight_safety(positions, ctrl_freq):
    dt = 1.0 / ctrl_freq
    vel = np.gradient(positions, axis=0) / dt
    acc = np.gradient(vel, axis=0) / dt
    issues = []
    pmin, pmax = positions.min(0), positions.max(0)
    for j in range(7):
        lo, hi = FR3_JOINT_POS_LIMITS[j]
        if pmin[j] < lo + 0.02 or pmax[j] > hi - 0.02:
            issues.append(f"J{j+1} pos [{pmin[j]:+.3f}, {pmax[j]:+.3f}] outside [{lo:+.3f}, {hi:+.3f}]")
        if float(np.abs(vel[:, j]).max()) > FR3_VEL_LIMIT[j] * SAFETY_MARGIN:
            issues.append(f"J{j+1} peak vel {np.abs(vel[:, j]).max():.2f} > {FR3_VEL_LIMIT[j]*SAFETY_MARGIN:.2f} rad/s")
        if float(np.abs(acc[:, j]).max()) > FR3_ACC_LIMIT[j] * SAFETY_MARGIN:
            issues.append(f"J{j+1} peak acc {np.abs(acc[:, j]).max():.1f} > {FR3_ACC_LIMIT[j]*SAFETY_MARGIN:.1f} rad/s^2")
    return len(issues) == 0, issues


def load_motion(motion_path):
    motion_path = Path(motion_path)
    with open(motion_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader]
    positions = np.asarray(rows, dtype=np.float64)
    if positions.shape[1] != 7:
        raise ValueError(f"Expected 7 columns, got {positions.shape}")
    json_path = motion_path.with_suffix(".json")
    if json_path.exists():
        meta = json.load(open(json_path))
    else:
        meta = {"control_freq_hz": 30.0, "joint_names": ARM_JOINT_NAMES, "name": motion_path.stem}
    if header != ARM_JOINT_NAMES:
        try:
            perm = [header.index(n) for n in ARM_JOINT_NAMES]
            positions = positions[:, perm]
            print(f"[INFO] Reordered motion columns to {ARM_JOINT_NAMES}")
        except ValueError:
            print(f"[WARN] Motion header {header} != {ARM_JOINT_NAMES}, using as-is")
    return positions, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", required=True, help="NUC bridge IP (e.g. 192.168.1.10)")
    p.add_argument("--state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
    p.add_argument("--cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT)
    p.add_argument("--motion", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--control_freq_override", type=float, default=None)
    p.add_argument("--record_freq", type=float, default=100.0)
    p.add_argument("--approach_time", type=float, default=4.0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force_unsafe", action="store_true")
    p.add_argument("--max_start_delta", type=float, default=0.5)
    p.add_argument("--kq", type=str, default=None, help="JSON list of 7 Kq (else Polymetis default)")
    p.add_argument("--kqd", type=str, default=None, help="JSON list of 7 Kqd")
    p.add_argument("--no_return_home", action="store_true", help="skip return-to-home at the end")
    args = p.parse_args()

    positions, meta = load_motion(args.motion)
    ctrl_freq = args.control_freq_override or float(meta.get("control_freq_hz", 30.0))
    ctrl_dt = 1.0 / ctrl_freq
    rec_dt = 1.0 / args.record_freq

    print(f"[INFO] Motion: {meta.get('name', 'unknown')}")
    print(f"[INFO] {positions.shape[0]} steps @ {ctrl_freq} Hz = {positions.shape[0]*ctrl_dt:.1f}s")
    print(f"[INFO] Joint range per dim: {(positions.max(0) - positions.min(0)).round(3)}")

    print("\n[PREFLIGHT] FR3 safety check...")
    safe, issues = preflight_safety(positions, ctrl_freq)
    if safe:
        print("  [OK] All joints within 70% of FR3 pos/vel/acc limits.")
    else:
        print(f"  [UNSAFE] {len(issues)} issue(s):")
        for i in issues:
            print(f"    - {i}")
        if not args.force_unsafe:
            print("  [ABORT] refusing to run. Pass --force_unsafe to override (DANGEROUS).")
            return
        print("  [WARN] --force_unsafe set; proceeding at your own risk!")

    if args.dry_run:
        return

    import json as _json
    kq = _json.loads(args.kq) if args.kq else None
    kqd = _json.loads(args.kqd) if args.kqd else None

    from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import PolymetisArmClient
    client = PolymetisArmClient(
        ip_address=args.ip, state_port=args.state_port, cmd_port=args.cmd_port,
        kq=kq, kqd=kqd, start_impedance=False,
    )

    print("[INFO] waiting for bridge state ...")
    for _ in range(100):
        if client.arm_data_received:
            break
        time.sleep(0.1)
    if not client.arm_data_received:
        print("[ERR] no state from NUC bridge. Is polymetis_joint_bridge.py running?")
        client.shutdown(); return

    def cur_pos():
        return client.arm_joint_positions.detach().cpu().numpy().astype(np.float64)

    def cur_vel():
        return client.arm_joint_velocities.detach().cpu().numpy().astype(np.float64)

    start = cur_pos()
    print(f"[INFO] current pose: {start.round(4).tolist()}")

    start_delta = float(np.abs(start - positions[0]).max())
    if start_delta > args.max_start_delta:
        print(f"[ABORT] current pose is {start_delta:.3f} rad from motion[0] "
              f"(max_start_delta={args.max_start_delta}). Move the arm near "
              f"{positions[0].round(3).tolist()} first, or raise --max_start_delta.")
        client.shutdown(); return

    print("[INFO] engaging joint impedance ...")
    client.start_joint_impedance()
    time.sleep(1.0)

    # Slow approach to motion[0]
    first = positions[0]
    n_appr = max(int(args.approach_time * ctrl_freq), 1)
    print(f"[INFO] approaching motion start over {args.approach_time:.1f}s (max_delta={np.abs(first-start).max():.3f} rad)")
    for i in range(n_appr):
        a = (i + 1) / n_appr
        client.publish_arm_joint_pos(start * (1 - a) + first * a)
        time.sleep(ctrl_dt)
    time.sleep(0.5)

    # Main replay: publish @ ctrl_freq, record @ record_freq (decoupled)
    n = positions.shape[0]
    total = n * ctrl_dt
    rec_t, rec_tg, rec_ac, rec_v = [], [], [], []
    last_rec = -rec_dt
    cur_target = positions[0].copy()
    client.publish_arm_joint_pos(cur_target)
    t0 = time.time()
    step_idx = -1
    watchdog = total + 2.0
    print(f"[INFO] replaying {n} steps over {total:.2f}s (timeout {watchdog:.1f}s)...")
    while True:
        t = time.time() - t0
        if t >= total or t >= watchdog:
            break
        idx = min(int(t / ctrl_dt), n - 1)
        if idx != step_idx:
            step_idx = idx
            cur_target = positions[idx].copy()
            client.publish_arm_joint_pos(cur_target)
        if t - last_rec >= rec_dt:
            last_rec = t
            rec_t.append(t)
            rec_tg.append(cur_target.copy())
            rec_ac.append(cur_pos())
            rec_v.append(cur_vel())
        time.sleep(0.0005)

    client.publish_arm_joint_pos(positions[-1])
    print(f"[INFO] done. recorded {len(rec_t)} frames over {rec_t[-1] if rec_t else 0:.2f}s.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "side": "real",
        "backend": "polymetis",
        "motion_meta": meta,
        "control_freq_hz": ctrl_freq,
        "record_freq_hz": args.record_freq,
        "joint_names": ARM_JOINT_NAMES,
        "timestamps": np.asarray(rec_t),
        "targets": np.asarray(rec_tg),
        "actuals": np.asarray(rec_ac),
        "velocities": np.asarray(rec_v),
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[OK] saved to {out_path}")

    if not args.no_return_home:
        end_pose = positions[-1]
        home = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
        n_ret = int(3.0 * ctrl_freq)
        print("[INFO] returning to home over 3s ...")
        for i in range(n_ret):
            a = (i + 1) / n_ret
            client.publish_arm_joint_pos(end_pose * (1 - a) + home * a)
            time.sleep(ctrl_dt)

    client.shutdown()


if __name__ == "__main__":
    main()
