#!/usr/bin/env python3
"""Replay a canonical HAND motion CSV on the real Sharpa Wave hand and record
target/actual joint angles for sim-vs-real comparison.

Uses Sharpa SDK directly (no ROS2).

Usage:
    python tools/sysid/replay_hand_motion_real.py \
        --motion tools/sysid/motions_hand/hand_sin_all.csv \
        --output logs/system_id/hand_replay/hand_sin_all_real.pkl
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from dexx.deploy_config import import_sharpa_sdk

_sharpa = import_sharpa_sdk()
SharpaWaveManager = _sharpa.SharpaWaveManager
ControlMode = _sharpa.ControlMode
ControlSource = _sharpa.ControlSource

HAND_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP",  "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE",   "ring_MCP_AA",   "ring_PIP",   "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",  "pinky_MCP_AA",  "pinky_PIP",  "pinky_DIP",
]

SHARPA_REAL_LIMITS = np.asarray([
    (-0.087,  1.833), (-0.307,  0.061), (-0.436,  1.309), (-0.307,  0.308),
    ( 0.000,  1.658), (-0.175,  1.466), (-0.309,  0.106), ( 0.000,  1.658),
    ( 0.000,  1.309), (-0.175,  1.466), (-0.136,  0.159), ( 0.000,  1.658),
    ( 0.000,  1.309), (-0.175,  1.466), (-0.027,  0.147), ( 0.000,  1.658),
    ( 0.000,  1.309), ( 0.008,  0.250), (-0.175,  1.466), (-0.026,  0.306),
    ( 0.000,  1.658), ( 0.000,  1.309),
], dtype=np.float64)


def load_motion(motion_path: str):
    motion_path = Path(motion_path)
    with open(motion_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, row)) for row in reader]
    positions = np.asarray(rows, dtype=np.float64)
    if positions.shape[1] != 22:
        raise ValueError(f"Hand motion must have 22 columns, got {positions.shape}")
    json_path = motion_path.with_suffix(".json")
    meta = {"control_freq_hz": 30.0, "joint_names": HAND_JOINT_SUFFIXES,
            "name": motion_path.stem}
    if json_path.exists():
        meta.update(json.load(open(json_path)))
    return positions, meta


def safety_check(positions: np.ndarray):
    lo = positions.min(axis=0); hi = positions.max(axis=0)
    issues = []
    for j in range(22):
        l, u = SHARPA_REAL_LIMITS[j]
        if lo[j] < l - 1e-3 or hi[j] > u + 1e-3:
            issues.append(
                f"J{j} ({HAND_JOINT_SUFFIXES[j]}) range "
                f"[{lo[j]:+.3f}, {hi[j]:+.3f}] outside limits "
                f"[{l:+.3f}, {u:+.3f}]"
            )
    return len(issues) == 0, issues


def connect_hand():
    manager = SharpaWaveManager.get_instance()
    time.sleep(1.0)
    for _ in range(10):
        devices = manager.get_all_device_sn()
        if devices:
            print(f"[replay] connected to Sharpa device {devices[0]}")
            return manager.connect(devices[0])
        time.sleep(1.0)
    raise RuntimeError("No Sharpa device found")


def init_hand(hand, speed_coef=0.3, current_coef=0.5):
    for fn, arg, name in [
        (hand.set_control_mode, ControlMode.POSITION, "control_mode"),
        (hand.set_speed_coeff, speed_coef, "speed_coeff"),
        (hand.set_current_coeff, current_coef, "current_coeff"),
        (hand.set_control_source, ControlSource.SDK, "control_source"),
    ]:
        err = fn(arg)
        if err.code != 0:
            raise RuntimeError(f"init step {name} failed: {err}")
    hand.start()
    time.sleep(0.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--control_freq_override", type=float, default=None)
    p.add_argument("--record_freq", type=float, default=100.0)
    p.add_argument("--approach_s", type=float, default=3.0,
                   help="Ramp time from current hand pose to motion[0].")
    p.add_argument("--speed_coef", type=float, default=0.3)
    p.add_argument("--current_coef", type=float, default=0.5)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--force_unsafe", action="store_true")
    args = p.parse_args()

    positions, meta = load_motion(args.motion)
    ctrl_freq = args.control_freq_override or float(meta.get("control_freq_hz", 30.0))
    ctrl_dt = 1.0 / ctrl_freq
    rec_dt = 1.0 / args.record_freq

    print(f"[replay] motion: {meta.get('name', 'unknown')}")
    print(f"[replay] {positions.shape[0]} steps @ {ctrl_freq} Hz "
          f"= {positions.shape[0]*ctrl_dt:.1f}s")
    print(f"[replay] per-joint target range:")
    rng = positions.max(axis=0) - positions.min(axis=0)
    for j in range(22):
        if rng[j] > 0.005:
            print(f"  j{j} {HAND_JOINT_SUFFIXES[j]:<18}: range={rng[j]:.3f} rad")

    safe, issues = safety_check(positions)
    if not safe:
        print("[UNSAFE] motion exceeds Sharpa reachable limits:")
        for i in issues:
            print(f"  {i}")
        if not args.force_unsafe:
            print("Aborting. Use --force_unsafe to override.")
            return

    if args.dry_run:
        return

    hand = connect_hand()
    init_hand(hand, speed_coef=args.speed_coef, current_coef=args.current_coef)

    # Read current pose
    cur = np.asarray(hand.get_states().angles, dtype=np.float64)
    print(f"[replay] current hand pose: {cur.round(2).tolist()}")

    # Slow approach to motion[0]
    n_appr = max(int(args.approach_s * ctrl_freq), 1)
    first = positions[0]
    print(f"[replay] approaching motion start over {args.approach_s:.1f}s "
          f"(max_delta={np.abs(cur - first).max():.3f} rad)")
    for i in range(n_appr):
        alpha = (i + 1) / n_appr
        interp = cur * (1 - alpha) + first * alpha
        hand.set_joint_position(interp.tolist())
        time.sleep(ctrl_dt)
    time.sleep(0.5)

    # Main replay with decoupled record loop
    n = positions.shape[0]
    total_duration = n * ctrl_dt
    rec_timestamps, rec_targets, rec_actuals = [], [], []
    last_rec_t = -rec_dt

    cur_target = positions[0].copy()
    hand.set_joint_position(cur_target.tolist())
    t_start = time.time()
    step_idx = -1
    watchdog = total_duration + 2.0
    print(f"[replay] replaying {n} steps over {total_duration:.2f}s")
    while True:
        t = time.time() - t_start
        if t >= total_duration:
            break
        if t >= watchdog:
            print(f"[WARN] watchdog at {t:.2f}s"); break

        new_idx = min(int(t / ctrl_dt), n - 1)
        if new_idx != step_idx:
            step_idx = new_idx
            cur_target = positions[step_idx].copy()
            hand.set_joint_position(cur_target.tolist())

        if t - last_rec_t >= rec_dt:
            last_rec_t = t
            rec_timestamps.append(t)
            rec_targets.append(cur_target.copy())
            angles = np.asarray(hand.get_states().angles, dtype=np.float64)
            rec_actuals.append(angles)

        time.sleep(0.0005)

    hand.set_joint_position(positions[-1].tolist())
    last_t = rec_timestamps[-1] if rec_timestamps else 0.0
    print(f"[replay] recorded {len(rec_timestamps)} frames over {last_t:.2f}s")

    # Save BEFORE return-to-home so Ctrl-C during return doesn't lose data.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "side": "real",
        "motion_meta": meta,
        "control_freq_hz": ctrl_freq,
        "record_freq_hz": args.record_freq,
        "joint_names": HAND_JOINT_SUFFIXES,
        "timestamps": np.asarray(rec_timestamps),
        "targets": np.asarray(rec_targets),
        "actuals": np.asarray(rec_actuals),
        "velocities": np.zeros_like(np.asarray(rec_actuals)),  # Sharpa SDK doesn't report per-joint vel
    }
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[OK] Saved {out_path}")

    # Slow return to a safe open pose [0]*22
    n_ret = int(3.0 * ctrl_freq)
    end_pose = positions[-1].copy()
    home = np.zeros(22, dtype=np.float64)
    for i in range(n_ret):
        alpha = (i + 1) / n_ret
        hand.set_joint_position((end_pose*(1-alpha) + home*alpha).tolist())
        time.sleep(ctrl_dt)


if __name__ == "__main__":
    main()
