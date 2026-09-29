#!/usr/bin/env python3
"""Arm step response test on the real Franka via ROS2 — the real-side
counterpart of `step_response_sim.py`.

Ramps the arm to the safe middle pose, then per joint: settle, step +step_size,
back to base, step -step_size, return — recording target vs actual at
--record_freq. Commands go out as sensor_msgs/JointState on the command topic
(default /teleop_joint_commands, QoS BEST_EFFORT / depth 1 / VOLATILE, matching
the robot-side controller); the arm state is read from /joint_states by joint
name. The ROS2 plumbing is shared with `replay_motion_ros2.py`.

Output pkl schema matches `step_response_sim.py`
({step_size, hold_time, record_freq, middle_pos, joints: {j: {settle, step_up,
step_down, step_neg, return}}}, each segment {timestamps, targets, actuals,
velocities, label}), so `analyze_step_response.py` consumes it unchanged.

The arm stiffness/damping on this path are whatever the robot-side controller
yaml sets.

Usage (from the repo root, dexmanip env, after `source /opt/ros/humble/setup.bash`):
    # print the plan only, no ROS traffic
    python tools/sysid/step_response_ros2.py --joints 0 3 --dry_run

    python tools/sysid/step_response_ros2.py \
        --output logs/system_id/step_response_real.pkl \
        --step_size 0.1 --hold_time 2.0

    python tools/sysid/analyze_step_response.py \
        --real logs/system_id/step_response_real.pkl \
        --sim  logs/system_id/step_response_sim.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np



def _load_replay_tool():
    """tools/sysid/replay_motion_ros2.py (tools/ is not a package)."""
    import importlib.util
    name = "replay_motion_ros2"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_replay = _load_replay_tool()
ARM_JOINT_NAMES, FR3_JOINT_POS_LIMITS = _replay.ARM_JOINT_NAMES, _replay.FR3_JOINT_POS_LIMITS
STALE_STATE_S, Ros2ArmIO, ramp = _replay.STALE_STATE_S, _replay.Ros2ArmIO, _replay.ramp

# Safe middle position for testing (same as step_response_sim.py)
SAFE_MIDDLE_POS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
MAX_SAFE_STEP = 0.3  # rad; larger steps need --force_unsafe
APPROACH_FREQ = 30.0


def preflight(args):
    issues = []
    if args.step_size > MAX_SAFE_STEP:
        issues.append(f"step_size {args.step_size} rad > {MAX_SAFE_STEP} rad")
    for j in args.joints:
        lo, hi = FR3_JOINT_POS_LIMITS[j]
        for q in (SAFE_MIDDLE_POS[j] - args.step_size, SAFE_MIDDLE_POS[j] + args.step_size):
            if q < lo + 0.02 or q > hi - 0.02:
                issues.append(f"J{j+1} step target {q:+.3f} outside [{lo:+.3f}, {hi:+.3f}]")
    return len(issues) == 0, issues


def record_segment(io, target_pos, duration, record_freq, label, stale):
    """Publish target_pos and record target/actual at record_freq for duration s."""
    dt = 1.0 / record_freq
    n_samples = int(duration * record_freq)
    timestamps, targets, actuals, velocities = [], [], [], []
    t0 = time.time()
    for i in range(n_samples):
        io.publish(target_pos)
        t = time.time() - t0
        pos, vel, age = io.state()
        stale[0] += int(age > STALE_STATE_S)
        timestamps.append(t)
        targets.append(target_pos.copy())
        actuals.append(pos)
        velocities.append(vel)
        sleep_t = t0 + (i + 1) * dt - time.time()
        if sleep_t > 0:
            time.sleep(sleep_t)
    return {
        "timestamps": np.array(timestamps),
        "targets": np.array(targets),
        "actuals": np.array(actuals),
        "velocities": np.array(velocities),
        "label": label,
    }


def run(io, args):
    print(f"[INFO] waiting for {args.joint_states_topic} ...")
    if not io.wait_for_state(5.0):
        print(f"[ERR] no arm state on {args.joint_states_topic} (need {ARM_JOINT_NAMES}). "
              f"Is the robot-side controller running, and is ROS_DOMAIN_ID the same as the robot's?")
        return
    start, _, _ = io.state()
    print(f"[INFO] current pose: {start.round(4).tolist()}")
    start_delta = float(np.abs(start - SAFE_MIDDLE_POS).max())
    if start_delta > args.max_start_delta:
        print(f"[ABORT] current pose is {start_delta:.3f} rad from the middle pose "
              f"{SAFE_MIDDLE_POS.tolist()} (max_start_delta={args.max_start_delta}). "
              f"Move the arm closer first, or raise --max_start_delta.")
        return

    print(f"\n[INFO] approaching middle pose over {args.approach_time}s ...")
    ramp(io, start, SAFE_MIDDLE_POS, args.approach_time, APPROACH_FREQ)
    time.sleep(1.0)

    all_results = {
        "side": "real",
        "backend": "ros2",
        "joint_names": ARM_JOINT_NAMES,
        "step_size": args.step_size,
        "hold_time": args.hold_time,
        "record_freq": args.record_freq,
        "middle_pos": SAFE_MIDDLE_POS.copy(),
        "joints": {},
    }
    stale = [0]
    n_rec = 0

    for j in args.joints:
        print(f"\n{'='*50}\nTesting joint {j} ({ARM_JOINT_NAMES[j]})\n{'='*50}")
        base = SAFE_MIDDLE_POS.copy()
        up = base.copy(); up[j] += args.step_size
        neg = base.copy(); neg[j] -= args.step_size

        seg = lambda tgt, dur, name: record_segment(  # noqa: E731
            io, tgt, dur, args.record_freq, f"j{j}_{name}", stale)
        print(f"  settle ({args.settle_time}s)")
        settle = seg(base, args.settle_time, "settle")
        print(f"  step UP +{args.step_size} rad (hold {args.hold_time}s)")
        step_up = seg(up, args.hold_time, "step_up")
        print(f"  step DOWN to base (hold {args.hold_time}s)")
        step_down = seg(base, args.hold_time, "step_down")
        print(f"  step NEG -{args.step_size} rad (hold {args.hold_time}s)")
        step_neg = seg(neg, args.hold_time, "step_neg")
        print("  return to base")
        ret = seg(base, args.settle_time, "return")

        all_results["joints"][j] = {
            "settle": settle, "step_up": step_up, "step_down": step_down,
            "step_neg": step_neg, "return": ret,
        }
        n_rec += sum(len(s["timestamps"]) for s in (settle, step_up, step_down, step_neg, ret))

        up_actual = step_up["actuals"][:, j]
        up_target = step_up["targets"][:, j]
        steady_err = np.abs(up_actual[-10:] - up_target[-10:]).mean()
        overshoot = (up_actual.max() - up_target[-1]) / args.step_size * 100
        print(f"  steady-state error: {steady_err:.4f} rad ({np.degrees(steady_err):.2f} deg)")
        print(f"  overshoot: {overshoot:.1f}%")

    io.publish(SAFE_MIDDLE_POS)
    if stale[0]:
        print(f"[WARN] {stale[0]}/{n_rec} samples used a /joint_states message older than "
              f"{STALE_STATE_S*1000:.0f} ms — check the state rate.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n[OK] saved to {args.output} ({len(args.joints)} joints, {args.step_size} rad steps)")
    time.sleep(0.5)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=str, default="logs/system_id/step_response_real.pkl")
    p.add_argument("--joint_states_topic", type=str, default="/joint_states")
    p.add_argument("--command_topic", type=str, default="/teleop_joint_commands")
    p.add_argument("--step_size", type=float, default=0.1, help="step size (rad)")
    p.add_argument("--hold_time", type=float, default=2.0, help="hold time per step (s)")
    p.add_argument("--settle_time", type=float, default=1.0, help="settle time before/after steps (s)")
    p.add_argument("--approach_time", type=float, default=5.0,
                   help="slow interpolation from the current pose to the middle pose (s)")
    p.add_argument("--record_freq", type=float, default=100.0, help="publish + record rate (Hz)")
    p.add_argument("--joints", type=int, nargs="+", default=list(range(7)),
                   help="joint indices to test, 0-6 (default all)")
    p.add_argument("--max_start_delta", type=float, default=0.5,
                   help="refuse to run if the current pose is further than this from the middle pose (rad)")
    p.add_argument("--dry_run", action="store_true", help="print the plan + preflight only; no ROS traffic")
    p.add_argument("--force_unsafe", action="store_true",
                   help="run even if the preflight check fails (DANGEROUS)")
    args = p.parse_args()

    if any(j < 0 or j > 6 for j in args.joints):
        p.error(f"--joints must be in 0..6, got {args.joints}")

    per_joint = 2 * args.settle_time + 3 * args.hold_time
    print(f"[INFO] joints {args.joints}, step ±{args.step_size} rad about {SAFE_MIDDLE_POS.tolist()}")
    print(f"[INFO] ~{args.approach_time + 1.0 + per_joint * len(args.joints):.1f}s total "
          f"({per_joint:.1f}s per joint) @ {args.record_freq} Hz")

    print("\n[PREFLIGHT] FR3 safety check...")
    safe, issues = preflight(args)
    if safe:
        print("  [OK] step targets inside FR3 limits.")
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

    io = Ros2ArmIO("step_response_ros2", args.joint_states_topic, args.command_topic)
    print(f"[INFO] commands -> {args.command_topic}, state <- {args.joint_states_topic}")
    try:
        run(io, args)
    except KeyboardInterrupt:
        print("\n[WARN] interrupted — nothing saved; the controller holds the last published target.")
    finally:
        io.shutdown()


if __name__ == "__main__":
    main()
