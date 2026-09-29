#!/usr/bin/env python3
"""Replay a canonical motion CSV on the real Franka via ROS2 and record
target/actual — the ROS2 counterpart of `replay_motion_polymetis.py`.

Publishes sensor_msgs/JointState arm targets on the command topic (default
/teleop_joint_commands, QoS BEST_EFFORT / depth 1 / VOLATILE — this must match
the robot-side joint impedance controller) and reads the measured arm state
from /joint_states, selecting fr3_joint1..7 BY NAME (the message may carry
other joints, e.g. the hand, in any order).

Output pkl schema is IDENTICAL to `replay_motion_polymetis.py` and
`replay_motion_sim.py` ({timestamps, targets, actuals, velocities, joint_names,
motion_meta, control_freq_hz, record_freq_hz, side, backend}), so
`analyze_motion.py` can diff it against the sim replay directly.

The arm stiffness/damping on this path are whatever the robot-side controller
yaml sets; there are no --kq/--kqd flags here.

Prereqs:
    # robot side: the joint impedance controller that listens on
    # /teleop_joint_commands must be running, and nothing else may publish there.
    source /opt/ros/humble/setup.bash   # rclpy + sensor_msgs

Usage (from the repo root, dexmanip env):
    # parse + preflight only, no ROS traffic
    python tools/sysid/replay_motion_ros2.py \
        --motion tools/sysid/motions/chirp_sweep.csv \
        --output logs/system_id/ros2/chirp_sweep_real.pkl --dry_run

    python tools/sysid/replay_motion_ros2.py \
        --motion tools/sysid/motions/chirp_sweep.csv \
        --output logs/system_id/ros2/chirp_sweep_real.pkl

    python tools/sysid/analyze_motion.py \
        --sim  logs/system_id/motion_replay/chirp_sweep_sim.pkl \
        --real logs/system_id/ros2/chirp_sweep_real.pkl \
        --out  logs/system_id/ros2/compare_chirp
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import threading
import time
from pathlib import Path

import numpy as np

ARM_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
HOME_POS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])

# FR3 safety limits (same as replay_motion_polymetis.py)
FR3_JOINT_POS_LIMITS = np.array([
    [-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007],
    [-3.0421, -0.1518], [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159],
])
FR3_VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
FR3_ACC_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])
SAFETY_MARGIN = 0.70
STALE_STATE_S = 0.1  # a /joint_states sample older than this counts as stale


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
    if positions.ndim != 2 or positions.shape[1] != 7:
        raise ValueError(f"Expected 7 columns, got {positions.shape}")
    json_path = motion_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
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


def import_ros2():
    """Import rclpy lazily so --help / --dry_run work without a ROS2 environment."""
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState
    except ImportError as e:
        sys.exit(f"[ERR] ROS2 Python packages not importable ({e}).\n"
                 f"      Run `source /opt/ros/humble/setup.bash` in this shell first "
                 f"(rclpy + sensor_msgs, Python 3.10).")
    return rclpy, SingleThreadedExecutor, QoSProfile, ReliabilityPolicy, DurabilityPolicy, JointState


class Ros2ArmIO:
    """Minimal ROS2 arm I/O: JointState command publisher + /joint_states reader
    (by joint name), spun on a background executor thread."""

    def __init__(self, node_name, joint_states_topic, command_topic):
        (rclpy, Executor, QoSProfile, Reliability, Durability, JointState) = import_ros2()
        self._rclpy = rclpy
        self._JointState = JointState
        rclpy.init()
        self.node = rclpy.create_node(node_name)
        qos_cmd = QoSProfile(depth=1, reliability=Reliability.BEST_EFFORT,
                             durability=Durability.VOLATILE)
        self._pub = self.node.create_publisher(JointState, command_topic, qos_cmd)
        self._lock = threading.Lock()
        self._pos = None
        self._vel = None
        self._stamp = 0.0
        self._missing_warned = False
        self.node.create_subscription(JointState, joint_states_topic, self._on_js, 10)
        self._executor = Executor()
        self._executor.add_node(self.node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    def _on_js(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        missing = [n for n in ARM_JOINT_NAMES if n not in idx or idx[n] >= len(msg.position)]
        if missing:
            if not self._missing_warned:
                self._missing_warned = True
                print(f"[WARN] /joint_states message lacks {missing}; ignoring such messages")
            return
        pos = np.array([msg.position[idx[n]] for n in ARM_JOINT_NAMES], dtype=np.float64)
        vel = np.array([msg.velocity[idx[n]] if idx[n] < len(msg.velocity) else 0.0
                        for n in ARM_JOINT_NAMES], dtype=np.float64)
        with self._lock:
            self._pos, self._vel, self._stamp = pos, vel, time.time()

    def state(self):
        """(pos, vel, age_s) of the latest arm state, or (None, None, inf)."""
        with self._lock:
            if self._pos is None:
                return None, None, float("inf")
            return self._pos.copy(), self._vel.copy(), time.time() - self._stamp

    def wait_for_state(self, timeout_s):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            if self.state()[0] is not None:
                return True
            time.sleep(0.05)
        return False

    def publish(self, target):
        msg = self._JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(ARM_JOINT_NAMES)
        msg.position = [float(x) for x in target]
        self._pub.publish(msg)

    def shutdown(self):
        try:
            self._executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if self._rclpy.ok():
            self._rclpy.shutdown()
        self._thread.join(timeout=1.0)


def ramp(io, start, end, duration_s, ctrl_freq):
    n = max(int(duration_s * ctrl_freq), 1)
    for i in range(n):
        a = (i + 1) / n
        io.publish(start * (1 - a) + end * a)
        time.sleep(1.0 / ctrl_freq)


def run(io, args, positions, meta, ctrl_freq):
    ctrl_dt = 1.0 / ctrl_freq
    rec_dt = 1.0 / args.record_freq

    print(f"[INFO] waiting for {args.joint_states_topic} ...")
    if not io.wait_for_state(5.0):
        print(f"[ERR] no arm state on {args.joint_states_topic} (need {ARM_JOINT_NAMES}). "
              f"Is the robot-side controller running, and is ROS_DOMAIN_ID the same as the robot's?")
        return
    start, _, _ = io.state()
    print(f"[INFO] current pose: {start.round(4).tolist()}")

    start_delta = float(np.abs(start - positions[0]).max())
    if start_delta > args.max_start_delta:
        print(f"[ABORT] current pose is {start_delta:.3f} rad from motion[0] "
              f"(max_start_delta={args.max_start_delta}). Move the arm near "
              f"{positions[0].round(3).tolist()} first, or raise --max_start_delta.")
        return

    # Slow approach to motion[0]
    first = positions[0]
    print(f"[INFO] approaching motion start over {args.approach_time:.1f}s "
          f"(max_delta={np.abs(first - start).max():.3f} rad)")
    ramp(io, start, first, args.approach_time, ctrl_freq)
    time.sleep(0.5)

    # Main replay: publish @ ctrl_freq, record @ record_freq (decoupled)
    n = positions.shape[0]
    total = n * ctrl_dt
    rec_t, rec_tg, rec_ac, rec_v = [], [], [], []
    n_stale = 0
    last_rec = -rec_dt
    cur_target = positions[0].copy()
    io.publish(cur_target)
    t0 = time.time()
    step_idx = -1
    watchdog = total + 2.0
    interrupted = False
    print(f"[INFO] replaying {n} steps over {total:.2f}s (timeout {watchdog:.1f}s)...")
    try:
        while True:
            t = time.time() - t0
            if t >= total or t >= watchdog:
                break
            idx = min(int(t / ctrl_dt), n - 1)
            if idx != step_idx:
                step_idx = idx
                cur_target = positions[idx].copy()
                io.publish(cur_target)
            if t - last_rec >= rec_dt:
                last_rec = t
                pos, vel, age = io.state()
                n_stale += int(age > STALE_STATE_S)
                rec_t.append(t)
                rec_tg.append(cur_target.copy())
                rec_ac.append(pos)
                rec_v.append(vel)
            time.sleep(0.0005)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[WARN] interrupted — the controller holds the last published target.")

    if not interrupted:
        io.publish(positions[-1])
    print(f"[INFO] done. recorded {len(rec_t)} frames over {rec_t[-1] if rec_t else 0:.2f}s.")
    if n_stale:
        print(f"[WARN] {n_stale}/{len(rec_t)} recorded samples used a /joint_states "
              f"message older than {STALE_STATE_S*1000:.0f} ms — check the state rate.")

    # Save before the return-to-home leg so the recording survives an abort there.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "side": "real",
        "backend": "ros2",
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
    print(f"[OK] saved to {out_path}" + (" (partial: interrupted)" if interrupted else ""))

    if not args.no_return_home and not interrupted:
        print("[INFO] returning to home over 3s ...")
        ramp(io, positions[-1], HOME_POS, 3.0, ctrl_freq)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True, type=str, help="motion CSV (tools/sysid/motions/*.csv)")
    p.add_argument("--output", required=True, type=str, help="output pkl path")
    p.add_argument("--joint_states_topic", type=str, default="/joint_states")
    p.add_argument("--command_topic", type=str, default="/teleop_joint_commands")
    p.add_argument("--control_freq_override", type=float, default=None,
                   help="override control_freq_hz from the motion's JSON sidecar")
    p.add_argument("--record_freq", type=float, default=100.0, help="state recording rate (Hz)")
    p.add_argument("--approach_time", type=float, default=4.0,
                   help="slow interpolation from the current pose to motion[0] (s)")
    p.add_argument("--dry_run", action="store_true", help="parse + preflight only; no ROS traffic")
    p.add_argument("--force_unsafe", action="store_true",
                   help="run even if the preflight safety check fails (DANGEROUS)")
    p.add_argument("--max_start_delta", type=float, default=0.5,
                   help="refuse to run if the current pose is further than this from motion[0] (rad)")
    p.add_argument("--no_return_home", action="store_true", help="skip return-to-home at the end")
    args = p.parse_args()

    positions, meta = load_motion(args.motion)
    ctrl_freq = args.control_freq_override or float(meta.get("control_freq_hz", 30.0))

    print(f"[INFO] Motion: {meta.get('name', 'unknown')}")
    print(f"[INFO] {positions.shape[0]} steps @ {ctrl_freq} Hz = {positions.shape[0] / ctrl_freq:.1f}s")
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

    io = Ros2ArmIO("replay_motion_ros2", args.joint_states_topic, args.command_topic)
    print(f"[INFO] commands -> {args.command_topic}, state <- {args.joint_states_topic}")
    try:
        run(io, args, positions, meta, ctrl_freq)
    finally:
        io.shutdown()


if __name__ == "__main__":
    main()
