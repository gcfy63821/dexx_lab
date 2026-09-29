#!/usr/bin/env python3
"""Generate canonical motion-files (CSV) for sim/real action-tracking tests.

Ported from https://github.com/NVIDIA-Isaac-Sim/sage (motion_files/so101/custom/*)
and adapted to Franka FR3 (7 arm joints).

Each output pair is {name}.csv + {name}.json:
  - CSV  : header row = joint names; subsequent rows = target positions (rad)
  - JSON : metadata (control_freq_hz, duration_s, description, base_pose)

All motions start and end at SAFE_MIDDLE_POS so real hardware can chain them
without discontinuities.

Usage:
    python tools/sysid/generate_motions.py \
        [--output_dir tools/sysid/motions] \
        [--control_freq 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARM_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]
# Same safe home pose the existing step_response_* scripts use.
SAFE_MIDDLE_POS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=np.float64)

# Franka FR3 datasheet joint limits (position [rad], velocity [rad/s], accel [rad/s^2]).
# We keep a safety margin = 70% of spec in safety_check(). Exceeding will usually
# trigger a reflex / halt on the real robot.
FR3_JOINT_POS_LIMITS = np.array([
    [-2.7437, 2.7437],     # joint 1
    [-1.7837, 1.7837],     # joint 2
    [-2.9007, 2.9007],     # joint 3
    [-3.0421, -0.1518],    # joint 4 (ONE-SIDED — must stay negative!)
    [-2.8065, 2.8065],     # joint 5
    [ 0.5445, 4.5169],     # joint 6 (MUST stay positive)
    [-3.0159, 3.0159],     # joint 7
], dtype=np.float64)

FR3_VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])  # rad/s
FR3_ACC_LIMIT = np.array([15.0,   7.5,   10.0,  12.5,   15.0,  20.0,  20.0])  # rad/s^2 (conservative)

SAFETY_MARGIN = 0.70  # only use 70% of rated maxima


def _safety_check(name: str, positions: np.ndarray, freq: float, raise_on_fail: bool = False) -> bool:
    """Check position / velocity / acceleration limits. Return True if safe.

    Computes per-joint max |vel|, max |acc| via finite differences at the
    control rate. Prints a per-joint table; raises if any joint is unsafe
    and raise_on_fail is True.
    """
    dt = 1.0 / freq
    vel = np.gradient(positions, axis=0) / dt                        # (N, 7)
    acc = np.gradient(vel, axis=0) / dt                              # (N, 7)

    pos_min = positions.min(axis=0); pos_max = positions.max(axis=0)
    vel_abs_max = np.abs(vel).max(axis=0)
    acc_abs_max = np.abs(acc).max(axis=0)

    ok = True
    warnings = []
    for j in range(7):
        pos_lo, pos_hi = FR3_JOINT_POS_LIMITS[j]
        # 0.02 rad hardware margin off the datasheet limits
        if pos_min[j] < pos_lo + 0.02 or pos_max[j] > pos_hi - 0.02:
            ok = False
            warnings.append(
                f"  J{j+1} POS [{pos_min[j]:+.3f}, {pos_max[j]:+.3f}] "
                f"violates limits [{pos_lo:+.3f}, {pos_hi:+.3f}]"
            )
        if vel_abs_max[j] > FR3_VEL_LIMIT[j] * SAFETY_MARGIN:
            ok = False
            warnings.append(
                f"  J{j+1} VEL {vel_abs_max[j]:.2f} > {FR3_VEL_LIMIT[j]*SAFETY_MARGIN:.2f} "
                f"rad/s (spec {FR3_VEL_LIMIT[j]:.2f})"
            )
        if acc_abs_max[j] > FR3_ACC_LIMIT[j] * SAFETY_MARGIN:
            ok = False
            warnings.append(
                f"  J{j+1} ACC {acc_abs_max[j]:.1f} > {FR3_ACC_LIMIT[j]*SAFETY_MARGIN:.1f} "
                f"rad/s^2 (spec {FR3_ACC_LIMIT[j]:.1f})"
            )

    status = "SAFE ✓" if ok else "UNSAFE ✗"
    print(f"  [{status}] {name} — vel_max={vel_abs_max.round(2).tolist()}, "
          f"acc_max={acc_abs_max.round(1).tolist()}")
    for w in warnings:
        print(w)
    if not ok and raise_on_fail:
        raise RuntimeError(f"Motion {name} failed safety check")
    return ok


def write_motion(out_dir: Path, name: str, positions: np.ndarray,
                 control_freq_hz: float, description: str, meta_extra: dict | None = None):
    assert positions.shape[1] == 7, positions.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"

    with open(csv_path, "w") as f:
        f.write(",".join(ARM_JOINT_NAMES) + "\n")
        for row in positions:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    meta = {
        "name": name,
        "joint_names": ARM_JOINT_NAMES,
        "control_freq_hz": float(control_freq_hz),
        "duration_s": float(len(positions) / control_freq_hz),
        "n_steps": int(len(positions)),
        "base_pose": SAFE_MIDDLE_POS.tolist(),
        "description": description,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] {name}: {positions.shape[0]} steps @ {control_freq_hz}Hz "
          f"({len(positions)/control_freq_hz:.1f}s) -> {csv_path}")


def _min_jerk(t: np.ndarray) -> np.ndarray:
    """5th-order minimum-jerk profile. s(0)=0, s(1)=1, s'=s''=0 at both endpoints.

    Using this instead of linear ramps eliminates the velocity / acceleration
    discontinuities that would otherwise show up at segment boundaries and
    trigger the FR3 accel-limit check.
    """
    t = np.clip(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _ramp_to(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Minimum-jerk interpolation, n rows inclusive of endpoints."""
    s = _min_jerk(np.linspace(0.0, 1.0, n))[:, None]
    return (1 - s) * start + s * end


def _fade_envelope(n: int, fade_n: int) -> np.ndarray:
    """1-D array of length n: fades in over fade_n samples, holds at 1, fades
    out over fade_n samples. All fades use minimum-jerk shape."""
    env = np.ones(n)
    fade_n = min(fade_n, n // 2)
    if fade_n > 0:
        env[:fade_n] = _min_jerk(np.linspace(0.0, 1.0, fade_n))
        env[-fade_n:] = _min_jerk(np.linspace(1.0, 0.0, fade_n))
    return env


def _prepend_approach(positions: np.ndarray, approach_s: float,
                      settle_s: float, freq: float) -> np.ndarray:
    """Prepend an approach-and-settle segment from SAFE_MIDDLE_POS."""
    n_appr = int(approach_s * freq)
    n_settle = int(settle_s * freq)
    first_row = positions[0]
    approach = _ramp_to(SAFE_MIDDLE_POS, first_row, n_appr)
    settle = np.tile(first_row, (n_settle, 1))
    return np.vstack([approach, settle, positions])


def _append_return(positions: np.ndarray, return_s: float,
                   settle_s: float, freq: float) -> np.ndarray:
    """Append a slow return to SAFE_MIDDLE_POS + settle."""
    n_ret = int(return_s * freq)
    n_settle = int(settle_s * freq)
    last_row = positions[-1]
    ret = _ramp_to(last_row, SAFE_MIDDLE_POS, n_ret)
    settle = np.tile(SAFE_MIDDLE_POS, (n_settle, 1))
    return np.vstack([positions, ret, settle])


# ------------------------------------------------------------------ motions

def gen_step_per_joint(freq: float, step_size: float = 0.08,
                       hold_s: float = 1.5, settle_s: float = 0.5,
                       ramp_s: float = 0.5) -> np.ndarray:
    # ramp_s=0.5s with min-jerk profile → max accel at mid-ramp
    # = 10 * step_size / ramp_s^2 ≈ 3.2 rad/s² (under j2 limit 5.25)
    """Sequentially step each joint up, back to center, down, back to center.

    Same protocol as tools/sysid/step_response_sim.py, just in CSV form so
    it can be replayed via the unified motion pipeline.
    """
    # Use a short linear ramp (ramp_s) between levels instead of instant jump,
    # so the commanded velocity stays bounded. step_size=0.08 / ramp=0.2s
    # yields peak vel 0.4 rad/s (well under FR3 2.175).
    rows = [SAFE_MIDDLE_POS.copy()]
    n_ramp = max(int(ramp_s * freq), 1)
    n_hold = int(hold_s * freq)
    n_settle = int(settle_s * freq)
    for j in range(7):
        prev = SAFE_MIDDLE_POS.copy()
        for delta in (+step_size, 0.0, -step_size, 0.0):
            target = SAFE_MIDDLE_POS.copy()
            target[j] += delta
            rows.extend(_ramp_to(prev, target, n_ramp))   # smooth transition
            rows.extend([target] * n_hold)                 # hold at step
            rows.extend([target] * n_settle)
            prev = target
    rows.append(SAFE_MIDDLE_POS)
    return np.asarray(rows)


def gen_chirp_sweep(freq: float, amplitude: float = 0.06,
                    f_start: float = 0.2, f_end: float = 1.3,
                    duration_s: float = 25.0, phase_shift: bool = True,
                    fade_s: float = 2.0) -> np.ndarray:
    # Safe defaults: A=0.06 rad, f_end=1.3 Hz
    #   peak vel  = 0.06 * 2π * 1.3  = 0.49 rad/s  (well under FR3 2.175)
    #   peak acc  = 0.06 * (2π*1.3)² = 4.0 rad/s²  (under j2 70%-limit 5.25)
    # fade_s of min-jerk envelope at start/end ensures zero velocity at boundaries.
    """Linear frequency chirp sin(2π·∫f(t) dt) on each joint.

    - f linear ramps from f_start to f_end Hz over duration_s
    - amplitude in rad (±amplitude from base pose)
    - if phase_shift=True, each joint gets its own phase offset so the sweep
      exercises multi-joint coordination (not just one at a time)

    Post-analysis: picking max_lag per joint per frequency band gives you the
    arm's phase response curve — far more informative than a single step test.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    # instantaneous freq linear in time -> phase = integral
    f_inst = f_start + (f_end - f_start) * (t / duration_s)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / freq  # discrete integral

    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(SAFE_MIDDLE_POS, (n, 1))
    for j in range(7):
        offset = j * np.pi / 3.5 if phase_shift else 0.0
        positions[:, j] += amplitude * env * np.sin(phase + offset)
    return positions


def gen_sinusoidal_single_joint(freq: float, joint: int = 4,
                                amplitude: float = 0.2, f_hz: float = 0.35,
                                duration_s: float = 14.0, fade_s: float = 1.5) -> np.ndarray:
    # Safe defaults (A=0.2, f=0.35Hz): peak vel 0.44, peak acc 0.97 rad/s²
    # fade_s of min-jerk envelope at start/end → zero vel/acc at boundaries.
    """Pure sinusoid on one joint — cleanest for measuring single-joint Bode.

    Holds every other joint at SAFE_MIDDLE_POS. Useful when chirp introduces
    too much cross-joint coupling noise for the joint you care about.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(SAFE_MIDDLE_POS, (n, 1))
    positions[:, joint] += amplitude * env * np.sin(2 * np.pi * f_hz * t)
    return positions


def gen_circular_wrist(freq: float, radius_rad: float = 0.2,
                       f_hz: float = 0.25, duration_s: float = 16.0,
                       fade_s: float = 1.5) -> np.ndarray:
    # Safe (r=0.2, f=0.25Hz): peak vel 0.31 rad/s, peak acc 0.49 rad/s²
    """Circular motion using joints 5 and 6 (wrist yaw + pitch).

    Good for detecting backlash / hysteresis in the wrist gimbal.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(SAFE_MIDDLE_POS, (n, 1))
    # cos starts at 1 so shift the start by phase such that env*cos begins at 0
    positions[:, 4] += radius_rad * env * np.sin(2 * np.pi * f_hz * t)
    positions[:, 5] += radius_rad * env * (1 - np.cos(2 * np.pi * f_hz * t)) * 0.5
    # ^ the second joint draws a 1-cos curve scaled — starts and peaks at 0/r,
    # so we still get a round-ish path without position discontinuity. For a
    # true circle, run a longer duration and ignore first/last cycle; here we
    # trade exact shape for safety at the endpoints.
    return positions


def gen_backlash_detection(freq: float, amplitude: float = 0.03,
                           n_reversals: int = 6, hold_s: float = 1.2,
                           ramp_s: float = 1.2) -> np.ndarray:
    """Small back-and-forth reversals per joint to expose backlash / stiction.

    Smaller amplitude than chirp so backlash (dead-band) shows up as flat
    segments at reversal points. Uses min-jerk ramps so peak accel stays
    under limit: for A=0.04, ramp=0.6s → max acc ≈ 2.2 rad/s² (safe on all).
    """
    rows = [SAFE_MIDDLE_POS.copy()]
    n_ramp = max(int(ramp_s * freq), 1)
    n_hold = int(hold_s * freq)
    for j in range(7):
        prev = SAFE_MIDDLE_POS.copy()
        for k in range(n_reversals):
            sign = 1.0 if k % 2 == 0 else -1.0
            target = SAFE_MIDDLE_POS.copy()
            target[j] += sign * amplitude
            rows.extend(_ramp_to(prev, target, n_ramp))
            rows.extend([target] * n_hold)
            prev = target
        # IMPORTANT: return this joint to center before moving to next joint,
        # otherwise the next joint's loop starts with this one off-center and
        # the subsequent ramp would have a 1-sample discontinuity.
        rows.extend(_ramp_to(prev, SAFE_MIDDLE_POS, n_ramp))
        rows.extend([SAFE_MIDDLE_POS] * int(0.5 * freq))
    rows.append(SAFE_MIDDLE_POS)
    return np.asarray(rows)


def gen_diagonal_sweep(freq: float, amplitude: float = 0.15,
                       f_hz: float = 0.25, duration_s: float = 16.0,
                       fade_s: float = 1.5) -> np.ndarray:
    # Safe (A=0.15, f=0.25Hz): peak vel 0.24 rad/s, peak acc 0.37 rad/s² per joint
    """All 7 joints sinusoid in phase — tests coordinated motion & Jacobian
    coupling, not just isolated joint dynamics.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(SAFE_MIDDLE_POS, (n, 1))
    for j in range(7):
        positions[:, j] += amplitude * env * np.sin(2 * np.pi * f_hz * t)
    return positions


def gen_coupled_joints(freq: float, amplitude: float = 0.12,
                       f_hz: float = 0.25, duration_s: float = 16.0,
                       fade_s: float = 1.5) -> np.ndarray:
    # Safe (A=0.12, f=0.25Hz): peak vel 0.19 rad/s, peak acc 0.30 rad/s² per joint
    """Anti-phase pairs: (j1, -j3), (j2, -j4), (j5, -j7) to expose cross-joint
    torque coupling. Motivated by SAGE's coupled_joints.txt.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    s = amplitude * env * np.sin(2 * np.pi * f_hz * t)
    positions = np.tile(SAFE_MIDDLE_POS, (n, 1))
    positions[:, 0] += s
    positions[:, 2] -= s
    positions[:, 1] += s
    positions[:, 3] -= s
    positions[:, 4] += s
    positions[:, 6] -= s
    return positions


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).resolve().parent / "motions"))
    p.add_argument("--control_freq", type=float, default=30.0,
                   help="Control rate in Hz (match deploy, default 30)")
    p.add_argument("--approach_s", type=float, default=5.0)
    p.add_argument("--settle_s", type=float, default=0.8)
    p.add_argument("--return_s", type=float, default=5.0)
    p.add_argument("--strict", action="store_true",
                   help="Refuse to write motions that fail FR3 safety check.")
    p.add_argument("--skip_safety_check", action="store_true",
                   help="Do not run the safety check at all (dangerous).")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    freq = args.control_freq
    print(f"Generating motions @ {freq} Hz into {out_dir}")

    motions = {
        "step_per_joint": (
            gen_step_per_joint(freq),
            "Sequential step up/center/down/center per joint. Mirrors existing "
            "step_response tests, re-expressed in unified motion-file format.",
        ),
        "chirp_sweep": (
            gen_chirp_sweep(freq),
            "Linear frequency chirp 0.2→3 Hz, amplitude 0.15 rad, 7 joints "
            "phase-shifted. Enables full Bode plot extraction post-analysis.",
        ),
        "sin_j1": (
            gen_sinusoidal_single_joint(freq, joint=0, amplitude=0.15, f_hz=0.3),
            "Pure sinusoid on joint 1 (shoulder yaw). Large inertia, primarily "
            "reveals whether sim's high K=1600 matches real's low K=200 behavior.",
        ),
        "sin_j2": (
            gen_sinusoidal_single_joint(freq, joint=1, amplitude=0.15, f_hz=0.3),
            "Pure sinusoid on joint 2 (shoulder pitch). Chirp showed +70ms lag "
            "here — check if it's real or coupling artifact.",
        ),
        "sin_j3": (
            gen_sinusoidal_single_joint(freq, joint=2, amplitude=0.18, f_hz=0.35),
            "Pure sinusoid on joint 3 (shoulder roll). Paired with sin_j2 for "
            "near-proximal coverage.",
        ),
        "sin_j4": (
            gen_sinusoidal_single_joint(freq, joint=3, amplitude=0.3, f_hz=0.5, duration_s=12.0),
            "Pure sinusoid on joint 4 (elbow). Clean for single-joint Bode.",
        ),
        "sin_j6": (
            gen_sinusoidal_single_joint(freq, joint=5, amplitude=0.25, f_hz=0.6, duration_s=12.0),
            "Pure sinusoid on joint 6 (wrist pitch). Lighter inertia → faster "
            "dynamics, stresses sim at high freq.",
        ),
        "circular_wrist": (
            gen_circular_wrist(freq),
            "Circular motion on joints 5&6; detects wrist backlash/hysteresis.",
        ),
        "backlash_detection": (
            gen_backlash_detection(freq),
            "Small reversals per joint to expose backlash / stiction.",
        ),
        "diagonal_sweep": (
            gen_diagonal_sweep(freq),
            "All joints in-phase sinusoid — coordinated motion test.",
        ),
        "coupled_joints": (
            gen_coupled_joints(freq),
            "Anti-phase joint pairs — cross-joint torque coupling stress test.",
        ),
    }

    n_written = 0
    n_unsafe = 0
    for name, (positions, description) in motions.items():
        positions = _prepend_approach(positions, args.approach_s, args.settle_s, freq)
        positions = _append_return(positions, args.return_s, args.settle_s, freq)
        if not args.skip_safety_check:
            safe = _safety_check(name, positions, freq)
            if not safe:
                n_unsafe += 1
                if args.strict:
                    print(f"  [SKIP] {name} is unsafe (--strict mode)")
                    continue
                print(f"  [WARN] writing {name} despite warnings "
                      f"(use --strict to block, or reduce amp/freq)")
        write_motion(out_dir, name, positions, freq, description,
                     meta_extra={"safety_checked": not args.skip_safety_check})
        n_written += 1

    print(f"\nGenerated {n_written} / {len(motions)} motions "
          f"({n_unsafe} unsafe).")
    print("Next steps:")
    print(f"  1. (Real)  # NUC: conda activate polymetis-local && python polymetis_joint_bridge.py")
    print(f"            python tools/sysid/replay_motion_polymetis.py --ip <NUC_IP> --motion {out_dir}/chirp_sweep.csv")
    print(f"  2. (Sim)  python tools/sysid/replay_motion_sim.py --motion {out_dir}/chirp_sweep.csv")
    print(f"  3. (Analyze) python tools/sysid/analyze_motion.py --real real.pkl --sim sim.pkl")


if __name__ == "__main__":
    main()
