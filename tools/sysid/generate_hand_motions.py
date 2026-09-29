#!/usr/bin/env python3
"""Generate canonical motion files (CSV) for Sharpa Wave hand sim/real tracking.

Analogous to `generate_motions.py` (which targets the 7-DOF FR3 arm). Here
we target the 22-DOF hand. Each joint's target is strictly clamped to
`SHARPA_REAL_LIMITS` (the probe-measured reachable range).

Each output pair is {name}.csv + {name}.json:
  - CSV  : header row = joint names (Sharpa/cfg order); subsequent rows = rad
  - JSON : metadata (control_freq_hz, duration_s, description, base_pose)

Ordering: Sharpa / cfg order, same as `real_hand.set_joint_position()` expects
and same as the retargeted opt_dof_pos.

Usage:
    python tools/sysid/generate_hand_motions.py \
        --output_dir tools/sysid/motions_hand \
        --control_freq 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Sharpa/cfg joint order — matches `cfg.actuated_joint_names` minus hand-side prefix
HAND_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP",  "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE",   "ring_MCP_AA",   "ring_PIP",   "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",  "pinky_MCP_AA",  "pinky_PIP",  "pinky_DIP",
]
assert len(HAND_JOINT_SUFFIXES) == 22

# Measured reachable range (normal mode, joint-limit probe)
# Keep in sync with dexx/tasks/franka_sharpa/sim2real/real_hand_limits.py
SHARPA_REAL_LIMITS = np.asarray([
    (-0.087,  1.833),   # 0  thumb_CMC_FE
    (-0.307,  0.061),   # 1  thumb_CMC_AA
    (-0.436,  1.309),   # 2  thumb_MCP_FE
    (-0.307,  0.308),   # 3  thumb_MCP_AA
    ( 0.000,  1.658),   # 4  thumb_IP
    (-0.175,  1.466),   # 5  index_MCP_FE
    (-0.309,  0.106),   # 6  index_MCP_AA
    ( 0.000,  1.658),   # 7  index_PIP
    ( 0.000,  1.309),   # 8  index_DIP
    (-0.175,  1.466),   # 9  middle_MCP_FE
    (-0.136,  0.159),   # 10 middle_MCP_AA
    ( 0.000,  1.658),   # 11 middle_PIP
    ( 0.000,  1.309),   # 12 middle_DIP
    (-0.175,  1.466),   # 13 ring_MCP_FE
    (-0.027,  0.147),   # 14 ring_MCP_AA
    ( 0.000,  1.658),   # 15 ring_PIP
    ( 0.000,  1.309),   # 16 ring_DIP
    ( 0.008,  0.250),   # 17 pinky_CMC
    (-0.175,  1.466),   # 18 pinky_MCP_FE
    (-0.026,  0.306),   # 19 pinky_MCP_AA
    ( 0.000,  1.658),   # 20 pinky_PIP
    ( 0.000,  1.309),   # 21 pinky_DIP
], dtype=np.float64)

LIM_LO = SHARPA_REAL_LIMITS[:, 0]
LIM_HI = SHARPA_REAL_LIMITS[:, 1]
LIM_MID = 0.5 * (LIM_LO + LIM_HI)
LIM_HALFRANGE = 0.5 * (LIM_HI - LIM_LO)

# Comfortable base pose for oscillatory tests. Each joint placed within its
# reachable range, picking "slight flex" for FE/PIP/DIP (0.3-0.4 rad) and 0
# for AA joints. Hand looks slightly closed / relaxed. All values verified
# to be within SHARPA_REAL_LIMITS.
BASE_POSE = np.asarray([
    # thumb
    0.40, 0.00, 0.30, 0.00, 0.40,
    # index
    0.30, 0.00, 0.50, 0.30,
    # middle
    0.30, 0.00, 0.50, 0.30,
    # ring
    0.30, 0.00, 0.50, 0.30,
    # pinky
    0.10,
    0.30, 0.00, 0.50, 0.30,
], dtype=np.float64)
assert BASE_POSE.shape == (22,)

# Per-joint amplitude for sinusoidal tests. Chosen so base±amp stays well
# inside SHARPA_REAL_LIMITS (≥ 0.04 rad margin). AA joints with tight real
# ranges (ring=±0.027, pinky=±0.026) get tiny amplitude.
AMPLITUDE = np.asarray([
    # thumb
    0.20, 0.03, 0.20, 0.10, 0.20,
    # index
    0.20, 0.05, 0.30, 0.20,
    # middle
    0.20, 0.04, 0.30, 0.20,
    # ring
    0.20, 0.01, 0.30, 0.20,       # ring_MCP_AA[14] coupled (±0.027 real range)
    # pinky
    0.05,
    0.20, 0.01, 0.30, 0.20,       # pinky_MCP_AA[19] coupled (±0.026 real range)
], dtype=np.float64)
assert AMPLITUDE.shape == (22,)

# Verify base+amp safe
_hi_check = BASE_POSE + AMPLITUDE
_lo_check = BASE_POSE - AMPLITUDE
for j in range(22):
    assert _hi_check[j] <= LIM_HI[j] - 0.01, (
        f"j{j} ({HAND_JOINT_SUFFIXES[j]}) base+amp "
        f"{_hi_check[j]:.3f} exceeds upper limit {LIM_HI[j]:.3f}")
    assert _lo_check[j] >= LIM_LO[j] + 0.01, (
        f"j{j} ({HAND_JOINT_SUFFIXES[j]}) base-amp "
        f"{_lo_check[j]:.3f} below lower limit {LIM_LO[j]:.3f}")


# --------------------------------------------------------------- helpers

def _min_jerk(t: np.ndarray) -> np.ndarray:
    """5th-order minimum-jerk profile. s(0)=0, s(1)=1, s'=s''=0 at both endpoints."""
    t = np.clip(t, 0.0, 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _ramp_to(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    s = _min_jerk(np.linspace(0.0, 1.0, n))[:, None]
    return (1 - s) * start + s * end


def _fade_envelope(n: int, fade_n: int) -> np.ndarray:
    env = np.ones(n)
    fade_n = min(fade_n, n // 2)
    if fade_n > 0:
        env[:fade_n] = _min_jerk(np.linspace(0.0, 1.0, fade_n))
        env[-fade_n:] = _min_jerk(np.linspace(1.0, 0.0, fade_n))
    return env


def write_motion(out_dir: Path, name: str, positions: np.ndarray,
                 control_freq_hz: float, description: str,
                 meta_extra: dict | None = None):
    assert positions.shape[1] == 22, positions.shape
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"

    with open(csv_path, "w") as f:
        f.write(",".join(HAND_JOINT_SUFFIXES) + "\n")
        for row in positions:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    meta = {
        "name": name,
        "joint_names": HAND_JOINT_SUFFIXES,
        "control_freq_hz": float(control_freq_hz),
        "duration_s": float(len(positions) / control_freq_hz),
        "n_steps": int(len(positions)),
        "base_pose": BASE_POSE.tolist(),
        "description": description,
        "dof": 22,
        "hand": "sharpa_wave",
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    # sanity check
    hi = positions.max(axis=0)
    lo = positions.min(axis=0)
    violations = np.where((lo < LIM_LO - 1e-6) | (hi > LIM_HI + 1e-6))[0]
    tag = "SAFE ✓" if len(violations) == 0 else "UNSAFE ✗"
    print(f"  [{tag}] {name}: {positions.shape[0]} steps @ {control_freq_hz}Hz "
          f"({len(positions)/control_freq_hz:.1f}s) -> {csv_path.name}")
    if len(violations) > 0:
        for j in violations:
            print(f"    J{j} ({HAND_JOINT_SUFFIXES[j]}) "
                  f"range [{lo[j]:.3f}, {hi[j]:.3f}] violates limits "
                  f"[{LIM_LO[j]:.3f}, {LIM_HI[j]:.3f}]")


def _prepend_approach(positions: np.ndarray, approach_s: float,
                      settle_s: float, freq: float) -> np.ndarray:
    n_appr = max(int(approach_s * freq), 1)
    n_settle = int(settle_s * freq)
    first_row = positions[0]
    approach = _ramp_to(BASE_POSE, first_row, n_appr)
    settle = np.tile(first_row, (n_settle, 1))
    return np.vstack([approach, settle, positions])


def _append_return(positions: np.ndarray, return_s: float,
                   settle_s: float, freq: float) -> np.ndarray:
    n_ret = max(int(return_s * freq), 1)
    n_settle = int(settle_s * freq)
    last_row = positions[-1]
    ret = _ramp_to(last_row, BASE_POSE, n_ret)
    settle = np.tile(BASE_POSE, (n_settle, 1))
    return np.vstack([positions, ret, settle])


# --------------------------------------------------------------- motions

def gen_sin_all_fingers(freq: float, duration_s: float = 20.0,
                        f_hz: float = 0.5, fade_s: float = 1.5) -> np.ndarray:
    """All 22 joints sinusoidally. Each joint with its own amplitude (from AMPLITUDE
    table) and a small phase offset so they don't all peak together. Best single
    test for hand PD matching.
    """
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(BASE_POSE, (n, 1))
    for j in range(22):
        # phase shift across joints
        phase = (j * np.pi / 8.0)
        positions[:, j] += AMPLITUDE[j] * env * np.sin(2 * np.pi * f_hz * t + phase)
    return positions


def gen_chirp_all_fingers(freq: float, duration_s: float = 25.0,
                          f_start: float = 0.2, f_end: float = 1.5,
                          fade_s: float = 2.0) -> np.ndarray:
    """Linear frequency chirp on all 22 joints. Amplitude per joint from
    AMPLITUDE table, clipped so base±amp stays within SHARPA_REAL_LIMITS."""
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    f_inst = f_start + (f_end - f_start) * (t / duration_s)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(BASE_POSE, (n, 1))
    for j in range(22):
        phase_off = j * np.pi / 8.0
        positions[:, j] += AMPLITUDE[j] * env * np.sin(phase + phase_off)
    return positions


def gen_finger_flex_sequence(freq: float,
                             flex_amp: float = 0.6,
                             hold_s: float = 1.0, ramp_s: float = 0.6,
                             settle_s: float = 0.5) -> np.ndarray:
    """Sequentially flex each of 5 fingers. For each finger, drive its MCP_FE,
    PIP, DIP (and IP for thumb) from base toward maximum safe flex, hold,
    return, settle. Clean per-finger response characterization."""
    # (finger_name, list of joint indices that constitute that finger's flex axis)
    fingers = [
        ("thumb",  [0, 2, 4]),         # CMC_FE, MCP_FE, IP
        ("index",  [5, 7, 8]),         # MCP_FE, PIP, DIP
        ("middle", [9, 11, 12]),
        ("ring",   [13, 15, 16]),
        ("pinky",  [18, 20, 21]),
    ]
    rows = [BASE_POSE.copy()]
    n_ramp = max(int(ramp_s * freq), 1)
    n_hold = int(hold_s * freq)
    n_settle = int(settle_s * freq)
    for name, joint_idxs in fingers:
        target = BASE_POSE.copy()
        for j in joint_idxs:
            safe_high = min(BASE_POSE[j] + flex_amp, LIM_HI[j] - 0.02)
            target[j] = safe_high
        # BASE → flexed
        rows.extend(_ramp_to(BASE_POSE.copy(), target, n_ramp))
        rows.extend([target] * n_hold)
        # flexed → BASE
        rows.extend(_ramp_to(target, BASE_POSE.copy(), n_ramp))
        rows.extend([BASE_POSE.copy()] * n_settle)
    return np.asarray(rows)


def gen_grasp_cycle(freq: float, cycles: int = 4,
                    period_s: float = 2.0, fade_s: float = 1.5) -> np.ndarray:
    """Full-hand grasp/release cycles — all 5 fingers flex in phase. Tests
    that the hand can do coordinated motions (relevant for manipulation)."""
    n = int(cycles * period_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    # 0 at t=0, 1 at peak flex (cosine ramp)
    phase = 0.5 * (1 - np.cos(2 * np.pi * t / period_s))  # [0, 1]
    flex_fraction = env * phase
    positions = np.tile(BASE_POSE, (n, 1))
    # Drive the long flexors and distals up to ~70% of their upper limit when
    # fully grasped. AA joints barely move.
    flex_joint_idxs = [0, 2, 4,     # thumb flexors
                       5, 7, 8,     # index
                       9, 11, 12,   # middle
                       13, 15, 16,  # ring
                       18, 20, 21]  # pinky
    for j in flex_joint_idxs:
        safe_high = LIM_HI[j] - 0.05
        reach = safe_high - BASE_POSE[j]
        positions[:, j] = BASE_POSE[j] + flex_fraction * reach
    return positions


def gen_thumb_opposition(freq: float, duration_s: float = 16.0,
                         f_hz: float = 0.3, fade_s: float = 1.5) -> np.ndarray:
    """Thumb CMC oscillation while index curls — exercises thumb opposition
    in real hand, which is often the hardest to retarget accurately."""
    n = int(duration_s * freq)
    t = np.arange(n) / freq
    env = _fade_envelope(n, int(fade_s * freq))
    positions = np.tile(BASE_POSE, (n, 1))
    # thumb CMC_FE sweeps forward/back, CMC_AA sweeps sideways
    positions[:, 0] += 0.25 * env * np.sin(2 * np.pi * f_hz * t)              # thumb_CMC_FE
    positions[:, 1] += 0.025 * env * np.sin(2 * np.pi * f_hz * t + np.pi/2)   # thumb_CMC_AA (tiny)
    # index curls in counter-phase
    positions[:, 5] += 0.30 * env * np.sin(2 * np.pi * f_hz * t + np.pi)      # index_MCP_FE
    positions[:, 7] += 0.30 * env * np.sin(2 * np.pi * f_hz * t + np.pi)      # index_PIP
    return positions


# --------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str,
                   default=str(Path(__file__).resolve().parent / "motions_hand"))
    p.add_argument("--control_freq", type=float, default=30.0)
    p.add_argument("--approach_s", type=float, default=3.0)
    p.add_argument("--settle_s", type=float, default=0.8)
    p.add_argument("--return_s", type=float, default=3.0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    freq = args.control_freq
    print(f"Generating hand motions @ {freq} Hz into {out_dir}")
    print(f"Base pose (22d, cfg order): {BASE_POSE.round(2).tolist()}")

    motions = {
        "hand_sin_all": (
            gen_sin_all_fingers(freq),
            "All 22 joints in phase-shifted sinusoids at 0.5 Hz. Per-joint "
            "amplitudes clamped to real-hand reachable range.",
        ),
        "hand_chirp_all": (
            gen_chirp_all_fingers(freq),
            "Linear frequency chirp 0.2->1.5 Hz on all 22 joints. Single-test "
            "Bode characterization.",
        ),
        "hand_flex_sequence": (
            gen_finger_flex_sequence(freq),
            "Sequentially flex each of 5 fingers; clean per-finger response "
            "curves without cross-finger coupling.",
        ),
        "hand_grasp_cycle": (
            gen_grasp_cycle(freq),
            "4x full grasp/release cycles @ 0.5 Hz. Tests coordinated closing.",
        ),
        "hand_thumb_opposition": (
            gen_thumb_opposition(freq),
            "Thumb oscillates while index curls in counter-phase. Tests "
            "opposition — often the weakest joint in retargeting.",
        ),
    }

    for name, (positions, description) in motions.items():
        positions = _prepend_approach(positions, args.approach_s, args.settle_s, freq)
        positions = _append_return(positions, args.return_s, args.settle_s, freq)
        write_motion(out_dir, name, positions, freq, description)

    print(f"\nGenerated {len(motions)} hand motions.")
    print("Next steps:")
    print(f"  python tools/sysid/replay_hand_motion_real.py "
          f"--motion {out_dir}/hand_sin_all.csv --output logs/system_id/hand_replay/hand_sin_all_real.pkl")
    print(f"  python tools/sysid/replay_hand_motion_sim.py  "
          f"--motion {out_dir}/hand_sin_all.csv --output logs/system_id/hand_replay/hand_sin_all_sim.pkl --headless")
    print(f"  python tools/sysid/analyze_motion.py         "
          f"--real ... --sim ... --out logs/system_id/hand_replay/compare_hand_sin_all")


if __name__ == "__main__":
    main()
