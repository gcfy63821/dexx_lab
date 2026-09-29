#!/usr/bin/env python3
"""SAGE-style sim-vs-real motion-replay analyzer.

Consumes the pkls produced by `replay_motion_sim.py` and `replay_motion_polymetis.py`
(or any compatible {timestamps, targets, actuals, velocities, joint_names} pkl)
and produces per-joint metrics + overlay plots.

Metrics (one row per joint):
    rmse         — position RMSE (rad) between sim and real actuals
    mape         — mean absolute percentage error (%, vs step amplitude)
    corr         — Pearson correlation of actuals over time
    cos_sim      — cosine similarity on zero-mean actuals
    max_lag_ms   — argmax of cross-correlation — NEGATIVE = sim lags real,
                   POSITIVE = sim leads real
    rmse_sim_tracking  — |sim_actual − target| — how well sim tracks its own target
    rmse_real_tracking — |real_actual − target| — how well real tracks its own target

Ported (loosely) from sage/sage/analysis.py.

Usage:
    python tools/sysid/analyze_motion.py \
        --sim  logs/system_id/motion_replay/chirp_sweep_sim.pkl \
        --real logs/system_id/motion_replay/chirp_sweep_real.pkl \
        --out  logs/system_id/motion_replay/compare_chirp
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def _resample(t_src: np.ndarray, x_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear-interpolate x_src (T, D) onto t_dst. From SAGE's _resample_waveform."""
    out = np.empty((len(t_dst), x_src.shape[1]), dtype=np.float64)
    for d in range(x_src.shape[1]):
        out[:, d] = np.interp(t_dst, t_src, x_src[:, d])
    return out


def _align(sim: dict, real: dict):
    """Align sim + real onto a common time axis using their own timestamps.

    Both sides start at t=0 (since each pkl timestamps are offsets from start
    of replay). Truncates to the shorter run.
    """
    t_end = min(sim["timestamps"][-1], real["timestamps"][-1])
    # Resample at ~ record_freq of sim to preserve detail
    rec_freq = float(sim.get("record_freq_hz", 100.0))
    n = int(t_end * rec_freq) + 1
    t_common = np.linspace(0, t_end, n)

    sim_act = _resample(sim["timestamps"], sim["actuals"], t_common)
    real_act = _resample(real["timestamps"], real["actuals"], t_common)
    sim_tgt = _resample(sim["timestamps"], sim["targets"], t_common)
    real_tgt = _resample(real["timestamps"], real["targets"], t_common)
    return t_common, sim_act, real_act, sim_tgt, real_tgt


def _metrics_per_joint(t: np.ndarray, sim_act: np.ndarray, real_act: np.ndarray,
                       sim_tgt: np.ndarray, real_tgt: np.ndarray) -> list[dict]:
    from scipy import signal
    from scipy.spatial import distance

    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    N_joints = sim_act.shape[1]
    # A joint is "active" iff its target range exceeds INACTIVE_RANGE_THRESH.
    # Below this, the joint is essentially stationary and cross-correlation /
    # waveform similarity metrics become meaningless (noise-dominated).
    INACTIVE_RANGE_THRESH = 0.02  # rad (~1.15°)
    rows = []
    for j in range(N_joints):
        s = sim_act[:, j]
        r = real_act[:, j]
        err = s - r
        tgt_range = float(max(sim_tgt[:, j].max() - sim_tgt[:, j].min(),
                              real_tgt[:, j].max() - real_tgt[:, j].min()))
        is_active = tgt_range >= INACTIVE_RANGE_THRESH

        # 1. RMSE
        rmse = float(np.sqrt(np.mean(err ** 2)))

        # 2. MAPE — normalize by range of actual motion (not raw %, which blows
        # up at rest). This is the SAGE convention with a small-motion guard.
        rng = max(r.max() - r.min(), 0.05)
        mape = float(np.mean(np.abs(err)) / rng * 100.0)

        # 3. Pearson correlation
        if r.std() > 1e-9 and s.std() > 1e-9:
            corr = float(np.corrcoef(s, r)[0, 1])
        else:
            corr = float("nan")

        # 4. Cosine similarity (on zero-mean signals)
        s0 = s - s.mean()
        r0 = r - r.mean()
        if np.linalg.norm(s0) > 1e-9 and np.linalg.norm(r0) > 1e-9:
            cos_sim = float(1.0 - distance.cosine(s0, r0))
        else:
            cos_sim = float("nan")

        # 5. Cross-correlation → time lag
        # Convention: signal.correlate(real, sim) peaks at lag k where
        # real[n+k] ≈ sim[n]. So k<0 means real's peak occurs EARLIER than
        # sim's → real leads, sim lags. Equivalently:
        #   max_lag_ms < 0  →  sim LAGS real (sim is delayed)
        #   max_lag_ms > 0  →  sim LEADS real (sim is early)
        # To fix sim-lags-real (negative lag), REDUCE sim damping or increase
        # sim stiffness so sim responds faster. action_delay_steps can only
        # ADD delay to sim, so it only helps when lag is positive.
        s_norm = (s - s.mean()) / (s.std() + 1e-9)
        r_norm = (r - r.mean()) / (r.std() + 1e-9)
        cc = signal.correlate(r_norm, s_norm, mode="full")
        lags = signal.correlation_lags(len(r_norm), len(s_norm))
        max_lag_samples = int(lags[int(np.argmax(cc))])
        max_lag_ms = float(max_lag_samples * dt * 1000.0)

        # 6. Tracking error per side (distance of actual to its own target)
        rmse_sim_track = float(np.sqrt(np.mean((sim_act[:, j] - sim_tgt[:, j]) ** 2)))
        rmse_real_track = float(np.sqrt(np.mean((real_act[:, j] - real_tgt[:, j]) ** 2)))

        # For inactive joints, cross-correlation and waveform metrics are
        # dominated by measurement noise. Null them out so they don't mislead.
        if not is_active:
            corr = float("nan")
            cos_sim = float("nan")
            max_lag_ms = float("nan")
            max_lag_samples = 0

        rows.append({
            "joint_idx": j,
            "active": bool(is_active),
            "tgt_range_rad": tgt_range,
            "rmse_rad": rmse,
            "rmse_deg": float(np.degrees(rmse)),
            "mape_pct": mape,
            "corr": corr,
            "cos_sim": cos_sim,
            "max_lag_ms": max_lag_ms,
            "max_lag_samples": max_lag_samples,
            "rmse_sim_tracking_rad": rmse_sim_track,
            "rmse_real_tracking_rad": rmse_real_track,
            "range_rad": float(rng),
        })
    return rows


def _print_table(rows: list[dict], joint_names: list[str]):
    print()
    print(f"{'joint':<14} {'active':>6} {'rmse(°)':>8} {'MAPE%':>7} "
          f"{'corr':>6} {'cos':>6} {'lag(ms)':>9} "
          f"{'sim_trk':>9} {'real_trk':>9}")
    print("-" * 86)
    for r, name in zip(rows, joint_names):
        act_mark = "yes" if r.get("active", True) else "no"
        corr_str = "   —  " if np.isnan(r["corr"]) else f"{r['corr']:>6.3f}"
        cos_str = "   —  " if np.isnan(r["cos_sim"]) else f"{r['cos_sim']:>6.3f}"
        lag_str = "    —    " if np.isnan(r["max_lag_ms"]) else f"{r['max_lag_ms']:>+9.1f}"
        print(f"{name:<14} {act_mark:>6} {r['rmse_deg']:>8.3f} {r['mape_pct']:>7.2f} "
              f"{corr_str} {cos_str} {lag_str} "
              f"{np.degrees(r['rmse_sim_tracking_rad']):>9.3f} "
              f"{np.degrees(r['rmse_real_tracking_rad']):>9.3f}")
    print("-" * 86)
    active_rows = [r for r in rows if r.get("active", True)]
    if active_rows:
        mean_lag = np.nanmean([r["max_lag_ms"] for r in active_rows])
        print(f"{'ACTIVE MEAN':<14} {'':>6} "
              f"{np.mean([r['rmse_deg'] for r in active_rows]):>8.3f} "
              f"{np.mean([r['mape_pct'] for r in active_rows]):>7.2f} "
              f"{np.nanmean([r['corr'] for r in active_rows]):>6.3f} "
              f"{np.nanmean([r['cos_sim'] for r in active_rows]):>6.3f} "
              f"{mean_lag:>+9.1f}")
    else:
        print("(no active joints)")
    print()
    print("Legend:")
    print("  rmse(mm°) — RMSE between sim-actual and real-actual (degrees)")
    print("  MAPE%     — mean abs err / range of real motion, percent")
    print("  corr/cos  — waveform similarity (1.0 = identical shape)")
    print("  lag(ms)   — negative = sim LAGS real (reduce sim damping to fix);")
    print("              positive = sim LEADS real (add action_delay_steps=round(+lag/ctrl_dt))")
    print("  sim_trk   — sim actual vs its own target (how well sim's PD tracks)")
    print("  real_trk  — real actual vs its own target (how well real controller tracks)")


def _plot_overlay(out_dir: Path, t: np.ndarray,
                  sim_act: np.ndarray, real_act: np.ndarray,
                  sim_tgt: np.ndarray, real_tgt: np.ndarray,
                  rows: list[dict], joint_names: list[str]):
    if not HAS_MPL:
        print("[WARN] matplotlib not available, skipping plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use only ACTIVE joints in plot so 22-DOF hand results aren't smothered
    # by 18 flat-line rows. Fall back to all joints if none are active.
    active_j = [j for j, r in enumerate(rows) if r.get("active", True)]
    if not active_j:
        active_j = list(range(len(rows)))
    N = len(active_j)

    # figure height scales with #joints (min 6in, max 28in)
    h = max(6.0, min(28.0, 2.3 * N))
    fig, axs = plt.subplots(N, 1, figsize=(12, h), sharex=True)
    if N == 1:
        axs = [axs]
    for k, j in enumerate(active_j):
        ax = axs[k]
        ax.plot(t, np.degrees(sim_tgt[:, j]), ls="--", color="gray", lw=0.8,
                label="target" if k == 0 else None)
        ax.plot(t, np.degrees(sim_act[:, j]), color="tab:blue", lw=1.2,
                label="sim" if k == 0 else None)
        ax.plot(t, np.degrees(real_act[:, j]), color="tab:red", lw=1.2,
                label="real" if k == 0 else None)
        r = rows[j]
        ax.set_ylabel(f"{joint_names[j]}\ndeg", fontsize=9)
        lag_str = "—" if np.isnan(r["max_lag_ms"]) else f"{r['max_lag_ms']:+.1f}ms"
        corr_str = "—" if np.isnan(r["corr"]) else f"{r['corr']:.3f}"
        ax.set_title(
            f"{joint_names[j]}: rmse={r['rmse_deg']:.2f}° "
            f"MAPE={r['mape_pct']:.1f}%  corr={corr_str}  lag={lag_str}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("time (s)")
    axs[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save_path = out_dir / "overlay_all_joints.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"[OK] Saved {save_path}")

    # Per-joint error plot (same N-active layout)
    fig2, axs2 = plt.subplots(N, 1, figsize=(12, h * 0.78), sharex=True)
    if N == 1:
        axs2 = [axs2]
    for k, j in enumerate(active_j):
        ax = axs2[k]
        err_sim_track = np.degrees(sim_act[:, j] - sim_tgt[:, j])
        err_real_track = np.degrees(real_act[:, j] - real_tgt[:, j])
        err_sim_real = np.degrees(sim_act[:, j] - real_act[:, j])
        ax.plot(t, err_sim_track, color="tab:blue", lw=0.8,
                label="sim - target" if k == 0 else None)
        ax.plot(t, err_real_track, color="tab:red", lw=0.8,
                label="real - target" if k == 0 else None)
        ax.plot(t, err_sim_real, color="black", lw=0.9,
                label="sim - real" if k == 0 else None)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{joint_names[j]}\nerr deg", fontsize=9)
        ax.grid(alpha=0.3)
    axs2[-1].set_xlabel("time (s)")
    axs2[0].legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    save_path = out_dir / "errors_per_joint.png"
    fig2.savefig(save_path, dpi=140)
    plt.close(fig2)
    print(f"[OK] Saved {save_path}")


def _save_csv(out_dir: Path, rows: list[dict], joint_names: list[str]):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "metrics.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint_name"] + list(rows[0].keys()))
        for r, name in zip(rows, joint_names):
            w.writerow([name] + [r[k] for k in r.keys()])
    print(f"[OK] Saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, type=str)
    ap.add_argument("--real", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    args = ap.parse_args()

    with open(args.sim, "rb") as f:
        sim = pickle.load(f)
    with open(args.real, "rb") as f:
        real = pickle.load(f)

    joint_names = sim.get("joint_names", real.get("joint_names"))
    print(f"[INFO] sim  : {len(sim['timestamps'])} frames, "
          f"{sim['timestamps'][-1]:.2f}s, motion={sim.get('motion_meta',{}).get('name','?')}")
    print(f"[INFO] real : {len(real['timestamps'])} frames, "
          f"{real['timestamps'][-1]:.2f}s, motion={real.get('motion_meta',{}).get('name','?')}")

    t, sim_act, real_act, sim_tgt, real_tgt = _align(sim, real)
    rows = _metrics_per_joint(t, sim_act, real_act, sim_tgt, real_tgt)

    _print_table(rows, joint_names)
    out_dir = Path(args.out)
    _save_csv(out_dir, rows, joint_names)
    _plot_overlay(out_dir, t, sim_act, real_act, sim_tgt, real_tgt, rows, joint_names)

    # Highlight suggestion based on ACTIVE-joint mean lag (ignore inactive
    # joints, whose cross-correlation is noise-dominated).
    active_rows = [r for r in rows if r.get("active", True)]
    print()
    print("=" * 60)
    if active_rows:
        mean_lag_ms = float(np.nanmean([r["max_lag_ms"] for r in active_rows]))
        ctrl_dt_ms = 1000.0 / sim.get("control_freq_hz", 30.0)
        print(f"  active joints     : {len(active_rows)}/{len(rows)} "
              f"(indices {[r['joint_idx'] for r in active_rows]})")
        print(f"  mean sim↔real lag : {mean_lag_ms:+.1f} ms (active joints only)")
        print(f"  control period    : {ctrl_dt_ms:.1f} ms")
        if mean_lag_ms >= 0:
            # Sim leads real — adding a delay to sim can close the gap.
            suggested_delay = int(round(mean_lag_ms / ctrl_dt_ms))
            print(f"  sim LEADS real → suggested action_delay_steps : {suggested_delay}")
        else:
            # Sim lags real — cannot be fixed by action_delay. Need to make
            # sim respond faster (reduce damping or increase stiffness).
            print(f"  sim LAGS real  → reduce sim damping on the slow joints")
            print(f"                    (action_delay won't help; only makes it worse)")
    else:
        print("  [WARN] no joint met the active threshold; metrics may be noise")
    print("=" * 60)


if __name__ == "__main__":
    main()
