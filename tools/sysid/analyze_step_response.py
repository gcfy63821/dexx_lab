#!/usr/bin/env python3
"""Analyze and compare step response data from sim and real.

Produces per-joint comparison plots and summary metrics.

Usage:
    # Compare sim vs real
    python tools/sysid/analyze_step_response.py \
        --real logs/system_id/step_response_real.pkl \
        --sim logs/system_id/step_response_sim.pkl

    # Real only
    python tools/sysid/analyze_step_response.py \
        --real logs/system_id/step_response_real.pkl

    # Sim only
    python tools/sysid/analyze_step_response.py \
        --sim logs/system_id/step_response_sim.pkl
"""

import argparse
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(timestamps, target, actual, step_size):
    """Compute step response metrics for a single joint.

    Within a segment, target is constant (the post-step value).
    We use actual[0] as the pre-step baseline to compute the step response.
    """
    empty = {"steady_state_error": 0, "overshoot_pct": 0, "rise_time": 0, "settling_time": 0,
             "peak_value_pct": 0, "time_to_peak": 0, "final_value_pct": 0,
             "pos_at_100ms": 0, "pos_at_200ms": 0, "pos_at_500ms": 0, "pos_at_1000ms": 0}
    if len(timestamps) == 0 or len(actual) == 0:
        return empty

    target_final = target[-1]
    actual_initial = actual[0]
    step = abs(target_final - actual_initial)

    if step < 1e-6:
        return empty

    direction = 1.0 if target_final > actual_initial else -1.0

    # Normalize response: 0 = initial position, 1 = target reached
    normalized = direction * (actual - actual_initial) / step

    # Rise time (10% to 90%)
    rise_time = 0
    t10 = None
    t90 = None
    for i, v in enumerate(normalized):
        if t10 is None and v >= 0.1:
            t10 = timestamps[i]
        if t90 is None and v >= 0.9:
            t90 = timestamps[i]
            break
    if t10 is not None and t90 is not None:
        rise_time = t90 - t10

    # Peak value and time to peak
    peak_idx = np.argmax(normalized)
    peak_value_pct = normalized[peak_idx] * 100
    time_to_peak = timestamps[peak_idx]

    # Overshoot
    overshoot_pct = max(0, (normalized[peak_idx] - 1.0)) * 100

    # Steady state error
    steady_actual = actual[-10:].mean()
    steady_err = abs(steady_actual - target_final)

    # Final value as percentage of step
    final_value_pct = normalized[-10:].mean() * 100

    # Position at key timestamps
    def pos_at_time(t_ms):
        t_s = t_ms / 1000.0
        idx = np.searchsorted(timestamps, t_s)
        if idx < len(normalized):
            return normalized[idx] * 100
        return normalized[-1] * 100

    pos_100ms = pos_at_time(100)
    pos_200ms = pos_at_time(200)
    pos_500ms = pos_at_time(500)
    pos_1000ms = pos_at_time(1000)

    # Settling time (within 2% of final value)
    settling_time = 0
    threshold = 0.02
    for i in range(len(normalized) - 1, -1, -1):
        if abs(normalized[i] - 1.0) > threshold:
            if i + 1 < len(timestamps):
                settling_time = timestamps[i + 1]
            break

    return {
        "rise_time": rise_time,
        "overshoot_pct": overshoot_pct,
        "steady_state_error": steady_err,
        "settling_time": settling_time,
        "peak_value_pct": peak_value_pct,
        "time_to_peak": time_to_peak,
        "final_value_pct": final_value_pct,
        "pos_at_100ms": pos_100ms,
        "pos_at_200ms": pos_200ms,
        "pos_at_500ms": pos_500ms,
        "pos_at_1000ms": pos_1000ms,
    }


def plot_joint(ax, j, real_data, sim_data, step_size, phase="step_up"):
    """Plot sim vs real step response for one joint."""
    ax.set_title(f"Joint {j} (fr3_joint{j+1}) — {phase}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")

    if real_data is not None and phase in real_data:
        d = real_data[phase]
        t = d["timestamps"]
        ax.plot(t, d["targets"][:, j], 'r--', alpha=0.5, label="target")
        ax.plot(t, d["actuals"][:, j], 'r-', linewidth=2, label="real actual")

    if sim_data is not None and phase in sim_data:
        d = sim_data[phase]
        t = d["timestamps"]
        ax.plot(t, d["actuals"][:, j], 'b-', linewidth=2, label="sim actual")
        if real_data is None:
            ax.plot(t, d["targets"][:, j], 'b--', alpha=0.5, label="target")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Analyze step response data")
    parser.add_argument("--real", type=str, default=None, help="Real step response pkl")
    parser.add_argument("--sim", type=str, default=None, help="Sim step response pkl")
    parser.add_argument("--output_dir", type=str, default="logs/system_id/plots")
    args = parser.parse_args()

    if args.real is None and args.sim is None:
        print("ERROR: Provide at least --real or --sim")
        return

    real = None
    sim = None
    if args.real:
        with open(args.real, "rb") as f:
            real = pickle.load(f)
        print(f"Real data: {args.real}")
    if args.sim:
        with open(args.sim, "rb") as f:
            sim = pickle.load(f)
        print(f"Sim data: {args.sim}")

    # Determine which joints were tested (normalize keys to int)
    if real:
        real["joints"] = {int(k): v for k, v in real["joints"].items()}
    if sim:
        sim["joints"] = {int(k): v for k, v in sim["joints"].items()}

    joints = set()
    if real:
        joints.update(real["joints"].keys())
    if sim:
        joints.update(sim["joints"].keys())
    joints = sorted(joints)

    step_size = real["step_size"] if real else sim["step_size"]
    print(f"\nStep size: {step_size} rad")
    print(f"Joints tested: {joints}")
    if real:
        print(f"  Real joint keys: {sorted(real['joints'].keys())}")
    if sim:
        print(f"  Sim joint keys: {sorted(sim['joints'].keys())}")
    if real and sim:
        common = sorted(set(real["joints"].keys()) & set(sim["joints"].keys()))
        print(f"  Common joints: {common}")

    # Print basic metrics table
    print(f"\n{'='*90}")
    print(f"{'Joint':>6} | {'Source':>6} | {'Rise(ms)':>10} | {'Overshoot%':>10} | {'SS Error(deg)':>13} | {'Settle(ms)':>10}")
    print(f"{'-'*90}")

    for j in joints:
        for source_name, data in [("Real", real), ("Sim", sim)]:
            if data is None or j not in data["joints"]:
                continue
            d = data["joints"][j]["step_up"]
            m = compute_metrics(d["timestamps"], d["targets"][:, j], d["actuals"][:, j], step_size)
            print(f"  J{j:>3} | {source_name:>6} | {m['rise_time']*1000:>8.1f}ms | {m['overshoot_pct']:>9.1f}% | {np.degrees(m['steady_state_error']):>11.2f}° | {m['settling_time']*1000:>8.1f}ms")

    # Print detailed time-domain comparison
    print(f"\n{'='*110}")
    print(f"{'Joint':>6} | {'Source':>6} | {'Peak%':>7} | {'t_peak(ms)':>10} | {'Final%':>7} | {'@100ms':>7} | {'@200ms':>7} | {'@500ms':>7} | {'@1000ms':>8}")
    print(f"{'-'*110}")

    for j in joints:
        for source_name, data in [("Real", real), ("Sim", sim)]:
            if data is None or j not in data["joints"]:
                continue
            d = data["joints"][j]["step_up"]
            m = compute_metrics(d["timestamps"], d["targets"][:, j], d["actuals"][:, j], step_size)
            print(f"  J{j:>3} | {source_name:>6} | {m['peak_value_pct']:>6.1f}% | {m['time_to_peak']*1000:>8.1f}ms | {m['final_value_pct']:>6.1f}% | {m['pos_at_100ms']:>6.1f}% | {m['pos_at_200ms']:>6.1f}% | {m['pos_at_500ms']:>6.1f}% | {m['pos_at_1000ms']:>7.1f}%")

    # Print tuning suggestions (only if both real and sim available)
    if real and sim:
        print(f"\n{'='*90}")
        print(f"  TUNING SUGGESTIONS (step_up phase)")
        print(f"{'-'*90}")
        for j in sorted(set(real["joints"].keys()) & set(sim["joints"].keys())):
            rd = real["joints"][j]["step_up"]
            sd = sim["joints"][j]["step_up"]
            rm = compute_metrics(rd["timestamps"], rd["targets"][:, j], rd["actuals"][:, j], step_size)
            sm = compute_metrics(sd["timestamps"], sd["targets"][:, j], sd["actuals"][:, j], step_size)

            suggestions = []
            # Compare final value
            real_final = rm["final_value_pct"]
            sim_final = sm["final_value_pct"]
            if sim_final < real_final - 5:
                suggestions.append(f"K↑ (sim final {sim_final:.0f}% < real {real_final:.0f}%)")
            elif sim_final > real_final + 5:
                suggestions.append(f"K↓ (sim final {sim_final:.0f}% > real {real_final:.0f}%)")

            # Compare speed at 200ms
            real_200 = rm["pos_at_200ms"]
            sim_200 = sm["pos_at_200ms"]
            if sim_200 > real_200 + 10:
                suggestions.append(f"D↑ (sim @200ms {sim_200:.0f}% >> real {real_200:.0f}%)")
            elif sim_200 < real_200 - 10:
                suggestions.append(f"D↓ or K↑ (sim @200ms {sim_200:.0f}% << real {real_200:.0f}%)")

            # Compare overshoot
            if sm["overshoot_pct"] > rm["overshoot_pct"] + 10:
                suggestions.append(f"D↑ (sim overshoot {sm['overshoot_pct']:.0f}% > real {rm['overshoot_pct']:.0f}%)")

            # SS error comparison
            sim_ss = np.degrees(sm["steady_state_error"])
            real_ss = np.degrees(rm["steady_state_error"])
            if sim_ss > real_ss + 0.3:
                suggestions.append(f"K↑ (sim SS {sim_ss:.2f}° > real {real_ss:.2f}°)")

            status = "✅" if len(suggestions) == 0 else "⚠️"
            print(f"  J{j}: {status} {'; '.join(suggestions) if suggestions else 'Good match'}")

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)

    for phase in ["step_up", "step_down", "step_neg"]:
        n_joints = len(joints)
        fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3 * n_joints), squeeze=False)
        fig.suptitle(f"Step Response Comparison — {phase}", fontsize=14)

        for i, j in enumerate(joints):
            real_j = real["joints"].get(j) if real else None
            sim_j = sim["joints"].get(j) if sim else None
            plot_joint(axes[i, 0], j, real_j, sim_j, step_size, phase)

        plt.tight_layout()
        path = os.path.join(args.output_dir, f"step_response_{phase}.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    # Summary comparison plot
    if real and sim:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Sim vs Real Summary", fontsize=14)

        metrics_names = ["rise_time", "overshoot_pct", "steady_state_error", "settling_time"]
        metric_labels = ["Rise Time (ms)", "Overshoot (%)", "Steady-State Error (deg)", "Settling Time (ms)"]
        metric_scale = [1000, 1, np.degrees(1), 1000]

        for idx, (m_name, m_label, m_scale) in enumerate(zip(metrics_names, metric_labels, metric_scale)):
            ax = axes[idx // 2, idx % 2]
            real_vals = []
            sim_vals = []
            joint_labels = []
            for j in joints:
                if j in real["joints"] and j in sim["joints"]:
                    rd = real["joints"][j]["step_up"]
                    sd = sim["joints"][j]["step_up"]
                    rm = compute_metrics(rd["timestamps"], rd["targets"][:, j], rd["actuals"][:, j], step_size)
                    sm = compute_metrics(sd["timestamps"], sd["targets"][:, j], sd["actuals"][:, j], step_size)
                    real_vals.append(rm[m_name] * m_scale)
                    sim_vals.append(sm[m_name] * m_scale)
                    joint_labels.append(f"J{j}")

            x = np.arange(len(joint_labels))
            w = 0.35
            ax.bar(x - w/2, real_vals, w, label="Real", color="red", alpha=0.7)
            ax.bar(x + w/2, sim_vals, w, label="Sim", color="blue", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(joint_labels)
            ax.set_ylabel(m_label)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(args.output_dir, "sim_vs_real_summary.png")
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
