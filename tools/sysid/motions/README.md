# Arm motion files

Franka FR3 joint-target trajectories for measuring the sim/real action-tracking
gap. Ported from the [SAGE](https://github.com/NVIDIA-Isaac-Sim/sage) project
(`motion_files/so101/custom/*`) and adapted to the 7-DOF FR3. The procedure that
uses them — replay in sim and on the arm, diff, read the metrics — is
[tutorial/07](../../../tutorial/07_dynamics_alignment/). The hand equivalents are
in `../motions_hand/`.

## Format

Each motion is two files:

| file | content |
|---|---|
| `{name}.csv` | header row = joint names (`fr3_joint1..7`); each following row = one target sample (rad) |
| `{name}.json` | metadata — `name`, `joint_names`, `control_freq_hz`, `duration_s`, `n_steps`, `base_pose`, `description`, `safety_checked` |

Rows are at `control_freq_hz` (30 Hz, matching deploy). Every motion starts and
ends at `base_pose` `[0, 0, 0, -1.57, 0, 1.57, 0]`, so motions chain and re-enter
safely.

## Motions

| name | purpose |
|---|---|
| `chirp_sweep` | 0.2 → 3 Hz linear chirp, all joints phase-shifted. The best single test — a full Bode sweep from one run |
| `step_per_joint` | step up / centre / down / centre on each joint in turn — rise time and overshoot |
| `sin_j1`, `sin_j2`, `sin_j3`, `sin_j4`, `sin_j6` | pure sinusoid on one joint — the cleanest single-joint signal |
| `circular_wrist` | joints 5 and 6 in a circle — wrist backlash / hysteresis |
| `backlash_detection` | small reversals per joint — dead-band and stiction |
| `diagonal_sweep` | all 7 joints in-phase sinusoid — coordinated-motion stress test |
| `coupled_joints` | anti-phase pairs (j1,-j3), (j2,-j4), (j5,-j7) — cross-joint coupling |

## Safety envelope

`generate_motions.py` checks every motion against **70% of the FR3 position,
velocity and acceleration limits** (datasheet values in `FR3_VEL_LIMIT` /
`FR3_ACC_LIMIT`) and prints per-joint peaks:

- **position:** at least 20 mrad from every joint limit (joint 4 is one-sided
  `[-3.04, -0.15]`, joint 6 is `[0.54, 4.52]`);
- **velocity:** peak < 0.7 × spec (j1–4: 2.175 rad/s, j5–7: 2.61 rad/s);
- **acceleration:** peak < 0.7 × spec (j2 is the tightest: 7.5 rad/s² → 5.25 usable).

Segment transitions use minimum-jerk profiles, and sinusoids are wrapped in
minimum-jerk fade-in/out envelopes, so every motion starts and stops at zero
velocity and acceleration. `replay_motion_polymetis.py` re-runs a preflight check
and refuses to command a motion that fails it (`--force_unsafe` overrides; don't).

Regenerate after editing amplitudes, frequencies or durations in
`generate_motions.py` — safer than hand-editing CSVs:

```bash
python tools/sysid/generate_motions.py --output_dir tools/sysid/motions --control_freq 30
```
