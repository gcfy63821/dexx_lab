# Evaluation

Four entry points:

- `scripts/eval.py` — batched quantitative evaluation of a point-cloud student,
  with strict success post-processing. **This is the protocol; quote its numbers.**
- `scripts/play.py` — interactive / visual rollout of a student checkpoint.
- `scripts/eval_teacher.py` — evaluation of a poseobs teacher checkpoint.
- `scripts/record_videos.py` — render teacher-rollout videos / snapshots.

## eval.py

```bash
python scripts/eval.py --task franka-sharpa-pointcloud \
    --load_path checkpoints/student_lean_v6_L1.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy \
    --num_envs 128 --out_dir logs/eval_cube --headless
```

The checkpoint decides the env-side point-cloud settings — point counts, crop
box, tactile representation, ablations (`pc_env_meta`) — and the lean student's
proprio slicing. Flags override them only when passed.

### What the default run does

The defaults are the reference protocol:

- **Every retarget augmentation variant** of each demo (`{demo}@{aug}`), reported
  per variant. `--no-expand_aug` evaluates only the base demos, under their base
  IDs (`DISABLE_AUG_EXPAND=1`).
- **First-to-finish collection** (`--per_demo_quota 0`) until `--max_episodes`
  (8000) or `--max_steps` (30000). It weights the aggregate toward demos whose
  episodes end sooner, so read the per-demo rates too; `--per_demo_quota N` takes
  N per demo, `-1` a balanced `ceil(max_episodes / n_demos)`
  ([tutorial/08](../tutorial/08_evaluation/)).
- **Physics DR on** (`--keep_physics_dr`): object mass, friction, centre of mass
  and **hand** PD gains stay randomized; `--no-keep_physics_dr` holds them at
  nominal. **Arm PD-gain randomisation and action-delay randomisation are always
  on** — `eval.py` does not disable them — so repeated runs are not bit-identical.
- **Deterministic `mu`**, `init_curriculum` off, resets uniform over the full demo.
- **Noise off:** point-cloud jitter / dropout / hand noise, tactile force noise,
  depth noise and object-pose noise are zeroed.

| Flag | Default | Purpose |
|---|---|---|
| `--load_path` | required | DAgger or PPO ckpt (arch auto-detected). |
| `--out_dir` | required | Output dir for `records.json`. |
| `--data_idx` | `None` | Demo indices to evaluate. |
| `--expand_aug` / `--no-expand_aug` | on | Evaluate every retarget augmentation variant (`{demo}@{aug}`); off = base demos only, under their base IDs (e.g. `rt/0416_grasp/cube_small_1`). |
| `--per_demo_quota` | `0` | `0` = first-to-finish; `N` = N episodes per demo; `-1` = `ceil(max_episodes / n_demos)` per demo. |
| `--keep_physics_dr` / `--no-keep_physics_dr` | on | Mass / friction / COM / hand-PD randomisation during eval. |
| `--camera_extrinsic` | unset → `calib/camera_align/current.npy` | Must match what the student was trained with. |
| `--num_envs` | `128` | Parallel envs. |
| `--max_episodes` | `8000` | Stop after this many episode terminations. |
| `--max_steps` | `30000` | Hard step ceiling (safety). |
| `--success_dist` | `0.05` | Closest-approach success threshold (`min_final_dist < this`). Post-hoc strict thresholds use `end_final_dist`, not this. |
| `--save_traj` | off | Save per-frame tracking arrays in `records.json`. |
| `--perturb_obj_xy` | `0.0` | Eval-time random XY perturbation of object init pos (m). |
| `--inject_jitter` / `--inject_dropout` / `--inject_hand_noise` | `None` | Override PC noise at eval time (sim2real preview); by default eval zeros all PC noise for a clean run. |

### strict3 protocol

`eval.py` computes strict2 / strict3 / strict5 itself and writes them to
`summary.json` under `strict`, plus a per-demo `strict3`. The separate
`success_rate_*` fields and `EpisodeRecord.succeeded` measure closest approach:
the object came within `--success_dist` of its target at some point during the
episode. They do not use the env's trajectory-completion success flag.
Endpoint distance and rotation come from the terminal step's reward diagnostics,
captured before the environment resets.

The headline metric, **strict3**, counts an episode as a success only if:

- `end_final_dist < 3 cm` at episode end (object reached the demo's final pose),
  **and**
- no object-position drift (stored as `obj_pos_drift` in `fail_causes`), and
- the episode was not a bad-init (`survival_len ≤ 5` are excluded).

This protocol does not require the env's success flag and deliberately does not
reject other failure causes. It is therefore not a subset of env-internal
reach-end success. Training's `Strict` is a separate conservative proxy: it
requires trajectory completion without failures and keeps bad inits in its
denominator as zeros (see [TRAINING.md](TRAINING.md)).

The gap between endpoint accuracy and closest-approach success is not constant:
it differs from demo to demo, and a demo can come within 5 cm almost every time
yet often finish outside 3 cm or with the object drifting. Reaching within 5 cm
at some point does not guarantee finishing within 3 cm without drift. Report
strict3, and report it per demo. The expected numbers for the shipped
checkpoints, with the exact commands, are in
[checkpoints/README.md](../checkpoints/README.md#reference-numbers-in-this-releases-simulation).

### succeeded-vs-terminated caveat

Do not read a per-step success rate as the episode success rate. `env.success_buf`
is **cleared in `_reset_idx`** (which runs inside `step()`), so reading it right
after `step()` returns 0 for envs that just terminated — the per-step
`success_rate` looks tiny. To measure env trajectory completion, select terminated
environments from `extras["succeeded_per_env"]`, captured before reset, and average
those episode flags (as PPO does). `eval.py` instead computes the closest-approach
and strict endpoint metrics from episode records; its `succeeded` field is not
the env flag.

## play.py

Interactive rollout (visual, fewer envs):

```bash
python scripts/play.py --task franka-sharpa-pointcloud \
    --load_path <ckpt.pth> --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 16
```

Key flags: `--num_envs` (default 16), `--max_episodes` (default 200),
`--max_steps`, `--label`. (Drop `--headless` to watch the GUI.)

⚠️ **`play.py`'s success rate is a different metric from `eval.py`'s.** Here a
success means the episode reached the end of the trajectory without a failure
termination; `eval.py`'s `success_rate_*` means the object came within
`--success_dist` (5 cm) of its target at some point, and its headline is strict3.
A short `--max_steps` truncates episodes and drives `play.py`'s number down
without the policy being any worse, so the two can differ by an order of
magnitude on the same checkpoint. Quote `eval.py`.

## record_videos.py

Renders one video per demo of an expert (teacher) rollout via the
`franka-sharpa-pointcloud-record` env. The script turns on `--enable_cameras`
itself; rendering is GPU-heavy, so keep `--num_envs` small:

```bash
python scripts/record_videos.py --task franka-sharpa-pointcloud-record \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx_list '["rt/0416_grasp/cube_small_2"]' \
    --out_dir logs/expert_rollout_videos
```

Notable flags: `--data_idx_list` (JSON list, one video per entry),
`--max_steps` (default 400), `--fps` (default 15), `--cam_yaw_deg`,
`--cam_z_offset`, `--snapshot_frames` (save PNG snapshots), `--snapshot_only`,
`--draw_force` (overlay per-fingertip contact-force arrows), `--save_per_env`.

## eval_teacher.py

`eval.py` and `play.py` take **point-cloud student** checkpoints. Evaluate a
**poseobs teacher** with `scripts/eval_teacher.py`:

```bash
python scripts/eval_teacher.py --task franka-sharpa-force-poseobs \
    --load_path checkpoints/teacher_poseobs.pth \
    --side right --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --modes random --success_dist 0.05 \
    --num_envs 64 --episodes_per_mode 256 --out_dir logs/eval_teacher --headless
```

`--modes` chooses where episodes start: `early`, `pre_contact`, `random`,
`per_stage` (default: all four). It writes `eval_table.md`, `summary.json` and
`records.json`; each record carries `init_frame`, `survival_len`,
`min_final_dist` and `fail_causes`, which is what you want when diagnosing a
failure mode.

**Run it after any asset change** as a behavioural regression. Two assets can be
identical item by item — gains, limits, inertia — and still differ by tens of
points; see the self-collision section of [ASSETS.md](../assets/ASSETS.md).
Measure a baseline with the shipped teacher before the change (the expected value
is in [checkpoints/README.md](../checkpoints/README.md#reference-numbers-in-this-releases-simulation))
and compare against it afterwards.

## Reference-rollout tools (internal)

- `scripts/collect_reference_rollouts.py` — collect policy rollouts
- `scripts/build_rollout_references.py` — build reference trajectories from them

These are off the main pipeline; use them only to rebuild reference trajectories.
