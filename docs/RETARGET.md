# Retargeting (MANO → Sharpa)

`scripts/retarget.py` maps a MANO human-hand demo onto the Sharpa Wave hand and
the Franka FR3 arm using a 2-stage IsaacLab optimization (stage 1: arm-only reach;
stage 2: hand + arm tracking). Output is a `.pkl` trajectory the training envs
consume.

Joint targets in the output (`opt_dof_pos`) are stored in **cfg / Sharpa order**
— the canonical hand order (see [JOINT_ORDERING.md](JOINT_ORDERING.md)).

The input demonstrations and the tools to preview, rotate and correct them are
described in [tools/dataset/README.md](../tools/dataset/README.md).

## Command

```bash
python scripts/retarget.py --side right --data_idx rt/0416_grasp/cube_small_2 \
    --dump_root logs/retarget --headless
```

- **Input:** source demo directory (MANO joints + `meta.json`) under
  `data/robotool_batch/<task>/<obj>/`. Object mesh paths in `meta.json` are
  relative to `data/robotool_batch/`.
- **Output:** `<obj>@<n>.pkl` under `<dump_root>/<task>/`. Without `--dump_root`
  the root is `data/retargeting/robotool_batch/mano2<dexhand>/`, which is also
  where the training loaders read from — for the example demo,
  `data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl`,
  one of the shipped files. Use a scratch `--dump_root` until you mean to replace
  the shipped data (see below); point training or eval at it with
  `--env_cfg robotool_batch_retarget_root=<dir>` (teacher, DAgger) or
  `--ref_root <dir>` (DAgger, eval).
- **Placement offsets** (`--target_offset_xy`, `--z_offset`) differ per sequence
  and are part of the data; the shipped demos' values are in
  [tutorial/01](../tutorial/01_frames_and_constants/). Omitting them moves the object.

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--data_idx` | `None` | Single demo index, e.g. `rt/0416_grasp/cube_small_2`. |
| `--task` | `None` | Retarget all sequences of a task (`--data_idx rt/{task}/*`). |
| `--side` | `right` | Hand side (`left`/`right`); selects hand + USD. |
| `--dexhand` | `sharpa` | Dexhand type. |
| `--iter` | `4000` | Max optimization iterations (see below). |
| `--stage1_iter` | `None` | Stage-1 (arm-only) iterations; if unset uses side ratio. |
| `--z_offset` | `0.0` | Extra Z offset applied to the trajectory (m). |
| `--target_offset_xy` | `None` | Override the side default XY offset, e.g. `0.3 0.1`. |
| `--obj_scale` | `1.0` | Uniform object mesh scale, saved into the pkl for training. |
| `--reachability_th` | `0.08` | Reachability gate threshold (m); see below. |
| `--no_real_hand_clamp` | off | Disable clamping `opt_dof_pos` to the Sharpa reachable range. |
| `--num_envs` | `1` | Parallel envs (for batch / augmented retargeting). |
| `--dump_root` | `None` | Output root for the retargeted pkls. |
| `--skip_existing` | off | Incremental: skip variants whose pkl already exists. |

### Augmentation (object-pose perturbation)

Generate extra variants of a demo with the object translated / rotated:

| Flag | Default | Purpose |
|---|---|---|
| `--aug_num` | `0` | Number of augmented variants per demo (0 = original only). |
| `--aug_radius` | `0.05` | XY translation radius (m), uniform in `[-r, r]`. |
| `--aug_yaw_deg` | `10.0` | Yaw range (deg), uniform in `[-y, y]`. |
| `--aug_seed` | `42` | RNG seed (same seed → same variants). |

## Reachability gate

After stage 1 + 2, the script measures the mean arm end-effector error. If it
exceeds `--reachability_th` (default **0.08 m**), the variant is marked
`reachable=False` and only a **partial pkl** is written.

Training dataset loaders check this flag and **skip** unreachable demos
(`robotool_batch_dataset_dexhand.py`: `if opt_params.get("reachable", True) is
False`). So a demo that fails the gate will silently not appear in training —
lower `--reachability_th` for stricter tracking, or raise it to admit borderline
demos.

## Retarget writes into the shipped data by default

The default output root is the location of the shipped example demos, so a run
without `--dump_root` overwrites them.

| Flag | Effect |
|---|---|
| `--dump_root <dir>` | write under another root and leave the shipped data alone |
| `--skip_existing` | skip variants whose pkl already exists (incremental augmentation) |
| `--force_overwrite` | allow an unreachable partial result to replace an existing reachable pkl |

**Guard, on by default:** an unreachable partial result never replaces an existing
reachable pkl; the script prints `[KEEP]` and skips it. Without that guard, a
short run with a small `--iter` would replace a good demo with a stub.

## Iterations

Use the default, `--iter 4000`. Too few iterations (a few hundred) do not
converge: the mean arm EE error stays above the 0.08 m gate and the demo is
marked unreachable. A converged run ends well below the gate.

Runtime depends on GPU and demo length; budget minutes to tens of minutes per
demo and batch accordingly. The script exits when it is done, so a shell loop
over sequences works ([tutorial/03](../tutorial/03_retarget/)).
