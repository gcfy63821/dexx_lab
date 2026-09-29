# Demonstration data tools

Inspect and fix the human demonstrations under `data/robotool_batch/` before
they are retargeted (lesson 03). Run everything from the repository root; all
tools default to `data/robotool_batch`.

| tool | what it does | writes |
|---|---|---|
| `vis_sequence.py` | preview a sequence in the browser: hand keypoints + bones and the object mesh, frame slider | nothing |
| `rotate_task_z.py` | rotate every sequence of a task (or one sequence) about a world axis and pivot — objects, hands, aux | the pkls in place, originals as `*.rotbak`; `--restore` undoes |
| `view_retarget.py` | after retargeting: robot (FK of the retargeted joints) + object + table + MANO targets in env-local, exactly as the env loads them; per-frame checks (object bottom vs table, EE and fingertip errors); `--summary_only` prints the frame-0 checks | nothing |
| `drop_test.py` | drop the object onto the table in Isaac Sim and record where its origin settles (`z_bottom_offset`) | `mano_joints_corrected.pkl` |
| `adjust_hand_offset.py` | slide a constant hand↔object offset per hand side | `mano_joints.pkl` (+ `*.bak`) |
| `adjust_hand_and_aux.py` | the same, plus place a static auxiliary object from `models/` | `mano_joints.pkl` (+ `*.bak`), `aux_object.json` |
| `adjust_object_offset.py` | nudge the whole object trajectory (optionally dragging hand and aux along) | `mano_joints_corrected.pkl` (+ `*.bak`), `aux_object.json` |
| `adjust_endpoint_correction.py` | pin the object's first and last frame and blend the correction across the sequence | `mano_joints_corrected.pkl` (+ `*.bak`) |
| `edit_frame_range_pose.py` | correct the object pose over a frame range, ramped into its neighbours | `mano_joints_corrected.pkl` (+ `*.frameedit.bak`) |
| `offset_editor.py` | browse every sequence and translate hand and/or object over a frame range | a **copy** of the sequence under a new name; originals untouched |

The viser tools open a page at `http://localhost:8080` (`--port` to change);
`drop_test.py` needs the Isaac Lab environment and `--headless` without a
display. Backups (`*.bak`, `*.rotbak`) are git-ignored — never commit them.

```bash
python tools/dataset/vis_sequence.py --sequence 0416_grasp/cube_small_2

python tools/dataset/rotate_task_z.py --task_dir data/robotool_batch/0416_grasp --dry_run
python tools/dataset/rotate_task_z.py --task_dir data/robotool_batch/0416_grasp --angle_deg 90
python tools/dataset/rotate_task_z.py --task_dir data/robotool_batch/0416_grasp --restore

python -u tools/dataset/drop_test.py --task 0416_grasp --headless
```

## After editing a demonstration

The policy never reads these files directly: it reads the **retargeted**
trajectory in `data/retargeting/robotool_batch/mano2sharpa_rh/<task>/<seq>@0.pkl`.
Any edit here — a rotation, an offset, a new `z_bottom_offset` — takes effect
only after re-running `scripts/retarget.py` for that sequence (docs/RETARGET.md),
and a policy trained on the old trajectory is trained on the old demo.

Order that avoids redoing work: rotate → drop test → offsets/corrections →
preview → retarget. `drop_test.py` rewrites `mano_joints_corrected.pkl` from
the optimized (or raw) pkl, so corrections baked into the corrected file by the
adjust tools are lost if you drop-test afterwards.

## Data layout

```
data/robotool_batch/
  models/<object_id>/                 object mesh (cleaned_mesh_10000.obj), URDF, optional tool.usd
  <task>/<seq>/
    meta.json                         task, sequence, object_ids, mano_sides, num_frames, obj_mesh_paths
    mano_joints.pkl                   MANO joint positions + object poses
    mano_joints_optimized.pkl         optional: hand-object contact optimised
    mano_joints_corrected.pkl         + original_data["z_bottom_offset"] from drop_test.py
    aux_object.json                   optional static auxiliary object
```

A sequence is addressed as `rt/<task>/<seq>` (e.g. `rt/0416_grasp/cube_small_2`).
The loader takes the first that exists of `mano_joints_corrected.pkl`,
`mano_joints_optimized.pkl`, `mano_joints.pkl`. Mesh paths in `meta.json` are
relative to `data/robotool_batch/`.

pkl contents: `left` / `right` hand dicts (`wrist`, `wrist_translation`,
`wrist_orientation` xyzw, and per-finger `*_proximal`, `*_intermediate`,
`*_distal`, `*_tip` positions, each `(T, 3)`) and `original_data` with
`tool_object_pose` (`(T, 4, 4)` or `(T, 7)`) and, after the drop test,
`z_bottom_offset`.

## Frames

The pkls store the capture's **camera frame** (y-down, z-forward). The
visualisers show a z-up frame, and every tool takes its sliders and pivots in
that displayed frame and converts back on save (`diag(1, -1, -1)`). The loader
maps camera → sim with `Rx(180°) = diag(1, -1, -1)` plus the table offset, so
what the tools display is what the environment loads (table top at
`TABLE_SURFACE_Z` = 0.415 m, `src/dexx/deploy_config.py`).
