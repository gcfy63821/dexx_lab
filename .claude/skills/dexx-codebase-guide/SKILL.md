---
name: dexx-codebase-guide
description: Use when installing, running or modifying this repository (Dex-X, Franka FR3 + Sharpa Wave dexterous hand in Isaac Lab) — setting up the environment, retargeting a human demo, training the PPO teacher, distilling the point-cloud DAgger student, evaluating checkpoints, or finding where something lives in the code.
---

# Dex-X codebase guide

Pipeline: **retarget** (MANO human demo → robot joint trajectory) → **teacher**
(PPO, privileged state, task `franka-sharpa-force-poseobs`) → **student** (DAgger,
point cloud + tactile + proprioception, task `franka-sharpa-pointcloud`) →
**deploy** (task `franka-sharpa-pointcloud-polymetis-deploy`, see the
`dexx-real-robot-deploy` skill).

Run every command from the repository root. Isaac scripts need `--headless` on a
machine without a display.

## 1. Environment

Pinned baseline: Isaac Sim 4.5, Isaac Lab v2.2.1, Python 3.10, torch 2.7 (cu128).

```bash
# conda + binary Isaac Sim
bash tutorial/00_setup/setup_env.sh --isaacsim /path/to/isaac-sim --isaaclab /path/to/IsaacLab
# or uv (pip Isaac Sim wheels need glibc >= 2.34; add --isaacsim for a binary Sim on older glibc)
bash tutorial/00_setup/setup_uv.sh --isaaclab /path/to/IsaacLab --venv .venv --accept-eula
source .venv/bin/activate
```

Then verify, cheapest first — stop at the first failure:

```bash
python tutorial/00_setup/check_install.py      # files, assets, demos, calibrations, checkpoints (1 s)
python tutorial/00_setup/check_imports.py      # every import the code makes
python tutorial/01_frames_and_constants/check_frames.py
bash tutorial/run_acceptance.sh                # short train + eval end to end
```

Know-how:
- Behind a slow PyPI use `UV_DEFAULT_INDEX=<mirror>/simple`; `TORCH_INDEX_URL` for the torch wheels.
- PyTorch3D prebuilt wheels need glibc >= 2.32; otherwise `setup_uv.sh` builds it from source and bounds `MAX_JOBS` by the cgroup memory (the build is killed by OOM otherwise).
- With a binary Isaac Sim, its bundled torch must not shadow the venv's: `setup_uv.sh --isaacsim` handles this; if `import torch` reports 2.5.x inside Isaac, rerun it.
- `pytest` inside a shell that sourced ROS picks up ROS plugins: run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests`.

## 2. Retarget a demo

Demos live in `data/robotool_batch/<task>/<seq>/` (preview, rotate, drop-test and
fix them with `tools/dataset/`, see `tools/dataset/README.md`). Retarget into a
scratch root, check, then use it:

```bash
python -u scripts/retarget.py --side right --data_idx rt/0416_grasp/cube_small_2 \
    --dump_root logs/retarget --headless
```

- Read the printed reachability line: `[OK] reachable: mean arm EE err = … m`. An
  unreachable result never overwrites a reachable one (`[KEEP]`).
- Any edit to a demo (rotation, offsets, `z_bottom_offset`) only takes effect after
  re-retargeting it.
- Point training at a scratch root with `--env_cfg robotool_batch_retarget_root=<dir>`.

## 3. Teacher (PPO)

```bash
python scripts/train_teacher.py --task franka-sharpa-force-poseobs --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 2048 --headless   # --no-wandb to skip W&B
python scripts/eval_teacher.py --task franka-sharpa-force-poseobs \
    --load_path <best.pth> --side right --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --modes random --num_envs 64 --episodes_per_mode 256 --out_dir logs/eval_teacher --headless
```

`checkpoints/teacher_poseobs.pth` is a pretrained teacher; use it unless you are
changing the teacher's task, observation or demos. Watch `Strict` in the training
log, not the reward alone.

## 4. Student (DAgger, point cloud)

The canonical command (docs/DISTILLATION.md) — keep every flag:

```bash
python scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]' \
    --num_envs 128 --dagger_iters 30 --rollout_steps 4096 --beta_init 1.0 --beta_decay 0.85 \
    --train_epochs 8 --batch_size 512 --lr 5e-5 --max_buffer 200000 --hand_body_subset minimal6 \
    --student_drop_slots obj_bps,tips_distance,obj_pose_tail \
    --camera_extrinsic calib/camera_align/current.npy --seed 42 --out_dir logs/dagger_pc --headless
```

- `--student_drop_slots obj_bps,tips_distance,obj_pose_tail` is what makes the
  student deployable (417-d proprio): those inputs do not exist on the robot.
- **Always pass `--camera_extrinsic`**, and deploy with the same file.
- Judge training by the loss curve and by `eval.py`; `roll_succ` in the log is a
  trend signal biased low by the short rollout window, not a success rate.
- The checkpoint records its own observation layout and point-cloud settings
  (`pc_env_meta`); eval and deploy restore them.

## 5. Evaluate

```bash
python scripts/eval.py --load_path logs/dagger_pc/dagger_final.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy --out_dir logs/eval --headless
```

- Defaults = the reference protocol: every augmentation variant, first-to-finish,
  physics randomisation on, 8000 episodes. `--no-expand_aug`, `--per_demo_quota -1`
  (balanced) and `--no-keep_physics_dr` change it; say which you used when you
  quote a number.
- Report **strict3** (object ends within 3 cm of the demo's final pose, no drift,
  bad inits excluded), per demo. The closest-approach rate is always higher.
- Per-demo rates on ~100 episodes carry about ±10 points of 95% interval; a
  40-episode run is a smoke test.
- Visual check: `python scripts/play.py --task franka-sharpa-pointcloud --load_path <ckpt> --side right --data_idx '[...]' --num_envs 16`.

## Where things live

| need | look in |
|---|---|
| geometry, table height, arm base, crop box, intrinsics, ports | `src/dexx/deploy_config.py` |
| arm/hand gains, armature, friction | `src/dexx/robot_constants.py` |
| env chain (base → force → critic_horizon → poseobs → pointcloud) | `src/dexx/tasks/franka_sharpa/`, map in `tutorial/CODE_MAP.md` |
| PPO / DAgger / networks | `src/dexx/algo/` |
| every tool, one line each | `docs/DEBUG_TOOLS.md` |
| hand joint orders (cfg/Sharpa order vs sorted USD order) | `docs/JOINT_ORDERING.md` — read before touching hand I/O |

## Rules of thumb

- Change one thing, re-run the cheapest check that covers it (`check_frames.py`,
  a 3-iteration DAgger smoke run, a 40-episode eval) before a long run.
- A constant that exists in two places will drift: edit geometry only in
  `deploy_config.py`.
- Isaac scripts must close the env / simulation context before
  `simulation_app.close()`, or the process hangs after writing its output.
