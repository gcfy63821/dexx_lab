# DAgger point-cloud distillation

Distills the privileged poseobs PPO teacher into a **point-cloud student** that
observes raw point clouds instead of ground-truth object pose. An optional PPO
fine-tune exists but is **not** part of the pipeline (§Step B).

```
poseobs PPO teacher (actor obs 557-d)
    └─ DAgger → PointCloudStudent (lean proprio 417-d + PointNet → 29 actions)
         └─ deploy (deploy/deploy_pc.py)
```

The env (`franka-sharpa-pointcloud`) inherits the poseobs env, so it still
publishes the full **557-d** actor observation, which the teacher labels from.
It additionally exposes raw point clouds in the obs dict — `scene_pc`,
`scene_mask`, `hand_pc`, `tactile_pc`, `tactile_force`. The **PointNet encoder
lives in the policy network, not the env**. The student's own proprio is a
**slice** of the 557: `--student_drop_slots obj_bps,tips_distance,obj_pose_tail`
leaves 417 dims (the "lean" student), the only kind `deploy_pc.py` accepts.

## Step A — DAgger

This is the canonical command. It is the recipe of the shipped
`checkpoints/student_lean_v6_L1.pth`:

```bash
python scripts/train_dagger_pc.py --task franka-sharpa-pointcloud \
    --teacher_ckpt checkpoints/teacher_poseobs.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]' \
    --num_envs 128 --dagger_iters 30 --rollout_steps 4096 \
    --beta_init 1.0 --beta_decay 0.85 \
    --train_epochs 8 --batch_size 512 --lr 5e-5 --max_buffer 200000 \
    --hand_body_subset minimal6 \
    --student_drop_slots obj_bps,tips_distance,obj_pose_tail \
    --camera_extrinsic calib/camera_align/current.npy \
    --seed 42 --out_dir logs/dagger_pc --headless
```

The shipped student's training conditions (arm mount, extrinsic, crop box) are
listed in [checkpoints/README.md](../checkpoints/README.md); a retrain here
reproduces the recipe, not the checkpoint. Several
defaults differ from this command (`--num_envs 64`, `--dagger_iters 20`,
`--beta_decay 0.8`, `--train_epochs 5`, `--seed 0`, no subset, no dropped slots),
so pass every flag. `--student_drop_slots` is what makes a student deployable:
the robot has no object pose, BPS shape code or mesh-based fingertip distances.

- **Output:** `dagger_final.pth` under `--out_dir`. Each iteration logs
  `roll_succ (n=…)`: the rollout policy's success rate over the `n` episodes
  that *ended* inside that rollout. Each rollout restarts from reset and lasts
  only `--rollout_steps / --num_envs` steps per env (32 with the command above),
  far shorter than an episode, so the count skews toward early terminations
  and the rate is **biased low — a trend signal, not a success rate**. Even at
  iteration 1 (β = 1, the rollout policy is the teacher) it reads far below the
  teacher's `eval_teacher.py` success rate. Judge the teacher with
  `eval_teacher.py` and the student with `eval.py`.
- The checkpoint records its own layout (`proprio_dim`, `student_drop_slots`,
  `student_keep_idx`, `student_obs_slots`) and the env-side point-cloud settings
  (`pc_env_meta`: point counts, crop box, tactile representation). Eval and deploy
  restore them and refuse a live slot map that disagrees.

Key DAgger flags:

| Flag | Default | Purpose |
|---|---|---|
| `--teacher_ckpt` | required | Poseobs teacher checkpoint. |
| `--dagger_iters` | `20` | DAgger outer iterations. |
| `--rollout_steps` | `4096` | Steps collected per iteration. |
| `--beta_init` / `--beta_decay` | `1.0` / `0.8` | Teacher-mixing schedule (convex action blend). |
| `--train_epochs` / `--batch_size` / `--lr` | `5` / `512` / `5e-5` | BC update. |
| `--max_buffer` | `200000` | Aggregated replay buffer size. |
| `--hand_body_subset` | `None` (cfg: `default11`) | `minimal5` / `minimal6` / `default11` / `dense22`. |
| `--student_drop_slots` | `None` | Actor-obs slots to slice out of the student's proprio, by name. |
| `--student_mask_slots` | `None` | Zero (not remove) proprio ranges for the student only. |
| `--camera_extrinsic` | unset → `calib/camera_align/current.npy` | Sim camera pose; deploy with the same file. |
| `--ref_root` | `None` | Read retargeted demos from another root. |
| `--student_ckpt` | `None` | Resume from a pretrained student. |

## Step B — PPO fine-tune (optional; not for lean students)

`scripts/train_ppo_pc.py` fine-tunes a DAgger student with asymmetric PPO (the
critic sees privileged state). **It refuses lean students** (checkpoints with a
`student_keep_idx`), so it cannot be applied to the shipped student or to the
Step A command's output — only to a full-observation student, which cannot be
deployed. It is kept for experiments.

```bash
python scripts/train_ppo_pc.py --task franka-sharpa-pointcloud \
    --dagger_ckpt <full-obs dagger_final.pth> \
    --side right --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy \
    --num_envs 64 --max_iters 100 \
    --mini_epochs 1 --minibatch_size 1024 \
    --lr 1e-5 --clip 0.05 --init_logstd -4.0 \
    --save_every_n 20 --out_dir logs/ppo_pc --headless
```

- **Output:** `ppo_final.pth` under `--out_dir`.
- `--init_logstd` defaults to `-2.0`; use `-4` (σ ≈ 0.018). With `logstd = 0`
  (σ = 1) over 29 action dims the sampled-action noise (norm ≈ 5.4) overwhelms the
  DAgger `mu` and destroys the policy before any learning.
- NaN guards, all needed together: `ratio.clamp(max=10)`, `logstd.clamp(-5, 2)`,
  Huber critic loss instead of MSE, no bf16 autocast in the update, and
  conservative hyperparameters (`critic_coef=0.5`, `lr=1e-5`, `clip=0.05`).
  Without them the PPO update diverges to NaN.

## Critical invariants

These are load-bearing — break one and training collapses or diverges:

- **Env inherits poseobs (557-d), not the force env.** The teacher's actor obs is
  the 557-d layout; the student's proprio is a slice of it.
- **Workspace crop** — `cfg.pc_workspace_min/max` (env-local, from
  `deploy_config.py`) drops table and curtain hits before subsampling, so
  `scene_pc` concentrates on the manipulation region.
- **Hand-body subset `minimal6`** (wrist + 5 fingertips), as in the shipped
  student. The MCP knuckles barely contact objects, so including them adds
  PointNet noise. The cfg default is `default11`; pass `--hand_body_subset minimal6`.
- **The DAgger buffer is fp32**, not fp16: `scene_pc` coordinates above 1 m exceed
  fp16 mantissa resolution.
- **`init_curriculum` is OFF** for the point-cloud env (the default in
  `franka_sharpa_pointcloud_env_cfg.py`). With it on, resets sample only from the
  last half of each demo, biasing training and eval toward the easy demo end.

### Hand-subset ↔ checkpoint alignment

A DAgger checkpoint saves `n_hand` (e.g. 6) but not the body names. `eval.py`,
`play.py` and `deploy_pc.py` pre-read `n_hand/n_scene/n_tactile` and pick the
matching subset **before** building the env. Without this, subset checkpoints
crash with `Sizes of tensors must match` in the PointNet forward (env produces 11
hand points, encoder weights sized for 6).

## Evaluating the student

Evaluate a trained student with `scripts/eval.py` ([EVAL.md](EVAL.md)). The
numbers the shipped student reproduces under the reference protocol, with the
exact command, are in
[checkpoints/README.md](../checkpoints/README.md#reference-numbers-in-this-releases-simulation).
