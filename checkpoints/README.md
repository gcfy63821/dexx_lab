# Checkpoints

This is the one place that describes the shipped checkpoints; other docs link here.

| file | what |
|---|---|
| `teacher_poseobs.pth` | PPO state expert (`franka-sharpa-force-poseobs`, 557-d privileged obs, tactile on). The teacher of the shipped student. |
| `student_lean_v6_L1.pth` | DAgger point-cloud student, lean: 417-d proprio (drops `obj_bps`, `tips_distance`, `obj_pose_tail` from the 557-d layout), 1024 scene / 6 hand / 25 tactile points. The checkpoint to deploy. |
| `teacher_poseobs_T0.pth` | an earlier poseobs teacher, kept for reference |

## Settings the shipped checkpoints expect

| | shipped checkpoints | release default |
|---|---|---|
| camera extrinsic | `calib/camera_align/current.npy` | the same file (fallback when `--camera_extrinsic` is omitted) |
| crop box z | 0.417 … 0.70, recorded in the checkpoint's `pc_env_meta` | `PC_WORKSPACE_MIN/MAX` in `src/dexx/deploy_config.py`, same values |
| arm base z | 0.432 | `ARM_BASE_Z = 0.415` (robot mounted level with the table) |

So retraining here reproduces the recipe, not the checkpoint. On the robot, deploy
with the crop floor raised to 0.422 (`--pc_workspace_min 0.0,-0.40,0.422`): the
real table sits a few mm higher in the measured cloud than in sim
([docs/DEPLOY.md](../docs/DEPLOY.md)).

## Reference numbers in this release's simulation

> The additional data augmentation used in our experiments is not included in
> this release. These results are provided to help confirm that the tutorial
> version of the code runs correctly.

Use these to check an install: a correct setup lands within a few points of
them, while a wrong extrinsic, crop box or asset lands far off. Both runs use the
four shipped demos and `calib/camera_align/current.npy`.

```bash
DEMOS='["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]'
python scripts/eval.py --load_path checkpoints/student_lean_v6_L1.pth --side right \
    --data_idx "$DEMOS" --camera_extrinsic calib/camera_align/current.npy \
    --out_dir logs/eval_ref_protocol --headless
python scripts/eval_teacher.py --task franka-sharpa-force-poseobs \
    --load_path checkpoints/teacher_poseobs.pth --side right --data_idx "$DEMOS" \
    --modes random --num_envs 64 --episodes_per_mode 256 \
    --out_dir logs/eval_ref_teacher --headless
```

Student, `eval.py` with its defaults (the reference protocol: every augmentation
variant — only `@0` ships — first-to-finish collection, physics DR on, 8000
episodes; strict = endpoint within N cm, no object drift, bad inits excluded; see
[docs/EVAL.md](../docs/EVAL.md)):

| demo | episodes | closest (5 cm) | strict3 |
|---|---|---|---|
| cube_small_1@0 | 1696 | 80.2% | 66.5% |
| cube_small_2@0 | 3607 | 68.0% | 62.2% |
| squeegee_1@0 | 1317 | 78.0% | 36.7% |
| squeegee_2@0 | 1380 | 91.5% | 29.5% |
| **all** | **8000** | **76.3%** (macro 79.4%) | **53.3%** (strict2 37.3%, strict5 64.4%) |

First-to-finish gives `cube_small_2` 45% of the episodes (its episodes end
soonest), so the aggregate leans on it; the per-demo rates are the comparable ones.

Teacher, `eval_teacher.py`, random initial states, 256 episodes: **94.9%**
(cube_small 100%, squeegee 87.4%).

The release sim differs from the one the shipped checkpoints were trained in (arm
base height, table above), so these are baselines for this release, not
training-time numbers. Physics DR, arm-gain and action-delay randomisation stay
on, so repeated runs are not bit-identical; a per-demo rate from about 100
episodes has a 95% interval of roughly ±10 points, and a 40-episode run is a smoke
test, not a measurement.

- Training command: [docs/DISTILLATION.md](../docs/DISTILLATION.md)
- Teacher: [docs/TRAINING.md](../docs/TRAINING.md)
- Robot run: [docs/DEPLOY.md](../docs/DEPLOY.md)
- Evaluation protocol: [docs/EVAL.md](../docs/EVAL.md)
