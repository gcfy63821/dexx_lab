# Dex-X

Dexterous manipulation reinforcement learning for a **Sharpa Wave** dexterous hand (22 DOF)
mounted on a **Franka FR3** arm (7 DOF), trained in **Isaac Lab** to imitate human hand
manipulation demos, with a sim-to-real deployment path.

A privileged **PPO teacher** is distilled into a deployable **point-cloud + tactile
student** with DAgger, and the student runs on the real robot.

> **This is the tutorial version of the Dex-X code.** It walks through the full
> pipeline, and we provide four sample demonstrations to run it on.

```
retarget              PPO teacher                 DAgger student                 deploy
MANO → Sharpa     →   franka-sharpa-          →   franka-sharpa-            →   franka-sharpa-pointcloud-
                      force-poseobs               pointcloud                     polymetis-deploy
                      (557-d privileged obs)      (417-d proprio + PointNet)
scripts/retarget.py   scripts/train_teacher.py    scripts/train_dagger_pc.py     deploy/deploy_pc.py
```

The environments form one inheritance chain: base → force → critic_horizon →
poseobs → pointcloud (configs `FrankaSharpaEnvCfg` → `FrankaSharpaCriticHorizonCfg`
→ `FrankaSharpaPoseObsCfg` → `FrankaSharpaPointCloudEnvCfg`). A fourth task,
`franka-sharpa-pointcloud-record`, renders videos.

## Shipped state

`checkpoints/` holds the PPO teacher and the deployable DAgger point-cloud
student. What each file is, the settings it expects (observation layout, crop
box, camera extrinsic, arm mount) and the evaluation numbers an
install should reproduce are recorded once, in
**[checkpoints/README.md](checkpoints/README.md)**.

## Install

Follow **[tutorial/00_setup](tutorial/00_setup/)**. The pinned baseline (Isaac Sim 4.5,
Isaac Lab v2.2.1, Python 3.10), both installer routes (conda + binary Sim, or uv +
pip Sim wheels) and the manual steps are in
**[MANUAL_SETUP.md](tutorial/00_setup/MANUAL_SETUP.md)**.

## Quickstart

Evaluate the shipped student in simulation (all commands run from the repository root):

```bash
python scripts/eval.py --task franka-sharpa-pointcloud \
    --load_path checkpoints/student_lean_v6_L1.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy \
    --max_episodes 400 --out_dir logs/eval_quickstart --headless
```

`--max_episodes 400` makes this a quick first check; without it `eval.py` runs
the full reference protocol (8000 episodes, [EVAL.md](docs/EVAL.md)).

Then: retarget a demo ([RETARGET.md](docs/RETARGET.md)), train a teacher
([TRAINING.md](docs/TRAINING.md)), distill a student
([DISTILLATION.md](docs/DISTILLATION.md)), read the metrics
([EVAL.md](docs/EVAL.md)), deploy ([DEPLOY.md](docs/DEPLOY.md)). The
[tutorial](tutorial/) walks the same path with the reasoning behind each step.

## Repository layout

| Path | What |
|---|---|
| `src/dexx/tasks/franka_sharpa/` | Isaac Lab environments (base → force → critic_horizon → poseobs → pointcloud) and the deploy envs |
| `src/dexx/tasks/hand_imitation/` | demo dataset loaders |
| `src/dexx/algo/` | PPO, DAgger point-cloud distillation, shared networks |
| `src/dexx/deploy_config.py` | sim2real geometry, intrinsics, crop box, ports ([tutorial/01](tutorial/01_frames_and_constants/)) |
| `src/dexx/robot_constants.py` | hand and arm gains, armature, friction |
| `scripts/` | retarget / train / eval / play / record entry points, asset builders |
| `deploy/` | real-robot runtime (Polymetis bridge, depth publisher, deploy entry point) |
| `assets/` | vendored FR3 + Sharpa Wave models, generated merged URDF, committed robot USDs |
| `data/` | four sample demos + their retargeted trajectories |
| `checkpoints/` | shipped teacher and student |
| `calib/camera_align/` | camera extrinsics — pass one with `--camera_extrinsic` |
| `tools/dataset/` | demonstration preview, rotation, drop test and manual fixes ([README](tools/dataset/README.md)) |
| `tools/calib/` | camera-calibration chain ([tutorial/06](tutorial/06_camera_calibration/)) |
| `tools/sysid/` | sim-vs-real motion replay for gain alignment ([tutorial/07](tutorial/07_dynamics_alignment/)) |
| `tutorial/` | the ten lessons |

## Documentation

**Start:** [tutorial/](tutorial/) — ten lessons in rebuild order, each with a check you can run.

**Reference:**
- [docs/RETARGET.md](docs/RETARGET.md) — MANO → Sharpa retargeting
- [docs/TRAINING.md](docs/TRAINING.md) — PPO teacher
- [docs/DISTILLATION.md](docs/DISTILLATION.md) — DAgger point-cloud student (the canonical command)
- [docs/EVAL.md](docs/EVAL.md) — evaluation protocol and metrics
- [docs/DEPLOY.md](docs/DEPLOY.md) — real-robot launch and safety
- [docs/JOINT_ORDERING.md](docs/JOINT_ORDERING.md) — hand joint orders (read before any sim2real work)
- [docs/DEBUG_TOOLS.md](docs/DEBUG_TOOLS.md) — index of the calibration, sysid and deploy tools
- [tutorial/CODE_MAP.md](tutorial/CODE_MAP.md) — where things live in the code

**Claude Code skills** ([.claude/skills/](.claude/skills/)) — step-by-step playbooks an
agent (or you) can follow: `dexx-codebase-guide` (setup → retarget → teacher →
student → eval), `dexx-real-robot-deploy`, `dexx-sim2real-testing`,
`dexx-camera-calibration` (easy_handeye initial guess → ICP → table levelling),
`dexx-new-hand-and-data` (new data / new hand / new env, with a check after every
step and viser checks of the initial hand and object pose).
Claude Code picks them up automatically when run from the repository root.

**Hardware & assets:**
- [assets/ASSETS.md](assets/ASSETS.md) — asset provenance, licensing, the hand swap
- [calib/camera_align/README.md](calib/camera_align/README.md) — the shipped extrinsics
- [checkpoints/README.md](checkpoints/README.md) — the shipped checkpoints, the settings they expect, install-check numbers
- [tools/dataset/README.md](tools/dataset/README.md) — demonstration data format and data tools
- [tools/sysid/motions/README.md](tools/sysid/motions/README.md) — sysid motion file format

## Citation

If you use this code, please cite:

```bibtex
@article{chen2026dex,
  title={Dex-x: Learning visual-tactile dexterous manipulation from human videos with simulated interaction},
  author={Chen, Ruoqu and Ruan, Feixiang and Cao, Liu and Wang, Zihao and Xu, Botian and Tong, Shiqin and Liu, Jiajun and Pei, Mingzhi and Zhang, Chenyu and Xing, Wanli and others},
  journal={arXiv preprint arXiv:2609.07747},
  year={2026}
}
```

## License

The project's own code is released under the **MIT License** (see [LICENSE](LICENSE));
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for vendored code and data.

`assets/franka_fr3/` (Franka FR3 description) and `assets/sharpa_wave/` (from
[sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml))
are third-party content redistributed under the **Apache License 2.0**; their
`LICENSE` / `LICENSE.txt` / `NOTICE.txt` are retained in those directories and must not be removed.
See [assets/ASSETS.md](assets/ASSETS.md) for the full provenance of every asset.
