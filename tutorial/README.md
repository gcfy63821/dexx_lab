# Dex-X tutorial

A dexterous manipulation pipeline that goes from a human hand demonstration to a
policy running on a real Franka + Sharpa hand, taught in the order you would
actually rebuild it.

Each lesson is a directory with a `README.md` you can read on its own and, where
it is possible without hardware, a script you can run. Lessons state what they
assume and end with a check you can perform, so you can tell whether you have it
working rather than whether you have read it.

## The ten lessons

| # | Lesson | Runs without hardware? |
|---|---|---|
| [00](00_setup/) | Setup, repo map, and a smoke test | yes |
| [01](01_frames_and_constants/) | Frames and constants — the sim2real contract | yes |
| [02](02_assets_and_joints/) | Assets and joint ordering | yes |
| [03](03_retarget/) | Retargeting a human demo onto the robot | yes |
| [04](04_teacher/) | The state expert (asymmetric PPO) | yes |
| [05](05_distillation/) | Distilling a point-cloud student (DAgger) | yes |
| [06](06_camera_calibration/) | Camera extrinsic calibration | needs a camera |
| [07](07_dynamics_alignment/) | Dynamics: PD/impedance gain alignment | sim part yes |
| [08](08_evaluation/) | Evaluation that means something | yes |
| [09](09_deploy/) | Deploying on the real robot | needs the robot |

Lessons 00–05 and 08 run end to end on one GPU with the four demonstrations this
repository ships. 06, 07 and 09 are the sim2real half; their sim-side content is
runnable, their hardware side is documented.

## If you are porting this to your own setup

Start from the row that matches what you are changing. Each lesson's **Porting**
section lists exactly which values move.

| What you are changing | Read, in this order |
|---|---|
| Nothing — you want to understand the pipeline | 00 → 01 → 03 → 04 → 05 → 08 |
| Table height, camera position, mount height | 01 → 06 → 08 |
| A different arm or a different hand | 02 → 01 → 07 → 03 |
| New task, new objects, new demonstrations | 03 → 04 → 05 |
| Getting it onto hardware | 01 → 06 → 07 → 09 |

## The three ideas worth taking away

Whatever you port, these are the parts that matter most.

**A frame convention is a contract, not a comment.** The table height, the arm
base height and the workspace crop appear in the retargeter, the simulator, the
point-cloud crop and the deploy client. When one copy drifts, nothing raises an
error — the object simply lands somewhere else. Lesson 01.

**A calibration is an input, not a constant.** The camera extrinsic describes
how the camera is mounted. If any stage reads a hard-coded or default value, a
moved camera silently changes the viewpoint the policy trains on, and nothing
warns you. Pass the calibration file explicitly at every stage. Lesson 06.

**Sim and real must describe the same plant.** The policy learns against whatever
stiffness, damping, latency and filtering the simulator gives it. If the real
arm's impedance controller differs, the policy is meeting a machine it never
trained on. Lesson 07.

## Where the code lives

An index by file and symbol name is in **[CODE_MAP.md](CODE_MAP.md)**.

The lessons teach the pipeline; the implementation stays where it is.

```
src/dexx/
  deploy_config.py            all sim2real geometry and ports  (lesson 01)
  tasks/franka_sharpa/        envs, observation assembly, point cloud (04, 05)
  algo/                       PPO, DAgger, the PointNet student (04, 05)
scripts/                      the entry points every lesson calls
deploy/                       the real-robot runtime (lesson 09)
calib/camera_align/           shipped camera extrinsics (lesson 06)
docs/                         reference pages the lessons link into
```

The lessons explain; the reference pages hold the canonical commands and flags —
[RETARGET](../docs/RETARGET.md), [TRAINING](../docs/TRAINING.md),
[DISTILLATION](../docs/DISTILLATION.md), [EVAL](../docs/EVAL.md),
[DEPLOY](../docs/DEPLOY.md), [JOINT_ORDERING](../docs/JOINT_ORDERING.md),
[tool index](../docs/DEBUG_TOOLS.md). What the shipped checkpoints were trained
with is in [checkpoints/README.md](../checkpoints/README.md).

## Verifying your changes

```bash
bash tutorial/run_acceptance.sh
```

Minutes, end to end (runtime depends on the GPU and first-run caches): static
checks, a runtime preflight, a three-iteration distillation, an evaluation with
physics randomization on, and assertions on what came out. Run it before and after
touching `src/`.
