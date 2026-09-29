# Code map

Where things are, and which lesson explains each one. Entries name files and
symbols, not line numbers — find them with `grep -n "def <name>" <file>`.

## The shape of the repository

```
src/dexx/
  deploy_config.py                  every sim2real constant             lesson 01
  robot_constants.py                hand + arm gains, armature, friction lessons 02, 07
  tasks/franka_sharpa/              the environments
  tasks/hand_imitation/             demo loading and transforms         lesson 03
  algo/{ppo,dagger,models}/         training algorithms                 lessons 04, 05
scripts/                            entry points — one per stage
deploy/                             the real-robot runtime              lesson 09
tools/{calib,sysid}/                calibration and system-ID           lessons 06, 07
tutorial/                           these lessons
```

## Environments: one inheritance chain

Each layer adds one thing. Read them in this order; each is small except the base.
All files are in `src/dexx/tasks/franka_sharpa/`.

| env class | file | adds |
|---|---|---|
| `FrankaSharpaEnv` | `franka_sharpa_env.py` | scene, actions, reward, reset — everything |
| `FrankaSharpaForceEnv` | `franka_sharpa_force_env.py` | contact sensing |
| `FrankaSharpaForceCriticHorizonEnv` | `franka_sharpa_force_critic_horizon_env.py` | the K-frame privileged block, the named slot map |
| `FrankaSharpaForcePoseObsEnv` | `franka_sharpa_force_poseobs_env.py` | the 7-d noisy object pose, with its noise model — task `franka-sharpa-force-poseobs` |
| `FrankaSharpaPointCloudEnv` | `franka_sharpa_pointcloud_env.py` | scene / hand / tactile point clouds — task `franka-sharpa-pointcloud` |

Two leaves hang off the point-cloud env: `FrankaSharpaPointCloudRecordEnv`
(`franka-sharpa-pointcloud-record`, videos) and `FrankaSharpaPointCloudDeployEnv`
(`franka-sharpa-pointcloud-polymetis-deploy`, the robot; its parents are the
`*_deploy_env*.py` files).

The configs form the matching chain: `FrankaSharpaEnvCfg` →
`FrankaSharpaCriticHorizonCfg` → `FrankaSharpaPoseObsCfg` →
`FrankaSharpaPointCloudEnvCfg`. A field set on a parent is visible to every child,
which is why `deploy_config.py` only has to be read in one place.

## Navigating `franka_sharpa_env.py`

The largest file by far. Grouped by what you would be looking for:

| you want | symbol | lesson |
|---|---|---|
| how an action becomes a joint target | `_pre_physics_step` | 07 |
| the PD control itself | `_apply_action` | 07 |
| what the actor observes | `compute_observations` | 04 |
| the reward (weights in its `reward_execute` sum) | `compute_imitation_reward` | 04 |
| episode termination | `_get_dones` | — |
| reset, curriculum, object placement | `_reset_idx` | 01 |
| domain randomization draws | `_rand_pd_scales`, `set_friction`, `set_com`, `set_mass` | 07 |
| loading the demonstrations | `_build_data` | 03 |
| which joints are the arm's | `_identify_arm_joints` | 02 |
| the scene: table, curtain, camera | `_setup_scene` | 06 |

The free functions at the bottom of the file — quaternion and frame helpers,
`scale`/`unscale`, rotation utilities — hold no state.

`_pre_physics_step` and `compute_observations` are the two worth reading in full.
Between them they define the policy's entire interface to the world.

## Where a number you care about lives

| number | where |
|---|---|
| table height, arm base, workspace crop, camera intrinsics, ports | `deploy_config.py` |
| hand stiffness / damping / armature / friction | `robot_constants.py` `HAND_GAINS` |
| **arm** stiffness / damping | `robot_constants.py` `ARM_TUNED_KP` / `ARM_TUNED_KD`, applied in `FrankaSharpaCriticHorizonCfg.__post_init__` |
| actuator cfgs (gains, armature, effort limits) | `robot_cfg` in `franka_sharpa_env_cfg.py` |
| action delay, EMA coefficients, delta scale | `franka_sharpa_env_cfg.py` |
| domain randomization ranges | `franka_sharpa_env_cfg.py` (`randomize_*`) |
| deploy e-stop limits | `franka_sharpa_env_cfg.py` (`arm_joint_*_limit`, `hand_joint_delta_limit`, `deploy_max_sensor_age_s`) |
| object pose noise model | `franka_sharpa_force_poseobs_cfg.py` |
| point counts, depth noise, crop | `franka_sharpa_pointcloud_env_cfg.py` |
| reward weights | `compute_imitation_reward` in `franka_sharpa_env.py` |
| camera extrinsic | `calib/camera_align/*.npy` — **not** a constant, see lesson 06 |

## Algorithms

| file | what |
|---|---|
| `algo/models/models.py` | `ActorCriticAsymmetric` and friends. The critic takes `cat([obs, priv_info])` |
| `algo/ppo/ppo.py` | PPO for the state expert |
| `algo/ppo/ppo_pointcloud.py`, `actor_critic_pointcloud.py` | the optional PPO fine-tune (full-obs students only; lesson 05) |
| `algo/dagger/dagger_pointcloud.py` | the DAgger loop, the convex blend, the lean-student slicing, checkpoint contents |
| `algo/dagger/pc_env_meta.py` | env-side point-cloud settings recorded in and restored from a checkpoint |
| `algo/dagger/pointcloud_student.py` | proprio ⊕ PointNet feature → action |
| `tasks/franka_sharpa/pointcloud/pointcloud_encoder.py` | the shared PointNet |

## Verifying a change

```bash
bash tutorial/run_acceptance.sh
```

Static checks, a runtime preflight, a three-iteration distillation, an evaluation
with physics randomization on, then assertions on the artefacts — the student's
width, its dropped channels, its slot map, and that the evaluation episodes came
out balanced across demonstrations.

"The imports still resolve" is not the same as "it still works". This is the
second one.
