# Third-party notices

The project's own code is released under the MIT License ([LICENSE](LICENSE)).
The files below contain code vendored or derived from other projects and remain
under their original licenses; their copyright headers are kept in the files and
must not be removed. Packages installed as dependencies (Isaac Lab, PyTorch,
pytorch3d, pytorch_kinematics, …) are not vendored and are not listed here.

## Code

| upstream | license | files |
|---|---|---|
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — Copyright (c) 2022-2025, The Isaac Lab Project Developers | BSD-3-Clause | `scripts/train_teacher.py`, `scripts/collect_reference_rollouts.py`, `src/dexx/tasks/franka_sharpa/agents/__init__.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_env.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_env.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_critic_horizon_env.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_poseobs_env.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_deploy_env_v2.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_critic_horizon_deploy_env.py`, `src/dexx/tasks/franka_sharpa/franka_sharpa_force_critic_horizon_deploy_env_v3.py`, `src/dexx/tasks/franka_sharpa/visual_raycaster.py`, `src/dexx/tasks/hand_imitation/deploy/ros2_action_publisher.py`, `src/dexx/tasks/hand_imitation/deploy/ros2_observation_subscriber.py`, `src/dexx/wrapper/sharpa_wave_env_wrapper.py` |
| [rsl_rl](https://github.com/leggedrobotics/rsl_rl) — Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION | BSD-3-Clause | `src/dexx/wrapper/vec_env.py`; `AdaptiveScheduler` in `src/dexx/algo/ppo/ppo.py` |
| [Hora](https://github.com/HaozhiQi/hora) (In-Hand Object Rotation via Rapid Motor Adaptation) — Copyright (c) 2022 Haozhi Qi | MIT | `src/dexx/algo/models/models.py`, `src/dexx/algo/models/running_mean_std.py`, `src/dexx/algo/ppo/ppo.py`, `src/dexx/algo/ppo/experience.py`, `src/dexx/utils/misc.py` |
| [rl_games](https://github.com/Denys88/rl_games) — Copyright (c) 2019 Denys88 (via Hora) | MIT | `src/dexx/algo/ppo/ppo.py`, `src/dexx/algo/ppo/experience.py` |
| [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) — Copyright (c) 2018-2022, NVIDIA Corporation (via Hora) | BSD-3-Clause | `src/dexx/algo/models/running_mean_std.py`, `src/dexx/utils/misc.py` |
| [SAGE](https://github.com/NVIDIA-Isaac-Sim/sage) — motion files and analysis, ported and adapted | to be confirmed by the maintainers | `tools/sysid/generate_motions.py`, `tools/sysid/analyze_motion.py` (and the motions it generates in `tools/sysid/motions/`) |

## Assets

`assets/franka_fr3/` and `assets/sharpa_wave/` are third-party models under the
Apache License 2.0; their `LICENSE` / `LICENSE.txt` / `NOTICE.txt` are kept in
those directories. [assets/ASSETS.md](assets/ASSETS.md) lists the provenance and
license of every asset.

## Data

The demonstrations in `data/robotool_batch/` (and the retargeted files derived
from them) are derived from MANO hand-model fits. MANO-derived data may be
subject to the [MANO license](https://mano.is.tue.mpg.de/license.html) terms,
which are separate from this repository's MIT License.

- **TODO (maintainers):** confirm the license of `data/robotool_batch/` (MANO-derived
  demonstrations and the object models in `data/robotool_batch/models/`) before
  publication, and state it here.
