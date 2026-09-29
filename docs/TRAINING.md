# PPO teacher training

`scripts/train_teacher.py` trains the **privileged PPO teacher** on the
`franka-sharpa-force-poseobs` task. The teacher is the source policy for DAgger
distillation ([DISTILLATION.md](DISTILLATION.md)).

## What "teacher" means here

- Task: **`franka-sharpa-force-poseobs`**, observation = **557d**. The observation
  is the critic-horizon force obs (550d) plus a **7d noisy object-pose tail**
  (position + quaternion). The tail acts as a privileged-but-noisy input that the
  actor is free to rely on during training.
- Because the pose is noisy, the DAgger student can later drop it and replace it
  with point clouds without losing much (see the distillation doc).

## Command

```bash
python scripts/train_teacher.py --task franka-sharpa-force-poseobs --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' --num_envs 2048 --headless
```

- **Input:** retargeted pkl(s) named by `--data_idx`.
- **Output:** teacher checkpoint (`.pth`). A pretrained one ships at
  `checkpoints/teacher_poseobs.pth` (training conditions in
  [checkpoints/README.md](../checkpoints/README.md)).

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--task` | `None` | Use `franka-sharpa-force-poseobs` for the teacher. |
| `--side` | `None` | Hand side (`left`/`right`). |
| `--data_idx` | `None` | JSON/Python list of demo indices, e.g. `'["rt/0416_grasp/cube_small_2"]'`. |
| `--num_envs` | `16384` | Parallel environments. Lower it (e.g. 2048) to fit memory. |
| `--seed` | `42` | Environment / training seed. |
| `--max_agent_steps` | `None` | Budget in **agent steps**, not iterations. One epoch is `num_envs x horizon_length` (e.g. 512 x 32 = 16384), so a small value stops before the first checkpoint. |
| `--load_path` | `None` | Checkpoint to load. |
| `--resume` | off | Resume training from `--load_path`. |
| `--env_cfg` | `[]` | Override env_cfg fields, e.g. `--env_cfg force_reward_weight=0.0`. |
| `--no_contact_force` | off | Ablation: disable the 5d scalar contact force in the obs. |
| `--video`, `--video_length`, `--video_interval` | | Record training videos. |
| `--wandb-project-name`, `--wandb-entity`, `--wandb-name` | `dex`, none, `dexmanip` | wandb logging. |
| `--no-wandb` | off | Train without wandb (no login needed). |

`--data_idx` accepts JSON (double quotes) or a Python-literal list.

## What to watch: `success_rate`, not just reward

The progress line and TensorBoard carry **two** episode success rates:

```
Mean Rewards: <reward> | Success: <survival>% | Strict: <strict>% | Current Best: <best reward>
```

| | meaning | TensorBoard |
|---|---|---|
| `Success` | reached the end of the trajectory without a failure termination — **survival** | `success_rate/iter` |
| `Strict` | survival **and** the object finished within 3 cm of its demo endpoint, with no object-position drift; episodes of at most 5 steps score zero — **training proxy** | `success_rate_strict/iter` |

Both are running means over the last 100 **completed** episodes.

`Strict` is a conservative training diagnostic, **not** the `strict3` evaluation
protocol. It requires trajectory completion without any failure termination and
counts bad inits as failures in the denominator. Evaluation excludes bad inits
from the denominator and rejects only object-position drift; it does not require
the env's success flag. For example, one successful episode plus one bad init
gives training `Strict = 50%`, but evaluation `strict3 = 100%`. Use `eval.py` for
the reported protocol metric; the training log keys retain their existing names.

The threshold lives in `STRICT_SUCCESS_DIST` (`franka_sharpa_env.py`) and is
passed into the reward function as a parameter rather than read from the module —
`compute_imitation_reward` is `@torch.jit.script`, and TorchScript cannot close
over a global float. Evaluation's strict3 always uses 3 cm; `--success_dist`
changes only the evaluation's closest-approach threshold.

Reward and success do not move together, and neither do the two success rates.
An epoch can gain a large amount of reward while **both** success rates fall.
Watching only the reward hides that completely.

### Why it is computed the way it is

`success_buf` is set to 1 on the step an episode ends and cleared in
`_reset_idx`. A mean over all environments at every step is therefore a
near-zero number that is **not** the episode success rate — the trap
`docs/EVAL.md` warns about. The correct quantity is taken only from the
environments that just terminated, on the step they terminate:

```python
# algo/ppo/ppo.py, alongside the existing episode_rewards update
done_indices = self.dones.nonzero(as_tuple=False)
self.episode_successes.update(infos['succeeded_per_env'][done_indices])
```

The env publishes `succeeded_per_env` / `failed_per_env` as per-environment
vectors for this. The older scalar keys `succeeded` / `failed_execute` are means
over all envs and are kept only for backward compatibility with existing log
names — do not read them as rates.

⚠️ **`_get_rewards` is overridden down the chain.** Both `FrankaSharpaEnv` and
`FrankaSharpaForceEnv` define it, and the force variant is what the teacher task
actually runs. Adding an extras key to the base class alone does nothing and
fails silently; PPO prints a one-shot warning if the key never arrives.

## Mixing demos with different metadata

`env._build_data` takes its key schema from the **first** demo in `--data_idx`.
For every key that demo has, a later demo lacking the key silently receives the
first demo's value; a key the first demo lacks is dropped for every demo. There is
no `KeyError` either way.

That matters for `obj_scale`, which `scripts/retarget.py --obj_scale` writes into
the pkl and the loader passes through. Older pkls do not carry it:

- legacy demo **first**: `obj_scale` disappears for all demos, and every object
  mesh is used at scale 1.0, including demos retargeted at another scale;
- legacy demo **later**: it inherits the first demo's scale.

Both are harmless when every scale is 1.0 (the shipped demos). Otherwise
re-retarget the legacy demo so every pkl carries `obj_scale`
([RETARGET.md](RETARGET.md)).
