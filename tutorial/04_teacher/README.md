# 04 — The state expert

**Goal:** understand what the expert is allowed to see, why that is not cheating,
and what the 557 and 148 numbers are made of.

## Run it

The command and flags are in [TRAINING.md](../../docs/TRAINING.md) — for one demo,
`--task franka-sharpa-force-poseobs --side right --num_envs 2048` with
`--data_idx '["rt/0416_grasp/cube_small_2"]'`. Lower `--num_envs` if it does not
fit in memory; the script's default is 16384.

A pretrained expert ships at `checkpoints/teacher_poseobs.pth`, so you can skip
straight to lesson 05 if you only want the distillation.

## Asymmetric actor-critic

The actor sees a 557-d observation. The critic sees that **concatenated with** a
148-d privileged block, 705 in total. Only the actor is ever distilled, so the
critic can look at things no camera could resolve — it exists to reduce variance
in the value estimate, not to act.

### The actor's 557

`obs_future_length = 1`, and the hand has 32 tracked bodies.

| block | dim | |
|---|---|---|
| proprioception | 79 | `q`(22) + `cos q`(22) + `sin q`(22) + wrist quat/vel/angvel(10) + 3 zeros |
| wrist reference | 23 | target and delta: pos, vel, quat, angular velocity |
| hand reference | 288 | 32 bodies × 3 × (Δpos, vel, Δvel) |
| target object pose | 7 | the demo's next-frame object pose |
| fingertip distances | 5 | demo-precomputed |
| object shape (BPS) | 128 | static encoding of the mesh |
| tactile | 20 | 5 contact forces + 15 contact positions |
| noisy object pose | 7 | a pose *estimate*, with realistic error |
| **total** | **557** | |

Note the wrist's absolute position is deliberately zeroed — the policy sees only
deltas to the reference, which is what makes the reference transferable.

The env publishes this layout at startup as a named slot map, which lesson 05 then
uses to drop channels by name:

```
proprioception[0:79] ref_tracking[79:390] target_obj_pose[390:397]
tips_distance[397:402] obj_bps[402:530] tactile[530:550] obj_pose_tail[550:557]
```

### The critic's 148

`40 + 18K + 18` with `K = 5` future frames: 22 hand joint velocities, friction,
mass, centre of mass, the current object state (13), then five frames of object
target and fingertip distances, a delta against the current object, and the
current object-to-fingertip distances.

Hand **joint velocity** is the largest single block and is genuinely privileged —
the actor gets positions only.

## Watch the success rate, not the reward

```
Mean Rewards: <reward> | Success: <survival>% | Strict: <strict>% | Current Best: <best reward>
```

Two rates, both running means over the last 100 completed episodes:

* **Success** — reached the trajectory's end without a failure termination. That
  is *survival*.
* **Strict** — survival **and** the object finished within 3 cm of its demo
  endpoint with no object-position drift; episodes of at most 5 steps (bad inits)
  count as failures. It is a conservative *training proxy*, **not** the `strict3`
  that `eval.py` reports: evaluation excludes bad inits from the denominator and
  does not require survival. [TRAINING.md](../../docs/TRAINING.md) has the exact
  difference.

Reward and the two rates do not have to move together: an epoch can gain reward
while both success rates fall. Select checkpoints on the rates.

Getting this number right is subtler than it looks. `success_buf` is 1 only on
the step an episode ends and is cleared at reset, so averaging it over all
environments every step gives a near-zero number that is not a success rate at
all. It has to be sampled from the environments that just terminated — which is
what `algo/ppo/ppo.py` does, the same way it treats the episode reward. [TRAINING.md](../../docs/TRAINING.md) has the details.

## The reward is not five weights

It is a flat sum of about 25 exponential-kernel terms with individually tuned
weights: object position at 8.0, object rotation at 6.0, wrist position at 4.0,
per-finger tracking at 0.5–0.9 multiplied by the absolute-tracking weight 2.0
(effective 1.0–1.8) with per-finger temperatures, plus contact, approach, no-slip
and smoothness terms. `compute_imitation_reward` in `franka_sharpa_env.py` holds it.

If you are porting, do not start by re-deriving this. Start by reproducing the
tracking terms and adding shaping only where a failure mode demands it.

## Porting

New task or objects: change `--data_idx`, nothing else.
New robot: the action dimension and the body count change, so the observation
width changes and no existing checkpoint loads. Expect to retrain.
