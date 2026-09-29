# 05 — Distilling a point-cloud student

**Goal:** replace privileged state with a camera and a PointNet, and understand
what the student is and is not given.

## Run it

Run the canonical command in [DISTILLATION.md](../../docs/DISTILLATION.md) §Step A
with `--out_dir logs/my_student`; later lessons read from there. It uses the four
shipped demos and the shipped teacher.

Loss should fall steadily across the run while β decays from 1.0 to about
0.009 (×0.85 per iteration over 30 iterations). Do not read `roll_succ` in the
per-iteration log as a success rate: it only counts episodes that end inside the
32-step rollout window, so it stays low and flat even at iteration 1, when the
rollout policy is the teacher. The loss curve and `eval.py` are what you check.

**Pass `--camera_extrinsic` explicitly.** Omitting it falls back to
`current.npy`, which is right today and silently wrong after the next
recalibration. Lesson 06 explains why.

## DAgger, precisely

The executed action is a **convex blend**, not a coin flip between two policies:

```python
action = beta * expert_action + (1 - beta) * student_action
```

β starts at 1.0 and is multiplied by 0.85 each iteration. The replay buffer is
aggregated across iterations (capacity 2×10⁵) and never cleared — that is what
makes this DAgger rather than iterated behaviour cloning. The loss is a plain MSE
against the expert's action.

## What the student gives up

The teacher keeps all 557 dims for labelling. The student's copy is **sliced**:

| dropped | dim | why |
|---|---|---|
| `obj_bps` | 128 | object shape must come from the point cloud, not a mesh encoding |
| `tips_distance` | 5 | demo-precomputed, unavailable at deploy |
| `obj_pose_tail` | 7 | the object pose *estimate* — the student must localize visually |

557 − 140 = **417**. Slicing rather than zeroing means the student has no dead
inputs and a layout mismatch fails on a shape error instead of silently feeding
unlearned signal.

Note what is **not** dropped: `target_obj_pose` (390:397), the demonstration's
*target* object pose. The student does not know where the object is; it does know
where the object is supposed to go.

## The point cloud

| source | points | per-point channels |
|---|---|---|
| scene (depth camera) | 1024 | xyz + type + force(0) |
| hand keypoints | 6 | wrist + 5 fingertips |
| tactile | 25 | 5 fingertips × 5 surface samples, force attached |

One unified cloud of **1055 × 5** through a shared PointNet with masked
max-pooling → 64 dims. The student MLP takes 417 + 64 = **481** and outputs the
same 29 actions.

## Check

The checkpoint records its own layout. After training:

```python
c = torch.load("logs/my_student/dagger_final.pth", weights_only=False)
print(c["proprio_dim"], c["student_drop_slots"], c["student_obs_slots"])
# 417  ['obj_bps','tips_distance','obj_pose_tail']  {'proprioception': (0,79), ...}
```

Evaluation re-applies that layout automatically and refuses to run if the live
slot map disagrees with the saved one.

## What about `scripts/train_ppo_pc.py`?

It fine-tunes a distilled student with asymmetric PPO, warm-started from the DAgger
checkpoint. **It is not part of the pipeline, and it refuses lean students** —
the only kind that can be deployed. It ships for experiments on full-observation
students;
[DISTILLATION.md](../../docs/DISTILLATION.md) §Step B has the flags.

## Porting

Changing the observation layout changes the slot map, which invalidates saved
`keep_idx`. The guard will catch it. Re-train the student; the teacher is
unaffected only if its own observation did not change.
