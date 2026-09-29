# 08 — Evaluation that means something

**Goal:** produce a number you would defend, which mostly means knowing what your
evaluator is quietly choosing for you.

## Run it

Use the command in [EVAL.md](../../docs/EVAL.md) with your lesson-05 student and
the four shipped demos:

```bash
python scripts/eval.py --load_path logs/my_student/dagger_final.pth \
  --data_idx '["rt/0416_grasp/cube_small_1","rt/0416_grasp/cube_small_2","rt/0420_manip/squeegee_1","rt/0420_manip/squeegee_2"]' \
  --num_envs 64 --max_episodes 200 --out_dir logs/my_eval \
  --camera_extrinsic calib/camera_align/current.npy --headless
```

The student's observation layout is restored from the checkpoint automatically,
and the run aborts if the live slot map disagrees with the saved one.

## Two defaults worth understanding

### Collecting the first N episodes biases the result

A successful episode ends sooner than a failing one. Collect "the first 200
episodes to finish" and whichever demonstration finishes fastest contributes the
most of them. A fixed episode budget can split very unevenly across demos, with
one demo contributing about twice as many episodes as another.

The aggregate then depends on an accident of timing. And the direction is not
predictable: the biased estimate comes out low when the over-represented demo
happens to be a weak one, and high when it is a strong one.

The default stays first-to-finish (`--per_demo_quota 0`), because that is the
reference protocol and new numbers should be comparable to numbers measured
with it. So read the summary's closest-approach
`success_rate_per_demo`, the demo-averaged **macro** rate and the episode-weighted
**micro** rate together: when they disagree, the aggregate is being pulled by the
mix of demos. For a balanced number pass `--per_demo_quota -1`
(`ceil(max_episodes / n_demos)` per demo); macro and micro then coincide.

### A clean evaluation hides what tactile is for

Physical domain randomization — object mass, friction, centre of mass, hand PD
gains — stays on by default, as in the reference protocol.
`--no-keep_physics_dr` holds it at nominal for a cleaner run (not fully
deterministic: arm PD-gain and action-delay randomization stay on either way;
[EVAL.md](../../docs/EVAL.md) lists exactly what is switched off). But the clean
run removes the only variation contact force could help with: a policy cannot
demonstrate a benefit from sensing grip force when every object weighs exactly
its nominal mass.

```bash
--no-keep_physics_dr    # mass, friction, COM and hand PD gains held at nominal
```

The same checkpoint can score much higher on a clean run than with physics DR
on. Neither number is wrong. They answer different questions, and a paper that reports
only the first is answering the easier one.

## The number to report is strict3, and the script computes it

The env's own success flag means "reached the end of the trajectory without a
failure termination". That is survival, not task success. The protocol metric,
**strict3**, checks whether the object finished within 3 cm of the demo's
final pose, with no object-position drift, excluding bad inits
(`survival_len <= 5`).

The separate `success_rate_*` fields in this evaluator check whether the object
came within `--success_dist` at any point in the episode. They are closest-approach
rates, not the env's trajectory-completion flags.

```
[EvalPC] strict success (end_final_dist < N cm, no obj_pos_drift, bad inits excluded: <b>/<n>):
    strict2   <k2>/<n-b>  ( <p2>%)
    strict3   <k3>/<n-b>  ( <p3>%)
    strict5   <k5>/<n-b>  ( <p5>%)
```

Note that strict3 deliberately excludes **only** object-position drift, not the
other failure causes. ORing them all in would change what the number means
without a reader being able to see it. It does not require the env's success
flag. Training's `Strict` is a separate conservative proxy: it requires successful
trajectory completion and counts bad inits as zeros rather than removing them
from the denominator. See [TRAINING.md](../../docs/TRAINING.md).

## Read the per-demo breakdown

The aggregate hides the interesting part. Put each demo's closest-approach rate
next to its strict3 rate. Per-demo rates can differ by tens of points between the
two: a demo that frequently approaches the target but finishes with poor endpoint
accuracy scores high on one and low on the other. An aggregate, on either metric
alone, hides that distinction. The commands that evaluate the shipped checkpoints
are in [checkpoints/README.md](../../checkpoints/README.md).

## Check

The summary JSON carries `per_demo_quota`, `success_rate_per_demo`,
`success_rate_macro` and `success_rate_micro`. If macro and micro differ, your
episodes are unbalanced and you should say so when you quote the number.
