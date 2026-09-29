# 03 — Retargeting a human demonstration onto the robot

**Goal:** turn a MANO hand trajectory into a robot joint trajectory the simulator
can replay, and verify it rather than trust it.

## Run it

The command and flags are in [RETARGET.md](../../docs/RETARGET.md). Point
`--dump_root` at a scratch directory such as `logs/retarget`: the default output
root is where the shipped demos live.

Two stages of Adam: stage 1 solves the arm alone to reach the wrist target, stage
2 opens up the hand and the arm together to match fingertips. The default
`--iter 4000` is the recommended value.

**Pass the sequence's placement offsets** (lesson 01's table). Omitting them moves
the object and confounds whatever you were actually testing.

## The reachability gate

After fitting, the mean arm end-effector error over the trajectory is compared
against `--reachability_th` (0.08 m). Above it, the variant is marked
`reachable=False` and a partial pkl is written, which the training loader skips.

Healthy values are a small fraction of the threshold — a couple of centimetres.
A sequence close to the threshold is telling you the demonstration does not fit
your robot's workspace, not that it needs more iterations.

## Two things that will waste your afternoon

**The output path is also the input path.** By default `scripts/retarget.py`
writes to `data/retargeting/robotool_batch/mano2sharpa_rh/<task>/<obj>@<n>.pkl`,
the same root the dataset loader reads — no copy step, but also no protection: a
retarget run replaces what training will use next. An unreachable result never
overwrites a reachable one (`[KEEP]`), but a worse reachable one will. Retarget
into a scratch `--dump_root`, check it, and then either move it into place or
point training at it (`--env_cfg robotool_batch_retarget_root=<dir>`, `--ref_root`).

**Isaac Sim can hang on exit.** If the simulation context is still alive when
`simulation_app.close()` runs, the process can finish its optimization, write its
pkl, and then sit at 100% CPU forever. `retarget.py` releases the context first
(as `env.close()` does in the gym scripts) and exits normally, so a plain loop
works:

```bash
for seq in 0416_grasp/cube_small_1 0416_grasp/cube_small_2; do
  python -u scripts/retarget.py --data_idx rt/$seq --side right --headless \
      --dump_root logs/retarget_scratch        # note -u: see below
done
```

If you write a new Isaac entry point, close the env (or clear the
`SimulationContext`) before `simulation_app.close()`, or it will hang the same way.

The `-u` matters: without it Python block-buffers stdout to a file and the log can
sit unchanged for a long time while the run is perfectly healthy. Do not diagnose
a stalled log before you have ruled that out.

## Verify before you train on it

Look at the demonstration itself first: `tools/dataset/vis_sequence.py --sequence
<task>/<seq>` shows the hand and object as the loader will see them. Rotating a
task, fixing the object's resting height or a hand offset are all in
[tools/dataset](../../tools/dataset/README.md); every such edit needs a re-run of
the retarget.

Check the reachability numbers the run printed (above), then look at the result
the way the environment will load it:

```bash
python tools/dataset/view_retarget.py --data_idx rt/<task>/<seq>@0 [--retarget_root logs/retarget]
```

It draws the robot (FK of the retargeted joints on the merged URDF), the object,
the table and the MANO targets in env-local coordinates, with a frame slider, and
prints the frame-0 checks (`--summary_only` prints them without a browser):
object bottom vs table, wrist height, end-effector position and rotation error,
per-finger tip error, fingertips within 8 mm of the object. Scrub through the
grasp: the closest fingertip-to-object distance is the number that tells you the
*hand-object relationship* survived, which is what lesson 01 is protecting.

`FK vs retarget's stored EE` replays the stored joints through the URDF and
compares with the body positions the retarget stored: more than a few mm means
the joints are being applied in the wrong order or to the wrong URDF (lesson 02).

## Porting: a new task or new objects

1. Collect the MANO demonstration and object trajectory in the source format.
2. Retarget it; record its placement offsets in lesson 01's table.
3. Check reachability, then `view_retarget.py`: frame 0 and the grasp.
4. Add it to `--data_idx` for the teacher (04) and the student (05).

A new object needs its mesh present, not just its trajectory: the teacher reads
the object's shape as a BPS encoding of the mesh, the simulator needs it for
contact, and the depth camera renders it. The lean student does not read BPS — it
sees shape only through the point cloud — but it is labelled by a teacher that does.
