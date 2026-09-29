# 01 — Frames and constants: the sim2real contract

**Goal:** know every number that has to mean the same thing in the retargeter, the
simulator, the point-cloud crop and the real robot — and know what happens when
one copy drifts.

**Why this is lesson 01:** frame bugs do not raise exceptions. They produce a
policy that trains beautifully and reaches for empty air. Every other lesson
assumes these are right.

## One file

The sim2real geometry, camera and comm constants live in
`src/dexx/deploy_config.py`. Edit there; the env cfgs, the deploy env, the depth
subscriber, the retargeter and the calibration tools take their values from it.

| constant | value | meaning |
|---|---|---|
| `TABLE_SURFACE_Z` | 0.415 | table top, env-local. Objects rest relative to this; the env, the env cfg's table box and the retargeter all read it |
| `ARM_BASE_Z` / `ARM_BASE_POS` | 0.415 / (-0.1, 0, 0.415) | `fr3_link0` origin, env-local |
| `SIM_INTRINSICS` | fx 193.33, fy 193.06, cx 160.08, cy 121.05 | depth camera at 320×240 (D455 640×480 decimated ×2) |
| `DEPTH_H`, `DEPTH_W` | 240, 320 | depth resolution |
| `PC_WORKSPACE_MIN/MAX` | (0.00, -0.40, 0.417) … (0.80, 0.25, 0.70) | point-cloud crop box, env-local; recorded in every student checkpoint and restored at eval/deploy |
| `POLYMETIS_STATE_PORT` / `_CMD_PORT` | 5560 / 5561 | ZMQ bridge to the arm |
| `CAMERA_ZMQ_ADDR_EXAMPLE` | an example address | shown in help text only; pass `--depth_zmq_addr` |
| `CAMERA_EXTRINSIC_DEFAULT_FILE` | `calib/camera_align/current.npy` | the extrinsic loaded when `--camera_extrinsic` is omitted (lesson 06) |
| `SHARPA_SDK_ENV` | `SHARPA_SDK_PYTHON` | env var naming the Sharpa SDK's `python/` directory |

Two things are deliberately **not** in that file:

- **The camera extrinsic.** It is a property of how the camera is bolted down, so
  it is a calibrated file in `calib/camera_align/`, passed with
  `--camera_extrinsic` (lesson 06). `deploy_config` only names the default file.
- **The bridge's ports.** `deploy/polymetis_joint_bridge.py` runs on the NUC,
  where `dexx` is not installed, so its `5560` / `5561` defaults are literals that
  must be kept in sync by hand.

## The table and the arm base are independent

They are equal in the reference setup (both 0.415) because the robot is bolted level
with the table surface. A different mount makes them differ.

That coincidence is a trap. The human demonstration is lifted into the scene by
the loader's `mujoco2gym` transform, whose translation is the table height.
Suppose it were written as `ARM_BASE_Z` rather than `TABLE_SURFACE_Z`, because the
two are the same number. Lower the base and the
demonstration's hand trajectory follows it down while the object, anchored to the
table, stays put. The grasp is then silently too low by exactly the change in
mount height.

The rule that falls out:

> A constant must be written as **what it physically is**, not as whatever other
> constant happens to share its value today.

The env and the retargeter both build that transform from `TABLE_SURFACE_Z`, and
`check_frames.py` asserts it. Moving the arm base does not move the demonstration.
(The deploy's wrist offset is a different quantity: the wrist is computed in the
arm-base frame, so it is moved to env-local by `ARM_BASE_POS`.)

### What *should* change when the base moves

Moving the robot changes which joint angles reach a given point. It does not
change where the object is or where the hand must go to grasp it. So after a
re-mount and a re-retarget, expect exactly this:

| quantity | expected |
|---|---|
| `object_pos` | **unchanged** — it is anchored to the table |
| `arm_joint_pos` | **changed** — the arm solves from a new base |
| closest fingertip-to-object distance | **unchanged** to within a millimetre or two |

If the fingertip-object relationship moved, the demonstration was dragged along
with the base and something is still coupled that should not be.

## Retarget placement offsets are part of the data

Each demonstration is retargeted with a placement offset deciding where its object
lands. They are CLI arguments and they differ per sequence. Re-running a retarget
without them moves the object 10 cm and nobody notices.

| sequence | arguments |
|---|---|
| `rt/0416_grasp/cube_small_1` | defaults (`--target_offset_xy 0.45 0.0`, `--z_offset 0`) |
| `rt/0416_grasp/cube_small_2` | defaults |
| `rt/0420_manip/squeegee_1` | `--target_offset_xy 0.45 0.1 --z_offset 0.01` |
| `rt/0420_manip/squeegee_2` | `--target_offset_xy 0.45 0.1 --z_offset 0.01` |

They are recoverable from data if lost: `target_offset_xy` **is** the first
object's xy, and `z_offset` is the difference in its z. Add a row when you
retarget a new sequence — lesson 03.

## The workspace crop

`PC_WORKSPACE_MIN/MAX` is applied before the point cloud is subsampled, so the
1024 points land on the manipulation region instead of being spent on the table
and the background curtain. The floor sits 2 mm above the table top on purpose:
level with the table top, the table takes most of the points; a few millimetres
higher, the object's base is cut off. The ceiling of 0.70 removes the robot's own
arm, which would otherwise take a large share of the points and whose pose the
policy already has from forward kinematics.

This box must match on both sides. Eval and deploy restore it from the checkpoint,
so the student sees the crop it trained on. A deliberate deviation is possible:
`--pc_workspace_min 0.0,-0.40,0.422` raises the floor 5 mm above training's, for
when the real table sits a few mm higher in the cloud than the simulated one and
would otherwise leak into it. Make such an override on purpose,
knowing why, and check the result with lesson 06's cropped overlay.

## Check

```bash
python tutorial/01_frames_and_constants/check_frames.py
```

Pure arithmetic, no simulator. It asserts the relationships rather than the
values, so it still passes after you re-mount your robot — and fails if you break
the decoupling.

## Porting

| you changed | edit | then |
|---|---|---|
| table height | `TABLE_SURFACE_Z` | re-retarget (03), re-crop check |
| arm mount height | `ARM_BASE_Z` | re-retarget (03); expect joints to change and object not to |
| camera | `SIM_INTRINSICS`, `DEPTH_H/W` | recalibrate the extrinsic (06) |
| workspace | `PC_WORKSPACE_MIN/MAX` | keep the deploy-side crop identical |
| arm networking | `POLYMETIS_*` ports | lesson 09 |

Changing either height invalidates existing retargets and every policy trained on
them. That is not a warning to be careful — it is a re-run.
