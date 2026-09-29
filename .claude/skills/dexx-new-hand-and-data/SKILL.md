---
name: dexx-new-hand-and-data
description: Use when bringing new demonstration data, a new object/task, a new environment, or a different dexterous hand (or arm mount) into this repo — step-by-step porting with a verification after every step, including mandatory viser checks of the hand and object initial position and rotation before and after retargeting, the hand base frame / offset / mimic-joint / joint-order pitfalls, data screening, and env smoke tests.
---

# New data, new environment, new dexterous hand

Retargeting is sensitive to where the hand and the object start, and a new hand
brings its own base rotation, offsets and joint conventions. Most porting bugs
are silent: training runs, losses fall, and the policy learns the wrong task.
So **every step below ends with a check, and the checks that involve geometry
are visual (viser)**. Do not move to the next step until the current check passes.

Order: new data (works with the current hand) → new hand → new environment.
If you change the hand and the data at the same time, you cannot tell which one
is wrong. Keep one known-good demo as a control through all of it.

## Part A — new demonstration data / new object

**A1. Look at the raw demo.**
```bash
python tools/dataset/vis_sequence.py --sequence <task>/<seq>        # http://localhost:8080
```
Check, scrubbing the whole sequence: the hand approaches, grasps and moves the
object the way you remember the recording (not "puts it down" when it picks up);
the object mesh is the right size and sits on — not in, not above — the table at
frame 0; the MANO hand is a right hand for `--side right`; no teleporting frames
at the start or end (trim them in the source data first).

**A2. Fix the object's resting height.**
```bash
python -u tools/dataset/drop_test.py --task <task> --headless
```
Check: `mano_joints_corrected.pkl` gets a `z_bottom_offset`; re-run A1 and see
the object resting on the table. Meshes whose origin is not at their centre
(common for tools) are exactly the ones that end up floating or buried.

**A3. Screen the grasp before spending compute.** A demo is only learnable if
the human fingertips actually touch the object. With `vis_sequence.py`, scrub to
the grasp and check that at least two fingertips are within a few millimetres of
the mesh for a sustained part of the sequence (a useful bar: ≥2 fingertips within
8 mm on more than 10% of frames). Demos that fail this train to zero success no
matter how the reward is tuned — replace them instead.

**A4. Place it in the robot's workspace.** Decide the placement per demo:
object initial xy at a common work point in front of the arm, `z_offset` if the
source table height differs, and a yaw of the whole demo if the wrist would have
to twist to its joint limits (`tools/dataset/rotate_task_z.py --angle_deg … --pivot x y z`,
pivoting on the object/work point, never on the world origin; `--dry_run` first,
`--restore` undoes). Record the arguments —
placement is part of the data. Offsets and rotation are applied to the object,
the MANO joints and the wrist together; never to only one of them.

**A5. Retarget into a scratch root.**
```bash
python -u scripts/retarget.py --side right --data_idx rt/<task>/<seq> --dump_root logs/retarget_new --headless
```
Check the printed line: `[OK] reachable: mean arm EE err = … m`. Unreachable →
the placement is wrong (A4), not the optimiser; fix the placement first.

**A6. Visual check of the retarget (mandatory).**
```bash
python tools/dataset/view_retarget.py --data_idx rt/<task>/<seq>@0 --retarget_root logs/retarget_new
```
Frame 0 — the initial state the env resets to:
- object bottom − table ≈ 0 mm (a few mm either way); object centre where you placed it;
- the robot hand's axes (`/ee_fk`) coincide with the MANO wrist target axes
  (`/wrist_target`): EE rotation error a few degrees, not tens; a constant
  large rotation error = hand base frame problem (Part B3);
- palm facing the object from the side the human approached; thumb on the
  correct side; no finger inside the object or the table;
- wrist well above the table; arm not folded against a joint limit.
Then scrub the whole sequence: yellow tip-error lines stay short through the
grasp; fingertips reach the object surface during contact (`robot fingertips
within 8 mm` > 0 at the grasp); no frame-to-frame jumps of the arm (IK branch
flips show as the elbow swinging through).
`--summary_only` prints the frame-0 numbers for logs and scripts.

**A7. Use it.** Point training at the scratch root
(`--env_cfg robotool_batch_retarget_root=logs/retarget_new`) or move the pkls
into `data/retargeting/…`. Run a short teacher training on this demo alone before
mixing it with others.

## Part B — a new dexterous hand (or a new mount / arm)

The hand-specific touch points in this repo — change all of them, and grep for
the old hand's name when done:

| what | where |
|---|---|
| merged arm+hand URDF: mount joint xyz/rpy, pruned links | `scripts/build_merged_urdf.py` (`DEFAULT_JOINT_XYZ`, `DEFAULT_JOINT_RPY`, `PRUNE_PATTERNS`) |
| robot USD + self-collision filter pairs | `scripts/build_robot_usd.py`, `SELF_COLLISION_FILTER_PAIRS` in `franka_sharpa_env_cfg.py` |
| hand bodies, dofs, MANO↔robot body mapping | a `DexHand` class in `src/dexx/tasks/hand_imitation/envs/` (like `sharpa.py`), registered with `DexHandFactory` |
| env cfg: action/obs sizes, actuated joint names, fingertip bodies, contact sensors, elastomer ids, gains | `src/dexx/tasks/franka_sharpa/franka_sharpa_env_cfg.py` |
| hand/arm PD, armature, friction | `src/dexx/robot_constants.py` |
| end-effector link (`<side>_hand_C_MC`) | `scripts/retarget.py`, `src/dexx/tasks/hand_imitation/deploy/arm_fk.py` (`SIM_EE_LINK`), `deploy/ros2/wrist_state_publisher.py` |
| wrist frame derived from MANO | `src/dexx/tasks/hand_imitation/dataset/robotool_batch_dataset_dexhand.py` |
| retarget: hand joint list, fingertip list, real-hand clamp | `scripts/retarget.py` (`_build_hand_joint_names`, `tip_list`, `--dexhand`) |
| hand point-cloud body subsets | `HAND_BODY_SUBSETS` in `src/dexx/algo/dagger/pc_env_meta.py` |
| real-hand SDK, joint limits, joint order on hardware | deploy env, `sim2real/real_hand_limits.py`, docs/JOINT_ORDERING.md |

**B1. Read the vendor URDF.** Count actuated and mimic (coupled) joints; list
joint limits; find the palm/base link, the fingertip links and any tactile links;
note the mount pose on the flange. Where the vendor's documents disagree (joint
order, which channel is which joint), write the question down and settle it on
the hardware in B6.

**B2. Build the merged URDF and look at it.** Merge arm + hand at the measured
mount pose (`build_merged_urdf.py`). If the hand has no fingertip link at the
pad (the last link's origin is often the last joint), add a virtual fingertip
link at the pad centre, measured from the mesh. Check in viser (load the URDF
with `viser.extras.ViserUrdf`): zero pose looks right; move each joint alone and
see the named finger move in the positive-closing direction; the hand sits on the
flange with the palm where the real one is. `python scripts/check_asset_equivalence.py`
for kinematic regressions against a reference asset.

**B3. Fix the hand base frame convention.** The pipeline assumes the end-effector
frame of the hand matches the wrist frame the loader derives from MANO (fingers
along +z, thumb side along +y for a right hand, origin at the palm base). A new
hand whose base link uses another convention will be silently rotated or offset
by a constant. Check: in `view_retarget.py`, with the retargeted robot, the
EE axes coincide with the wrist target axes — a constant rotation error
(tens of degrees, or 90/180°) or a constant offset along the palm means the
convention differs. Fix it in the URDF (mount rpy / a fixed frame at the palm
base used as the EE link), not by tuning retarget weights.

**B4. Mimic / coupled joints.** pytorch_kinematics and many tools ignore
`<mimic>`: they treat coupled joints as free. Give the retargeter and the env an
explicit mimic table (child = multiplier × parent + offset), keep mimic joints out
of the action space, and write them from their parents every step. Check: in
viser, drive only the parents and see the children follow as on the real hand.

**B5. Register the DexHand.** Body names, dof names, `hand2dex_mapping` (MANO
keypoint → robot bodies; fingertips to the pad/virtual tip links, not to the
last joint), contact bodies. Check: every body and dof name exists in the URDF (`view_retarget.py` raises
on a missing joint; the env fails to resolve a missing body); `to_dex` /
`to_hand` round-trip for every MANO keypoint.

**B6. Joint order and sign, end to end.** Different tools order joints
differently (URDF parse order, pytorch_kinematics chain order, Isaac's sorted
order, the real SDK's channel order). Always map by **name**, and print when a
remap happens. Check: (1) FK replay — stored retarget joints through the URDF
reproduce the stored body positions (`view_retarget.py`: `FK vs retarget's
stored EE` ≲ a few mm); (2) on the real hand, command one joint at a time at low
speed and confirm the finger and direction (docs/JOINT_ORDERING.md).

**B7. Make the retargeter work for the hand, and prove it.** Generalise the
hand-specific lists in `scripts/retarget.py` (joint names, tip list, EE link,
mimic). Before trusting its error on the new hand, run the modified retargeter on
the **old** hand and a known demo and reproduce the old error — if it cannot, the
problem is the code, not the new hand's kinematics. Then A5–A6 for the new hand;
compare fingertip error with the old hand on the same demo.

**B8. Physics.** PD gains, armature (rotor inertia) and friction per joint;
self-collision filter pairs for adjacent links; convex decomposition for finger
collision meshes. Check in sim: 100 zero-action steps with the hand at rest — no
joint outside its limits, no jitter; a scripted full close and open reaches the
limits; self-collision on does not lock the fingers. Missing armature and dropped
filter pairs are the two failures that look like a bad policy.

## Part C — a new environment (task, table, objects, sensors)

**C1. Geometry in one place.** Table height, arm base, crop box live in
`src/dexx/deploy_config.py`; the env, the retargeter and deploy read them from
there. Changing the table means re-retargeting every demo. Check:
`python tutorial/01_frames_and_constants/check_frames.py`, then A6 on a demo.

**C2. Spaces and indices.** Declare action/observation sizes before the env is
built and resolve joint/body indices after the articulation exists; assert the
built observation size equals the declared one (a mismatch is otherwise silent).
Map every joint by name.

**C3. Multiple objects.** Spawn after the environments are cloned and look up
which object each env actually got from the stage; give each object mesh/USD a
unique file name (identical names collide in caches); load an object's own USD
without overriding its mass unless you mean to.

**C4. Contact and tactile.** One contact sensor per fingertip with the object
as filter (one sensor over several bodies silently drops the filter). Check:
non-zero contact force when a replayed grasp closes on the object.

**C5. Falsify the env before training** (minutes, not hours):
- zero action → success ≈ 0 (if it "succeeds", the success test is wrong);
- replaying the retargeted reference gives much higher reward than zero action;
- done/reward flags are read after the step that produced them (off-by-one
  flags look like a working env);
- an open-loop replay of the reference with PD control is a baseline, not a
  success rate: RL has to close the gap it leaves.

**C6. Short training, then look.** A short teacher run on one demo; watch the
strict success in the log; render or `play.py` and look at the grasp from the
object's side, not only the numbers. Then scale to more demos.

## Rules

- One change at a time; keep a known-good demo and the old hand as controls.
- Geometry problems are checked visually, not by reading losses.
- Never compensate a frame or offset error with retarget weights, reward terms
  or randomisation — fix the frame.
- Record every placement, offset and rotation you choose; they are part of the
  data.
