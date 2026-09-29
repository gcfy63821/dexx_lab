# 06 — Camera extrinsic calibration

**Goal:** make the simulated depth camera sit where the real one sits, and be able
to prove it.

## What the number is

A 4×4 **camera-in-armbase** transform in the ROS optical convention (x right,
y down, z forward). Everything else about the camera — intrinsics, resolution,
depth clamps — lives in `deploy_config.py` as a constant. The extrinsic does not,
because it is a property of how you bolted the camera down, not of the camera.

Shipped calibrations are in `calib/camera_align/`.

## Look at them first

```bash
python tutorial/06_camera_calibration/inspect_extrinsic.py
```

No camera needed. It validates each matrix (orthonormal rotation, `det = +1`,
proper bottom row), describes where the camera is and which way it looks, and
reports the drift between any two.

That last part is the lesson. The drift between two calibrations is the
viewpoint error a policy inherits if it is trained with one and deployed with the
other. Nothing in training warns about it: a policy trained on a stale extrinsic
converges and its losses look fine. (The extrinsic the shipped student expects
is listed in [checkpoints/README.md](../../checkpoints/README.md).)

> **A calibration is an input, not a constant.** If it can go stale, it must be
> passed explicitly at every stage that consumes it.

So every stage takes it as a flag:

```bash
--camera_extrinsic calib/camera_align/current.npy
```

on `train_dagger_pc.py`, `train_ppo_pc.py`, `eval.py`, `play.py` and
`deploy_pc.py` alike (`deploy_pc.py` requires it). Omit it on the others and the
code loads `calib/camera_align/current.npy`
(`deploy_config.default_camera_extrinsic`) and says so; there is no hard-coded
matrix, and a missing file is an error. That fallback is right only until the
next recalibration, so pass the file for any run you mean to reproduce. The
shipped files and their drifts are listed in
[calib/camera_align/README.md](../../calib/camera_align/README.md).

## Calibrating your own

The toolchain ships in `tools/calib/`. Every step below is a real command.

What you need: the robot powered and reachable over Polymetis (bridge up, see
[DEPLOY.md](../../docs/DEPLOY.md)), the camera host running
`deploy/realsense_depth_zmq_pub.py`, and an initial guess for the extrinsic. For a
new mount, get it from a marker-based hand-eye calibration (below); if the camera
has only been nudged, start from `current.npy`.

### Initial extrinsic: easy_handeye (new mount)

ICP only converges from a guess within a few centimetres and degrees. A fresh
mount gets that guess from [easy_handeye2](https://github.com/marcoesposito1988/easy_handeye2)
(ROS 2; [easy_handeye](https://github.com/IFL-CAMP/easy_handeye) on ROS 1),
**eye-on-base**: camera fixed, an ArUco/ChArUco marker rigidly mounted on the arm.

1. Bring up the arm under ROS 2 (franka_ros2) so that TF has `fr3_link0` (or
   `base`) and the flange `fr3_link8`, the RealSense driver with its colour
   stream, and a marker tracker (e.g. `aruco_ros`) publishing the marker frame.
2. Launch the calibration with `calibration_type: eye_on_base`,
   `robot_base_frame: fr3_link0` (or `base`), `robot_effector_frame: fr3_link8`,
   `tracking_base_frame: camera_color_optical_frame`, and your marker frame.
   Move the arm freehand (`freehand_robot_movement: true`) or through its
   generated poses; take 15–20 samples that vary the marker **orientation** a lot
   (tilt it ±30° about two axes, not just translate it), all with the marker
   fully visible and in focus. Compute and save; the result is
   `~/.ros2/easy_handeye2/calibrations/<name>.calib`.
3. Convert it to this codebase's convention (the depth optical frame in
   `fr3_link0`). easy_handeye tracked the *colour* camera; the deploy
   back-projects *depth*, so pass the camera's depth-to-colour extrinsic
   (print it on the camera host):

   ```bash
   python tools/calib/handeye_to_extrinsic.py --print_realsense_extrinsic   # camera host
   python tools/calib/handeye_to_extrinsic.py \
       --calib ~/.ros2/easy_handeye2/calibrations/<name>.calib \
       --color_T_depth <TX TY TZ QX QY QZ QW printed above> \
       --out calib/camera_align/extrinsic_YYYYMMDD_handeye.npy
   ```

   If `robot_base_frame` was `base`, confirm it coincides with the arm base:
   `ros2 run tf2_ros tf2_echo base fr3_link0` should print identity.
4. Use the file as `--init` for the ICP below. Do not train or deploy on it
   directly: a marker estimate is typically off by a few millimetres to a
   centimetre and about a degree, which is enough to move the crop boundary
   through the table.

### Recommended: multi-pose arm ICP, then level to the table

This is the two-step procedure that produced the shipped `current.npy`.
It splits the six degrees of freedom by what each reference
can actually pin down:

* **The arm** constrains all six, because it is a complex 3-D shape. But one
  configuration seen from one viewpoint can leave a DOF weakly observed (a
  mostly-vertical arm says little about rotation about its own axis), and ICP of
  a one-sided cloud against closed meshes can slide by a few tenths of a degree.
  So capture several *different* poses and solve them jointly.
* **The table** constrains only three — height and two tilts — but constrains
  them exactly, because sim puts it perfectly flat at `TABLE_SURFACE_Z`.

So: take yaw and in-plane translation from the arm, then re-fit height and tilt
against the table.

**a. Capture 3–5 poses.** Move the arm by teleop or hand-guiding to a
configuration that is in view and spread out, hold it still, and capture:

```bash
python tools/calib/calibrate_extrinsic_live_icp.py capture \
    --session logs/calib_live --ip <NUC_IP> \
    --depth_zmq_addr tcp://<CAM_HOST>:5562
```

Repeat for each pose, into the same `--session`. Each capture stores the
median of 21 depth frames and the joint angles read from the Polymetis bridge
at the same moment (`pose_XX.npz`); it never commands the arm. It refuses a
capture if the arm moved while the frames were collected, or if the joint
vector is bit-identical to an earlier pose — the signature of a bridge serving
a stale reading.

**b. Solve all poses jointly against the arm:**

```bash
python tools/calib/calibrate_extrinsic_live_icp.py solve \
    --session logs/calib_live \
    --init calib/camera_align/current.npy \
    --min_z 0.44 \
    --out calib/camera_align/extrinsic_YYYYMMDD_icp.npy
```

The sim reference is the FR3 arm from the merged URDF, posed by forward
kinematics at each pose's recorded joints, with faces turned away from the
camera culled. Real points farther than `--near_arm` (10 cm) from it are
dropped. `--min_z` (env-local metres; here 2.5 cm above the table) also drops
the table points next to the arm, which have nothing to match in an arm-only
reference and otherwise pull the fit. The hand is not in the reference — its
joints are not on the bridge — unless you pass `--hand_from_pkl` with the real
hand held at that demo frame. It prints the correction, the ICP residual, and
the table height/tilt before and after, and dumps `real_solved.ply` /
`sim_target.ply` to `logs/calib_live_icp/` for inspection.

**c. Level to the table:**

```bash
python tools/calib/level_extrinsic_to_table.py \
    --init calib/camera_align/extrinsic_YYYYMMDD_icp.npy \
    --session logs/calib_live \
    --out calib/camera_align/extrinsic_YYYYMMDD_tablelevel.npy
```

Same captures, now back-projected with the ICP result. Points within 6 cm of the
FK-posed arm are excluded (an arm parked over the table drags the plane fit and
reads as tilt), a trimmed plane is fitted, and the extrinsic is rotated about
the workspace centre so the plane is level, then shifted so it sits at
`TABLE_SURFACE_Z`. Yaw and in-plane translation are left as the ICP found them.
Only valid when the real table top is bare: a mat's thickness becomes a height
error.

Both tools require `--out`, refuse `current.npy`, and refuse to overwrite an
existing file without `--overwrite`. Then check the result by eye with the viser
overlay (step 5 below, `--init_extrinsic` set to the levelled file) and record
the drift (see "Record what you changed" below).

### Alternative: single-frame ICP and manual alignment

One retargeted demo frame, one ICP, and a viser session with sliders. Use it when the arm cannot be moved freely, when there is no bare
table to level against, or as a manual fallback to nudge a result by eye.

#### 1. Put the arm somewhere both sides agree on

Calibration works by matching a real picture of the robot against a simulated
one, so both have to be in the same configuration. Drive the real arm to a
demonstration frame and hold it there:

```bash
python deploy/move_to_frame_polymetis.py --ip <NUC_IP> \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --hold
```

Pick a frame where the arm is *in view and spread out*. A folded arm gives ICP
almost nothing to lock onto. `--dry_run` prints the target and the start-pose
delta without moving.

#### 2. Capture the real cloud

With the camera host publishing depth on ZMQ:

```bash
python tools/calib/capture_multiframe_zmq.py \
    --addr tcp://<CAM_HOST>:5562 --n_frames 10 \
    --out_dir logs/calib_real
```

Ten frames of a held-still scene are accumulated into one dense cloud
(`logs/calib_real/accum.npz`, points in the **camera** frame). Overlaying frames
fills in dropout and shows you how much the sensor jitters. The publisher sends
depth only, so the cloud is uncoloured. `--addr` is required.

#### 3. Render the matching sim cloud

Forward kinematics on the same frame, sampled to a point cloud:

```bash
python tools/calib/gen_sim_frame_ply.py \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --out logs/calib_sim_frame.ply
```

This is the reference the real cloud gets aligned to. The table plane is included
(at `TABLE_SURFACE_Z`) because it is a large, unambiguous surface — useful for
pinning down height and tilt even though it says nothing about yaw.

#### 4. Align automatically (ICP)

```bash
python tools/calib/icp_align_extrinsic.py \
    --real_npz logs/calib_real/accum.npz \
    --init_extrinsic calib/camera_align/current.npy \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --box -0.2 0.6 -0.2 0.2 0.0 0.6 \
    --out logs/calib_icp_it1.npy
```

Kabsch/SVD ICP in the arm-base frame: apply the initial extrinsic to the real
cloud, crop both to the same box, match, reject correspondences beyond
`--max_corr`, and fold the resulting rigid transform into the extrinsic.

**Iterate.** Feed its output back in as `--init_extrinsic` two or three times; the
translation correction should shrink each round. If it does not, the initial
guess is too far off or the crop box contains something that is not in the sim
model (a clamp, a cable, your hand).

#### 5. Verify by eye, and nudge

ICP will converge confidently to a wrong answer on a scene that is mostly a flat
table. This step is not optional.

```bash
python tools/calib/live_calibrate_extrinsic.py \
    --real_pc_npz logs/calib_real/accum.npz \
    --sim_frame_pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --sim_frame_idx 0 \
    --init_extrinsic logs/calib_icp_it1.npy \
    --out calib/camera_align/extrinsic_YYYYMMDD.npy --port 8080
```

Write to a **new, dated file**. `current.npy` is what the shipped student was
trained with; overwriting it silently changes the viewpoint of every run that
relies on the default.

Opens a viser session in your browser: the real (uncoloured) cloud over the sim robot
meshes, with six sliders for translation and rotation. Look along each axis in
turn. The arm's *silhouette* is what you are matching — if the real link edges
sit inside or outside the sim mesh consistently, that is a translation error; if
they diverge along the arm, it is rotation.

Save writes the 4×4 to `--out`.

#### 6. Verify what the policy will actually see

The policy never sees the raw cloud — it sees the cropped one. A small extrinsic
error can remove different things on each side.

```bash
python tools/calib/viz_cropped_overlay.py \
    --real_npz logs/calib_real/accum.npz \
    --extrinsic calib/camera_align/extrinsic_YYYYMMDD.npy \
    --sim_link0_ply logs/calib_sim_frame.ply --port 8081
```

(`logs/calib_real/accum.npz` is step 2's capture of the same arm
pose; the tool moves it into env-local with the extrinsic and crops it.)

(`logs/calib_sim_frame.ply` is step 3's output, in the `fr3_link0` frame by default.)

Both clouds get the same workspace box (lesson 01; the tool's default is
`PC_WORKSPACE_MIN/MAX`, pass `--ws_min/--ws_max` to test an override such as a
raised 0.422 floor). Confirm the table is gone from both and that the object
and hand points that survive line up.

### Record what you changed (either route)

```bash
python tutorial/06_camera_calibration/inspect_extrinsic.py
```

It prints the drift between your new file and the previous one. A few millimetres
is re-measurement noise. A few centimetres means the camera actually moved — and
every policy trained before that move is now trained on the wrong viewpoint.

### The constants these tools read

The calibration tools take their defaults — table height, arm-base position, crop
box, depth resolution and intrinsics — from `dexx.deploy_config`, the same place
the trainer reads them. That matters: a tool with its own hard-coded copy of, say, the
arm-base height keeps the old value after a re-mount and calibrates against the
wrong geometry.

## Sanity checks that catch most errors

* **Table plane.** Back-project the real depth and fit a plane. It should come out
  at `TABLE_SURFACE_Z` and level. A tilt of half a degree is enough to make the
  table survive the crop on one side of the image and not the other.
* **Object count.** After cropping, the object should be a few hundred points. If
  it is a handful, the crop floor is too high or the extrinsic is off in z.
* **Re-run the inspector** against your previous calibration. A few millimetres is
  re-measurement noise; a few centimetres means the camera moved.

## Porting

Moving the camera, changing lens or resolution, or re-mounting the arm all
invalidate the extrinsic — re-mounting the arm because the transform is expressed
*in the arm base frame*.

Changing resolution or camera model also means new `SIM_INTRINSICS` (lesson 01).
Check the numbers are the **depth** stream's, not the colour stream's: on a D455
those differ substantially, and using colour intrinsics narrows the modelled field
of view and quietly drops a third of the workspace.

`check_frames.py` from lesson 01 prints the horizontal FoV your intrinsics imply —
compare it against the datasheet.
