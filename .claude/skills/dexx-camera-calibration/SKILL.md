---
name: dexx-camera-calibration
description: Use when calibrating, refining, checking or switching the RealSense camera extrinsic (camera-in-armbase 4x4) for this repo — initial eye-on-base hand-eye calibration with easy_handeye/easy_handeye2, converting it with handeye_to_extrinsic.py, multi-pose arm ICP plus table levelling, viser overlay checks, and managing calib/camera_align/*.npy files for training and deploy.
---

# Camera extrinsic calibration

What is being calibrated: a 4×4 transform of the **depth optical frame**
(x right, y down, z forward) in the arm base `fr3_link0`, stored as
`calib/camera_align/<name>.npy`. The sim renders the training point cloud from
this pose and the deploy back-projects real depth with it, so **train and deploy
must use the same file**. Procedure details: tutorial/06_camera_calibration.

## When to (re)calibrate

- New camera mount, or the camera/arm/table was moved or re-mounted.
- `ARM_BASE_Z` changed (the extrinsic is relative to the arm base).
- Symptoms: the policy reaches consistently offset; table points leak through the
  crop floor on one side; the real cropped cloud does not overlay the sim one.

A new calibration is a new input: a policy trained with the old file is out of
distribution with the new one. Plan to retrain the student (or at least evaluate
it in sim with the new file) after a large change.

## 0. Initial guess — easy_handeye (new mount only)

ICP needs a start within a few cm / degrees. For a new mount use
[easy_handeye2](https://github.com/marcoesposito1988/easy_handeye2) (ROS 2) or
easy_handeye (ROS 1), **eye-on-base**:

- Marker (ArUco/ChArUco, printed flat, measured size entered exactly) rigidly on
  the flange or the hand; camera fixed.
- Frames: `robot_base_frame: fr3_link0` (or `base` if it coincides — check
  `ros2 run tf2_ros tf2_echo base fr3_link0` is identity), `robot_effector_frame:
  fr3_link8`, `tracking_base_frame: camera_color_optical_frame`, marker frame from
  the tracker (e.g. `aruco_ros`).
- Samples: 15–20, spanning **orientation** (tilt the marker ±30° about two axes),
  not just position; marker fully visible, in focus, not at the image edge;
  wait for the arm to settle before each sample. Discard samples where the
  tracker flickers.
- Save → `~/.ros2/easy_handeye2/calibrations/<name>.calib`, then convert (colour →
  depth frame, since the deploy uses depth):

```bash
python tools/calib/handeye_to_extrinsic.py --print_realsense_extrinsic          # on the camera host
python tools/calib/handeye_to_extrinsic.py --calib ~/.ros2/easy_handeye2/calibrations/<name>.calib \
    --color_T_depth <TX TY TZ QX QY QZ QW> --out calib/camera_align/extrinsic_YYYYMMDD_handeye.npy
```

Never train or deploy on the hand-eye result directly: it is a starting point.

## 1. Refine — multi-pose arm ICP (yaw + in-plane translation)

Camera host publishing depth, bridge up, arm moved by hand/teleop to 3–5
*different* configurations in view (spread out, elbow and wrist varied, arm not
parked over the table):

```bash
python tools/calib/calibrate_extrinsic_live_icp.py capture --session logs/calib_live \
    --ip <NUC_IP> --depth_zmq_addr tcp://<CAM_HOST>:5562            # once per pose
python tools/calib/calibrate_extrinsic_live_icp.py solve --session logs/calib_live \
    --init calib/camera_align/extrinsic_YYYYMMDD_handeye.npy --min_z 0.44 \
    --out calib/camera_align/extrinsic_YYYYMMDD_icp.npy
```

- `capture` never commands the arm; it refuses a pose if the arm moved during the
  frames or the joints are identical to a previous pose (stale bridge).
- `--min_z` drops table points next to the arm (they have nothing to match in an
  arm-only reference and pull the fit).
- Inspect `logs/calib_live_icp/real_solved.ply` vs `sim_target.ply`; the printed
  residual should drop and the table tilt should be small.

## 2. Level to the table (height + two tilts)

```bash
python tools/calib/level_extrinsic_to_table.py --init calib/camera_align/extrinsic_YYYYMMDD_icp.npy \
    --session logs/calib_live --out calib/camera_align/extrinsic_YYYYMMDD_tablelevel.npy
```

The table constrains exactly the three DOF the arm constrains least. Only valid
on a bare table top (a mat's thickness becomes a height error); points near the
arm are excluded by FK automatically.

## 3. Verify what the policy will see

```bash
python tools/calib/capture_multiframe_zmq.py --addr tcp://<CAM_HOST>:5562 --n_frames 10 --out_dir logs/calib_real
python tools/calib/gen_sim_frame_ply.py --pkl <demo@0.pkl> --frame 0 --out logs/calib_sim_frame.ply
python tools/calib/viz_cropped_overlay.py --real_npz logs/calib_real/accum.npz \
    --extrinsic calib/camera_align/extrinsic_YYYYMMDD_tablelevel.npy \
    --sim_link0_ply logs/calib_sim_frame.ply --port 8081
```

Put the real arm at the same demo frame first
(`deploy/move_to_frame_polymetis.py --frame 0 --hold`). Pass: arm silhouettes
coincide, the table is gone from both clouds, the object sits where sim puts it.
Edges consistently inside/outside the mesh = translation; divergence along the
arm = rotation. For a manual nudge use `tools/calib/live_calibrate_extrinsic.py`
(viser sliders) on the same capture.

## 4. Adopt it

```bash
python tutorial/06_camera_calibration/inspect_extrinsic.py   # validity + drift between all files
```

- Write every calibration to a new dated file; `handeye_to_extrinsic.py`,
  `calibrate_extrinsic_live_icp.py` and `level_extrinsic_to_table.py` refuse
  `current.npy` as `--out` (the viser slider tool does not check, so give it a
  dated `--out` yourself). Switch deliberately (copy to `current.npy` or pass the new file explicitly).
- Pass `--camera_extrinsic <file>` to training, eval, play **and** deploy;
  `deploy_pc.py` requires it.
- Record the drift from the previous file. More than ~1 cm / ~1° is a different
  viewpoint for the policy: re-evaluate in sim with the new file, retrain if it drops.
- If the real table sits a few mm higher than sim in the cloud, raise the crop
  floor at deploy (`--pc_workspace_min x,y,z`) rather than editing the extrinsic.
