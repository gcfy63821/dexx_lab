---
name: dexx-real-robot-deploy
description: Use when bringing up, running, tuning or debugging the real Franka FR3 + Sharpa Wave hand deploy of a point-cloud student — Polymetis NUC bridge, RealSense depth host, deploy_pc.py, ROS2 backend, e-stops, first-run checklist, and what to do when the robot behaves differently from simulation.
---

# Real-robot deploy and tuning

Three machines (docs/DEPLOY.md has every flag):

| machine | runs |
|---|---|
| NUC (real-time kernel) | Polymetis server + `deploy/polymetis_joint_bridge.py` (ZMQ state :5560, cmd :5561) |
| camera host | `deploy/realsense_depth_zmq_pub.py` (D455 640×480, ×2 decimation → 320×240 depth, ZMQ :5562) |
| inference PC (GPU) | `deploy/deploy_pc.py` — policy, Sharpa hand over USB (SDK), arm and depth over the network |

Use a wired link between the NUC and the inference PC for the 30 Hz control
loop.

## Bring-up, in order

1. **Robot.** Desk: FCI active, joints unlocked, the Sharpa hand's end-effector
   load set (mass, COM). Hand E-stop within reach.
2. **NUC.**
   ```bash
   python launch_robot.py robot_client=franka_hardware   # Polymetis server, 1 kHz, needs sudo for RT
   python deploy/polymetis_joint_bridge.py               # copy this file to the NUC; needs pyzmq msgpack msgpack-numpy
   ```
   The bridge rejects non-finite targets, targets outside the FR3 limits and steps
   over `--max_target_step` (0.5 rad).
3. **Camera host.** `python deploy/realsense_depth_zmq_pub.py --bind_port 5562`
4. **Inference PC.**
   ```bash
   export SHARPA_SDK_PYTHON=/path/to/SharpaWaveSDK/python
   python deploy/test_polymetis_arm.py --ip <NUC_IP>        # read-only: joints, flange and wrist update
   python deploy/move_to_frame_polymetis.py --ip <NUC_IP> \
       --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl --frame 0 --hold
   python deploy/deploy_pc.py --load_path checkpoints/student_lean_v6_L1.pth --side right \
       --data_idx '["rt/0416_grasp/cube_small_2"]' --camera_extrinsic calib/camera_align/current.npy \
       --pc_workspace_min 0.0,-0.40,0.422 --polymetis_ip <NUC_IP> --depth_zmq_addr tcp://<CAM_HOST>:5562 --headless
   ```
   At each reset: Enter = straight to the demo start, `y` = via the Franka home
   pose (when a straight joint path would sweep the hand across the table). During
   a rollout `r` resets, `q` quits.

## First-run checklist

The shipped student was trained for the arm mount stated in
`checkpoints/README.md`; check that against your robot before deploying it.

Before trusting any rollout, confirm in this order:

1. `ARM_BASE_Z` / `TABLE_SURFACE_Z` in `deploy_config.py` match the real mount;
   `check_frames.py` passes.
2. The extrinsic passed to `deploy_pc.py` is the file the student was **trained**
   with, and it was calibrated for the current camera and arm mount
   (`dexx-camera-calibration`).
3. Hand joint order verified end to end on the real hand (docs/JOINT_ORDERING.md):
   command one joint, watch the right finger move.
4. `test_polymetis_arm.py` prints a wrist (`right_hand_C_MC`) position that is
   plausible in `fr3_link0`, and joint readings that change when you move the arm.
5. In the deploy log, `[V3] FK→world offset captured` ≈ the arm base position
   (-0.1, 0, 0.415). Anything else shifts the hand point cloud against the scene.
6. The first cropped point cloud contains the object and no table
   (raise the crop floor a few mm with `--pc_workspace_min` if table points get in).
7. Run with the default action ramp (15 steps), object in place, e-stop in hand.

## Parameters that define the plant

- **Polymetis joint impedance:** deploy passes no gains, so Polymetis'
  `default_Kq/default_Kqd` from `robot_client/franka_hardware.yaml` apply
  (`[40, 30, 50, 25, 35, 25, 10]` / `[4, 6, 5, 5, 3, 2, 1]` in stock fairo). Check
  which Polymetis checkout the NUC actually imports (`python -c "import polymetis; print(polymetis.__file__)"`)
  — a second checkout can carry other defaults. Training uses different
  (simulated) gains, fitted so the sim arm responds like the real one; never copy
  them into Polymetis. `--polymetis_kq/--polymetis_kqd` deliberately change the
  real controller; any change requires re-running the sysid comparison.
- **Wrist observation:** both arm clients report `right_hand_C_MC` (sim's end
  effector) by FK of the measured joints on the sim URDF. Polymetis' own
  `get_ee_pose()` is the flange, 135° about the tool axis and ~3.5 cm away — never
  feed it to the policy. The quaternion sign and the centre-of-mass linear
  velocity follow Isaac Lab's conventions (`src/dexx/tasks/hand_imitation/deploy/arm_fk.py`).
- **E-stops:** arm target jump > 0.3 rad, arm joint velocity > 2.5 rad/s, hand
  target > 1.5 rad from measured, any arm/hand/depth input older than 0.25 s,
  tactile channel silent > 1.0 s, any non-finite target. The tactile limit is
  `TACTILE_MAX_AGE_S` in `franka_sharpa_force_critic_horizon_deploy_env_v3.py`;
  the others are fields of `FrankaSharpaEnvCfg`.
- **Action ramp:** `--action_ramp_steps` (default 15) scales the whole action in;
  the hand drifts toward mid-range while it ramps. Use 0 only when you know why.

## ROS2 backend (experimental)

`--arm_backend ros2 --depth_backend ros2`, with a ros2_control joint-impedance
controller on the robot PC (docs/DEPLOY.md "ROS2 backend": gains, hand load via
`deploy/ros2/apply_sharpa_load.py`) and `deploy/ros2/wrist_state_publisher.py`
running on the inference PC before deploy. Same `ROS_DOMAIN_ID` everywhere; check
`ros2 topic hz /joint_states /franka_wrist_state` first.

## When the robot behaves differently from simulation

Diagnose before retraining; each row is a measurement, not a guess:

| symptom | first check |
|---|---|
| arm shakes, sim is smooth | impedance too stiff / command EMA / delay (`dexx-sim2real-testing`, dynamics) |
| arm lags or overshoots | damping, latency |
| reaches to the wrong place, consistently | extrinsic or arm-base height; overlay the cropped clouds |
| hand does something odd from the first step | joint order, hand gains, the action ramp |
| grasp slips | object mass/friction outside the randomised range; hand current coefficient |
| e-stop "stale sensor data" | the named stream: bridge/NUC load, depth host, USB hand |
| e-stop "arm joint delta" right after reset | the arm did not reach the demo start; route via home, check the object isn't blocking |

Every rollout writes `logs/deploy_debug/*.npz` (actions, targets, joint states,
wrist, tactile, full observation). Compare it with the same demo in simulation
before changing anything.
