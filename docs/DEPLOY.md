# Real-robot deployment (Polymetis + ZMQ + Sharpa SDK)

> **Hardware-specific.** Requires the Franka FR3 + Sharpa hand, a RealSense depth
> camera on a camera host, a NUC running Polymetis, and the Sharpa SDK on the
> inference PC. The arm runs through a Polymetis joint bridge (ZMQ) and depth
> streams from the camera host over ZMQ. The reference path uses no ROS; an
> experimental ROS2 backend is described [below](#ros2-backend-experimental).

Deploys a **lean DAgger point-cloud student** through the
`franka-sharpa-pointcloud-polymetis-deploy` env, which builds the sim student's
observation dict from hardware: scene points from the ZMQ depth stream,
`hand_pc` / `tactile_pc` from `pytorch_kinematics` FK, `tactile_force` from the
Sharpa SDK F6 sensors, arm joint/EE state from the Polymetis bridge.

`deploy_pc.py` refuses:

- **PPO checkpoints** — only DAgger students are deployed;
- **any student that reads the object-pose tail** — the deploy env publishes 550
  proprio dims (there is no object pose estimator on the robot), and a checkpoint
  whose kept or masked proprio indices reach dim 550 or beyond is rejected at
  startup. In practice that means a lean student, trained with
  `--student_drop_slots obj_bps,tips_distance,obj_pose_tail`
  ([DISTILLATION.md](DISTILLATION.md)).

## Settings for the shipped student

`checkpoints/student_lean_v6_L1.pth` is deployed with:

- extrinsic `calib/camera_align/current.npy`, the one the student was trained with;
- the crop restored from the checkpoint, with the floor raised to 0.422
  (`--pc_workspace_min 0.0,-0.40,0.422`): the real table sits a few mm higher in
  the cloud than in sim;
- **Polymetis default joint-impedance gains** (no `--polymetis_kq/--polymetis_kqd`;
  see [Polymetis parameters](#polymetis-parameters)). Sim training uses
  `ARM_TUNED_KP/KD` (`src/dexx/robot_constants.py`); the two controllers differ,
  so the numbers are not meant to be equal
  ([tutorial/07](../tutorial/07_dynamics_alignment/)).

The arm mount the student expects is listed in
[checkpoints/README.md](../checkpoints/README.md). On a mount other than the
one it was trained on, set `ARM_BASE_Z` to match or retrain
([tutorial/01](../tutorial/01_frames_and_constants/)).

## Architecture

```
Inference PC (Python, ~30Hz)
  ├─ arm: PolymetisArmClient  ──ZMQ cmd :5561──►  polymetis_joint_bridge (NUC)
  │        joint targets                            └► Polymetis server ► FR3
  │        arm/EE state       ◄──ZMQ state :5560──┘  (joint impedance)
  ├─ depth: RealSenseDepthZmqSubscriber ◄──ZMQ :5562── realsense_depth_zmq_pub (camera host)
  └─ hand: Sharpa SDK set_joint_position()  →  hand   (cfg/Sharpa joint order!)
```

All ports and addresses live in `src/dexx/deploy_config.py`
(`POLYMETIS_STATE_PORT`, `POLYMETIS_CMD_PORT`, `CAMERA_ZMQ_ADDR_EXAMPLE`).

## Launch order

**1. NUC** (in the env that has the `polymetis` package):
```bash
# a. Polymetis server (connects the real Franka; FCI activated + joints unlocked)
launch_robot.py robot_client=franka_hardware      # parameters: see "Polymetis parameters" below
# b. joint bridge: Polymetis <-> ZMQ (state :5560, cmd :5561)
python deploy/polymetis_joint_bridge.py           # --robot_ip localhost by default
```

**2. Camera host** (needs `pyrealsense2 pyzmq msgpack msgpack-numpy`):
```bash
python deploy/realsense_depth_zmq_pub.py --bind_port 5562
# depth only: D455 640x480, x2 decimation -> 320x240 fp32 metres (matches the sim intrinsics)
```

**3. Inference PC** (Isaac Lab env + `requirements-deploy.txt`; the Sharpa hand on USB):
```bash
export SHARPA_SDK_PYTHON=/path/to/SharpaWaveSDK/python   # unless `sharpa` is importable
python deploy/test_polymetis_arm.py --ip <NUC_IP>        # bridge sanity check

# (optional) move the arm to a demo frame first
python deploy/move_to_frame_polymetis.py --ip <NUC_IP> \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \
    --frame 0 --hold

python deploy/deploy_pc.py \
    --load_path checkpoints/student_lean_v6_L1.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy \
    --pc_workspace_min 0.0,-0.40,0.422 \
    --polymetis_ip <NUC_IP> --depth_zmq_addr tcp://<CAM_HOST>:5562 --headless
```

Actions are scaled in over the first 15 policy steps (0.5 s, `--action_ramp_steps`,
default 15; `0` turns it off). The whole action is scaled, so while it ramps the
hand targets move toward the middle of their range.

Put the real object where `--data_idx`'s demo starts. At every reset the env
asks `Route via home first? [y/N]`: Enter moves straight to the demo start (the
normal case after a rollout), `y` detours via the Franka home pose when a
straight joint interpolation would sweep the hand across the table, Ctrl+C
aborts. Reset refuses to move the arm if its state is stale. During a rollout,
`r` resets and `q` quits.

## Safety

- **E-stop** (hold position, then reset) when any of these trips; the limits are
  fields of `FrankaSharpaEnvCfg` in `franka_sharpa_env_cfg.py` unless noted:

  | check | limit |
  |---|---|
  | arm target jumps more than | `arm_joint_delta_limit` = 0.3 rad |
  | arm joint velocity above | `arm_joint_vel_limit` = 2.5 rad/s |
  | hand target further from the measured angle than | `hand_joint_delta_limit` = 1.5 rad |
  | newest arm state, hand state (during a rollout) or depth frame older than | `deploy_max_sensor_age_s` = 0.25 s |
  | any tactile channel without a good frame for (during a rollout, tactile on) | `TACTILE_MAX_AGE_S` = 1.0 s (`franka_sharpa_force_critic_horizon_deploy_env_v3.py`) |

  Tactile gets its own, longer limit because the Sharpa tactile stream runs on a
  ~130 ms cycle of its own and a channel legitimately returns nothing on many polls.

- **Startup fails loudly** when the bridge sends no arm state within 10 s, the
  camera sends no depth frame within 5 s, or no Sharpa hand is found.
- Keep the FR3 e-stop in hand.

## Polymetis parameters

The NUC runs Polymetis with the following settings. Polymetis records the
configuration it resolved at every launch in Hydra's
`outputs/<date>/<time>/.hydra/config.yaml`; compare against that file on your NUC.

| parameter | value |
|---|---|
| Polymetis | fairo build (Python 3.8 env), `python launch_robot.py robot_client=franka_hardware`; config from `fairo/polymetis/polymetis/conf/` |
| server | gRPC `:50051`, `hz: 1000`, `use_real_time: true` (needs sudo); FR3 at `robot_ip: 172.16.0.2` (Franka default); FCI activated and joints unlocked in Desk first |
| controller | joint impedance, `robot.start_joint_impedance()` with no gains, engaged by the deploy client at start |
| **gains** | **`default_Kq = [40, 30, 50, 25, 35, 25, 10]`**, **`default_Kqd = [4, 6, 5, 5, 3, 2, 1]`** (`robot_client/franka_hardware.yaml`, `metadata_cfg`; deploy passes no `--polymetis_kq/--polymetis_kqd`) |
| end effector | `panda_link8` (the flange) from `robot_model/franka_panda.yaml`: the pose the bridge publishes. The deploy does **not** use it for the policy (see "Wrist observation" below) |
| client filters and limits | `limit_rate: true`, `lpf_cutoff_frequency: 100`; joint position limits = FR3 limits minus 0.1 rad, joint velocity 2.075 / 2.51 rad/s, torques 86 / 11.5 Nm; collision thresholds 40 Nm / 40 N; safety controller on (margins: joint 0.2 rad, velocity 0.5 rad/s, Cartesian 0.05 m) |
| targets | `update_desired_joint_positions(q)`, 7 joint positions at the policy rate (~30 Hz); the 1 kHz impedance loop tracks them |
| bridge | `deploy/polymetis_joint_bridge.py`: connects to `localhost:50051` (`enforce_version=False`), publishes state at 200 Hz on `:5560`, receives commands on `:5561`; rejects non-finite targets, targets outside the FR3 limits and steps over `--max_target_step` (0.5 rad) |
| state staleness | deploy e-stops if arm state is older than 0.25 s (`deploy_max_sensor_age_s`) |
| e-stop | holds the current joint position as the target; if arm state is stale, sends nothing and Polymetis holds its last target |
| on exit | `terminate_current_policy()` |

Different Polymetis checkouts can carry different client defaults. Check which
checkout `import polymetis` resolves to on your NUC and that its
`robot_client/franka_hardware.yaml` holds the gains above.

### Wrist observation

The policy's wrist observations (`wrist_quat`, `delta_wrist_quat`,
`delta_wrist_pos`, wrist velocities) are defined on sim's end effector,
`right_hand_C_MC` (the Sharpa hand base). Polymetis' own EE pose is the flange,
`panda_link8`: 135° about the tool axis and ~3.5 cm from it. The deploy
therefore computes the wrist itself, by FK of the measured joints on the sim's
URDF (`src/dexx/tasks/hand_imitation/deploy/arm_fk.py`), with velocities
`J(q)·dq` from the measured joint velocities, and ignores the flange pose (kept
as `flange_position` for diagnostics). Feeding the flange pose instead would give
the policy a wrist observation off by that fixed rotation and offset.

Training uses different gains (`ARM_TUNED_KP/KD`, [tutorial/07](../tutorial/07_dynamics_alignment/));
other gains can be passed with `--polymetis_kq/--polymetis_kqd` (give both) to
deliberately change the real controller. Never pass the sim gains there; any
change requires re-running the sysid comparison.

## ROS2 backend (experimental)

The arm and/or the depth camera can go through ROS2 (Humble) instead:
`--arm_backend ros2`, `--depth_backend ros2`. **Not validated with the
point-cloud student**; Polymetis + ZMQ is the reference path. Test it with the
e-stop in hand.

**Robot PC: the arm controller.** A fork of `franka_ros2` with a
`DexhandJointImpedanceController`:
[github.com/gcfy63821/franka_ros2_ws](https://github.com/gcfy63821/franka_ros2_ws),
branch `humble` (reference commit `b81a3dc`). Build it with `colcon build` in
a Humble workspace, set `robot_ip` in `franka_bringup/config/franka_arm.config.yaml`,
then

```bash
ros2 launch franka_bringup dexhand_joint_impedance_controller.launch.py
python deploy/ros2/apply_sharpa_load.py --mass 1.48 --com -0.00163 0.0061 0.04042   # unless Desk set it
```

Controller parameters:

| parameter | value |
|---|---|
| control law | `tau = K (q_target - q) - D dq_f`, `dq_f = 0.01 dq_f + 0.99 dq`; gravity compensated by libfranka |
| `update_rate` / `thread_priority` | 1000 Hz / 98 (real-time kernel) |
| `k_gains` | **`[450, 450, 450, 450, 225, 225, 112.5]`** — edit `franka_bringup/config/controllers.yaml`; the repo ships the `[200, 200, 200, 200, 100, 100, 50]` baseline |
| `d_gains` | **`[45, 45, 45, 45, 22.5, 22.5, 11.25]`** (baseline `[20, 20, 20, 20, 10, 10, 5]`) |
| hand load (FCI) | m = 1.48 kg, COM = (-0.00163, 0.0061, 0.04042) m in the flange frame — a Desk end-effector profile for the Sharpa hand (auto-loaded at launch), or `deploy/ros2/apply_sharpa_load.py` |
| `joint_command_topic` | `/teleop_joint_commands` (`sensor_msgs/JointState`, `position[0..6]` read **by index**; QoS best-effort, depth 1, volatile) |
| `max_joint_distance` | 0.5 rad: a target further than this from the previous one is rejected |
| `command_timeout` | 0.5 s: warns, keeps holding the last target |
| `joint_state_rate` | 30 Hz (`franka_arm.config.yaml`); check `ros2 topic hz /joint_states` |

The 450/45 set is this backend's real-controller gains (K and D both 2.25× the
baseline). Sim gains (`ARM_TUNED_KP/KD`) and real gains are different
controllers and are not meant to be equal; do not copy one into the other. Any
change to the real gains requires re-running the sysid comparison
([tutorial/07](../tutorial/07_dynamics_alignment/)). Check `ros2 param get` / the
YAML on the robot PC before a run.

**Camera: the stock `realsense2_camera` driver**, configured to produce what
`realsense_depth_zmq_pub.py` produces (640×480 depth, ×2 decimation → 320×240,
not aligned to color):

```bash
ros2 launch realsense2_camera rs_launch.py \
    depth_module.depth_profile:=640x480x30 enable_color:=false \
    decimation_filter.enable:=true decimation_filter.filter_magnitude:=2
# older driver versions name the profile depth_module.profile
ros2 topic echo --once /camera/camera/depth/camera_info   # expect 320x240, fx ~ 193
```

The subscriber rejects frames that are not 320×240 instead of resizing them.

**Inference PC** (two terminals, both with `source /opt/ros/humble/setup.bash`
and the same `ROS_DOMAIN_ID` as the robot PC):

```bash
# 1. wrist state: FK of /joint_states on the sim's URDF -> /franka_wrist_state @ 200 Hz
python deploy/ros2/wrist_state_publisher.py --side right --rate 200
ros2 topic hz /franka_wrist_state          # ~200 Hz before going on

# 2. the policy
python deploy/deploy_pc.py \
    --load_path checkpoints/student_lean_v6_L1.pth --side right \
    --data_idx '["rt/0416_grasp/cube_small_2"]' \
    --camera_extrinsic calib/camera_align/current.npy \
    --pc_workspace_min 0.0,-0.40,0.422 \
    --arm_backend ros2 --depth_backend ros2 --headless
```

The comm layer:
`ros2_observation_subscriber.py` (`/joint_states`, `/franka_wrist_state`) and
`ros2_action_publisher.py` (`/teleop_joint_commands`), under
`src/dexx/tasks/hand_imitation/deploy/`, spun on one background executor by
`ros2_arm_client.py`. `--ros2_namespace` prefixes all three topics (empty by
default). What differs from Polymetis:

- **Wrist frame.** `wrist_state_publisher.py` runs FK of the measured joints on
  the sim's URDF to `right_hand_C_MC` and finite-differences the velocities at
  200 Hz: the same frame the Polymetis client computes (see "Wrist
  observation"). The subscriber's PoseStamped fallback (`current_pose`, no
  velocities, the `fr3_hand` body) is refused: deploy waits for
  `/franka_wrist_state` and exits if it does not come.
- **Staleness.** The stale-sensor e-stop (0.25 s) watches both `/joint_states`
  and `/franka_wrist_state`.
- **Gains** are in the controller YAML; `--polymetis_kq/--polymetis_kqd` are
  refused with `--arm_backend ros2`.
- **Stop.** On exit or e-stop the controller holds the last target (there is no
  policy to terminate). Non-finite targets are dropped, not sent as 0 rad.
- **Tools.** `tools/sysid/replay_motion_ros2.py` and `step_response_ros2.py`
  replace the Polymetis replay for gain alignment. The helpers
  `move_to_frame_polymetis.py` and `test_polymetis_arm.py` have no ROS2
  version; the deploy env's reset already ramps the arm to the demo start.

## deploy_pc.py flags

| Flag | Default | Purpose |
|---|---|---|
| `--load_path` | required | Lean DAgger student checkpoint. |
| `--arm_backend` | `polymetis` | `polymetis` (reference) or `ros2` (experimental). |
| `--depth_backend` | `zmq` | `zmq` (reference) or `ros2` (experimental). |
| `--polymetis_ip` | required with `--arm_backend polymetis` | NUC bridge IP. |
| `--depth_zmq_addr` | required with `--depth_backend zmq` | Camera-host depth publisher, e.g. `tcp://<CAM_HOST>:5562`. |
| `--ros2_namespace` | `""` | Prefix of the ROS2 state and command topics. |
| `--depth_topic` | `/camera/camera/depth/image_rect_raw` | ROS2 depth topic (320×240, 16UC1 or 32FC1). |
| `--camera_extrinsic` | required | 4×4 camera-in-armbase `.npy` the student was trained with. |
| `--data_idx` | `None` | Demo whose object and wrist/joint targets the env uses. |
| `--side` | `right` | Hand side. |
| `--pc_workspace_min` / `--pc_workspace_max` | from ckpt | Crop box override, `"x,y,z"` env-local. |
| `--pc_no_crop` | off | Disable the crop to inspect the raw cloud (policy is out of distribution). |
| `--pc_*`, `--no_contact_force`, `--no_tactile` | from ckpt | Deliberate deviations from training (ablations). |
| `--polymetis_state_port` / `--polymetis_cmd_port` | `5560` / `5561` | Bridge ZMQ ports. |
| `--polymetis_kq` / `--polymetis_kqd` | Polymetis defaults | 7 comma-separated joint-impedance gains (give both). |
| `--depth_height` / `--depth_width` | `240` / `320` | Depth resolution. |
| `--action_ramp_steps` | `15` | Ramp the action in over the first N policy steps (0.5 s); `0` = off. |
| `--max_steps` | `10000000` | Stop after this many policy steps. |

## Sharpa joint order on the real hand

The real hand's `set_joint_position()` expects **cfg / Sharpa order**, which is not
the same as the sorted USD order the policy acts in. Sending sorted-order targets
to the hand maps joints to the wrong motors. See [JOINT_ORDERING.md](JOINT_ORDERING.md)
for the exact conversion — this is the single most safety-critical detail in deploy.
