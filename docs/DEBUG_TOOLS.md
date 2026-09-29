# Tool index

One line per tool; the linked lesson or doc has the procedure and the flags.

## Demonstration data — [tools/dataset/README.md](../tools/dataset/README.md)

| tool | purpose |
|---|---|
| `tools/dataset/vis_sequence.py` | preview a demonstration (hand keypoints + object) in the browser |
| `tools/dataset/rotate_task_z.py` | rotate a task's demonstrations about a world axis; `--restore` undoes |
| `tools/dataset/view_retarget.py` | the **retargeted** robot (FK) + object + table + MANO targets in env-local, as the env loads them; frame-0 checks (`--summary_only` prints them) |
| `tools/dataset/drop_test.py` | settle the object on the table in sim → `z_bottom_offset` |
| `tools/dataset/adjust_*.py`, `edit_frame_range_pose.py`, `offset_editor.py` | manual hand/object/aux offset and pose fixes |

## Camera calibration — [tutorial/06](../tutorial/06_camera_calibration/)

| tool | purpose |
|---|---|
| `tools/calib/handeye_to_extrinsic.py` | step 0 (new mount): easy_handeye(2) eye-on-base result → initial 4×4 in `fr3_link0`, colour → depth frame |
| `tools/calib/calibrate_extrinsic_live_icp.py` | recommended step 1: `capture` (depth + bridge joints) at several arm poses, `solve` joint ICP against the FK-posed sim arm → 4×4 |
| `tools/calib/level_extrinsic_to_table.py` | recommended step 2: keep yaw/in-plane translation, re-fit height + tilt to the table plane (arm excluded by FK) |
| `deploy/move_to_frame_polymetis.py` | drive the real arm to a retargeted demo frame so sim and real match |
| `tools/calib/capture_multiframe_zmq.py` | accumulate N depth frames from the ZMQ publisher into one camera-frame cloud |
| `tools/calib/gen_sim_frame_ply.py` | URDF FK at that frame → sim robot + table point cloud |
| `tools/calib/icp_align_extrinsic.py` | Kabsch/SVD ICP of the real cloud onto sim → refined 4×4 |
| `tools/calib/live_calibrate_extrinsic.py` | browser (viser) overlay with sliders; saves the 4×4 |
| `tools/calib/viz_cropped_overlay.py` | cropped real vs cropped sim cloud — what the policy sees |
| `tutorial/06_camera_calibration/inspect_extrinsic.py` | validate the shipped 4×4s and report drift between them |

## Dynamics system-ID — [tutorial/07](../tutorial/07_dynamics_alignment/)

| tool | purpose |
|---|---|
| `tools/sysid/motions/`, `motions_hand/` | pre-generated arm / hand motions ([format](../tools/sysid/motions/README.md)) |
| `tools/sysid/generate_motions.py`, `generate_hand_motions.py` | regenerate or add motions, with an FR3 safety check |
| `tools/sysid/replay_motion_sim.py` | sim replay at the training gains; `--arm_kp/--arm_kd` for a candidate |
| `tools/sysid/replay_motion_polymetis.py` | real-arm replay through the Polymetis bridge |
| `tools/sysid/replay_motion_ros2.py`, `step_response_ros2.py` | real-arm replay / step response over ROS2 (`/teleop_joint_commands` → `/joint_states`); same pkl schema |
| `tools/sysid/replay_hand_motion_sim.py`, `replay_hand_motion_real.py` | the hand equivalents (real side via the Sharpa SDK) |
| `tools/sysid/step_response_sim.py`, `analyze_step_response.py` | per-joint step response: rise time, overshoot |
| `tools/sysid/analyze_motion.py` | sim-vs-real per-joint metrics and overlay plots |

## Assets — [assets/ASSETS.md](../assets/ASSETS.md)

| tool | purpose |
|---|---|
| `scripts/build_merged_urdf.py` | merge the vendored FR3 + Sharpa Wave URDFs |
| `scripts/build_robot_usd.py` | convert the merged URDF to the committed robot USD, with self-collision filters |
| `scripts/check_asset_equivalence.py` | merged-URDF link frames vs a reference USD, no simulator |
| `tools/calibrate_elastomer_ids.py` | find the fingertip-elastomer collision-shape indices for friction DR |

## Deploy runtime — [docs/DEPLOY.md](DEPLOY.md)

| tool | runs on | purpose |
|---|---|---|
| `deploy/polymetis_joint_bridge.py` | NUC | Polymetis ↔ ZMQ (state :5560, cmd :5561) |
| `deploy/realsense_depth_zmq_pub.py` | camera host | depth-only ZMQ publisher (D455, 320×240) |
| `deploy/test_polymetis_arm.py` | workstation | sanity-check the bridge and arm state |
| `deploy/ros2/apply_sharpa_load.py` | robot PC (ROS2 backend) | push the Sharpa hand load to FCI (`set_load`) |
| `deploy/ros2/wrist_state_publisher.py` | workstation (ROS2 backend) | `/joint_states` → FK → `/franka_wrist_state` @ 200 Hz; start before deploy |
| `deploy/deploy_pc.py` | workstation | run a lean student on the robot |
| `src/dexx/tasks/hand_imitation/deploy/polymetis_arm_client.py` | workstation | arm client used by the deploy env |
| `src/dexx/scripts/deploy/realsense_depth_zmq_subscriber.py` | workstation | depth subscriber used by the deploy env |
| `src/dexx/tasks/hand_imitation/deploy/ros2_arm_client.py` | workstation | ROS2 arm backend (`--arm_backend ros2`, experimental): wraps `ros2_observation_subscriber.py` + `ros2_action_publisher.py` |
| `src/dexx/scripts/deploy/ros2_depth_subscriber.py` | workstation | ROS2 depth subscriber (`--depth_backend ros2`, experimental) |
| `src/dexx/tasks/franka_sharpa/franka_sharpa_pointcloud_deploy_env.py` | workstation | the deploy env (and its parents) |

Tools that touch hardware also need Polymetis, the RealSense SDK or the Sharpa
SDK on their respective hosts ([MANUAL_SETUP.md](../tutorial/00_setup/MANUAL_SETUP.md)).
