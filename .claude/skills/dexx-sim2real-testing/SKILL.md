---
name: dexx-sim2real-testing
description: Use when checking or closing the sim-to-real gap for this repo's Franka FR3 + Sharpa hand policies — layered tests from static frames and joint order, through arm/hand dynamics system-ID (chirp/step replay sim vs real), point-cloud and observation parity, to robustness sweeps in sim and staged closed-loop runs on the robot.
---

# Sim2real testing

Principle: **isolate one layer at a time, cheapest first, and measure before you
change anything.** A policy failure on hardware is usually geometry, dynamics or
perception, and the three look alike from the outside. Retrain only after a
measurement says which one it is.

## Layer 0 — static geometry (no robot motion)

| check | how | pass |
|---|---|---|
| frame constants are consistent | `python tutorial/01_frames_and_constants/check_frames.py` | all `[ok]` |
| constants match the physical setup | tape-measure arm base height vs table; compare with `ARM_BASE_Z`, `TABLE_SURFACE_Z` in `src/dexx/deploy_config.py` | within a few mm |
| robot asset = what training used | `python scripts/check_asset_equivalence.py --reference_usd <usd> --side right` | frames match |
| hand joint order | command single joints on the real hand through the SDK; compare with docs/JOINT_ORDERING.md | the named finger moves |
| extrinsic sane | `python tutorial/06_camera_calibration/inspect_extrinsic.py` | valid 4×4, camera where it physically is |

## Layer 1 — static observation parity (robot held still)

Put the real arm at a demo frame and compare what the policy would see with sim
at the same frame:

```bash
python deploy/move_to_frame_polymetis.py --ip <NUC_IP> \
    --pkl data/retargeting/robotool_batch/mano2sharpa_rh/<task>/<seq>@0.pkl --frame 0 --hold
python deploy/test_polymetis_arm.py --ip <NUC_IP>     # joints + flange + wrist (right_hand_C_MC)
```

- Joint readings should match the demo frame to within the impedance
  steady-state error (fractions of a degree on the proximal joints).
- The wrist the deploy uses is FK of these joints on the sim URDF — so a wrong
  wrist means wrong joints, wrong `ARM_BASE_POS`, or a wrong URDF.
- Point clouds: overlay the real *cropped* cloud on the sim *cropped* cloud
  (`dexx-camera-calibration`, verify step). The object and hand must land on
  their sim counterparts; the table must be gone from both.

## Layer 2 — dynamics (system-ID)

Replay the same motion file in sim and on the robot and diff them
(tutorial/07, `tools/sysid/motions/README.md` for the format):

```bash
python tools/sysid/replay_motion_sim.py --motion tools/sysid/motions/chirp_sweep.csv \
    --output logs/sysid/chirp_sim.pkl --headless
python tools/sysid/replay_motion_polymetis.py --ip <NUC_IP> \
    --motion tools/sysid/motions/chirp_sweep.csv --output logs/sysid/chirp_real.pkl
# ROS2 backend: tools/sysid/replay_motion_ros2.py (same pkl schema)
python tools/sysid/analyze_motion.py --sim logs/sysid/chirp_sim.pkl \
    --real logs/sysid/chirp_real.pkl --out logs/sysid/chirp_cmp
```

- Start with `step_per_joint.csv`: stiffness sets rise time, damping sets
  overshoot. Match rise time first, then overshoot, then latency
  (`action_delay_max` covers the worst measured delay, not the mean), then the
  command EMA. Latency measured against badly tuned gains measures the gains.
- Iterate candidate gains in sim only (`--arm_kp/--arm_kd`) against one real
  recording; the robot does not need to move again.
- Fit the simulator to the robot, never the other way round, and re-measure after
  any controller change (gains, filter, Polymetis version).
- Hand: `tools/sysid/replay_hand_motion_sim.py` / `replay_hand_motion_real.py`
  with `tools/sysid/motions_hand/*.csv`. Fingers that stop short of contact in sim
  but not on the robot (or vice versa) point at armature or PD gains.
- Keep the arm replay's physics rate high (`replay_motion_sim.py` defaults to
  480 Hz): with the explicit arm PD, a low rate lets integration error dominate
  the recorded response.

## Layer 3 — robustness in simulation (before the robot sees a new checkpoint)

Stress the student with what hardware will do to it; a checkpoint that collapses
here will collapse on the robot:

```bash
python scripts/eval.py --load_path <ckpt> --side right --data_idx '[...]' \
    --camera_extrinsic calib/camera_align/current.npy --out_dir logs/eval_stress --headless \
    --inject_jitter 0.003 --inject_dropout 0.1 --inject_hand_noise 0.003 --perturb_obj_xy 0.02
```

- Physics randomisation is on by default in `eval.py`; compare against
  `--no-keep_physics_dr` to see how much of the score depends on nominal mass and
  friction.
- Modality ablations (`--no_tactile`, `--no_contact_force`, `--pc_ablate_*`) tell
  you what the student leans on — useful when one sensor is flaky on hardware.
- Evaluate with the extrinsic you will deploy with. Evaluating with another
  calibration of the same mount predicts the cost of a stale calibration.

## Layer 4 — staged closed loop on the robot

1. Arm-only motion without the policy (`move_to_frame_polymetis.py`, sysid replays)
   — confirms the arm path and e-stops.
2. Policy with the object absent, hand clear of the table — confirms the loop runs
   at ~30 Hz without stale-sensor e-stops.
3. Policy on the easiest demo, object in place, action ramp on, e-stop in hand.
4. Then harder demos and object placements.

After every rollout, compare `logs/deploy_debug/*.npz` with the same demo in
simulation: arm/hand targets vs measured, wrist trajectory vs demo, tactile and
contact force, loop rate (`loop_hz`). The first divergence in time is the layer to
fix.

## Anti-patterns

- Tuning the policy (reward, recipe) to compensate a plant mismatch.
- Changing two things between robot runs.
- Trusting a success rate from fewer than ~30 real trials; report trials and
  conditions with it.
- Deploying a checkpoint whose training extrinsic, arm mount or gains you cannot
  name.
