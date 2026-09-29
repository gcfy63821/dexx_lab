#!/usr/bin/env python3
"""Arm step response test in Isaac Sim.

Mirrors the real test: sends step commands to each joint, records response.
Uses the same arm PD gains as franka_sharpa_env_cfg.py.

Usage:
    python tools/sysid/step_response_sim.py \
        --output logs/system_id/step_response_sim.pkl \
        --step_size 0.1 --hold_time 2.0
"""

import argparse
import os
import pickle

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Arm step response test in sim")
parser.add_argument("--output", type=str, default="logs/system_id/step_response_sim.pkl")
parser.add_argument("--step_size", type=float, default=0.1)
parser.add_argument("--hold_time", type=float, default=2.0)
parser.add_argument("--settle_time", type=float, default=1.0)
parser.add_argument("--joints", type=int, nargs="+", default=list(range(7)))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

# Same safe middle position as real test
SAFE_MIDDLE_POS = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])

from dexx import deploy_config as _dcfg
try:
    from dexx.robot_constants import ARM_TUNED_KP, ARM_TUNED_KD
except ImportError:  # fallback: the critic-horizon cfg re-exports the tuned arm set
    from dexx.tasks.franka_sharpa.franka_sharpa_critic_horizon_cfg import ARM_TUNED_KP, ARM_TUNED_KD


def main():
    # Setup sim
    sim_cfg = SimulationCfg(
        # 480, not 120: the arm PD here is an explicit torque loop (the implicit
        # actuator is zeroed so gravity compensation can be added) and it is
        # unstable at 120 Hz on this robot — see replay_motion_sim.py.
        dt=1.0 / 480.0,
        device="cuda:0",
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8, max_velocity_iteration_count=0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0

    scene = InteractiveScene(SceneCfg())
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # Robot with matched gains (same as franka_sharpa_env_cfg.py)
    # Resolve the robot the same way the training env does, so the replay runs
    # against the asset the policy was trained on (collision filters, armature
    # and gains included) rather than a look-alike.
    from dexx.tasks.franka_sharpa.franka_sharpa_env_cfg import franka_sharpa_robot_usd
    usd_path = franka_sharpa_robot_usd("right")
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.1, angular_damping=0.1,
                max_linear_velocity=1000.0, max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0, max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(_dcfg.ARM_BASE_POS), rot=tuple(_dcfg.ARM_BASE_ROT),
            joint_pos={
                "fr3_joint1": 0.0, "fr3_joint2": 0.0, "fr3_joint3": 0.0,
                "fr3_joint4": -1.57, "fr3_joint5": 0.0, "fr3_joint6": 1.57, "fr3_joint7": 0.0,
            },
        ),
        actuators={
            # Arm: per-joint gains tuned via step response comparison (round 2).
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
                # The training set (dexx.robot_constants.ARM_TUNED_KP/KD).
                stiffness=dict(ARM_TUNED_KP),
                damping=dict(ARM_TUNED_KD),
            ),
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                stiffness=500.0, damping=30.0,
            ),
        },
    )

    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions()
    sim.reset()

    # Find arm joint indices
    arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
    arm_indices = [robot.joint_names.index(n) for n in arm_joint_names]
    device = sim.device
    sim_dt = sim_cfg.dt

    print(f"Arm joint indices: {arm_indices}")
    print(f"Sim dt: {sim_dt}, device: {device}")

    # Setup gravity compensation: store PD gains, zero PhysX drives
    arm_actuator = robot.actuators["arm_joints"]
    arm_K = arm_actuator.stiffness.clone()
    arm_D = arm_actuator.damping.clone()
    if arm_K.ndim == 1:
        arm_K = arm_K.unsqueeze(0)
        arm_D = arm_D.unsqueeze(0)
    # Zero PhysX PD drives — must write to PhysX directly
    arm_actuator.stiffness[:] = 0.0
    arm_actuator.damping[:] = 0.0
    cur_stiffness = robot.root_physx_view.get_dof_stiffnesses()
    cur_damping = robot.root_physx_view.get_dof_dampings()
    arm_phys_ids = arm_actuator.joint_indices.cpu().tolist()
    for idx in arm_phys_ids:
        cur_stiffness[:, idx] = 0.0
        cur_damping[:, idx] = 0.0
    robot.root_physx_view.set_dof_stiffnesses(cur_stiffness, torch.arange(1))
    robot.root_physx_view.set_dof_dampings(cur_damping, torch.arange(1))
    print(f"Gravity compensation enabled. PhysX drives zeroed for arm joints.")
    print(f"Manual PD: K={arm_K[0].cpu().numpy()}, D={arm_D[0].cpu().numpy()}")

    # Debug: print joint ordering info
    print(f"robot.joint_names: {robot.joint_names}")
    print(f"robot.num_joints: {robot.num_joints}")
    print(f"arm_indices (in joint_names): {arm_indices}")

    # Check actuator joint mapping
    arm_act = robot.actuators["arm_joints"]
    print(f"arm_actuator joint_indices: {arm_act.joint_indices}")
    print(f"arm_K shape: {arm_K.shape}, values: {arm_K}")

    _debug_step = [0]

    def apply_arm_effort(target_arm_pos_tensor):
        """Compute PD + gravity + coriolis and apply via effort."""
        gravity_forces = robot.root_physx_view.get_gravity_compensation_forces()
        coriolis_forces = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        if not isinstance(gravity_forces, torch.Tensor):
            gravity_forces = torch.tensor(gravity_forces, device=device, dtype=torch.float32)
        if not isinstance(coriolis_forces, torch.Tensor):
            coriolis_forces = torch.tensor(coriolis_forces, device=device, dtype=torch.float32)
        gravity_forces = gravity_forces.reshape(1, -1)
        coriolis_forces = coriolis_forces.reshape(1, -1)

        arm_pos = robot.data.joint_pos[:, arm_indices]
        arm_vel = robot.data.joint_vel[:, arm_indices]
        pos_err = target_arm_pos_tensor - arm_pos
        tau_pd = arm_K * pos_err + arm_D * (-arm_vel)

        # gravity/coriolis are in PhysX DOF order — use actuator joint_indices for correct mapping
        arm_phys_ids = arm_act.joint_indices
        arm_gravity = gravity_forces[:, arm_phys_ids]
        arm_coriolis = coriolis_forces[:, arm_phys_ids]

        arm_effort = tau_pd + arm_gravity + arm_coriolis

        all_effort = torch.zeros(1, robot.num_joints, device=device)
        all_effort[:, arm_indices] = arm_effort
        robot.set_joint_effort_target(all_effort)

        _debug_step[0] += 1
        if _debug_step[0] <= 3 or _debug_step[0] % 240 == 0:
            print(f"[EFFORT #{_debug_step[0]}]")
            print(f"  gravity shape: {gravity_forces.shape}, arm_phys_ids: {arm_phys_ids}")
            print(f"  pos_err: {pos_err[0].cpu().numpy()}")
            print(f"  tau_pd:  {tau_pd[0].cpu().detach().numpy()}")
            print(f"  gravity: {arm_gravity[0].cpu().numpy()}")
            print(f"  coriolis:{arm_coriolis[0].cpu().numpy()}")
            print(f"  total:   {arm_effort[0].cpu().detach().numpy()}")

    # Set to middle position
    all_jp = torch.zeros(1, robot.num_joints, device=device)
    for i, idx in enumerate(arm_indices):
        all_jp[0, idx] = SAFE_MIDDLE_POS[i]
    all_jv = torch.zeros_like(all_jp)
    robot.write_joint_state_to_sim(all_jp, all_jv)

    # Build target tensor for middle pos
    mid_target = torch.tensor(SAFE_MIDDLE_POS, device=device, dtype=torch.float32).unsqueeze(0)

    # Step to settle with gravity comp
    for _ in range(240):  # 2s
        apply_arm_effort(mid_target)
        robot.set_joint_position_target(all_jp)  # hand position targets
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(sim_dt)

    def record_segment(target_arm_pos, duration_s, label=""):
        """Set target and record for duration with gravity compensation."""
        n_steps = int(duration_s / sim_dt)
        timestamps = []
        targets = []
        actuals = []
        velocities = []

        target_tensor = torch.tensor(target_arm_pos, device=device, dtype=torch.float32).unsqueeze(0)

        for step in range(n_steps):
            apply_arm_effort(target_tensor)
            robot.set_joint_position_target(all_jp)  # hand targets unchanged
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(sim_dt)

            t = step * sim_dt
            timestamps.append(t)
            targets.append(target_arm_pos.copy())
            actual = robot.data.joint_pos[0, arm_indices].cpu().numpy()
            vel = robot.data.joint_vel[0, arm_indices].cpu().numpy()
            actuals.append(actual)
            velocities.append(vel)

        return {
            "timestamps": np.array(timestamps),
            "targets": np.array(targets),
            "actuals": np.array(actuals),
            "velocities": np.array(velocities),
            "label": label,
        }

    all_results = {
        "step_size": args_cli.step_size,
        "hold_time": args_cli.hold_time,
        "record_freq": 1.0 / sim_dt,
        "middle_pos": SAFE_MIDDLE_POS.copy(),
        "sim_dt": sim_dt,
        "joints": {},
    }

    for j in args_cli.joints:
        print(f"\n{'='*50}")
        print(f"Testing joint {j} (fr3_joint{j+1})")
        print(f"{'='*50}")

        base_pos = SAFE_MIDDLE_POS.copy()

        print(f"  Settling...")
        settle_data = record_segment(base_pos, args_cli.settle_time, f"j{j}_settle")

        step_pos = base_pos.copy()
        step_pos[j] += args_cli.step_size
        print(f"  Step UP +{args_cli.step_size} rad...")
        step_up_data = record_segment(step_pos, args_cli.hold_time, f"j{j}_step_up")

        print(f"  Step DOWN...")
        step_down_data = record_segment(base_pos, args_cli.hold_time, f"j{j}_step_down")

        step_neg_pos = base_pos.copy()
        step_neg_pos[j] -= args_cli.step_size
        print(f"  Step NEG...")
        step_neg_data = record_segment(step_neg_pos, args_cli.hold_time, f"j{j}_step_neg")

        print(f"  Return...")
        return_data = record_segment(base_pos, args_cli.settle_time, f"j{j}_return")

        all_results["joints"][j] = {
            "settle": settle_data,
            "step_up": step_up_data,
            "step_down": step_down_data,
            "step_neg": step_neg_data,
            "return": return_data,
        }

        up_actual = step_up_data["actuals"][:, j]
        up_target = step_up_data["targets"][:, j]
        steady_err = np.abs(up_actual[-10:] - up_target[-10:]).mean()
        overshoot = (up_actual.max() - up_target[-1]) / args_cli.step_size * 100 if args_cli.step_size > 0 else 0
        print(f"  Steady-state error: {steady_err:.4f} rad ({np.degrees(steady_err):.2f} deg)")
        print(f"  Overshoot: {overshoot:.1f}%")

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    with open(args_cli.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved to {args_cli.output}")

    # Release the simulation context first; otherwise simulation_app.close() hangs.
    _sim = sim_utils.SimulationContext.instance()
    if _sim is not None:
        _sim.clear_all_callbacks()
        _sim.clear_instance()
    simulation_app.close()


if __name__ == "__main__":
    main()
