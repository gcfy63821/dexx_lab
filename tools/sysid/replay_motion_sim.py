#!/usr/bin/env python3
"""Replay a canonical motion CSV in Isaac Sim with the same PD + gravity comp
configuration as the training env, and record target/actual for comparison
against the real-robot recording.

Output pkl schema matches `replay_motion_polymetis.py` so `analyze_motion.py` can
diff them directly.

Usage:
    python tools/sysid/replay_motion_sim.py \
        --motion tools/sysid/motions/chirp_sweep.csv \
        --output logs/system_id/motion_replay/chirp_sweep_sim.pkl \
        --headless
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ARM_JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

parser = argparse.ArgumentParser(description="Replay a motion CSV in Isaac Sim.")
parser.add_argument("--motion", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--control_freq_override", type=float, default=None)
parser.add_argument("--record_freq", type=float, default=100.0)
# 480 Hz, not 120. The arm PD here is an EXPLICIT
# torque loop (the implicit actuator is zeroed so gravity comp can be added), and
# at 120 Hz it is unstable for this robot: a chirp tracks cleanly at low
# frequency and then diverges, while at 480 Hz the same motion tracks within
# its target amplitude. Lower this only if you check the result.
parser.add_argument("--physics_freq", type=float, default=480.0)
parser.add_argument("--approach_s", type=float, default=2.0)
parser.add_argument("--arm_kp", type=str, default=None,
                    help="Comma-separated 7 stiffness values overriding the training "
                         "gains, e.g. '1600,1600,1200,800,500,300,150'.")
parser.add_argument("--arm_kd", type=str, default=None,
                    help="Comma-separated 7 damping values. The Polymetis "
                         "realignment candidate is '85,135,110,25,18,10,5' "
                         "(ARM_KD_POLYMETIS_IT2); the training default is "
                         "'145,135,110,100,50,30,15'.")
parser.add_argument("--dry_run", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --- pre-parse motion so we can dry-run without launching Isaac ---
def _load_motion_csv(path: str):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = [list(map(float, row)) for row in r]
    import numpy as _np
    positions = _np.asarray(rows, dtype=_np.float64)
    json_path = Path(path).with_suffix(".json")
    meta = {}
    if json_path.exists():
        meta = json.load(open(json_path))
    if header != ARM_JOINT_NAMES:
        try:
            perm = [header.index(n) for n in ARM_JOINT_NAMES]
            positions = positions[:, perm]
        except ValueError:
            pass
    return positions, meta


_positions_np, _meta = _load_motion_csv(args_cli.motion)
_ctrl_freq = args_cli.control_freq_override or float(_meta.get("control_freq_hz", 30.0))
_ctrl_dt = 1.0 / _ctrl_freq

print(f"[INFO] Motion: {_meta.get('name', 'unknown')}  "
      f"{_positions_np.shape[0]} steps @ {_ctrl_freq} Hz "
      f"({_positions_np.shape[0]*_ctrl_dt:.1f}s)")
if args_cli.dry_run:
    sys.exit(0)


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

# Arm gains: default to what TRAINING uses, so the replay measures the plant the
# trained policy actually meets. --arm_kp / --arm_kd override for evaluating a
# candidate set (e.g. ARM_KD_POLYMETIS_IT2 from the Polymetis realignment).
from dexx import deploy_config as _dcfg  # noqa: E402
try:
    from dexx.robot_constants import ARM_TUNED_KP, ARM_TUNED_KD  # noqa: E402
except ImportError:  # fallback: the critic-horizon cfg re-exports the tuned arm set
    from dexx.tasks.franka_sharpa.franka_sharpa_critic_horizon_cfg import (  # noqa: E402
        ARM_TUNED_KP, ARM_TUNED_KD,
    )

_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def _resolve_gains(flag, default):
    if not flag:
        return dict(default)
    vals = [float(x) for x in flag.split(",")]
    if len(vals) != 7:
        raise SystemExit(f"expected 7 comma-separated values, got {len(vals)}: {flag!r}")
    return dict(zip(_JOINTS, vals))


_ARM_KP = _resolve_gains(args_cli.arm_kp, ARM_TUNED_KP)
_ARM_KD = _resolve_gains(args_cli.arm_kd, ARM_TUNED_KD)
print(f"[replay_sim] arm KP = {[_ARM_KP[j] for j in _JOINTS]}"
      + ("  (override)" if args_cli.arm_kp else "  (training default)"), flush=True)
print(f"[replay_sim] arm KD = {[_ARM_KD[j] for j in _JOINTS]}"
      + ("  (override)" if args_cli.arm_kd else "  (training default)"), flush=True)
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass


def main():
    sim_dt = 1.0 / args_cli.physics_freq
    sim_cfg = SimulationCfg(
        dt=sim_dt,
        device="cuda:0",
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=0),
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        num_envs: int = 1
        env_spacing: float = 1.0

    scene = InteractiveScene(SceneCfg())
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

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
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fr3_joint.*"],
                # Default to the gains TRAINING uses, so the replay answers
                # "what plant does my trained policy actually meet?". Override
                # with --arm_kp / --arm_kd to evaluate a candidate set.
                stiffness=dict(_ARM_KP),
                damping=dict(_ARM_KD),
            ),
            "hand_joints": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"], stiffness=500.0, damping=30.0,
            ),
        },
    )

    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    scene.clone_environments(copy_from_source=False)
    scene.filter_collisions()
    sim.reset()

    device = sim.device
    arm_indices = [robot.joint_names.index(n) for n in ARM_JOINT_NAMES]

    # Gravity-compensation setup (identical to step_response_sim.py)
    arm_actuator = robot.actuators["arm_joints"]
    arm_K = arm_actuator.stiffness.clone()
    arm_D = arm_actuator.damping.clone()
    if arm_K.ndim == 1:
        arm_K = arm_K.unsqueeze(0); arm_D = arm_D.unsqueeze(0)
    arm_actuator.stiffness[:] = 0.0
    arm_actuator.damping[:] = 0.0
    cur_s = robot.root_physx_view.get_dof_stiffnesses()
    cur_d = robot.root_physx_view.get_dof_dampings()
    for idx in arm_actuator.joint_indices.cpu().tolist():
        cur_s[:, idx] = 0.0; cur_d[:, idx] = 0.0
    robot.root_physx_view.set_dof_stiffnesses(cur_s, torch.arange(1))
    robot.root_physx_view.set_dof_dampings(cur_d, torch.arange(1))

    # The PD gains are indexed by the ACTUATOR's joint order; the measured
    # positions are indexed by ARM_JOINT_NAMES' order in robot.joint_names.
    # If those disagree, every gain multiplies the wrong joint's error and the
    # replay diverges silently. Assert rather than discover it in the metrics.
    _act_ids = arm_actuator.joint_indices
    _act_ids = _act_ids.cpu().tolist() if hasattr(_act_ids, "cpu") else list(_act_ids)
    print(f"[replay_sim] arm joint order — by name : {arm_indices}", flush=True)
    print(f"[replay_sim] arm joint order — actuator: {_act_ids}", flush=True)
    print(f"[replay_sim] names: {[robot.joint_names[i] for i in arm_indices]}", flush=True)
    if _act_ids != arm_indices:
        raise RuntimeError(
            f"arm actuator joint order {_act_ids} != name order {arm_indices}. "
            f"The PD gains and the measured positions would be indexed "
            f"differently and the replay would diverge. Reorder the gains to "
            f"match, or index both through the same list.")

    def apply_arm_effort(target_tensor: torch.Tensor):
        gforce = robot.root_physx_view.get_gravity_compensation_forces()
        cforce = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        if not isinstance(gforce, torch.Tensor):
            gforce = torch.as_tensor(gforce, device=device, dtype=torch.float32)
        if not isinstance(cforce, torch.Tensor):
            cforce = torch.as_tensor(cforce, device=device, dtype=torch.float32)
        gforce = gforce.reshape(1, -1); cforce = cforce.reshape(1, -1)
        arm_pos = robot.data.joint_pos[:, arm_indices]
        arm_vel = robot.data.joint_vel[:, arm_indices]
        pos_err = target_tensor - arm_pos
        tau_pd = arm_K * pos_err + arm_D * (-arm_vel)
        arm_phys_ids = arm_actuator.joint_indices
        arm_effort = tau_pd + gforce[:, arm_phys_ids] + cforce[:, arm_phys_ids]
        all_effort = torch.zeros(1, robot.num_joints, device=device)
        all_effort[:, arm_indices] = arm_effort
        robot.set_joint_effort_target(all_effort)

    # --- load motion and ramp to first frame ---
    positions = _positions_np.astype(np.float32)
    N = positions.shape[0]
    home = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=np.float32)

    # set robot home
    all_jp = torch.zeros(1, robot.num_joints, device=device)
    for i, idx in enumerate(arm_indices):
        all_jp[0, idx] = float(home[i])
    robot.write_joint_state_to_sim(all_jp, torch.zeros_like(all_jp))
    all_hand_target = all_jp.clone()

    # approach from home to motion[0]
    n_approach = int(args_cli.approach_s / sim_dt)
    first_target = torch.as_tensor(positions[0], device=device, dtype=torch.float32).unsqueeze(0)
    home_t = torch.as_tensor(home, device=device, dtype=torch.float32).unsqueeze(0)
    print(f"[INFO] Approaching motion start over {args_cli.approach_s:.1f}s...")
    for i in range(n_approach):
        alpha = (i + 1) / n_approach
        interp = (1 - alpha) * home_t + alpha * first_target
        apply_arm_effort(interp)
        robot.set_joint_position_target(all_hand_target)
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(sim_dt)

    # --- main replay ---
    rec_timestamps, rec_targets, rec_actuals, rec_vels = [], [], [], []
    rec_dt = 1.0 / args_cli.record_freq
    next_rec_t = 0.0

    total_duration = N * _ctrl_dt
    n_phys_steps = int(total_duration / sim_dt)
    print(f"[INFO] Replaying {N} steps ({total_duration:.2f}s) at physics "
          f"dt={sim_dt:.4f}s ({n_phys_steps} phys steps)")

    cur_target_np = positions[0]
    cur_target_t = first_target
    for phys_step in range(n_phys_steps):
        t = phys_step * sim_dt
        idx = min(int(t / _ctrl_dt), N - 1)
        if not np.array_equal(positions[idx], cur_target_np):
            cur_target_np = positions[idx]
            cur_target_t = torch.as_tensor(cur_target_np, device=device,
                                           dtype=torch.float32).unsqueeze(0)

        apply_arm_effort(cur_target_t)
        robot.set_joint_position_target(all_hand_target)
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(sim_dt)

        # record at record_freq
        if t >= next_rec_t:
            next_rec_t += rec_dt
            rec_timestamps.append(t)
            rec_targets.append(cur_target_np.copy())
            act = robot.data.joint_pos[0, arm_indices].detach().cpu().numpy().astype(np.float64)
            vel = robot.data.joint_vel[0, arm_indices].detach().cpu().numpy().astype(np.float64)
            rec_actuals.append(act)
            rec_vels.append(vel)

    print(f"[INFO] Recorded {len(rec_timestamps)} frames "
          f"({rec_timestamps[-1]:.2f}s).")

    out_path = Path(args_cli.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "side": "sim",
        "motion_meta": _meta,
        "control_freq_hz": _ctrl_freq,
        "record_freq_hz": args_cli.record_freq,
        "physics_freq_hz": args_cli.physics_freq,
        "joint_names": ARM_JOINT_NAMES,
        "timestamps": np.asarray(rec_timestamps),
        "targets": np.asarray(rec_targets),
        "actuals": np.asarray(rec_actuals),
        "velocities": np.asarray(rec_vels),
        "sim_config": {
            "arm_K": arm_K[0].cpu().numpy().tolist(),
            "arm_D": arm_D[0].cpu().numpy().tolist(),
            "gravity_comp": True,
        },
    }
    # A diverged replay produces confident, meaningless metrics downstream.
    # Refuse to save one rather than let it reach analyze_motion.py.
    _err = np.abs(np.asarray(rec_actuals) - np.asarray(rec_targets))
    _max_err = float(_err.max()) if _err.size else 0.0
    if _max_err > 1.0:
        raise RuntimeError(
            f"replay diverged: max |actual - target| = {_max_err:.1f} rad against a "
            f"target range of ~0.12 rad. The PD torque loop went unstable — raise "
            f"--physics_freq (480 is the default for this reason) or lower the "
            f"stiffness. Not saving, because the metrics would look plausible.")
    print(f"[replay_sim] max tracking error {_max_err:.4f} rad — stable")

    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[OK] Saved to {out_path}")

    # Release the simulation context first; otherwise simulation_app.close() hangs.
    _sim = sim_utils.SimulationContext.instance()
    if _sim is not None:
        _sim.clear_all_callbacks()
        _sim.clear_instance()
    simulation_app.close()


if __name__ == "__main__":
    main()
