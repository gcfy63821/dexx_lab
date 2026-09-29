#!/usr/bin/env python3
"""
Drop-test objects in IsaacLab to find their resting position on the table.

For each sequence:
  1. Loads the object mesh and frame-0 rotation
  2. Spawns the object above the table in IsaacLab with correct orientation
  3. Lets it fall under gravity until settled
  4. Records z_bottom_offset = final_z - table_surface_z
  5. Saves metadata in mano_joints_corrected.pkl for the dataloader

Usage:
    # Single sequence (with viewer)
    python tools/dataset/drop_test.py --sequence blue_cup/blue_cup_1

    # All sequences (headless)
    python tools/dataset/drop_test.py --all --headless

    # Specific task
    python tools/dataset/drop_test.py --task blue_cup --headless
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

# Isaac's shutdown ends the process without flushing a block-buffered stdout.
sys.stdout.reconfigure(line_buffering=True)

# ---- IsaacLab AppLauncher (must be before other sim imports) ----
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Drop-test objects to find table resting position"
)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--sequence", type=str, help="e.g. blue_cup/blue_cup_1")
group.add_argument("--task", type=str, help="Process all sequences under a task")
group.add_argument("--all", action="store_true", help="Process all sequences")

parser.add_argument("--data_dir", type=str, default="data/robotool_batch")
parser.add_argument(
    "--drop_height", type=float, default=0.05,
    help="Height above table surface to drop from (m)",
)
parser.add_argument(
    "--settle_time", type=float, default=3.0,
    help="Max seconds to simulate for settling",
)
parser.add_argument(
    "--velocity_threshold", type=float, default=0.01,
    help="Linear velocity threshold to consider settled (m/s)",
)
parser.add_argument("--object_mass", type=float, default=0.05, help="Object mass (kg)")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument(
    "--batch_size", type=int, default=50,
    help="Max objects to simulate in one batch",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Post-launcher imports ----
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg

import isaacsim.core.utils.prims as prim_utils

from dexx import deploy_config as _dcfg

# Table geometry, as in franka_sharpa_env_cfg.py: a 3 cm slab whose top is the
# table surface.
TABLE_THICKNESS = 0.03
TABLE_SURFACE_Z = _dcfg.TABLE_SURFACE_Z
TABLE_POS_Z = TABLE_SURFACE_Z - TABLE_THICKNESS / 2

# Full rotation from raw camera frame to sim frame:
#   dataloader:  rot = Rx(+90°) @ Rz(-90°)
#   retarget:    mujoco2gym_rot = Rz(-90°) @ Rx(+90°)
#   combined:    mujoco2gym_rot @ rot = Rx(180°) = diag(1, -1, -1)
# This flips y and z, converting camera frame (y-down, z-forward) to sim frame.
ROT_RAW_TO_SIM = np.array(
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
)

OBJECT_SPACING = 0.5  # meters between objects along x-axis


def find_sequences(data_dir: str, task: str | None = None):
    """Find all valid sequence directories."""
    sequences = []
    for task_name in sorted(os.listdir(data_dir)):
        task_dir = os.path.join(data_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        if task_name == "models":
            continue
        if task and task_name != task:
            continue
        for seq_name in sorted(os.listdir(task_dir)):
            seq_dir = os.path.join(task_dir, seq_name)
            if os.path.isdir(seq_dir) and os.path.exists(
                os.path.join(seq_dir, "meta.json")
            ):
                sequences.append((f"{task_name}/{seq_name}", seq_dir))
    return sequences


def load_sequence(seq_dir: str):
    """Load data needed for drop test: pkl data, mesh path, frame-0 pose."""
    meta_path = os.path.join(seq_dir, "meta.json")
    if not os.path.exists(meta_path):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Load pkl (prefer optimized > original; skip corrected since we regenerate it)
    pkl_path = os.path.join(seq_dir, "mano_joints.pkl")
    opt_path = os.path.join(seq_dir, "mano_joints_optimized.pkl")
    if os.path.exists(opt_path):
        pkl_path = opt_path
    if not os.path.exists(pkl_path):
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Object mesh
    obj_ids = meta.get("object_ids", [])
    if not obj_ids:
        return None
    obj_id = obj_ids[0]
    # Load from centralized models dir
    data_root = os.path.dirname(os.path.dirname(seq_dir))  # data/robotool_batch
    mesh_path = os.path.join(data_root, "models", obj_id, "cleaned_mesh_10000.obj")
    if not os.path.exists(mesh_path):
        return None

    # Frame-0 pose
    tp = np.array(data["original_data"]["tool_object_pose"])
    if tp.ndim == 3 and tp.shape[1] == 4:
        pose0 = tp[0].astype(np.float64)
    elif tp.ndim == 2 and tp.shape[1] == 7:
        p = tp[0]
        pose0 = np.eye(4, dtype=np.float64)
        pose0[:3, :3] = Rot.from_quat(p[:4]).as_matrix()
        pose0[:3, 3] = p[4:7]
    else:
        return None

    return {
        "data": data,
        "pkl_path": pkl_path,
        "mesh_path": mesh_path,
        "pose0": pose0,
        "seq_dir": seq_dir,
    }


def ensure_urdf(mesh_path: str):
    """Ensure a URDF exists for the mesh (generate if needed).

    Uses the same generate_urdf_from_obj pattern as the existing codebase.
    """
    urdf_path = mesh_path.replace(".obj", ".urdf")
    if os.path.exists(urdf_path):
        return urdf_path

    obj_filename = os.path.basename(mesh_path)
    urdf_content = f"""<?xml version="1.0"?>
<robot name="drop_test_object">
  <link name="base">
    <visual>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0.0 0.0 0.0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.0 0.0 0.0"/>
      <mass value="{args.object_mass}"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
</robot>"""
    with open(urdf_path, "w") as f:
        f.write(urdf_content)
    print(f"  Generated URDF: {urdf_path}")
    return urdf_path


def rotmat_to_quat_wxyz(R: np.ndarray) -> tuple:
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion."""
    q = Rot.from_matrix(R).as_quat()  # scipy returns (x, y, z, w)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def run_batch(entries: list[dict], sim: SimulationContext, batch_idx: int):
    """Spawn a batch of objects, simulate drop, record results."""
    n = len(entries)
    print(f"\n[BATCH {batch_idx}] Spawning {n} objects...")

    # Spawn table wide enough for all objects
    table_width = max(2.0, (n + 1) * OBJECT_SPACING)
    table_center_x = (n - 1) * OBJECT_SPACING / 2
    table_prim = "/World/table"

    table_cfg = sim_utils.CuboidCfg(
        size=(table_width, 2.0, TABLE_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.002,
            rest_offset=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
    )
    table_cfg.func(table_prim, table_cfg, translation=(table_center_x, 0.0, TABLE_POS_Z))

    # Spawn each object above the table
    drop_z = TABLE_SURFACE_Z + args.drop_height
    obj_prim_paths = []

    for i, entry in enumerate(entries):
        urdf_path = ensure_urdf(entry["mesh_path"])
        R_sim = ROT_RAW_TO_SIM @ entry["pose0"][:3, :3]
        quat = rotmat_to_quat_wxyz(R_sim)

        prim_path = f"/World/object_{i}"
        obj_prim_paths.append(prim_path)

        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=False,
            joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.object_mass),
        )
        urdf_cfg.func(
            prim_path,
            urdf_cfg,
            translation=(i * OBJECT_SPACING, 0.0, drop_z),
            orientation=quat,
        )
        print(f"  [{i}] {entry['name']} at x={i * OBJECT_SPACING:.1f}, z={drop_z:.3f}")

    # Reset sim to initialize physics
    sim.reset()
    print(f"[INFO] Simulating for up to {args.settle_time}s...")

    # Create rigid body view for all objects
    # The URDF converter creates /World/object_i/base as the rigid body link
    physics_view = sim.physics_sim_view
    rb_view = physics_view.create_rigid_body_view("/World/object_*/base")

    # Step until all settled or timeout
    max_steps = int(args.settle_time * 120)
    stable_count = 0

    for step in range(max_steps):
        sim.step()

        if step % 60 == 59:  # Check every 0.5s
            velocities = rb_view.get_velocities()  # (N, 6)
            if hasattr(velocities, "cpu"):
                velocities = velocities.cpu().numpy()
            else:
                velocities = np.array(velocities)

            speeds = np.linalg.norm(velocities[:, :3], axis=1)
            n_settled = int(np.sum(speeds < args.velocity_threshold))
            elapsed = (step + 1) / 120
            print(
                f"  {elapsed:.1f}s: {n_settled}/{n} settled "
                f"(max speed: {speeds.max():.4f} m/s)"
            )

            if n_settled == n:
                stable_count += 1
                if stable_count >= 3:  # Stable for 1.5s
                    print(f"  All objects settled at {elapsed:.1f}s")
                    break
            else:
                stable_count = 0

    # Read final transforms
    transforms = rb_view.get_transforms()  # (N, 7): [x, y, z, qx, qy, qz, qw]
    if hasattr(transforms, "cpu"):
        transforms = transforms.cpu().numpy()
    else:
        transforms = np.array(transforms)

    # Print results and save
    print(f"\n{'=' * 60}")
    print(f"{'Sequence':<40} {'final_z':>8} {'z_bottom_offset':>16}")
    print(f"{'=' * 60}")

    for i, entry in enumerate(entries):
        final_z = float(transforms[i, 2])
        z_bottom_offset = final_z - TABLE_SURFACE_Z
        entry["z_bottom_offset"] = z_bottom_offset
        entry["final_z"] = final_z
        print(f"{entry['name']:<40} {final_z:>8.4f} {z_bottom_offset:>16.4f}")

        # Save to corrected pkl
        data = entry["data"]
        original_data = data.get("original_data", {})
        original_data["z_bottom_offset"] = z_bottom_offset
        data["original_data"] = original_data

        out_path = os.path.join(entry["seq_dir"], "mano_joints_corrected.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  -> Saved {out_path}")

    # Clean up prims for next batch
    for prim_path in obj_prim_paths:
        if prim_utils.is_prim_path_valid(prim_path):
            prim_utils.delete_prim(prim_path)
    if prim_utils.is_prim_path_valid(table_prim):
        prim_utils.delete_prim(table_prim)


def main():
    # Collect sequences
    if args.sequence:
        seq_list = [(args.sequence, os.path.join(args.data_dir, args.sequence))]
    elif args.task:
        seq_list = find_sequences(args.data_dir, task=args.task)
    else:
        seq_list = find_sequences(args.data_dir)

    # Load data for each
    entries = []
    for name, seq_dir in seq_list:
        info = load_sequence(seq_dir)
        if info is None:
            print(f"[SKIP] {name}")
            continue
        info["name"] = name
        entries.append(info)

    if not entries:
        print("[ERROR] No valid sequences to process")
        simulation_app.close()
        return

    print(f"[INFO] {len(entries)} sequences to drop-test")

    if args.dry_run:
        for e in entries:
            R_sim = ROT_RAW_TO_SIM @ e["pose0"][:3, :3]
            euler = Rot.from_matrix(R_sim).as_euler("xyz", degrees=True)
            print(f"  {e['name']}: euler(xyz)={euler}")
        simulation_app.close()
        return

    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            enable_ccd=True,
        ),
    )
    sim = SimulationContext(sim_cfg)

    # Ground plane
    ground_cfg = GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Process in batches
    batch_size = args.batch_size
    num_batches = (len(entries) + batch_size - 1) // batch_size

    for b in range(num_batches):
        batch = entries[b * batch_size : (b + 1) * batch_size]
        run_batch(batch, sim, b)

    print(f"\n[DONE] Processed {len(entries)} sequences")
    # Release the simulation context first; otherwise simulation_app.close() hangs.
    _sim = sim_utils.SimulationContext.instance()
    if _sim is not None:
        _sim.clear_all_callbacks()
        _sim.clear_instance()
    simulation_app.close()


if __name__ == "__main__":
    main()
