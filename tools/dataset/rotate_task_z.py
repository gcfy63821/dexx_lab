#!/usr/bin/env python3
"""
Apply a global rotation (default 90° around world z-axis) to every sequence
in a task folder. Rotates:

  - tool_object_pose   (in `original_data`; supports (T,4,4) or (T,7) quat+pos)
  - hand joint positions for left/right (wrist + 20 finger joints)
  - wrist_translation
  - wrist_rotation     (if present; matrix or xyzw quat)
  - mano_pose[:, :3]   (global orientation; only with --update_mano_pose)
  - aux_object.json    (aux_pos_world + aux_quat_wxyz_world)

Frame conventions
-----------------
The pkl stores positions/rotations in the **camera frame** (z-down). The
viser visualizers display them after `flip_z = diag(1, -1, -1)`, so what
you SEE is z-up.

This script interprets the rotation axis/pivot in the **z-up display frame**
(i.e. what you see in the visualizer). Internally it conjugates by flip_z
to apply the equivalent transform to the stored camera-frame data:

    R_storage_eq = flip_z @ R_world @ flip_z
    p_storage_new = R_storage_eq @ p_storage + flip_z @ (pivot - R_world @ pivot)
    R_storage_new = R_storage_eq @ R_storage           (for rotation fields)

Usage
-----
    # The common case: rotate every seq in a task folder 90° about world z
    python tools/dataset/rotate_task_z.py \\
        --task_dir data/robotool_batch/0423_dual

    # Custom angle / axis / pivot
    python tools/dataset/rotate_task_z.py \\
        --task_dir data/robotool_batch/0423_dual \\
        --angle_deg -90 --axis z --pivot 0.5 0 0

    # Single seq, also update mano_pose global axis-angle, dry-run first
    python tools/dataset/rotate_task_z.py \\
        --seq data/robotool_batch/0423_dual/blue_cup_1 \\
        --update_mano_pose --dry_run

    # Restore originals from .rotbak (undo)
    python tools/dataset/rotate_task_z.py \\
        --task_dir data/robotool_batch/0423_dual --restore
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot


PKL_NAMES = [
    "mano_joints.pkl",
    "mano_joints_optimized.pkl",
    "mano_joints_corrected.pkl",
]
AUX_SIDECAR = "aux_object.json"

JOINT_NAMES = [
    "wrist",
    "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]
POSITION_FIELDS = JOINT_NAMES + ["wrist_translation"]
ROTATION_FIELDS = ["wrist_rotation"]   # (T,3,3) or (T,4) xyzw

AXIS_VEC = {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}

FLIP_Z = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

BACKUP_SUFFIX = ".rotbak"


# --------------------------- core math ---------------------------------

def build_world_rotation(angle_deg: float, axis_letter: str) -> np.ndarray:
    """Rotation matrix for `angle_deg` about world `axis_letter` (z-up frame)."""
    axis = AXIS_VEC[axis_letter]
    rv = np.deg2rad(angle_deg) * axis
    return Rot.from_rotvec(rv).as_matrix()


def storage_transforms(R_world: np.ndarray, pivot_world: np.ndarray):
    """Precompute the equivalent (R, t) operation in storage (camera) frame.

    World-frame op:  p_world_new = R_world @ p_world + d_world,
                     d_world = pivot - R_world @ pivot

    Storage-frame op:  p_storage_new = R_eq @ p_storage + t_eq
                     R_eq = flip_z @ R_world @ flip_z
                     t_eq = flip_z @ d_world
    """
    d_world = pivot_world - R_world @ pivot_world
    R_eq = FLIP_Z @ R_world @ FLIP_Z
    t_eq = FLIP_Z @ d_world
    return R_eq, t_eq


def transform_positions(arr: np.ndarray, R_eq: np.ndarray, t_eq: np.ndarray) -> np.ndarray:
    """Rotate a (T, 3) array of camera-frame positions."""
    out = arr.astype(np.float64) @ R_eq.T + t_eq
    return out


def transform_rotation_matrices(R_arr: np.ndarray, R_eq: np.ndarray) -> np.ndarray:
    """Rotate a (T, 3, 3) array of camera-frame rotation matrices."""
    # R_new = R_eq @ R_old; broadcast via einsum
    return np.einsum("ij,tjk->tik", R_eq, R_arr.astype(np.float64))


def transform_rotvecs(rv_arr: np.ndarray, R_eq: np.ndarray) -> np.ndarray:
    """Rotate a (T, 3) array of camera-frame axis-angle vectors."""
    R_old = Rot.from_rotvec(rv_arr).as_matrix()      # (T,3,3)
    R_new = transform_rotation_matrices(R_old, R_eq)
    return Rot.from_matrix(R_new).as_rotvec()


# --------------------------- per-file rotators -------------------------

def rotate_pkl(pkl_path: str,
                R_eq: np.ndarray, t_eq: np.ndarray,
                update_mano_pose: bool = False,
                dry_run: bool = False) -> Optional[Dict[str, int]]:
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    summary = {"obj_frames": 0, "hand_pos_fields": 0,
               "hand_rot_fields": 0, "mano_pose": 0}

    # --- tool_object_pose ---
    od = data.get("original_data", {})
    tp = od.get("tool_object_pose")
    if tp is not None:
        tp_arr = np.asarray(tp).astype(np.float64).copy()
        if tp_arr.ndim == 3 and tp_arr.shape[1:] == (4, 4):
            R_old = tp_arr[:, :3, :3]
            t_old = tp_arr[:, :3, 3]
            tp_arr[:, :3, :3] = transform_rotation_matrices(R_old, R_eq)
            tp_arr[:, :3, 3]  = transform_positions(t_old, R_eq, t_eq)
            summary["obj_frames"] = tp_arr.shape[0]
            od["tool_object_pose"] = tp_arr
        elif tp_arr.ndim == 2 and tp_arr.shape[1] == 7:
            q_old = tp_arr[:, :4]                       # xyzw assumed
            p_old = tp_arr[:, 4:7]
            R_old = Rot.from_quat(q_old).as_matrix()    # (T,3,3)
            R_new = transform_rotation_matrices(R_old, R_eq)
            tp_arr[:, :4]  = Rot.from_matrix(R_new).as_quat()
            tp_arr[:, 4:7] = transform_positions(p_old, R_eq, t_eq)
            summary["obj_frames"] = tp_arr.shape[0]
            od["tool_object_pose"] = tp_arr
        else:
            print(f"  [warn] unexpected tool_object_pose shape {tp_arr.shape}; skipping")
        data["original_data"] = od

    # --- hand fields ---
    for side in ("left", "right"):
        if side not in data or not isinstance(data[side], dict):
            continue
        sd = data[side]

        for field in POSITION_FIELDS:
            if field in sd:
                arr = np.asarray(sd[field])
                if arr.ndim == 2 and arr.shape[-1] == 3 and len(arr) > 0:
                    sd[field] = transform_positions(arr, R_eq, t_eq).astype(np.float32)
                    summary["hand_pos_fields"] += 1

        for field in ROTATION_FIELDS:
            if field in sd:
                arr = np.asarray(sd[field])
                if arr.ndim == 3 and arr.shape[1:] == (3, 3) and len(arr) > 0:
                    sd[field] = transform_rotation_matrices(arr, R_eq).astype(np.float32)
                    summary["hand_rot_fields"] += 1
                elif arr.ndim == 2 and arr.shape[-1] == 4 and len(arr) > 0:
                    R_old = Rot.from_quat(arr).as_matrix()
                    R_new = transform_rotation_matrices(R_old, R_eq)
                    sd[field] = Rot.from_matrix(R_new).as_quat().astype(np.float32)
                    summary["hand_rot_fields"] += 1

        if update_mano_pose and "mano_pose" in sd:
            mp = np.asarray(sd["mano_pose"]).astype(np.float64).copy()
            if mp.ndim == 2 and mp.shape[1] >= 3 and len(mp) > 0:
                mp[:, :3] = transform_rotvecs(mp[:, :3], R_eq)
                sd["mano_pose"] = mp.astype(np.float32)
                summary["mano_pose"] += 1

        data[side] = sd

    if dry_run:
        return summary

    bak = pkl_path + BACKUP_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(pkl_path, bak)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)
    return summary


def rotate_aux(aux_path: str,
                R_world: np.ndarray, pivot_world: np.ndarray,
                dry_run: bool = False) -> bool:
    """aux_object.json stores world-frame pose directly (z-up). No flip_z needed."""
    if not os.path.exists(aux_path):
        return False
    with open(aux_path, "r") as f:
        aux = json.load(f)

    pos = np.asarray(aux.get("aux_pos_world", [0, 0, 0]), dtype=np.float64)
    q_wxyz = aux.get("aux_quat_wxyz_world", [1, 0, 0, 0])
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    R_old = Rot.from_quat(q_xyzw).as_matrix()

    R_new = R_world @ R_old
    pos_new = R_world @ (pos - pivot_world) + pivot_world

    q_new_xyzw = Rot.from_matrix(R_new).as_quat()
    aux["aux_pos_world"] = pos_new.tolist()
    aux["aux_quat_wxyz_world"] = [float(q_new_xyzw[3]),
                                    float(q_new_xyzw[0]),
                                    float(q_new_xyzw[1]),
                                    float(q_new_xyzw[2])]

    if dry_run:
        return True

    bak = aux_path + BACKUP_SUFFIX
    if not os.path.exists(bak):
        shutil.copy2(aux_path, bak)
    with open(aux_path, "w") as f:
        json.dump(aux, f, indent=2)
    return True


# --------------------------- restore -----------------------------------

def restore_from_backup(seq_dir: str) -> int:
    """Restore any `*.rotbak` siblings inside seq_dir. Returns # files restored."""
    n = 0
    for name in os.listdir(seq_dir):
        if name.endswith(BACKUP_SUFFIX):
            target = os.path.join(seq_dir, name[:-len(BACKUP_SUFFIX)])
            src = os.path.join(seq_dir, name)
            shutil.copy2(src, target)
            print(f"    restored {target}")
            n += 1
    return n


# --------------------------- driver ------------------------------------

def discover_sequences(task_dir: str) -> List[str]:
    out = []
    for n in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, n)
        if not os.path.isdir(sub):
            continue
        if any(os.path.exists(os.path.join(sub, x)) for x in PKL_NAMES):
            out.append(sub)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--task_dir", help="Process every seq inside this task folder")
    grp.add_argument("--seq", help="Process this single seq dir")

    p.add_argument("--angle_deg", type=float, default=90.0,
                   help="Rotation angle in degrees (default 90)")
    p.add_argument("--axis", choices=["x", "y", "z"], default="z",
                   help="Axis in z-up display frame (default z)")
    p.add_argument("--pivot", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="Rotation pivot in z-up display frame (default world origin)")
    p.add_argument("--skip_corrected", action="store_true",
                   help="Don't touch mano_joints_corrected.pkl (only rotate raw + optimized)")
    p.add_argument("--update_mano_pose", action="store_true",
                   help="Also rotate the first 3 axis-angle components of mano_pose "
                        "(global wrist orientation). Off by default — joint positions "
                        "are what the retarget pipeline uses.")
    p.add_argument("--dry_run", action="store_true",
                   help="Compute and print summary without writing")
    p.add_argument("--restore", action="store_true",
                   help="Restore *.rotbak files to their originals and exit. "
                        "Ignores other rotation args.")
    args = p.parse_args()

    seqs = [args.seq] if args.seq else discover_sequences(args.task_dir)
    if not seqs:
        print(f"[ERROR] no sequences found")
        sys.exit(1)

    # ---- restore branch ----
    if args.restore:
        print(f"[restore] {len(seqs)} seq(s)")
        total = 0
        for s in seqs:
            print(f"  {s}")
            total += restore_from_backup(s)
        print(f"[DONE] restored {total} file(s) from .rotbak")
        return

    # ---- build transforms ----
    R_world = build_world_rotation(args.angle_deg, args.axis)
    pivot_world = np.array(args.pivot, dtype=np.float64)
    R_eq, t_eq = storage_transforms(R_world, pivot_world)

    pkl_targets = list(PKL_NAMES)
    if args.skip_corrected and "mano_joints_corrected.pkl" in pkl_targets:
        pkl_targets.remove("mano_joints_corrected.pkl")

    mode_str = "DRY-RUN" if args.dry_run else "WRITE"
    print(f"[rotate_task_z] {mode_str}")
    print(f"  axis={args.axis}, angle={args.angle_deg}°, pivot={args.pivot} (z-up display frame)")
    print(f"  pkls={pkl_targets}, update_mano_pose={args.update_mano_pose}")
    print(f"  {len(seqs)} sequence(s)\n")

    grand = {"seqs": 0, "pkls": 0, "aux": 0,
             "obj_frames": 0, "hand_pos_fields": 0,
             "hand_rot_fields": 0, "mano_pose": 0}

    for seq_dir in seqs:
        print(f"[seq] {seq_dir}")
        grand["seqs"] += 1
        for name in pkl_targets:
            pkl_path = os.path.join(seq_dir, name)
            s = rotate_pkl(pkl_path, R_eq, t_eq,
                            update_mano_pose=args.update_mano_pose,
                            dry_run=args.dry_run)
            if s is None:
                continue
            grand["pkls"] += 1
            for k in ("obj_frames", "hand_pos_fields",
                       "hand_rot_fields", "mano_pose"):
                grand[k] += s[k]
            print(f"  - {name:32s}  obj={s['obj_frames']:4d}  "
                  f"hand_pos={s['hand_pos_fields']:2d}  "
                  f"hand_rot={s['hand_rot_fields']:2d}  "
                  f"mano_pose={s['mano_pose']:2d}")

        aux_path = os.path.join(seq_dir, AUX_SIDECAR)
        if rotate_aux(aux_path, R_world, pivot_world, dry_run=args.dry_run):
            grand["aux"] += 1
            print(f"  - {AUX_SIDECAR:32s}  rotated")

    suffix = "  (dry run, no files written)" if args.dry_run else ""
    print(f"\n[DONE]{suffix}")
    print(f"  seqs={grand['seqs']}  pkls={grand['pkls']}  aux={grand['aux']}")
    print(f"  total obj frames rotated:  {grand['obj_frames']}")
    print(f"  total hand pos fields:     {grand['hand_pos_fields']}")
    print(f"  total hand rot fields:     {grand['hand_rot_fields']}")
    print(f"  total mano_pose updates:   {grand['mano_pose']}")
    if not args.dry_run:
        print(f"\n  Originals backed up to <name>{BACKUP_SUFFIX} (only on first run).")
        print(f"  Undo with: --restore --task_dir <same dir>")


if __name__ == "__main__":
    main()
