#!/usr/bin/env python3
"""
Manually adjust the MAIN (operated) object's xy/z position via viser sliders,
AFTER `drop_test.py` has already corrected its z to sit on the table.

Use case: drop_test put the object on the table, but its xy is still wherever
the raw camera-frame trajectory put it — that may not be a good position in
the robot workspace. Slide dx/dy/dz to nudge the whole trajectory, optionally
dragging the hand and aux along so relative geometry is preserved.

Reads:
  - mano_joints_corrected.pkl  (preferred; produced by drop_test.py)
    falls back to mano_joints_optimized.pkl or mano_joints.pkl if absent
  - aux_object.json  (sidecar, optional; produced by adjust_hand_and_aux.py)

Writes (bakes offset into existing files):
  - mano_joints_corrected.pkl  ← shifted tool_object_pose (+ hand fields if
                                   "Also shift hand" is checked)
  - aux_object.json            ← shifted aux_pos_world (if "Also shift aux"
                                   is checked AND sidecar exists)
  - mano_joints_corrected.pkl.bak  (first-time backup)
  - aux_object.json.bak            (first-time backup)

Usage:
  # single sequence
  python tools/dataset/adjust_object_offset.py \\
      --pkl data/robotool_batch/0422_multi/peg_hole_1/mano_joints_corrected.pkl

  # whole task
  python tools/dataset/adjust_object_offset.py \\
      --task_dir data/robotool_batch/0422_multi
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as Rot

# ---- Hand skeleton (matches adjust_hand_offset.py) ----

FINGER_CHAINS = {
    "thumb":  ["wrist", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip"],
    "index":  ["wrist", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    "middle": ["wrist", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    "ring":   ["wrist", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    "pinky":  ["wrist", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip"],
}

JOINT_NAMES = [
    "wrist",
    "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]
POSITION_FIELDS = JOINT_NAMES + ["wrist_translation"]

FINGER_COLORS = {
    "thumb":  (255, 100, 100),
    "index":  (100, 255, 100),
    "middle": (100, 100, 255),
    "ring":   (255, 255, 100),
    "pinky":  (255, 100, 255),
}
LEFT_WRIST_COLOR = (100, 149, 237)
RIGHT_WRIST_COLOR = (255, 160, 122)
OBJ_COLOR = (180, 180, 180)
AUX_COLOR = (110, 220, 110)
EDITED_MARK = "✦"
AUX_MARK = "Ⓐ"
AUX_SIDECAR_NAME = "aux_object.json"


# ----------------------------- coord helpers -----------------------------

def flip_z(v: np.ndarray) -> np.ndarray:
    """Flip y,z axes (camera frame z-down ↔ display frame z-up). Involutive."""
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


# ----------------------------- discovery ---------------------------------

def pick_pkl(seq_dir: str) -> Optional[str]:
    """Prefer mano_joints_corrected.pkl > optimized > original."""
    for name in ("mano_joints_corrected.pkl",
                 "mano_joints_optimized.pkl",
                 "mano_joints.pkl"):
        p = os.path.join(seq_dir, name)
        if os.path.exists(p):
            return p
    return None


def discover_sequences(task_dir: str) -> List[Tuple[str, str]]:
    results = []
    for name in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, name)
        if not os.path.isdir(sub):
            continue
        meta = os.path.join(sub, "meta.json")
        pkl = pick_pkl(sub)
        if pkl is not None and os.path.isfile(meta):
            results.append((name, pkl))
    return results


def derive_data_root(any_pkl_path: str) -> str:
    """For pkl at .../data/robotool_batch/{task}/{seq}/*.pkl, return .../data/robotool_batch/."""
    seq_dir = os.path.dirname(os.path.abspath(any_pkl_path))
    return os.path.dirname(os.path.dirname(seq_dir))


def seq_label(name: str, pkl_path: str) -> str:
    """Dropdown label markers."""
    seq_dir = os.path.dirname(pkl_path)
    edited = os.path.exists(os.path.join(seq_dir, "mano_joints_corrected.pkl.bak"))
    has_aux = os.path.exists(os.path.join(seq_dir, AUX_SIDECAR_NAME))
    marks = []
    if edited:
        marks.append(EDITED_MARK)
    if has_aux:
        marks.append(AUX_MARK)
    prefix = "".join(marks) if marks else " "
    return f"{prefix} {name}"


# ----------------------------- loading -----------------------------------

def load_aux_sidecar(seq_dir: str) -> Optional[dict]:
    p = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def load_aux_mesh(data_root: str, aux_id: str):
    p = os.path.join(data_root, "models", aux_id, "cleaned_mesh_10000.obj")
    if not os.path.exists(p):
        return None, None
    m = trimesh.load(p, process=False, force="mesh")
    return m.vertices.copy().astype(np.float32), m.faces.copy()


def load_sequence_data(pkl_path: str) -> dict:
    """Load pkl + meta + main object mesh + aux sidecar (+ aux mesh) for a seq."""
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    meta_path = os.path.join(seq_dir, "meta.json")

    with open(pkl_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    data_root = derive_data_root(pkl_path)

    # Main object mesh
    obj_vertices = None
    obj_faces = None
    obj_id = None
    if meta.get("object_ids"):
        obj_id = meta["object_ids"][0]
        obj_path = os.path.join(data_root, "models", obj_id, "cleaned_mesh_10000.obj")
        if os.path.exists(obj_path):
            mesh = trimesh.load(obj_path, process=False, force="mesh")
            obj_vertices = mesh.vertices.copy().astype(np.float32)
            obj_faces = mesh.faces.copy()
        else:
            print(f"[WARN] Object mesh not found: {obj_path}")

    obj_poses = None
    if obj_vertices is not None and "original_data" in mano_data:
        tp = mano_data["original_data"].get("tool_object_pose")
        if tp is not None:
            obj_poses = np.asarray(tp)

    mano_sides = meta.get("mano_sides", [])
    has_left = "left" in mano_sides and "left" in mano_data and \
        len(np.asarray(mano_data["left"].get("wrist", []))) > 0
    has_right = "right" in mano_sides and "right" in mano_data and \
        len(np.asarray(mano_data["right"].get("wrist", []))) > 0

    # Aux
    aux_info = load_aux_sidecar(seq_dir)
    aux_verts = None
    aux_faces = None
    if aux_info is not None and aux_info.get("aux_obj_id"):
        aux_verts, aux_faces = load_aux_mesh(data_root, aux_info["aux_obj_id"])
        if aux_verts is None:
            print(f"[WARN] aux mesh not found for {aux_info['aux_obj_id']}")

    return {
        "pkl_path": pkl_path,
        "seq_dir": seq_dir,
        "mano_data": mano_data,
        "meta": meta,
        "obj_id": obj_id,
        "obj_vertices": obj_vertices,
        "obj_faces": obj_faces,
        "obj_poses": obj_poses,
        "num_frames": int(meta.get("num_frames", 0)),
        "has_left": has_left,
        "has_right": has_right,
        "aux_info": aux_info,
        "aux_verts": aux_verts,
        "aux_faces": aux_faces,
    }


# ----------------------------- viz primitives ----------------------------

def get_hand_positions_with_offset(hand_data: dict, frame_idx: int,
                                    display_offset: np.ndarray) -> dict:
    """Hand joints at frame_idx, converted to display frame, plus offset."""
    positions = {}
    for name in JOINT_NAMES:
        if name in hand_data:
            arr = np.asarray(hand_data[name])
            if arr.ndim == 2 and frame_idx < len(arr):
                positions[name] = flip_z(arr[frame_idx]) + display_offset
    return positions


def transform_obj_vertices(vertices: np.ndarray, pose) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3]
    elif pose.shape == (7,):
        R = Rot.from_quat(pose[:4]).as_matrix()
        t = pose[4:7]
    else:
        raise ValueError(f"Unexpected object pose shape: {pose.shape}")
    return (R @ vertices.T).T + t


def aux_pose_to_R_t(aux_info: dict):
    pos = np.asarray(aux_info.get("aux_pos_world", [0, 0, 0]), dtype=np.float64)
    q_wxyz = aux_info.get("aux_quat_wxyz_world", [1, 0, 0, 0])
    q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
    R = Rot.from_quat(q_xyzw).as_matrix()
    return R, pos, float(aux_info.get("aux_scale", 1.0))


# ----------------------------- bake offset --------------------------------

_FLIP_Z_MAT = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def bake_offset_into_pkl(pkl_path: str,
                          offset_disp: np.ndarray,
                          rpy_deg: np.ndarray,
                          shift_hand: bool,
                          backup_suffix: str = ".bak") -> Tuple[bool, str]:
    """Apply (rotate around main's frame-0 position by R(roll,pitch,yaw), then
    translate by offset_disp) in z-up visualizer frame to:
       - tool_object_pose (per frame rotation + translation), and optionally
       - all hand POSITION_FIELDS.

    rpy_deg is [roll(X), pitch(Y), yaw(Z)] in degrees, intrinsic XYZ Euler in
    z-up visualizer frame. Stored arrays are in camera (z-down) frame, so the
    visualizer rotation is conjugated:  R_cam = flip_z · R_vis · flip_z. The
    translation (dx, dy, dz) becomes (dx, −dy, −dz)."""
    if not os.path.exists(pkl_path):
        return False, f"pkl not found: {pkl_path}"

    bak = pkl_path + backup_suffix
    if not os.path.exists(bak):
        shutil.copy2(pkl_path, bak)
        backed_up = True
    else:
        backed_up = False

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    off_stored = np.array(
        [offset_disp[0], -offset_disp[1], -offset_disp[2]], dtype=np.float64
    )
    R_vis = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    R_cam = _FLIP_Z_MAT @ R_vis @ _FLIP_Z_MAT
    has_rot = not np.allclose(rpy_deg, 0.0)

    actions = []

    # Pivot in stored (camera) frame = main object's frame-0 translation
    od = data.get("original_data", {})
    tp = od.get("tool_object_pose", None)
    if tp is None:
        return False, "no tool_object_pose in pkl; cannot determine pivot"
    tp_arr = np.asarray(tp).astype(np.float64)

    if tp_arr.ndim == 3 and tp_arr.shape[1:] == (4, 4):
        pivot_stored = tp_arr[0, :3, 3].copy()
        for t in range(tp_arr.shape[0]):
            old_t = tp_arr[t, :3, 3].copy()
            old_R = tp_arr[t, :3, :3].copy()
            new_t = R_cam @ (old_t - pivot_stored) + pivot_stored + off_stored
            new_R = R_cam @ old_R if has_rot else old_R
            tp_arr[t, :3, 3] = new_t
            tp_arr[t, :3, :3] = new_R
        actions.append(
            f"tool_object_pose 4x4 (T={tp_arr.shape[0]}, "
            f"rpy={np.asarray(rpy_deg).tolist()}°, off={offset_disp.tolist()})"
        )
    elif tp_arr.ndim == 2 and tp_arr.shape[1] == 7:
        # [qx, qy, qz, qw, tx, ty, tz]
        pivot_stored = tp_arr[0, 4:7].copy()
        for t in range(tp_arr.shape[0]):
            old_q_xyzw = tp_arr[t, :4]
            old_t = tp_arr[t, 4:7].copy()
            new_t = R_cam @ (old_t - pivot_stored) + pivot_stored + off_stored
            if has_rot:
                old_R = Rot.from_quat(old_q_xyzw).as_matrix()
                new_R = R_cam @ old_R
                new_q_xyzw = Rot.from_matrix(new_R).as_quat()
                tp_arr[t, :4] = new_q_xyzw
            tp_arr[t, 4:7] = new_t
        actions.append(
            f"tool_object_pose 7d (T={tp_arr.shape[0]}, "
            f"rpy={np.asarray(rpy_deg).tolist()}°, off={offset_disp.tolist()})"
        )
    else:
        return False, f"unexpected tool_object_pose shape: {tp_arr.shape}"

    od["tool_object_pose"] = tp_arr
    data["original_data"] = od

    # Hand position fields
    if shift_hand:
        for side in ("left", "right"):
            if side not in data or not isinstance(data[side], dict):
                continue
            n_shifted = 0
            for field in POSITION_FIELDS:
                if field in data[side]:
                    arr = np.asarray(data[side][field]).astype(np.float64)
                    if arr.ndim == 2 and arr.shape[-1] == 3:
                        # p_new = R_cam @ (p_old - pivot) + pivot + off_stored
                        new_arr = (arr - pivot_stored) @ R_cam.T + pivot_stored + off_stored
                        data[side][field] = new_arr.astype(np.float32)
                        n_shifted += 1
            if n_shifted > 0:
                actions.append(f"hand {side} ({n_shifted} fields)")

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    msg = f"saved {pkl_path}: " + ", ".join(actions)
    if backed_up:
        msg += f"  [backup: {bak}]"
    else:
        msg += "  [existing .bak kept]"
    return True, msg


def bake_offset_into_aux_sidecar(seq_dir: str,
                                   offset_disp: np.ndarray,
                                   rpy_deg: np.ndarray,
                                   pivot_vis: np.ndarray,
                                   backup_suffix: str = ".bak") -> Tuple[bool, str]:
    """Rotate aux around `pivot_vis` (visualizer z-up frame) by R(roll,pitch,
    yaw), then translate by `offset_disp`. Both pivot and aux live in the same
    z-up world so no flip is needed."""
    p = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if not os.path.exists(p):
        return False, "no aux sidecar to shift"

    bak = p + backup_suffix
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
        backed_up = True
    else:
        backed_up = False

    with open(p, "r") as f:
        sidecar = json.load(f)

    R_vis = Rot.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    has_rot = not np.allclose(rpy_deg, 0.0)

    # Position
    pos = np.asarray(sidecar.get("aux_pos_world", [0.0, 0.0, 0.0]), dtype=np.float64)
    new_pos = R_vis @ (pos - pivot_vis) + pivot_vis + offset_disp
    sidecar["aux_pos_world"] = [float(new_pos[0]), float(new_pos[1]), float(new_pos[2])]

    # Orientation: left-multiply by R_vis
    if has_rot:
        q_wxyz = sidecar.get("aux_quat_wxyz_world", [1.0, 0.0, 0.0, 0.0])
        q_xyzw = [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
        old_R = Rot.from_quat(q_xyzw).as_matrix()
        new_R = R_vis @ old_R
        new_q_xyzw = Rot.from_matrix(new_R).as_quat()
        sidecar["aux_quat_wxyz_world"] = [
            float(new_q_xyzw[3]),
            float(new_q_xyzw[0]),
            float(new_q_xyzw[1]),
            float(new_q_xyzw[2]),
        ]
        sidecar["aux_euler_xyz_deg"] = (
            Rot.from_matrix(new_R).as_euler("xyz", degrees=True).tolist()
        )

    with open(p, "w") as f:
        json.dump(sidecar, f, indent=2)

    msg = (
        f"saved {p}: aux rpy={np.asarray(rpy_deg).tolist()}°, "
        f"pos += {offset_disp.tolist()} → {sidecar['aux_pos_world']}"
    )
    if backed_up:
        msg += f"  [backup: {bak}]"
    else:
        msg += "  [existing .bak kept]"
    return True, msg


def pivot_vis_from_pkl_data(mano_data: dict) -> np.ndarray:
    """Main object's frame-0 position in z-up visualizer frame (= flip_z of
    camera-frame translation). Used as rotation pivot."""
    od = mano_data.get("original_data", {})
    tp = od.get("tool_object_pose", None)
    if tp is None:
        return np.zeros(3, dtype=np.float64)
    arr = np.asarray(tp)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        t = arr[0, :3, 3].astype(np.float64)
    elif arr.ndim == 2 and arr.shape[1] == 7:
        t = arr[0, 4:7].astype(np.float64)
    else:
        return np.zeros(3, dtype=np.float64)
    return np.array([t[0], -t[1], -t[2]], dtype=np.float64)


# ----------------------------- main --------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adjust main (operated) object xy/z position via viser; "
                    "optionally drag hand and aux along."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl", type=str,
                       help="Path to a single mano_joints*.pkl to edit "
                            "(prefer mano_joints_corrected.pkl)")
    group.add_argument("--task_dir", type=str,
                       help="Path to a task folder containing exp subdirs")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--slider_range", type=float, default=0.5,
                        help="Offset slider range in meters (default 0.5)")
    args = parser.parse_args()

    # ---- Build sequence list ----
    if args.pkl:
        if not os.path.exists(args.pkl):
            print(f"[ERROR] pkl not found: {args.pkl}")
            sys.exit(1)
        name = os.path.basename(os.path.dirname(os.path.abspath(args.pkl)))
        sequences: List[Tuple[str, str]] = [(name, args.pkl)]
    else:
        if not os.path.isdir(args.task_dir):
            print(f"[ERROR] task_dir not found: {args.task_dir}")
            sys.exit(1)
        sequences = discover_sequences(args.task_dir)
        if not sequences:
            print(f"[ERROR] No sequences in {args.task_dir}")
            sys.exit(1)
        print(f"[INFO] {len(sequences)} sequences:")
        for n, p in sequences:
            tags = []
            if os.path.exists(os.path.join(os.path.dirname(p),
                                            "mano_joints_corrected.pkl.bak")):
                tags.append("obj-edited")
            if os.path.exists(os.path.join(os.path.dirname(p), AUX_SIDECAR_NAME)):
                tags.append("has-aux")
            tag_str = " [" + ", ".join(tags) + "]" if tags else ""
            print(f"  - {n}  ({os.path.basename(p)}){tag_str}")

    # ---- Viser ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    # ---- GUI: sequence nav ----
    seq_labels = [seq_label(n, p) for n, p in sequences]
    seq_dropdown = server.gui.add_dropdown("Sequence", options=seq_labels,
                                            initial_value=seq_labels[0])
    prev_btn = server.gui.add_button("◀ Prev")
    next_btn = server.gui.add_button("Next ▶")
    seq_status = server.gui.add_markdown(f"**1 / {len(sequences)}**")

    # ---- GUI: playback / display ----
    server.gui.add_markdown("---")
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    show_object_cb = server.gui.add_checkbox("Show main object", initial_value=True)
    show_aux_cb = server.gui.add_checkbox("Show aux", initial_value=True)
    show_hand_cb = server.gui.add_checkbox("Show hand", initial_value=True)
    show_bones_cb = server.gui.add_checkbox("Show bones", initial_value=True)
    joint_radius = server.gui.add_slider("Joint radius", min=0.001, max=0.015,
                                          step=0.001, initial_value=0.005)

    # ---- GUI: object offset (display frame, z-up world) ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Main Object Transform (display frame)**")
    R_ = args.slider_range
    STEP = 0.002
    ox = server.gui.add_slider("Obj dx (m)", min=-R_, max=R_, step=STEP, initial_value=0.0)
    oy = server.gui.add_slider("Obj dy (m)", min=-R_, max=R_, step=STEP, initial_value=0.0)
    oz = server.gui.add_slider("Obj dz (m)", min=-R_, max=R_, step=STEP, initial_value=0.0)
    # Rotation: fine-grained sliders (0.5° step). Yaw full ±180°, roll/pitch
    # ±45° (typical small-tilt corrections; widen later if needed).
    ROT_STEP = 0.5
    oyaw   = server.gui.add_slider("Obj yaw   (Z, deg)", min=-180, max=180, step=ROT_STEP, initial_value=0.0)
    opitch = server.gui.add_slider("Obj pitch (Y, deg)", min=-45,  max=45,  step=ROT_STEP, initial_value=0.0)
    oroll  = server.gui.add_slider("Obj roll  (X, deg)", min=-45,  max=45,  step=ROT_STEP, initial_value=0.0)
    server.gui.add_markdown(
        "_rotation pivot = main's frame-0 position; intrinsic XYZ Euler in z-up frame; translation applied after rotation_"
    )

    server.gui.add_markdown("Also shift / rotate...")
    also_shift_hand_cb = server.gui.add_checkbox(
        "...hand (keeps grasp aligned)", initial_value=True,
    )
    also_shift_aux_cb = server.gui.add_checkbox(
        "...aux  (translation only — aux never rotates with main)",
        initial_value=True,
    )

    # ---- GUI: save / reset ----
    server.gui.add_markdown("---")
    save_btn = server.gui.add_button("Save (bake offsets, overwrite files)")
    reset_btn = server.gui.add_button("Reset sliders to 0")
    status_md = server.gui.add_markdown("_ready_")

    # ---- State ----
    state = {"idx": 0, "data": None, "suppress_events": False,
             "pivot_vis": np.zeros(3, dtype=np.float64)}
    handles = {}

    def clear_handles(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def current_offset() -> np.ndarray:
        return np.array([ox.value, oy.value, oz.value], dtype=np.float64)

    def current_rpy_deg() -> np.ndarray:
        return np.array([float(oroll.value), float(opitch.value), float(oyaw.value)],
                         dtype=np.float64)

    def current_rotmat_vis() -> np.ndarray:
        return Rot.from_euler("xyz", current_rpy_deg(), degrees=True).as_matrix()

    def apply_transform_vis(pts: np.ndarray,
                             R_vis: np.ndarray,
                             pivot_vis: np.ndarray,
                             offset_disp: np.ndarray) -> np.ndarray:
        """pts: (...,3) array in z-up visualizer frame. Returns rotated+translated."""
        return (pts - pivot_vis) @ R_vis.T + pivot_vis + offset_disp

    # ---- Draw ----
    def draw_hand(side: str, frame_idx: int,
                   R_vis: np.ndarray, pivot_vis: np.ndarray, offset_disp: np.ndarray):
        prefix = f"/{side}_hand"
        clear_handles(prefix)
        d = state["data"]
        present = d["has_left"] if side == "left" else d["has_right"]
        if not present or not show_hand_cb.value:
            return
        hand_data = d["mano_data"].get(side, {})
        # Get positions in z-up frame at zero offset; apply transform after.
        positions = get_hand_positions_with_offset(hand_data, frame_idx,
                                                    np.zeros(3, dtype=np.float64))
        if not positions:
            return

        wrist_color = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        r = joint_radius.value

        transformed = {
            name: apply_transform_vis(p.astype(np.float64), R_vis, pivot_vis, offset_disp)
            for name, p in positions.items()
        }

        for name, pos in transformed.items():
            color = wrist_color if name == "wrist" else (200, 200, 200)
            if name != "wrist":
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key,
                radius=r * 1.5 if name == "wrist" else r,
                position=pos,
                color=color,
            )
        if show_bones_cb.value:
            for finger, chain in FINGER_CHAINS.items():
                color = FINGER_COLORS[finger]
                for i in range(len(chain) - 1):
                    a, b = chain[i], chain[i + 1]
                    if a in transformed and b in transformed:
                        key = f"{prefix}/bone_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key,
                            positions=np.stack([transformed[a], transformed[b]]),
                            color=color,
                            line_width=2.0,
                        )

    def draw_main_object(frame_idx: int,
                          R_vis: np.ndarray, pivot_vis: np.ndarray,
                          offset_disp: np.ndarray):
        prefix = "/object"
        clear_handles(prefix)
        d = state["data"]
        if (not show_object_cb.value or d["obj_poses"] is None
                or d["obj_vertices"] is None):
            return
        if frame_idx >= len(d["obj_poses"]):
            return
        try:
            verts = transform_obj_vertices(d["obj_vertices"], d["obj_poses"][frame_idx])
        except ValueError as e:
            print(f"[WARN] {e}")
            return
        verts = flip_z(verts)  # camera → z-up visualizer
        verts = apply_transform_vis(verts, R_vis, pivot_vis, offset_disp)
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32),
            faces=d["obj_faces"], color=OBJ_COLOR,
        )

    def draw_aux(R_vis: np.ndarray, pivot_vis: np.ndarray, offset_disp: np.ndarray):
        prefix = "/aux"
        clear_handles(prefix)
        d = state["data"]
        if not show_aux_cb.value or d["aux_info"] is None or d["aux_verts"] is None:
            return
        R_aux, pos, scale = aux_pose_to_R_t(d["aux_info"])
        # aux verts in z-up world: R_aux @ (local * scale) + pos
        verts = (R_aux @ (d["aux_verts"] * scale).T).T + pos
        verts = apply_transform_vis(verts, R_vis, pivot_vis, offset_disp)
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32),
            faces=d["aux_faces"], color=AUX_COLOR,
        )

    def redraw():
        if state["data"] is None:
            return
        f = int(frame_slider.value)
        off = current_offset()
        R_vis_full = current_rotmat_vis()
        pivot = state["pivot_vis"]
        I = np.eye(3, dtype=np.float64)
        zero = np.zeros(3, dtype=np.float64)

        # Object always uses full transform
        draw_main_object(f, R_vis_full, pivot, off)

        # Hand uses transform only if "also shift hand" is on
        if also_shift_hand_cb.value:
            draw_hand("left", f, R_vis_full, pivot, off)
            draw_hand("right", f, R_vis_full, pivot, off)
        else:
            draw_hand("left", f, I, pivot, zero)
            draw_hand("right", f, I, pivot, zero)

        # Aux: rotation NEVER applies (aux is a static reference like a slot;
        # rotating it with main would defeat the purpose of correcting main
        # orientation against a fixed target). Translation follows checkbox.
        if also_shift_aux_cb.value:
            draw_aux(I, pivot, off)
        else:
            draw_aux(I, pivot, zero)

    # ---- Sequence switching ----
    def load_idx(idx: int):
        idx = int(np.clip(idx, 0, len(sequences) - 1))
        name, pkl_path = sequences[idx]
        print(f"[LOAD] {idx + 1}/{len(sequences)}: {name}  ({pkl_path})")
        state["data"] = load_sequence_data(pkl_path)
        state["idx"] = idx

        state["suppress_events"] = True
        try:
            nf = state["data"]["num_frames"]
            frame_slider.max = max(nf - 1, 0)
            frame_slider.value = 0
            ox.value = 0.0
            oy.value = 0.0
            oz.value = 0.0
            oyaw.value = 0.0
            opitch.value = 0.0
            oroll.value = 0.0
            state["pivot_vis"] = pivot_vis_from_pkl_data(state["data"]["mano_data"])
            has_obj = (state["data"]["obj_poses"] is not None
                       and state["data"]["obj_vertices"] is not None)
            show_object_cb.value = has_obj
            seq_dropdown.value = seq_labels[idx]
            aux_present = state["data"]["aux_info"] is not None
            seq_status.content = (
                f"**{idx + 1} / {len(sequences)}** — `{name}` ({os.path.basename(pkl_path)})  "
                f"({nf} frames, obj={'yes' if has_obj else 'no'}, "
                f"aux={'yes' if aux_present else 'no'}, "
                f"pivot=[{state['pivot_vis'][0]:+.2f},{state['pivot_vis'][1]:+.2f},{state['pivot_vis'][2]:+.2f}])"
            )
        finally:
            state["suppress_events"] = False

        clear_handles()
        redraw()

    def on_any_update(_event=None):
        if state["suppress_events"]:
            return
        redraw()

    redraw_widgets = [
        frame_slider,
        show_object_cb, show_aux_cb, show_hand_cb, show_bones_cb, joint_radius,
        ox, oy, oz, oyaw, opitch, oroll,
        also_shift_hand_cb, also_shift_aux_cb,
    ]
    for w in redraw_widgets:
        w.on_update(on_any_update)

    @seq_dropdown.on_update
    def _on_dropdown(_event):
        if state["suppress_events"]:
            return
        idx = seq_labels.index(seq_dropdown.value)
        if idx != state["idx"]:
            load_idx(idx)

    @prev_btn.on_click
    def _on_prev(_event):
        if state["idx"] > 0:
            load_idx(state["idx"] - 1)

    @next_btn.on_click
    def _on_next(_event):
        if state["idx"] < len(sequences) - 1:
            load_idx(state["idx"] + 1)

    @save_btn.on_click
    def _on_save(_event):
        d = state["data"]
        off = current_offset()
        rpy = current_rpy_deg()
        if np.allclose(off, 0.0) and np.allclose(rpy, 0.0):
            status_md.content = "_no change: offset and rpy all zero_"
            return

        pivot = state["pivot_vis"]
        lines = []

        # 1) pkl (object trajectory + optionally hand)
        ok, msg = bake_offset_into_pkl(
            d["pkl_path"], off, rpy_deg=rpy, shift_hand=also_shift_hand_cb.value,
        )
        lines.append(("OK  " if ok else "FAIL") + " " + msg)
        print("[SAVE pkl]", msg)

        # 2) aux sidecar (optional, translation only — aux never rotates here)
        if also_shift_aux_cb.value:
            if d["aux_info"] is None:
                lines.append("info  no aux sidecar, skipping aux shift")
            else:
                ok2, msg2 = bake_offset_into_aux_sidecar(
                    d["seq_dir"], off,
                    rpy_deg=np.zeros(3, dtype=np.float64),
                    pivot_vis=pivot,
                )
                lines.append(("OK  " if ok2 else "FAIL") + " " + msg2)
                print("[SAVE aux]", msg2)
        else:
            lines.append("info  Also shift aux unchecked, aux untouched")

        # Refresh label markers
        name, pkl_path = sequences[state["idx"]]
        new_label = seq_label(name, pkl_path)
        if new_label != seq_labels[state["idx"]]:
            seq_labels[state["idx"]] = new_label
            state["suppress_events"] = True
            try:
                seq_dropdown.options = seq_labels
                seq_dropdown.value = new_label
            finally:
                state["suppress_events"] = False

        # Reload so view reflects baked state, then zero sliders
        state["data"] = load_sequence_data(pkl_path)
        state["pivot_vis"] = pivot_vis_from_pkl_data(state["data"]["mano_data"])
        state["suppress_events"] = True
        try:
            ox.value = 0.0
            oy.value = 0.0
            oz.value = 0.0
            oyaw.value = 0.0
            opitch.value = 0.0
            oroll.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "**Saved.**\n\n" + "\n\n".join(lines)

    @reset_btn.on_click
    def _on_reset(_event):
        state["suppress_events"] = True
        try:
            ox.value = 0.0
            oy.value = 0.0
            oz.value = 0.0
            oyaw.value = 0.0
            opitch.value = 0.0
            oroll.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "_sliders reset_"

    # Initial load
    load_idx(0)

    try:
        while True:
            if playing.value and state["data"] is not None:
                nf = state["data"]["num_frames"]
                if nf > 0:
                    frame_slider.value = (int(frame_slider.value) + 1) % nf
                    time.sleep(1.0 / fps_slider.value)
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()
