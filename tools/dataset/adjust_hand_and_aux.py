#!/usr/bin/env python3
"""
Manually adjust hand-object offset AND place a static auxiliary object
via viser sliders.

This is an extension of `adjust_hand_offset.py`:
  - Same hand offset sliders (left / right dx,dy,dz baked into mano_joints.pkl)
  - NEW: pick an aux mesh from data/robotool_batch/models/ and place it in
    the scene with position + yaw/roll/pitch sliders. The chosen pose is
    saved to a sidecar `aux_object.json` next to the pkl (does NOT modify
    mano_joints.pkl). The downstream env loader checks for this sidecar to
    decide whether the demo has a static aux object.

Sidecar format (data/.../{seq}/aux_object.json):
    {
      "aux_obj_id":        "blue_bowl",
      "aux_pos_world":     [x, y, z],            # z-up world frame (display)
      "aux_quat_wxyz_world": [w, x, y, z],
      "aux_euler_xyz_deg": [roll, pitch, yaw],   # human-readable
      "aux_scale":         1.0
    }

Usage (single file):
    python tools/dataset/adjust_hand_and_aux.py \
        --pkl data/robotool_batch/0416_grasp/cube_small_1/mano_joints.pkl \
        --aux_obj_id blue_cup

Usage (task folder):
    python tools/dataset/adjust_hand_and_aux.py \
        --task_dir data/robotool_batch/0416_grasp
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

# ---- Hand skeleton (matches adjust_hand_offset.py / vis_sequence.py) ----

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
AUX_COLOR = (110, 220, 110)         # greenish to distinguish from main object

EDITED_MARK = "✓"
AUX_MARK = "Ⓐ"
AUX_SIDECAR_NAME = "aux_object.json"


# ----------------------------- coord helpers -----------------------------

def flip_z(v: np.ndarray) -> np.ndarray:
    """Flip y,z axes (camera frame z-down ↔ display frame z-up). Involutive."""
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


def euler_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[float]:
    """Intrinsic XYZ euler (degrees) → unit quaternion in wxyz order."""
    q_xyzw = Rot.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True).as_quat()
    return [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


# ----------------------------- discovery ---------------------------------

def discover_sequences(task_dir: str) -> List[Tuple[str, str]]:
    """Return [(seq_name, pkl_path), ...] for every subdir with mano_joints.pkl + meta.json."""
    results = []
    for name in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, name)
        if not os.path.isdir(sub):
            continue
        pkl = os.path.join(sub, "mano_joints.pkl")
        meta = os.path.join(sub, "meta.json")
        if os.path.isfile(pkl) and os.path.isfile(meta):
            results.append((name, pkl))
    return results


def discover_aux_models(data_root: str) -> List[str]:
    """List model dirs under data_root/models/ that contain cleaned_mesh_10000.obj."""
    models_dir = os.path.join(data_root, "models")
    if not os.path.isdir(models_dir):
        return []
    out = []
    for name in sorted(os.listdir(models_dir)):
        sub = os.path.join(models_dir, name)
        mesh = os.path.join(sub, "cleaned_mesh_10000.obj")
        if os.path.isdir(sub) and os.path.exists(mesh):
            out.append(name)
    return out


def derive_data_root(any_pkl_path: str) -> str:
    """For pkl at .../data/robotool_batch/{task}/{seq}/mano_joints.pkl, return .../data/robotool_batch/."""
    seq_dir = os.path.dirname(os.path.abspath(any_pkl_path))
    return os.path.dirname(os.path.dirname(seq_dir))


def seq_label(name: str, pkl_path: str) -> str:
    """Dropdown label; mark sequences with .bak (hand offset edited) and/or aux sidecar."""
    edited = os.path.exists(pkl_path + ".bak")
    has_aux = os.path.exists(os.path.join(os.path.dirname(pkl_path), AUX_SIDECAR_NAME))
    marks = []
    if edited:
        marks.append(EDITED_MARK)
    if has_aux:
        marks.append(AUX_MARK)
    prefix = "".join(marks) if marks else " "
    return f"{prefix} {name}"


# ----------------------------- loading -----------------------------------

def load_sequence_data(pkl_path: str) -> dict:
    """Load pkl + meta + main object mesh for a sequence."""
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    meta_path = os.path.join(seq_dir, "meta.json")

    with open(pkl_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    obj_vertices = None
    obj_faces = None
    if meta.get("object_ids"):
        obj_id = meta["object_ids"][0]
        data_root = derive_data_root(pkl_path)
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

    return {
        "pkl_path": pkl_path,
        "seq_dir": seq_dir,
        "mano_data": mano_data,
        "meta": meta,
        "obj_vertices": obj_vertices,
        "obj_faces": obj_faces,
        "obj_poses": obj_poses,
        "num_frames": int(meta.get("num_frames", 0)),
        "has_left": has_left,
        "has_right": has_right,
    }


def load_aux_sidecar(seq_dir: str) -> Optional[dict]:
    p = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def save_aux_sidecar(seq_dir: str, payload: dict) -> str:
    p = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    return p


def remove_aux_sidecar(seq_dir: str) -> Optional[str]:
    p = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if os.path.exists(p):
        os.remove(p)
        return p
    return None


# ----------------------------- viz primitives ----------------------------

def get_hand_positions_with_offset(hand_data: dict, frame_idx: int,
                                    display_offset: np.ndarray) -> dict:
    """Extract joint positions for a frame, convert to display frame, add offset."""
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


def transform_aux_vertices(verts_local: np.ndarray,
                            pos: np.ndarray, euler_deg: np.ndarray,
                            scale: float) -> np.ndarray:
    """Apply scale → intrinsic XYZ euler rotation → translation, in z-up world frame."""
    R = Rot.from_euler("xyz", euler_deg, degrees=True).as_matrix()
    return (R @ (verts_local * scale).T).T + pos


# ----------------------------- pkl bake ----------------------------------

def apply_offset_to_pkl(pkl_path: str, offsets_display: dict, backup_suffix: str = ".bak"):
    if not os.path.exists(pkl_path):
        return False, f"not found: {pkl_path}"

    bak = pkl_path + backup_suffix
    if not os.path.exists(bak):
        shutil.copy2(pkl_path, bak)
        backed_up = True
    else:
        backed_up = False

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    changed = []
    for side, off_disp in offsets_display.items():
        if side not in data or not isinstance(data[side], dict):
            continue
        if np.allclose(off_disp, 0.0):
            continue
        # display (z-up) → stored (z-down): (dx, -dy, -dz)
        off_stored = np.array([off_disp[0], -off_disp[1], -off_disp[2]], dtype=np.float32)
        for field in POSITION_FIELDS:
            if field in data[side]:
                arr = np.asarray(data[side][field])
                if arr.ndim == 2 and arr.shape[-1] == 3:
                    data[side][field] = (arr + off_stored).astype(arr.dtype)
        changed.append(side)

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    msg = f"saved {pkl_path}"
    if changed:
        msg += f" (offset applied to: {', '.join(changed)})"
    if backed_up:
        msg += f" [backup: {bak}]"
    else:
        msg += " [existing .bak kept]"
    return True, msg


# ----------------------------- main --------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hand offset + aux object placement via viser.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl", type=str,
                       help="Path to a single mano_joints.pkl to edit")
    group.add_argument("--task_dir", type=str,
                       help="Path to a task folder containing exp subdirs")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--also_update_optimized", action="store_true",
                        help="Apply hand offset to mano_joints_optimized.pkl / _corrected.pkl too")
    parser.add_argument("--slider_range", type=float, default=0.15,
                        help="Hand offset slider range in meters (default 0.15)")
    parser.add_argument("--aux_obj_id", type=str, default=None,
                        help="Initial aux model id (subdir under data/robotool_batch/models/). "
                             "If omitted, defaults to first available.")
    parser.add_argument("--aux_slider_range", type=float, default=0.5,
                        help="Aux position slider range in meters (default 0.5)")
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
        print(f"[INFO] Found {len(sequences)} sequences in {args.task_dir}:")
        for n, p in sequences:
            tags = []
            if os.path.exists(p + ".bak"):
                tags.append("hand-edited")
            if os.path.exists(os.path.join(os.path.dirname(p), AUX_SIDECAR_NAME)):
                tags.append("has-aux")
            tag_str = " [" + ", ".join(tags) + "]" if tags else ""
            print(f"  - {n}{tag_str}")

    # ---- Aux model discovery ----
    data_root = derive_data_root(sequences[0][1])
    aux_models = discover_aux_models(data_root)
    if not aux_models:
        print(f"[WARN] No aux models found under {data_root}/models/. "
              f"You can still adjust hand offsets but not place an aux object.")
    initial_aux = args.aux_obj_id or (aux_models[0] if aux_models else None)
    if args.aux_obj_id and args.aux_obj_id not in aux_models:
        print(f"[ERROR] --aux_obj_id {args.aux_obj_id!r} not found. "
              f"Available: {aux_models}")
        sys.exit(1)

    # ---- Viser server ----
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
    show_object_cb = server.gui.add_checkbox("Show Object", initial_value=True)
    show_bones_cb = server.gui.add_checkbox("Show Bones", initial_value=True)
    joint_radius = server.gui.add_slider("Joint Radius", min=0.001, max=0.015,
                                          step=0.001, initial_value=0.005)

    # ---- GUI: hand offset ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Hand Offset (display frame, meters)**")
    R_ = args.slider_range
    STEP = 0.001

    server.gui.add_markdown("Left hand")
    show_left_cb = server.gui.add_checkbox("Show Left", initial_value=False)
    lx = server.gui.add_slider("Left dx", min=-R_, max=R_, step=STEP, initial_value=0.0)
    ly = server.gui.add_slider("Left dy", min=-R_, max=R_, step=STEP, initial_value=0.0)
    lz = server.gui.add_slider("Left dz", min=-R_, max=R_, step=STEP, initial_value=0.0)

    server.gui.add_markdown("Right hand")
    show_right_cb = server.gui.add_checkbox("Show Right", initial_value=False)
    rx = server.gui.add_slider("Right dx", min=-R_, max=R_, step=STEP, initial_value=0.0)
    ry = server.gui.add_slider("Right dy", min=-R_, max=R_, step=STEP, initial_value=0.0)
    rz = server.gui.add_slider("Right dz", min=-R_, max=R_, step=STEP, initial_value=0.0)

    off_sliders = {"left": (lx, ly, lz), "right": (rx, ry, rz)}
    show_side = {"left": show_left_cb, "right": show_right_cb}

    # ---- GUI: aux object ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Aux Object (static, z-up world frame)**")

    has_aux_cb = server.gui.add_checkbox("Has aux in this sequence", initial_value=False)
    if aux_models:
        aux_id_dropdown = server.gui.add_dropdown(
            "Aux model", options=aux_models, initial_value=initial_aux,
        )
    else:
        aux_id_dropdown = None
    show_aux_cb = server.gui.add_checkbox("Show Aux", initial_value=True)

    A_ = args.aux_slider_range
    AUX_STEP = 0.005
    aux_x = server.gui.add_slider("Aux x", min=-A_, max=A_, step=AUX_STEP, initial_value=0.0)
    aux_y = server.gui.add_slider("Aux y", min=-A_, max=A_, step=AUX_STEP, initial_value=0.0)
    aux_z = server.gui.add_slider("Aux z", min=-A_, max=A_, step=AUX_STEP, initial_value=0.05)
    aux_yaw = server.gui.add_slider("Aux yaw  (Z, deg)", min=-180, max=180, step=1, initial_value=0)
    aux_pitch = server.gui.add_slider("Aux pitch (Y, deg)", min=-180, max=180, step=1, initial_value=0)
    aux_roll = server.gui.add_slider("Aux roll  (X, deg)", min=-180, max=180, step=1, initial_value=0)
    aux_scale = server.gui.add_slider("Aux scale", min=0.1, max=3.0, step=0.05, initial_value=1.0)

    # ---- Save / reset ----
    server.gui.add_markdown("---")
    save_btn = server.gui.add_button("Save (hand offset → pkl  +  aux → sidecar)")
    reset_hand_btn = server.gui.add_button("Reset hand sliders")
    reset_aux_btn = server.gui.add_button("Reset aux sliders")
    status_md = server.gui.add_markdown("_ready_")

    # ---- State ----
    state = {"idx": 0, "data": None, "suppress_events": False}
    aux_mesh = {"verts": None, "faces": None, "id": initial_aux}
    handles = {}

    def clear_handles(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def current_hand_offset(side: str) -> np.ndarray:
        sx, sy, sz = off_sliders[side]
        return np.array([sx.value, sy.value, sz.value], dtype=np.float32)

    def current_aux_pose():
        pos = np.array([aux_x.value, aux_y.value, aux_z.value], dtype=np.float32)
        euler = np.array([aux_roll.value, aux_pitch.value, aux_yaw.value], dtype=np.float32)
        return pos, euler, float(aux_scale.value)

    def load_aux_mesh(aux_id: str) -> bool:
        path = os.path.join(data_root, "models", aux_id, "cleaned_mesh_10000.obj")
        if not os.path.exists(path):
            print(f"[WARN] aux mesh not found: {path}")
            return False
        mesh = trimesh.load(path, process=False, force="mesh")
        aux_mesh["verts"] = mesh.vertices.copy().astype(np.float32)
        aux_mesh["faces"] = mesh.faces.copy()
        aux_mesh["id"] = aux_id
        return True

    if initial_aux is not None:
        load_aux_mesh(initial_aux)

    # ---- Draw functions ----
    def draw_hand(side: str, frame_idx: int):
        prefix = f"/{side}_hand"
        clear_handles(prefix)
        d = state["data"]
        present = d["has_left"] if side == "left" else d["has_right"]
        if not present or not show_side[side].value:
            return
        hand_data = d["mano_data"].get(side, {})
        offset = current_hand_offset(side)
        positions = get_hand_positions_with_offset(hand_data, frame_idx, offset)
        if not positions:
            return

        wrist_color = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        r = joint_radius.value

        for name, pos in positions.items():
            if name == "wrist":
                color = wrist_color
            else:
                color = (200, 200, 200)
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key,
                radius=r * 1.5 if name == "wrist" else r,
                position=pos.astype(np.float64),
                color=color,
            )

        if show_bones_cb.value:
            for finger, chain in FINGER_CHAINS.items():
                color = FINGER_COLORS[finger]
                for i in range(len(chain) - 1):
                    a, b = chain[i], chain[i + 1]
                    if a in positions and b in positions:
                        key = f"{prefix}/bone_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key,
                            positions=np.stack([positions[a].astype(np.float64),
                                                positions[b].astype(np.float64)]),
                            color=color,
                            line_width=2.0,
                        )

    def draw_main_object(frame_idx: int):
        prefix = "/object"
        clear_handles(prefix)
        d = state["data"]
        if not show_object_cb.value or d["obj_poses"] is None or d["obj_vertices"] is None:
            return
        if frame_idx >= len(d["obj_poses"]):
            return
        try:
            verts = transform_obj_vertices(d["obj_vertices"], d["obj_poses"][frame_idx])
        except ValueError as e:
            print(f"[WARN] {e}")
            return
        verts = flip_z(verts)
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32), faces=d["obj_faces"], color=OBJ_COLOR,
        )

    def draw_aux():
        prefix = "/aux"
        clear_handles(prefix)
        if not (has_aux_cb.value and show_aux_cb.value):
            return
        if aux_mesh["verts"] is None or aux_mesh["faces"] is None:
            return
        pos, euler, s = current_aux_pose()
        verts = transform_aux_vertices(aux_mesh["verts"], pos, euler, s)
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32), faces=aux_mesh["faces"], color=AUX_COLOR,
        )
        # Small axis indicator at aux origin
        handles[f"{prefix}/origin"] = server.scene.add_frame(
            f"{prefix}/origin",
            wxyz=tuple(euler_to_quat_wxyz(*euler)),
            position=tuple(pos.astype(np.float64)),
            axes_length=0.05,
            axes_radius=0.0015,
        )

    def redraw():
        if state["data"] is None:
            return
        f = int(frame_slider.value)
        draw_hand("left", f)
        draw_hand("right", f)
        draw_main_object(f)
        draw_aux()

    # ---- Sequence switching ----
    def load_idx(idx: int):
        idx = int(np.clip(idx, 0, len(sequences) - 1))
        name, pkl_path = sequences[idx]
        print(f"[LOAD] {idx + 1}/{len(sequences)}: {name}  ({pkl_path})")
        state["data"] = load_sequence_data(pkl_path)
        state["idx"] = idx

        state["suppress_events"] = True
        try:
            # Frame slider
            nf = state["data"]["num_frames"]
            frame_slider.max = max(nf - 1, 0)
            frame_slider.value = 0

            # Hand presence
            show_left_cb.value = state["data"]["has_left"]
            show_right_cb.value = state["data"]["has_right"]
            has_obj = (state["data"]["obj_poses"] is not None
                       and state["data"]["obj_vertices"] is not None)
            show_object_cb.value = has_obj

            # Reset hand offset sliders
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0

            # Aux: load from sidecar if exists, else clear
            sidecar = load_aux_sidecar(state["data"]["seq_dir"])
            if sidecar is not None:
                aux_id = sidecar.get("aux_obj_id", initial_aux)
                if aux_id_dropdown is not None and aux_id in aux_models:
                    aux_id_dropdown.value = aux_id
                if aux_id is not None:
                    load_aux_mesh(aux_id)
                pos = sidecar.get("aux_pos_world", [0.0, 0.0, 0.05])
                eul = sidecar.get("aux_euler_xyz_deg", [0.0, 0.0, 0.0])
                aux_x.value = float(pos[0])
                aux_y.value = float(pos[1])
                aux_z.value = float(pos[2])
                aux_roll.value = float(eul[0])
                aux_pitch.value = float(eul[1])
                aux_yaw.value = float(eul[2])
                aux_scale.value = float(sidecar.get("aux_scale", 1.0))
                has_aux_cb.value = True
            else:
                has_aux_cb.value = False
                # leave aux sliders at last position for fast iteration
                # (user can reset via the "Reset aux sliders" button)

            seq_dropdown.value = seq_labels[idx]
            seq_status.content = (
                f"**{idx + 1} / {len(sequences)}** — `{name}`  "
                f"({nf} frames, L={state['data']['has_left']}, R={state['data']['has_right']}, "
                f"aux={'yes' if has_aux_cb.value else 'no'})"
            )
        finally:
            state["suppress_events"] = False

        clear_handles()
        redraw()

    def on_any_update(_event=None):
        if state["suppress_events"]:
            return
        redraw()

    # Wire all sliders / checkboxes that affect rendering
    redraw_widgets = [
        frame_slider, show_object_cb, show_bones_cb, joint_radius,
        show_left_cb, show_right_cb, lx, ly, lz, rx, ry, rz,
        has_aux_cb, show_aux_cb,
        aux_x, aux_y, aux_z, aux_yaw, aux_pitch, aux_roll, aux_scale,
    ]
    for w in redraw_widgets:
        w.on_update(on_any_update)

    if aux_id_dropdown is not None:
        @aux_id_dropdown.on_update
        def _on_aux_id_change(_event):
            if state["suppress_events"]:
                return
            new_id = aux_id_dropdown.value
            if not load_aux_mesh(new_id):
                status_md.content = f"_aux mesh not found: {new_id}_"
            redraw()

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
        offsets = {side: current_hand_offset(side) for side in off_sliders}
        any_hand_change = not all(np.allclose(v, 0.0) for v in offsets.values())

        lines = []

        # 1) Hand offset → pkl
        if any_hand_change:
            targets = [d["pkl_path"]]
            if args.also_update_optimized:
                for name in ("mano_joints_optimized.pkl", "mano_joints_corrected.pkl"):
                    p = os.path.join(d["seq_dir"], name)
                    if os.path.exists(p):
                        targets.append(p)
            for p in targets:
                ok, msg = apply_offset_to_pkl(p, offsets)
                lines.append(("OK  " if ok else "FAIL") + " " + msg)
                print("[SAVE pkl]", msg)
        else:
            lines.append("info  hand sliders all zero — pkl untouched")

        # 2) Aux pose → sidecar (or remove sidecar if has_aux unchecked)
        if has_aux_cb.value:
            if aux_mesh["id"] is None:
                lines.append("FAIL  has_aux is checked but no aux model selected")
            else:
                pos, euler, s = current_aux_pose()
                payload = {
                    "aux_obj_id": aux_mesh["id"],
                    "aux_pos_world": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "aux_quat_wxyz_world": euler_to_quat_wxyz(*euler),
                    "aux_euler_xyz_deg": [float(euler[0]), float(euler[1]), float(euler[2])],
                    "aux_scale": s,
                }
                p = save_aux_sidecar(d["seq_dir"], payload)
                lines.append(f"OK  aux sidecar saved: {p}")
                print("[SAVE aux]", p, payload)
        else:
            removed = remove_aux_sidecar(d["seq_dir"])
            if removed:
                lines.append(f"OK  aux sidecar removed: {removed}")
                print("[SAVE aux removed]", removed)
            else:
                lines.append("info  has_aux unchecked, no existing sidecar")

        # Reload current seq + refresh label markers
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

        # Re-load to pick up baked hand state; aux sliders stay at user's chosen values
        prev_aux_state = (
            has_aux_cb.value, aux_x.value, aux_y.value, aux_z.value,
            aux_roll.value, aux_pitch.value, aux_yaw.value, aux_scale.value,
            (aux_id_dropdown.value if aux_id_dropdown is not None else None),
        )
        state["data"] = load_sequence_data(pkl_path)
        state["suppress_events"] = True
        try:
            # Zero hand sliders (offset already baked)
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0
            # Restore aux sliders (saved values are now reflected in sidecar too)
            (
                has_aux_cb.value, aux_x.value, aux_y.value, aux_z.value,
                aux_roll.value, aux_pitch.value, aux_yaw.value, aux_scale.value,
                _aux_id,
            ) = prev_aux_state
            if aux_id_dropdown is not None and _aux_id is not None:
                aux_id_dropdown.value = _aux_id
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "**Saved.**\n\n" + "\n\n".join(lines)

    @reset_hand_btn.on_click
    def _on_reset_hand(_event):
        state["suppress_events"] = True
        try:
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "_hand sliders reset_"

    @reset_aux_btn.on_click
    def _on_reset_aux(_event):
        state["suppress_events"] = True
        try:
            aux_x.value = 0.0
            aux_y.value = 0.0
            aux_z.value = 0.05
            aux_yaw.value = 0.0
            aux_pitch.value = 0.0
            aux_roll.value = 0.0
            aux_scale.value = 1.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "_aux sliders reset_"

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
