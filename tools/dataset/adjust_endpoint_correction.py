#!/usr/bin/env python3
"""
Pin trajectory endpoints (frame 0 + final frame) of the main object to
user-specified corrections, then linearly blend the correction across all
frames so the middle of the trajectory preserves the captured motion shape
while endpoints are fixed.

Use case: object detection at frame 0 and at the final frame is noisy → object
appears floating at start and sinking into table at end. The middle frames are
usually fine. Slide `start dz` down and `end dz` up; the system applies

    delta(t) = (1 − α(t)) · delta_start + α(t) · delta_end
    α(t)     = t / (T − 1)

to the object translation AND the same blend to every hand joint position
(if "shift hand" is checked). Aux is static, never modified.

Reads:  mano_joints_corrected.pkl (preferred; falls back to optimized / original)
Writes: same pkl  +  first-time .bak backup.

Usage:
  python tools/dataset/adjust_endpoint_correction.py \\
      --pkl data/robotool_batch/.../mano_joints_corrected.pkl

  python tools/dataset/adjust_endpoint_correction.py \\
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

# ---- Skeleton ----

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
START_GHOST_COLOR = (100, 220, 120)   # green
END_GHOST_COLOR = (220, 100, 100)     # red
TRACE_COLOR = (200, 200, 80)
AUX_COLOR = (110, 220, 110)

EDITED_MARK = "✸"
AUX_SIDECAR_NAME = "aux_object.json"


# ----------------------------- helpers ----------------------------------

def flip_z(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


def pick_pkl(seq_dir: str) -> Optional[str]:
    for name in ("mano_joints_corrected.pkl",
                 "mano_joints_optimized.pkl",
                 "mano_joints.pkl"):
        p = os.path.join(seq_dir, name)
        if os.path.exists(p):
            return p
    return None


def discover_sequences(task_dir: str) -> List[Tuple[str, str]]:
    out = []
    for name in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, name)
        if not os.path.isdir(sub):
            continue
        meta = os.path.join(sub, "meta.json")
        pkl = pick_pkl(sub)
        if pkl and os.path.isfile(meta):
            out.append((name, pkl))
    return out


def derive_data_root(any_pkl_path: str) -> str:
    seq_dir = os.path.dirname(os.path.abspath(any_pkl_path))
    return os.path.dirname(os.path.dirname(seq_dir))


def seq_label(name: str, pkl_path: str) -> str:
    seq_dir = os.path.dirname(pkl_path)
    # Use a dedicated suffix to mark "endpoint correction already applied"
    edited = os.path.exists(os.path.join(seq_dir, "mano_joints_corrected.pkl.endpt.bak"))
    has_aux = os.path.exists(os.path.join(seq_dir, AUX_SIDECAR_NAME))
    marks = []
    if edited:
        marks.append(EDITED_MARK)
    if has_aux:
        marks.append("Ⓐ")
    prefix = "".join(marks) if marks else " "
    return f"{prefix} {name}"


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


# ----------------------------- loading ----------------------------------

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
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    meta_path = os.path.join(seq_dir, "meta.json")
    with open(pkl_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    data_root = derive_data_root(pkl_path)

    obj_vertices = None
    obj_faces = None
    obj_id = None
    if meta.get("object_ids"):
        obj_id = meta["object_ids"][0]
        obj_path = os.path.join(data_root, "models", obj_id, "cleaned_mesh_10000.obj")
        if os.path.exists(obj_path):
            m = trimesh.load(obj_path, process=False, force="mesh")
            obj_vertices = m.vertices.copy().astype(np.float32)
            obj_faces = m.faces.copy()

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

    aux_info = load_aux_sidecar(seq_dir)
    aux_verts = None
    aux_faces = None
    if aux_info is not None and aux_info.get("aux_obj_id"):
        aux_verts, aux_faces = load_aux_mesh(data_root, aux_info["aux_obj_id"])

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
        "aux_info": aux_info,
        "aux_verts": aux_verts,
        "aux_faces": aux_faces,
    }


# ----------------------------- interpolation -----------------------------

def alpha_curve(T: int, mode: str = "linear") -> np.ndarray:
    """Return α[t] ∈ [0,1] for t = 0..T-1 according to interpolation mode."""
    if T <= 1:
        return np.zeros(T, dtype=np.float64)
    t = np.arange(T, dtype=np.float64) / (T - 1)
    if mode == "linear":
        return t
    if mode == "cosine":
        # ease in/out: 0 at 0, 1 at 1, zero derivative at both ends
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if mode == "smoothstep":
        # 3t² − 2t³
        return t * t * (3 - 2 * t)
    return t


def lerp_deltas(delta_start_stored: np.ndarray,
                  delta_end_stored: np.ndarray,
                  T: int, mode: str = "linear") -> np.ndarray:
    """Return (T, 3) per-frame stored-frame correction."""
    a = alpha_curve(T, mode).reshape(T, 1)
    return (1.0 - a) * delta_start_stored + a * delta_end_stored


# ----------------------------- bake -------------------------------------

def bake_endpoint_correction(
    pkl_path: str,
    delta_start_disp: np.ndarray,
    delta_end_disp: np.ndarray,
    shift_hand: bool,
    interp_mode: str,
    backup_suffix: str = ".endpt.bak",
) -> Tuple[bool, str]:
    """Linear (or cosine/smoothstep) blend between start/end corrections,
    applied to:
      - tool_object_pose translation (per frame), and
      - hand POSITION_FIELDS (per frame, if shift_hand).

    Stored arrays are in camera (z-down) frame, so display offsets are flipped
    via (dx, -dy, -dz) before being blended.
    """
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

    ds = np.array([delta_start_disp[0], -delta_start_disp[1], -delta_start_disp[2]],
                  dtype=np.float64)
    de = np.array([delta_end_disp[0], -delta_end_disp[1], -delta_end_disp[2]],
                  dtype=np.float64)

    actions = []

    # 1) tool_object_pose translations
    od = data.get("original_data", {})
    tp = od.get("tool_object_pose", None)
    if tp is None:
        return False, "no tool_object_pose in pkl"
    tp_arr = np.asarray(tp).astype(np.float64).copy()

    if tp_arr.ndim == 3 and tp_arr.shape[1:] == (4, 4):
        T = tp_arr.shape[0]
        deltas = lerp_deltas(ds, de, T, interp_mode)  # (T, 3)
        tp_arr[:, :3, 3] = tp_arr[:, :3, 3] + deltas
        actions.append(f"obj 4x4 (T={T}, mode={interp_mode})")
    elif tp_arr.ndim == 2 and tp_arr.shape[1] == 7:
        T = tp_arr.shape[0]
        deltas = lerp_deltas(ds, de, T, interp_mode)
        tp_arr[:, 4:7] = tp_arr[:, 4:7] + deltas
        actions.append(f"obj 7d (T={T}, mode={interp_mode})")
    else:
        return False, f"unexpected tool_object_pose shape: {tp_arr.shape}"
    od["tool_object_pose"] = tp_arr
    data["original_data"] = od

    # 2) hand fields (each field may have its own T_f)
    if shift_hand:
        for side in ("left", "right"):
            if side not in data or not isinstance(data[side], dict):
                continue
            n_shifted = 0
            for field in POSITION_FIELDS:
                if field in data[side]:
                    arr = np.asarray(data[side][field]).astype(np.float64)
                    if arr.ndim == 2 and arr.shape[-1] == 3:
                        T_f = arr.shape[0]
                        deltas_f = lerp_deltas(ds, de, T_f, interp_mode)
                        data[side][field] = (arr + deltas_f).astype(np.float32)
                        n_shifted += 1
            if n_shifted > 0:
                actions.append(f"hand {side} ({n_shifted} fields)")

    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    msg = (f"saved {pkl_path}  start={delta_start_disp.tolist()}  "
           f"end={delta_end_disp.tolist()}  | " + ", ".join(actions))
    msg += f"  [backup: {bak}]" if backed_up else "  [existing .endpt.bak kept]"
    return True, msg


# ----------------------------- main -------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pin object trajectory endpoints + linear-in-time blend correction."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl", type=str)
    group.add_argument("--task_dir", type=str)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--slider_range", type=float, default=0.15,
                        help="Endpoint correction slider range in meters (default 0.15)")
    args = parser.parse_args()

    # ---- Sequences ----
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
            print(f"[ERROR] no sequences in {args.task_dir}")
            sys.exit(1)

    print(f"[INFO] {len(sequences)} sequences")
    for n, p in sequences:
        tags = []
        if os.path.exists(os.path.join(os.path.dirname(p),
                                        "mano_joints_corrected.pkl.endpt.bak")):
            tags.append("endpt-edited")
        if os.path.exists(os.path.join(os.path.dirname(p), AUX_SIDECAR_NAME)):
            tags.append("has-aux")
        tag = " [" + ", ".join(tags) + "]" if tags else ""
        print(f"  - {n}  ({os.path.basename(p)}){tag}")

    # ---- Viser ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    seq_labels = [seq_label(n, p) for n, p in sequences]
    seq_dropdown = server.gui.add_dropdown("Sequence", options=seq_labels,
                                            initial_value=seq_labels[0])
    prev_btn = server.gui.add_button("◀ Prev")
    next_btn = server.gui.add_button("Next ▶")
    seq_status = server.gui.add_markdown(f"**1 / {len(sequences)}**")

    server.gui.add_markdown("---")
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    show_object_cb = server.gui.add_checkbox("Show main (current frame)", initial_value=True)
    show_start_ghost_cb = server.gui.add_checkbox("Show start ghost (frame 0)", initial_value=True)
    show_end_ghost_cb = server.gui.add_checkbox("Show end ghost (final)", initial_value=True)
    show_trace_cb = server.gui.add_checkbox("Show object center trace", initial_value=True)
    show_hand_cb = server.gui.add_checkbox("Show hand", initial_value=True)
    show_bones_cb = server.gui.add_checkbox("Show bones", initial_value=True)
    show_aux_cb = server.gui.add_checkbox("Show aux (static)", initial_value=True)
    joint_radius = server.gui.add_slider("Joint radius", min=0.001, max=0.015,
                                          step=0.001, initial_value=0.005)
    trace_radius = server.gui.add_slider("Trace radius", min=0.001, max=0.010,
                                          step=0.001, initial_value=0.003)

    # ---- Endpoint correction sliders (display frame, meters) ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Start (frame 0) correction**")
    R_ = args.slider_range
    STEP = 0.001
    sx = server.gui.add_slider("Start dx", min=-R_, max=R_, step=STEP, initial_value=0.0)
    sy = server.gui.add_slider("Start dy", min=-R_, max=R_, step=STEP, initial_value=0.0)
    sz = server.gui.add_slider("Start dz", min=-R_, max=R_, step=STEP, initial_value=0.0)

    server.gui.add_markdown("**End (final frame) correction**")
    ex = server.gui.add_slider("End dx", min=-R_, max=R_, step=STEP, initial_value=0.0)
    ey = server.gui.add_slider("End dy", min=-R_, max=R_, step=STEP, initial_value=0.0)
    ez = server.gui.add_slider("End dz", min=-R_, max=R_, step=STEP, initial_value=0.0)

    server.gui.add_markdown("---")
    interp_dropdown = server.gui.add_dropdown(
        "Blend curve", options=["linear", "cosine", "smoothstep"],
        initial_value="linear",
    )
    shift_hand_cb = server.gui.add_checkbox(
        "Also shift hand (keeps grasp aligned)", initial_value=True,
    )

    server.gui.add_markdown("---")
    save_btn = server.gui.add_button("Save (bake correction into pkl)")
    reset_btn = server.gui.add_button("Reset sliders to 0")
    status_md = server.gui.add_markdown("_ready_")

    # ---- State ----
    state = {"idx": 0, "data": None, "suppress_events": False}
    handles = {}

    def clear_handles(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def delta_start_disp() -> np.ndarray:
        return np.array([sx.value, sy.value, sz.value], dtype=np.float64)

    def delta_end_disp() -> np.ndarray:
        return np.array([ex.value, ey.value, ez.value], dtype=np.float64)

    def per_frame_delta_disp(T_total: int) -> np.ndarray:
        """(T, 3) array of per-frame correction in z-up display frame."""
        return lerp_deltas(delta_start_disp(), delta_end_disp(),
                           T_total, interp_dropdown.value)

    # ---- Draw ----
    def draw_main_at_frame(frame_idx: int, delta_disp: np.ndarray,
                            color, prefix: str, alpha_hint: bool = False):
        clear_handles(prefix)
        d = state["data"]
        if d["obj_poses"] is None or d["obj_vertices"] is None:
            return
        if frame_idx >= len(d["obj_poses"]):
            return
        try:
            verts = transform_obj_vertices(d["obj_vertices"], d["obj_poses"][frame_idx])
        except ValueError as e:
            print(f"[WARN] {e}")
            return
        verts = flip_z(verts) + delta_disp
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32),
            faces=d["obj_faces"], color=color,
        )

    def draw_trace():
        prefix = "/trace"
        clear_handles(prefix)
        if not show_trace_cb.value:
            return
        d = state["data"]
        if d["obj_poses"] is None or d["obj_vertices"] is None:
            return
        T = len(d["obj_poses"])
        centroid_local = d["obj_vertices"].mean(axis=0).astype(np.float64)
        deltas = per_frame_delta_disp(T)  # (T, 3) display
        centers = []
        for t in range(T):
            try:
                c_cam = transform_obj_vertices(centroid_local[None, :],
                                                 d["obj_poses"][t])[0]
            except ValueError:
                continue
            c_vis = np.array([c_cam[0], -c_cam[1], -c_cam[2]]) + deltas[t]
            centers.append(c_vis)
        if len(centers) < 2:
            return
        centers = np.array(centers)
        # Polyline
        key = f"{prefix}/line"
        handles[key] = server.scene.add_spline_catmull_rom(
            key, positions=centers, color=TRACE_COLOR, line_width=2.0,
        )
        # Sparse dots every ~10 frames
        r = trace_radius.value
        for i in range(0, len(centers), max(1, len(centers) // 30)):
            kk = f"{prefix}/dot_{i}"
            handles[kk] = server.scene.add_icosphere(
                kk, radius=r, position=centers[i],
                color=TRACE_COLOR,
            )

    def draw_hand(side: str, frame_idx: int, delta_disp: np.ndarray):
        prefix = f"/{side}_hand"
        clear_handles(prefix)
        d = state["data"]
        present = d["has_left"] if side == "left" else d["has_right"]
        if not present or not show_hand_cb.value:
            return
        hand_data = d["mano_data"].get(side, {})
        positions = {}
        for name in JOINT_NAMES:
            if name in hand_data:
                arr = np.asarray(hand_data[name])
                if arr.ndim == 2 and frame_idx < len(arr):
                    positions[name] = flip_z(arr[frame_idx]) + delta_disp
        if not positions:
            return

        wrist_color = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        r = joint_radius.value
        for name, pos in positions.items():
            color = wrist_color if name == "wrist" else (200, 200, 200)
            if name != "wrist":
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key, radius=r * 1.5 if name == "wrist" else r,
                position=pos.astype(np.float64), color=color,
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
                            color=color, line_width=2.0,
                        )

    def draw_aux():
        prefix = "/aux"
        clear_handles(prefix)
        d = state["data"]
        if not show_aux_cb.value or d["aux_info"] is None or d["aux_verts"] is None:
            return
        R, pos, scale = aux_pose_to_R_t(d["aux_info"])
        verts = (R @ (d["aux_verts"] * scale).T).T + pos
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=verts.astype(np.float32),
            faces=d["aux_faces"], color=AUX_COLOR,
        )

    def redraw():
        d = state["data"]
        if d is None or d["obj_poses"] is None:
            clear_handles()
            return
        T = len(d["obj_poses"])
        deltas = per_frame_delta_disp(T)
        f = int(frame_slider.value)
        f = min(max(f, 0), T - 1)
        # Current frame's per-frame delta
        delta_cur = deltas[f]
        # Start / end deltas
        delta_0 = deltas[0]
        delta_T = deltas[-1]

        # Main object current
        if show_object_cb.value:
            draw_main_at_frame(f, delta_cur, OBJ_COLOR, "/object_cur")
        else:
            clear_handles("/object_cur")

        # Ghosts
        if show_start_ghost_cb.value:
            draw_main_at_frame(0, delta_0, START_GHOST_COLOR, "/object_start")
        else:
            clear_handles("/object_start")

        if show_end_ghost_cb.value:
            draw_main_at_frame(T - 1, delta_T, END_GHOST_COLOR, "/object_end")
        else:
            clear_handles("/object_end")

        # Trace
        draw_trace()

        # Hand
        hand_delta = delta_cur if shift_hand_cb.value else np.zeros(3, dtype=np.float64)
        draw_hand("left", f, hand_delta)
        draw_hand("right", f, hand_delta)

        # Aux (always at sidecar pose, never per-frame-corrected)
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
            nf = state["data"]["num_frames"]
            frame_slider.max = max(nf - 1, 0)
            frame_slider.value = 0
            for w in (sx, sy, sz, ex, ey, ez):
                w.value = 0.0
            has_obj = (state["data"]["obj_poses"] is not None
                       and state["data"]["obj_vertices"] is not None)
            show_object_cb.value = has_obj
            seq_dropdown.value = seq_labels[idx]
            aux_present = state["data"]["aux_info"] is not None
            seq_status.content = (
                f"**{idx + 1} / {len(sequences)}** — `{name}` "
                f"({os.path.basename(pkl_path)})  "
                f"(T={nf}, obj={'yes' if has_obj else 'no'}, "
                f"aux={'yes' if aux_present else 'no'})"
            )
        finally:
            state["suppress_events"] = False
        clear_handles()
        redraw()

    def on_any_update(_event=None):
        if state["suppress_events"]:
            return
        redraw()

    for w in (frame_slider,
              show_object_cb, show_start_ghost_cb, show_end_ghost_cb,
              show_trace_cb, show_hand_cb, show_bones_cb, show_aux_cb,
              joint_radius, trace_radius,
              sx, sy, sz, ex, ey, ez,
              interp_dropdown, shift_hand_cb):
        w.on_update(on_any_update)

    @seq_dropdown.on_update
    def _on_seq(_e):
        if state["suppress_events"]:
            return
        idx = seq_labels.index(seq_dropdown.value)
        if idx != state["idx"]:
            load_idx(idx)

    @prev_btn.on_click
    def _on_prev(_e):
        if state["idx"] > 0:
            load_idx(state["idx"] - 1)

    @next_btn.on_click
    def _on_next(_e):
        if state["idx"] < len(sequences) - 1:
            load_idx(state["idx"] + 1)

    @save_btn.on_click
    def _on_save(_e):
        ds = delta_start_disp()
        de = delta_end_disp()
        if np.allclose(ds, 0.0) and np.allclose(de, 0.0):
            status_md.content = "_no change: start and end deltas both zero_"
            return
        d = state["data"]
        ok, msg = bake_endpoint_correction(
            d["pkl_path"], ds, de,
            shift_hand=shift_hand_cb.value,
            interp_mode=interp_dropdown.value,
        )
        line = ("OK  " if ok else "FAIL") + " " + msg
        print("[SAVE]", msg)

        # Refresh label
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

        # Reload, zero sliders
        state["data"] = load_sequence_data(pkl_path)
        state["suppress_events"] = True
        try:
            for w in (sx, sy, sz, ex, ey, ez):
                w.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "**Saved.**\n\n" + line

    @reset_btn.on_click
    def _on_reset(_e):
        state["suppress_events"] = True
        try:
            for w in (sx, sy, sz, ex, ey, ez):
                w.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "_sliders reset_"

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
