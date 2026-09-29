#!/usr/bin/env python3
"""
Interactively edit object pose for a specific FRAME RANGE, bake into
mano_joints_corrected.pkl.

Use case: detection went off only on the last few frames (or some middle
range), but the rest of the trajectory is fine. Pick the range, slide
xyz/rpy deltas to align the object visually, save. The script applies a
smooth ramp-in (and optional ramp-out) so the corrected segment blends with
the unedited neighbors — no pose jumps.

Edits are in z-up world frame (same as you see in viser):
  - dx/dy/dz  → translation in meters
  - roll/pitch/yaw → rotation (deg, intrinsic XYZ Euler)

Bake converts back to the pkl's camera-frame storage (z-down) and writes
to `mano_joints_corrected.pkl` with a `.frameedit.bak` backup on first save.

Usage:
    python tools/dataset/edit_frame_range_pose.py \\
        --seq data/robotool_batch/0423_dual/blue_cup_1

    # Or task-level browsing:
    python tools/dataset/edit_frame_range_pose.py \\
        --task_dir data/robotool_batch/0423_dual
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as Rot


# ---------- skeleton constants (matches other visualizers) ----------

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
FINGER_COLORS = {
    "thumb":  (255, 100, 100),
    "index":  (100, 255, 100),
    "middle": (100, 100, 255),
    "ring":   (255, 255, 100),
    "pinky":  (255, 100, 255),
}
LEFT_WRIST_COLOR  = (100, 149, 237)
RIGHT_WRIST_COLOR = (255, 160, 122)
OBJ_COLOR         = (180, 180, 180)
EDIT_GHOST_COLOR  = (250, 180, 90)     # frames inside edit range
ORIG_GHOST_COLOR  = (90, 90, 90)       # source pose at current frame (faint)
START_MARK_COLOR  = (100, 220, 120)
END_MARK_COLOR    = (220, 100, 100)
AUX_COLOR         = (110, 220, 110)
EDITED_MARK       = "✦"
AUX_SIDECAR_NAME  = "aux_object.json"


# --------------------------- helpers -----------------------------------

def flip_z(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


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
    out = []
    for n in sorted(os.listdir(task_dir)):
        sub = os.path.join(task_dir, n)
        if not os.path.isdir(sub):
            continue
        meta = os.path.join(sub, "meta.json")
        pkl = pick_pkl(sub)
        if pkl and os.path.isfile(meta):
            out.append((n, pkl))
    return out


def derive_data_root(pkl_path: str) -> str:
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    return os.path.dirname(os.path.dirname(seq_dir))


def seq_label(name: str, pkl_path: str) -> str:
    seq_dir = os.path.dirname(pkl_path)
    edited = os.path.exists(os.path.join(
        seq_dir, "mano_joints_corrected.pkl.frameedit.bak"
    ))
    marks = []
    if edited:
        marks.append(EDITED_MARK)
    return f"{''.join(marks) if marks else ' '} {name}"


def transform_obj_vertices(vertices: np.ndarray, pose) -> np.ndarray:
    """Apply 4x4 (or 7-vec) pose to vertices."""
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


# --------------------------- loading -----------------------------------

def load_sequence_data(pkl_path: str) -> dict:
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    meta_path = os.path.join(seq_dir, "meta.json")
    with open(pkl_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    data_root = derive_data_root(pkl_path)

    obj_vertices = obj_faces = None
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
    has_left  = "left"  in mano_sides and "left"  in mano_data and \
        len(np.asarray(mano_data["left"].get("wrist", []))) > 0
    has_right = "right" in mano_sides and "right" in mano_data and \
        len(np.asarray(mano_data["right"].get("wrist", []))) > 0

    aux_info = None
    aux_verts = aux_faces = None
    aux_path = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if os.path.exists(aux_path):
        with open(aux_path) as f:
            aux_info = json.load(f)
        if aux_info.get("aux_obj_id"):
            mesh = os.path.join(data_root, "models", aux_info["aux_obj_id"],
                                 "cleaned_mesh_10000.obj")
            if os.path.exists(mesh):
                m = trimesh.load(mesh, process=False, force="mesh")
                aux_verts = m.vertices.copy().astype(np.float32)
                aux_faces = m.faces.copy()

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


# ----- frame-range alpha ramp ------------------------------------------

def compute_alpha_per_frame(T: int,
                              edit_start: int, edit_end: int,
                              ramp_in: int, ramp_out: int) -> np.ndarray:
    """Return (T,) alpha values in [0, 1]:
        0 before edit_start / after edit_end,
        ramps from 0 → 1 over ramp_in frames at start of range,
        ramps from 1 → 0 over ramp_out frames at end of range,
        1 throughout the interior of the range.
    """
    alpha = np.zeros(T, dtype=np.float64)
    edit_start = max(0, min(edit_start, T - 1))
    edit_end   = max(edit_start, min(edit_end, T - 1))
    for f in range(edit_start, edit_end + 1):
        a_in  = 1.0 if ramp_in <= 0 else min(1.0, (f - edit_start + 1) / float(ramp_in))
        a_out = 1.0 if ramp_out <= 0 else min(1.0, (edit_end - f + 1) / float(ramp_out))
        alpha[f] = min(a_in, a_out)
    return alpha


# ----- edit application ------------------------------------------------

def apply_edit_world(R_orig_cam: np.ndarray,
                      t_orig_cam: np.ndarray,
                      dt_world: np.ndarray,
                      R_delta_world: np.ndarray,
                      alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply world-frame delta translation + rotation (both scaled by alpha)
    to a camera-frame pose. Returns new (R_cam, t_cam).

    Stored in pkl is camera frame (z-down). Display / edit is in z-up world.
    flip_z = diag(1, -1, -1) converts both ways (involution).

    Math:
       1. Convert R_orig to world frame:  R_world = flip_z @ R_orig_cam @ flip_z
       2. Apply alpha-scaled world delta:
          R_world_new = R_delta_alpha @ R_world
          t_world_new = t_world + alpha * dt_world
       3. Convert back:  R_cam_new = flip_z @ R_world_new @ flip_z
                         t_cam_new = flip_z @ t_world_new
    """
    FLIP_Z = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    t_world = FLIP_Z @ t_orig_cam.astype(np.float64)
    R_world = FLIP_Z @ R_orig_cam.astype(np.float64) @ FLIP_Z

    # Alpha-scaled rotation: linear in axis-angle space (axis preserved,
    # angle multiplied by alpha)
    rv = Rot.from_matrix(R_delta_world).as_rotvec() * alpha
    R_delta_alpha = Rot.from_rotvec(rv).as_matrix()

    R_world_new = R_delta_alpha @ R_world
    t_world_new = t_world + alpha * dt_world

    R_cam_new = FLIP_Z @ R_world_new @ FLIP_Z
    t_cam_new = FLIP_Z @ t_world_new
    return R_cam_new, t_cam_new


def euler_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Intrinsic XYZ Euler (degrees) → 3x3 rotation matrix."""
    return Rot.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg],
                           degrees=True).as_matrix()


# ----- save (bake into pkl) --------------------------------------------

def bake_into_pkl(pkl_path: str,
                   alpha_per_frame: np.ndarray,
                   dt_world: np.ndarray,
                   R_delta_world: np.ndarray,
                   shift_hand: bool,
                   out_filename: str = "mano_joints_corrected.pkl",
                   backup_suffix: str = ".frameedit.bak") -> Tuple[bool, str]:
    """Apply alpha-weighted world-frame delta to each frame of tool_object_pose
    (and optionally hand POSITION_FIELDS), write to `out_filename` in the
    same seq dir. Backs up existing output on first overwrite.
    """
    if not os.path.exists(pkl_path):
        return False, f"source pkl not found: {pkl_path}"

    seq_dir = os.path.dirname(pkl_path)
    out_path = os.path.join(seq_dir, out_filename)

    # Backup if writing to existing file
    if os.path.exists(out_path):
        bak = out_path + backup_suffix
        if not os.path.exists(bak):
            shutil.copy2(out_path, bak)
            backed_up = bak
        else:
            backed_up = None
    else:
        backed_up = None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    od = data.get("original_data", {})
    tp = od.get("tool_object_pose")
    if tp is None:
        return False, "no tool_object_pose in pkl"
    tp_arr = np.asarray(tp).astype(np.float64).copy()
    T = tp_arr.shape[0]
    actions = []

    if tp_arr.ndim == 3 and tp_arr.shape[1:] == (4, 4):
        for f in range(T):
            a = float(alpha_per_frame[f])
            if a <= 0.0:
                continue
            R_new, t_new = apply_edit_world(
                tp_arr[f, :3, :3], tp_arr[f, :3, 3],
                dt_world, R_delta_world, a,
            )
            tp_arr[f, :3, :3] = R_new
            tp_arr[f, :3, 3] = t_new
        actions.append(f"obj 4x4 (T={T})")
    elif tp_arr.ndim == 2 and tp_arr.shape[1] == 7:
        for f in range(T):
            a = float(alpha_per_frame[f])
            if a <= 0.0:
                continue
            R_cam = Rot.from_quat(tp_arr[f, :4]).as_matrix()
            t_cam = tp_arr[f, 4:7]
            R_new, t_new = apply_edit_world(R_cam, t_cam, dt_world,
                                              R_delta_world, a)
            tp_arr[f, :4] = Rot.from_matrix(R_new).as_quat()
            tp_arr[f, 4:7] = t_new
        actions.append(f"obj 7d (T={T})")
    else:
        return False, f"unexpected tool_object_pose shape: {tp_arr.shape}"

    od["tool_object_pose"] = tp_arr
    data["original_data"] = od

    # Hand position fields: optional — translation only, no rotation
    if shift_hand:
        FLIP_Z = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
        dt_cam = FLIP_Z @ dt_world.astype(np.float64)
        POSITION_FIELDS = JOINT_NAMES + ["wrist_translation"]
        for side in ("left", "right"):
            if side not in data or not isinstance(data[side], dict):
                continue
            n = 0
            for field in POSITION_FIELDS:
                if field in data[side]:
                    arr = np.asarray(data[side][field]).astype(np.float64).copy()
                    if arr.ndim == 2 and arr.shape[-1] == 3:
                        T_f = arr.shape[0]
                        per_frame_alpha = alpha_per_frame[:T_f] \
                            if T_f <= len(alpha_per_frame) \
                            else np.concatenate([alpha_per_frame,
                                                  np.zeros(T_f - len(alpha_per_frame))])
                        arr += per_frame_alpha[:, None] * dt_cam[None, :]
                        data[side][field] = arr.astype(np.float32)
                        n += 1
            if n > 0:
                actions.append(f"hand {side} ({n} fields, translation only)")

    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    msg = f"saved {out_path}: " + ", ".join(actions)
    if backed_up:
        msg += f"  [backup: {backed_up}]"
    return True, msg


# --------------------------- main GUI ----------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactively edit object pose for a frame range, bake into pkl."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seq", type=str,
                       help="Single seq dir (e.g. data/robotool_batch/0423_dual/blue_cup_1)")
    group.add_argument("--task_dir", type=str,
                       help="Task dir; browse all seqs inside via dropdown")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--xyz_range",     type=float, default=0.15,
                        help="Translation slider range, meters (default 0.15)")
    parser.add_argument("--rpy_range_deg", type=float, default=180.0,
                        help="Rotation slider range, degrees (default 180)")
    args = parser.parse_args()

    if args.seq:
        if not os.path.isdir(args.seq):
            print(f"[ERROR] seq dir not found: {args.seq}")
            sys.exit(1)
        pkl = pick_pkl(args.seq)
        if pkl is None:
            print(f"[ERROR] no pkl in {args.seq}")
            sys.exit(1)
        name = os.path.basename(os.path.abspath(args.seq))
        sequences = [(name, pkl)]
    else:
        if not os.path.isdir(args.task_dir):
            print(f"[ERROR] task_dir not found: {args.task_dir}")
            sys.exit(1)
        sequences = discover_sequences(args.task_dir)
        if not sequences:
            print(f"[ERROR] no sequences in {args.task_dir}")
            sys.exit(1)
    print(f"[INFO] {len(sequences)} sequence(s)")
    for n, p in sequences:
        print(f"  - {n}  ({os.path.basename(p)})")

    # ---- viser ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    # ---- GUI: nav ----
    seq_labels = [seq_label(n, p) for n, p in sequences]
    seq_dropdown = server.gui.add_dropdown("Sequence", options=seq_labels,
                                            initial_value=seq_labels[0])
    prev_btn = server.gui.add_button("◀ Prev")
    next_btn = server.gui.add_button("Next ▶")
    seq_status = server.gui.add_markdown(f"**1 / {len(sequences)}**")

    # ---- GUI: playback ----
    server.gui.add_markdown("---")
    frame_slider = server.gui.add_slider("Frame (view)", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    show_object_cb = server.gui.add_checkbox("Show object (current frame, edited)", initial_value=True)
    show_orig_cb = server.gui.add_checkbox("Show original at current (faint)", initial_value=True)
    show_start_ghost_cb = server.gui.add_checkbox("Show edit_start ghost", initial_value=True)
    show_end_ghost_cb = server.gui.add_checkbox("Show edit_end ghost", initial_value=True)
    show_trace_cb = server.gui.add_checkbox("Show object center trace", initial_value=True)
    show_hand_cb = server.gui.add_checkbox("Show hand", initial_value=True)
    show_bones_cb = server.gui.add_checkbox("Show bones", initial_value=True)
    show_aux_cb = server.gui.add_checkbox("Show aux (if any)", initial_value=True)
    joint_radius = server.gui.add_slider("Joint radius", min=0.001, max=0.015,
                                          step=0.001, initial_value=0.005)
    trace_radius = server.gui.add_slider("Trace radius", min=0.001, max=0.010,
                                          step=0.001, initial_value=0.003)

    # ---- GUI: edit range ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**Edit range** (frames inside this range get the delta)")
    edit_start = server.gui.add_slider("edit_start", min=0, max=1, step=1, initial_value=0)
    edit_end   = server.gui.add_slider("edit_end",   min=0, max=1, step=1, initial_value=1)
    ramp_in    = server.gui.add_slider("ramp_in  (frames, 0 → full at start)",
                                        min=0, max=30, step=1, initial_value=3)
    ramp_out   = server.gui.add_slider("ramp_out (frames, full → 0 at end)",
                                        min=0, max=30, step=1, initial_value=0)

    # ---- GUI: edit delta (world z-up) ----
    server.gui.add_markdown("---")
    server.gui.add_markdown("**XYZ-RPY delta** (world frame; intrinsic XYZ Euler)")
    R_ = args.xyz_range
    XYZ_STEP = 0.001
    dx = server.gui.add_slider("dx (m)", min=-R_, max=R_, step=XYZ_STEP, initial_value=0.0)
    dy = server.gui.add_slider("dy (m)", min=-R_, max=R_, step=XYZ_STEP, initial_value=0.0)
    dz = server.gui.add_slider("dz (m)", min=-R_, max=R_, step=XYZ_STEP, initial_value=0.0)
    RPY = args.rpy_range_deg
    RPY_STEP = 0.5
    droll  = server.gui.add_slider("roll  (X, deg)", min=-RPY, max=RPY, step=RPY_STEP, initial_value=0.0)
    dpitch = server.gui.add_slider("pitch (Y, deg)", min=-RPY, max=RPY, step=RPY_STEP, initial_value=0.0)
    dyaw   = server.gui.add_slider("yaw   (Z, deg)", min=-RPY, max=RPY, step=RPY_STEP, initial_value=0.0)
    shift_hand_cb = server.gui.add_checkbox(
        "Also translate hand by (dx,dy,dz) over the range",
        initial_value=False,
    )

    # ---- GUI: save ----
    server.gui.add_markdown("---")
    save_btn  = server.gui.add_button("Save (bake into mano_joints_corrected.pkl)")
    reset_btn = server.gui.add_button("Reset deltas + range")
    status_md = server.gui.add_markdown("_ready_")

    # ---- state ----
    state = {"idx": 0, "data": None, "suppress_events": False}
    handles = {}

    def clear_handles(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def current_delta_world():
        dt = np.array([dx.value, dy.value, dz.value], dtype=np.float64)
        Rd = euler_to_rotmat(droll.value, dpitch.value, dyaw.value)
        return dt, Rd

    def current_alpha_curve() -> np.ndarray:
        d = state["data"]
        if d is None or d["obj_poses"] is None:
            return np.zeros(1)
        T = len(d["obj_poses"])
        return compute_alpha_per_frame(
            T,
            int(edit_start.value), int(edit_end.value),
            int(ramp_in.value), int(ramp_out.value),
        )

    # ---- draw helpers ----
    def edited_pose_at(frame_idx: int):
        """Return (R_cam, t_cam) for given frame after applying alpha-edit."""
        d = state["data"]
        pose = np.asarray(d["obj_poses"][frame_idx])
        if pose.shape == (4, 4):
            R_cam, t_cam = pose[:3, :3], pose[:3, 3]
        else:
            R_cam = Rot.from_quat(pose[:4]).as_matrix()
            t_cam = pose[4:7]
        alpha = current_alpha_curve()
        a = float(alpha[frame_idx]) if frame_idx < len(alpha) else 0.0
        if a <= 0.0:
            return R_cam, t_cam
        dt_w, R_delta_w = current_delta_world()
        return apply_edit_world(R_cam, t_cam, dt_w, R_delta_w, a)

    def draw_obj_mesh(frame_idx: int, R_cam: np.ndarray, t_cam: np.ndarray,
                       color, prefix: str):
        clear_handles(prefix)
        d = state["data"]
        if d["obj_vertices"] is None or d["obj_faces"] is None:
            return
        verts_cam = (R_cam @ d["obj_vertices"].T).T + t_cam
        verts_disp = flip_z(verts_cam)
        handles[f"{prefix}/mesh"] = server.scene.add_mesh_simple(
            f"{prefix}/mesh", vertices=verts_disp.astype(np.float32),
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
        centers = []
        colors  = []
        alpha = current_alpha_curve()
        for t in range(T):
            pose = np.asarray(d["obj_poses"][t])
            if pose.shape == (4, 4):
                R_cam, t_cam = pose[:3, :3], pose[:3, 3]
            else:
                R_cam = Rot.from_quat(pose[:4]).as_matrix()
                t_cam = pose[4:7]
            if alpha[t] > 0.0:
                R_cam, t_cam = apply_edit_world(R_cam, t_cam,
                                                  *current_delta_world(),
                                                  float(alpha[t]))
            c_cam = R_cam @ centroid_local + t_cam
            centers.append(flip_z(c_cam))
            # Edited = orange-ish, untouched = yellow-ish
            base = np.array([200, 200, 80], dtype=np.float64)
            edit = np.array([255, 140, 50], dtype=np.float64)
            colors.append(tuple((base + alpha[t] * (edit - base)).astype(int)))

        if len(centers) < 2:
            return
        centers = np.array(centers)
        handles[f"{prefix}/line"] = server.scene.add_spline_catmull_rom(
            f"{prefix}/line", positions=centers, color=(200, 200, 80), line_width=2.0,
        )
        r = trace_radius.value
        n_dots = min(60, len(centers))
        step = max(1, len(centers) // n_dots)
        for i in range(0, len(centers), step):
            kk = f"{prefix}/dot_{i}"
            handles[kk] = server.scene.add_icosphere(
                kk, radius=r, position=centers[i], color=colors[i],
            )

    def draw_hand(side: str, frame_idx: int):
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
                    pos = flip_z(arr[frame_idx])
                    # If "also translate hand" is on, add dt_world * alpha
                    if shift_hand_cb.value:
                        alpha = current_alpha_curve()
                        a = float(alpha[frame_idx]) if frame_idx < len(alpha) else 0.0
                        if a > 0:
                            dt_w, _ = current_delta_world()
                            pos = pos + a * dt_w
                    positions[name] = pos
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
        R_a, pos_a, scale = aux_pose_to_R_t(d["aux_info"])
        verts = (R_a @ (d["aux_verts"] * scale).T).T + pos_a
        handles[f"{prefix}/mesh"] = server.scene.add_mesh_simple(
            f"{prefix}/mesh", vertices=verts.astype(np.float32),
            faces=d["aux_faces"], color=AUX_COLOR,
        )

    def redraw():
        d = state["data"]
        if d is None or d["obj_poses"] is None:
            clear_handles()
            return
        T = len(d["obj_poses"])
        f = int(frame_slider.value)
        f = min(max(f, 0), T - 1)

        # current frame: edited
        if show_object_cb.value:
            R_cam, t_cam = edited_pose_at(f)
            draw_obj_mesh(f, R_cam, t_cam, OBJ_COLOR, "/object_cur")
        else:
            clear_handles("/object_cur")

        # original at current frame (faint)
        if show_orig_cb.value:
            pose = np.asarray(d["obj_poses"][f])
            if pose.shape == (4, 4):
                R_cam_o, t_cam_o = pose[:3, :3], pose[:3, 3]
            else:
                R_cam_o = Rot.from_quat(pose[:4]).as_matrix()
                t_cam_o = pose[4:7]
            draw_obj_mesh(f, R_cam_o, t_cam_o, ORIG_GHOST_COLOR, "/object_orig")
        else:
            clear_handles("/object_orig")

        # range markers — edit_start / edit_end frames (in their EDITED pose)
        if show_start_ghost_cb.value:
            es = int(edit_start.value)
            es = min(max(es, 0), T - 1)
            R_cam_s, t_cam_s = edited_pose_at(es)
            draw_obj_mesh(es, R_cam_s, t_cam_s, START_MARK_COLOR, "/object_start_ghost")
        else:
            clear_handles("/object_start_ghost")

        if show_end_ghost_cb.value:
            ee = int(edit_end.value)
            ee = min(max(ee, 0), T - 1)
            R_cam_e, t_cam_e = edited_pose_at(ee)
            draw_obj_mesh(ee, R_cam_e, t_cam_e, END_MARK_COLOR, "/object_end_ghost")
        else:
            clear_handles("/object_end_ghost")

        draw_trace()
        draw_hand("left",  f)
        draw_hand("right", f)
        draw_aux()

    # ---- sequence switching ----
    def load_idx(idx: int):
        idx = int(np.clip(idx, 0, len(sequences) - 1))
        name, pkl_path = sequences[idx]
        print(f"[LOAD] {idx + 1}/{len(sequences)}: {name}  ({pkl_path})")
        state["data"] = load_sequence_data(pkl_path)
        state["idx"] = idx
        state["suppress_events"] = True
        try:
            nf = state["data"]["num_frames"]
            T = max(nf, 1)
            frame_slider.max = T - 1
            frame_slider.value = 0
            edit_start.max = T - 1
            edit_end.max   = T - 1
            # Default: edit the LAST 10 frames (user's main use case)
            default_start = max(0, T - 10)
            edit_start.value = default_start
            edit_end.value   = T - 1
            ramp_in.value = min(3, max(0, T - default_start - 1))
            ramp_out.value = 0
            for w in (dx, dy, dz, droll, dpitch, dyaw):
                w.value = 0.0

            has_obj = (state["data"]["obj_poses"] is not None
                       and state["data"]["obj_vertices"] is not None)
            show_object_cb.value = has_obj
            seq_dropdown.value = seq_labels[idx]
            aux_present = state["data"]["aux_info"] is not None
            seq_status.content = (
                f"**{idx + 1} / {len(sequences)}** — `{name}` "
                f"(T={nf}, obj={'yes' if has_obj else 'no'}, "
                f"aux={'yes' if aux_present else 'no'})  "
                f"defaults: edit f{default_start}..f{T-1}"
            )
        finally:
            state["suppress_events"] = False
        clear_handles()
        redraw()

    def on_any_update(_e=None):
        if state["suppress_events"]:
            return
        redraw()

    redraw_widgets = [
        frame_slider,
        show_object_cb, show_orig_cb, show_start_ghost_cb, show_end_ghost_cb,
        show_trace_cb, show_hand_cb, show_bones_cb, show_aux_cb,
        joint_radius, trace_radius,
        edit_start, edit_end, ramp_in, ramp_out,
        dx, dy, dz, droll, dpitch, dyaw, shift_hand_cb,
    ]
    for w in redraw_widgets:
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
        d = state["data"]
        if d is None or d["obj_poses"] is None:
            status_md.content = "_no obj_poses to edit_"
            return
        es = int(edit_start.value)
        ee = int(edit_end.value)
        if es > ee:
            status_md.content = "_edit_start > edit_end; nothing to do_"
            return
        dt_w, R_delta_w = current_delta_world()
        if np.allclose(dt_w, 0.0) and np.allclose(
            Rot.from_matrix(R_delta_w).as_rotvec(), 0.0
        ):
            status_md.content = "_all deltas zero — nothing to bake_"
            return

        alpha = current_alpha_curve()
        n_active = int((alpha > 0).sum())
        status_md.content = (
            f"_baking {n_active} frames "
            f"[{es}..{ee}] (ramp_in={int(ramp_in.value)}, "
            f"ramp_out={int(ramp_out.value)}) ..._"
        )

        ok, msg = bake_into_pkl(
            pkl_path=d["pkl_path"],
            alpha_per_frame=alpha,
            dt_world=dt_w,
            R_delta_world=R_delta_w,
            shift_hand=shift_hand_cb.value,
            out_filename="mano_joints_corrected.pkl",
        )
        print("[SAVE]", msg)

        if ok:
            # Reload from corrected.pkl so subsequent edits stack
            name, _ = sequences[state["idx"]]
            new_pkl = os.path.join(d["seq_dir"], "mano_joints_corrected.pkl")
            sequences[state["idx"]] = (name, new_pkl)
            seq_labels[state["idx"]] = seq_label(name, new_pkl)
            state["suppress_events"] = True
            try:
                seq_dropdown.options = seq_labels
                seq_dropdown.value = seq_labels[state["idx"]]
                for w in (dx, dy, dz, droll, dpitch, dyaw):
                    w.value = 0.0
            finally:
                state["suppress_events"] = False
            # Reload data so source is now corrected.pkl
            state["data"] = load_sequence_data(new_pkl)
            redraw()
            status_md.content = (
                f"**Saved.** Baked {n_active} frames. "
                f"Source now = mano_joints_corrected.pkl.\n\n"
                f"```\n{msg}\n```"
            )
        else:
            status_md.content = f"**FAIL**: {msg}"

    @reset_btn.on_click
    def _on_reset(_e):
        state["suppress_events"] = True
        try:
            for w in (dx, dy, dz, droll, dpitch, dyaw):
                w.value = 0.0
            d = state["data"]
            if d is not None and d["obj_poses"] is not None:
                T = len(d["obj_poses"])
                edit_start.value = max(0, T - 10)
                edit_end.value = T - 1
                ramp_in.value = min(3, max(0, 9))
                ramp_out.value = 0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "_reset_"

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
