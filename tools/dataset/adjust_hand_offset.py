#!/usr/bin/env python3
"""
Manually adjust hand-object offset in mano_joints.pkl via viser sliders.

Use case: raw annotation has a fixed translation bias between hand and object.
This tool loads a trajectory (or a whole task folder), shows hand + object,
lets you drag x/y/z sliders (per hand side) to find a constant offset that
aligns them, then bakes the offset into the pkl (original backed up as *.bak).

Usage (single file):
    python tools/dataset/adjust_hand_offset.py \
        --pkl data/robotool_batch/0416_grasp/cube_small_1/mano_joints.pkl

Usage (task folder — browse each sequence with a dropdown / Prev / Next):
    python tools/dataset/adjust_hand_offset.py \
        --task_dir data/robotool_batch/0416_grasp

Optional:
    --port 8080
    --also_update_optimized    Also apply offset to mano_joints_optimized.pkl
                               and mano_joints_corrected.pkl in the same dir.
    --slider_range 0.15        Offset slider range in meters.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from typing import List, Tuple

import numpy as np
import trimesh
import viser

# ---- Hand skeleton definition (matches vis_sequence.py) ----

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

# Fields that are 3D positions in the stored (camera) frame and should be shifted
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

EDITED_MARK = "✓"


def flip_z(v: np.ndarray) -> np.ndarray:
    """Flip y,z axes (camera frame z-down → display frame z-up). Involutive."""
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


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


def seq_label(name: str, pkl_path: str) -> str:
    """Dropdown label; mark edited sequences (those with a .bak already)."""
    return f"{EDITED_MARK} {name}" if os.path.exists(pkl_path + ".bak") else f"  {name}"


def load_sequence_data(pkl_path: str):
    """Load pkl + meta + object mesh for a sequence. Returns a dict of state."""
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
        # pkl_path is .../data/robotool_batch/{task}/{exp}/mano_joints.pkl
        data_root = os.path.dirname(os.path.dirname(seq_dir))
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
        from scipy.spatial.transform import Rotation as Rot
        R = Rot.from_quat(pose[:4]).as_matrix()
        t = pose[4:7]
    else:
        raise ValueError(f"Unexpected object pose shape: {pose.shape}")
    return (R @ vertices.T).T + t


def apply_offset_to_pkl(pkl_path: str, offsets_display: dict, backup_suffix: str = ".bak"):
    """Bake a per-side display-frame offset into the pkl (stored in camera frame).

    display_offset in z-up frame; stored arrays are in z-down frame.
    Stored delta = (dx, -dy, -dz).
    """
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


def main():
    parser = argparse.ArgumentParser(description="Manually adjust hand-object offset via viser.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl", type=str,
                       help="Path to a single mano_joints.pkl to edit")
    group.add_argument("--task_dir", type=str,
                       help="Path to a task folder containing exp subdirs, e.g. data/robotool_batch/0416_grasp")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--also_update_optimized", action="store_true",
                        help="Also apply the same offset to mano_joints_optimized.pkl / _corrected.pkl")
    parser.add_argument("--slider_range", type=float, default=0.15,
                        help="Slider range for offset in meters (default 0.15)")
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
            print(f"[ERROR] No sequences (subdir with mano_joints.pkl + meta.json) found in {args.task_dir}")
            sys.exit(1)
        print(f"[INFO] Found {len(sequences)} sequences in {args.task_dir}:")
        for n, p in sequences:
            edited = " (edited)" if os.path.exists(p + ".bak") else ""
            print(f"  - {n}{edited}")

    # ---- Viser server ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser at http://localhost:{args.port}")
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    # ---- GUI (created once, updated on sequence switch) ----
    seq_labels = [seq_label(n, p) for n, p in sequences]
    seq_dropdown = server.gui.add_dropdown("Sequence", options=seq_labels,
                                            initial_value=seq_labels[0])
    prev_btn = server.gui.add_button("◀ Prev")
    next_btn = server.gui.add_button("Next ▶")
    seq_status = server.gui.add_markdown(f"**1 / {len(sequences)}**")

    server.gui.add_markdown("---")
    # frame_slider.max will be updated on sequence switch
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    show_object_cb = server.gui.add_checkbox("Show Object", initial_value=True)
    show_bones_cb = server.gui.add_checkbox("Show Bones", initial_value=True)
    joint_radius = server.gui.add_slider("Joint Radius", min=0.001, max=0.015,
                                          step=0.001, initial_value=0.005)

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Hand Offset (display frame, meters)**")
    R_ = args.slider_range
    STEP = 0.001

    # Always create both sides' controls; show/hide driven by per-sequence presence.
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

    server.gui.add_markdown("---")
    save_btn = server.gui.add_button("Save (bake offset into pkl)")
    reset_btn = server.gui.add_button("Reset sliders to 0")
    status_md = server.gui.add_markdown("_ready_")

    # ---- State container (mutable; updated on sequence switch) ----
    state = {"idx": 0, "data": None, "suppress_events": False}
    handles = {}

    def clear_handles(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def current_offset(side: str) -> np.ndarray:
        sx, sy, sz = off_sliders[side]
        return np.array([sx.value, sy.value, sz.value], dtype=np.float32)

    def draw_hand(side: str, frame_idx: int):
        prefix = f"/{side}_hand"
        clear_handles(prefix)
        d = state["data"]
        present = d["has_left"] if side == "left" else d["has_right"]
        if not present or not show_side[side].value:
            return
        hand_data = d["mano_data"].get(side, {})
        offset = current_offset(side)
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

    def draw_object(frame_idx: int):
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

    def redraw():
        if state["data"] is None:
            return
        f = int(frame_slider.value)
        draw_hand("left", f)
        draw_hand("right", f)
        draw_object(f)

    def load_idx(idx: int):
        """Switch to sequence `idx`: load data, reset sliders, reconfigure widgets."""
        idx = int(np.clip(idx, 0, len(sequences) - 1))
        name, pkl_path = sequences[idx]
        print(f"[LOAD] {idx + 1}/{len(sequences)}: {name}  ({pkl_path})")
        state["data"] = load_sequence_data(pkl_path)
        state["idx"] = idx

        # Suppress on_update side effects while we bulk-update widget values.
        state["suppress_events"] = True
        try:
            # Frame slider
            nf = state["data"]["num_frames"]
            frame_slider.max = max(nf - 1, 0)
            frame_slider.value = 0

            # Side visibility reflects data presence
            show_left_cb.value = state["data"]["has_left"]
            show_right_cb.value = state["data"]["has_right"]

            # Object toggle only if object available
            has_obj = (state["data"]["obj_poses"] is not None
                       and state["data"]["obj_vertices"] is not None)
            show_object_cb.value = has_obj

            # Reset offset sliders
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0

            # Keep dropdown consistent when navigated via Prev/Next
            seq_dropdown.value = seq_labels[idx]

            seq_status.content = (
                f"**{idx + 1} / {len(sequences)}** — `{name}`  "
                f"({nf} frames, L={state['data']['has_left']}, R={state['data']['has_right']})"
            )
        finally:
            state["suppress_events"] = False

        clear_handles()
        redraw()

    def on_any_update(_event=None):
        if state["suppress_events"]:
            return
        redraw()

    # Wire events
    for widget in [frame_slider, show_object_cb, show_bones_cb, joint_radius,
                   show_left_cb, show_right_cb, lx, ly, lz, rx, ry, rz]:
        widget.on_update(on_any_update)

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
        offsets = {side: current_offset(side) for side in off_sliders}
        if all(np.allclose(v, 0.0) for v in offsets.values()):
            status_md.content = "_no change: all offsets are zero_"
            return

        targets = [d["pkl_path"]]
        if args.also_update_optimized:
            for name in ("mano_joints_optimized.pkl", "mano_joints_corrected.pkl"):
                p = os.path.join(d["seq_dir"], name)
                if os.path.exists(p):
                    targets.append(p)

        lines = []
        for p in targets:
            ok, msg = apply_offset_to_pkl(p, offsets)
            lines.append(("OK  " if ok else "FAIL") + " " + msg)
            print("[SAVE]", msg)

        # Mark this sequence as edited in the dropdown label
        idx = state["idx"]
        name, pkl_path = sequences[idx]
        new_label = seq_label(name, pkl_path)
        if new_label != seq_labels[idx]:
            seq_labels[idx] = new_label
            state["suppress_events"] = True
            try:
                seq_dropdown.options = seq_labels
                seq_dropdown.value = new_label
            finally:
                state["suppress_events"] = False

        # Reload the just-saved pkl so the view shows the baked state, and zero sliders.
        state["data"] = load_sequence_data(pkl_path)
        state["suppress_events"] = True
        try:
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0
        finally:
            state["suppress_events"] = False
        redraw()
        status_md.content = "**Saved.**\n\n" + "\n\n".join(lines)

    @reset_btn.on_click
    def _on_reset(_event):
        state["suppress_events"] = True
        try:
            for sx, sy, sz in off_sliders.values():
                sx.value = 0.0
                sy.value = 0.0
                sz.value = 0.0
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
