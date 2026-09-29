#!/usr/bin/env python3
"""
Browse every sequence in robotool_batch, translate the hand and/or the object
(whole trajectory or one frame range), and save the result under a NEW name.

Differs from the neighbouring scripts in four ways, which is the whole point:

  - `adjust_object_offset.py` takes --pkl or --task_dir and edits one task's
    sequences; this walks the entire data root, so you pick from all of them in
    one dropdown (with a substring filter, because there are ~170).
  - The offset target is a choice, not a fixed role plus an "also shift" tick.
    An Object checkbox and a Hand dropdown (None/Left/Right/Both, rebuilt per
    sequence so an absent hand cannot be picked); shifting only the hand is a
    first-class operation here.
  - A frame range can be selected, with cosine ramps at both ends so the edited
    segment blends into its neighbours instead of stepping.
  - Nothing is overwritten. Save copies the sequence directory to a new name and
    writes the edited pkl there, so the original stays byte-identical and there
    is no .bak to keep track of.

TRANSLATION ONLY. Rotation is deliberately absent: rotating the hand about an
object pivot without re-solving the finger joints breaks the grasp, and
`edit_frame_range_pose.py` already covers object rotation. Adding a rotation
slider here would invite exactly the edit that silently corrupts a demo.

FRAMES ARE Z-UP. Sliders are in the frame you see in viser. The pkl stores
camera frame (z-down), so a display offset (dx, dy, dz) is written as
(dx, -dy, -dz). Getting this backwards is an easy mistake, so the conversion
lives in exactly one function, `disp_to_stored`.

Usage:
    python tools/dataset/offset_editor.py
    python tools/dataset/offset_editor.py --filter squeegee
    python tools/dataset/offset_editor.py \\
        --data_root data/robotool_batch --port 8081
"""

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as Rot


# ---- Hand skeleton (matches adjust_object_offset.py) ----

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

# "Wrist only" edits must touch BOTH: they hold the same numbers (verified
# identical on 0420_manip/squeegee_1), the viewer draws `wrist`, and
# robotool_batch_dataset_dexhand.py:421 reads `wrist_translation`. Moving one
# and not the other makes the edit either invisible or ineffective.
WRIST_FIELDS = ("wrist", "wrist_translation")

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
GHOST_COLOR = (90, 90, 100)
AUX_COLOR = (110, 220, 110)
AUX_SIDECAR_NAME = "aux_object.json"

PKL_PREFERENCE = ("mano_joints_corrected.pkl",
                  "mano_joints_optimized.pkl",
                  "mano_joints.pkl")


# ----------------------------- coord helpers -----------------------------

def flip_z(v: np.ndarray) -> np.ndarray:
    """Flip y,z (camera frame z-down <-> display frame z-up). Involutive."""
    out = np.array(v, dtype=np.float64, copy=True)
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


def disp_to_stored(offset_disp: np.ndarray) -> np.ndarray:
    """The ONLY place the display->storage sign flip is written."""
    o = np.asarray(offset_disp, dtype=np.float64)
    return np.array([o[0], -o[1], -o[2]], dtype=np.float64)


# ----------------------------- discovery ---------------------------------

def pick_pkl(seq_dir: str) -> Optional[str]:
    for name in PKL_PREFERENCE:
        p = os.path.join(seq_dir, name)
        if os.path.exists(p):
            return p
    return None


def discover_all_sequences(data_root: str) -> List[Tuple[str, str, str]]:
    """Walk data_root/<task>/<seq>/meta.json. Returns (task, seq, pkl_path).

    Skips the sibling directories that are not tasks (models/, visualizer/,
    ...) implicitly: they contain no <sub>/meta.json.
    """
    out = []
    for task in sorted(os.listdir(data_root)):
        tdir = os.path.join(data_root, task)
        if not os.path.isdir(tdir) or task in ("models", "visualizer"):
            continue
        for seq in sorted(os.listdir(tdir)):
            sdir = os.path.join(tdir, seq)
            if not os.path.isdir(sdir):
                continue
            if not os.path.isfile(os.path.join(sdir, "meta.json")):
                continue
            pkl = pick_pkl(sdir)
            if pkl is not None:
                out.append((task, seq, pkl))
    return out


def seq_label(task: str, seq: str, pkl_path: str) -> str:
    short = {"mano_joints_corrected.pkl": "corr",
             "mano_joints_optimized.pkl": "opt",
             "mano_joints.pkl": "raw"}.get(os.path.basename(pkl_path), "?")
    return f"{task}/{seq}  [{short}]"


def derive_data_root(pkl_path: str) -> str:
    """.../data/robotool_batch/<task>/<seq>/x.pkl -> .../data/robotool_batch"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(pkl_path))))


# ----------------------------- loading -----------------------------------

# Set from --data_root. A sequence saved to an absolute path outside the data
# tree has no models/ three levels up, so mesh lookup would silently fail and
# the copy would be unviewable in this very tool. Fall back to the CLI root.
_FALLBACK_ROOT: Optional[str] = None


def load_mesh(data_root: str, obj_id: str):
    roots = [data_root]
    if _FALLBACK_ROOT and os.path.abspath(_FALLBACK_ROOT) != os.path.abspath(data_root):
        roots.append(_FALLBACK_ROOT)
    for r in roots:
        p = os.path.join(r, "models", obj_id, "cleaned_mesh_10000.obj")
        if os.path.exists(p):
            m = trimesh.load(p, process=False, force="mesh")
            return m.vertices.copy().astype(np.float32), m.faces.copy()
    return None, None


def load_sequence_data(pkl_path: str) -> dict:
    seq_dir = os.path.dirname(os.path.abspath(pkl_path))
    with open(pkl_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(os.path.join(seq_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    data_root = derive_data_root(pkl_path)

    obj_id = (meta.get("object_ids") or [None])[0]
    obj_verts, obj_faces = (None, None)
    if obj_id:
        obj_verts, obj_faces = load_mesh(data_root, obj_id)
        if obj_verts is None:
            print(f"[WARN] mesh not found for object '{obj_id}'")

    od = mano_data.get("original_data", {})
    obj_poses = None
    tp = od.get("tool_object_pose")
    if tp is not None and np.asarray(tp).size > 0:
        obj_poses = np.asarray(tp)

    tgt = od.get("target_object_pose")
    has_target = tgt is not None and np.asarray(tgt).size > 0

    sides = meta.get("mano_sides", [])
    def _present(s):
        return (s in sides and s in mano_data
                and len(np.asarray(mano_data[s].get("wrist", []))) > 0)

    aux_info = None
    aux_path = os.path.join(seq_dir, AUX_SIDECAR_NAME)
    if os.path.exists(aux_path):
        with open(aux_path, "r") as f:
            aux_info = json.load(f)
    aux_verts, aux_faces = (None, None)
    if aux_info and aux_info.get("aux_obj_id"):
        aux_verts, aux_faces = load_mesh(data_root, aux_info["aux_obj_id"])

    # Frame count: trust the arrays over meta, which can be wrong.
    n_meta = int(meta.get("num_frames", 0))
    n_arr = 0
    if obj_poses is not None:
        n_arr = max(n_arr, len(obj_poses))
    for s in ("left", "right"):
        if _present(s):
            n_arr = max(n_arr, len(np.asarray(mano_data[s]["wrist"])))
    num_frames = n_arr if n_arr > 0 else n_meta

    return {
        "pkl_path": pkl_path,
        "pkl_name": os.path.basename(pkl_path),
        "seq_dir": seq_dir,
        "mano_data": mano_data,
        "meta": meta,
        "obj_id": obj_id,
        "obj_vertices": obj_verts,
        "obj_faces": obj_faces,
        "obj_poses": obj_poses,
        "has_target_obj": has_target,
        "num_frames": num_frames,
        "num_frames_meta": n_meta,
        "has_left": _present("left"),
        "has_right": _present("right"),
        "aux_info": aux_info,
        "aux_verts": aux_verts,
        "aux_faces": aux_faces,
    }


# ----------------------------- alpha ramp --------------------------------

def compute_alpha(T: int, whole: bool, start: int, end: int,
                  ramp_in: int, ramp_out: int) -> np.ndarray:
    """(T,) weights in [0,1] scaling the offset per frame.

    Whole-trajectory mode is a flat 1.0 — a rigid shift of everything, no ramp,
    because there are no neighbouring frames to blend with.

    Range mode uses a raised-cosine rather than a linear ramp: linear is
    continuous in position but not in velocity, and a velocity step at the seam
    is exactly what the retarget's arm-velocity term punishes.
    """
    a = np.zeros(int(T), dtype=np.float64)
    if T <= 0:
        return a
    if whole:
        a[:] = 1.0
        return a
    s = int(np.clip(start, 0, T - 1))
    e = int(np.clip(end, s, T - 1))
    for f in range(s, e + 1):
        w_in = 1.0 if ramp_in <= 0 else min(1.0, (f - s + 1) / float(ramp_in))
        w_out = 1.0 if ramp_out <= 0 else min(1.0, (e - f + 1) / float(ramp_out))
        w = min(w_in, w_out)
        a[f] = 0.5 - 0.5 * np.cos(np.pi * w)      # raised cosine
    return a


# ----------------------------- edit application --------------------------

def apply_edits(mano_data: dict, edits: List[dict]) -> Tuple[dict, List[str]]:
    """Apply a list of staged edits in order onto one deep copy.

    Translations commute, so order does not change the result — but they are
    applied in the order staged anyway, because the log then reads the same way
    the user built it, which is what makes a review pass meaningful.

    Each edit: {"offset": (3,), "alpha": (T,), "object": bool,
                "left": bool, "right": bool}
    """
    data = copy.deepcopy(mano_data)
    notes: List[str] = []
    for i, e in enumerate(edits):
        data, sub = _apply_one(data, np.asarray(e["offset"], dtype=np.float64),
                               np.asarray(e["alpha"], dtype=np.float64),
                               bool(e["object"]), bool(e["left"]), bool(e["right"]),
                               e.get("fields", "all"))
        notes.append(f"[{i + 1}] " + "; ".join(sub))
    return data, notes


def apply_offset(mano_data: dict,
                 offset_disp: np.ndarray,
                 alpha: np.ndarray,
                 do_object: bool,
                 do_left: bool,
                 do_right: bool,
                 hand_fields: str = "all") -> Tuple[dict, List[str]]:
    """Return a deep-copied pkl dict with the offset baked in, plus a log.

    Deep copy rather than in-place so the loaded sequence stays pristine and the
    viewer keeps showing the original after a save.
    """
    return _apply_one(copy.deepcopy(mano_data), offset_disp, alpha,
                      do_object, do_left, do_right, hand_fields)


def _apply_one(data: dict,
               offset_disp: np.ndarray,
               alpha: np.ndarray,
               do_object: bool,
               do_left: bool,
               do_right: bool,
               hand_fields: str = "all") -> Tuple[dict, List[str]]:
    """In-place on `data` (caller owns the copy). Returns (data, log).

    hand_fields: "all"   -> every joint plus wrist_translation (rigid hand)
                 "wrist" -> only WRIST_FIELDS, fingers stay where they are
    """
    off = disp_to_stored(offset_disp)
    notes: List[str] = []

    if do_object:
        od = data.get("original_data", {})
        tp = od.get("tool_object_pose")
        if tp is None or np.asarray(tp).size == 0:
            notes.append("object: no tool_object_pose, skipped")
        else:
            arr = np.asarray(tp).astype(np.float64)
            T = min(len(arr), len(alpha))
            if arr.ndim == 3 and arr.shape[1:] == (4, 4):
                arr[:T, :3, 3] += alpha[:T, None] * off[None, :]
            elif arr.ndim == 2 and arr.shape[1] == 7:
                # [qx, qy, qz, qw, tx, ty, tz]
                arr[:T, 4:7] += alpha[:T, None] * off[None, :]
            else:
                notes.append(f"object: unexpected shape {arr.shape}, skipped")
                arr = None
            if arr is not None:
                od["tool_object_pose"] = arr
                data["original_data"] = od
                notes.append(f"object: tool_object_pose ({T} frames)")

    for side, do in (("left", do_left), ("right", do_right)):
        if not do:
            continue
        if side not in data or not isinstance(data[side], dict):
            notes.append(f"{side} hand: absent, skipped")
            continue
        n_fields = 0
        fields = WRIST_FIELDS if hand_fields == "wrist" else POSITION_FIELDS
        for field in fields:
            if field not in data[side]:
                continue
            arr = np.asarray(data[side][field]).astype(np.float64)
            if arr.ndim == 2 and arr.shape[-1] == 3:
                T = min(len(arr), len(alpha))
                arr[:T] += alpha[:T, None] * off[None, :]
                data[side][field] = arr.astype(np.float32)
                n_fields += 1
        notes.append(f"{side} hand: {n_fields} field(s)"
                     + (" [wrist only]" if hand_fields == "wrist" else ""))

    return data, notes


# ----------------------------- save --------------------------------------

def resolve_dst_dir(data_root: str, src_task: str, raw: str) -> str:
    """Interpret the 'Save as' box.

      squeegee_1_fixed        -> data_root/<src_task>/squeegee_1_fixed
      0420_manip/sq_fixed     -> data_root/0420_manip/sq_fixed
      /abs/path/whatever      -> exactly that
    """
    raw = raw.strip().rstrip("/")
    if not raw:
        raise ValueError("empty name")
    if os.path.isabs(raw):
        return raw
    parts = raw.split("/")
    if len(parts) == 1:
        return os.path.join(data_root, src_task, parts[0])
    if len(parts) == 2:
        return os.path.join(data_root, parts[0], parts[1])
    raise ValueError(f"name has too many path components: {raw!r}")


def save_as_new_sequence(src_dir: str,
                         dst_dir: str,
                         pkl_name: str,
                         new_pkl_data: dict,
                         provenance: dict,
                         overwrite: bool) -> Tuple[bool, str]:
    """Copy the sequence directory to dst_dir, then write the edited pkl there.

    The whole directory is copied — meta.json, the other pkl variants, the aux
    sidecar — so the result is a self-contained sequence the existing loaders
    can read without knowing it was derived. Only the pkl that was loaded gets
    rewritten; the other variants are copied verbatim and are therefore STALE
    with respect to the edit, which the provenance block records.
    """
    if os.path.abspath(src_dir) == os.path.abspath(dst_dir):
        return False, "destination is the source; this tool never overwrites in place"
    if os.path.exists(dst_dir):
        if not overwrite:
            return False, f"exists: {dst_dir}  (tick 'Overwrite' to replace)"
        shutil.rmtree(dst_dir)

    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns("*.bak", "*.bak.*", "__pycache__"))

    with open(os.path.join(dst_dir, pkl_name), "wb") as f:
        pickle.dump(new_pkl_data, f)

    stale = [n for n in PKL_PREFERENCE
             if n != pkl_name and os.path.exists(os.path.join(dst_dir, n))]

    meta_path = os.path.join(dst_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["task"] = os.path.basename(os.path.dirname(dst_dir))
        meta["sequence"] = os.path.basename(dst_dir)
        hist = meta.get("edit_history", [])
        rec = dict(provenance)
        rec["stale_pkls"] = stale
        hist.append(rec)
        meta["edit_history"] = hist
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)

    msg = f"wrote {dst_dir}  (edited {pkl_name})"
    if stale:
        msg += f"  [NOT edited, copied as-is: {', '.join(stale)}]"
    return True, msg


# ----------------------------- viz helpers -------------------------------

def obj_verts_at(vertices: np.ndarray, pose) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        R, t = pose[:3, :3], pose[:3, 3]
    elif pose.shape == (7,):
        R, t = Rot.from_quat(pose[:4]).as_matrix(), pose[4:7]
    else:
        raise ValueError(f"unexpected object pose shape: {pose.shape}")
    return (R @ vertices.T).T + t


def aux_R_t_scale(aux_info: dict):
    pos = np.asarray(aux_info.get("aux_pos_world", [0, 0, 0]), dtype=np.float64)
    q = aux_info.get("aux_quat_wxyz_world", [1, 0, 0, 0])
    R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R, pos, float(aux_info.get("aux_scale", 1.0))


# ----------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Browse all robotool_batch sequences; offset hand and/or "
                    "object over the whole trajectory or a frame range; save "
                    "under a new name.")
    ap.add_argument("--data_root", type=str, default="data/robotool_batch")
    ap.add_argument("--filter", type=str, default="",
                    help="Initial substring filter on 'task/seq'")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--slider_range", type=float, default=0.5,
                    help="Offset slider half-range in meters (default 0.5)")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    global _FALLBACK_ROOT
    _FALLBACK_ROOT = data_root
    if not os.path.isdir(data_root):
        print(f"[ERROR] data_root not found: {data_root}")
        sys.exit(1)

    all_seqs = discover_all_sequences(data_root)
    if not all_seqs:
        print(f"[ERROR] no sequences under {data_root}")
        sys.exit(1)
    print(f"[INFO] {len(all_seqs)} sequences under {data_root}")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    # viser silently moves to the next free port when --port is taken, so print
    # what it actually bound rather than what was asked for.
    actual_port = server.get_port()
    print(f"[INFO] viser at http://localhost:{actual_port}"
          + ("" if actual_port == args.port else f"  (asked for {args.port})"))
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    # ---------------- GUI: selection ----------------
    server.gui.add_markdown("### Sequence")
    filter_box = server.gui.add_text("Filter (substring)", initial_value=args.filter)
    seq_dropdown = server.gui.add_dropdown("Sequence", options=["(none)"],
                                           initial_value="(none)")
    with server.gui.add_folder("Nav"):
        prev_btn = server.gui.add_button("◀ Prev")
        next_btn = server.gui.add_button("Next ▶")
    seq_status = server.gui.add_markdown("_loading_")

    # ---------------- GUI: playback ----------------
    server.gui.add_markdown("### View")
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)
    with server.gui.add_folder("Show"):
        show_obj_cb = server.gui.add_checkbox("Object", initial_value=True)
        show_hand_cb = server.gui.add_checkbox("Hand", initial_value=True)
        show_bones_cb = server.gui.add_checkbox("Bones", initial_value=True)
        show_aux_cb = server.gui.add_checkbox("Aux", initial_value=True)
        show_ghost_cb = server.gui.add_checkbox("Ghost (un-edited)", initial_value=True)
        joint_radius = server.gui.add_slider("Joint radius", min=0.001, max=0.015,
                                             step=0.001, initial_value=0.005)

    # ---------------- GUI: what moves ----------------
    server.gui.add_markdown("### Apply offset to")
    tgt_obj_cb = server.gui.add_checkbox("Object", initial_value=True)
    # A dropdown rather than one checkbox per side: the options are rebuilt per
    # sequence to list only the hands that sequence actually has, so an absent
    # hand can never be selected and then silently do nothing on save.
    hand_target = server.gui.add_dropdown(
        "Hand", options=["None", "Left", "Right", "Both"], initial_value="None")
    hand_fields_dd = server.gui.add_dropdown(
        "Hand fields", options=["All joints", "Wrist only"],
        initial_value="All joints")
    server.gui.add_markdown(
        "_**Wrist only** moves the wrist keypoint and leaves every finger where "
        "it is — the hand shape changes. Note the retarget takes "
        "`wrist - 0.25*(middle_proximal - wrist)`, so a wrist-only shift of d "
        "moves its target wrist by **1.25 d**._")
    target_hint = server.gui.add_markdown("")

    # ---------------- GUI: range ----------------
    server.gui.add_markdown("### Frame range")
    range_mode = server.gui.add_dropdown(
        "Mode", options=["Whole trajectory", "Frame range"],
        initial_value="Whole trajectory")
    start_slider = server.gui.add_slider("Start", min=0, max=1, step=1, initial_value=0)
    end_slider = server.gui.add_slider("End", min=0, max=1, step=1, initial_value=1)
    ramp_in_slider = server.gui.add_slider("Ramp in (frames)", min=0, max=60,
                                           step=1, initial_value=10)
    ramp_out_slider = server.gui.add_slider("Ramp out (frames)", min=0, max=60,
                                            step=1, initial_value=10)
    set_start_btn = server.gui.add_button("Start = current frame")
    set_end_btn = server.gui.add_button("End = current frame")
    range_status = server.gui.add_markdown("_whole trajectory_")

    # ---------------- GUI: offset ----------------
    server.gui.add_markdown("### Offset (display frame, z-up, meters)")
    R_ = float(args.slider_range)
    dx = server.gui.add_slider("dx", min=-R_, max=R_, step=0.002, initial_value=0.0)
    dy = server.gui.add_slider("dy", min=-R_, max=R_, step=0.002, initial_value=0.0)
    dz = server.gui.add_slider("dz", min=-R_, max=R_, step=0.002, initial_value=0.0)
    with server.gui.add_folder("Exact values (overrides sliders when Apply pressed)"):
        nx = server.gui.add_number("dx", initial_value=0.0, step=0.001)
        ny = server.gui.add_number("dy", initial_value=0.0, step=0.001)
        nz = server.gui.add_number("dz", initial_value=0.0, step=0.001)
        apply_num_btn = server.gui.add_button("Apply numbers to sliders")
    # In Frame range mode the preview shows offset * alpha[current frame], so a
    # frame inside a ramp moves by a fraction of the slider value. With the
    # default ramp of 10 and the frame slider parked at 0 (where it lands on
    # load) the weight is 0.02 — the picture barely moves while the slider says
    # 0.10, which reads as a broken scale. This readout names the number, and
    # the checkbox lets you judge magnitude at full weight.
    offset_eff_md = server.gui.add_markdown("")
    full_weight_cb = server.gui.add_checkbox(
        "Preview at full weight (ignore ramp)", initial_value=False)
    reset_btn = server.gui.add_button("Reset offset to 0")

    # ---------------- GUI: staging ----------------
    server.gui.add_markdown("### Staged edits")
    stage_btn = server.gui.add_button("＋ Stage current offset")
    with server.gui.add_folder("Staged list"):
        staged_md = server.gui.add_markdown("_none staged_")
        undo_btn = server.gui.add_button("Undo last")
        clear_stage_btn = server.gui.add_button("Clear all")

    # ---------------- GUI: save ----------------
    server.gui.add_markdown("### Save as new sequence")
    saveas_box = server.gui.add_text("Name or task/name or /abs/path",
                                     initial_value="")
    overwrite_cb = server.gui.add_checkbox("Overwrite if exists", initial_value=False)
    save_btn = server.gui.add_button("Save copy")
    status_md = server.gui.add_markdown("_ready_")

    # ---------------- state ----------------
    state: Dict = {"idx": 0, "data": None, "quiet": False, "view": [],
                   "staged": []}
    handles: Dict[str, object] = {}

    def clear(prefix: str = ""):
        for k in [k for k in handles if k.startswith(prefix)]:
            handles[k].remove()
            del handles[k]

    def offset_vec() -> np.ndarray:
        return np.array([dx.value, dy.value, dz.value], dtype=np.float64)

    def hand_flags() -> Tuple[bool, bool]:
        v = hand_target.value
        return (v in ("Left", "Both"), v in ("Right", "Both"))

    def staged_shift_at(f: int) -> Dict[str, np.ndarray]:
        """Cumulative shift per drawable group at frame f.

        Wrist and fingers are tracked separately because a "Wrist only" edit
        moves one and not the other; a single per-hand vector could not express
        that and the preview would lie about what is being saved.
        """
        acc = {k: np.zeros(3) for k in
               ("obj", "l_wrist", "l_rest", "r_wrist", "r_rest")}
        for e in state["staged"]:
            a = e["alpha"]
            w = float(a[f]) if f < len(a) else 0.0
            if w == 0.0:
                continue
            v = np.asarray(e["offset"], dtype=np.float64) * w
            wrist_only = e.get("fields", "all") == "wrist"
            if e["object"]:
                acc["obj"] = acc["obj"] + v
            for side, key in (("left", "l"), ("right", "r")):
                if not e[side]:
                    continue
                acc[f"{key}_wrist"] = acc[f"{key}_wrist"] + v
                if not wrist_only:
                    acc[f"{key}_rest"] = acc[f"{key}_rest"] + v
        return acc

    def describe_edit(e: dict) -> str:
        tg = []
        if e["object"]:
            tg.append("obj")
        if e["left"]:
            tg.append("L")
        if e["right"]:
            tg.append("R")
        if e.get("fields", "all") == "wrist" and (e["left"] or e["right"]):
            tg[-1] = tg[-1] + "wrist"
        off = np.asarray(e["offset"])
        rng = ("all frames" if e["whole"] else
               f"{e['start']}–{e['end']} (ramp {e['ramp_in']}/{e['ramp_out']}, "
               f"{int((np.asarray(e['alpha']) > 1e-9).sum())}f)")
        return (f"`{'+'.join(tg)}`  "
                f"[{off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f}]  ·  {rng}")

    def refresh_staged_md():
        st = state["staged"]
        if not st:
            staged_md.content = "_none staged — Save will use the live slider_"
            return
        lines = [f"**{len(st)} staged** (applied in order on Save)  \n"]
        for i, e in enumerate(st):
            lines.append(f"{i + 1}. {describe_edit(e)}")
        # Net per-target translation, so a review pass has one number to sanity
        # check instead of re-adding the list by hand.
        net = {}
        for key in ("object", "left", "right"):
            v = np.zeros(3)
            for e in st:
                if e[key]:
                    v = v + np.asarray(e["offset"])
            if np.linalg.norm(v) > 1e-9:
                net[key] = v
        if net:
            lines.append("")
            for k, v in net.items():
                lines.append(f"_net {k}: [{v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f}] "
                             f"(full-weight frames only)_")
        staged_md.content = "  \n".join(lines)

    def alpha_now() -> np.ndarray:
        d = state["data"]
        T = d["num_frames"] if d else 0
        return compute_alpha(T,
                             range_mode.value == "Whole trajectory",
                             int(start_slider.value), int(end_slider.value),
                             int(ramp_in_slider.value), int(ramp_out_slider.value))

    # ---------------- drawing ----------------
    def draw_hand(side: str, f: int, shift_wrist: np.ndarray,
                  shift_rest: np.ndarray, ghost: bool):
        prefix = f"/{'ghost_' if ghost else ''}{side}_hand"
        clear(prefix)
        d = state["data"]
        present = d["has_left"] if side == "left" else d["has_right"]
        if not present or not show_hand_cb.value:
            return
        hd = d["mano_data"].get(side, {})
        pos = {}
        for name in JOINT_NAMES:
            arr = np.asarray(hd.get(name, []))
            if arr.ndim == 2 and f < len(arr):
                pos[name] = flip_z(arr[f]) + (shift_wrist if name == "wrist"
                                              else shift_rest)
        if not pos:
            return
        r = joint_radius.value
        if ghost:
            for name, p in pos.items():
                key = f"{prefix}/j_{name}"
                handles[key] = server.scene.add_icosphere(
                    key, radius=r * 0.6, position=p, color=GHOST_COLOR)
            return
        wc = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        for name, p in pos.items():
            color = wc if name == "wrist" else (200, 200, 200)
            if name != "wrist":
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
            key = f"{prefix}/j_{name}"
            handles[key] = server.scene.add_icosphere(
                key, radius=r * 1.5 if name == "wrist" else r,
                position=p, color=color)
        if show_bones_cb.value:
            for finger, chain in FINGER_CHAINS.items():
                for i in range(len(chain) - 1):
                    a, b = chain[i], chain[i + 1]
                    if a in pos and b in pos:
                        key = f"{prefix}/b_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key, points=np.stack([pos[a], pos[b]]),
                            color=FINGER_COLORS[finger], line_width=2.0)

    def draw_object(f: int, shift: np.ndarray, ghost: bool):
        prefix = f"/{'ghost_' if ghost else ''}object"
        clear(prefix)
        d = state["data"]
        if (not show_obj_cb.value or d["obj_poses"] is None
                or d["obj_vertices"] is None or f >= len(d["obj_poses"])):
            return
        try:
            v = obj_verts_at(d["obj_vertices"], d["obj_poses"][f])
        except ValueError as e:
            print(f"[WARN] {e}")
            return
        v = flip_z(v) + shift
        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key, vertices=v.astype(np.float32), faces=d["obj_faces"],
            color=GHOST_COLOR if ghost else OBJ_COLOR,
            wireframe=bool(ghost))

    def draw_aux():
        clear("/aux")
        d = state["data"]
        if not show_aux_cb.value or not d["aux_info"] or d["aux_verts"] is None:
            return
        R, pos, s = aux_R_t_scale(d["aux_info"])
        v = (R @ (d["aux_verts"] * s).T).T + pos
        handles["/aux/mesh"] = server.scene.add_mesh_simple(
            "/aux/mesh", vertices=v.astype(np.float32), faces=d["aux_faces"],
            color=AUX_COLOR)

    def redraw():
        d = state["data"]
        if d is None:
            return
        f = int(frame_slider.value)
        a = alpha_now()
        w = float(a[f]) if f < len(a) else 0.0
        zero = np.zeros(3, dtype=np.float64)
        do_l, do_r = hand_flags()

        # Preview = everything staged so far PLUS the live slider, so what you
        # look at is what a save would produce. Staged edits alone would hide
        # the current drag; the live one alone would hide the accumulated work.
        acc = staged_shift_at(f)
        w_prev = 1.0 if full_weight_cb.value else w
        live = offset_vec() * w_prev
        live_wrist_only = hand_fields_dd.value == "Wrist only"

        p_obj = acc["obj"] + (live if tgt_obj_cb.value else zero)
        p_lw = acc["l_wrist"] + (live if do_l else zero)
        p_lr = acc["l_rest"] + (live if (do_l and not live_wrist_only) else zero)
        p_rw = acc["r_wrist"] + (live if do_r else zero)
        p_rr = acc["r_rest"] + (live if (do_r and not live_wrist_only) else zero)

        moved = max(np.linalg.norm(v) for v in
                    (p_obj, p_lw, p_lr, p_rw, p_rr)) > 1e-9
        ghost_on = show_ghost_cb.value and moved

        draw_object(f, p_obj, ghost=False)
        draw_hand("left", f, p_lw, p_lr, ghost=False)
        draw_hand("right", f, p_rw, p_rr, ghost=False)
        if ghost_on:
            if np.linalg.norm(p_obj) > 1e-9:
                draw_object(f, zero, ghost=True)
            if max(np.linalg.norm(p_lw), np.linalg.norm(p_lr)) > 1e-9:
                draw_hand("left", f, zero, zero, ghost=True)
            if max(np.linalg.norm(p_rw), np.linalg.norm(p_rr)) > 1e-9:
                draw_hand("right", f, zero, zero, ghost=True)
        else:
            clear("/ghost_")
        draw_aux()

        off_now = offset_vec()
        if np.allclose(off_now, 0.0):
            offset_eff_md.content = ""
        else:
            eff = off_now * w
            if full_weight_cb.value:
                offset_eff_md.content = (
                    f"👁 previewing at **full weight**; at frame {f} the saved "
                    f"shift is **{w:.2f}×** = "
                    f"[{eff[0]:+.3f}, {eff[1]:+.3f}, {eff[2]:+.3f}]")
            elif w < 0.995:
                offset_eff_md.content = (
                    f"⚠️ frame {f} is in a ramp: weight **{w:.2f}**, so you are "
                    f"seeing [{eff[0]:+.3f}, {eff[1]:+.3f}, {eff[2]:+.3f}], not "
                    f"the full [{off_now[0]:+.3f}, {off_now[1]:+.3f}, {off_now[2]:+.3f}]")
            else:
                offset_eff_md.content = (
                    f"frame {f}: weight 1.00 — showing the full offset")

        parts = []
        if tgt_obj_cb.value:
            parts.append("object")
        suffix = " (wrist only)" if hand_fields_dd.value == "Wrist only" else ""
        if do_l:
            parts.append("left hand" + suffix)
        if do_r:
            parts.append("right hand" + suffix)
        if not parts:
            target_hint.content = "⚠️ **nothing selected** — the offset moves nothing"
        else:
            frozen = []
            if not tgt_obj_cb.value and state["data"]["obj_poses"] is not None:
                frozen.append("object")
            if not do_l and state["data"]["has_left"]:
                frozen.append("left hand")
            if not do_r and state["data"]["has_right"]:
                frozen.append("right hand")
            target_hint.content = ("moving **" + " + ".join(parts) + "**"
                                   + (f"; staying put: {', '.join(frozen)}" if frozen else ""))

        if range_mode.value == "Whole trajectory":
            range_status.content = (
                f"_whole trajectory — every frame shifted by the full offset_  \n"
                f"frame {f}: weight **1.00**")
        else:
            s, e = int(start_slider.value), int(end_slider.value)
            n_edited = int((a > 1e-9).sum())
            range_status.content = (
                f"_frames **{s}–{e}** ({n_edited} frames touched incl. ramps), "
                f"cosine ramp {int(ramp_in_slider.value)} in / "
                f"{int(ramp_out_slider.value)} out_  \n"
                f"frame {f}: weight **{w:.2f}**")

    # ---------------- filtered list ----------------
    def visible_indices() -> List[int]:
        q = filter_box.value.strip().lower()
        if not q:
            return list(range(len(all_seqs)))
        return [i for i, (t, s, _) in enumerate(all_seqs)
                if q in f"{t}/{s}".lower()]

    def refresh_dropdown(keep_idx: Optional[int] = None):
        vis = state["view"] = visible_indices()
        opts = [seq_label(*all_seqs[i]) for i in vis] or ["(no match)"]
        state["quiet"] = True
        try:
            seq_dropdown.options = opts
            if keep_idx is not None and keep_idx in vis:
                seq_dropdown.value = seq_label(*all_seqs[keep_idx])
            else:
                seq_dropdown.value = opts[0]
        finally:
            state["quiet"] = False
        return vis

    # ---------------- load ----------------
    def load_global(idx: int):
        idx = int(np.clip(idx, 0, len(all_seqs) - 1))
        task, seq, pkl = all_seqs[idx]
        print(f"[LOAD] {task}/{seq}  ({os.path.basename(pkl)})")
        dropped = len(state["staged"])
        state["staged"] = []          # staged edits belong to one sequence
        state["data"] = load_sequence_data(pkl)
        state["idx"] = idx
        d = state["data"]
        nf = d["num_frames"]

        state["quiet"] = True
        try:
            frame_slider.max = max(nf - 1, 0)
            frame_slider.value = 0
            start_slider.max = max(nf - 1, 0)
            end_slider.max = max(nf - 1, 0)
            start_slider.value = 0
            end_slider.value = max(nf - 1, 0)
            dx.value = dy.value = dz.value = 0.0
            opts = ["None"]
            if d["has_left"]:
                opts.append("Left")
            if d["has_right"]:
                opts.append("Right")
            if d["has_left"] and d["has_right"]:
                opts.append("Both")
            hand_target.options = opts
            # Default to moving every hand present, together with the object:
            # a rigid shift of the whole scene is the least surprising thing to
            # see when a slider is dragged. Narrow it by unticking Object or
            # picking a single side.
            hand_target.value = ("Both" if len(opts) == 4
                                 else (opts[1] if len(opts) == 2 else "None"))
            show_obj_cb.value = d["obj_poses"] is not None
            saveas_box.value = f"{seq}_edit"
            if seq_dropdown.value != seq_label(task, seq, pkl):
                seq_dropdown.value = seq_label(task, seq, pkl)
        finally:
            state["quiet"] = False

        pos_in_view = (state["view"].index(idx) + 1) if idx in state["view"] else 0
        warn = ""
        if d["num_frames_meta"] and d["num_frames_meta"] != nf:
            warn = f"  ⚠️ meta says {d['num_frames_meta']} frames, arrays say {nf}"
        seq_status.content = (
            f"**{pos_in_view} / {len(state['view'])}** (of {len(all_seqs)} total)  \n"
            f"`{task}/{seq}` — {d['pkl_name']}  \n"
            f"{nf} frames · obj={d['obj_id'] or 'none'} · "
            f"hands={'L' if d['has_left'] else ''}{'R' if d['has_right'] else ''}"
            f"{' · aux' if d['aux_info'] else ''}"
            f"{' · has target_object_pose (never touched)' if d['has_target_obj'] else ''}"
            f"{warn}")
        refresh_staged_md()
        if dropped:
            status_md.content = (f"⚠️ _discarded {dropped} staged edit(s) on "
                                 f"sequence switch — they were not saved_")
        clear()
        redraw()

    def step_view(delta: int):
        vis = state["view"]
        if not vis:
            return
        cur = state["idx"]
        pos = vis.index(cur) if cur in vis else 0
        load_global(vis[int(np.clip(pos + delta, 0, len(vis) - 1))])

    # ---------------- callbacks ----------------
    def on_update(_e=None):
        if state["quiet"]:
            return
        redraw()

    for w in (frame_slider, show_obj_cb, show_hand_cb, show_bones_cb, show_aux_cb,
              show_ghost_cb, joint_radius, tgt_obj_cb, hand_target, hand_fields_dd,
              start_slider, end_slider, ramp_in_slider,
              ramp_out_slider, dx, dy, dz, full_weight_cb):
        w.on_update(on_update)

    @range_mode.on_update
    def _on_range_mode(_e):
        if state["quiet"]:
            return
        # Landing on a ramp frame right after switching is what makes the
        # preview look scaled down, so jump to a full-weight frame once.
        if range_mode.value == "Frame range":
            a = alpha_now()
            full = np.flatnonzero(a >= 0.995)
            f = int(frame_slider.value)
            if len(full) and (f >= len(a) or a[f] < 0.995):
                state["quiet"] = True
                try:
                    frame_slider.value = int(full[len(full) // 2])
                finally:
                    state["quiet"] = False
        redraw()

    @filter_box.on_update
    def _on_filter(_e):
        if state["quiet"]:
            return
        vis = refresh_dropdown(keep_idx=state["idx"])
        if vis and state["idx"] not in vis:
            load_global(vis[0])
        else:
            redraw()

    @seq_dropdown.on_update
    def _on_pick(_e):
        if state["quiet"]:
            return
        vis = state["view"]
        labels = [seq_label(*all_seqs[i]) for i in vis]
        if seq_dropdown.value in labels:
            gi = vis[labels.index(seq_dropdown.value)]
            if gi != state["idx"]:
                load_global(gi)

    @prev_btn.on_click
    def _prev(_e):
        step_view(-1)

    @next_btn.on_click
    def _next(_e):
        step_view(+1)

    @set_start_btn.on_click
    def _set_start(_e):
        state["quiet"] = True
        try:
            start_slider.value = int(frame_slider.value)
            if end_slider.value < start_slider.value:
                end_slider.value = start_slider.value
            range_mode.value = "Frame range"
        finally:
            state["quiet"] = False
        redraw()

    @set_end_btn.on_click
    def _set_end(_e):
        state["quiet"] = True
        try:
            end_slider.value = int(frame_slider.value)
            if start_slider.value > end_slider.value:
                start_slider.value = end_slider.value
            range_mode.value = "Frame range"
        finally:
            state["quiet"] = False
        redraw()

    @apply_num_btn.on_click
    def _apply_num(_e):
        state["quiet"] = True
        try:
            lim = float(args.slider_range)
            dx.value = float(np.clip(nx.value, -lim, lim))
            dy.value = float(np.clip(ny.value, -lim, lim))
            dz.value = float(np.clip(nz.value, -lim, lim))
        finally:
            state["quiet"] = False
        redraw()
        status_md.content = (
            f"_offset = [{dx.value:+.4f}, {dy.value:+.4f}, {dz.value:+.4f}]_"
            + ("  ⚠️ clipped to slider range" if
               (abs(nx.value) > args.slider_range or abs(ny.value) > args.slider_range
                or abs(nz.value) > args.slider_range) else ""))

    @reset_btn.on_click
    def _reset(_e):
        state["quiet"] = True
        try:
            dx.value = dy.value = dz.value = 0.0
        finally:
            state["quiet"] = False
        redraw()
        status_md.content = "_offset reset_"

    @stage_btn.on_click
    def _stage(_e):
        d = state["data"]
        if d is None:
            return
        off = offset_vec()
        if np.allclose(off, 0.0):
            status_md.content = "_offset is zero; nothing to stage_"
            return
        do_l, do_r = hand_flags()
        if not (tgt_obj_cb.value or do_l or do_r):
            status_md.content = "_nothing selected: tick Object and/or pick a Hand_"
            return
        whole = range_mode.value == "Whole trajectory"
        state["staged"].append({
            "offset": off.copy(),
            "alpha": alpha_now().copy(),
            "object": bool(tgt_obj_cb.value),
            "left": bool(do_l),
            "right": bool(do_r),
            "fields": ("wrist" if hand_fields_dd.value == "Wrist only" else "all"),
            "whole": whole,
            "start": int(start_slider.value),
            "end": int(end_slider.value),
            "ramp_in": int(ramp_in_slider.value),
            "ramp_out": int(ramp_out_slider.value),
        })
        # Zero the sliders so the next edit starts from the staged state rather
        # than double-counting the one just banked.
        state["quiet"] = True
        try:
            dx.value = dy.value = dz.value = 0.0
        finally:
            state["quiet"] = False
        refresh_staged_md()
        redraw()
        status_md.content = f"_staged #{len(state['staged'])}; sliders zeroed_"

    @undo_btn.on_click
    def _undo(_e):
        if not state["staged"]:
            status_md.content = "_nothing staged_"
            return
        e = state["staged"].pop()
        refresh_staged_md()
        redraw()
        status_md.content = f"_removed: {describe_edit(e)}_"

    @clear_stage_btn.on_click
    def _clear_stage(_e):
        n = len(state["staged"])
        state["staged"] = []
        refresh_staged_md()
        redraw()
        status_md.content = f"_cleared {n} staged edit(s)_"

    @save_btn.on_click
    def _save(_e):
        d = state["data"]
        if d is None:
            status_md.content = "_nothing loaded_"
            return
        off = offset_vec()
        do_l, do_r = hand_flags()
        live_nonzero = not np.allclose(off, 0.0)
        staged = list(state["staged"])

        # Two flows, and the ambiguous middle is refused rather than guessed:
        #   nothing staged  -> the live slider IS the edit (one-shot use)
        #   staged + live=0 -> save exactly what was reviewed
        #   staged + live!=0-> refuse. Silently folding an unstaged drag into a
        #                      reviewed set defeats the point of reviewing, and
        #                      silently dropping it loses work.
        if staged and live_nonzero:
            status_md.content = (
                "**unstaged offset present.** Stage it (＋) to include it, or "
                "Reset offset to 0 to save only the reviewed list.")
            return
        if not staged:
            if not live_nonzero:
                status_md.content = "_nothing staged and offset is zero_"
                return
            if not (tgt_obj_cb.value or do_l or do_r):
                status_md.content = "_nothing selected: tick Object and/or pick a Hand_"
                return
            staged = [{
                "offset": off.copy(), "alpha": alpha_now().copy(),
                "object": bool(tgt_obj_cb.value), "left": bool(do_l),
                "right": bool(do_r),
                "fields": ("wrist" if hand_fields_dd.value == "Wrist only" else "all"),
                "whole": range_mode.value == "Whole trajectory",
                "start": int(start_slider.value), "end": int(end_slider.value),
                "ramp_in": int(ramp_in_slider.value),
                "ramp_out": int(ramp_out_slider.value),
            }]
        try:
            dst = resolve_dst_dir(data_root, all_seqs[state["idx"]][0],
                                  saveas_box.value)
        except ValueError as e:
            status_md.content = f"**bad name:** {e}"
            return

        new_data, notes = apply_edits(d["mano_data"], staged)

        def _edit_record(e: dict) -> dict:
            o = np.asarray(e["offset"], dtype=np.float64)
            return {
                "offset_display_zup_m": [round(float(v), 6) for v in o],
                "offset_stored_camframe_m": [round(float(v), 6)
                                             for v in disp_to_stored(o)],
                "targets": {"object": bool(e["object"]),
                            "left_hand": bool(e["left"]),
                            "right_hand": bool(e["right"])},
                "hand_fields": e.get("fields", "all"),
                "range": ("whole" if e["whole"] else
                          {"start": int(e["start"]), "end": int(e["end"]),
                           "ramp_in": int(e["ramp_in"]),
                           "ramp_out": int(e["ramp_out"]),
                           "ramp": "raised_cosine",
                           "frames_touched": int((np.asarray(e["alpha"]) > 1e-9).sum())}),
            }

        prov = {
            "tool": "offset_editor.py",
            "when": datetime.now().isoformat(timespec="seconds"),
            "source": os.path.relpath(d["seq_dir"], data_root),
            "edited_pkl": d["pkl_name"],
            "n_edits": len(staged),
            "edits": [_edit_record(e) for e in staged],
        }

        ok, msg = save_as_new_sequence(d["seq_dir"], dst, d["pkl_name"],
                                       new_data, prov, overwrite_cb.value)
        print(("[SAVE] " if ok else "[SAVE FAIL] ") + msg)
        if not ok:
            status_md.content = f"**failed:** {msg}"
            return

        # Register the new sequence so it can be reviewed without a restart.
        new_pkl = os.path.join(dst, d["pkl_name"])
        entry = (os.path.basename(os.path.dirname(dst)), os.path.basename(dst),
                 new_pkl)
        if entry not in all_seqs:
            all_seqs.append(entry)
            all_seqs.sort()
        refresh_dropdown(keep_idx=None)
        state["idx"] = all_seqs.index(entry)
        refresh_dropdown(keep_idx=state["idx"])

        state["staged"] = []
        refresh_staged_md()
        redraw()
        status_md.content = ("**Saved.**  \n" + msg + "  \n\n"
                             + "  \n".join(f"- {n}" for n in notes)
                             + "  \n\n_original untouched; use the dropdown to "
                               "open the copy_")

    # ---------------- go ----------------
    vis = refresh_dropdown()
    load_global(vis[0] if vis else 0)

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
        print("\n[INFO] stopped.")


if __name__ == "__main__":
    main()
