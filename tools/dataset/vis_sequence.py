#!/usr/bin/env python3
"""
Visualize batch-processed RoboTool data using viser.
Shows hand keypoints (spheres + bones) and object mesh per frame.

Usage:
    # Visualize a specific sequence
    python tools/dataset/vis_sequence.py --sequence blue_cup/blue_cup_1

    # Specify data root and port
    python tools/dataset/vis_sequence.py \
        --sequence blue_cup/blue_cup_1 \
        --data_dir data/robotool_batch \
        --port 8080

Requires: pip install viser trimesh
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import trimesh
import viser

# ---- Hand skeleton definition ----

# Finger chains: wrist -> proximal -> intermediate -> distal -> tip
FINGER_CHAINS = {
    "thumb":  ["wrist", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip"],
    "index":  ["wrist", "index_proximal", "index_intermediate", "index_distal", "index_tip"],
    "middle": ["wrist", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip"],
    "ring":   ["wrist", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip"],
    "pinky":  ["wrist", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip"],
}

# All joint names
JOINT_NAMES = [
    "wrist",
    "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
]

# Colors
FINGER_COLORS = {
    "thumb":  (255, 100, 100),
    "index":  (100, 255, 100),
    "middle": (100, 100, 255),
    "ring":   (255, 255, 100),
    "pinky":  (255, 100, 255),
}
LEFT_WRIST_COLOR = (100, 149, 237)   # cornflower blue
RIGHT_WRIST_COLOR = (255, 160, 122)  # light salmon
OBJ_COLOR = (180, 180, 180)


def load_sequence(seq_dir: str):
    """Load mano_joints.pkl, meta.json, and object mesh from a sequence directory."""
    mano_path = os.path.join(seq_dir, "mano_joints.pkl")
    meta_path = os.path.join(seq_dir, "meta.json")

    with open(mano_path, "rb") as f:
        mano_data = pickle.load(f)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load optimized data if available
    opt_path = os.path.join(seq_dir, "mano_joints_optimized.pkl")
    opt_data = None
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt_data = pickle.load(f)
        print(f"[INFO] Loaded optimized data from {opt_path}")

    # Load object mesh
    obj_mesh = None
    obj_faces = None
    obj_vertices = None
    if meta["object_ids"]:
        obj_id = meta["object_ids"][0]
        # Load from centralized models dir
        data_root = os.path.dirname(os.path.dirname(seq_dir))  # data/robotool_batch
        obj_path = os.path.join(data_root, "models", obj_id, "cleaned_mesh_10000.obj")
        if os.path.exists(obj_path):
            obj_mesh = trimesh.load(obj_path, process=False, force="mesh")
            obj_vertices = obj_mesh.vertices.copy().astype(np.float32)
            obj_faces = obj_mesh.faces.copy()

    return mano_data, meta, obj_vertices, obj_faces, opt_data


def flip_z(v: np.ndarray) -> np.ndarray:
    """View-side Rx(180°): (x, y, z) -> (x, -y, -z). Used for data with
    z_flipped=false (raw z-down OpenCV) to render z-up. Data with
    z_flipped=true is already z-up on disk, so this is skipped.
    """
    out = v.copy()
    out[..., 1] *= -1
    out[..., 2] *= -1
    return out


# View-side rotation modes (applied AFTER all data-coord adjustments).
# Each mode is a 180° rotation about world X / Y / Z; "none" is identity.
# Both hand keypoints and object vertices get the same transform so they
# stay in the same frame.
VIEW_FLIP_MODES = ("none", "rx180", "ry180", "rz180")


def apply_view_flip(v: np.ndarray, mode: str) -> np.ndarray:
    """View-side 180° rotation. Same transform is applied to hand + object."""
    if mode == "none":
        return v
    out = v.copy()
    if mode == "rx180":
        # (x, y, z) -> (x, -y, -z)
        out[..., 1] *= -1
        out[..., 2] *= -1
    elif mode == "ry180":
        # (x, y, z) -> (-x, y, -z)
        out[..., 0] *= -1
        out[..., 2] *= -1
    elif mode == "rz180":
        # (x, y, z) -> (-x, -y, z)
        out[..., 0] *= -1
        out[..., 1] *= -1
    return out


def get_hand_positions(hand_data: dict, frame_idx: int, already_flipped: bool) -> dict:
    """Extract all joint positions for a given frame. Returns {joint_name: (3,) ndarray}.

    If `already_flipped` is True (data on disk was pre-rotated by Rx(180°)
    when the raw capture was converted; meta.json z_flipped), no view-side rotation is applied and the
    visualizer faithfully displays what retarget will load.
    """
    positions = {}
    for name in JOINT_NAMES:
        if name in hand_data:
            arr = np.array(hand_data[name])
            if frame_idx < len(arr):
                p = arr[frame_idx]
                positions[name] = p.copy() if already_flipped else flip_z(p)
    return positions


def transform_obj_vertices(vertices: np.ndarray, pose_4x4: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation to object vertices."""
    R = pose_4x4[:3, :3]
    t = pose_4x4[:3, 3]
    return (R @ vertices.T).T + t


def build_bone_segments(positions: dict) -> tuple:
    """Build line segments for bone connections. Returns (starts, ends, colors)."""
    starts = []
    ends = []
    colors = []
    for finger, chain in FINGER_CHAINS.items():
        color = FINGER_COLORS[finger]
        for i in range(len(chain) - 1):
            if chain[i] in positions and chain[i + 1] in positions:
                starts.append(positions[chain[i]])
                ends.append(positions[chain[i + 1]])
                colors.append(color)
    return starts, ends, colors


def main():
    parser = argparse.ArgumentParser(description="Visualize RoboTool batch data with viser")
    parser.add_argument("--sequence", type=str, required=True,
                        help="Sequence path relative to data_dir (e.g., blue_cup/blue_cup_1)")
    parser.add_argument("--data_dir", type=str, default="data/robotool_batch",
                        help="Root directory for batch-processed data")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--view_flip", type=str, default="none", choices=VIEW_FLIP_MODES,
                        help="Initial view-side 180° rotation applied to BOTH hand and "
                             "object before rendering. Options: none / rx180 / ry180 / "
                             "rz180. Live-toggleable in the GUI; this just sets the "
                             "starting value. Use rx180 to flip a z-up scene to z-down "
                             "(or vice-versa).")
    args = parser.parse_args()

    seq_dir = os.path.join(args.data_dir, args.sequence)
    if not os.path.isdir(seq_dir):
        print(f"[ERROR] Sequence directory not found: {seq_dir}")
        sys.exit(1)

    print(f"[INFO] Loading {args.sequence}...")
    mano_data, meta, obj_vertices, obj_faces, opt_data = load_sequence(seq_dir)

    num_frames = meta["num_frames"]
    mano_sides = meta["mano_sides"]
    already_flipped = bool(meta.get("z_flipped", False))
    print(f"[INFO] z_flipped on disk: {already_flipped}, "
          f"rot_z_deg on disk: {meta.get('rot_z_deg', 0.0)}")
    # z_flipped=true: the entire scene was pre-rotated by
    # Rx(180°) at raw-data conversion -- proper rotation, mesh untouched, every
    # field consistent. The visualizer applies NO additional view-side flip,
    # so what you see here = exactly what retarget will load.
    # z_flipped=false: raw is z-down OpenCV, viewer flips to z-up
    # for display; retarget applies its own coord conversion via mujoco2gym.
    has_left = "left" in mano_sides and "left" in mano_data and len(mano_data["left"].get("wrist", [])) > 0
    has_right = "right" in mano_sides and "right" in mano_data and len(mano_data["right"].get("wrist", [])) > 0
    has_object = obj_vertices is not None and "original_data" in mano_data
    has_optimized = opt_data is not None

    # Get object poses
    obj_poses = None
    if has_object:
        tp = mano_data["original_data"].get("tool_object_pose")
        if tp is not None:
            obj_poses = np.array(tp)

    print(f"[INFO] {num_frames} frames, left={has_left}, right={has_right}, object={has_object}")

    # ---- Viser server ----
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"[INFO] Viser server at http://localhost:{args.port}")

    # GUI controls
    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    playing = server.gui.add_checkbox("Play", initial_value=False)
    fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=15)

    show_left = server.gui.add_checkbox("Show Left Hand", initial_value=has_left)
    show_right = server.gui.add_checkbox("Show Right Hand", initial_value=has_right)
    show_object = server.gui.add_checkbox("Show Object", initial_value=has_object)
    show_bones = server.gui.add_checkbox("Show Bones", initial_value=True)
    joint_radius = server.gui.add_slider("Joint Radius", min=0.001, max=0.015, step=0.001, initial_value=0.005)

    # 180° view rotation (applies to hand + object together). Set via CLI
    # --view_flip; toggle live in the dropdown.
    view_flip = server.gui.add_dropdown(
        "View flip (180°)",
        options=list(VIEW_FLIP_MODES),
        initial_value=args.view_flip,
    )

    # Optimized data controls
    if has_optimized:
        server.gui.add_markdown("---")
        server.gui.add_markdown("**Optimized Data** (green)")
        show_opt_left = server.gui.add_checkbox("Show Opt Left", initial_value=has_left and "left" in opt_data)
        show_opt_right = server.gui.add_checkbox("Show Opt Right", initial_value=has_right and "right" in opt_data)
        show_opt_bones = server.gui.add_checkbox("Show Opt Bones", initial_value=True)

    # World axes
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.002)

    # Handle storage
    handles = {}

    def clear_handles(prefix: str):
        keys_to_remove = [k for k in handles if k.startswith(prefix)]
        for k in keys_to_remove:
            handles[k].remove()
            del handles[k]

    def draw_hand(hand_data: dict, frame_idx: int, side: str):
        """Draw hand keypoints and bones for one frame."""
        prefix = f"/{side}_hand"
        clear_handles(prefix)

        if (side == "left" and not show_left.value) or (side == "right" and not show_right.value):
            return

        positions = get_hand_positions(hand_data, frame_idx, already_flipped)
        if not positions:
            return

        # Apply user-selected view flip (rx180 / ry180 / rz180 / none).
        vf = view_flip.value
        if vf != "none":
            positions = {n: apply_view_flip(p, vf) for n, p in positions.items()}

        wrist_color = LEFT_WRIST_COLOR if side == "left" else RIGHT_WRIST_COLOR
        r = joint_radius.value

        # Draw joint spheres
        for name, pos in positions.items():
            color = wrist_color if name == "wrist" else None
            if color is None:
                # Find which finger this joint belongs to
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = FINGER_COLORS[finger]
                        break
                if color is None:
                    color = (200, 200, 200)

            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key,
                radius=r * 1.5 if name == "wrist" else r,
                position=pos.astype(np.float64),
                color=color,
            )

        # Draw bones
        if show_bones.value:
            for finger, chain in FINGER_CHAINS.items():
                color = FINGER_COLORS[finger]
                for i in range(len(chain) - 1):
                    if chain[i] in positions and chain[i + 1] in positions:
                        p0 = positions[chain[i]].astype(np.float64)
                        p1 = positions[chain[i + 1]].astype(np.float64)

                        # Use spline with 2 points to draw a line segment
                        key = f"{prefix}/bone_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key,
                            positions=np.stack([p0, p1]),
                            color=color,
                            line_width=2.0,
                        )

    # Colors for optimized hand (green-tinted versions)
    OPT_FINGER_COLORS = {
        "thumb":  (100, 200, 100),
        "index":  (50, 220, 50),
        "middle": (80, 180, 120),
        "ring":   (100, 220, 80),
        "pinky":  (60, 200, 140),
    }
    OPT_WRIST_COLOR = (80, 220, 80)

    def draw_hand_opt(hand_data: dict, frame_idx: int, side: str):
        """Draw optimized hand keypoints and bones."""
        prefix = f"/{side}_hand_opt"
        clear_handles(prefix)

        if not has_optimized:
            return
        show_flag = show_opt_left.value if side == "left" else show_opt_right.value
        if not show_flag:
            return

        positions = get_hand_positions(hand_data, frame_idx, already_flipped)
        if not positions:
            return

        # Apply user-selected view flip (must match the non-opt hand).
        vf = view_flip.value
        if vf != "none":
            positions = {n: apply_view_flip(p, vf) for n, p in positions.items()}

        r = joint_radius.value

        for name, pos in positions.items():
            if name == "wrist":
                color = OPT_WRIST_COLOR
            else:
                color = (100, 200, 100)
                for finger, chain in FINGER_CHAINS.items():
                    if name in chain:
                        color = OPT_FINGER_COLORS[finger]
                        break

            key = f"{prefix}/joint_{name}"
            handles[key] = server.scene.add_icosphere(
                key,
                radius=r * 1.5 if name == "wrist" else r,
                position=pos.astype(np.float64),
                color=color,
            )

        if show_opt_bones.value:
            for finger, chain in FINGER_CHAINS.items():
                color = OPT_FINGER_COLORS[finger]
                for i in range(len(chain) - 1):
                    if chain[i] in positions and chain[i + 1] in positions:
                        p0 = positions[chain[i]].astype(np.float64)
                        p1 = positions[chain[i + 1]].astype(np.float64)
                        key = f"{prefix}/bone_{finger}_{i}"
                        handles[key] = server.scene.add_spline_catmull_rom(
                            key,
                            positions=np.stack([p0, p1]),
                            color=color,
                            line_width=2.0,
                        )

    def draw_object(frame_idx: int):
        """Draw object mesh for one frame."""
        prefix = "/object"
        clear_handles(prefix)

        if not show_object.value or obj_poses is None or obj_vertices is None:
            return
        if frame_idx >= len(obj_poses):
            return

        pose = obj_poses[frame_idx]
        if pose.shape == (4, 4):
            verts = transform_obj_vertices(obj_vertices, pose)
        elif pose.shape == (7,):
            from scipy.spatial.transform import Rotation as Rot
            R = Rot.from_quat(pose[:4]).as_matrix()
            t = pose[4:7]
            verts = (R @ obj_vertices.T).T + t
        else:
            return

        # Match the hand convention: z_flipped=false data needs view-side
        # Rx(180°) to render z-up; z_flipped=true data is already z-up on disk.
        if not already_flipped:
            verts = flip_z(verts)

        # Apply user-selected view flip (same transform as the hand).
        vf = view_flip.value
        if vf != "none":
            verts = apply_view_flip(verts, vf)

        key = f"{prefix}/mesh"
        handles[key] = server.scene.add_mesh_simple(
            key,
            vertices=verts.astype(np.float32),
            faces=obj_faces,
            color=OBJ_COLOR,
        )

    def update_frame(frame_idx: int):
        if has_left:
            draw_hand(mano_data["left"], frame_idx, "left")
        if has_right:
            draw_hand(mano_data["right"], frame_idx, "right")
        if has_optimized:
            if "left" in opt_data and opt_data["left"]:
                draw_hand_opt(opt_data["left"], frame_idx, "left")
            if "right" in opt_data and opt_data["right"]:
                draw_hand_opt(opt_data["right"], frame_idx, "right")
        draw_object(frame_idx)

    # Initial render
    update_frame(0)

    # Event handlers
    @frame_slider.on_update
    def _on_frame(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_left.on_update
    def _on_left(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_right.on_update
    def _on_right(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_object.on_update
    def _on_obj(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @show_bones.on_update
    def _on_bones(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @joint_radius.on_update
    def _on_radius(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    @view_flip.on_update
    def _on_view_flip(event: viser.GuiEvent) -> None:
        update_frame(int(frame_slider.value))

    if has_optimized:
        @show_opt_left.on_update
        def _on_opt_left(event: viser.GuiEvent) -> None:
            update_frame(int(frame_slider.value))

        @show_opt_right.on_update
        def _on_opt_right(event: viser.GuiEvent) -> None:
            update_frame(int(frame_slider.value))

        @show_opt_bones.on_update
        def _on_opt_bones(event: viser.GuiEvent) -> None:
            update_frame(int(frame_slider.value))

    # Playback loop
    try:
        while True:
            if playing.value:
                frame = int(frame_slider.value)
                frame = (frame + 1) % num_frames
                frame_slider.value = frame
                update_frame(frame)
                time.sleep(1.0 / fps_slider.value)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped.")


if __name__ == "__main__":
    main()
