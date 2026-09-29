"""Live viser-based camera-extrinsic refinement on the real robot.

Subscribes to RealSense depth → back-projects with sim intrinsics → applies a
CANDIDATE T_cam_in_armbase → shows the resulting point cloud in viser, in the
arm-base (fr3_link0) frame, alongside reference scene objects:

  - world axes at the arm-base origin (= table top)
  - table-top plane outline
  - workspace bbox outline (matches sim's pc_workspace_min/max)
  - target object marker at the expected cube position (CLI)

GUI sliders apply a small `(tx, ty, tz, rx, ry, rz)` DELTA on top of
the --init_extrinsic (default: the shipped current.npy). Drag until the PC clearly shows table + cube where
they physically are. Click "Save" → writes a 4x4 .npy you can pass to deploy
via `--camera_extrinsic <file>`.

Offline against a captured cloud (the usual path — see tutorial/06):

  python tools/calib/live_calibrate_extrinsic.py \\
      --real_pc_npz logs/calib_real/accum.npz \\
      --sim_frame_pkl data/retargeting/robotool_batch/mano2sharpa_rh/0416_grasp/cube_small_2@0.pkl \\
      --sim_frame_idx 0 \\
      --init_extrinsic calib/camera_align/current.npy \\
      --out calib/camera_align/extrinsic_YYYYMMDD.npy --port 8080

Write to a new, dated file, never over current.npy (calib/camera_align/README.md).

Live against the camera host's depth publisher (deploy/realsense_depth_zmq_pub.py):

  --depth_zmq_addr tcp://<CAM_HOST>:5562

Open http://<host>:8080 in a browser. Drag sliders, watch the cloud move, save.
"""
from __future__ import annotations
import argparse
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
from dexx import deploy_config as _dcfg
import torch




def _euler_xyz_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    """rx, ry, rz in radians, XYZ intrinsic order."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def _rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to (x, y, z, w) quaternion (viser wants xyzw)."""
    # Shepperd's method
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth_zmq_addr", type=str, default=None,
                   help="Live mode: ZMQ addr of realsense_depth_zmq_pub.py, "
                        "e.g. tcp://<CAM_HOST>:5562.")
    p.add_argument("--depth_height", type=int, default=_dcfg.DEPTH_H)
    p.add_argument("--depth_width", type=int, default=_dcfg.DEPTH_W)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--subsample", type=int, default=15000,
                   help="Subsample real PC for viz (per frame).")
    p.add_argument("--target_obj_fr3", type=str, default="0.55,0.05,0.03",
                   help='Where the demo expects the object in fr3_link0 frame, '
                        '"x,y,z" m. Drawn as a red sphere.')
    p.add_argument("--target_obj_size", type=float, default=0.03,
                   help="Half-extent of the red reference cube (m).")
    p.add_argument("--update_hz", type=float, default=3.0,
                   help="How often viser fetches a new live depth frame.")
    p.add_argument("--max_translation_m", type=float, default=0.30)
    p.add_argument("--max_rotation_deg", type=float, default=30.0)
    p.add_argument("--out", type=str, required=True,
                   help="Path to save final 4x4 extrinsic .npy on Save click; use a new, "
                        "dated file, not calib/camera_align/current.npy.")
    p.add_argument("--depth_file", type=str, default=None,
                   help="Offline mode: load depth from this .npy file "
                        "(units: meters, shape (H, W)). Use to run viser locally on "
                        "a copied dump from the robot.")
    p.add_argument("--sim_scene_ply", type=str, default=None,
                   help="Path to sim scene_pc PLY (e.g. from tools/calib/gen_sim_frame_ply.py). "
                        "Loaded as orange reference cloud. Subtracts arm_base_pos to "
                        "convert env-local → fr3_link0 frame so it overlays the real PC.")
    p.add_argument("--sim_hand_ply", type=str, default=None,
                   help="Path to sim hand_pc PLY (yellow markers).")
    p.add_argument("--real_pc_npz", type=str, default=None,
                   help="OFFLINE real-PC mode: skip depth entirely and load a "
                        "captured point cloud directly. Accepts .npz with a 'points' "
                        "(N,3, camera frame, meters) [+ optional 'colors'] array, or a "
                        ".ply/.npy of xyz. Use when you have a live RealSense capture "
                        "instead of a sim-intrinsic depth image.")
    p.add_argument("--sim_ply_frame", type=str, default="env_local",
                   choices=["env_local", "link0"],
                   help="Frame the --sim_scene_ply / --sim_hand_ply points are in. "
                        "'env_local' (default): subtract arm_base_pos to reach fr3_link0. "
                        "'link0': already in fr3_link0, no shift (use for gen_sim_frame_ply "
                        "--frame0 output).")
    p.add_argument("--init_extrinsic", type=str, default=None,
                   help="4x4 .npy to use as the baseline T_cam_in_armbase instead of "
                        "the hardcoded DEFAULT (use for a new camera mount).")
    p.add_argument("--color_file", type=str, default=None,
                   help="(H,W,3) uint8 RGB .npy aligned to --depth_file, so the real PC "
                        "renders in true camera color instead of flat blue.")
    p.add_argument("--sim_frame_pkl", type=str, default=None,
                   help="Retarget pkl to load the FULL sim robot (FR3+Sharpa) MESHES + table "
                        "directly into viser at frame --sim_frame_idx (fr3_link0 frame). "
                        "Cleaner than a sampled --sim_scene_ply.")
    p.add_argument("--sim_frame_idx", type=int, default=0)
    p.add_argument("--sim_table_z", type=float, default=_dcfg.TABLE_SURFACE_Z,
                   help="table-top z in env-local; drawn at (table_z - arm_base_z) in link0.")
    p.add_argument("--point_size", type=float, default=0.005,
                   help="real PC point size in viser (smaller = finer). Try 0.0015.")
    args = p.parse_args()

    # Pull these in both modes.
    SIM_INTRINSICS = _dcfg.SIM_INTRINSICS

    sub = None
    static_depth_np: Optional[np.ndarray] = None
    preloaded_pc_cam: Optional[np.ndarray] = None   # (N,3) camera-frame points, offline real-PC mode
    preloaded_pc_col: Optional[np.ndarray] = None   # (N,3) uint8 RGB, optional

    if args.real_pc_npz is not None:
        # ---- OFFLINE real-PC mode: load captured camera-frame points directly ----
        path = args.real_pc_npz
        if path.endswith(".npz"):
            z = np.load(path, allow_pickle=True)
            preloaded_pc_cam = np.asarray(z["points"], dtype=np.float32).reshape(-1, 3)
            if "colors" in z:
                preloaded_pc_col = np.asarray(z["colors"], dtype=np.uint8).reshape(-1, 3)
                print(f"[live-calib] real_pc_npz has colors -> true-RGB accumulated cloud")
        elif path.endswith(".npy"):
            preloaded_pc_cam = np.load(path).astype(np.float32).reshape(-1, 3)
        elif path.endswith(".ply"):
            with open(path) as fh:
                lines = fh.readlines()
            s = next(i for i, L in enumerate(lines) if L.strip() == "end_header") + 1
            preloaded_pc_cam = np.array(
                [list(map(float, L.split()[:3])) for L in lines[s:] if L.strip()],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"--real_pc_npz must be .npz/.npy/.ply, got {path}")
        print(f"[live-calib] OFFLINE real-PC mode: {preloaded_pc_cam.shape[0]} camera-frame "
              f"points from {path}")

    # DepthToPointCloud only needed when we actually back-project a depth image.
    if preloaded_pc_cam is None:
        from dexx.tasks.franka_sharpa.pointcloud.depth_to_pointcloud import (
            DepthToPointCloud,
        )

    if args.real_pc_npz is not None:
        pass  # no live depth or depth_file needed
    elif args.depth_file is not None:
        static_depth_np = np.load(args.depth_file).astype(np.float32)
        if static_depth_np.ndim != 2:
            raise ValueError(f"expected (H,W) depth, got {static_depth_np.shape}")
        # Override H/W from the file
        args.depth_height, args.depth_width = static_depth_np.shape
        print(f"[live-calib] OFFLINE mode: depth from {args.depth_file} "
              f"shape={static_depth_np.shape} "
              f"valid_frac={(static_depth_np > 0).mean():.2%}")

    else:
        # ---- Live depth from the camera host (CPU buffer; no GPU needed) ----
        if not args.depth_zmq_addr:
            p.error("pass --real_pc_npz / --depth_file for offline mode, or "
                    "--depth_zmq_addr for live mode")
        from dexx.scripts.deploy.realsense_depth_zmq_subscriber import (
            RealSenseDepthZmqSubscriber,
        )
        sub = RealSenseDepthZmqSubscriber(
            addr=args.depth_zmq_addr,
            height=args.depth_height, width=args.depth_width, device="cpu",
        )
        print(f"[live-calib] subscribed to {args.depth_zmq_addr}; waiting for first frame…")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if sub.get_latest() is not None:
                break
            time.sleep(0.05)
        if sub.get_latest() is None:
            print("[live-calib] ERROR: no depth frame in 10s. Is realsense_depth_zmq_pub.py running?")
            return 1
        print("[live-calib] depth OK; opening viser…")

    # Optional per-pixel RGB aligned to --depth_file (for true-color real PC).
    color_img = None
    if args.color_file is not None and os.path.exists(args.color_file):
        color_img = np.load(args.color_file).astype(np.uint8)
        if color_img.ndim != 3 or color_img.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) color, got {color_img.shape}")
        print(f"[live-calib] color from {args.color_file} shape={color_img.shape} "
              f"-> real PC will render in true RGB")

    d2pc = None
    if preloaded_pc_cam is None:
        d2pc = DepthToPointCloud(
            height=args.depth_height, width=args.depth_width,
            fx=SIM_INTRINSICS["fx"], fy=SIM_INTRINSICS["fy"],
            cx=SIM_INTRINSICS["cx"], cy=SIM_INTRINSICS["cy"],
            device="cpu", accepts_normalized=False,
        )

    # Current best extrinsic (refined by user).
    if args.init_extrinsic is not None:
        T_base = np.load(args.init_extrinsic).astype(np.float32)
        assert T_base.shape == (4, 4), f"init_extrinsic must be 4x4, got {T_base.shape}"
        print(f"[live-calib] baseline T_cam_in_armbase from {args.init_extrinsic}")
    else:
        T_base = _dcfg.default_camera_extrinsic().astype(np.float32).copy()
    print(f"[live-calib] baseline T_cam_in_armbase:\n{T_base}")

    # ---- viser ----
    import viser
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.scene.world_axes.visible = True

    # Reference geometry in arm-base frame.
    tgt = np.array([float(x) for x in args.target_obj_fr3.split(",")], dtype=np.float32)
    # Demo target object (red cube)
    server.scene.add_box(
        "/ref/target_obj",
        dimensions=(2 * args.target_obj_size,) * 3,
        position=tuple(tgt.tolist()),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        color=(255, 30, 30),
    )
    # Table-top plane outline (workspace_min[x] to max[x] × min[y] to max[y] at z=0)
    # Workspace box, expressed in fr3_link0 frame: the env-local crop minus the
    # arm-base height. Both come from dexx.deploy_config so a re-mount or a new
    # crop propagates here instead of silently disagreeing with the trainer.
    ws_min = np.asarray(_dcfg.PC_WORKSPACE_MIN, dtype=np.float64) - np.array([0.0, 0.0, _dcfg.ARM_BASE_Z])
    ws_max = np.asarray(_dcfg.PC_WORKSPACE_MAX, dtype=np.float64) - np.array([0.0, 0.0, _dcfg.ARM_BASE_Z])
    # 4 lines = workspace bbox outline at z=0 (table top)
    corners_xy = [
        (ws_min[0], ws_min[1], 0.0),
        (ws_max[0], ws_min[1], 0.0),
        (ws_max[0], ws_max[1], 0.0),
        (ws_min[0], ws_max[1], 0.0),
        (ws_min[0], ws_min[1], 0.0),
    ]
    server.scene.add_spline_catmull_rom(
        "/ref/workspace_outline_table",
        positions=tuple(corners_xy),
        color=(80, 200, 255),
        line_width=2.0,
    )

    # ---- Optional: sim reference point cloud (overlay target) ----
    # env_local: convert to fr3_link0 by subtracting arm_base_pos.
    # link0: already in fr3_link0 (e.g. gen_sim_frame_ply.py --link0), no shift.
    _ARM_BASE_IN_ENV_LOCAL = _dcfg.arm_base_pos_np()
    _sim_shift = (np.zeros(3, dtype=np.float32) if args.sim_ply_frame == "link0"
                  else _ARM_BASE_IN_ENV_LOCAL)
    def _load_pc_xyz(path: str) -> np.ndarray:
        """Load (N,3) xyz from .ply (ASCII) or .npy."""
        if path.endswith(".npy"):
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                raise ValueError(f"expected (N,3) npy, got {arr.shape}")
            return arr[:, :3]
        with open(path) as fh:
            lines = fh.readlines()
        s = next(i for i, L in enumerate(lines) if L.strip() == "end_header") + 1
        return np.array(
            [list(map(float, L.split()[:3])) for L in lines[s:] if L.strip()],
            dtype=np.float32,
        )

    sim_scene_handle = None
    if args.sim_scene_ply is not None and os.path.exists(args.sim_scene_ply):
        pts_sim = _load_pc_xyz(args.sim_scene_ply) - _sim_shift[None]
        sim_scene_handle = server.scene.add_point_cloud(
            "/ref/sim_scene_pc",
            points=pts_sim.astype(np.float32),
            colors=(255, 140, 0),     # orange — sim ref
            point_size=0.004,
        )
        print(f"[live-calib] loaded sim_scene_pc: {pts_sim.shape[0]} points from {args.sim_scene_ply}")

    sim_hand_handle = None
    if args.sim_hand_ply is not None and os.path.exists(args.sim_hand_ply):
        pts_hand = _load_pc_xyz(args.sim_hand_ply) - _sim_shift[None]
        sim_hand_handle = server.scene.add_point_cloud(
            "/ref/sim_hand_pc",
            points=pts_hand.astype(np.float32),
            colors=(255, 240, 0),     # yellow — sim hand bodies
            point_size=0.010,
        )
        print(f"[live-calib] loaded sim_hand_pc: {pts_hand.shape[0]} points from {args.sim_hand_ply}")

    # ---- Optional: load the FULL sim robot MESHES + table directly (fr3_link0 frame) ----
    # The whole viser scene is in the fr3_link0 / arm-base frame (the real PC is
    # transformed to arm-base via the extrinsic). URDF meshes are link0-native, so
    # they need NO shift; the table sits at (table_z - arm_base_z) in link0.
    sim_mesh_handles = []
    if args.sim_frame_pkl is not None and os.path.exists(args.sim_frame_pkl):
        import pickle as _pickle
        import yourdfpy as _yourdfpy
        _URDF = "assets/generated/fr3_with_right_sharpa_wave.urdf"
        _d = _pickle.load(open(args.sim_frame_pkl, "rb"))
        _arm = np.asarray(_d["opt_arm_joint_pos"])[args.sim_frame_idx]     # (7,)
        _hand = np.asarray(_d["opt_dof_pos"])[args.sim_frame_idx]          # (22,)
        _base_z = float(np.asarray(_d["arm_base_pos"], dtype=np.float64)[2])
        _u = _yourdfpy.URDF.load(_URDF, load_meshes=True, build_scene_graph=True)
        _u.update_cfg(dict(zip(list(_u.actuated_joint_names),
                               np.concatenate([_arm, _hand]).tolist())))
        _scene = _u.scene
        for _i, (_gname, _geom) in enumerate(_scene.geometry.items()):
            _T = _scene.graph.get(_gname)[0]
            _m = _geom.copy().apply_transform(_T)   # posed in fr3_link0
            h = server.scene.add_mesh_trimesh(f"/ref/sim_robot/{_i}", _m)
            sim_mesh_handles.append(h)
        _table_top_z = args.sim_table_z - _base_z    # link0 z of table surface
        htab = server.scene.add_box(
            "/ref/sim_table",
            dimensions=(1.05, 1.10, 0.02),
            position=(0.325, 0.0, _table_top_z - 0.01),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            color=(150, 150, 150),
        )
        sim_mesh_handles.append(htab)
        print(f"[live-calib] loaded sim robot: {len(sim_mesh_handles)-1} meshes + table "
              f"(top z={_table_top_z:+.3f} link0) from {args.sim_frame_pkl} frame {args.sim_frame_idx}")

    # ---- GUI sliders ----
    gh = server.gui
    gh.add_markdown("## Camera extrinsic — drag to align")
    gh.add_markdown(
        "Sliders apply a **delta** on top of the initial extrinsic.\n"
        "Goal: blue real PC should land on **table top (z=0)** and the **red cube** should "
        "overlap with the actual cube in the PC."
    )

    tx_s = gh.add_slider("Δtx (m)", -args.max_translation_m, args.max_translation_m, 0.001, 0.0)
    ty_s = gh.add_slider("Δty (m)", -args.max_translation_m, args.max_translation_m, 0.001, 0.0)
    tz_s = gh.add_slider("Δtz (m)", -args.max_translation_m, args.max_translation_m, 0.001, 0.0)
    rx_s = gh.add_slider("Δrx (°)", -args.max_rotation_deg, args.max_rotation_deg, 0.1, 0.0)
    ry_s = gh.add_slider("Δry (°)", -args.max_rotation_deg, args.max_rotation_deg, 0.1, 0.0)
    rz_s = gh.add_slider("Δrz (°)", -args.max_rotation_deg, args.max_rotation_deg, 0.1, 0.0)

    info_t = gh.add_text("Translation", initial_value="-")
    info_q = gh.add_text("Quaternion (xyzw)", initial_value="-")
    refresh_btn = gh.add_button("Refresh depth frame")
    save_btn = gh.add_button("💾 Save extrinsic to file")
    reset_btn = gh.add_button("Reset sliders")

    # Overlay toggles
    show_sim_scene = gh.add_checkbox("Show sim scene_pc (orange)",
                                     initial_value=sim_scene_handle is not None)
    show_sim_hand = gh.add_checkbox("Show sim hand_pc (yellow)",
                                    initial_value=sim_hand_handle is not None)

    show_sim_robot = gh.add_checkbox("Show sim robot+table (grey mesh)",
                                     initial_value=len(sim_mesh_handles) > 0)

    @show_sim_scene.on_update
    def _(_):
        if sim_scene_handle is not None:
            sim_scene_handle.visible = show_sim_scene.value
    @show_sim_hand.on_update
    def _(_):
        if sim_hand_handle is not None:
            sim_hand_handle.visible = show_sim_hand.value
    @show_sim_robot.on_update
    def _(_):
        for h in sim_mesh_handles:
            h.visible = show_sim_robot.value

    state = {
        "depth_latest_np": None,           # (H, W) m
        "pc_cam": None,                    # (N, 3) cam frame
        "pc_col": None,                    # (N, 3) uint8 RGB, or None
        "pc_handle": None,
    }

    def fetch_depth_and_unproject():
        if preloaded_pc_cam is not None:
            # Offline real-PC mode: points already in camera frame; just subsample once.
            pts = preloaded_pc_cam
            cols = preloaded_pc_col
            if pts.shape[0] > args.subsample:
                sel = np.random.choice(pts.shape[0], size=args.subsample, replace=False)
                pts = pts[sel]
                if cols is not None:
                    cols = cols[sel]
            state["pc_cam"] = pts
            state["pc_col"] = cols
            return pts
        if static_depth_np is not None:
            d_np = static_depth_np
        else:
            d = sub.get_latest()
            if d is None:
                return None
            d_np = d.cpu().numpy().astype(np.float32)
        state["depth_latest_np"] = d_np
        pts_cam, valid = d2pc.back_project(torch.from_numpy(d_np).unsqueeze(0))
        vmask = valid[0].numpy().astype(bool)
        pts = pts_cam[0].numpy()[vmask]
        # per-point RGB from the aligned color image (row-major flatten matches back_project)
        cols = None
        if color_img is not None and color_img.shape[:2] == d_np.shape:
            cols = color_img.reshape(-1, 3)[vmask]
        if pts.shape[0] > args.subsample:
            sel = np.random.choice(pts.shape[0], size=args.subsample, replace=False)
            pts = pts[sel]
            if cols is not None:
                cols = cols[sel]
        state["pc_cam"] = pts
        state["pc_col"] = cols
        return pts

    def current_T():
        # Compose: T_new = T_delta @ T_base where T_delta is small Euler+translation.
        # We treat delta as applied in arm-base frame (left-multiply) so user sees
        # the camera (and its PC) shift wholesale.
        dt = np.array([tx_s.value, ty_s.value, tz_s.value], dtype=np.float32)
        dR = _euler_xyz_to_rot(np.deg2rad(rx_s.value), np.deg2rad(ry_s.value), np.deg2rad(rz_s.value))
        T_delta = np.eye(4, dtype=np.float32)
        T_delta[:3, :3] = dR
        T_delta[:3, 3] = dt
        return T_delta @ T_base

    def redraw():
        if state["pc_cam"] is None:
            fetch_depth_and_unproject()
        if state["pc_cam"] is None:
            return
        T = current_T()
        R = T[:3, :3]; t = T[:3, 3]
        pts_arm = state["pc_cam"] @ R.T + t  # (N, 3) in arm-base frame
        if state["pc_handle"] is not None:
            try:
                state["pc_handle"].remove()
            except Exception:
                pass
        _cols = state["pc_col"] if state["pc_col"] is not None else (80, 160, 255)
        state["pc_handle"] = server.scene.add_point_cloud(
            "/real_pc",
            points=pts_arm.astype(np.float32),
            colors=_cols,
            point_size=args.point_size,
        )
        info_t.value = f"({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})"
        info_q.value = ", ".join(f"{v:+.4f}" for v in _rot_to_quat_xyzw(R))

    def _on_slider_change(_evt=None):
        redraw()
    for s in (tx_s, ty_s, tz_s, rx_s, ry_s, rz_s):
        s.on_update(_on_slider_change)

    @refresh_btn.on_click
    def _on_refresh(_evt):
        pts = fetch_depth_and_unproject()
        n = 0 if pts is None else pts.shape[0]
        print(f"[live-calib] refreshed PC: {n} points")
        redraw()

    @save_btn.on_click
    def _on_save(_evt):
        T = current_T()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, T.astype(np.float32))
        print(f"\n[live-calib] saved extrinsic to {args.out}")
        print(f"T_cam_in_armbase =\n{T}")
        print(f"To use: deploy_pc.py --camera_extrinsic {args.out}")

    @reset_btn.on_click
    def _on_reset(_evt):
        for s in (tx_s, ty_s, tz_s, rx_s, ry_s, rz_s):
            s.value = 0.0
        redraw()

    # Initial fetch + draw.
    fetch_depth_and_unproject()
    redraw()
    print(f"\n[live-calib] viser at http://0.0.0.0:{args.port}  (open in browser)")
    print(f"[live-calib] press Ctrl+C to quit")

    # Background timer to refresh depth periodically (so user sees live scene).
    # In OFFLINE mode (static depth/PC) the scene never changes, so DON'T re-fetch —
    # re-subsampling every tick reshuffles the random point subset and makes the
    # cloud "jitter". Only run the live refresh loop when reading a live source.
    stop_evt = threading.Event()
    _is_offline = (static_depth_np is not None) or (preloaded_pc_cam is not None)
    if not _is_offline:
        def _refresh_loop():
            period = 1.0 / max(0.1, args.update_hz)
            while not stop_evt.is_set():
                fetch_depth_and_unproject()
                redraw()
                stop_evt.wait(period)
        refresh_thr = threading.Thread(target=_refresh_loop, daemon=True)
        refresh_thr.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[live-calib] shutting down")
    finally:
        stop_evt.set()
        if sub is not None:
            sub.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
