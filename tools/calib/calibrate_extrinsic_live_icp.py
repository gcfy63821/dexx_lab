#!/usr/bin/env python3
"""Six-DOF camera-extrinsic calibration from the live robot, over several arm poses.

Step 1 of the recommended procedure (tutorial/06). Registers the real depth
cloud against the sim FR3 arm, posed by forward kinematics from the joint
angles recorded at the same moment as the depth. Step 2,
`level_extrinsic_to_table.py`, then re-fits height and tilt against the table.

Why the arm: a table plane constrains only three degrees of freedom (height
and two tilts) and says nothing about yaw or in-plane translation. The arm is
a complex 3-D shape and constrains all six.

Why several poses: one arm configuration seen from one viewpoint can leave a
DOF weakly observed (a mostly-vertical arm barely constrains rotation about
its own axis). Capture a few genuinely different poses and solve jointly:

    # move the arm (teleop / hand-guiding), hold it still, then for each pose:
    python tools/calib/calibrate_extrinsic_live_icp.py capture \\
        --session logs/calib_live --ip <NUC_IP> \\
        --depth_zmq_addr tcp://<CAM_HOST>:5562
    # ... repeat 3-5 times, moving the arm between captures ...

    python tools/calib/calibrate_extrinsic_live_icp.py solve \\
        --session logs/calib_live \\
        --init calib/camera_align/current.npy \\
        --out calib/camera_align/extrinsic_YYYYMMDD_icp.npy

`capture` stores the fused depth frame and the joint angles that were true at
the same moment (read-only: it never commands the arm), so nothing depends on
the arm having reached a commanded pose.

The hand is NOT part of the reference by default: the Sharpa joint angles are
not on the Polymetis bridge, so posing it would add error rather than remove
it. Real points on the hand fall outside the correspondence threshold and are
rejected. `--hand_from_pkl` adds it when the real hand is held at a known
retargeted frame. `--include_table` adds the table plane (see its help for why
that is usually the wrong choice).

Frames: the extrinsic is camera-in-armbase (ROS optical); points are compared
in env-local, i.e. arm-base frame + `deploy_config.ARM_BASE_POS`, with the
table top at `deploy_config.TABLE_SURFACE_Z`.

Output always goes to a NEW file given by --out; `calib/camera_align/current.npy`
is never written (see calib/camera_align/README.md).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np

from dexx import deploy_config as _dcfg

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


def urdf_path(side: str = "right") -> str:
    """Merged FR3 + Sharpa Wave URDF for the given hand side."""
    return os.path.join(REPO_ROOT, "assets", "generated", f"fr3_with_{side}_sharpa_wave.urdf")


URDF_PATH = urdf_path("right")

ARM_BASE_POS = _dcfg.arm_base_pos_np()
TABLE_Z = float(_dcfg.TABLE_SURFACE_Z)
DEPTH_H, DEPTH_W = _dcfg.DEPTH_H, _dcfg.DEPTH_W
DEPTH_MIN_M, DEPTH_MAX_M = 0.10, 1.50

# Where the table is looked for (env-local, metres).
TABLE_SEARCH_LO = np.array([0.05, -0.45, TABLE_Z - 0.10])
TABLE_SEARCH_HI = np.array([0.75, 0.30, TABLE_Z + 0.10])

_ARM_LINK_RE = re.compile(r"^fr3_link(\d+)$")


def load_sibling(name: str):
    """Import another tools/calib/<name>.py (tools/ is not a package)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy helpers (also used by level_extrinsic_to_table.py and the tests)
# ─────────────────────────────────────────────────────────────────────────────
def intrinsics_from_args(values) -> dict:
    """--intrinsics FX FY CX CY, or deploy_config.SIM_INTRINSICS when omitted."""
    if values is None:
        return dict(_dcfg.SIM_INTRINSICS)
    fx, fy, cx, cy = (float(v) for v in values)
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)


def backproject(depth: np.ndarray, intr: dict,
                dmin: float = DEPTH_MIN_M, dmax: float = DEPTH_MAX_M) -> np.ndarray:
    """(H,W) depth in metres -> (N,3) camera-frame points (ROS optical)."""
    fx, fy, cx, cy = (float(intr[k]) for k in ("fx", "fy", "cx", "cy"))
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    ok = np.isfinite(depth) & (depth > dmin) & (depth < dmax)
    z = depth[ok]
    return np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], axis=1)


def cam_to_env(pts_cam: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Camera-frame points -> env-local via a camera-in-armbase 4x4."""
    return pts_cam @ T[:3, :3].T + T[:3, 3] + ARM_BASE_POS


def voxel_ds(P: np.ndarray, v: float) -> np.ndarray:
    if v <= 0 or not len(P):
        return P
    keys = np.floor(P / v).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return P[np.sort(idx)]


def cull_backfaces(pts: np.ndarray, nrm: np.ndarray, cam_pos: np.ndarray,
                   max_cos: float = -0.15) -> np.ndarray:
    """Mask of surface samples whose normal faces the camera."""
    v = pts - cam_pos[None]
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return (v * nrm).sum(1) < max_cos


def table_grid(step: float = 0.008, table_z: float = TABLE_Z) -> np.ndarray:
    """Sim table-top samples (env-local), for --include_table."""
    xs = np.arange(-0.10, 0.85, step)
    ys = np.arange(-0.50, 0.35, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, table_z)], 1).astype(np.float32)


def in_table_search_box(pts: np.ndarray) -> np.ndarray:
    return np.all((pts > TABLE_SEARCH_LO) & (pts < TABLE_SEARCH_HI), axis=1)


def fit_plane(p: np.ndarray, table_z: float = TABLE_Z, iters: int = 8):
    """Trimmed least-squares plane. Returns (unit normal with n_z > 0,
    median height of the inliers minus table_z, in metres)."""
    keep = np.ones(len(p), bool)
    n = np.array([0.0, 0.0, 1.0])
    for _ in range(iters):
        q = p[keep]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        n = vt[-1]
        n = n if n[2] > 0 else -n
        r = np.abs((p - c) @ n)
        keep = r < max(0.004, np.percentile(r, 70))
    return n, float(np.median(p[keep][:, 2] - table_z))


def tilt_deg(n: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))


def table_stats(pts_env: np.ndarray, table_z: float = TABLE_Z):
    """(median height error mm, tilt deg) of the table in an env-local cloud,
    or None if too few points fall in the table search box."""
    p = pts_env[in_table_search_box(pts_env)]
    if len(p) < 500:
        return None
    n, dz = fit_plane(p, table_z, iters=6)
    return dz * 1000.0, tilt_deg(n)


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def check_out_path(path: str, overwrite: bool) -> str:
    """Refuse to write the shipped default extrinsic, or clobber a file silently."""
    out = os.path.abspath(path)
    default = os.path.abspath(os.path.join(REPO_ROOT, _dcfg.CAMERA_EXTRINSIC_DEFAULT_FILE))
    if out == default or os.path.basename(out) == os.path.basename(default):
        sys.exit(f"refusing to write {path}: write new calibrations to a new, dated file "
                 f"(e.g. calib/camera_align/extrinsic_YYYYMMDD.npy), see "
                 f"calib/camera_align/README.md")
    if os.path.exists(out) and not overwrite:
        sys.exit(f"{path} exists; choose a new file name or pass --overwrite")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sim arm reference (URDF forward kinematics)
# ─────────────────────────────────────────────────────────────────────────────
def sim_robot_points(arm_q: np.ndarray, n_pts: int, exclude_links=(),
                     hand_q: np.ndarray | None = None, with_normals: bool = True,
                     urdf: str = URDF_PATH):
    """Surface samples (env-local) of the posed robot, and their outward normals.

    Arm links only (fr3_link0..7) unless `hand_q` is given, in which case the
    Sharpa hand meshes are posed with it and included too. `urdf` selects the
    hand side (see `urdf_path`).

    Normals enable backface culling, which is what makes this ICP converge: the
    sim target is a set of CLOSED meshes, while the real cloud only sees the
    faces turned toward the camera. Without culling, nearest-neighbour matching
    pairs real points with hidden surfaces and the solution slides along the
    geometry.
    """
    import trimesh
    import yourdfpy

    u = yourdfpy.URDF.load(urdf, load_meshes=True, build_scene_graph=True)
    names = list(u.actuated_joint_names)
    q = np.zeros(len(names))
    q[:7] = np.asarray(arm_q, dtype=np.float64)[:7]
    if hand_q is not None:
        hand_q = np.asarray(hand_q, dtype=np.float64)
        q[7:7 + len(hand_q)] = hand_q
    u.update_cfg(dict(zip(names, q.tolist())))

    scene = u.scene
    parents = scene.graph.transforms.parents
    exclude = {int(x) for x in exclude_links}
    keep, area = [], 0.0
    for gname, geom in scene.geometry.items():
        link = parents.get(gname, "")
        m_arm = _ARM_LINK_RE.match(link)
        if m_arm is None and hand_q is None:
            continue
        if m_arm is not None and int(m_arm.group(1)) in exclude:
            continue
        m = geom.copy().apply_transform(scene.graph.get(gname)[0])
        keep.append(m)
        area += m.area
    if not keep:
        raise RuntimeError(f"no fr3_link* meshes found in {urdf}")
    P, N = [], []
    for m in keep:
        k = max(200, int(n_pts * (m.area / max(area, 1e-6))))
        pts, fid = trimesh.sample.sample_surface(m, k)
        P.append(np.asarray(pts))
        N.append(np.asarray(m.face_normals[fid]))
    P = np.concatenate(P, 0).astype(np.float32) + ARM_BASE_POS
    if not with_normals:
        return P
    return P, np.concatenate(N, 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# capture
# ─────────────────────────────────────────────────────────────────────────────
def read_fresh_joints(client, wait_s: float = 5.0) -> np.ndarray:
    """Arm joint angles from a bridge state that arrived after this call."""
    n0 = client.wrist_msg_count
    t0 = time.time()
    while client.wrist_msg_count == n0:
        if time.time() - t0 > wait_s:
            raise RuntimeError(f"no new arm state within {wait_s}s — is "
                               f"polymetis_joint_bridge.py running on the NUC?")
        time.sleep(0.01)
    return client.arm_joint_positions.detach().cpu().numpy().astype(np.float64)[:7]


def grab_depth(addr: str, n: int, wait_s: float):
    """Median of n distinct depth frames from the camera host's ZMQ publisher."""
    from dexx.scripts.deploy.realsense_depth_zmq_subscriber import RealSenseDepthZmqSubscriber

    sub = RealSenseDepthZmqSubscriber(addr=addr, height=DEPTH_H, width=DEPTH_W, device="cpu")
    frames, seen, t0 = [], 0, time.time()
    try:
        while len(frames) < n and time.time() - t0 < wait_s:
            if sub.n_received != seen:
                seen = sub.n_received
                d = sub.get_latest()
                if d is not None:
                    frames.append(d.cpu().numpy().astype(np.float32))
            time.sleep(0.02)
        rejected = sub.n_rejected
    finally:
        sub.shutdown()
    if not frames:
        extra = (f" ({rejected} frames rejected for a size other than "
                 f"{DEPTH_H}x{DEPTH_W})" if rejected else "")
        raise RuntimeError(f"no depth from {addr}{extra}")
    return np.median(np.stack(frames), axis=0), len(frames)


def cmd_capture(a):
    from dexx.tasks.hand_imitation.deploy.polymetis_arm_client import PolymetisArmClient

    os.makedirs(a.session, exist_ok=True)
    # Read-only: never start the impedance controller, never terminate a policy.
    client = PolymetisArmClient(ip_address=a.ip, state_port=a.state_port,
                                cmd_port=a.cmd_port, start_impedance=False)
    try:
        q = read_fresh_joints(client)

        # A bridge serving a frozen joint vector while the arm is moved pairs a
        # new cloud with a stale pose, and the ICP still "converges" — to the
        # wrong answer. An exact repeat of an earlier capture is the signature.
        for prev in sorted(glob.glob(os.path.join(a.session, "pose_*.npz"))):
            if np.array_equal(np.load(prev)["joints"], q):
                raise RuntimeError(
                    f"joint vector is bit-identical to {os.path.basename(prev)}. Either "
                    f"the arm did not move, or the bridge is serving a stale reading — "
                    f"restart polymetis_joint_bridge.py and confirm the values change.")

        depth, n = grab_depth(a.depth_zmq_addr, a.n_frames, a.wait_seconds)
        q2 = read_fresh_joints(client)
    finally:
        client.shutdown(terminate_policy=False)

    drift = float(np.abs(q2 - q).max())
    if drift > a.max_joint_drift:
        raise RuntimeError(
            f"arm moved {drift:.4f} rad while the depth frames were being collected "
            f"(limit {a.max_joint_drift}); hold it still and retry")
    k = len(glob.glob(os.path.join(a.session, "pose_*.npz")))
    path = os.path.join(a.session, f"pose_{k:02d}.npz")
    np.savez(path, depth=depth, joints=q, n_frames=n, drift=drift, t=time.time())
    print(f"[capture] pose {k}: {n} frames fused, joint drift {drift*1000:.2f} mrad")
    print(f"[capture] joints = {np.round(q, 4)}")
    print(f"[capture] -> {path}   (total captured: {k+1})")
    if k + 1 < 3:
        print(f"[capture] move the arm to a DIFFERENT configuration and capture "
              f"again — {3-(k+1)} more recommended before solving")


# ─────────────────────────────────────────────────────────────────────────────
# solve
# ─────────────────────────────────────────────────────────────────────────────
def cmd_solve(a):
    _icp_tool = load_sibling("icp_align_extrinsic")  # Kabsch/SVD ICP + PLY writer
    icp, write_ply = _icp_tool.icp, _icp_tool.write_ply

    out = check_out_path(a.out, a.overwrite)
    intr = intrinsics_from_args(a.intrinsics)
    print(f"[solve] back-projecting with intrinsics {intr}")

    hand_q = None
    if a.hand_from_pkl:
        import pickle
        with open(a.hand_from_pkl, "rb") as f:
            hand_q = np.asarray(pickle.load(f)["opt_dof_pos"])[a.hand_frame]
        print(f"[solve] {a.side} hand INCLUDED, {len(hand_q)} joints from "
              f"{os.path.basename(a.hand_from_pkl)} frame {a.hand_frame}")

    poses = sorted(glob.glob(os.path.join(a.session, "pose_*.npz")))
    if not poses:
        sys.exit(f"no captures in {a.session}")
    T_init = np.load(a.init).astype(np.float64)
    print(f"[solve] {len(poses)} poses, init = {a.init}")
    if len(poses) < 3:
        print(f"[solve] WARNING: only {len(poses)} pose(s). Yaw and in-plane translation "
              f"may stay weakly constrained. Prefer 3+ distinct configurations.")

    exclude = [int(x) for x in a.exclude_links.split(",") if x.strip()]
    lo, hi = np.array(a.box[::2]), np.array(a.box[1::2])
    cam_pos = T_init[:3, 3] + ARM_BASE_POS
    src_all, tgt_all = [], []
    for p in poses:
        z = np.load(p)
        real = cam_to_env(backproject(z["depth"], intr), T_init)
        sim, nrm = sim_robot_points(z["joints"], a.sim_pts, exclude_links=exclude, hand_q=hand_q,
                                    urdf=urdf_path(a.side))
        n_sim0 = len(sim)
        if a.cull_backfaces:
            sim = sim[cull_backfaces(sim, nrm, cam_pos)]
        n_sim1 = len(sim)
        if a.include_table:
            sim = np.concatenate([sim, table_grid()], 0)
        rm = np.all((real > lo) & (real < hi), axis=1)
        sm = np.all((sim > lo) & (sim < hi), axis=1)
        r, s = voxel_ds(real[rm], a.voxel), voxel_ds(sim[sm], a.voxel)
        if a.min_z is not None:
            n0 = len(r)
            r = r[r[:, 2] > a.min_z]
            print(f"    min_z {a.min_z}: {n0} -> {len(r)} real points")
        msg = f"  {os.path.basename(p)}: real {rm.sum()}->{len(r)}"
        if a.near_arm > 0:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(sim).query(r, k=1)
            n_before = len(r)
            r = r[d < a.near_arm]
            msg = (f"  {os.path.basename(p)}: real {rm.sum()}->{n_before}->{len(r)} "
                   f"(near-arm {a.near_arm*100:.0f}cm)")
        print(f"{msg}  sim {n_sim0}->{n_sim1} (culled)->{len(s)}")
        src_all.append(r)
        tgt_all.append(s)

    # Joint solve: one rigid correction explaining every pose at once. The
    # correction lives in the camera-to-armbase transform, which all poses
    # share; the arm geometry differs per pose, and that difference is exactly
    # what pins the DOFs a single pose leaves loose.
    src = np.concatenate(src_all, 0)
    tgt = np.concatenate(tgt_all, 0)
    print(f"[solve] stacked: {len(src)} real vs {len(tgt)} sim points")
    T_corr, rmse, inl, tot = icp(src, tgt, a.max_corr, iters=a.iters)

    # T_corr acts in env-local; the extrinsic is in the arm-base frame, which
    # differs by a pure translation, so the rotation composes directly and the
    # translation picks up (R - I) @ ARM_BASE_POS.
    R, t = T_corr[:3, :3], T_corr[:3, 3]
    T_new = np.eye(4)
    T_new[:3, :3] = R @ T_init[:3, :3]
    T_new[:3, 3] = R @ (T_init[:3, 3] + ARM_BASE_POS) + t - ARM_BASE_POS

    print(f"\n[solve] correction: {rot_angle_deg(R):.3f} deg, "
          f"{np.linalg.norm(T_new[:3, 3] - T_init[:3, 3])*1000:.1f} mm   "
          f"rmse {rmse*1000:.2f} mm, inliers {inl}/{tot}")

    print(f"\n{'extrinsic':16s} {'table bias':>12s} {'table tilt':>12s}")
    z0 = np.load(poses[0])
    for lbl, T in (("init", T_init), ("solved", T_new)):
        st = table_stats(cam_to_env(backproject(z0["depth"], intr), T))
        print(f"{lbl:16s} {st[0]:>10.1f}mm {st[1]:>11.2f}deg" if st
              else f"{lbl:16s}  (no table found)")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(out, T_new.astype(np.float32))
    print(f"\n[solve] saved -> {a.out}")
    print(np.array2string(T_new, precision=6, suppress_small=True))
    os.makedirs(a.dump_dir, exist_ok=True)
    write_ply(os.path.join(a.dump_dir, "real_solved.ply"), src @ R.T + t)
    write_ply(os.path.join(a.dump_dir, "sim_target.ply"), tgt)
    print(f"[solve] PLYs (env-local) -> {a.dump_dir}")
    print("[solve] next: level height and tilt against the table with "
          "tools/calib/level_extrinsic_to_table.py")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Multi-pose ICP of the live depth cloud against the FK-posed sim arm.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="record one (depth, joints) pair; read-only on the arm")
    c.add_argument("--session", required=True, help="directory collecting pose_XX.npz")
    c.add_argument("--ip", required=True, help="NUC bridge IP (polymetis_joint_bridge.py host)")
    c.add_argument("--state_port", type=int, default=_dcfg.POLYMETIS_STATE_PORT)
    c.add_argument("--cmd_port", type=int, default=_dcfg.POLYMETIS_CMD_PORT,
                   help="unused for commands (capture never moves the arm); "
                        "the client connects it regardless")
    c.add_argument("--depth_zmq_addr", required=True,
                   help=f"camera-host depth publisher, e.g. {_dcfg.CAMERA_ZMQ_ADDR_EXAMPLE}")
    c.add_argument("--n_frames", type=int, default=21, help="depth frames fused by median")
    c.add_argument("--wait_seconds", type=float, default=15.0)
    c.add_argument("--max_joint_drift", type=float, default=0.002,
                   help="rad; reject the capture if the arm moved this much "
                        "while the frames were being collected")
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve", help="joint ICP over all captured poses")
    s.add_argument("--session", required=True)
    s.add_argument("--init", required=True, help="initial camera-in-armbase 4x4 .npy")
    s.add_argument("--out", required=True,
                   help="NEW .npy for the result, e.g. calib/camera_align/extrinsic_YYYYMMDD_icp.npy "
                        "(current.npy is refused)")
    s.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out")
    s.add_argument("--intrinsics", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"),
                   default=None,
                   help="back-projection intrinsics for the real depth (default: "
                        "deploy_config.SIM_INTRINSICS). The D455's own decimated depth "
                        "intrinsics differ by ~1.5 px in cx, a lateral error that grows "
                        "with depth and that no rigid transform can absorb.")
    s.add_argument("--box", type=float, nargs=6,
                   default=[-0.15, 0.85, -0.50, 0.40, 0.40, 1.10],
                   help="x_lo x_hi y_lo y_hi z_lo z_hi in env-local")
    s.add_argument("--sim_pts", type=int, default=200000)
    s.add_argument("--voxel", type=float, default=0.006)
    s.add_argument("--max_corr", type=float, default=0.05)
    s.add_argument("--iters", type=int, default=60)
    s.add_argument("--include_table", action="store_true",
                   help="add the table plane to the target. NOT valid when the real "
                        "table is covered by a mat: sim puts the top at exactly "
                        "TABLE_SURFACE_Z, so the mat's thickness enters the fit as a "
                        "height error and drags tilt with it. Prefer the separate "
                        "level_extrinsic_to_table.py step.")
    s.add_argument("--cull_backfaces", action="store_true", default=True,
                   help="drop sim samples facing away from the camera (default on)")
    s.add_argument("--no_cull", dest="cull_backfaces", action="store_false")
    s.add_argument("--near_arm", type=float, default=0.10,
                   help="keep only real points within this many metres of the predicted "
                        "arm surface; removes table, background and most of the hand "
                        "before they can bias the fit (0 disables)")
    s.add_argument("--hand_from_pkl", type=str, default=None,
                   help="retarget pkl to read the 22 hand joints (opt_dof_pos) from; ADDS "
                        "the Sharpa hand meshes to the reference. Only valid when the real "
                        "hand is held at --hand_frame of that demo: the bridge does not "
                        "publish hand joints, so a mismatch cannot be detected.")
    s.add_argument("--hand_frame", type=int, default=0)
    s.add_argument("--side", choices=("right", "left"), default="right",
                   help="hand mounted on the arm; selects fr3_with_<side>_sharpa_wave.urdf "
                        "(only the hand meshes posed by --hand_from_pkl differ)")
    s.add_argument("--exclude_links", type=str, default="",
                   help="comma-separated fr3 link numbers to drop from the reference, e.g. "
                        "'0': link0 sits on the table, so the near-arm filter sweeps up "
                        "table/mat points around it that the URDF has no geometry for")
    s.add_argument("--min_z", type=float, default=None,
                   help="drop real points below this env-local height before matching")
    s.add_argument("--dump_dir", default="logs/calib_live_icp")
    s.set_defaults(func=cmd_solve)
    return ap


def main():
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
